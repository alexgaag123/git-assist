"""Пайплайн индексации: собирает из рабочего дерева готовый к запросам снапшот.

Это стадия оркестрации в потоке воркера — materialize → ingest → chunk → embed → upsert →
граф → READY → activate. Связывает детерминированные строительные блоки common
(ingest_tree, chunk_documents, Embedder, три стора) под одним snapshot_id и ведёт снапшот
по жизненному циклу, чтобы чтение всегда видело только полностью готовую сборку.

Чистая оркестрация над интерфейсами common: импортирует только common.*, никогда apps/api,
и зависит только от протоколов Embedder и трёх сторов — конкретные адаптеры (Ollama,
Qdrant, Neo4j, Postgres) собираются фабриками common.

index_tree — это сама сборка (до READY); activate_snapshot — атомарное переключение,
которое переводит готовую сборку в ACTIVE, откуда её уже видно чтению.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.chunk import chunk_documents
from common.config import Settings
from common.embed import Embedder
from common.graph import build_graph_edges
from common.ingest import ingest_tree
from common.snapshot import SnapshotRef, SnapshotStatus, can_transition
from common.stores import GraphStore, MetaStore, VectorRecord, VectorStore


@dataclass(frozen=True)
class IndexResult:
    """Итог одной сборки индекса — для логирования и тестов.

    status — статус жизненного цикла, в котором остался снапшот (READY при успехе).
    """

    snapshot_id: str
    n_documents: int
    n_chunks: int
    n_edges: int
    status: SnapshotStatus


_EMBED_BATCH_SIZE = 32


def _embed_in_batches(embedder: Embedder, texts: list[str]) -> list[tuple[float, ...]]:
    vectors: list[tuple[float, ...]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        vectors.extend(embedder.embed(texts[start : start + _EMBED_BATCH_SIZE]))
    return vectors


def index_tree(
    root: str,
    ref: SnapshotRef,
    *,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    graph_store: GraphStore,
    meta_store: MetaStore,
) -> IndexResult:
    """Собрать индекс рабочего дерева root под снапшотом ref.

    Прогоняет стадии сборки по порядку под snapshot_id = ref.snapshot_id и ведёт снапшот
    через PENDING → BUILDING → READY: разбирает дерево в документы, режет их на чанки,
    эмбеддит, апсертит векторы, резолвит внутренние импорты в рёбра графа — и по пути
    регистрирует документы и чанки в мета-сторе. Возвращает IndexResult с итогами сборки.

    Сборка идемпотентна: повторный запуск с тем же ref по тому же дереву заменяет те же
    документы/чанки/векторы (все ключи детерминированы) и даёт тот же результат. Пустое
    дерево — легальный случай: ноль документов, ноль чанков, статус всё равно READY.

    Чанки эмбеддятся батчами по _EMBED_BATCH_SIZE, а не одним вызовом на весь снапшот —
    один запрос на реальный репозиторий легко даёт тысячи чанков, а батчи держат
    отдельные HTTP-запросы к Ollama разумного размера.

    Возвращает ValueError, если embedder.dim не совпадает с settings.embed_dim — векторный стор
    создаётся под embed_dim, так что несовпадающий эмбеддер это ошибка конфигурации, и её
    ловят до того, как что-либо записано.
    """

    if embedder.dim != settings.embed_dim:
        raise ValueError(
            f"embedder.dim ({embedder.dim}) must equal settings.embed_dim ({settings.embed_dim})"
        )

    snapshot_id = ref.snapshot_id
    meta_store.set_status(snapshot_id, SnapshotStatus.PENDING, repo=ref.repo)
    meta_store.set_status(snapshot_id, SnapshotStatus.BUILDING, repo=ref.repo)

    documents = ingest_tree(root, snapshot_id)
    meta_store.put_documents(documents)

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    meta_store.put_chunks(chunks)

    vectors = _embed_in_batches(embedder, [chunk.text for chunk in chunks])
    records = [
        VectorRecord(chunk_id=chunk.chunk_id, snapshot_id=snapshot_id, vector=vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    vector_store.upsert(records)

    edges = build_graph_edges(documents)
    graph_store.add_edges(edges)

    meta_store.set_status(snapshot_id, SnapshotStatus.READY, repo=ref.repo)
    return IndexResult(
        snapshot_id=snapshot_id,
        n_documents=len(documents),
        n_chunks=len(chunks),
        n_edges=len(edges),
        status=SnapshotStatus.READY,
    )


def activate_snapshot(snapshot_id: str, *, repo: str, meta_store: MetaStore) -> None:
    """Атомарно переключить snapshot_id в ACTIVE-снапшот репозитория repo.

    Переводит READY-снапшот в ACTIVE и разжалует прежний ACTIVE того же репозитория в
    RETIRED — так что ACTIVE ровно один на репозиторий, и чтение никогда не увидит
    недостроенный индекс. Снапшоты других репозиториев не трогаются — у каждого репозитория
    свой независимый активный снапшот.

    Возвращает ValueError, если snapshot_id не в статусе READY (нельзя активировать
    незавершённую сборку) — это заодно защищает от повторной активации уже активного
    снапшота: тихого no-op не будет, будет явная ошибка.
    """

    status = meta_store.get_status(snapshot_id)
    if status != SnapshotStatus.READY or not can_transition(
        SnapshotStatus.READY, SnapshotStatus.ACTIVE
    ):
        raise ValueError(
            f"cannot activate snapshot {snapshot_id!r}: status is {status} (expected READY)"
        )

    previous = meta_store.active_snapshot(repo)
    if previous is not None and previous != snapshot_id:
        meta_store.set_status(previous, SnapshotStatus.RETIRED, repo=repo)

    meta_store.set_status(snapshot_id, SnapshotStatus.ACTIVE, repo=repo)

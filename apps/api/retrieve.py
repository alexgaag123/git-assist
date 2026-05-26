"""Retrieval: находим и ранжируем чанки из активного снапшота под запрос.

Это read-сторона системы, поверхность запросов apps/api. Поток такой: запрос →
эмбеддинг → векторный поиск (по активному снапшоту) → маппинг хитов обратно в
чанки → сборка ранжированного контекста — детерминированное ядро полного
продакшен-пути запрос → эмбеддинг (Ollama) → векторный поиск → реранк → сборка
контекста → ответ (Ollama). Реранкинг и генерация — тяжёлые, завязанные на
модели этапы за собственными интерфейсами; этот модуль останавливается на
ранжированном поиске.

Чистая оркестрация над интерфейсами common: импортирует только common.* (никогда
apps/worker) и зависит только от протоколов Embedder/VectorStore/MetaStore —
конкретные адаптеры (Ollama, Qdrant, Postgres) собираются фабриками common.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from common.config import Settings
from common.embed import Embedder
from common.fusion import exact_name_boost, symbol_name_for
from common.hyde import HydeExpander
from common.rerank import Reranker
from common.stores import MetaStore, VectorStore

DEFAULT_TOP_K = 5
"""Сколько результатов возвращает retrieve, если limit не задан."""

DEFAULT_PREFETCH_K = 30
"""Размер пула прификтча для retrieve_pipeline перед реранкингом."""


@dataclass(frozen=True)
class RetrievalResult:
    """Один результат поиска: SearchHit, склеенный со своим чанком.

    Несёт path и text, чтобы сборка контекста (и HTTP-API) обходилась без
    второго похода в стор; score — это similarity векторного поиска (или,
    после реранка, оценка реранкера), snapshot_id — снапшот, из которого читан
    чанк.
    """

    chunk_id: str
    path: str
    text: str
    score: float
    snapshot_id: str


def retrieve(
    query: str,
    *,
    repo: str,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    meta_store: MetaStore,
    limit: int = DEFAULT_TOP_K,
) -> list[RetrievalResult]:
    """Достать limit самых релевантных чанков под query из активного снапшота repo.

    Эмбеддит query тем же эмбеддером, что использовался при индексации, ищет по
    векторам активного снапшота и гидрирует каждый хит обратно в RetrievalResult
    через реестр чанков. Результаты приходят уже в порядке ранжирования — по
    убыванию похожести, с детерминированным тай-брейком по chunk_id.

    Возвращает пустой список, если для repo нет активного снапшота (индексировать
    ещё нечего) — без эмбеддинга и поиска. Хит, чей чанк отсутствует в мета-сторе
    (устаревший id), просто пропускается, не роняя весь запрос.

    Возвращает ValueError, если embedder.dim не совпадает с settings.embed_dim: раз
    активные векторы были записаны с размерностью settings.embed_dim, эмбеддер
    другой размерности — это ошибка конфигурации (то же самое проверяет guard в
    index_tree на этапе индексации).
    """

    if embedder.dim != settings.embed_dim:
        raise ValueError(
            f"embedder.dim ({embedder.dim}) must equal settings.embed_dim ({settings.embed_dim})"
        )

    snapshot_id = meta_store.active_snapshot(repo)
    if snapshot_id is None:
        return []

    query_vector = embedder.embed([query])[0]
    hits = vector_store.search(snapshot_id, query_vector, limit)

    results: list[RetrievalResult] = []
    for hit in hits:
        chunk = meta_store.get_chunk(hit.chunk_id)
        if chunk is None:
            continue
        results.append(
            RetrievalResult(
                chunk_id=hit.chunk_id,
                path=chunk.path,
                text=chunk.text,
                score=hit.score,
                snapshot_id=chunk.snapshot_id,
            )
        )
    return results


def retrieve_pipeline(
    query: str,
    *,
    repo: str,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    meta_store: MetaStore,
    reranker: Reranker,
    limit: int = DEFAULT_TOP_K,
    prefetch_limit: int = DEFAULT_PREFETCH_K,
    expander: HydeExpander | None = None,
) -> list[RetrievalResult]:
    """Двухэтапный поиск с реранком: прификтч → гидрация → реранк → буст точных имён → top-k.

    Эмбеддит query, достаёт из векторного стора активного снапшота repo более
    широкий пул кандидатов (prefetch_limit), реранкает их через reranker,
    применяет жёсткий буст exact_name_boost как финальный арбитр порядка и
    обрезает до limit.

    Порядок этапов (буст — последнее и решающее слово):

    1. Guard по размерности — ValueError, если embedder.dim не совпадает с
       settings.embed_dim.
    2. Активный снапшот — если для repo ничего не активно, сразу пустой список.
    3. HyDE-расширение — если передан expander, expander.expand(query) даёт
       текст для эмбеддинга; без expander эмбеддится исходный запрос без
       изменений.
    4. Прификтч — расширенный текст эмбеддится, из векторного стора достаётся
       до prefetch_limit кандидатов.
    5. Гидрация — каждый хит превращается в свой Chunk; устаревшие хиты (чанк
       отсутствует в meta_store) молча пропускаются.
    6. Реранк — reranker.rerank(query, texts, top_k=None) оценивает и
       переупорядочивает весь гидрированный пул по исходному query (не по
       расширенному тексту); оценка реранкера попадает в RetrievalResult.score.
    7. Буст точных имён — exact_name_boost выносит вперёд кандидатов, чьё имя
       символа (по стему пути) точно совпадает с токеном запроса, невзирая на
       оценку реранкера — точное совпадение по имени никогда не проигрывает
       нечёткому семантическому соседу, буст побеждает реранкер.
    8. Обрезка — возвращается boosted[:limit].

    Размер пула prefetch_limit ограничивает то, с чем вообще может работать
    реранкер: слишком узкий прификтч режет recall, даже если limit большой.

    Возвращает пустой список, если для repo нет активного снапшота, без
    эмбеддинга и поиска. Хит с отсутствующим в meta_store чанком просто
    пропускается, не роняя запрос.

    Возвращает ValueError при несовпадении embedder.dim и settings.embed_dim — тот
    же guard, что и в retrieve.
    """
    if embedder.dim != settings.embed_dim:
        raise ValueError(
            f"embedder.dim ({embedder.dim}) must equal settings.embed_dim ({settings.embed_dim})"
        )

    snapshot_id = meta_store.active_snapshot(repo)
    if snapshot_id is None:
        return []

    text_to_embed = expander.expand(query) if expander is not None else query
    query_vector = embedder.embed([text_to_embed])[0]
    hits = vector_store.search(snapshot_id, query_vector, prefetch_limit)

    candidates: list[RetrievalResult] = []
    for hit in hits:
        chunk = meta_store.get_chunk(hit.chunk_id)
        if chunk is None:
            continue
        candidates.append(
            RetrievalResult(
                chunk_id=hit.chunk_id,
                path=chunk.path,
                text=chunk.text,
                score=hit.score,
                snapshot_id=chunk.snapshot_id,
            )
        )

    if not candidates:
        return []

    rr = reranker.rerank(query, [c.text for c in candidates], top_k=None)
    reranked = [replace(candidates[r.index], score=r.score) for r in rr]

    boosted = exact_name_boost(reranked, query, lambda result: symbol_name_for(result.path))
    return boosted[:limit]


def assemble_context(
    results: list[RetrievalResult],
    *,
    max_chars: int | None = None,
) -> str:
    """Склеить ранжированные results в один блок контекста, готовый для промпта.

    Каждый результат становится секцией "# {path}\n{text}"; секции соединяются
    пустой строкой в порядке ранжирования. Пустой results даёт пустую строку.

    Если задан max_chars, секции добавляются по порядку, пока суммарная длина
    (с учётом разделителей) укладывается в бюджет; первая секция, которая бы
    его превысила, останавливает сборку — так длинный хвост никогда не
    взрывает контекстное окно генерации. Топовая секция включается всегда,
    даже если сама по себе превышает бюджет — лучший хит никогда не
    выбрасывается молча. None означает отсутствие бюджета.
    """

    if not results:
        return ""

    sections = [f"# {result.path}\n{result.text}" for result in results]
    if max_chars is None:
        return "\n\n".join(sections)

    separator = "\n\n"
    chosen = [sections[0]]
    length = len(sections[0])
    for section in sections[1:]:
        candidate = length + len(separator) + len(section)
        if candidate > max_chars:
            break
        length = candidate
        chosen.append(section)
    return separator.join(chosen)

"""Границы хранения: три стора, с которыми говорит остальная система.

Поиск и импакт-анализ опираются на три бэкенда — векторный стор (Qdrant), графовый стор
(Neo4j) и стор метаданных (Postgres). Каждый описан здесь как чистый Protocol, поэтому
вызывающий код зависит от формы интерфейса, а не от конкретной реализации; фабрики
make_vector_store/make_graph_store в этом модуле и make_meta_store в common.db выбирают
конкретный адаптер по URL.

Каждая запись несёт snapshot_id, и каждое чтение ограничено одним снапшотом; вся система
всегда обращается только к ACTIVE снапшоту, которым владеет MetaStore. Направление рёбер
графа единое по всей системе: ребро означает «src зависит от dst».
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from common.models import Chunk, Document
from common.snapshot import SnapshotStatus


@dataclass(frozen=True)
class VectorRecord:
    """Чанк с посчитанным эмбеддингом: его id, снапшот-владелец и плотный вектор."""

    chunk_id: str
    snapshot_id: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchHit:
    """Один результат векторного поиска: chunk_id и его similarity-score."""

    chunk_id: str
    score: float


@dataclass(frozen=True)
class GraphEdge:
    """Ребро зависимости src -> dst (src зависит от dst) внутри одного снапшота.

    Узлы — непрозрачные строковые ключи: путь к файлу или имя символа. kind описывает тип
    связи (например, "import").
    """

    src: str
    dst: str
    kind: str
    snapshot_id: str


@runtime_checkable
class VectorStore(Protocol):
    """Чанки с посчитанными эмбеддингами, доступные для поиска по схожести внутри снапшота."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Вставить или заменить records, идемпотентно по chunk_id."""
        ...

    def search(self, snapshot_id: str, query: tuple[float, ...], limit: int) -> list[SearchHit]:
        """Топ limit записей в snapshot_id, отсортированных по убыванию схожести."""
        ...

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Удалить все записи с этим snapshot_id."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """Граф зависимостей файлов/символов, по которому считается импакт внутри снапшота."""

    def add_edges(self, edges: Sequence[GraphEdge]) -> None:
        """Добавить edges, идемпотентно по (src, dst, kind, snapshot_id)."""
        ...

    def dependents(self, snapshot_id: str, node: str) -> set[str]:
        """Узлы, напрямую зависящие от node (все src, у которых dst == node)."""
        ...

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Удалить все рёбра с этим snapshot_id."""
        ...


@runtime_checkable
class MetaStore(Protocol):
    """Жизненный цикл снапшотов плюс реестр документов и чанков."""

    def set_status(self, snapshot_id: str, status: SnapshotStatus, *, repo: str = "") -> None:
        """Записать статус жизненного цикла snapshot_id, принадлежащего repo."""
        ...

    def get_status(self, snapshot_id: str) -> SnapshotStatus | None:
        """Сохранённый статус snapshot_id, либо None, если он неизвестен."""
        ...

    def active_snapshot(self, repo: str) -> str | None:
        """snapshot_id, который сейчас ACTIVE у repo, либо None, если такого нет."""
        ...

    def put_documents(self, documents: Sequence[Document]) -> None:
        """Вставить или заменить documents, идемпотентно по doc_id."""
        ...

    def get_document(self, doc_id: str) -> Document | None:
        """Документ с этим doc_id, либо None, если его нет."""
        ...

    def put_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Вставить или заменить chunks, идемпотентно по chunk_id."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Чанк с этим chunk_id, либо None, если его нет."""
        ...


DEFAULT_VECTOR_COLLECTION = "chunks"
"""Имя коллекции Qdrant, которое использует make_vector_store (одна коллекция на бэкенд;
изоляция по снапшотам достигается фильтром по полю snapshot_id в payload)."""


def make_vector_store(url: str, *, embed_dim: int) -> VectorStore:
    """Выбрать реализацию VectorStore по строке подключения url.

    - URL Qdrant (``qdrant://…``, ``http://…``, ``https://…`` или ``:memory:``) →
      QdrantVectorStore поверх клиента, который строится лениво из url и настраивается
      по embed_dim (соединение не открывается в момент вызова; адаптер импортируется
      лениво, чтобы импорт common.stores оставался лёгким).
    - что угодно другое → ValueError с названием неподдерживаемой схемы.

    Вызывающий код выбирает бэкенд через settings.vector_url и settings.embed_dim, не
    импортируя конкретный класс стора напрямую.
    """

    if url == ":memory:" or url.startswith(("qdrant://", "http://", "https://")):
        from common.qdrant_vector import QdrantVectorStore

        return QdrantVectorStore(DEFAULT_VECTOR_COLLECTION, embed_dim, url=url)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported vector-store URL scheme: {scheme!r} (url={url!r})")


_NEO4J_SCHEMES = ("neo4j://", "neo4j+s://", "bolt://", "bolt+s://")


def make_graph_store(url: str, *, auth: tuple[str, str] | None = None) -> GraphStore:
    """Выбрать реализацию GraphStore по строке подключения url.

    - URL Neo4j (``neo4j://…``, ``neo4j+s://…``, ``bolt://…``, ``bolt+s://…``) →
      Neo4jGraphStore поверх драйвера, который строится лениво из url и auth (соединение
      не открывается в момент вызова; адаптер импортируется лениво, чтобы импорт
      common.stores оставался лёгким).
    - что угодно другое → ValueError с названием неподдерживаемой схемы.

    Вызывающий код выбирает бэкенд через settings.graph_url и передаёт креды через
    settings.graph_user/settings.graph_password, не импортируя конкретный класс стора
    напрямую.
    """

    if url.startswith(_NEO4J_SCHEMES):
        from common.neo4j_graph import Neo4jGraphStore

        return Neo4jGraphStore(url=url, auth=auth)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported graph-store URL scheme: {scheme!r} (url={url!r})")

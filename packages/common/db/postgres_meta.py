"""MetaStore поверх настоящего Postgres.

PostgresMetaStore хранит жизненный цикл снапшотов и реестр документов/чанков в
Postgres, реализуя протокол MetaStore. ORM-таблицы живут в common.db.models; этот
модуль — адаптер поверх переданного движка SQLAlchemy.

Записи — идемпотентные upsert по первичному ключу через диалектный
``INSERT ... ON CONFLICT DO UPDATE``; чтения восстанавливают обычные доменные
записи (колонка ``acl``, которой у них нет, отбрасывается). ``active_snapshot``
возвращает наименьший ``snapshot_id`` со статусом ``ACTIVE`` — детерминированно,
даже если ``ACTIVE`` помечено больше одной строки. Каждый вызов идёт в своей
транзакции. Конструктор стора не открывает соединение: можно передать готовый
Engine либо ``url``, из которого движок строится лениво при первом обращении
(так стор для ``postgresql://`` можно создать даже без установленного драйвера
— см. make_meta_store).
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Session

from common.db.models import (
    ChunkRow,
    DocumentRow,
    SnapshotRow,
    chunk_to_row,
    document_to_row,
    row_to_chunk,
    row_to_document,
    row_to_status,
    snapshot_to_row,
)
from common.models import Chunk, Document
from common.snapshot import SnapshotStatus

_SNAPSHOT_COLUMNS = ("snapshot_id", "status", "repo", "acl")
_DOCUMENT_COLUMNS = ("doc_id", "snapshot_id", "path", "kind", "text", "acl")
_CHUNK_COLUMNS = ("chunk_id", "doc_id", "snapshot_id", "path", "index", "text", "acl")


def _row_values(row: DeclarativeBase, columns: tuple[str, ...]) -> dict[str, object]:
    return {column: getattr(row, column) for column in columns}


class PostgresMetaStore:
    """Реализация MetaStore поверх движка SQLAlchemy."""

    def __init__(self, engine: Engine | None = None, *, url: str | None = None) -> None:
        if engine is None and url is None:
            raise ValueError("PostgresMetaStore requires either an engine or a url")
        self._engine = engine
        self._url = url

    def _get_engine(self) -> Engine:
        """Возвращает движок, при первом обращении лениво строя его из ``url``.

        Отложенный вызов create_engine избавляет конструктор от зависимости на
        драйвер: DBAPI Postgres импортируется только при реальном обращении к
        базе, так что фабрика может вернуть стор для ``postgresql://``-URL даже
        без установленного драйвера.
        """

        if self._engine is None:
            assert self._url is not None
            self._engine = create_engine(self._url)
        return self._engine

    def _upsert(
        self,
        model: type[DeclarativeBase],
        rows: Sequence[DeclarativeBase],
        columns: tuple[str, ...],
        pk: str,
    ) -> None:
        """Массовый upsert ``rows`` модели ``model``.

        При конфликте по ``pk`` заменяются все остальные колонки.
        """

        if not rows:
            return
        stmt = pg_insert(model).values([_row_values(row, columns) for row in rows])
        stmt = stmt.on_conflict_do_update(
            index_elements=[pk],
            set_={column: getattr(stmt.excluded, column) for column in columns if column != pk},
        )
        with Session(self._get_engine()) as session:
            session.execute(stmt)
            session.commit()

    def set_status(self, snapshot_id: str, status: SnapshotStatus, *, repo: str = "") -> None:
        self._upsert(
            SnapshotRow,
            [snapshot_to_row(snapshot_id, status, repo=repo)],
            _SNAPSHOT_COLUMNS,
            "snapshot_id",
        )

    def get_status(self, snapshot_id: str) -> SnapshotStatus | None:
        with Session(self._get_engine()) as session:
            row = session.get(SnapshotRow, snapshot_id)
            return row_to_status(row) if row is not None else None

    def active_snapshot(self, repo: str) -> str | None:
        stmt = (
            select(SnapshotRow.snapshot_id)
            .where(SnapshotRow.status == SnapshotStatus.ACTIVE.value, SnapshotRow.repo == repo)
            .order_by(SnapshotRow.snapshot_id)
        )
        with Session(self._get_engine()) as session:
            return session.scalars(stmt).first()

    def put_documents(self, documents: Sequence[Document]) -> None:
        self._upsert(
            DocumentRow,
            [document_to_row(document) for document in documents],
            _DOCUMENT_COLUMNS,
            "doc_id",
        )

    def get_document(self, doc_id: str) -> Document | None:
        with Session(self._get_engine()) as session:
            row = session.get(DocumentRow, doc_id)
            return row_to_document(row) if row is not None else None

    def put_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._upsert(
            ChunkRow,
            [chunk_to_row(chunk) for chunk in chunks],
            _CHUNK_COLUMNS,
            "chunk_id",
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with Session(self._get_engine()) as session:
            row = session.get(ChunkRow, chunk_id)
            return row_to_chunk(row) if row is not None else None

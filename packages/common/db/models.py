"""SQLAlchemy ORM-модели для Postgres-based meta-store.

Три таблицы стоят за PostgresMetaStore: реестр жизненного цикла снапшотов
(``snapshots``) и реестр документов/чанков (``documents``, ``chunks``). Каждая
строка несёт тот же детерминированный 16-символьный hex-id, что уже генерирует
доменный слой (``snapshot_id`` / ``doc_id`` / ``chunk_id``), в качестве
первичного ключа — база сама id никогда не генерирует — плюс тег ``acl``
(по умолчанию ``""``), чтобы поиск мог предфильтровать по тегам доступа без
второго стора.

``snapshot_id`` и ``acl`` — это колонки, по которым потом фильтруют, поэтому они
проиндексированы. Строки маппятся один в один на Document/Chunk из
common.models и SnapshotStatus из common.snapshot; у колонки ``acl`` пока нет
аналога в доменной записи, поэтому при чтении она отбрасывается (протокол
возвращает обычные записи). ``acl`` по умолчанию ``""``, так что вызывающий код,
который её вообще не выставляет, получает round-trip без изменений.

``Base.metadata`` — это цель alembic-миграций. Модуль — чистое ORM-определение:
его импорт не открывает соединение.
"""

from __future__ import annotations

from sqlalchemy import Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from common.models import Chunk, Document, FileKind
from common.snapshot import SnapshotStatus


class Base(DeclarativeBase):
    """Декларативная база; ``Base.metadata`` — цель миграций."""


class SnapshotRow(Base):
    """Статус жизненного цикла одного снапшота, ключ — ``snapshot_id``."""

    __tablename__ = "snapshots"

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    repo: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    acl: Mapped[str] = mapped_column(String, nullable=False, default="")


class DocumentRow(Base):
    """Текст одного файла в рамках снапшота, ключ — ``doc_id``."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    acl: Mapped[str] = mapped_column(String, nullable=False, default="")

    __table_args__ = (Index("ix_documents_snapshot_id_path", "snapshot_id", "path"),)


class ChunkRow(Base):
    """Непрерывный кусок документа, ключ — ``chunk_id``."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    acl: Mapped[str] = mapped_column(String, nullable=False, default="")


def snapshot_to_row(
    snapshot_id: str, status: SnapshotStatus, *, repo: str = "", acl: str = ""
) -> SnapshotRow:
    """Собирает SnapshotRow из id, статуса жизненного цикла и владеющего ``repo``."""

    return SnapshotRow(snapshot_id=snapshot_id, status=status.value, repo=repo, acl=acl)


def row_to_status(row: SnapshotRow) -> SnapshotStatus:
    """Восстанавливает SnapshotStatus, хранящийся в ``row``."""

    return SnapshotStatus(row.status)


def document_to_row(document: Document, *, acl: str = "") -> DocumentRow:
    """Собирает DocumentRow из Document (``acl`` по умолчанию ``""``)."""

    return DocumentRow(
        doc_id=document.doc_id,
        snapshot_id=document.snapshot_id,
        path=document.path,
        kind=document.kind.value,
        text=document.text,
        acl=acl,
    )


def row_to_document(row: DocumentRow) -> Document:
    """Восстанавливает запись Document из ``row`` (``acl`` отбрасывается)."""

    return Document(
        path=row.path,
        kind=FileKind(row.kind),
        text=row.text,
        snapshot_id=row.snapshot_id,
    )


def chunk_to_row(chunk: Chunk, *, acl: str = "") -> ChunkRow:
    """Собирает ChunkRow из Chunk (``acl`` по умолчанию ``""``)."""

    return ChunkRow(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        snapshot_id=chunk.snapshot_id,
        path=chunk.path,
        index=chunk.index,
        text=chunk.text,
        acl=acl,
    )


def row_to_chunk(row: ChunkRow) -> Chunk:
    """Восстанавливает запись Chunk из ``row`` (``acl`` отбрасывается)."""

    return Chunk(
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        path=row.path,
        index=row.index,
        text=row.text,
        snapshot_id=row.snapshot_id,
    )

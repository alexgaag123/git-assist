"""Первоначальная схема meta-store.

Revision ID: 0001_initial
Revises:
Create Date: spec 13

Создаёт таблицы ``snapshots`` / ``documents`` / ``chunks`` и их индексы — один в
один с метаданными Base из common.db.models. Идентификатор ревизии зафиксирован
(без автогенерации в рантайме), поэтому миграция детерминирована; имя файла —
просто слаг, alembic находит ревизию по значению ``revision`` ниже.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Создаёт три таблицы meta-store и их индексы."""

    op.create_table(
        "snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("repo", sa.String(), nullable=False, server_default=""),
        sa.Column("acl", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.create_index("ix_snapshots_status", "snapshots", ["status"])
    op.create_index("ix_snapshots_repo", "snapshots", ["repo"])

    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("acl", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index("ix_documents_snapshot_id", "documents", ["snapshot_id"])
    op.create_index("ix_documents_snapshot_id_path", "documents", ["snapshot_id", "path"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("acl", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_snapshot_id", "chunks", ["snapshot_id"])


def downgrade() -> None:
    """Удаляет всё, что создал upgrade, в обратном порядке."""

    op.drop_index("ix_chunks_snapshot_id", table_name="chunks")
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index("ix_documents_snapshot_id_path", table_name="documents")
    op.drop_index("ix_documents_snapshot_id", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_snapshots_status", table_name="snapshots")
    op.drop_table("snapshots")

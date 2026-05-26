"""Окружение alembic-миграций для Postgres-based meta-store.

Целится в метаданные Base из common.db.models и берёт URL базы из конфигурации
meta-store — переменной окружения ``GIT_ASSIST_META_URL``, с фоллбэком на
``sqlalchemy.url`` из ``alembic.ini`` для разового запуска из консоли — вместо
захардкоженного DSN. Импорт этого модуля не открывает соединение; оно
устанавливается, только когда alembic реально накатывает миграцию.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from common.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_URL_ENV_VAR = "GIT_ASSIST_META_URL"


def _database_url() -> str:
    """Достаёт URL meta-store из окружения или из ``alembic.ini``."""

    url = os.environ.get(_URL_ENV_VAR) or config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError(f"no database URL: set {_URL_ENV_VAR} or sqlalchemy.url in alembic.ini")
    return url


def run_migrations_offline() -> None:
    """Прогоняет миграции без живого соединения — просто печатает SQL по URL."""

    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Прогоняет миграции через живой движок SQLAlchemy."""

    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

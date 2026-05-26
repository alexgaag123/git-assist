"""Адаптеры хранилищ поверх настоящей базы данных.

Единственная реализация MetaStore здесь — PostgresMetaStore, за протоколом MetaStore.
Импорт этого пакета не должен открывать соединение с базой.
"""

from __future__ import annotations

from common.stores import MetaStore


def make_meta_store(url: str) -> MetaStore:
    """Выбирает реализацию MetaStore по строке подключения ``url``.

    - ``postgresql...`` (``postgresql://`` / ``postgresql+psycopg://``) →
      PostgresMetaStore поверх движка, который строится лениво из ``url`` — без
      соединения и без импорта драйвера в момент вызова.
    - всё остальное → ValueError с указанием неподдерживаемой схемы.

    Вызывающий код выбирает бэкенд через ``settings.meta_url``, а не
    импортирует конкретный класс стора напрямую.
    """

    if url.startswith("postgresql"):
        from common.db.postgres_meta import PostgresMetaStore

        return PostgresMetaStore(url=url)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported meta-store URL scheme: {scheme!r} (url={url!r})")

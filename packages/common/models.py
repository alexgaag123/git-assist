"""Основные доменные записи, которые передаются между ингестом, индексацией, хранением и поиском.

Замороженные dataclass с детерминированными id, без I/O. Эти записи пересекают границы
всех пакетов, поэтому должны легко создаваться, сравниваться, а их id — быть стабильными
на любой машине и в любом запуске.

Все id строятся по одной схеме — sha256 от частей, склеенных через перевод строки, обрезанный
до 16 hex-символов — так же, как в common/snapshot.py, чтобы любой идентификатор в системе
собирался одинаково детерминированным способом.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


def _hash_id(*parts: str) -> str:
    """Детерминированный sha256[:16] от частей parts, склеенных через перевод строки."""

    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class FileKind(StrEnum):
    """Как файл трактуется при ингесте и поиске."""

    CODE = "code"
    DOC = "doc"
    CONFIG = "config"
    OTHER = "other"


@dataclass(frozen=True)
class Document:
    """Текст одного файла внутри снапшота.

    doc_id — детерминированный хеш (snapshot_id, path), поэтому один и тот же файл в
    одном и том же снапшоте всегда даёт один и тот же id.
    """

    path: str
    kind: FileKind
    text: str
    snapshot_id: str

    @property
    def doc_id(self) -> str:
        return _hash_id(self.snapshot_id, self.path)


@dataclass(frozen=True)
class Chunk:
    """Непрерывный кусок Document — единица эмбеддинга и поиска.

    chunk_id адресуется по содержимому: это хеш (path, нормализованный текст),
    который считает слой чанкинга (content_chunk_id), а не позиционный хеш
    (doc_id, index). Независимость от позиции означает, что вставка контента перед чанком
    не меняет id последующих чанков, поэтому инкрементальный реиндекс переэмбеддит только
    то, что реально изменилось. index нужен только для порядка/отображения.
    """

    chunk_id: str
    doc_id: str
    path: str
    index: int
    text: str
    snapshot_id: str


@dataclass(frozen=True)
class Commit:
    """Метаданные git-коммита — вход для анализа влияния коммита."""

    sha: str
    parents: tuple[str, ...]
    changed_paths: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class SymbolRef:
    """Именованное вхождение символа в файле (например, import или def)."""

    path: str
    name: str
    kind: str

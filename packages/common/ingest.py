"""Ingestion: превращение рабочего дерева в записи Document.

Первый этап потока воркера (materialize → **ingest** → chunk → embed → …).
Логика чистая, на стандартной библиотеке и детерминированная — тестируется
на временной директории без живых сервисов; apps/worker лишь оркеструет её.

Модуль начинается с classify, которая определяет FileKind файла по одному
только пути. Обход дерева, производящий сами Document, строится поверх неё.
"""

from __future__ import annotations

import os

from common.models import Document, FileKind

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
        ".egg-info",
    }
)

_CODE_EXTS: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".hpp",
        ".rb",
        ".php",
        ".cs",
        ".scala",
        ".swift",
        ".sh",
        ".bash",
    }
)

_DOC_EXTS: frozenset[str] = frozenset({".md", ".markdown", ".rst", ".txt", ".adoc"})

_CONFIG_EXTS: frozenset[str] = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".properties",
    }
)

_CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "dockerfile",
        "makefile",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
    }
)


def classify(path: str) -> FileKind:
    """Определить FileKind файла по одному пути.

    Решение принимается только по пути: сначала расширение файла (без учёта
    регистра), затем короткий список известных базовых имён для файлов без
    расширения или с неинформативным расширением. Всё нераспознанное — OTHER.
    """

    name = os.path.basename(path).lower()
    if name in _CONFIG_NAMES:
        return FileKind.CONFIG

    ext = os.path.splitext(name)[1]
    if ext in _CODE_EXTS:
        return FileKind.CODE
    if ext in _DOC_EXTS:
        return FileKind.DOC
    if ext in _CONFIG_EXTS:
        return FileKind.CONFIG
    return FileKind.OTHER


def _read_text(path: str) -> str | None:
    """Прочитать path как UTF-8, вернуть None, если файл похож на бинарный.

    Файл считается бинарным (и пропускается), если содержит нулевой байт или
    не декодируется как строгий UTF-8.
    """

    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def ingest_tree(root: str, snapshot_id: str, *, include_other: bool = False) -> list[Document]:
    """Обойти root и построить по одному Document на каждый сохранённый файл.

    Обход детерминированный и рекурсивный, директории VCS/шумовые (кэши, venv,
    билд-артефакты — см. _SKIP_DIRS) пропускаются по имени. Каждый обычный
    файл классифицируется, читается как UTF-8 и превращается в Document, чей
    path — относительный к root, со слэшами вперёд.

    Бинарные файлы (нулевой байт или невалидный UTF-8) пропускаются. Файлы
    FileKind.OTHER пропускаются, если не задан include_other. Результат
    отсортирован по path.
    """

    documents: list[Document] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for filename in sorted(filenames):
            abs_path = os.path.join(dirpath, filename)
            if not os.path.isfile(abs_path):
                continue
            kind = classify(abs_path)
            if kind is FileKind.OTHER and not include_other:
                continue
            text = _read_text(abs_path)
            if text is None:
                continue
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            documents.append(Document(path=rel_path, kind=kind, text=text, snapshot_id=snapshot_id))

    documents.sort(key=lambda d: d.path)
    return documents

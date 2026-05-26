"""Chunking: разбиение Document на ограниченные перекрывающиеся окна.

Этап между ingestion и embedding в потоке воркера (ingest → **chunk** →
embed → …). Логика чистая, на стандартной библиотеке и детерминированная —
тестируется без живых сервисов; apps/worker лишь оркеструет её позже.

chunk_document структурный для кода и документации: код режется по
верхнеуровневым сущностям tree-sitter (один чанк на сущность, плюс ведущий
преамбульный чанк для текста модульного уровня до первой сущности),
markdown режется по секциям (по заголовкам), и только простой текст — или
код, чей язык не поддерживается или не парсится — откатывается на скользящее
символьное окно chunk_text. Этот откат никогда не должен незаметно стать
основным путём. chunk_documents разворачивает это по всему корпусу.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from common.models import Chunk, Document, _hash_id
from common.structure import language_for_path, parse_entities

_MARKDOWN_EXTS: frozenset[str] = frozenset({".md", ".markdown"})


def normalize_chunk_text(text: str) -> str:
    """Детерминированно нормализовать text для адресации по содержимому.

    Приводит окончания строк (\\r\\n/\\r → \\n), обрезает завершающие
    пробелы в каждой строке и убирает пустые строки в начале/конце. Два
    чанка, различающихся только случайными пробелами, поэтому хешируются в
    один и тот же content_chunk_id.
    """

    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def content_chunk_id(path: str, text: str) -> str:
    """Id чанка, адресованный по содержимому: _hash_id(path, normalize_chunk_text(text)).

    Id зависит только от пути файла и нормализованного содержимого чанка —
    никогда от позиции, doc_id или snapshot_id, поэтому вставка контента перед
    чанком оставляет его id неизменным между переиндексациями. path
    ограничивает id по файлу, чтобы одинаковый контент в двух файлах
    оставался различным в сторах, ключом которых служит chunk_id.
    """

    return _hash_id(path, normalize_chunk_text(text))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Разбить text на перекрывающиеся символьные окна.

    Детерминированное скользящее окно: окна начинаются на 0, step, 2*step, …,
    где step = chunk_size - chunk_overlap, каждое окно —
    text[start:start + chunk_size]. Обход останавливается, как только окно
    достигает конца текста, так что лишнее хвостовое окно (целиком
    содержащееся в предыдущем) не порождается.

    Пустой текст даёт []; текст короче chunk_size даёт ровно один чанк (весь
    текст целиком). Вызывающий код передаёт chunk_size/chunk_overlap из
    Settings, который гарантирует chunk_size > 0 и
    0 <= chunk_overlap < chunk_size (то есть step >= 1).
    """

    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks


def _entity_chunk_texts(text: str, language: str) -> list[str] | None:
    """Разрезать text по верхнеуровневым сущностям tree-sitter, либо None для отката.

    Возвращает по одной строке на верхнеуровневую сущность (её полный байтовый
    диапазон, включая ведущие декораторы/ключевое слово export) в порядке
    исходника, перед которыми — один преамбульный чанк для непустого текста
    модульного уровня до первой сущности. Возвращает None, если язык не дал
    ни одной распарсенной сущности (не поддерживается, ошибка парсинга или
    нет верхнеуровневых объявлений) — тогда вызывающий код откатывается на
    символьное окно, и это единственный путь к такому откату.
    """

    entities = parse_entities(text, language)
    if not entities:
        return None

    source = text.encode("utf-8")
    texts: list[str] = []
    preamble = source[: entities[0].start_byte].decode("utf-8", "replace")
    if preamble.strip():
        texts.append(preamble)
    texts.extend(
        source[entity.start_byte : entity.end_byte].decode("utf-8", "replace")
        for entity in entities
    )
    return texts


def _markdown_section_texts(text: str) -> list[str]:
    """Разбить markdown-text на секции: заголовок и тело до следующего заголовка.

    ATX-заголовок — это строка, начинающаяся с #. Контент до первого
    заголовка становится собственной ведущей секцией. Пустые секции
    отбрасываются. Текст без заголовков даёт одну секцию (весь документ).
    """

    sections: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [section for section in sections if section.strip()]


def _dispatch_chunk_texts(document: Document, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Выбрать стратегию нарезки для document и вернуть сырые тексты чанков.

    Код с поддерживаемым языком и ≥1 распарсенной сущностью → чанки по
    сущностям; markdown → чанки по секциям; всё остальное (включая код,
    который не парсится) → символьное окно.
    """

    language = language_for_path(document.path)
    if language is not None:
        entity_texts = _entity_chunk_texts(document.text, language)
        if entity_texts is not None:
            return entity_texts
    elif os.path.splitext(document.path)[1].lower() in _MARKDOWN_EXTS:
        sections = _markdown_section_texts(document.text)
        if sections:
            return sections
    return chunk_text(document.text, chunk_size, chunk_overlap)


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Разбить document на записи Chunk.

    Диспетчеризация по языку/типу: код → один чанк на верхнеуровневую
    сущность tree-sitter (плюс преамбульный чанк модульного уровня),
    markdown → один чанк на секцию, иначе → символьное окно chunk_text
    (документированный откат, достижимый только когда структурный парсер не
    применим). Каждый чанк несёт doc_id, path и snapshot_id документа, с
    нулевым index по порядку нарезки и id, адресованным по содержимому
    (content_chunk_id), стабильным между переиндексациями и не зависящим от
    позиции. Документ с пустым текстом даёт [].
    """

    doc_id = document.doc_id
    return [
        Chunk(
            chunk_id=content_chunk_id(document.path, text),
            doc_id=doc_id,
            path=document.path,
            index=index,
            text=text,
            snapshot_id=document.snapshot_id,
        )
        for index, text in enumerate(_dispatch_chunk_texts(document, chunk_size, chunk_overlap))
    ]


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Развернуть chunk_document по documents, сохраняя порядок ввода.

    Удобная обёртка для пайплайна индексации; без дополнительной сортировки
    (вызывающий код передаёт уже детерминированный, отсортированный по path
    список из ingest_tree). index каждого документа начинается заново с 0.
    """

    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size, chunk_overlap)
    ]

"""Структурный парсинг через tree-sitter: исходник файла → его верхнеуровневые сущности.

Структурный аналог символьного окна из common.chunk. Код парсится через tree-sitter до
верхнеуровневых деклараций (функции, классы, …), чтобы резка на чанки могла проходить по
настоящим синтаксическим границам. tree-sitter парсит прямо в процессе, без внешних сервисов.

Грамматики импортируются лениво внутри parse_entities, чтобы импорт пакета common
оставался дешёвым, а зависимость от tree-sitter была нужна только там, где парсинг реально
происходит. Неизвестный язык или любая ошибка парсинга дают пустой список сущностей —
тогда чанкинг откатывается на обычное символьное окно, и только в этом случае.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node


@dataclass(frozen=True)
class Entity:
    """Верхнеуровневая декларация в исходном файле.

    start_line/end_line — 0-based и включительные; start_byte/end_byte — это диапазон
    байтов UTF-8 всей декларации целиком (включая ведущие декораторы или ключевое слово
    export), поэтому чанк, вырезанный по этому диапазону, самодостаточен.
    """

    kind: str
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "tsx",
}
"""Расширение файла → имя языка. Поддерживаются только эти четыре."""

_KIND_BY_TYPE: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
    "typescript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "type_alias_declaration": "type",
    },
}
"""На каждый язык — карта тип узла tree-sitter → свободная строка kind, которую мы
записываем. Сущностями становятся только узлы, чей тип есть в этой карте; всё остальное
(импорты, объявления пакета, обычные выражения) пропускается."""

_KIND_BY_TYPE["tsx"] = _KIND_BY_TYPE["typescript"]
"""tsx использует ту же грамматику деклараций, что и TypeScript."""

_WRAPPER_TYPES = frozenset({"decorated_definition", "export_statement"})
"""Узлы-обёртки, под которыми настоящая декларация лежит как дочерний узел (декораторы
Python, export в TS)."""


def language_for_path(path: str) -> str | None:
    """Язык tree-sitter по расширению path, либо None, если оно не поддерживается."""

    _, ext = os.path.splitext(path)
    return _LANG_BY_EXT.get(ext.lower())


def _grammar(language: str):  # type: ignore[no-untyped-def]
    """Лениво импортирует грамматику для language и возвращает её Language.

    Импорт происходит здесь, а не на уровне модуля, чтобы зависимость от tree-sitter
    трогалась только тогда, когда парсинг реально запускается. KeyError для неизвестного
    языка обрабатывается вызывающей стороной.
    """

    from tree_sitter import Language

    if language == "python":
        import tree_sitter_python as ts_python

        return Language(ts_python.language())
    if language == "go":
        import tree_sitter_go as ts_go

        return Language(ts_go.language())
    if language == "typescript":
        import tree_sitter_typescript as ts_typescript

        return Language(ts_typescript.language_typescript())
    if language == "tsx":
        import tree_sitter_typescript as ts_typescript

        return Language(ts_typescript.language_tsx())
    raise KeyError(language)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _kind_and_name(node: Node, kinds: dict[str, str], source: bytes) -> tuple[str, str] | None:
    """Определяет (kind, name) верхнеуровневого узла, разворачивая декораторы/export.

    Возвращает None, если узел — не распознанная декларация или у него нет извлекаемого
    имени (например, деструктурирующий export const {a, b} = …), такие узлы просто
    пропускаются. Для Go имя лежит на уровень глубже (type_declaration → type_spec → имя),
    поэтому при отсутствии имени у самого узла ищем его среди именованных детей.
    """

    decl = node
    if decl.type in _WRAPPER_TYPES:
        inner = next((c for c in decl.named_children if c.type in kinds), None)
        if inner is None:
            return None
        decl = inner

    kind = kinds.get(decl.type)
    if kind is None:
        return None

    name_node = decl.child_by_field_name("name")
    if name_node is None:
        for child in decl.named_children:
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                break
    if name_node is None:
        return None

    return kind, _node_text(name_node, source)


def parse_entities(text: str, language: str) -> list[Entity]:
    """Парсит text как language → список верхнеуровневых сущностей в порядке исходника.

    Неизвестный язык, ошибка импорта грамматики или любая синтаксическая ошибка в text
    дают пустой список (вызывающая сторона откатывается на символьное окно). Возвращаются
    только верхнеуровневые декларации; вложенные определения и недекларативные выражения
    игнорируются.
    """

    kinds = _KIND_BY_TYPE.get(language)
    if kinds is None:
        return []

    try:
        from tree_sitter import Parser

        parser = Parser(_grammar(language))
        source = text.encode("utf-8")
        root = parser.parse(source).root_node
    except Exception:
        return []

    if root.has_error:
        return []

    entities: list[Entity] = []
    for node in root.named_children:
        resolved = _kind_and_name(node, kinds, source)
        if resolved is None:
            continue
        kind, name = resolved
        entities.append(
            Entity(
                kind=kind,
                name=name,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            )
        )
    return entities

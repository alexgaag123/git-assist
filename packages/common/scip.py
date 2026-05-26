"""SCIP cross-references: точные, компиляторные occurrences символов → рёбра зависимостей.

Обычный граф импортов грубый — он работает на уровне модулей и резолвится по имени.
SCIP-индексаторы (scip-python / scip-typescript / scip-go) выдают occurrences компиляторного
качества: каждый помечен глобальной SCIP-строкой символа (полным дескриптором), поэтому два
occurrence считаются одним и тем же символом только когда так говорит компилятор — а не когда
у них случайно совпали голые имена. Этот модуль превращает такие occurrences в GraphEdge вида
"reference".

Преобразование чистое, как и build_graph_edges: на вход — уже декодированные
SymbolOccurrence, на выход — рёбра. Декодирование SCIP protobuf и запуск самих индексаторов
(scip-python/-typescript/-go) — забота обвязки воркера; здесь только построитель рёбер
внутри common.

Направление рёбер как везде в системе: ребро (src, dst) значит "src зависит от dst". Файл,
который *ссылается* на символ, зависит от файла, где символ *определён* — то есть ссылка в B
на символ, определённый в A, даёт ребро B -> A.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from common.stores import GraphEdge

REFERENCE_EDGE_KIND = "reference"
"""Значение GraphEdge.kind для компиляторной cross-reference (в отличие от обычного "import")."""


class OccurrenceRole(StrEnum):
    """Определяет ли occurrence символ или просто ссылается на него."""

    DEFINITION = "definition"
    REFERENCE = "reference"


@dataclass(frozen=True)
class SymbolOccurrence:
    """Одно вхождение SCIP-символа в файле.

    symbol — глобальная SCIP-строка символа (полный дескриптор), поэтому равенство —
    это компиляторное тождество, а не совпадение голых имён. role различает место
    определения и место использования. Диапазоны строк/символов намеренно опущены: граф
    ссылок работает от файла к файлу, а диапазоны только добавили бы шум, который построитель
    рёбер всё равно отбрасывает.
    """

    path: str
    symbol: str
    role: OccurrenceRole


def scip_reference_edges(
    occurrences: Iterable[SymbolOccurrence],
    snapshot_id: str,
) -> list[GraphEdge]:
    """Резолвит SCIP-occurrences в рёбра "reference" файл→файл внутри одного снапшота.

    Occurrences группируются по symbol на множество путей, где символ определён, и
    множество путей, где на него ссылаются. Для каждого символа каждая пара (путь ссылки,
    путь определения) с разными путями даёт GraphEdge(src=путь_ссылки, dst=путь_определения,
    kind="reference", snapshot_id) — ссылающийся файл зависит от файла с определением.

    Occurrences в одном и том же файле ребра не дают, и символ без определения в этом
    индексе (или у которого все ссылки — в его же файле определения) ничего не добавляет.
    Рёбра дедуплицируются и возвращаются отсортированными по (src, dst), как и в
    build_graph_edges.
    """

    definitions: dict[str, set[str]] = {}
    references: dict[str, set[str]] = {}
    for occurrence in occurrences:
        bucket = definitions if occurrence.role is OccurrenceRole.DEFINITION else references
        bucket.setdefault(occurrence.symbol, set()).add(occurrence.path)

    edges: set[GraphEdge] = set()
    for symbol, def_paths in definitions.items():
        for ref_path in references.get(symbol, set()):
            for def_path in def_paths:
                if ref_path != def_path:
                    edges.add(
                        GraphEdge(
                            src=ref_path,
                            dst=def_path,
                            kind=REFERENCE_EDGE_KIND,
                            snapshot_id=snapshot_id,
                        )
                    )

    return sorted(edges, key=lambda e: (e.src, e.dst))

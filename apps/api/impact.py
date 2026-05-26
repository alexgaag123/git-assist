"""Анализ влияния коммита: что затронется ниже по цепочке зависимостей.

Граф зависимостей (рёбра import между файлами внутри снапшота) уже построен на этапе
индексации — этот модуль его обходит: коммит → изменённые файлы → обратное замыкание
зависимостей → итоговый impact-набор, в рамках ACTIVE-снапшота нужного репозитория.

Импортирует только common.*, никогда apps/worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.graph import transitive_dependents
from common.models import Commit
from common.stores import GraphStore, MetaStore


@dataclass(frozen=True)
class ImpactResult:
    """Итог анализа влияния коммита на конкретный снапшот.

    snapshot_id — снапшот, по которому считали ("" если активного не было). changed —
    изменённые пути коммита, отсортированные и без дублей (это семена обхода). impacted —
    транзитивное замыкание обратных зависимостей, отсортировано и не включает changed.
    """

    snapshot_id: str
    changed: tuple[str, ...]
    impacted: tuple[str, ...]

    @property
    def affected(self) -> tuple[str, ...]:
        """Объединение changed и impacted, отсортированное — полный радиус поражения."""

        return tuple(sorted(set(self.changed) | set(self.impacted)))


def analyze_impact(
    commit: Commit,
    *,
    repo: str,
    graph_store: GraphStore,
    meta_store: MetaStore,
    max_depth: int | None = None,
) -> ImpactResult:
    """Считает, что затронется при изменении commit, относительно активного снапшота repo.

    Семена обхода — изменённые пути коммита, отсортированные и без дублей (repo-relative
    POSIX-пути, совпадающие с ключами узлов графа). Если для repo нет активного снапшота —
    графа для обхода нет, impacted пуст, а changed просто эхом повторяет вход.

    Иначе impact-набор считается как transitive_dependents от семян, отсортированный.
    max_depth ограничивает глубину обхода (1 — только прямые зависимые, None — полное
    замыкание).
    """

    changed = tuple(sorted(set(commit.changed_paths)))

    snapshot_id = meta_store.active_snapshot(repo)
    if snapshot_id is None:
        return ImpactResult(snapshot_id="", changed=changed, impacted=())

    impacted = transitive_dependents(graph_store, snapshot_id, changed, max_depth=max_depth)
    return ImpactResult(snapshot_id=snapshot_id, changed=changed, impacted=tuple(sorted(impacted)))

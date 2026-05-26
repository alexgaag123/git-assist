"""Основа снапшотов.

Снапшот — это один неизменяемый билд индекса над репозиторием на конкретном коммите.
Каждая точка в Qdrant, узел в Neo4j и строка в Postgres помечены snapshot_id. Запросы
всегда читают только ACTIVE-снапшот; замена собирается в фоне через жизненный цикл
ниже и переключается атомарно.

Жизненный цикл детерминированный: одинаковый ввод всегда даёт один и тот же snapshot_id,
и переход между статусами не зависит от порядка вызовов внешних сторов.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class SnapshotStatus(StrEnum):
    """Жизненный цикл снапшота. В запросах участвует только ACTIVE."""

    PENDING = "pending"
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"


_TRANSITIONS: dict[SnapshotStatus, frozenset[SnapshotStatus]] = {
    SnapshotStatus.PENDING: frozenset({SnapshotStatus.BUILDING, SnapshotStatus.RETIRED}),
    SnapshotStatus.BUILDING: frozenset({SnapshotStatus.READY, SnapshotStatus.RETIRED}),
    SnapshotStatus.READY: frozenset({SnapshotStatus.ACTIVE, SnapshotStatus.RETIRED}),
    SnapshotStatus.ACTIVE: frozenset({SnapshotStatus.RETIRED}),
    SnapshotStatus.RETIRED: frozenset(),
}


def can_transition(current: SnapshotStatus, target: SnapshotStatus) -> bool:
    """Проверяет, разрешён ли переход из current в target.

    Разрешены только переходы вперёд: снапшот никогда не двигается назад, а неудачный
    билд ретайрится, а не откатывается. Активация (READY -> ACTIVE) и понижение
    предыдущего активного снапшота (ACTIVE -> RETIRED) — это одно атомарное переключение.
    """

    return target in _TRANSITIONS[current]


@dataclass(frozen=True)
class SnapshotRef:
    """Естественный ключ снапшота: (repo, branch, commit_sha).

    snapshot_id — детерминированный хеш этой тройки, поэтому один и тот же коммит
    на одной и той же ветке всегда даёт один и тот же id, на любой машине и в любом запуске.
    """

    repo: str
    branch: str
    commit_sha: str

    @property
    def snapshot_id(self) -> str:
        payload = "\n".join((self.repo, self.branch, self.commit_sha))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

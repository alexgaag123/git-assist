"""Реализация GraphStore поверх Neo4j.

Модуль содержит адаптер графового хранилища и чистые вспомогательные функции,
которые превращают домен (GraphEdge) в Cypher-запросы и обратно. Хелперы не зависят от
клиента — их можно импортировать без пакета neo4j и без запущенного сервера; сам адаптер
работает поверх переданного драйвера.

Что важно знать про устройство хелперов:

- Граф использует один лейбл узла, NODE_LABEL, узлы несут поля key (непрозрачный
  идентификатор файла или символа) и snapshot_id. Все рёбра зависимости — один тип связи,
  DEPENDS_ON_TYPE, с полями kind (например "import" / "reference") и snapshot_id. Таким
  образом граф полностью изолирован по snapshot_id.
- add_edges делает MERGE обеих конечных вершин ребра и самой связи по ключу
  (src, dst, kind, snapshot_id) — вершины фиксируются по src/dst, а свойства связи по
  kind/snapshot_id. Повторное добавление того же ребра идемпотентно.
- dependents находит обратных соседей узла в рамках снапшота: все src, у которых есть
  ребро src -> node (то есть src зависит от node), возвращаются под ключом
  DEPENDENTS_RESULT_KEY.
- delete_snapshot делает DETACH DELETE для каждого узла с данным тегом снапшота, вместе с
  узлом удаляются и все его связи (все рёбра снапшота соединяют узлы этого же снапшота).
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from common.stores import GraphEdge


class _Neo4jSession(Protocol):
    """Минимальная форма сессии драйвера: run плюс вход/выход как менеджер контекста."""

    def run(
        self, query: str, parameters: Mapping[str, object]
    ) -> Iterable[Mapping[str, object]]: ...

    def __enter__(self) -> _Neo4jSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None: ...


class _Neo4jDriver(Protocol):
    """Минимальная форма драйвера: фабрика session(), возвращающая _Neo4jSession."""

    def session(self) -> _Neo4jSession: ...


NODE_LABEL = "Node"
"""Лейбл каждого узла графа; узлы несут поля key и snapshot_id."""

DEPENDS_ON_TYPE = "DEPENDS_ON"
"""Тип связи каждого ребра зависимости; связи несут поля kind и snapshot_id."""

DEPENDENTS_RESULT_KEY = "key"
"""Псевдоним колонки, которую возвращает DEPENDENTS_CYPHER и читает records_to_dependents."""

ADD_EDGES_CYPHER = (
    "UNWIND $edges AS edge "
    f"MERGE (src:{NODE_LABEL} {{key: edge.src, snapshot_id: edge.snapshot_id}}) "
    f"MERGE (dst:{NODE_LABEL} {{key: edge.dst, snapshot_id: edge.snapshot_id}}) "
    f"MERGE (src)-[:{DEPENDS_ON_TYPE} {{kind: edge.kind, snapshot_id: edge.snapshot_id}}]->(dst)"
)
"""Идемпотентно MERGE-ит обе вершины и связь каждого ребра (ключ — полный кортеж полей)."""

DEPENDENTS_CYPHER = (
    f"MATCH (src:{NODE_LABEL} {{snapshot_id: $snapshot_id}})"
    f"-[:{DEPENDS_ON_TYPE} {{snapshot_id: $snapshot_id}}]->"
    f"(dst:{NODE_LABEL} {{key: $node, snapshot_id: $snapshot_id}}) "
    f"RETURN DISTINCT src.key AS {DEPENDENTS_RESULT_KEY}"
)
"""Обратные соседи узла $node в рамках $snapshot_id (узлы, которые от него зависят)."""

DELETE_SNAPSHOT_CYPHER = f"MATCH (n:{NODE_LABEL} {{snapshot_id: $snapshot_id}}) DETACH DELETE n"
"""Удаляет каждый узел с данным $snapshot_id (а через DETACH — и его связи)."""


def add_edges_params(edges: Sequence[GraphEdge]) -> dict[str, list[dict[str, str]]]:
    """Параметры для ADD_EDGES_CYPHER: одна строка на каждое ребро из edges."""

    return {
        "edges": [
            {
                "src": edge.src,
                "dst": edge.dst,
                "kind": edge.kind,
                "snapshot_id": edge.snapshot_id,
            }
            for edge in edges
        ]
    }


def dependents_params(snapshot_id: str, node: str) -> dict[str, str]:
    """Параметры для DEPENDENTS_CYPHER."""

    return {"snapshot_id": snapshot_id, "node": node}


def delete_snapshot_params(snapshot_id: str) -> dict[str, str]:
    """Параметры для DELETE_SNAPSHOT_CYPHER."""

    return {"snapshot_id": snapshot_id}


def records_to_dependents(records: Iterable[Mapping[str, object]]) -> set[str]:
    """Превращает записи результата DEPENDENTS_CYPHER в множество ключей зависимых узлов."""

    return {str(record[DEPENDENTS_RESULT_KEY]) for record in records}


class Neo4jGraphStore:
    """GraphStore поверх базы Neo4j.

    Реализует протокол GraphStore: идемпотентный add_edges (Cypher MERGE по полному
    ключу src/dst/kind/snapshot_id), обратный dependents в рамках одного снапшота и
    scoped delete_snapshot через DETACH DELETE.

    Конструктор не открывает соединение. Передай либо готовый драйвер (любой объект, чей
    session() удовлетворяет _Neo4jDriver), либо url, из которого драйвер строится лениво
    при первом использовании (_build_driver). Если не передать ни то, ни другое — будет
    ValueError.
    """

    def __init__(
        self,
        *,
        driver: _Neo4jDriver | None = None,
        url: str | None = None,
        auth: tuple[str, str] | None = None,
    ) -> None:
        if driver is None and url is None:
            raise ValueError("Neo4jGraphStore requires either a driver or a url")
        self._driver = driver
        self._url = url
        self._auth = auth

    @staticmethod
    def _build_driver(url: str, auth: tuple[str, str] | None) -> _Neo4jDriver:
        """Лениво импортирует neo4j и строит драйвер из url и auth.

        Ленивый импорт нужен, чтобы `import common.neo4j_graph` не тянул за собой пакет
        neo4j без необходимости (например, если драйвер инжектирован напрямую). cast
        говорит mypy, что настоящий neo4j.Driver удовлетворяет нашему узкому структурному
        протоколу _Neo4jDriver — это действительно так на рантайме (session() совместим, а
        более богатые keyword-only параметры реального драйвера — просто надмножество
        того, что нам нужно).
        """
        from typing import cast  # noqa: PLC0415

        from neo4j import GraphDatabase  # noqa: PLC0415

        return cast(_Neo4jDriver, GraphDatabase.driver(url, auth=auth))

    def _get_driver(self) -> _Neo4jDriver:
        """Возвращает драйвер, лениво строя его из url при первом обращении."""

        if self._driver is None:
            assert self._url is not None
            self._driver = self._build_driver(self._url, self._auth)
        return self._driver

    def add_edges(self, edges: Sequence[GraphEdge]) -> None:
        """Идемпотентно MERGE-ит каждое ребро; для пустого списка — no-op без похода к базе."""

        if not edges:
            return
        with self._get_driver().session() as session:
            session.run(ADD_EDGES_CYPHER, add_edges_params(edges))

    def dependents(self, snapshot_id: str, node: str) -> set[str]:
        """Прямые зависимые узла node в рамках snapshot_id (узлы с ребром src -> node)."""

        with self._get_driver().session() as session:
            result = session.run(DEPENDENTS_CYPHER, dependents_params(snapshot_id, node))
            return records_to_dependents(result)

    def delete_snapshot(self, snapshot_id: str) -> None:
        """DETACH DELETE для каждого узла, помеченного данным snapshot_id."""

        with self._get_driver().session() as session:
            session.run(DELETE_SNAPSHOT_CYPHER, delete_snapshot_params(snapshot_id))

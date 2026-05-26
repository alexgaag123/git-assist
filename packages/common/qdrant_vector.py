"""Реализация VectorStore поверх Qdrant.

Здесь живёт адаптер векторного хранилища (QdrantVectorStore) и чистые функции
преобразования между доменными VectorRecord/SearchHit и точками/коллекциями Qdrant. Эти
функции не требуют клиента (импортируются без работающего сервера); сам адаптер построен
поверх них и может работать как против встроенного локального бэкенда ``:memory:``,
так и против настоящего сервера.

Ключевые решения:

- Коллекция использует один именованный плотный вектор (DENSE_VECTOR_NAME), размер которого
  берётся из settings.embed_dim (никогда не хардкодится — см. dense_vector_params). Разреженный
  bm25-вектор и более богатый payload (путь / acl / диапазон строк) пока не реализованы.
- Id точки в Qdrant обязан быть либо беззнаковым int, либо UUID, а chunk_id — это 16-значная
  hex-строка. point_id детерминированно превращает chunk_id в UUID, поэтому повторный upsert
  того же чанка обновляет ту же точку (идемпотентность). Настоящий chunk_id лежит в payload,
  чтобы scored_point_to_hit мог восстановить его при поиске.

Как пайплайн поиска мог бы лечь на Query API Qdrant (пока не сделано):

retrieve_pipeline гоняет двухэтапный поиск (prefetch → fusion → boost → rerank)
полностью на клиенте. Ту же логическую схему можно перенести на серверный Query API
Qdrant так:

1. Два параллельных prefetch-запроса (dense и в перспективе sparse-bm25), каждый как
   отдельный models.Prefetch с фильтром по snapshot_id/acl_tags и лимитом prefetch_limit
   (~30). Разреженного BM25-вектора пока нет — можно временно гонять два dense prefetch
   с разными векторами запроса как структурную заглушку: механизм prefetch+RRF от этого
   не меняется.
2. Обе ветки несут одинаковый must-фильтр по snapshot_id (плюс в будущем acl_tags — ACL в
   payload тоже пока не реализован, но привязка уже задокументирована здесь).
   _snapshot_filter строит фильтрующую половину, отвечающую за snapshot_id.
3. Внешний вызов query_points передаёт обе prefetch-ветки и просит слить их на сервере через
   FusionQuery(fusion=Fusion.RRF). Математика RRF в Qdrant идентична reciprocal_rank_fusion
   из common.fusion (только по позиции в ранге, k=60), так что серверное и клиентское
   слияние семантически эквивалентны.
4. Кандидаты (ScoredPoint), которые вернул query_points, дальше на клиенте гидрируются,
   получают буст по точному совпадению имени символа и реранкаются через Reranker
   (Qwen/Qwen3-Reranker-0.6B) до итогового limit (~15) — точно так же, как делает
   retrieve_pipeline. Финальный порядок — это попарный скор реранкера, ничьи разбиваются
   по возрастанию chunk_id.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from common.stores import SearchHit, VectorRecord

DENSE_VECTOR_NAME = "dense"
"""Имя плотного вектора в коллекции Qdrant (разреженный bm25 пока не реализован)."""

PAYLOAD_CHUNK_KEY = "chunk_id"
"""Ключ payload'а с доменным chunk_id (id самой точки — это производный UUID, не он)."""

PAYLOAD_SNAPSHOT_KEY = "snapshot_id"
"""Ключ payload'а с snapshot_id владельца; по нему фильтруют search и delete."""

_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
"""Фиксированное пространство имён для детерминированного вывода UUID точки из chunk_id."""


def dense_vector_params(embed_dim: int) -> VectorParams:
    """Конфиг плотного косинусного вектора, размер берётся из embed_dim (единый источник)."""

    return VectorParams(size=embed_dim, distance=Distance.COSINE)


def point_id(chunk_id: str) -> str:
    """Детерминированный UUID (строкой) для chunk_id — стабилен между вызовами и процессами."""

    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def record_to_point(record: VectorRecord) -> PointStruct:
    """VectorRecord → точка Qdrant: производный id, именованный плотный вектор, payload."""

    return PointStruct(
        id=point_id(record.chunk_id),
        vector={DENSE_VECTOR_NAME: list(record.vector)},
        payload={
            PAYLOAD_CHUNK_KEY: record.chunk_id,
            PAYLOAD_SNAPSHOT_KEY: record.snapshot_id,
        },
    )


def scored_point_to_hit(point: ScoredPoint) -> SearchHit:
    """Найденная точка Qdrant → SearchHit, chunk_id восстанавливается из payload."""

    payload = point.payload or {}
    return SearchHit(chunk_id=str(payload[PAYLOAD_CHUNK_KEY]), score=point.score)


def _snapshot_filter(snapshot_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key=PAYLOAD_SNAPSHOT_KEY, match=MatchValue(value=snapshot_id))]
    )


class QdrantVectorStore:
    """VectorStore поверх коллекции Qdrant.

    Хранит один именованный плотный вектор (DENSE_VECTOR_NAME) на чанк, размером embed_dim,
    и отвечает на поиск похожести в рамках одного снапшота через фильтр по payload'у.
    Реализует протокол VectorStore: идемпотентный upsert по chunk_id, search в рамках
    снапшота с ранжированием по убыванию похожести, delete_snapshot тоже в рамках снапшота.

    Конструктор не открывает соединение: передай либо готовый QdrantClient, либо url, из
    которого клиент будет собран лениво при первом обращении (так же, как ленивый engine у
    PostgresMetaStore). Коллекция создаётся один раз, при первом использовании, размером embed_dim.
    """

    def __init__(
        self,
        collection_name: str,
        embed_dim: int,
        *,
        client: QdrantClient | None = None,
        url: str | None = None,
    ) -> None:
        if client is None and url is None:
            raise ValueError("QdrantVectorStore requires either a client or a url")
        self._collection_name = collection_name
        self._embed_dim = embed_dim
        self._client = client
        self._url = url
        self._collection_ready = False

    @staticmethod
    def _build_client(url: str) -> QdrantClient:
        """Собрать клиента из url; значение ``:memory:`` выбирает встроенный in-process бэкенд."""

        if url == ":memory:":
            return QdrantClient(location=":memory:")
        if url.startswith("qdrant://"):
            url = "http://" + url[len("qdrant://") :]
        return QdrantClient(url=url)

    def _get_client(self) -> QdrantClient:
        """Вернуть клиента, лениво собрав его из url при первом обращении."""

        if self._client is None:
            assert self._url is not None
            self._client = self._build_client(self._url)
        return self._client

    def _ensure_collection(self, client: QdrantClient) -> None:
        """Создать коллекцию (один раз) размером embed_dim, если она ещё не существует."""

        if self._collection_ready:
            return
        if not client.collection_exists(self._collection_name):
            client.create_collection(
                self._collection_name,
                vectors_config={DENSE_VECTOR_NAME: dense_vector_params(self._embed_dim)},
            )
        self._collection_ready = True

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        client = self._get_client()
        self._ensure_collection(client)
        client.upsert(
            self._collection_name,
            points=[record_to_point(record) for record in records],
        )

    def search(self, snapshot_id: str, query: tuple[float, ...], limit: int) -> list[SearchHit]:
        client = self._get_client()
        self._ensure_collection(client)
        response = client.query_points(
            self._collection_name,
            query=list(query),
            using=DENSE_VECTOR_NAME,
            query_filter=_snapshot_filter(snapshot_id),
            limit=limit,
            with_payload=True,
        )
        return [scored_point_to_hit(point) for point in response.points]

    def delete_snapshot(self, snapshot_id: str) -> None:
        client = self._get_client()
        self._ensure_collection(client)
        client.delete(
            self._collection_name,
            points_selector=FilterSelector(filter=_snapshot_filter(snapshot_id)),
        )

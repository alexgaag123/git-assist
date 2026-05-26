"""Ответ с опорой на контекст: сначала поиск, потом генерация с проверкой цитат.

Модуль связывает retrieve_pipeline с генератором ответов, превращает каждый найденный
чанк в ContextChunk и передаёт их в generator.generate.

Если активного снапшота нет (или ничего не нашлось), контекст пустой — генератор вернёт
статус INSUFFICIENT_CONTEXT: система никогда не выдумывает ответ, если не на что опереться.
"""

from __future__ import annotations

from api.retrieve import DEFAULT_PREFETCH_K, DEFAULT_TOP_K, retrieve_pipeline
from common.config import Settings
from common.embed import Embedder
from common.generate import Answer, ContextChunk, Generator
from common.hyde import HydeExpander
from common.rerank import Reranker
from common.stores import MetaStore, VectorStore


def answer(
    query: str,
    *,
    repo: str,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    meta_store: MetaStore,
    reranker: Reranker,
    generator: Generator,
    limit: int = DEFAULT_TOP_K,
    prefetch_limit: int = DEFAULT_PREFETCH_K,
    expander: HydeExpander | None = None,
) -> Answer:
    """Отвечает на query утверждениями, опирающимися на чанки активного снапшота repo.

    Прогоняет retrieve_pipeline, превращает каждый результат в ContextChunk и возвращает
    generator.generate(query, context). Если для repo нет ACTIVE-снапшота (или поиск
    ничего не нашёл) — контекст пуст, генератор вернёт INSUFFICIENT_CONTEXT.

    Возвращает ValueError, если размерность embedder не совпадает с settings.embed_dim.
    """
    results = retrieve_pipeline(
        query,
        repo=repo,
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        meta_store=meta_store,
        reranker=reranker,
        limit=limit,
        prefetch_limit=prefetch_limit,
        expander=expander,
    )
    context = [ContextChunk(chunk_id=r.chunk_id, text=r.text) for r in results]
    return generator.generate(query, context)

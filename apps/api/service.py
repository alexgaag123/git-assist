"""Сервисный слой API: собирает зависимости запроса и отдаёт готовые операции.

Это точка сборки apps/api. Retrieve/assemble_context и analyze_impact — чистые
библиотечные функции поверх протоколов common. Этот модуль собирает нужные им
зависимости в один ApiService и оборачивает каждую операцию в форму
запрос/ответ с to_dict(), отдающим JSON-совместимые примитивы, которые
транспортный адаптер навешивает на HTTP.

Чистая оркестрация над интерфейсами common и соседями по api.*: импортирует
только common.* и api.* (никогда apps/worker — это одна из границ пакетов проекта).
Конкретные адаптеры (Ollama/Qdrant/Neo4j/Postgres) выбираются фабриками common
по settings.*_url, поэтому этот модуль зависит только от протоколов.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.answer import answer as run_answer
from api.impact import analyze_impact
from api.retrieve import (
    DEFAULT_PREFETCH_K,
    DEFAULT_TOP_K,
    RetrievalResult,
    assemble_context,
    retrieve_pipeline,
)
from common.config import Settings
from common.embed import Embedder
from common.generate import Answer, AnswerStatus, Generator, Statement
from common.hyde import HydeExpander
from common.models import Commit
from common.rerank import Reranker
from common.stores import GraphStore, MetaStore, VectorStore


@dataclass(frozen=True)
class SearchResponse:
    """Готовый результат операции поиска.

    query возвращается как есть; snapshot_id — активный снапшот, по которому
    искали (пустая строка, если ничего не было активно); results — ранжированные
    хиты; context — собранный блок, готовый для промпта. to_dict рендерит всё в
    JSON-примитивы — у каждого результата опущен свой snapshot_id, потому что
    он и так есть на уровне всего ответа.
    """

    query: str
    snapshot_id: str
    results: tuple[RetrievalResult, ...]
    context: str

    def to_dict(self) -> dict[str, object]:
        """Отрендерить в JSON-совместимый словарь (транспортный адаптер кодирует его на провод)."""

        return {
            "query": self.query,
            "snapshot_id": self.snapshot_id,
            "results": [
                {
                    "chunk_id": result.chunk_id,
                    "path": result.path,
                    "text": result.text,
                    "score": result.score,
                }
                for result in self.results
            ],
            "context": self.context,
        }


@dataclass(frozen=True)
class AnswerResponse:
    """Готовый результат операции обоснованного ответа (grounded answer).

    query возвращается как есть; snapshot_id — активный снапшот, по которому
    отвечали (пустая строка, если ничего не было активно); status —
    AnswerStatus; statements — обоснованные утверждения с цитатами. to_dict
    рендерит всё в JSON-примитивы.
    """

    query: str
    snapshot_id: str
    status: AnswerStatus
    statements: tuple[Statement, ...]

    def to_dict(self) -> dict[str, object]:
        """Отрендерить в JSON-совместимый словарь (транспортный адаптер кодирует его на провод)."""

        return {
            "query": self.query,
            "snapshot_id": self.snapshot_id,
            "status": self.status.value,
            "statements": [
                {"text": s.text, "chunk_ids": list(s.chunk_ids)} for s in self.statements
            ],
        }


@dataclass(frozen=True)
class ImpactResponse:
    """Готовый результат операции commit-impact.

    snapshot_id — активный снапшот, по которому шёл анализ (пустая строка, если
    ничего не было активно); changed — отсортированные, без дублей изменённые
    пути (семена обхода); impacted — транзитивное замыкание обратных
    зависимостей; affected — отсортированное объединение обоих — полный радиус
    поражения. to_dict рендерит кортежи как JSON-списки.
    """

    snapshot_id: str
    changed: tuple[str, ...]
    impacted: tuple[str, ...]
    affected: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Отрендерить в JSON-совместимый словарь (транспортный адаптер кодирует его на провод)."""

        return {
            "snapshot_id": self.snapshot_id,
            "changed": list(self.changed),
            "impacted": list(self.impacted),
            "affected": list(self.affected),
        }


@dataclass(frozen=True)
class ApiService:
    """Зависимости запроса, собранные один раз, чтобы хендлерам не таскать по шесть аргументов.

    Embedder и три стора — настоящие клиенты Ollama/Qdrant/Neo4j/Postgres, за
    протоколами common. Индексация и активация снапшотов — забота apps/worker;
    этот сервис только читает активный снапшот. reranker собирается из
    settings.rerank_url фабрикой make_reranker (OllamaReranker), generator — из
    settings.gen_url через make_generator (OllamaGenerator), expander — из
    settings.hyde_url через make_expander (OllamaHydeExpander).
    """

    settings: Settings
    embedder: Embedder
    vector_store: VectorStore
    graph_store: GraphStore
    meta_store: MetaStore
    reranker: Reranker
    generator: Generator
    expander: HydeExpander

    def search(
        self,
        query: str,
        *,
        repo: str,
        limit: int = DEFAULT_TOP_K,
        max_context_chars: int | None = None,
        prefetch_limit: int = DEFAULT_PREFETCH_K,
    ) -> SearchResponse:
        """Достать ранжированные чанки под query из снапшота repo и собрать контекст.

        Обёртка над retrieve_pipeline и assemble_context. snapshot_id отдаётся
        пустой строкой, если для repo ничего не активно (retrieve_pipeline и
        так вернёт пустой список). Guard по размерности эмбеддинга всё ещё
        работает — при рассинхроне эмбеддера бросается ValueError, роутер
        превращает это в 400.
        """

        results = retrieve_pipeline(
            query,
            repo=repo,
            settings=self.settings,
            embedder=self.embedder,
            vector_store=self.vector_store,
            meta_store=self.meta_store,
            reranker=self.reranker,
            expander=self.expander,
            limit=limit,
            prefetch_limit=prefetch_limit,
        )
        snapshot_id = self.meta_store.active_snapshot(repo) or ""
        context = assemble_context(results, max_chars=max_context_chars)
        return SearchResponse(
            query=query,
            snapshot_id=snapshot_id,
            results=tuple(results),
            context=context,
        )

    def impact(
        self,
        changed_paths: Sequence[str],
        *,
        repo: str,
        max_depth: int | None = None,
    ) -> ImpactResponse:
        """Посчитать нижестоящий impact коммита, затронувшего changed_paths в repo.

        Обёртка над analyze_impact: собирает минимальный Commit (для обхода
        важны только changed_paths, sha/parents/message пустые). snapshot_id
        отдаётся пустой строкой, если для repo ничего не активно (analyze_impact
        и так вернёт пустой impacted). max_depth ограничивает глубину замыкания
        (1 — только прямые зависимые, None — полное замыкание).
        """

        commit = Commit(sha="", parents=(), changed_paths=tuple(changed_paths), message="")
        result = analyze_impact(
            commit,
            repo=repo,
            graph_store=self.graph_store,
            meta_store=self.meta_store,
            max_depth=max_depth,
        )
        return ImpactResponse(
            snapshot_id=result.snapshot_id,
            changed=result.changed,
            impacted=result.impacted,
            affected=result.affected,
        )

    def answer(
        self,
        query: str,
        *,
        repo: str,
        limit: int = DEFAULT_TOP_K,
        prefetch_limit: int = DEFAULT_PREFETCH_K,
    ) -> AnswerResponse:
        """Дать обоснованный ответ на query из активного снапшота repo.

        Обёртка над api.answer.answer, которая гоняет поиск и генерацию.
        snapshot_id отдаётся пустой строкой, если для repo ничего не активно
        (генератор в этом случае вернёт AnswerStatus.INSUFFICIENT_CONTEXT).
        ValueError из guard'а по размерности эмбеддинга пробрасывается до
        роутера, который превращает его в 400.
        """

        result: Answer = run_answer(
            query,
            repo=repo,
            settings=self.settings,
            embedder=self.embedder,
            vector_store=self.vector_store,
            meta_store=self.meta_store,
            reranker=self.reranker,
            generator=self.generator,
            limit=limit,
            prefetch_limit=prefetch_limit,
            expander=self.expander,
        )
        snapshot_id = self.meta_store.active_snapshot(repo) or ""
        return AnswerResponse(
            query=query,
            snapshot_id=snapshot_id,
            status=result.status,
            statements=result.statements,
        )

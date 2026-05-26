"""FastAPI ASGI-обвязка вебхука воркера.

create_app возвращает приложение с двумя роутами: GET /health — проверка живости, и
POST /webhook — принимает push-события GitLab, проверяет заголовок X-Gitlab-Token и
запускает build_snapshot фоновой задачей, сразу отвечая 202.

build_service собирает WorkerService из Settings через make_*-фабрики; все они ленивые —
соединение при сборке ещё не открывается.

Модульная переменная app строится из load_settings() при импорте. Фабрики строят клиенты
лениво — ни одно соединение не открывается при импорте, так что импортировать worker.asgi
безопасно и без живых сервисов.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from common.config import Settings, load_settings
from common.db import make_meta_store
from common.embed import Embedder, make_embedder
from common.stores import (
    GraphStore,
    MetaStore,
    VectorStore,
    make_graph_store,
    make_vector_store,
)
from worker.pipeline import build_snapshot

_BAD_BODY = JSONResponse({"error": "malformed request body"}, status_code=400)
_UNAUTHORIZED = JSONResponse({"error": "invalid or missing X-Gitlab-Token"}, status_code=401)


@dataclass(frozen=True)
class WorkerService:
    """Набор зависимостей, нужных build_snapshot."""

    settings: Settings
    embedder: Embedder
    vector_store: VectorStore
    graph_store: GraphStore
    meta_store: MetaStore


def build_service(settings: Settings) -> WorkerService:
    """Собрать WorkerService из settings через make_*-фабрики.

    Все фабрики ленивые — сетевое соединение при вызове не открывается.
    """
    return WorkerService(
        settings=settings,
        embedder=make_embedder(
            settings.embed_url, embed_dim=settings.embed_dim, model=settings.embed_model
        ),
        vector_store=make_vector_store(settings.vector_url, embed_dim=settings.embed_dim),
        graph_store=make_graph_store(
            settings.graph_url, auth=(settings.graph_user, settings.graph_password)
        ),
        meta_store=make_meta_store(settings.meta_url),
    )


def create_app(service: WorkerService) -> FastAPI:
    """Вернуть FastAPI-приложение с роутами /health и /webhook."""

    _app = FastAPI()

    @_app.get("/health")
    async def health() -> JSONResponse:  # noqa: RUF029
        return JSONResponse({"status": "ok"})

    @_app.post("/webhook")
    async def webhook(request: Request) -> JSONResponse:  # noqa: RUF029
        """Проверить токен, разобрать тело push-события и запустить сборку снапшота в фоне."""
        token = request.headers.get("X-Gitlab-Token", "")
        if token != service.settings.webhook_secret:
            return _UNAUTHORIZED

        try:
            raw = await request.json()
        except Exception:
            return _BAD_BODY
        if not isinstance(raw, dict):
            return _BAD_BODY

        try:
            ref: str = raw["ref"]
            commit_sha: str = raw["checkout_sha"]
            project = raw.get("project")
            repository = raw.get("repository")
            if not isinstance(project, dict) or not isinstance(repository, dict):
                return _BAD_BODY
            repo_path = (
                project.get("http_url_to_repo")
                or project.get("git_http_url")
                or repository.get("git_http_url")
            )
            if not isinstance(repo_path, str):
                return _BAD_BODY
            branch = ref.removeprefix("refs/heads/")
        except (KeyError, TypeError, AttributeError):
            return _BAD_BODY

        task = BackgroundTask(
            build_snapshot,
            repo_path,
            branch,
            commit_sha,
            settings=service.settings,
            embedder=service.embedder,
            vector_store=service.vector_store,
            graph_store=service.graph_store,
            meta_store=service.meta_store,
        )
        return JSONResponse(
            {"status": "accepted", "commit_sha": commit_sha},
            status_code=202,
            background=task,
        )

    return _app


app: FastAPI = create_app(build_service(load_settings()))

"""FastAPI-обвязка (ASGI-адаптер) поверх чистого роутера handle_request.

create_app — фабрика, которая навешивает на FastAPI-приложение единый
хендлер, а тот целиком делегирует работу в handle_request. Вся маршрутизация,
валидация и формирование ошибок живут в роутере — этот адаптер только достаёт
JSON из тела запроса и переносит статус и тело ответа роутера на транспорт.

build_service собирает ApiService из объекта Settings через фабричные make_*
функции пакета common. Все фабрики ленивые — ни одно соединение не открывается
в момент конструирования.

Модульный объект app собирается из load_settings() прямо при импорте. Фабрики строят
клиенты лениво — ни одно соединение не открывается при импорте, даже когда дефолтные
URL указывают на реальные Qdrant/Neo4j/Ollama, так что импорт api.asgi безопасен без
поднятых сервисов.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.app import handle_request
from api.service import ApiService
from common.config import Settings, load_settings
from common.db import make_meta_store
from common.embed import make_embedder
from common.generate import make_generator
from common.hyde import make_expander
from common.rerank import make_reranker
from common.stores import make_graph_store, make_vector_store

_BAD_BODY = JSONResponse({"error": "request body must be a JSON object"}, status_code=400)


def build_service(settings: Settings) -> ApiService:
    """Собрать ApiService из settings через фабричные функции.

    Все фабрики ленивые — ни одно сетевое соединение не открывается в момент вызова.
    """
    embedder = make_embedder(
        settings.embed_url, embed_dim=settings.embed_dim, model=settings.embed_model
    )
    vector_store = make_vector_store(settings.vector_url, embed_dim=settings.embed_dim)
    graph_store = make_graph_store(
        settings.graph_url, auth=(settings.graph_user, settings.graph_password)
    )
    meta_store = make_meta_store(settings.meta_url)
    reranker = make_reranker(settings.rerank_url, model=settings.rerank_model)
    generator = make_generator(settings.gen_url, model=settings.gen_model)
    expander = make_expander(settings.hyde_url, model=settings.hyde_model)
    return ApiService(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        graph_store=graph_store,
        meta_store=meta_store,
        reranker=reranker,
        generator=generator,
        expander=expander,
    )


def create_app(service: ApiService) -> FastAPI:
    """Собрать FastAPI-приложение, где все роуты делегируют в handle_request."""

    _app = FastAPI()

    @_app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        response_model=None,
    )
    async def handler(request: Request, full_path: str) -> JSONResponse:  # noqa: RUF029
        path = "/" + full_path
        method = request.method

        if method != "GET":
            try:
                raw = await request.json()
            except (json.JSONDecodeError, ValueError):
                return _BAD_BODY
            if not isinstance(raw, dict):
                return _BAD_BODY
            body: dict[str, object] = raw
        else:
            body = {}
        resp = handle_request(service, method, path, body)
        return JSONResponse(resp.body, status_code=resp.status)

    return _app


app: FastAPI = create_app(build_service(load_settings()))

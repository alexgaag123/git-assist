"""Чистый роутер запросов для API-поверхности, без привязки к транспорту.

Принимает уже разобранный запрос (method, path, body), маппит его на операции ApiService
и возвращает Response (статус + готовое к сериализации тело) — никогда не бросает исключение
наружу из-за клиентской ошибки, всегда отдаёт правильный код (200/400/404/405).

Роутер сознательно чистый: получает уже раскодированный body и возвращает обычный
dataclass. Сам JSON-over-HTTP ввод-вывод (ASGI/FastAPI-адаптер) — отдельный слой снаружи,
это нужно, чтобы поверхность оставалась тестируемой независимо от транспорта.
"""

from __future__ import annotations

from dataclasses import dataclass

from api.main import health
from api.retrieve import DEFAULT_PREFETCH_K, DEFAULT_TOP_K
from api.service import ApiService

_ROUTES: dict[str, str] = {
    "/health": "GET",
    "/search": "POST",
    "/impact": "POST",
    "/answer": "POST",
}


@dataclass(frozen=True)
class Response:
    """Ответ роутера: HTTP-статус и тело, готовое к сериализации в JSON."""

    status: int
    body: dict[str, object]


def handle_request(
    service: ApiService,
    method: str,
    path: str,
    body: dict[str, object],
) -> Response:
    """Направляет разобранный запрос в сервис и формирует ответ.

    GET /health → liveness; POST /search → ApiService.search; POST /impact →
    ApiService.impact; POST /answer → ApiService.answer. Неизвестный path — 404,
    известный path не с тем методом — 405. Отсутствующие/неверно типизированные поля
    запроса, а также ValueError из сервиса (например проверка размерности эмбеддера
    в retrieve_pipeline) — всё это репортится как 400, наружу исключение никогда не
    улетает.
    """
    allowed = _ROUTES.get(path)
    if allowed is None:
        return Response(status=404, body={"error": f"unknown path: {path}"})
    if method != allowed:
        return Response(status=405, body={"error": f"method not allowed: {method} {path}"})

    try:
        if path == "/health":
            return Response(status=200, body=_health_body())
        if path == "/search":
            return Response(status=200, body=_handle_search(service, body))
        if path == "/impact":
            return Response(status=200, body=_handle_impact(service, body))
        return Response(status=200, body=_handle_answer(service, body))
    except ValueError as exc:
        return Response(status=400, body={"error": str(exc)})


def _health_body() -> dict[str, object]:
    return {**health()}


def _handle_search(service: ApiService, body: dict[str, object]) -> dict[str, object]:
    query = body.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError("'query' must be a non-empty string")
    repo = body.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError("'repo' must be a non-empty string")
    limit = _optional_int(body, "limit")
    max_context_chars = _optional_int(body, "max_context_chars")
    response = service.search(
        query,
        repo=repo,
        limit=DEFAULT_TOP_K if limit is None else limit,
        max_context_chars=max_context_chars,
    )
    return response.to_dict()


def _handle_impact(service: ApiService, body: dict[str, object]) -> dict[str, object]:
    changed_paths = body.get("changed_paths")
    if not isinstance(changed_paths, list) or not all(
        isinstance(item, str) for item in changed_paths
    ):
        raise ValueError("'changed_paths' must be a list of strings")
    repo = body.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError("'repo' must be a non-empty string")
    max_depth = _optional_int(body, "max_depth")
    response = service.impact(changed_paths, repo=repo, max_depth=max_depth)
    return response.to_dict()


def _handle_answer(service: ApiService, body: dict[str, object]) -> dict[str, object]:
    query = body.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError("'query' must be a non-empty string")
    repo = body.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ValueError("'repo' must be a non-empty string")
    limit = _optional_int(body, "limit")
    prefetch_limit = _optional_int(body, "prefetch_limit")
    response = service.answer(
        query,
        repo=repo,
        limit=DEFAULT_TOP_K if limit is None else limit,
        prefetch_limit=DEFAULT_PREFETCH_K if prefetch_limit is None else prefetch_limit,
    )
    return response.to_dict()


def _optional_int(body: dict[str, object], key: str) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer")
    return value

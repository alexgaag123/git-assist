"""Доменный слой расширения запроса по HyDE (Hypothetical Document Embeddings).

HydeExpander превращает сырой запрос в более богатый текст для эмбеддинга — улучшает
полноту поиска за счёт того, что эмбеддится гипотетический ответ, а не голый запрос.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from common.transport import Transport, _http_post_json

HYDE_PROMPT = (
    "You are a code search assistant. Given a user query, write a SHORT hypothetical\n"
    "code snippet or documentation paragraph that a real repository file might contain\n"
    "if it answered the query. Output only the snippet — no explanation, no markdown\n"
    "fences. Keep it under 200 tokens."
)


@runtime_checkable
class HydeExpander(Protocol):
    """Превращает запрос в текст для эмбеддинга вместо сырого запроса.

    Реализация обязана возвращать непустую строку для любого непустого запроса;
    пустой запрос даёт "".
    """

    def expand(self, query: str) -> str:
        """Возвращает текст, который нужно эмбеддить вместо query."""
        ...


class OllamaHydeExpander:
    """HyDE-расширитель поверх эндпоинта Ollama /api/chat (например, Qwen3).

    Отправляет обычный (без format) chat-запрос и возвращает message.content — плоский
    конверт {"message": {"content": "..."}}, без обёртки "choices", как у OpenAI. При
    создании соединение не открывается — транспорт вызывается только внутри expand.
    Пустой запрос сразу возвращает пустую строку, не вызывая транспорт.
    """

    def __init__(self, base_url: str, model: str, *, transport: Transport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport

    def expand(self, query: str) -> str:
        """Возвращает гипотетический фрагмент кода/документации для query через Ollama."""
        if not query:
            return ""
        transport = self._transport or _http_post_json
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user", "content": query},
            ],
            "stream": False,
        }
        response = transport(f"{self._base_url}/api/chat", payload)
        if not isinstance(response, Mapping):
            raise ValueError("Ollama: response must be a JSON object")
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Ollama: response missing 'message'")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama: message missing string 'content'")
        return content


def make_expander(url: str, *, model: str = "") -> HydeExpander:
    """Выбирает реализацию HydeExpander по URL.

    - ollama://<host>:<port> → OllamaHydeExpander с именем модели model (обязателен для
      этой схемы).
    - что угодно ещё → ValueError с указанием неподдерживаемой схемы.

    Вызывающий код выбирает реализацию через settings.hyde_url и settings.hyde_model,
    не импортируя конкретный класс расширителя напрямую.
    """
    if url.startswith("ollama://"):
        if not model:
            raise ValueError("ollama:// expander requires a model name (settings.hyde_model)")
        return OllamaHydeExpander("http://" + url[len("ollama://") :], model)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported HyDE expander URL scheme: {scheme!r} (url={url!r})")

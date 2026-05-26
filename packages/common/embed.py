"""Абстракция эмбеддера текста и фабрика, выбирающая реализацию по URL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from common.transport import Transport, _http_post_json, truncate_and_normalize


@runtime_checkable
class Embedder(Protocol):
    """Отображает текст в плотные векторы фиксированной ширины, детерминированно и батчем."""

    @property
    def dim(self) -> int:
        """Размерность каждого вектора (по ней настраивается векторный стор)."""
        ...

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Эмбеддит батч, возвращая по одному вектору длины dim на каждый вход, по порядку."""
        ...


def _parse_ollama_embed_response(response: object, *, expected: int) -> list[list[float]]:
    """Проверяет ответ Ollama /api/embed и превращает его в список векторов float.

    Ollama возвращает {"embeddings": [[float, ...], ...]} в порядке входных данных.
    """
    if not isinstance(response, Mapping):
        raise ValueError(f"Ollama /api/embed: expected an object, got {type(response).__name__}")
    vectors_raw = response.get("embeddings")
    if not isinstance(vectors_raw, list):
        raise ValueError("Ollama /api/embed: response missing 'embeddings' array")
    vectors: list[list[float]] = []
    for item in vectors_raw:
        if not isinstance(item, list):
            raise ValueError("Ollama /api/embed: each embedding must be a list of floats")
        vectors.append([float(component) for component in item])
    if len(vectors) != expected:
        raise ValueError(f"Ollama /api/embed: expected {expected} vectors, got {len(vectors)}")
    return vectors


class OllamaEmbedder:
    """Реализация Embedder поверх эндпоинта Ollama /api/embed (Qwen/Qwen3-Embedding-4B).

    Создаётся с базовым URL сервера, целевой размерностью (== settings.embed_dim) и именем
    модели. Каждый полученный вектор обрезается до dim (Matryoshka) и заново
    L2-нормализуется (common.transport.truncate_and_normalize). При создании соединение
    не открывается — транспорт вызывается только внутри embed.
    """

    def __init__(
        self, base_url: str, dim: int, model: str, *, transport: Transport | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dim = dim
        self._model = model
        self._transport = transport

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        transport = self._transport or _http_post_json
        response = transport(
            f"{self._base_url}/api/embed", {"model": self._model, "input": list(texts)}
        )
        vectors = _parse_ollama_embed_response(response, expected=len(texts))
        return [
            truncate_and_normalize(vector, self._dim, source="Ollama /api/embed")
            for vector in vectors
        ]


def make_embedder(url: str, *, embed_dim: int, model: str = "") -> Embedder:
    """Выбирает реализацию Embedder по URL.

    - ollama://<host>:<port> → OllamaEmbedder с именем модели model (обязателен для этой
      схемы) поверх лениво создаваемого транспорта.
    - что угодно ещё → ValueError с указанием неподдерживаемой схемы.

    Вызывающий код выбирает размер через settings.embed_dim и модель через
    settings.embed_model, не импортируя конкретный класс эмбеддера напрямую.
    """
    if url.startswith("ollama://"):
        if not model:
            raise ValueError("ollama:// embedder requires a model name (settings.embed_model)")
        return OllamaEmbedder("http://" + url[len("ollama://") :], embed_dim, model)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported embedder URL scheme: {scheme!r} (url={url!r})")

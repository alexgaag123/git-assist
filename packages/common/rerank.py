"""Реранкинг: переупорядочивание кандидатов поиска по релевантности запросу.

Реранкинг — это второй проход скоринга. Первый проход (векторный поиск)
возвращает top-k чанков по сходству эмбеддингов; реранкер затем пересчитывает
скор этих немногих кандидатов относительно запроса более точной,
query-aware моделью и переупорядочивает их. Модуль фиксирует интерфейс —
протокол Reranker и запись RerankResult — плюс единственную реализацию,
OllamaReranker (Qwen/Qwen3-Reranker-0.6B).

Qwen3-Reranker — decoder-only модель: пара запрос/документ скорится через
logits токенов "yes"/"no" после обычного causal-LM прохода (sentence-transformers
5.4 называет это модулем LogitScore). OllamaReranker обслуживает её через Ollama:
бьёт в /api/generate с logprobs и сам считает score = softmax(yes, no).

Реранкер используется на стороне поиска (см. retrieve_pipeline в apps/api).
RerankResult.index указывает обратно на позицию во входном списке кандидатов, так
что вызывающий код мапит скоры на свои собственные результаты без необходимости
эхом возвращать текст.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from common.transport import Transport, _http_post_json


def _sort_and_truncate(results: list[RerankResult], top_k: int | None) -> list[RerankResult]:
    """Отсортировать results по убыванию score, ничьи — по возрастанию index; обрезать по top_k."""
    results.sort(key=lambda result: (-result.score, result.index))
    if top_k is not None:
        results = results[:top_k]
    return results


@dataclass(frozen=True)
class RerankResult:
    """Один переранжированный кандидат: его index во входном списке и score релевантности."""

    index: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Детерминированно пересчитывает скор кандидатов поиска относительно запроса."""

    def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankResult]:
        """Оценить каждого кандидата относительно query.

        Возвращает по одному RerankResult на сохранённого кандидата,
        отсортированные по убыванию score с ничьими по возрастанию index,
        обрезанные до top_k, если он задан. Пустой documents даёт [].
        """
        ...


def make_reranker(url: str, *, model: str = "") -> Reranker:
    """Выбрать реализацию Reranker по URL реранкера.

    - ollama://<host>:<port> → OllamaReranker с именем модели model (обязателен
      для этой схемы) поверх лениво создаваемого транспорта.
    - что угодно ещё → ValueError с именем неподдерживаемой схемы.

    Вызывающий код выбирает реализацию через settings.rerank_url и
    settings.rerank_model, не импортируя конкретный класс реранкера напрямую.
    """
    if url.startswith("ollama://"):
        if not model:
            raise ValueError("ollama:// reranker requires a model name (settings.rerank_model)")
        return OllamaReranker("http://" + url[len("ollama://") :], model)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported reranker URL scheme: {scheme!r} (url={url!r})")


_OLLAMA_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
_OLLAMA_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def _ollama_prompt(query: str, document: str) -> str:
    """Собрать ChatML-промпт Qwen3-Reranker с пустым think-блоком (reasoning выключен)."""
    user = f"<Instruct>: {_OLLAMA_INSTRUCT}\n<Query>: {query}\n<Document>: {document}"
    return (
        f"<|im_start|>system\n{_OLLAMA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _parse_yes_no_score(response: object) -> float:
    """Извлечь score = softmax(yes, no) из top_logprobs первого сгенерированного токена.

    Ollama /api/generate с logprobs=true возвращает logprobs[0].top_logprobs — список
    {"token", "logprob"} по кандидатам следующего токена. Токен, не попавший в топ,
    считается настолько маловероятным, что берётся дефолтный logprob -20.0.
    """
    if not isinstance(response, Mapping):
        raise ValueError(f"Ollama /api/generate: expected an object, got {type(response).__name__}")
    logprobs = response.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs:
        raise ValueError("Ollama /api/generate: missing 'logprobs' in response")
    top = logprobs[0].get("top_logprobs") if isinstance(logprobs[0], Mapping) else None
    if not isinstance(top, list):
        raise ValueError("Ollama /api/generate: missing 'top_logprobs' in response")
    by_token: dict[str, float] = {}
    for item in top:
        if not isinstance(item, Mapping):
            continue
        token = str(item.get("token", "")).strip().lower()
        if token in ("yes", "no") and token not in by_token:
            by_token[token] = float(item["logprob"])
    yes_p = math.exp(by_token.get("yes", -20.0))
    no_p = math.exp(by_token.get("no", -20.0))
    return yes_p / (yes_p + no_p)


class OllamaReranker:
    """Reranker поверх Ollama, скорящий кандидатов через Qwen3-Reranker-0.6B.

    Qwen3-Reranker — decoder-only модель: релевантность пары запрос/документ
    выражается через logits токенов "yes"/"no" после обычного causal-LM прохода
    (см. докстринг модуля). У Ollama нет отдельного /rerank-эндпоинта под эту
    схему, поэтому каждый кандидат — отдельный вызов /api/generate с num_predict=1
    и logprobs=true; score — softmax(P(yes), P(no)).

    Конструктор не открывает соединение — транспорт вызывается только внутри
    rerank; пустой documents возвращает [] без вызова.
    """

    def __init__(self, base_url: str, model: str, *, transport: Transport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport

    def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []
        transport = self._transport or _http_post_json
        results = [
            RerankResult(index=index, score=self._score(transport, query, document))
            for index, document in enumerate(documents)
        ]
        return _sort_and_truncate(results, top_k)

    def _score(self, transport: Transport, query: str, document: str) -> float:
        payload = {
            "model": self._model,
            "prompt": _ollama_prompt(query, document),
            "raw": True,
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True,
            "top_logprobs": 20,
        }
        response = transport(f"{self._base_url}/api/generate", payload)
        return _parse_yes_no_score(response)

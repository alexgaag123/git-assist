"""Генерация: сборка обоснованных ответов на естественном языке из найденных чанков контекста.

Генерация — последний этап пути запроса (найти → реранжировать → сгенерировать). Generator
получает исходный запрос и ранжированный список записей ContextChunk, а возвращает Answer
из объектов Statement, каждый из которых привязан к чанкам, которые его подтверждают. Этот
модуль фиксирует только доменный слой — сами записи и Protocol Generator, — так что
остальная система работает с формой интерфейса, а не с конкретной моделью под капотом.
Конкретный генератор (OllamaGenerator) и проверка обоснованности реализованы ниже.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from common.transport import Transport, _http_post_json

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")
"""Тот же регэксп-разбор токенов, что в rerank.py; продублирован, чтобы этот модуль
оставался независимым от rerank.py."""


class AnswerStatus(StrEnum):
    """Итог попытки генерации."""

    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass(frozen=True)
class ContextChunk:
    """Один обосновывающий чанк, переданный генератору.

    Несёт стабильный chunk_id и text чанка, чтобы генератор мог формировать Statement
    с цитированием конкретных chunk_id, не импортируя при этом записи поиска из
    apps/api.
    """

    chunk_id: str
    text: str


@dataclass(frozen=True)
class Statement:
    """Одно утверждение в ответе, обоснованное подмножеством чанков контекста.

    chunk_ids перечисляет каждый ContextChunk, подтверждающий это утверждение. Пустой
    кортеж допустим (например, для мета-утверждений), но генераторам стоит стремиться
    процитировать хотя бы один чанк на утверждение.
    """

    text: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """Ответ генератора: статус и ноль или более обоснованных утверждений.

    status равен ANSWERED, если из переданного контекста удалось получить хотя бы одно
    утверждение, и INSUFFICIENT_CONTEXT, если контекст был пуст или слишком слаб, чтобы
    на него можно было опереться.
    """

    status: AnswerStatus
    statements: tuple[Statement, ...]

    @property
    def answer_text(self) -> str:
        """Тексты утверждений, склеенные через пустую строку; при их отсутствии — ""."""
        return "\n\n".join(s.text for s in self.statements)


def enforce_grounding(
    statements: Sequence[Statement],
    allowed_chunk_ids: Collection[str],
) -> Answer:
    """Вернуть Answer, в котором цитируются только чанки из allowed_chunk_ids.

    Для каждого Statement из statements:

    1. Фильтруем его chunk_ids — оставляем только те id, что входят в allowed_chunk_ids,
       сохраняя исходный порядок и стабильно убирая дубликаты (первое вхождение остаётся).
    2. Если после фильтрации кортеж пуст — утверждение отбрасывается целиком.
    3. Из выживших утверждений собираем новые Statement с тем же текстом, но
       отфильтрованными chunk_ids.

    Итог: если выжило хотя бы одно утверждение — Answer(ANSWERED, ...); если не выжило
    ни одного (в том числе когда statements изначально пуст) — Answer(INSUFFICIENT_CONTEXT, ()).

    allowed_chunk_ids принимает любой Collection (например, set — для O(1) проверки
    вхождения), а не только list.
    """
    survivors: list[Statement] = []
    allowed: Collection[str] = allowed_chunk_ids
    for stmt in statements:
        seen: set[str] = set()
        filtered: list[str] = []
        for cid in stmt.chunk_ids:
            if cid in allowed and cid not in seen:
                filtered.append(cid)
                seen.add(cid)
        if filtered:
            survivors.append(Statement(text=stmt.text, chunk_ids=tuple(filtered)))
    if survivors:
        return Answer(AnswerStatus.ANSWERED, tuple(survivors))
    return Answer(AnswerStatus.INSUFFICIENT_CONTEXT, ())


@runtime_checkable
class Generator(Protocol):
    """Формирует обоснованный Answer из запроса и найденных чанков контекста.

    Реализации обязаны быть детерминированными: один и тот же query и context всегда
    дают один и тот же Answer. Если context пуст, генератор должен немедленно вернуть
    Answer(INSUFFICIENT_CONTEXT, ()), не пытаясь ничего сгенерировать — вызывающий код
    полагается на это короткое замыкание, чтобы не тратить впустую инференс.
    """

    def generate(self, query: str, context: Sequence[ContextChunk]) -> Answer:
        """Сгенерировать Answer, обоснованный context.

        query — исходный запрос на естественном языке. context — ранжированные записи
        ContextChunk (первая — самая релевантная). Пустой context → сразу
        Answer(INSUFFICIENT_CONTEXT, ()).
        """
        ...


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _TOKEN_SPLIT.split(text.lower()) if tok}


@runtime_checkable
class Verifier(Protocol):
    """Подтверждает ли совокупность доказательств утверждение?"""

    def entails(self, statement: str, evidence: Sequence[str]) -> bool: ...


class LexicalEntailmentVerifier:
    """Детерминированный верификатор по лексическому пересечению.

    Утверждение считается подтверждённым, если доля его буквенно-цифровых токенов
    в нижнем регистре, покрытых объединением токенов доказательств, не меньше threshold.
    Пустое утверждение → True (подтверждать нечего). Непустое утверждение без токенов
    доказательств → False.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def entails(self, statement: str, evidence: Sequence[str]) -> bool:
        stmt_tokens = _tokenize(statement)
        if not stmt_tokens:
            return True
        ev_tokens: set[str] = set()
        for ev in evidence:
            ev_tokens |= _tokenize(ev)
        if not ev_tokens:
            return False
        covered = len(stmt_tokens & ev_tokens) / len(stmt_tokens)
        return covered >= self.threshold


def verify_statements(
    statements: Sequence[Statement],
    context: Sequence[ContextChunk],
    verifier: Verifier,
) -> tuple[Statement, ...]:
    """Оставить только те утверждения, чьи процитированные доказательства их подтверждают.

    Для каждого утверждения его chunk_ids резолвятся против context, после чего
    вызывается verifier.entails. Утверждения, чьи процитированные чанки отсутствуют
    в context, получают пустой набор доказательств. Порядок входных данных сохраняется.
    """
    chunk_map: dict[str, str] = {c.chunk_id: c.text for c in context}
    survivors: list[Statement] = []
    for stmt in statements:
        evidence = [chunk_map[cid] for cid in stmt.chunk_ids if cid in chunk_map]
        if verifier.entails(stmt.text, evidence):
            survivors.append(stmt)
    return tuple(survivors)


ANSWER_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["status", "statements"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_context"]},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "chunk_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}
"""JSON Schema, встроенная в системный промпт, чтобы модель выдавала структурированный ответ."""


def _parse_answer_json(content: str, *, source: str) -> list[Statement]:
    """Провалидировать JSON-строку content по ANSWER_JSON_SCHEMA и вернуть сырые Statement.

    Отсутствующий "statements" (например, когда модель вернула только
    status: insufficient_context) трактуется как пустой список; если поле присутствует,
    но не список — ошибка. source называет вызывающий бэкенд в сообщении об ошибке.
    """
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: content is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{source}: content JSON must be an object")
    raw_stmts = parsed.get("statements", [])
    if not isinstance(raw_stmts, list):
        raise ValueError(f"{source}: 'statements' must be an array")
    statements: list[Statement] = []
    for i, item in enumerate(raw_stmts):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: statements[{i}] must be an object")
        text = item.get("text")
        chunk_ids_raw = item.get("chunk_ids")
        if not isinstance(text, str):
            raise ValueError(f"{source}: statements[{i}] missing string 'text'")
        if not isinstance(chunk_ids_raw, list):
            raise ValueError(f"{source}: statements[{i}] missing 'chunk_ids' array")
        for j, cid in enumerate(chunk_ids_raw):
            if not isinstance(cid, str):
                raise ValueError(f"{source}: statements[{i}].chunk_ids[{j}] must be a string")
        statements.append(Statement(text=text, chunk_ids=tuple(str(c) for c in chunk_ids_raw)))
    return statements


def _parse_ollama_chat_response(response: object) -> list[Statement]:
    """Провалидировать ответ Ollama /api/chat и вернуть сырые Statement.

    Ожидаемый конверт: {"message": {"content": "<строка JSON>"}} — плоский, без обёртки
    "choices", как у OpenAI-совместимых API. Сам content валидирует _parse_answer_json.
    """
    if not isinstance(response, Mapping):
        raise ValueError("Ollama: response must be a JSON object")
    message = response.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Ollama: response missing 'message'")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama: message missing string 'content'")
    return _parse_answer_json(content, source="Ollama")


class OllamaGenerator:
    """Generator поверх эндпоинта Ollama /api/chat (например, Qwen3).

    Ollama принимает JSON Schema прямо в поле format и валидирует структуру на своей
    стороне. Конверт ответа плоский: {"message": {"content": "<строка JSON>"}}, без
    "choices" — парсит _parse_ollama_chat_response.

    Конструктор не открывает соединения; транспорт вызывается только внутри generate.
    Пустой context обрывает выполнение раньше, не дёргая транспорт.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport
        self._verifier = verifier

    def generate(self, query: str, context: Sequence[ContextChunk]) -> Answer:
        """Вернуть Answer, обоснованный context, через эндпоинт Ollama."""
        if not context:
            return Answer(AnswerStatus.INSUFFICIENT_CONTEXT, ())

        transport = self._transport or _http_post_json
        verifier = self._verifier or LexicalEntailmentVerifier()

        labelled = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering assistant. "
                    "Answer using ONLY the provided context chunks. "
                    "Cite the chunk_ids that support each statement."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{labelled}\n\nQuestion: {query}",
            },
        ]
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "format": ANSWER_JSON_SCHEMA,
            "stream": False,
        }
        response = transport(f"{self._base_url}/api/chat", payload)
        raw_statements = _parse_ollama_chat_response(response)

        allowed_ids = {c.chunk_id for c in context}
        grounded = enforce_grounding(raw_statements, allowed_chunk_ids=allowed_ids)
        if grounded.status is not AnswerStatus.ANSWERED:
            return grounded

        verified = verify_statements(grounded.statements, context, verifier)
        if verified:
            return Answer(AnswerStatus.ANSWERED, verified)
        return Answer(AnswerStatus.INSUFFICIENT_CONTEXT, ())


def make_generator(url: str, *, model: str = "") -> Generator:
    """Выбрать реализацию Generator по строке подключения url.

    - ``ollama://<host>:<port>`` → OllamaGenerator с именем модели model (обязателен для
      этой схемы).
    - что угодно другое → ValueError с названием неподдерживаемой схемы.

    Вызывающий код выбирает бэкенд через settings.gen_url и модель через
    settings.gen_model, не импортируя конкретный класс генератора напрямую.
    """
    if url.startswith("ollama://"):
        if not model:
            raise ValueError("ollama:// generator requires a model name (settings.gen_model)")
        return OllamaGenerator("http://" + url[len("ollama://") :], model)
    scheme = url.split("://", 1)[0]
    raise ValueError(f"unsupported generator URL scheme: {scheme!r} (url={url!r})")

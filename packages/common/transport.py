"""Общий HTTP-транспорт и векторная утилита, разделяемые адаптерами Ollama.

Каждый адаптер (embed.py, rerank.py, generate.py, hyde.py) говорит с Ollama по HTTP
на JSON. Сам HTTP-вызов инжектируется как callable-транспорт (по умолчанию —
urllib), который вызывается лениво при первом запросе, так что конструирование
адаптера не открывает соединение.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Mapping

Transport = Callable[[str, Mapping[str, object]], object]

_HTTP_TIMEOUT = 120.0
"""Своп моделей в Ollama при нехватке VRAM (см. rerank.py/embed.py/generate.py) может
занимать десятки секунд на холодной загрузке."""


def _http_post_json(url: str, payload: Mapping[str, object]) -> object:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        body: object = json.loads(response.read().decode("utf-8"))
    return body


def truncate_and_normalize(vector: list[float], dim: int, *, source: str) -> tuple[float, ...]:
    """Обрезать Matryoshka-вектор до dim и заново L2-нормализовать.

    Qwen/Qwen3-Embedding-4B обучен по схеме Matryoshka (нативные векторы до ~2560
    измерений) — обрезка до settings.embed_dim с повторной нормализацией обязательна
    для корректного косинуса. source называет вызывающий эндпоинт в сообщении об ошибке.
    """
    if len(vector) < dim:
        raise ValueError(f"{source}: vector width {len(vector)} is smaller than embed_dim {dim}")
    truncated = vector[:dim]
    norm = math.sqrt(sum(component * component for component in truncated))
    if norm == 0.0:
        return tuple(truncated)
    return tuple(component / norm for component in truncated)

"""Liveness-проверка API: минимальный payload без побочных эффектов."""

from __future__ import annotations


def health() -> dict[str, str]:
    """Пейлоад liveness-проверки: без GPU, без I/O, без внешних зависимостей."""

    return {"status": "ok"}

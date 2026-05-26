"""Точка входа воркера — поднимает ASGI-приложение вебхука под uvicorn.

Само FastAPI-приложение собирается в worker.asgi при импорте, на дефолтных настройках;
фабрики строят клиенты лениво, так что импортировать этот модуль безопасно даже без
живых сервисов. Здесь только раннер uvicorn: команда `python -m worker.main` в контейнере
поднимает вебхук на порту 8001.
"""

from __future__ import annotations

from worker.asgi import app

WORKER_PORT = 8001


def main() -> None:
    """Поднять приложение вебхука на всех интерфейсах, порт 8001."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)


if __name__ == "__main__":
    main()

"""Типизированные настройки системы, читаемые из окружения.

Каждый модуль берёт конфигурацию из одного неизменяемого объекта Settings.
Дефолты рассчитаны на запуск внутри docker-compose (см. docker-compose.yml,
.env.example): сторы и модельные сервисы указывают на контейнеры по их
именам в сети compose. Локальный запуск вне Docker или без Postgres-креды
требует переопределить соответствующие GIT_ASSIST_*_URL через окружение.

Функция load_settings принимает окружение явным словарём, а не читает
os.environ напрямую — так логика загрузки остаётся детерминированной и
тестируемой без побочных эффектов.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_CLONE_TOKEN_FILE = Path("/run/secrets/gitlab_clone_token")

ENV_PREFIX = "GIT_ASSIST_"


@dataclass(frozen=True)
class Settings:
    """Неизменяемая конфигурация индексации и запросов к репозиторию.

    Дефолты нацелены на docker-compose: все поля *_url указывают на реальные
    сервисы по их именам в сети compose (qdrant, neo4j, postgres, ollama), с
    теми же кредами, что в .env.example. Реальное окружение переопределяет их
    через GIT_ASSIST_*_URL.
    """

    repo_path: str = "."
    """Рабочее дерево, которое нужно проиндексировать."""

    vector_url: str = "qdrant://qdrant:6333"
    """URL векторного стора — адрес Qdrant (QdrantVectorStore)."""

    graph_url: str = "neo4j://neo4j:7687"
    """URL графового стора — адрес Neo4j (Neo4jGraphStore)."""

    graph_user: str = "neo4j"
    """Логин Neo4j. GIT_ASSIST_GRAPH_USER."""

    graph_password: str = "gitassist"
    """Пароль Neo4j — совпадает с NEO4J_AUTH из docker-compose. GIT_ASSIST_GRAPH_PASSWORD."""

    meta_url: str = "postgresql://gitassist:gitassist@postgres:5432/gitassist"
    """URL стора метаданных — строка подключения Postgres (PostgresMetaStore).
    Дефолт совпадает с кредами docker-compose (.env.example); в любом другом
    окружении переопределяется через GIT_ASSIST_META_URL."""

    embed_url: str = "ollama://ollama:11434"
    """URL эмбеддера — адрес Ollama, обслуживающего Qwen/Qwen3-Embedding-4B (OllamaEmbedder)."""

    embed_model: str = "qwen3-embedding:4b"
    """Имя модели, передаваемое в Ollama при эмбеддинге. GIT_ASSIST_EMBED_MODEL."""

    rerank_url: str = "ollama://ollama:11434"
    """URL реранкера — адрес Ollama, обслуживающего Qwen/Qwen3-Reranker-0.6B
    (OllamaReranker, см. rerank.py)."""

    rerank_model: str = "hf.co/QuantFactory/Qwen3-Reranker-0.6B-GGUF:Q4_K_M"
    """Имя модели, передаваемое в Ollama при реранкинге. GIT_ASSIST_RERANK_MODEL."""

    gen_url: str = "ollama://ollama:11434"
    """URL генератора — адрес Ollama (OllamaGenerator)."""

    gen_model: str = "qwen3:4b"
    """Имя модели, передаваемое в Ollama при генерации ответа. GIT_ASSIST_GEN_MODEL."""

    hyde_url: str = "ollama://ollama:11434"
    """URL HyDE-расширителя запроса — адрес Ollama (OllamaHydeExpander).
    Переопределяется через GIT_ASSIST_HYDE_URL."""

    hyde_model: str = "qwen3:4b"
    """Id модели, отправляемый в HyDE chat-completion запросе.
    GIT_ASSIST_HYDE_MODEL."""

    webhook_secret: str = ""
    """Общий секрет, сверяемый с заголовком X-Gitlab-Token на POST /webhook.
    GIT_ASSIST_WEBHOOK_SECRET; пустой дефолт, чтобы гейту не требовался секрет."""

    gitlab_clone_token: str = ""
    """Deploy-токен для клонирования приватных GitLab-репозиториев по HTTP
    (http://oauth2:<token>@gitlab/...); пустой дефолт означает анонимный/
    внутренний доступ без авторизации. GIT_ASSIST_GITLAB_CLONE_TOKEN."""

    embed_dim: int = 1024
    """Размерность эмбеддингов — единственный источник истины для размеров
    векторного стора.

    Дефолт — усечённая по Matryoshka размерность активной модели эмбеддингов
    (Qwen/Qwen3-Embedding-4B, нативно до 2560 измерений). Любой векторный стор
    создаётся из этого значения, а не из захардкоженного числа."""

    chunk_size: int = 1000
    """Целевой размер чанка в символах."""

    chunk_overlap: int = 100
    """Перекрытие чанков в символах; должно соблюдать 0 <= chunk_overlap < chunk_size."""


def _parse_int(env: Mapping[str, str], field: str, default: int) -> int:
    key = ENV_PREFIX + field.upper()
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key}: expected an integer, got {raw!r}") from exc


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Собрать Settings из окружения с префиксом GIT_ASSIST_.

    Каждое поле переопределяется через GIT_ASSIST_<FIELD> (например
    GIT_ASSIST_EMBED_DIM); незаданные ключи берут дефолт из датакласса.
    Если env не передан, используется os.environ процесса; тесты всегда
    передают явный словарь, чтобы логика оставалась детерминированной.

    Возвращает ValueError на нечисловые int-поля или если проверки не проходят:
    embed_dim > 0, chunk_size > 0 и 0 <= chunk_overlap < chunk_size.
    """
    if env is None:
        env = os.environ

    defaults = Settings()
    settings = Settings(
        repo_path=env.get(ENV_PREFIX + "REPO_PATH", defaults.repo_path),
        vector_url=env.get(ENV_PREFIX + "VECTOR_URL", defaults.vector_url),
        graph_url=env.get(ENV_PREFIX + "GRAPH_URL", defaults.graph_url),
        graph_user=env.get(ENV_PREFIX + "GRAPH_USER", defaults.graph_user),
        graph_password=env.get(ENV_PREFIX + "GRAPH_PASSWORD", defaults.graph_password),
        meta_url=env.get(ENV_PREFIX + "META_URL", defaults.meta_url),
        embed_url=env.get(ENV_PREFIX + "EMBED_URL", defaults.embed_url),
        embed_model=env.get(ENV_PREFIX + "EMBED_MODEL", defaults.embed_model),
        rerank_url=env.get(ENV_PREFIX + "RERANK_URL", defaults.rerank_url),
        rerank_model=env.get(ENV_PREFIX + "RERANK_MODEL", defaults.rerank_model),
        gen_url=env.get(ENV_PREFIX + "GEN_URL", defaults.gen_url),
        gen_model=env.get(ENV_PREFIX + "GEN_MODEL", defaults.gen_model),
        hyde_url=env.get(ENV_PREFIX + "HYDE_URL", defaults.hyde_url),
        hyde_model=env.get(ENV_PREFIX + "HYDE_MODEL", defaults.hyde_model),
        webhook_secret=env.get(ENV_PREFIX + "WEBHOOK_SECRET")
        or env.get("GITLAB_WEBHOOK_SECRET", defaults.webhook_secret),
        gitlab_clone_token=env.get(ENV_PREFIX + "GITLAB_CLONE_TOKEN", defaults.gitlab_clone_token)
        or (_CLONE_TOKEN_FILE.read_text().strip() if _CLONE_TOKEN_FILE.exists() else ""),
        embed_dim=_parse_int(env, "embed_dim", defaults.embed_dim),
        chunk_size=_parse_int(env, "chunk_size", defaults.chunk_size),
        chunk_overlap=_parse_int(env, "chunk_overlap", defaults.chunk_overlap),
    )

    if settings.embed_dim <= 0:
        raise ValueError(f"embed_dim must be > 0, got {settings.embed_dim}")
    if settings.chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {settings.chunk_size}")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError(
            "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size "
            f"(got chunk_overlap={settings.chunk_overlap}, chunk_size={settings.chunk_size})"
        )

    return settings

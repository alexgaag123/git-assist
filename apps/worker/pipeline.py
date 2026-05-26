"""Полная сборка снапшота: materialize → index → activate.

Оркеструет весь воркер-пайплайн для одного коммита: клонирует репозиторий (если это
удалённый HTTP-URL), достаёт дерево во временную директорию, прогоняет через index_tree,
затем вызывает activate_snapshot, чтобы перевести готовую сборку в ACTIVE. Все временные
директории безусловно подчищаются через ExitStack, даже при исключении.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from urllib.parse import ParseResult, urlparse, urlunparse

from common.config import Settings
from common.embed import Embedder
from common.snapshot import SnapshotRef
from common.stores import GraphStore, MetaStore, VectorStore
from worker.index import IndexResult, activate_snapshot, index_tree
from worker.materialize import materialize_commit


def _inject_token(url: str, token: str) -> str:
    """Вшить token как oauth2:<token>@ в HTTP(S) URL для клонирования."""
    if not token:
        return url
    p = urlparse(url)
    host = p.hostname or ""
    if p.port:
        host = f"{host}:{p.port}"
    netloc = f"oauth2:{token}@{host}"
    return urlunparse(ParseResult(p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def build_snapshot(
    repo_path: str,
    branch: str,
    commit_sha: str,
    *,
    settings: Settings,
    embedder: Embedder,
    vector_store: VectorStore,
    graph_store: GraphStore,
    meta_store: MetaStore,
) -> IndexResult:
    """Достать commit_sha, проиндексировать дерево, активировать снапшот и вернуть результат.

    repo_path — локальный путь или HTTP(S)-URL репозитория; если это URL, репозиторий
    сначала клонируется во временный bare-клон с авторизацией через
    settings.gitlab_clone_token. branch нужен только для детерминированного snapshot_id.
    embedder.dim обязан совпадать с settings.embed_dim. Возвращает IndexResult готовой
    (ACTIVE) сборки.

    Все временные директории удаляются до возврата из функции, даже если index_tree
    бросит исключение.
    """
    ref = SnapshotRef(repo=repo_path, branch=branch, commit_sha=commit_sha)
    is_remote = urlparse(repo_path).scheme in ("http", "https")

    with contextlib.ExitStack() as stack:
        if is_remote:
            clone_dir = stack.enter_context(tempfile.TemporaryDirectory())
            clone_url = _inject_token(repo_path, settings.gitlab_clone_token)
            subprocess.run(
                ["git", "clone", "--bare", clone_url, clone_dir],
                check=True,
                capture_output=True,
            )
            local_repo = clone_dir
        else:
            local_repo = repo_path

        tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
        materialize_commit(local_repo, commit_sha, tmpdir)
        result = index_tree(
            tmpdir,
            ref,
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            graph_store=graph_store,
            meta_store=meta_store,
        )

    activate_snapshot(ref.snapshot_id, repo=ref.repo, meta_store=meta_store)
    return result

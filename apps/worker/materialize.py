"""Достаёт дерево файлов git-коммита в указанную директорию, не трогая индекс репозитория.

Через `git archive` получаем детерминированный tar-поток для любого SHA, затем
распаковываем стандартным tarfile. Никаких побочных эффектов на рабочую директорию —
можно безопасно запускать параллельно для одного и того же репозитория.
"""

from __future__ import annotations

import io
import subprocess
import tarfile


def materialize_commit(repo_path: str, commit_sha: str, dest: str) -> None:
    """Извлечь дерево коммита commit_sha в директорию dest.

    repo_path — путь к git-репозиторию (bare или обычному), commit_sha — любая ревизия,
    у которой есть дерево (коммит, тег и т.п.), dest — уже существующая директория.

    Возвращает ValueError, если git завершился с ошибкой (битый SHA, не репозиторий и т.д.) —
    в сообщении будет декодированный stderr от git.
    """
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit_sha],
        cwd=repo_path,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.decode())

    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tf:
        tf.extractall(dest, filter="data")

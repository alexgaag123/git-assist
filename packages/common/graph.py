"""Извлечение графа зависимостей: превращает файлы снапшота в рёбра импортов.

Анализ влияния коммита отвечает на вопрос «что затронется ниже по цепочке, если
поменять вот эти файлы» — для этого нужен граф зависимостей файлов/символов. Этот модуль
строит рёбра такого графа по документам, которые уже подготовил ingestion — чисто и
детерминированно, разбирая Python-исходники стандартным модулем ast и резолвя импорты в
другие файлы внутри того же снапшота. Никакого I/O и внешних зависимостей — модуль
тестируется юнит-тестами так же просто, как ingest/chunk/embed; воркер только вызывает его
на своём месте в пайплайне сборки, перед переводом снапшота в READY.

Направление ребра в системе всегда одно и то же: ребро (src, dst) значит «src зависит от
dst», то есть импорт B в файле A — это ребро A -> B. В граф попадают только внутренние
зависимости: импорт, который резолвится в файл снапшота, становится ребром; импорты
сторонних пакетов и стандартной библиотеки (для которых нет подходящего файла) просто
пропускаются.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from common.models import Document, FileKind
from common.stores import GraphEdge, GraphStore


def extract_import_modules(text: str, *, path: str = "") -> list[str]:
    """Отсортированный список уникальных абсолютных имён модулей, импортируемых в ``text``.

    Разбирает ``text`` через ast.parse и обходит все узлы import / from ... import
    (и на верхнем уровне, и вложенные):

    - ``import a.b.c`` / ``import a.b.c as x`` → ``"a.b.c"`` (составное имя, алиас отбрасывается).
    - ``from a.b import c, d`` → и ``"a.b"``, и ``"a.b.c"``, ``"a.b.d"`` (базовый модуль и
      каждый кандидат вида ``base.name``) — так резолвер может сматчить и импорт пакета
      (``from pkg import helpers`` → ``pkg/helpers.py``), и импорт конкретного модуля
      (``from pkg.mod import func`` → ``pkg/mod.py``).
    - ``from a import *`` → только ``"a"`` (имя ``*`` пропускается).

    Относительные импорты (``node.level > 0``, например ``from . import x``,
    ``from .mod import y``) резолвятся относительно ``path`` — пути файла в репозитории —
    подъёмом на ``level`` директорий-пакетов вверх от директории, где лежит ``path`` (это
    повторяет то, как сам Python резолвит относительные импорты — они привязаны к пакету, а
    не к имени модуля). Если такой подъём выходит выше корня репозитория, импорт считается
    нерезолвируемым и пропускается. Если ``path`` не передан (по умолчанию пустая строка),
    относительные импорты пропускаются целиком — вызывающему просто неоткуда взять контекст
    пакета.

    На неразбираемом исходнике (``SyntaxError`` / ``ValueError``, например из-за нулевых
    байт) возвращается пустой список — один битый файл просто не даёт рёбер и не валит сборку.
    """

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None:
                    continue
                base = node.module
            else:
                if not path:
                    continue
                package = _relative_package(path, node.level)
                if package is None:
                    continue
                base = _join_module(package, node.module) if node.module else package

            if base:
                modules.add(base)
            for alias in node.names:
                if alias.name != "*":
                    candidate = _join_module(base, alias.name)
                    if candidate:
                        modules.add(candidate)

    return sorted(modules)


def _join_module(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _relative_package(path: str, level: int) -> str | None:
    """Пакет, относительно которого резолвится импорт уровня ``level`` в файле ``path``.

    ``level=1`` (``from . import x``) — это пакет, содержащий ``path`` (его директория);
    каждый следующий уровень поднимается ещё на одну директорию выше. Возвращает ``None``,
    если такой подъём достигает или выходит за пределы корня репозитория — так же, как сам
    Python падает с ошибкой «attempted relative import beyond top-level package», когда
    подниматься уже некуда.
    """

    directory = path.split("/")[:-1]
    climb = level - 1
    if climb >= len(directory):
        return None
    return ".".join(directory[: len(directory) - climb])


def _module_name(path: str) -> str:
    """Отображает POSIX-путь ``.py``-файла (относительно корня репо) в его модульное имя.

    ``a/b/c.py`` → ``a.b.c``; ``a/b/__init__.py`` → имя самого пакета ``a.b``.
    Корневой ``__init__.py`` даёт пустую строку (пакета нет) — она никогда не совпадёт ни с
    одним кандидатом импорта.
    """

    parts = path.split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _module_candidates(path: str) -> list[str]:
    """Все абсолютные имена модулей, под которыми мог бы импортироваться файл ``path``.

    Обычно это просто результат ``_module_name``. Но если файл лежит под ведущей
    директорией ``src/`` (стандартный src-layout: пакет физически лежит под ``src/``, но
    этот сегмент никогда не входит в его импортируемое имя — ``src/pkg/mod.py``
    импортируется как ``pkg.mod``, а не ``src.pkg.mod``), дополнительно возвращается имя
    модуля без этого сегмента — так абсолютные внутренние импорты в src-layout репозиториях
    тоже резолвятся.
    """

    candidates = [_module_name(path)]
    parts = path.split("/")
    if len(parts) > 1 and parts[0] == "src":
        candidates.append(_module_name("/".join(parts[1:])))
    return [c for c in candidates if c]


def build_graph_edges(documents: Iterable[Document]) -> list[GraphEdge]:
    """Резолвит внутренние Python-импорты документов снапшота в рёбра файл→файл.

    Рассматриваются только документы вида CODE, чей путь оканчивается на ``.py``. Сперва
    строится карта модуль → путь по всем кандидатам ``_module_candidates`` каждого документа
    (при коллизии побеждает лексикографически меньший путь), затем для каждого такого
    документа резолвится каждый кандидат из ``extract_import_modules`` (и абсолютный, и —
    с учётом собственного пути документа — относительный), если он есть в этой карте. Каждое
    успешное разрешение в *другой* путь даёт ``GraphEdge(src=doc.path, dst=resolved,
    kind="import", snapshot_id=doc.snapshot_id)`` — рёбер файла в самого себя не бывает.
    Результат дедуплицируется и сортируется по ``(src, dst)``.

    Импорты, которые не резолвятся ни в один файл снапшота (сторонние пакеты, стандартная
    библиотека), просто пропускаются — анализ влияния интересуют только зависимости внутри
    самого репозитория. Предполагается, что все документы принадлежат одному снапшоту; в
    ребре записывается snapshot_id импортирующего файла.
    """

    py_docs = [doc for doc in documents if doc.kind is FileKind.CODE and doc.path.endswith(".py")]

    module_to_path: dict[str, str] = {}
    for doc in py_docs:
        for module in _module_candidates(doc.path):
            existing = module_to_path.get(module)
            if existing is None or doc.path < existing:
                module_to_path[module] = doc.path

    edges: set[GraphEdge] = set()
    for doc in py_docs:
        for candidate in extract_import_modules(doc.text, path=doc.path):
            resolved = module_to_path.get(candidate)
            if resolved is not None and resolved != doc.path:
                edges.add(
                    GraphEdge(
                        src=doc.path,
                        dst=resolved,
                        kind="import",
                        snapshot_id=doc.snapshot_id,
                    )
                )

    return sorted(edges, key=lambda e: (e.src, e.dst))


def transitive_dependents(
    graph_store: GraphStore,
    snapshot_id: str,
    seeds: Iterable[str],
    *,
    max_depth: int | None = None,
) -> set[str]:
    """Все узлы, транзитивно зависящие от ``seeds``, в пределах одного снапшота.

    Обход в ширину по обратным рёбрам: начиная от ``seeds``, фронт волны на каждом шаге
    расширяется через ``GraphStore.dependents`` (прямые импортёры узла). Возвращает
    достигнутые узлы, *исключая сами seeds* — нас интересует, что затронется изменением
    seeds, а не они сами. Seed, достижимый из другого seed, тоже исключается.

    Множество посещённых узлов защищает от зацикливания (если A импортирует B, а B — A, всё
    равно завершится). ``max_depth`` ограничивает число шагов от seeds: ``1`` — только прямые
    зависимые, ``None`` — полное замыкание; при ``max_depth=0`` (как и при пустых ``seeds``)
    результат пустой.
    """

    seed_set = set(seeds)
    impacted: set[str] = set()
    frontier = seed_set
    depth = 0
    while frontier and (max_depth is None or depth < max_depth):
        nxt: set[str] = set()
        for node in frontier:
            for dep in graph_store.dependents(snapshot_id, node):
                if dep not in seed_set and dep not in impacted:
                    nxt.add(dep)
        impacted |= nxt
        frontier = nxt
        depth += 1
    return impacted

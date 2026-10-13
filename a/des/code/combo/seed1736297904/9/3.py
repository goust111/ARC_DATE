from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
MIN_LINE_LEN = 5
MIN_VISIBLE_CELLS = 2
MAX_GENERATION_ATTEMPTS = 400
STACK_BUILD_ATTEMPTS = 500

Coord = tuple[int, int]


@dataclass(frozen=True)
class LineSpec:
    color: int
    kind: str
    key: int
    cells: tuple[Coord, ...]
    cellset: frozenset[Coord]


@dataclass(frozen=True)
class LocalLine:
    kind: str
    key: int
    cells: tuple[Coord, ...]
    cellset: frozenset[Coord]


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    colors, counts = np.unique(grid, return_counts=True)
    return {int(color): int(count) for color, count in zip(colors, counts)}


def _legend_info(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    counts = _color_counts(grid)
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for col in (0, w - 1):
        rows: list[int] = []
        colors: list[int] = []
        valid = True
        for r in range(h):
            color = int(grid[r, col])
            if color == BG:
                continue
            if counts.get(color, 0) != 1:
                valid = False
                break
            rows.append(r)
            colors.append(color)
        if valid and len(rows) >= 2:
            candidates.append((len(rows), col, rows, colors))

    assert candidates, "expected a border legend made of singleton colors"
    best_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_len]
    assert len(best) == 1, "legend side must be unique"
    _, legend_col, legend_rows, legend_colors = best[0]
    return legend_col, legend_rows, legend_colors


def _play_columns(shape: tuple[int, int], legend_col: int) -> tuple[int, int]:
    _h, w = shape
    if legend_col == 0:
        return 1, w - 1
    return 0, w - 2


def _infer_line_kind(coords: np.ndarray) -> tuple[str, int]:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    options: list[tuple[str, int]] = []
    if np.all(rows == rows[0]):
        options.append(("h", int(rows[0])))
    if np.all(cols == cols[0]):
        options.append(("v", int(cols[0])))
    deltas = rows - cols
    if np.all(deltas == deltas[0]):
        options.append(("d1", int(deltas[0])))
    sums = rows + cols
    if np.all(sums == sums[0]):
        options.append(("d2", int(sums[0])))
    assert len(options) == 1, "path pixels must determine one straight orientation"
    return options[0]


def _actual_line_cells(shape: tuple[int, int], legend_col: int, kind: str, key: int) -> tuple[Coord, ...]:
    h, _w = shape
    c0, c1 = _play_columns(shape, legend_col)
    cells: list[Coord] = []
    if kind == "h":
        cells = [(key, c) for c in range(c0, c1 + 1)]
    elif kind == "v":
        cells = [(r, key) for r in range(h)]
    elif kind == "d1":
        for r in range(h):
            c = r - key
            if c0 <= c <= c1:
                cells.append((r, c))
    else:
        assert kind == "d2"
        for r in range(h):
            c = key - r
            if c0 <= c <= c1:
                cells.append((r, c))
    assert len(cells) >= 2, "reconstructed path must be non-trivial"
    return tuple(cells)


def _recover_paths(grid: np.ndarray) -> tuple[int, list[int], dict[int, LineSpec]]:
    legend_col, _legend_rows, legend_colors = _legend_info(grid)
    legend_set = set(legend_colors)
    path_colors = [int(color) for color in np.unique(grid) if int(color) not in legend_set and int(color) != BG]
    assert len(path_colors) == len(legend_colors), "legend size must match the number of paths"

    specs: dict[int, LineSpec] = {}
    for color in path_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 2, "each path must remain visible in at least two cells"
        kind, key = _infer_line_kind(coords)
        cells = _actual_line_cells(grid.shape, legend_col, kind, key)
        cellset = frozenset(cells)
        visible = {tuple(map(int, coord)) for coord in coords}
        assert visible.issubset(cellset), "visible pixels must lie on the recovered full line"
        specs[color] = LineSpec(color=color, kind=kind, key=key, cells=cells, cellset=cellset)
    return legend_col, legend_colors, specs


def _layer_order(grid: np.ndarray, specs: dict[int, LineSpec]) -> list[int]:
    colors = list(specs)
    edges = {color: set() for color in colors}
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = specs[a].cellset & specs[b].cellset
            assert len(inter) <= 1, "two straight paths may intersect in at most one cell"
            if not inter:
                continue
            cell = next(iter(inter))
            visible = int(grid[cell])
            if visible == a:
                edges[b].add(a)
            elif visible == b:
                edges[a].add(b)
            else:
                raise AssertionError("intersection must show one of the crossing paths")

    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indegree[dst] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in colors if color in remaining and indegree[color] == 0]
        assert len(ready) == 1, "occlusion clues must imply a unique stack order"
        current = ready[0]
        remaining.remove(current)
        order.append(current)
        for dst in edges[current]:
            indegree[dst] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden stacking order of straight paths from visible crossings,
    then repaint the full paths using the border legend from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_col, legend_colors, specs = _recover_paths(g)
    order = _layer_order(g, specs)

    out = np.zeros_like(g)
    out[:, legend_col] = g[:, legend_col]
    for rank, color in enumerate(order):
        paint = int(legend_colors[rank])
        for r, c in specs[color].cells:
            out[r, c] = paint
    return out


def _local_line_cells(h: int, w: int, kind: str, key: int) -> tuple[Coord, ...]:
    cells: list[Coord] = []
    if kind == "h":
        cells = [(key, c) for c in range(w)]
    elif kind == "v":
        cells = [(r, key) for r in range(h)]
    elif kind == "d1":
        for r in range(h):
            c = r - key
            if 0 <= c < w:
                cells.append((r, c))
    else:
        assert kind == "d2"
        for r in range(h):
            c = key - r
            if 0 <= c < w:
                cells.append((r, c))
    return tuple(cells)


def _local_candidates(h: int, w: int) -> list[LocalLine]:
    candidates: list[LocalLine] = []
    for r in range(h):
        cells = _local_line_cells(h, w, "h", r)
        if len(cells) >= MIN_LINE_LEN:
            candidates.append(LocalLine("h", r, cells, frozenset(cells)))
    for c in range(w):
        cells = _local_line_cells(h, w, "v", c)
        if len(cells) >= MIN_LINE_LEN:
            candidates.append(LocalLine("v", c, cells, frozenset(cells)))
    for d in range(-(w - 1), h):
        cells = _local_line_cells(h, w, "d1", d)
        if len(cells) >= MIN_LINE_LEN:
            candidates.append(LocalLine("d1", d, cells, frozenset(cells)))
    for s in range(h + w - 1):
        cells = _local_line_cells(h, w, "d2", s)
        if len(cells) >= MIN_LINE_LEN:
            candidates.append(LocalLine("d2", s, cells, frozenset(cells)))
    return candidates


def _local_intersection(a: LocalLine, b: LocalLine) -> Coord | None:
    inter = a.cellset & b.cellset
    if not inter:
        return None
    assert len(inter) == 1
    return next(iter(inter))


def _visible_count(stack: list[LocalLine], idx: int) -> int:
    line = stack[idx]
    hidden = 0
    for higher in stack[idx + 1 :]:
        if _local_intersection(line, higher) is not None:
            hidden += 1
    return len(line.cells) - hidden


def _choose_stack(rng: random.Random, h: int, w: int, n_paths: int) -> list[LocalLine] | None:
    candidates = _local_candidates(h, w)
    for _ in range(STACK_BUILD_ATTEMPTS):
        stack: list[LocalLine] = []
        usage: dict[Coord, int] = {}
        first = rng.choice(candidates)
        stack.append(first)
        for cell in first.cells:
            usage[cell] = 1

        while len(stack) < n_paths:
            prev = stack[-1]
            pool: list[LocalLine] = []
            for cand in candidates:
                if cand in stack:
                    continue
                if _local_intersection(prev, cand) is None:
                    continue
                if any(usage.get(cell, 0) >= 2 for cell in cand.cells):
                    continue
                pool.append(cand)
            if not pool:
                break
            chosen = rng.choice(pool)
            stack.append(chosen)
            for cell in chosen.cells:
                usage[cell] = usage.get(cell, 0) + 1

        if len(stack) != n_paths:
            continue
        if min(_visible_count(stack, idx) for idx in range(len(stack))) < MIN_VISIBLE_CELLS:
            continue
        return stack
    return None


def _shift_cells(cells: tuple[Coord, ...], legend_side: str) -> tuple[Coord, ...]:
    if legend_side == "left":
        return tuple((r, c + 1) for r, c in cells)
    return cells


def _render_scene(
    h: int,
    w: int,
    legend_side: str,
    legend_rows: list[int],
    legend_colors: list[int],
    stack: list[LocalLine],
    paint_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    legend_col = 0 if legend_side == "left" else w - 1
    for row, color in zip(legend_rows, legend_colors):
        grid[row, legend_col] = int(color)
    for line, color in zip(stack, paint_colors):
        for r, c in _shift_cells(line.cells, legend_side):
            grid[r, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(9, 13)
    play_w = rng.randint(8, 12)
    w = play_w + 1
    n_paths = rng.choice([3, 4])
    legend_side = rng.choice(["left", "right"])

    stack = _choose_stack(rng, h, play_w, n_paths)
    if stack is None:
        return None

    palette = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2 * n_paths)
    path_colors = palette[:n_paths]
    legend_colors = palette[n_paths:]
    legend_start = rng.randint(0, h - n_paths)
    legend_rows = list(range(legend_start, legend_start + n_paths))

    inp = _render_scene(h, w, legend_side, legend_rows, legend_colors, stack, path_colors)
    out = _render_scene(h, w, legend_side, legend_rows, legend_colors, stack, legend_colors)

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-stacking puzzle")

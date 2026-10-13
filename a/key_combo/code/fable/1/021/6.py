from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BACKGROUND = Color.BLACK
PALETTE = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
GRID_SIDE_MIN = 8
GRID_SIDE_MAX = 18
GEN_RETRY_LIMIT = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _parse_sources(grid: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BACKGROUND, "expected black background"
    assert int(g[0, 0]) == BACKGROUND, "top-left corner must stay black to avoid direction ambiguity"
    assert np.all(g[1:, 1:] == BACKGROUND), "only edge source pixels may be non-background in the input"

    left = {int(r): int(g[r, 0]) for r in range(1, g.shape[0]) if int(g[r, 0]) != BACKGROUND}
    top = {int(c): int(g[0, c]) for c in range(1, g.shape[1]) if int(g[0, c]) != BACKGROUND}
    assert left and top, "expected at least one source on each emitting edge"

    source_colors = list(left.values()) + list(top.values())
    assert len(source_colors) == len(set(source_colors)), "source colors must be pairwise distinct"
    return left, top


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    left, top = _parse_sources(g)
    h, w = g.shape
    out = g.copy()

    horizontal_heads = {row: 0 for row in left}
    vertical_heads = {col: 0 for col in top}
    stats = {
        "ties": [],
        "blocked_h": [],
        "blocked_v": [],
        "edge_h": [],
        "edge_v": [],
    }

    while horizontal_heads or vertical_heads:
        proposals: dict[tuple[int, int], list[tuple[str, int]]] = {}
        next_horizontal: dict[int, int] = {}
        next_vertical: dict[int, int] = {}

        for row, head_col in horizontal_heads.items():
            next_col = int(head_col) + 1
            if next_col >= w:
                stats["edge_h"].append(int(row))
                continue
            if int(out[row, next_col]) != BACKGROUND:
                stats["blocked_h"].append({"row": int(row), "cell": (int(row), int(next_col))})
                continue
            proposals.setdefault((int(row), int(next_col)), []).append(("H", int(row)))

        for col, head_row in vertical_heads.items():
            next_row = int(head_row) + 1
            if next_row >= h:
                stats["edge_v"].append(int(col))
                continue
            if int(out[next_row, col]) != BACKGROUND:
                stats["blocked_v"].append({"col": int(col), "cell": (int(next_row), int(col))})
                continue
            proposals.setdefault((int(next_row), int(col)), []).append(("V", int(col)))

        for cell, rays in proposals.items():
            if len(rays) > 1:
                stats["ties"].append({"cell": cell, "rays": tuple(rays)})

        for cell, rays in proposals.items():
            if len(rays) != 1:
                continue
            r, c = cell
            ray_type, axis = rays[0]
            if ray_type == "H":
                out[r, c] = int(left[axis])
                next_horizontal[int(axis)] = int(c)
            else:
                out[r, c] = int(top[axis])
                next_vertical[int(axis)] = int(r)

        horizontal_heads = next_horizontal
        vertical_heads = next_vertical

    if return_stats:
        return out, stats
    return out


def _naive_local_race(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    left, top = _parse_sources(g)
    h, w = g.shape
    out = np.full((h, w), BACKGROUND, dtype=int)

    for row, color in left.items():
        out[row, 0] = int(color)
    for col, color in top.items():
        out[0, col] = int(color)

    for row, color in left.items():
        for col in range(1, w):
            if col not in top:
                out[row, col] = int(color)
                continue
            if row == col:
                out[row, col] = BACKGROUND
            elif row > col:
                out[row, col] = int(color)
            else:
                out[row, col] = int(top[col])

    for col, color in top.items():
        for row in range(1, h):
            if row not in left:
                out[row, col] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate simultaneous rightward and downward edge rays that race for cells, block later arrivals, and cancel on ties.

    Input structure:
    - The background is black.
    - The top-left corner is black, so every colored source belongs unambiguously to either the left edge or the top edge.
    - The left edge contains one or more distinct-colored source pixels below the corner, and the top edge contains one or more distinct-colored source pixels to the right of the corner.
    - Every non-source interior cell starts as background.

    Rule:
    - Each left-edge source emits a horizontal ray to the right, and each top-edge source emits a vertical ray downward.
    - All active rays advance one cell per tick in parallel.
    - If a ray's next cell was already painted by another ray on an earlier tick, that ray stops without entering the cell.
    - If a horizontal and a vertical ray reach the same empty cell on the same tick, that cell stays black and both rays stop.
    - Any ray that is never stopped keeps painting cells until it reaches the canvas boundary.

    Key insight:
    - Cell ownership depends on the global time-ordered simulation, not only on local row/column distances, because an early stop removes that ray from all later crossings.
    """
    return _simulate(np.asarray(grid, dtype=int))


def _build_candidate() -> np.ndarray:
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    common_limit = min(h, w) - 1
    assert common_limit >= 5

    t, a, b, c, d = sorted(random.sample(range(1, common_limit + 1), 5))
    left_rows = [t, b, d]
    top_cols = [t, a, c]

    extra_row_pool = list(range(d + 1, h))
    extra_col_pool = list(range(d + 1, w))
    if extra_row_pool and random.random() < 0.45:
        left_rows.append(random.choice(extra_row_pool))
    if extra_col_pool and random.random() < 0.45:
        top_cols.append(random.choice(extra_col_pool))

    total_sources = len(left_rows) + len(top_cols)
    colors = random.sample(PALETTE, total_sources)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for row, color in zip(left_rows, colors[: len(left_rows)]):
        grid[int(row), 0] = int(color)
    for col, color in zip(top_cols, colors[len(left_rows) :]):
        grid[0, int(col)] = int(color)
    return grid


def _has_required_semantics(grid: np.ndarray) -> bool:
    out, stats = _simulate(grid, return_stats=True)
    naive = _naive_local_race(grid)

    if not stats["ties"]:
        return False
    if not stats["blocked_h"] or not stats["blocked_v"]:
        return False
    if np.array_equal(out, naive):
        return False

    for event in stats["ties"]:
        tie_r, tie_c = event["cell"]
        if not in_bounds(out.shape, tie_r, tie_c) or int(out[tie_r, tie_c]) != BACKGROUND:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_candidate()
        if _has_required_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for synchronized racing rays")

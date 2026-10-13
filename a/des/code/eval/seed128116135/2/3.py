import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _marker_color(grid: np.ndarray, bg: int) -> int:
    top_colors = {int(v) for v in np.unique(grid[0]) if int(v) != bg}
    left_colors = {int(v) for v in np.unique(grid[:, 0]) if int(v) != bg}
    common = top_colors & left_colors
    assert len(common) == 1, "expected exactly one shared edge marker color"
    return next(iter(common))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use grey edge ticks as row/column selectors and keep only their intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid is too small for edge markers and interior content"

    bg = _bg_color(g)
    marker = _marker_color(g, bg)

    keep_rows = [r for r in range(1, h) if int(g[r, 0]) == marker]
    keep_cols = [c for c in range(1, w) if int(g[0, c]) == marker]
    assert keep_rows, "expected at least one marked row"
    assert keep_cols, "expected at least one marked column"

    return g[np.ix_(keep_rows, keep_cols)].copy()


_GRID_H_RANGE = (10, 20)
_GRID_W_RANGE = (10, 20)
_RECT_H_RANGE = (5, 11)
_RECT_W_RANGE = (5, 11)
_KEEP_RANGE = (2, 6)
_GEN_ATTEMPTS = 400
_PATTERN_TYPES = ("row", "col", "woven")


def _sample_indices(rng: random.Random, start: int, length: int) -> list[int]:
    count_hi = min(_KEEP_RANGE[1], length)
    count = rng.randint(_KEEP_RANGE[0], count_hi)
    base = list(range(start, start + length))
    for _ in range(80):
        picked = sorted(rng.sample(base, count))
        if len(picked) >= 2 and max(np.diff(picked)) >= 2:
            if len(picked) < 3:
                return picked
            gaps = [b - a for a, b in zip(picked, picked[1:])]
            if len(set(gaps)) > 1:
                return picked
    return sorted(rng.sample(base, count))


def _fill_pattern(
    grid: np.ndarray,
    r0: int,
    c0: int,
    rect_h: int,
    rect_w: int,
    colors: list[int],
    pattern_type: str,
    rng: random.Random,
) -> None:
    if pattern_type == "row":
        phase = rng.randrange(len(colors))
        for i in range(rect_h):
            grid[r0 + i, c0 : c0 + rect_w] = colors[(phase + i) % len(colors)]
        return

    if pattern_type == "col":
        phase = rng.randrange(len(colors))
        for j in range(rect_w):
            grid[r0 : r0 + rect_h, c0 + j] = colors[(phase + j) % len(colors)]
        return

    row_colors = colors[:]
    col_colors = colors[:]
    rng.shuffle(row_colors)
    rng.shuffle(col_colors)
    for i in range(rect_h):
        for j in range(rect_w):
            if (i + j) % 2 == 0:
                grid[r0 + i, c0 + j] = row_colors[i % len(row_colors)]
            else:
                grid[r0 + i, c0 + j] = col_colors[j % len(col_colors)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*_GRID_H_RANGE)
    w = rng.randint(*_GRID_W_RANGE)

    bg = rng.randrange(10)
    remaining = [c for c in range(10) if c != bg]
    marker = rng.choice(remaining)
    remaining.remove(marker)

    color_count = rng.randint(2, min(4, len(remaining)))
    pattern_colors = rng.sample(remaining, color_count)

    rect_h = rng.randint(_RECT_H_RANGE[0], min(_RECT_H_RANGE[1], h - 3))
    rect_w = rng.randint(_RECT_W_RANGE[0], min(_RECT_W_RANGE[1], w - 3))
    if rect_h < _KEEP_RANGE[0] or rect_w < _KEEP_RANGE[0]:
        return None

    r0 = rng.randint(2, h - rect_h - 1)
    c0 = rng.randint(2, w - rect_w - 1)

    grid = np.full((h, w), bg, dtype=int)
    pattern_type = rng.choice(_PATTERN_TYPES)
    _fill_pattern(grid, r0, c0, rect_h, rect_w, pattern_colors, pattern_type, rng)

    keep_rows = _sample_indices(rng, r0, rect_h)
    keep_cols = _sample_indices(rng, c0, rect_w)
    if len(keep_rows) < 2 or len(keep_cols) < 2:
        return None

    grid[keep_rows, 0] = marker
    grid[0, keep_cols] = marker

    out = main(grid)
    if out.shape[0] < 2 or out.shape[1] < 2:
        return None
    if len(np.unique(out)) < 2:
        return None

    sampled_rows = {tuple(int(v) for v in row) for row in out}
    sampled_cols = {tuple(int(v) for v in col) for col in out.T}
    if len(sampled_rows) == 1 and len(sampled_cols) == 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
_RNG = random.Random()


def _axis_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (color, start, length) runs on one axis, skipping the corner cell."""
    runs: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    i = 1
    n = len(values)
    while i < n:
        color = int(values[i])
        if color == BG:
            i += 1
            continue
        j = i + 1
        while j < n and int(values[j]) == color:
            j += 1
        assert color not in seen_colors, "each axis color must be unique"
        seen_colors.add(color)
        runs.append((color, i, j - i))
        i = j
    return runs


def _draw_rect(grid: np.ndarray, color: int, r0: int, h: int, c0: int, w: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build colored rectangles from matching tick segments on the left and top axes.

    The leftmost column encodes rectangle heights and row starts.
    The top row encodes rectangle widths and column starts.
    Matching colors define a filled rectangle in the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "grid too small for two axes and an interior"

    left_runs = _axis_runs(g[:, 0])
    top_runs = _axis_runs(g[0, :])
    top_by_color = {color: (c0, width) for color, c0, width in top_runs}

    out = g.copy()
    for color, r0, height in left_runs:
        if color not in top_by_color:
            continue
        c0, width = top_by_color[color]
        assert r0 + height <= h and c0 + width <= w, "rectangle must fit the grid"
        _draw_rect(out, color, r0, height, c0, width)
    return out


def _distribute_units(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[_RNG.randrange(slots)] += 1
    return parts


def _sample_axis_layout(axis_len: int, n_segments: int) -> list[tuple[int, int]] | None:
    """
    Sample non-touching runs inside indices 1..axis_len inclusive.

    axis_len is the usable length after removing the shared corner cell.
    """
    if n_segments <= 0:
        return []

    max_len = min(5, axis_len - (n_segments - 1))
    if max_len < 1:
        return None

    min_len = 2 if axis_len >= 3 * n_segments - 1 and _RNG.random() < 0.7 else 1
    if axis_len < n_segments * min_len + (n_segments - 1):
        min_len = 1
    if axis_len < n_segments * min_len + (n_segments - 1):
        return None

    lengths = [min_len] * n_segments
    capacity = [max_len - min_len] * n_segments
    extra = axis_len - (n_segments - 1) - sum(lengths)
    while extra > 0 and any(c > 0 for c in capacity):
        idxs = [i for i, c in enumerate(capacity) if c > 0]
        idx = _RNG.choice(idxs)
        lengths[idx] += 1
        capacity[idx] -= 1
        extra -= 1

    slack = axis_len - sum(lengths) - (n_segments - 1)
    gaps = [0] + [1] * (n_segments - 1) + [0]
    for i, add in enumerate(_distribute_units(slack, n_segments + 1)):
        gaps[i] += add

    layout: list[tuple[int, int]] = []
    pos = 1 + gaps[0]
    for i, length in enumerate(lengths):
        layout.append((pos, length))
        if i < n_segments - 1:
            pos += length + gaps[i + 1]
    return layout


def _choose_counts() -> tuple[int, int, int]:
    mode = _RNG.choice(("balanced", "left_extra", "top_extra", "both_extra"))
    if mode == "balanced":
        return _RNG.randint(1, 4), 0, 0
    if mode == "left_extra":
        return _RNG.randint(1, 3), _RNG.randint(1, 2), 0
    if mode == "top_extra":
        return _RNG.randint(1, 3), 0, _RNG.randint(1, 2)
    return _RNG.randint(1, 3), _RNG.randint(1, 2), _RNG.randint(1, 2)


def _validate_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or min(grid.shape) < 3:
        return False
    if int(grid[0, 0]) != BG:
        return False
    if np.any(grid[1:, 1:] != BG):
        return False

    left_runs = _axis_runs(grid[:, 0])
    top_runs = _axis_runs(grid[0, :])
    left_colors = {color for color, _, _ in left_runs}
    top_colors = {color for color, _, _ in top_runs}
    if not left_colors or not top_colors:
        return False

    matches = left_colors & top_colors
    if not matches:
        return False

    # Keep samples visually meaningful.
    area_scores = []
    top_by_color = {color: width for color, _, width in top_runs}
    for color, _, height in left_runs:
        if color in top_by_color:
            area_scores.append(height * top_by_color[color])
    if not area_scores or max(area_scores) < 2:
        return False

    out = main(grid)
    return np.any(out[1:, 1:] != BG)


def _gen_attempt() -> np.ndarray | None:
    n_match, n_left_only, n_top_only = _choose_counts()
    total_colors = n_match + n_left_only + n_top_only
    if total_colors > len(PALETTE):
        return None

    left_count = n_match + n_left_only
    top_count = n_match + n_top_only

    min_axis_h = max(5, left_count + (left_count - 1) + 2)
    min_axis_w = max(5, top_count + (top_count - 1) + 2)
    axis_h = _RNG.randint(min_axis_h, max(min_axis_h, 18))
    axis_w = _RNG.randint(min_axis_w, max(min_axis_w, 18))

    left_layout = _sample_axis_layout(axis_h, left_count)
    top_layout = _sample_axis_layout(axis_w, top_count)
    if left_layout is None or top_layout is None:
        return None

    colors = list(_RNG.sample(PALETTE, total_colors))
    matched = colors[:n_match]
    left_only = colors[n_match : n_match + n_left_only]
    top_only = colors[n_match + n_left_only :]

    left_colors = matched + left_only
    top_colors = matched + top_only
    _RNG.shuffle(left_colors)
    _RNG.shuffle(top_colors)

    grid = np.zeros((axis_h + 1, axis_w + 1), dtype=int)
    for color, (start, length) in zip(left_colors, left_layout):
        grid[start : start + length, 0] = color
    for color, (start, length) in zip(top_colors, top_layout):
        grid[0, start : start + length] = color

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")

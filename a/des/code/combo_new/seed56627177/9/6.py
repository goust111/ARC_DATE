from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and bool(np.all(np.diff(indices) == 1))


def _find_strip(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], list[int]]:
    top_cols = np.flatnonzero(grid[0] != bg)
    left_rows = np.flatnonzero(grid[:, 0] != bg)
    has_top = bool(top_cols.size)
    has_left = bool(left_rows.size)
    assert has_top != has_left, "expected exactly one strip on the top row or left column"

    if has_top:
        assert _is_contiguous(top_cols), "top strip must be contiguous"
        colors = [int(grid[0, int(c)]) for c in top_cols]
        cells = {(0, int(c)) for c in top_cols}
    else:
        assert _is_contiguous(left_rows), "left strip must be contiguous"
        colors = [int(grid[int(r), 0]) for r in left_rows]
        cells = {(int(r), 0) for r in left_rows}

    assert len(colors) == len(set(colors)), "strip colors must be distinct"
    return cells, colors


def _find_pattern_bbox(grid: np.ndarray, bg: int, strip_cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    mask = grid != bg
    for r, c in strip_cells:
        mask[int(r), int(c)] = False

    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one nested-rectangle pattern"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert bool(np.all(patch != bg)), "pattern bbox must be fully filled by rectangle layers"
    assert int(coords.shape[0]) == int(patch.size), "pattern must occupy a single full rectangle bbox"
    return int(r0), int(c0), int(r1), int(c1)


def _layer_ids(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows), np.minimum(cols, width - 1 - cols))


def _recolor_patch(patch: np.ndarray, strip_colors: list[int]) -> np.ndarray:
    height, width = patch.shape
    layer_ids = _layer_ids(height, width)
    n_layers = int(layer_ids.max()) + 1
    assert n_layers == len(strip_colors), "strip length must match the number of rectangle layers"

    layer_colors: list[int] = []
    out = patch.copy()
    for layer in range(n_layers):
        vals = np.unique(patch[layer_ids == layer])
        assert len(vals) == 1, "each layer must have exactly one color"
        layer_colors.append(int(vals[0]))
        out[layer_ids == layer] = int(strip_colors[layer])

    assert len(set(layer_colors)) == n_layers, "all layers must use distinct colors"
    assert set(layer_colors) == set(strip_colors), "strip colors must match the pattern palette"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the colors of a nested rectangular target using the edge strip as the new outer-to-inner palette.

    Input structure:
    - The background is the dominant color.
    - One edge carries a contiguous one-cell strip: either on the top row or on the left column.
    - Away from the edge strip, the foreground is a single filled rectangular bbox made of concentric one-pixel layers, each layer using a different color.

    Rule:
    - Read the edge strip in its visible order: left-to-right on the top row, or top-to-bottom on the left column.
    - Ignore the strip itself and locate the nested rectangle pattern.
    - Preserve every layer's geometry, but recolor layer k with the k-th strip color, from the outermost layer inward.

    Key insight:
    - The strip does not label specific old colors; it specifies the new layer order, while the nested rectangle's shape only supplies the layer skeleton.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    strip_cells, strip_colors = _find_strip(g, bg)
    r0, c0, r1, c1 = _find_pattern_bbox(g, bg, strip_cells)

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = _recolor_patch(g[r0 : r1 + 1, c0 : c1 + 1], strip_colors)
    return out


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
ARC_MAX_SIZE = 30
LAYER_COUNT_RANGE = (2, 6)
EXTRA_SIDE_RANGE = (0, 6)


def _sample_pattern_shape() -> tuple[int, int, int]:
    n_layers = random.randint(*LAYER_COUNT_RANGE)
    min_side = random.choice((2 * n_layers - 1, 2 * n_layers))
    other_side = min_side + random.randint(*EXTRA_SIDE_RANGE)
    if random.random() < 0.5:
        return n_layers, min_side, other_side
    return n_layers, other_side, min_side


def _make_pattern(height: int, width: int, layer_colors: list[int]) -> np.ndarray:
    layer_ids = _layer_ids(height, width)
    pattern = np.empty((height, width), dtype=int)
    for layer, color in enumerate(layer_colors):
        pattern[layer_ids == layer] = int(color)
    return pattern


def _sample_strip_order(layer_colors: list[int]) -> list[int]:
    order = layer_colors[:]
    while True:
        random.shuffle(order)
        if order != layer_colors:
            return order[:]


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.sum(grid == bg))
    non_bg_colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    return all(bg_count > int(np.sum(grid == color)) for color in non_bg_colors)

def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_VALUES)
    available = [color for color in COLOR_VALUES if color != bg]
    n_layers, pattern_h, pattern_w = _sample_pattern_shape()
    if n_layers > len(available):
        return None

    layer_colors = random.sample(available, n_layers)
    strip_colors = _sample_strip_order(layer_colors)
    orientation = random.choice(("top", "left"))

    if orientation == "top":
        min_h = pattern_h + 3
        min_w = max(pattern_w + 2, n_layers + 2)
    else:
        min_h = max(pattern_h + 2, n_layers + 2)
        min_w = pattern_w + 3

    if min_h > ARC_MAX_SIZE or min_w > ARC_MAX_SIZE:
        return None

    height = random.randint(min_h, ARC_MAX_SIZE)
    width = random.randint(min_w, ARC_MAX_SIZE)
    if orientation == "top":
        r0 = random.randint(2, height - pattern_h - 1)
        c0 = random.randint(1, width - pattern_w - 1)
    else:
        r0 = random.randint(1, height - pattern_h - 1)
        c0 = random.randint(2, width - pattern_w - 1)

    grid = np.full((height, width), bg, dtype=int)
    pattern = _make_pattern(pattern_h, pattern_w, layer_colors)
    grid[r0 : r0 + pattern_h, c0 : c0 + pattern_w] = pattern

    if orientation == "top":
        start = random.randint(1, width - n_layers - 1)
        for offset, color in enumerate(strip_colors):
            grid[0, start + offset] = int(color)
    else:
        start = random.randint(1, height - n_layers - 1)
        for offset, color in enumerate(strip_colors):
            grid[start + offset, 0] = int(color)

    if not _background_is_dominant(grid, bg):
        return None

    expected = grid.copy()
    expected[r0 : r0 + pattern_h, c0 : c0 + pattern_w] = _recolor_patch(pattern, strip_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-rectangle reordering input")

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location
import random


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    assert boundary.size, "expected a nonempty boundary"
    colors = np.unique(boundary)
    assert colors.size == 1, "each nested rectangle boundary must have one color"
    return int(colors[0])


def _layer_order(rect: np.ndarray) -> list[int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    order: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        order.append(color)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        assert (top, bottom, left, right) != previous, "peeling must remove at least one row or column"

    assert len(order) >= 2, "expected at least two nested colors"
    assert len(order) == len(set(order)), "each nested layer color must be unique"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the nested color layers inside each solid rectangular foreground object.

    Input structure
    - The background is the most common color.
    - Every 4-connected foreground object is a filled rectangle.
    - Each rectangle is made from inward nested rectangular color layers, with a
      uniform boundary at every peel step.

    Rule
    - For each rectangle independently, read its layer colors from outermost to
      innermost by repeatedly peeling the current uniform boundary.
    - Recolor every pixel of each layer to the color at the opposite position in
      that outer-to-inner order.
    - Keep the background unchanged.

    Key insight
    - The transformation reverses the order of the colored rectangular layers,
      not the one-cell geometric distance shells; thick layers move as whole
      color masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = g.copy()
    components = components_mask(g != bg, connectivity=4)
    assert components, "expected at least one foreground rectangle"

    for coords in components:
        r0, c0, r1, c1 = bbox_coords(coords)
        rect_mask = g[r0 : r1 + 1, c0 : c1 + 1] != bg
        assert np.all(rect_mask), "foreground component must fill its bounding box"

        rect = g[r0 : r1 + 1, c0 : c1 + 1]
        layer_colors = _layer_order(rect)
        color_map = dict(zip(layer_colors, reversed(layer_colors)))

        out_rect = out[r0 : r1 + 1, c0 : c1 + 1]
        for src, dst in color_map.items():
            out_rect[rect == src] = dst

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle placement bounds.
_GRID_ROW_RANGE = (16, 20)
_GRID_COL_RANGE = (16, 19)
_RECT_SIZE_RANGE = (6, 13)
_RECT_COUNT_RANGE = (1, 3)
_MIN_RECT_GAP = 1

# Layer construction and palette balancing.
_LAYER_MIN_COUNT = 3
_LAYER_MAX_COUNT = 5
_LAYER_MAX_THICKNESS = 2
_COLOR_MIN, _COLOR_MAX = 1, 9
_COLOR_BALANCE_CANDIDATES = 4
_GEN_COLORS = list(range(_COLOR_MIN, _COLOR_MAX + 1))

# Validation and retry controls.
_BACKGROUND_COLOR = 0
_BINCNT_MIN_LENGTH = 10
_GEN_MAX_ATTEMPTS = 300


def _rect_sizes() -> range:
    return range(_RECT_SIZE_RANGE[0], _RECT_SIZE_RANGE[1] + 1)


def _rand_rect_extent(limit: int) -> int:
    return random.randint(_RECT_SIZE_RANGE[0], min(_RECT_SIZE_RANGE[1], limit))


def _fitting_pairs(limit: int) -> list[tuple[int, int]]:
    return [(a, b) for a in _rect_sizes() for b in _rect_sizes() if a + b + _MIN_RECT_GAP <= limit]


def _fitting_pair_with_single(limit: int) -> list[tuple[int, int, int]]:
    return [
        (a, b, single) for a in _rect_sizes() for b in _rect_sizes() for single in _rect_sizes()
        if max(a, b) + single + _MIN_RECT_GAP <= limit
    ]


def _gen_pick_thicknesses(rect_h: int, rect_w: int, n_layers: int) -> list[int] | None:
    """Choose legal ring thicknesses while reserving room for later layers."""
    thicks: list[int] = []
    h, w = int(rect_h), int(rect_w)
    for i in range(n_layers - 1):
        remaining_layers = n_layers - i - 1
        min_after = 2 * (remaining_layers - 1) + 1
        max_t = min(_LAYER_MAX_THICKNESS, (min(h, w) - min_after) // 2)
        if max_t < 1:
            return None
        t = random.randint(_MIN_RECT_GAP, max_t)
        thicks.append(t)
        h -= 2 * t
        w -= 2 * t
    return thicks


def _gen_layer_areas(rect_h: int, rect_w: int, thicks: list[int]) -> list[int]:
    areas: list[int] = []
    h, w = int(rect_h), int(rect_w)
    for t in thicks:
        inner_h, inner_w = h - 2 * t, w - 2 * t
        areas.append(h * w - inner_h * inner_w)
        h, w = inner_h, inner_w
    areas.append(h * w)
    return areas


def _gen_layered_rect(
    rect_h: int, rect_w: int, color_usage: dict[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the sampled rectangle and its layer-reversed oracle result."""
    max_layers = min(_LAYER_MAX_COUNT, (min(rect_h, rect_w) + 1) // 2)
    if max_layers < _LAYER_MIN_COUNT:
        return None
    n_layers = random.randint(_LAYER_MIN_COUNT, max_layers)
    thicks = _gen_pick_thicknesses(rect_h, rect_w, n_layers)
    if thicks is None:
        return None

    areas = _gen_layer_areas(rect_h, rect_w, thicks)
    colors: list[int | None] = [None] * n_layers
    available = _GEN_COLORS.copy()
    for layer_i in sorted(range(n_layers), key=lambda i: areas[i], reverse=True):
        ranked = sorted(available, key=lambda c: (color_usage[c], random.random()))
        top_n = min(_COLOR_BALANCE_CANDIDATES, len(ranked))
        color = random.choice(ranked[:top_n])
        colors[layer_i] = color
        available.remove(color)
        color_usage[color] += areas[layer_i]

    layer_colors = [int(c) for c in colors]
    rect = np.full((rect_h, rect_w), _BACKGROUND_COLOR, dtype=int)
    out = np.zeros_like(rect)
    top, left = 0, 0
    bottom, right = int(rect_h), int(rect_w)
    for i, color in enumerate(layer_colors):
        out_color = layer_colors[-1 - i]
        rect[top:bottom, left:right] = color
        out[top:bottom, left:right] = out_color
        if i < len(thicks):
            t = thicks[i]
            top += t
            bottom -= t
            left += t
            right -= t
    return rect, out


def _gen_one_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    h = _rand_rect_extent(H)
    w = _rand_rect_extent(W)
    canvas = np.full((H, W), _BACKGROUND_COLOR, dtype=int)
    patch = np.ones((h, w), dtype=int)
    r, c = random_free_location(canvas, patch, transparent=_BACKGROUND_COLOR)
    return [(r, c, h, w)]


def _gen_two_row_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    width_pairs = _fitting_pairs(W)
    if not width_pairs:
        raise ValueError("no row layout")
    w0, w1 = random.choice(width_pairs)
    h0 = _rand_rect_extent(H)
    h1 = _rand_rect_extent(H)
    gap = random.randint(_MIN_RECT_GAP, W - w0 - w1)
    c_base = random.randint(0, W - w0 - gap - w1)
    shelf_h = max(h0, h1)
    r_base = random.randint(0, H - shelf_h)
    r0 = r_base + random.randint(0, shelf_h - h0)
    r1 = r_base + random.randint(0, shelf_h - h1)
    return [(r0, c_base, h0, w0), (r1, c_base + w0 + gap, h1, w1)]


def _gen_two_col_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    height_pairs = _fitting_pairs(H)
    if not height_pairs:
        raise ValueError("no column layout")
    h0, h1 = random.choice(height_pairs)
    w0 = _rand_rect_extent(W)
    w1 = _rand_rect_extent(W)
    gap = random.randint(_MIN_RECT_GAP, H - h0 - h1)
    r_base = random.randint(0, H - h0 - gap - h1)
    shelf_w = max(w0, w1)
    c_base = random.randint(0, W - shelf_w)
    c0 = c_base + random.randint(0, shelf_w - w0)
    c1 = c_base + random.randint(0, shelf_w - w1)
    return [(r_base, c0, h0, w0), (r_base + h0 + gap, c1, h1, w1)]


def _gen_two_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    if random.random() < 0.5:
        return _gen_two_row_layout(H, W)
    return _gen_two_col_layout(H, W)


def _gen_three_row_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    height_triples = _fitting_pair_with_single(H)
    width_pairs = _fitting_pairs(W)
    if not height_triples or not width_pairs:
        raise ValueError("no three-row layout")

    h0, h1, h2 = random.choice(height_triples)
    w0, w1 = random.choice(width_pairs)
    w2 = _rand_rect_extent(W)
    pair_h = max(h0, h1)
    gap_r = random.randint(_MIN_RECT_GAP, H - pair_h - h2)
    r_base = random.randint(0, H - pair_h - gap_r - h2)
    pair_first = random.random() < 0.5
    if pair_first:
        pair_r, single_r = r_base, r_base + pair_h + gap_r
    else:
        single_r, pair_r = r_base, r_base + h2 + gap_r

    gap_c = random.randint(_MIN_RECT_GAP, W - w0 - w1)
    c_pair = random.randint(0, W - w0 - gap_c - w1)
    r0 = pair_r + random.randint(0, pair_h - h0)
    r1 = pair_r + random.randint(0, pair_h - h1)
    c2 = random.randint(0, W - w2)
    return [
        (r0, c_pair, h0, w0),
        (r1, c_pair + w0 + gap_c, h1, w1),
        (single_r, c2, h2, w2),
    ]


def _gen_three_col_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    width_triples = _fitting_pair_with_single(W)
    height_pairs = _fitting_pairs(H)
    if not width_triples or not height_pairs:
        raise ValueError("no three-column layout")

    w0, w1, w2 = random.choice(width_triples)
    h0, h1 = random.choice(height_pairs)
    h2 = _rand_rect_extent(H)
    pair_w = max(w0, w1)
    gap_c = random.randint(_MIN_RECT_GAP, W - pair_w - w2)
    c_base = random.randint(0, W - pair_w - gap_c - w2)
    pair_first = random.random() < 0.5
    if pair_first:
        pair_c, single_c = c_base, c_base + pair_w + gap_c
    else:
        single_c, pair_c = c_base, c_base + w2 + gap_c

    gap_r = random.randint(_MIN_RECT_GAP, H - h0 - h1)
    r_pair = random.randint(0, H - h0 - gap_r - h1)
    c0 = pair_c + random.randint(0, pair_w - w0)
    c1 = pair_c + random.randint(0, pair_w - w1)
    r2 = random.randint(0, H - h2)
    return [
        (r_pair, c0, h0, w0),
        (r_pair + h0 + gap_r, c1, h1, w1),
        (r2, single_c, h2, w2),
    ]


def _gen_three_layout(H: int, W: int) -> list[tuple[int, int, int, int]]:
    if random.random() < 0.5:
        return _gen_three_row_layout(H, W)
    return _gen_three_col_layout(H, W)


def _gen_layout(H: int, W: int, count: int) -> list[tuple[int, int, int, int]]:
    if count == 1:
        return _gen_one_layout(H, W)
    if count == 2:
        return _gen_two_layout(H, W)
    return _gen_three_layout(H, W)


def _gen_attempt() -> np.ndarray | None:
    """Sample one complete grid and validate it against the solver oracle."""
    H = random.randint(*_GRID_ROW_RANGE)
    W = random.randint(*_GRID_COL_RANGE)
    count = random.randint(*_RECT_COUNT_RANGE)
    try:
        rect_specs = _gen_layout(H, W, count)
    except ValueError:
        return None

    made = _gen_render_rectangles(H, W, rect_specs)
    if made is None:
        return None
    grid, expected = made

    if not _gen_background_is_dominant(grid):
        return None
    if not _gen_matches_solver(grid, expected):
        return None
    return grid


def _gen_render_rectangles(
    H: int, W: int, rect_specs: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Render input and oracle grids, or None if a rectangle cannot be layered."""
    grid = np.full((H, W), _BACKGROUND_COLOR, dtype=int)
    expected = np.zeros_like(grid)
    color_usage = {color: 0 for color in _GEN_COLORS}
    for r, c, h, w in rect_specs:
        made = _gen_layered_rect(h, w, color_usage)
        if made is None:
            return None
        rect, out_rect = made
        paste(grid, rect, r, c)
        paste(expected, out_rect, r, c)
    return grid, expected


def _gen_background_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.reshape(-1), minlength=_BINCNT_MIN_LENGTH)
    return counts[_BACKGROUND_COLOR] > int(np.max(counts[_COLOR_MIN:]))


def _gen_matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to construct a valid 8dae5dfc input")

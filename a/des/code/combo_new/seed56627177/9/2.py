from __future__ import annotations

import random

import numpy as np


ARC_GRID_MAX = 30
PALETTE_SIZE = 10
RECT_COUNT_RANGE = (2, 5)
TOP_HEIGHT_RANGE = (2, 5)
TOP_WIDTH_RANGE = (2, 5)
EXPAND_PER_SIDE_MAX = 2
EXPAND_TOTAL_MAX = 4
MIN_VISIBLE_PIXELS = 3
GEN_RETRIES = 400
CHAIN_RETRIES = 120
LAYER_RETRIES = 60


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=PALETTE_SIZE)
    return int(np.argmax(counts))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_shape(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    h, w = _rect_shape(rect)
    return h * w


def _sort_key(color: int, rects: dict[int, tuple[int, int, int, int]]) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = rects[color]
    return _rect_area(rects[color]), r0, c0, r1, c1, color


def _topological_order(
    colors: list[int],
    rects: dict[int, tuple[int, int, int, int]],
    coords_by_color: dict[int, np.ndarray],
) -> list[int]:
    graph = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower in colors:
        r0, c0, r1, c1 = rects[lower]
        for upper in colors:
            if upper == lower:
                continue
            pts = coords_by_color[upper]
            inside = (
                (pts[:, 0] >= r0)
                & (pts[:, 0] <= r1)
                & (pts[:, 1] >= c0)
                & (pts[:, 1] <= c1)
            )
            if inside.any() and lower not in graph[upper]:
                graph[upper].add(lower)
                indegree[lower] += 1

    order: list[int] = []
    ready = sorted((color for color in colors if indegree[color] == 0), key=lambda color: _sort_key(color, rects))
    while ready:
        color = ready.pop(0)
        order.append(color)
        for nxt in sorted(graph[color], key=lambda item: _sort_key(item, rects)):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda item: _sort_key(item, rects))

    assert len(order) == len(colors), "expected an acyclic occlusion order"
    return order


def _pack_rectangles(
    order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    dims = {color: _rect_shape(rects[color]) for color in order}
    out_h = max(h for h, _w in dims.values())
    out_w = sum(w for _h, w in dims.values()) + len(order) - 1
    assert out_h <= ARC_GRID_MAX and out_w <= ARC_GRID_MAX, "packed output must fit ARC bounds"

    out = np.full((out_h, out_w), bg, dtype=int)
    col = 0
    for color in order:
        h, w = dims[color]
        out[:h, col : col + w] = color
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each occluded solid rectangle and pack the layers from front to back.

    Input structure:
    - The grid has one dominant background color and several distinct foreground colors.
    - Each foreground color shows only the still-visible part of one solid axis-aligned rectangle.
    - Higher rectangles occlude lower ones, so lower colors may appear as rings, strips, L-shapes, or disconnected fragments.
    - At least one rectangle is fully visible, so its original size appears directly.

    Rule:
    - For each foreground color, take the minimal bounding box of its visible pixels as that rectangle's original full extent.
    - If pixels of color B appear inside color A's bounding box, then B is above A in the stack.
    - Topologically sort the colors from highest layer to lowest layer using those occlusion relations.
    - Output the reconstructed full rectangles as solid color blocks placed left-to-right in that layer order, with one background column between blocks and background padding below shorter rectangles.

    Key insight:
    - The visible fragments only reveal clues: their bounding boxes recover the hidden full rectangles, and the foreign colors appearing inside those boxes reveal the occlusion order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground rectangle"

    coords_by_color = {color: np.argwhere(g == color) for color in colors}
    rects = {color: _bbox(coords_by_color[color]) for color in colors}
    order = _topological_order(colors, rects, coords_by_color)
    return _pack_rectangles(order, rects, bg)


def _normalize_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    min_r = min(rect[0] for rect in rects)
    min_c = min(rect[1] for rect in rects)
    return [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in rects]


def _build_chain(rng: random.Random, n_rects: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(CHAIN_RETRIES):
        rects = [(0, 0, rng.randint(*TOP_HEIGHT_RANGE) - 1, rng.randint(*TOP_WIDTH_RANGE) - 1)]
        ok = True
        for _layer in range(1, n_rects):
            prev = rects[-1]
            placed = False
            for _ in range(LAYER_RETRIES):
                top = rng.randint(0, EXPAND_PER_SIDE_MAX)
                bottom = rng.randint(0, EXPAND_PER_SIDE_MAX)
                left = rng.randint(0, EXPAND_PER_SIDE_MAX)
                right = rng.randint(0, EXPAND_PER_SIDE_MAX)
                growth = top + bottom + left + right
                if growth == 0 or growth > EXPAND_TOTAL_MAX:
                    continue
                r0, c0, r1, c1 = prev
                rects.append((r0 - top, c0 - left, r1 + bottom, c1 + right))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        rects = _normalize_rects(rects)
        widths = [_rect_shape(rect)[1] for rect in rects]
        heights = [_rect_shape(rect)[0] for rect in rects]
        if sum(widths) + n_rects - 1 > ARC_GRID_MAX:
            continue
        if max(heights) > ARC_GRID_MAX:
            continue
        return rects
    return None


def _shift_rects(
    rects: list[tuple[int, int, int, int]],
    row_shift: int,
    col_shift: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift) for r0, c0, r1, c1 in rects]


def _render_scene(
    rects_top_to_bottom: list[tuple[int, int, int, int]],
    colors_top_to_bottom: list[int],
    bg: int,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, rect in reversed(list(zip(colors_top_to_bottom, rects_top_to_bottom))):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _rects_to_dict(
    colors_top_to_bottom: list[int],
    rects_top_to_bottom: list[tuple[int, int, int, int]],
) -> dict[int, tuple[int, int, int, int]]:
    return {color: rect for color, rect in zip(colors_top_to_bottom, rects_top_to_bottom)}


def _valid_visibility(
    grid: np.ndarray,
    bg: int,
    colors_top_to_bottom: list[int],
    rects_top_to_bottom: list[tuple[int, int, int, int]],
) -> bool:
    for idx, (color, rect) in enumerate(zip(colors_top_to_bottom, rects_top_to_bottom)):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False
        if _bbox(coords) != rect:
            return False
        if int(coords.shape[0]) < MIN_VISIBLE_PIXELS:
            return False
        if idx == 0 and int(coords.shape[0]) != _rect_area(rect):
            return False

    counts = np.bincount(grid.ravel(), minlength=PALETTE_SIZE)
    if int(np.argmax(counts)) != bg:
        return False
    return True


def _expected_output(
    colors_top_to_bottom: list[int],
    rects_top_to_bottom: list[tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    rects = _rects_to_dict(colors_top_to_bottom, rects_top_to_bottom)
    return _pack_rectangles(colors_top_to_bottom, rects, bg)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.randint(*RECT_COUNT_RANGE)
    rects_local = _build_chain(rng, n_rects)
    if rects_local is None:
        return None

    local_h = max(rect[2] for rect in rects_local) + 1
    local_w = max(rect[3] for rect in rects_local) + 1
    if local_h > ARC_GRID_MAX or local_w > ARC_GRID_MAX:
        return None

    max_extra_h = ARC_GRID_MAX - local_h
    max_extra_w = ARC_GRID_MAX - local_w
    extra_h = rng.randint(0, min(8, max_extra_h))
    extra_w = rng.randint(0, min(8, max_extra_w))
    top_pad = rng.randint(0, extra_h)
    left_pad = rng.randint(0, extra_w)
    bottom_pad = extra_h - top_pad
    right_pad = extra_w - left_pad

    grid_h = local_h + top_pad + bottom_pad
    grid_w = local_w + left_pad + right_pad
    if grid_h > ARC_GRID_MAX or grid_w > ARC_GRID_MAX:
        return None

    colors = rng.sample(range(PALETTE_SIZE), n_rects + 1)
    bg = colors[0]
    rect_colors_top_to_bottom = colors[1:]
    rects_shifted = _shift_rects(rects_local, top_pad, left_pad)
    grid = _render_scene(rects_shifted, rect_colors_top_to_bottom, bg, (grid_h, grid_w))

    if not _valid_visibility(grid, bg, rect_colors_top_to_bottom, rects_shifted):
        return None

    expected = _expected_output(rect_colors_top_to_bottom, rects_shifted, bg)
    if expected.shape[0] > ARC_GRID_MAX or expected.shape[1] > ARC_GRID_MAX:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-rectangle scene")

from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LAYER_COUNT_RANGE = (2, 5)
MAX_OUTER_SIDE = 28
MAX_GENERATION_ATTEMPTS = 300


def _border_values(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    vals: list[int] = []
    for c in range(c0, c1 + 1):
        vals.append(int(grid[r0, c]))
    for r in range(r0 + 1, r1):
        vals.append(int(grid[r, c1]))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            vals.append(int(grid[r1, c]))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            vals.append(int(grid[r, c0]))
    return np.array(vals, dtype=int)


def _uniform_nonbg_border_color(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    vals = _border_values(grid, bbox)
    assert vals.size > 0, "frame border must be non-empty"
    color = int(vals[0])
    assert color != BG and np.all(vals == color), "expected a monochrome rectangular frame"
    return color


def _maybe_uniform_nonbg_border_color(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int | None:
    vals = _border_values(grid, bbox)
    if vals.size == 0:
        return None
    color = int(vals[0])
    if color == BG or not np.all(vals == color):
        return None
    return color


def _extract_fill_color(region: np.ndarray) -> int | None:
    filled = np.argwhere(region != BG)
    if filled.size == 0:
        return None

    (r0, c0), (r1, c1) = filled.min(axis=0), filled.max(axis=0)
    color_values = np.unique(region[region != BG])
    assert len(color_values) == 1, "inner fill must be monochrome"
    color = int(color_values[0])

    patch = region[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    assert np.all(patch == color), "inner fill must be a solid rectangle"

    mask = np.zeros_like(region, dtype=bool)
    mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    assert not np.any((region != BG) & ~mask), "inner fill must be a single rectangle"
    return color


def _extract_sequence(grid: np.ndarray) -> tuple[list[int], int]:
    g = np.asarray(grid, dtype=int)
    filled = np.argwhere(g != BG)
    assert filled.size > 0, "expected at least one non-background structure"

    (outer_r0, outer_c0), (outer_r1, outer_c1) = filled.min(axis=0), filled.max(axis=0)
    current = (int(outer_r0), int(outer_c0), int(outer_r1), int(outer_c1))
    colors: list[int] = []

    while True:
        r0, c0, r1, c1 = current
        colors.append(_uniform_nonbg_border_color(g, current))

        inner_r0, inner_c0 = r0 + 1, c0 + 1
        inner_r1, inner_c1 = r1 - 1, c1 - 1
        if inner_r0 > inner_r1 or inner_c0 > inner_c1:
            break

        next_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        can_descend = False
        if next_bbox[0] <= next_bbox[2] and next_bbox[1] <= next_bbox[3]:
            gap_bbox = (inner_r0, inner_c0, inner_r1, inner_c1)
            if np.all(_border_values(g, gap_bbox) == BG):
                can_descend = _maybe_uniform_nonbg_border_color(g, next_bbox) is not None

        if can_descend:
            current = next_bbox
            continue

        fill = _extract_fill_color(g[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1])
        if fill is not None:
            colors.append(fill)
        break

    return colors, int(outer_r1 - outer_r0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Linearize nested frame colors into left-to-right vertical stripes.

    Input structure:
    - The background is black.
    - A single structure consists of several one-pixel-thick rectangular frames
      nested concentrically, with a one-pixel black gap between consecutive frames.
    - Each frame has one uniform color, and the innermost frame may contain an
      additional monochrome filled rectangle somewhere inside its interior.

    Rule:
    - Detect the nested frames from the outermost layer inward and read their
      colors in that spatial order.
    - If the innermost frame encloses a solid monochrome filled rectangle,
      append that fill color after the frame colors.
    - Return a new grid whose height equals the outer frame height and whose
      columns are full-height stripes colored by the extracted sequence.

    Key insight:
    - The two-dimensional nesting depth is the signal: every deeper rectangle
      contributes the next item in a one-dimensional color sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors, height = _extract_sequence(g)
    out = np.repeat(np.array(colors, dtype=int)[None, :], height, axis=0)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _stripe_output(height: int, colors: list[int]) -> np.ndarray:
    return np.repeat(np.array(colors, dtype=int)[None, :], int(height), axis=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layers = rng.randint(*LAYER_COUNT_RANGE)
    max_inner_side = MAX_OUTER_SIDE - 4 * (layers - 1)
    if max_inner_side < 3:
        return None

    inner_h = rng.randint(3, max_inner_side)
    inner_w = rng.randint(3, max_inner_side)
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)

    has_fill = bool(rng.randint(0, 1))
    needed_colors = layers + int(has_fill)
    colors = rng.sample(PALETTE, needed_colors)
    frame_colors = colors[:layers]
    fill_color = colors[-1] if has_fill else None

    grid_h = rng.randint(outer_h + 2, 30)
    grid_w = rng.randint(outer_w + 2, 30)
    top = rng.randint(1, grid_h - outer_h - 1)
    left = rng.randint(1, grid_w - outer_w - 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)

    for layer, color in enumerate(frame_colors):
        inset = 2 * layer
        bbox = (
            top + inset,
            left + inset,
            top + outer_h - 1 - inset,
            left + outer_w - 1 - inset,
        )
        _draw_frame(grid, bbox, int(color))

    if has_fill:
        inner_top = top + 2 * (layers - 1) + 1
        inner_left = left + 2 * (layers - 1) + 1
        inner_hole_h = inner_h - 2
        inner_hole_w = inner_w - 2
        fill_h = rng.randint(1, inner_hole_h)
        fill_w = rng.randint(1, inner_hole_w)
        fill_r0 = rng.randint(inner_top, inner_top + inner_hole_h - fill_h)
        fill_c0 = rng.randint(inner_left, inner_left + inner_hole_w - fill_w)
        grid[fill_r0 : fill_r0 + fill_h, fill_c0 : fill_c0 + fill_w] = int(fill_color)

    expected = _stripe_output(outer_h, frame_colors + ([int(fill_color)] if has_fill else []))
    try:
        got = main(grid)
    except Exception:
        return None
    if np.array_equal(got, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")

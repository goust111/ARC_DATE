from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PALETTE = [BLUE, RED, YELLOW, 6, 7, 8, 9]


def _parse_start_bar(grid: np.ndarray) -> tuple[int, int]:
    green_cells = np.argwhere(grid == GREEN)
    assert green_cells.shape[0] == 2, "expected exactly two green start-bar pixels"
    cells = sorted((int(r), int(c)) for r, c in green_cells)
    (r0, c0), (r1, c1) = cells
    assert c0 == 0 and c1 == 0, "start bar must lie on the left edge"
    assert r1 == r0 + 1, "start bar must be a vertical 2-pixel bar"
    return r0, r1


def _parse_counter(grid: np.ndarray) -> int:
    col = grid[:, -1]
    grey_rows = [int(r) for r in np.where(col == GREY)[0]]
    assert 3 <= len(grey_rows) <= 7, "counter length must be 3..7"
    assert grey_rows == list(range(len(grey_rows))), "counter must be top-aligned and contiguous"
    assert int(np.count_nonzero(grid == GREY)) == len(grey_rows), "grey pixels must only appear in the counter"
    return len(grey_rows)


def _parse_stripes(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    r = h - 1
    stripes: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    while r >= 0 and not np.all(grid[r] == BLACK):
        row = grid[r]
        color = int(row[0])
        assert color != BLACK, "bottom control region must be colored"
        assert np.all(row == color), "each stripe row must be full-width and uniform"
        height = 1
        r -= 1
        while r >= 0 and np.all(grid[r] == color):
            height += 1
            r -= 1
        assert 1 <= height <= 2, "stripe height must be 1 or 2"
        assert color not in seen_colors, "stripe colors must be distinct"
        seen_colors.add(color)
        stripes.append((color, height))

    assert 2 <= len(stripes) <= 4, "expected 2..4 bottom stripes"
    stripe_rows = sum(height for _color, height in stripes)
    assert r == h - stripe_rows - 1, "stripes must occupy a single bottom block"
    return stripes


def _shape_radius(stripes: list[tuple[int, int]]) -> int:
    return sum(height for _color, height in stripes) - 1


def _build_nested_shape(stripes: list[tuple[int, int]]) -> np.ndarray:
    inner_color, inner_h = stripes[0]
    side = 2 * inner_h - 1
    shape = np.full((side, side), inner_color, dtype=int)
    for color, height in stripes[1:]:
        new_side = shape.shape[0] + 2 * height
        wrapped = np.full((new_side, new_side), color, dtype=int)
        wrapped[height : height + shape.shape[0], height : height + shape.shape[1]] = shape
        shape = wrapped
    return shape


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert max(h, w) <= 30

    start_top, ray_row = _parse_start_bar(g)
    k = _parse_counter(g)
    stripes = _parse_stripes(g)
    radius = _shape_radius(stripes)
    stripe_rows = sum(height for _color, height in stripes)
    stripe_top = h - stripe_rows
    center_r, center_c = ray_row, k

    assert radius <= k, "counter must place the center far enough from the left edge"
    assert center_r - radius >= 0, "nested shape would cross the top boundary"
    assert center_c - radius >= 0, "nested shape would cross the left boundary"
    assert center_r + radius < stripe_top, "nested shape must stay above the stripe block"
    assert center_c + radius < w, "nested shape would cross the right boundary"
    assert start_top >= 0
    assert stripe_top > ray_row

    shape = _build_nested_shape(stripes)
    return {
        "grid": g,
        "shape_h": h,
        "shape_w": w,
        "start_top": start_top,
        "ray_row": ray_row,
        "k": k,
        "center": (center_r, center_c),
        "radius": radius,
        "stripe_top": stripe_top,
        "stripes": stripes,
        "shape": shape,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the gray counter to set a green ray length, then build concentric nested squares at the ray endpoint from the bottom color stripes.

    Input structure:
    - The grid is mostly black.
    - A 2-pixel vertical green start bar sits on the left edge; its lower pixel emits a horizontal ray.
    - The top of the rightmost column contains a contiguous gray counter, and the bottom contains 2-4 full-width color stripes of height 1-2.

    Rule:
    - Count the gray pixels in the rightmost column to obtain K, then erase the gray counter.
    - Starting from the lower green bar pixel, extend a green horizontal ray K steps to the right, keeping the start bar green.
    - Read the bottom stripes from bottom to top as (color, height) controls.
    - At the ray endpoint, place a solid square of side 2*h1-1 in the first stripe color, then wrap each higher stripe outward as a square frame of thickness equal to that stripe's height.
    - Erase the original bottom stripe block to black, but preserve the green ray over the constructed squares.

    Key insight:
    - The counter fixes the center position indirectly through the ray length, while the bottom stripe stack independently encodes the nested square colors and layer thicknesses.
    """
    info = _analyze(grid)
    h, w = info["shape_h"], info["shape_w"]
    out = np.full((h, w), BLACK, dtype=int)

    center_r, center_c = info["center"]
    shape = info["shape"]
    radius = info["radius"]
    r0 = center_r - radius
    c0 = center_c - radius
    out[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] = shape

    out[info["start_top"], 0] = GREEN
    out[info["ray_row"], : info["k"] + 1] = GREEN
    return out


def _rand_stripes() -> list[tuple[int, int]] | None:
    n = random.randint(2, 4)
    colors = random.sample(PALETTE, n)

    for _ in range(40):
        heights = [random.randint(1, 2) for _ in range(n)]
        if sum(heights) <= 7:
            return list(zip(colors, heights))
    return None


def _gen_attempt() -> np.ndarray | None:
    stripes = _rand_stripes()
    if stripes is None:
        return None

    radius = _shape_radius(stripes)
    stripe_rows = sum(height for _color, height in stripes)
    min_k = max(3, radius)
    if min_k > 7:
        return None
    k = random.randint(min_k, 7)

    min_w = max(18, k + radius + 4)
    max_w = min(30, max(min_w, 26))
    if min_w > max_w:
        return None
    w = random.randint(min_w, max_w)

    min_h = max(18, 2 * radius + stripe_rows + 4)
    max_h = min(30, max(min_h, 28))
    if min_h > max_h:
        return None
    h = random.randint(min_h, max_h)

    stripe_top = h - stripe_rows
    ray_row_min = max(1, radius)
    ray_row_max = stripe_top - radius - 1
    if ray_row_min > ray_row_max:
        return None
    ray_row = random.randint(ray_row_min, ray_row_max)
    start_top = ray_row - 1

    grid = np.full((h, w), BLACK, dtype=int)
    grid[start_top, 0] = GREEN
    grid[ray_row, 0] = GREEN
    grid[:k, -1] = GREY

    r = h - 1
    for color, height in stripes:
        grid[r - height + 1 : r + 1, :] = color
        r -= height

    if not _semantic_check(grid):
        return None
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False

    center_r, center_c = info["center"]
    radius = info["radius"]
    stripe_top = info["stripe_top"]
    shape = info["shape"]
    r0 = center_r - radius
    c0 = center_c - radius

    expected = np.full_like(grid, BLACK)
    expected[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] = shape
    expected[info["start_top"], 0] = GREEN
    expected[info["ray_row"], : info["k"] + 1] = GREEN
    if not np.array_equal(out, expected):
        return False

    if np.any(out[stripe_top:, :] != BLACK):
        return False
    if int(np.count_nonzero(out == GREY)) != 0:
        return False
    if int(np.count_nonzero(grid == GREY)) != info["k"]:
        return False
    if out[center_r, center_c] != GREEN:
        return False
    if shape.shape[0] < 3:
        return False
    if info["k"] == 0 or len(info["stripes"]) < 2:
        return False
    if not any(height == 2 for _color, height in info["stripes"]) and not any(
        height == 1 for _color, height in info["stripes"]
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")

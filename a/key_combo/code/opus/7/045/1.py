from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
STRIPE_WIDTH = 3
SCATTER_COL_START = 4

GEN_COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _parse_stripes(grid: np.ndarray) -> list[tuple[int, int]]:
    stripes: list[tuple[int, int]] = []
    rows, _cols = grid.shape
    r = 0
    while r < rows:
        row = grid[r, :STRIPE_WIDTH]
        color = int(row[0])
        if color == BLACK or not bool(np.all(row == color)):
            break
        start = r
        while r < rows and bool(np.all(grid[r, :STRIPE_WIDTH] == color)):
            r += 1
        stripes.append((color, r - start))
    return stripes


def _find_unique_scatter_pixel(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    scatter = grid[:, SCATTER_COL_START:]
    colors = [int(color) for color in np.unique(scatter) if int(color) != BLACK]
    counts = {color: int(np.count_nonzero(scatter == color)) for color in colors}
    unique_colors = [color for color, count in counts.items() if count == 1]
    assert len(unique_colors) == 1
    color = unique_colors[0]
    coords = np.argwhere(scatter == color)
    assert coords.shape[0] == 1
    r, c = map(int, coords[0])
    return color, (r, c + SCATTER_COL_START)


def _render_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    center_color: int,
    thickness: int,
) -> np.ndarray:
    rows, cols = map(int, shape)
    r, c = map(int, center)
    out = np.zeros((rows, cols), dtype=int)
    r0 = max(0, r - thickness)
    r1 = min(rows - 1, r + thickness)
    c0 = max(0, c - thickness)
    c1 = min(cols - 1, c + thickness)
    out[r0 : r1 + 1, c0 : c1 + 1] = RED
    out[r, c] = int(center_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the uniquely frequent scatter color and use its stripe height as the thickness of a red square around it.

    Input structure:
    - The grid has a black background.
    - Columns 0-2 contain 3-4 contiguous horizontal color stripes stacked from the top, each stripe using one distinct non-black color and having a color-specific height.
    - Columns 4 and to the right contain scattered single pixels drawn from the stripe colors; exactly one scatter color appears once, while every other scatter color appears at least twice.

    Rule:
    - Count non-black scatter pixels in columns 4+ and find the unique color C whose frequency is exactly 1; its only scatter pixel is the target location P.
    - Read the stripe height H of color C from the left stripe stack.
    - Produce an all-black output grid of the same size.
    - Fill the square centered at P with radius H in red, clipping to the grid if necessary, then restore the center pixel P to color C.

    Key insight:
    - The right half tells you which color/location matters by frequency uniqueness, while the left stripe stack independently encodes how large the output mark must be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] >= SCATTER_COL_START + 1

    stripes = _parse_stripes(g)
    assert 3 <= len(stripes) <= 4
    stripe_heights = {color: height for color, height in stripes}
    assert len(stripe_heights) == len(stripes)

    color, center = _find_unique_scatter_pixel(g)
    assert color in stripe_heights
    return _render_output(g.shape, center, color, stripe_heights[color])


def _well_separated(cell: tuple[int, int], used: list[tuple[int, int]], gap: int) -> bool:
    r, c = cell
    return all(max(abs(r - ur), abs(c - uc)) > gap for ur, uc in used)


def _pick_cells(
    candidates: list[tuple[int, int]],
    count: int,
    used: list[tuple[int, int]],
    gap: int,
) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    blocked = used[:]
    for cell in shuffled:
        if _well_separated(cell, blocked, gap):
            chosen.append(cell)
            blocked.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    stripe_count = random.randint(3, 4)
    stripe_colors = random.sample(GEN_COLOR_POOL, stripe_count)
    stripe_heights = [random.randint(1, 4) for _ in range(stripe_count)]

    total_stripe_rows = sum(stripe_heights)
    max_thickness = max(stripe_heights)
    min_rows = max(12, total_stripe_rows + 1, 2 * max_thickness + 1)
    min_cols = max(12, SCATTER_COL_START + 2 * max_thickness + 1)
    rows = random.randint(min_rows, max(min_rows, 20))
    cols = random.randint(min_cols, max(min_cols, 20))

    grid = np.zeros((rows, cols), dtype=int)

    stripe_height_by_color: dict[int, int] = {}
    r = 0
    for color, height in zip(stripe_colors, stripe_heights):
        grid[r : r + height, :STRIPE_WIDTH] = color
        stripe_height_by_color[int(color)] = int(height)
        r += height

    unique_color = random.choice(stripe_colors)
    unique_thickness = stripe_height_by_color[int(unique_color)]

    target_candidates = [
        (rr, cc)
        for rr in range(unique_thickness, rows - unique_thickness)
        for cc in range(SCATTER_COL_START + unique_thickness, cols - unique_thickness)
    ]
    if not target_candidates:
        return None

    target = random.choice(target_candidates)
    used = [target]
    grid[target] = int(unique_color)

    other_colors = [color for color in stripe_colors if color != unique_color]
    random.shuffle(other_colors)
    all_scatter_candidates = [
        (rr, cc)
        for rr in range(rows)
        for cc in range(SCATTER_COL_START, cols)
    ]
    for color in other_colors:
        copies = random.choice([2, 2, 3, 3, 4, 5])
        cells = _pick_cells(all_scatter_candidates, copies, used, gap=1)
        if cells is None:
            return None
        for cell in cells:
            grid[cell] = int(color)
        used.extend(cells)

    expected = _render_output(grid.shape, target, int(unique_color), unique_thickness)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")

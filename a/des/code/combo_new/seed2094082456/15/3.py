import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

STRIPE_COUNT_RANGE = (2, 4)
GRID_WIDTH_RANGE = (10, 24)
REGION_HEIGHT_MIN = 2
REGION_HEIGHT_CAP = 6

DOTS_PER_COLOR_CAP = 5
REFERENCE_INCLUDE_PROB = 0.5

GEN_ATTEMPTS = 300


def _find_stripes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    stripe_rows: list[int] = []
    stripe_colors: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color == BG:
            continue
        if np.all(row == color):
            stripe_rows.append(int(r))
            stripe_colors.append(color)

    assert stripe_rows, "expected at least one full-width stripe"
    assert len(set(stripe_colors)) == len(stripe_colors), "stripe colors must be unique"
    assert all(
        stripe_rows[i + 1] - stripe_rows[i] >= 2 for i in range(len(stripe_rows) - 1)
    ), "stripes must be one row tall and separated by regions"
    return stripe_rows, stripe_colors


def _region_bounds(height: int, stripe_rows: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for stripe_r in stripe_rows:
        bounds.append((start, stripe_r))
        start = stripe_r + 1
    bounds.append((start, height))
    return bounds


def _reference_color(region_index: int, stripe_colors: list[int]) -> int:
    if region_index < len(stripe_colors):
        return int(stripe_colors[region_index])
    return int(stripe_colors[-1])


def _priority_order(region_index: int, stripe_colors: list[int]) -> list[int]:
    reference = _reference_color(region_index, stripe_colors)
    return [reference] + [color for color in reversed(stripe_colors) if color != reference]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restack each region's colored dots against the bottom according to the neighboring stripe order.

    Input structure:
    - The background is black.
    - One-pixel-tall horizontal stripes span the full width of the grid, and each stripe has a distinct color.
    - The stripes divide the canvas into horizontal regions containing scattered single-pixel dots whose colors all come from the stripe palette.

    Rule:
    - Keep every stripe unchanged and treat the rows between successive stripes, plus the top and bottom edge regions, as separate areas.
    - For each area, clear its dots, choose the reference stripe immediately below it, and in the bottom edge region use its sole adjacent stripe above as the reference.
    - Repaint the area's dots in bottom-up layers: the reference stripe color occupies the bottom layer, and the remaining colors follow the global stripe order from bottom to top; dots of the same color share one row and keep their original columns.

    Key insight:
    - The stripes are both separators and an ordering legend: a region's nearest lower stripe sets the first layer, while the full vertical stripe order determines the rest of the stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    stripe_rows, stripe_colors = _find_stripes(g)
    bounds = _region_bounds(g.shape[0], stripe_rows)
    palette = set(stripe_colors)

    out = g.copy()
    for region_index, (r0, r1) in enumerate(bounds):
        if r0 >= r1:
            continue

        region = g[r0:r1, :]
        dot_colors = {int(v) for v in np.unique(region) if int(v) != BG}
        assert dot_colors <= palette, "region dots must use stripe colors only"
        assert len(dot_colors) <= (r1 - r0), "region must be tall enough for its color layers"

        out[r0:r1, :] = BG
        dst_row = r1 - 1
        for color in _priority_order(region_index, stripe_colors):
            if color not in dot_colors:
                continue
            cols = np.flatnonzero(np.any(region == color, axis=0))
            if cols.size:
                out[dst_row, cols] = int(color)
                dst_row -= 1

    return out


def _choose_region_colors(
    stripe_colors: list[int], reference: int, max_layers: int
) -> list[int]:
    layer_count = random.randint(1, max_layers)
    others = [color for color in stripe_colors if color != reference]

    include_reference = True
    if layer_count <= len(others) and others:
        include_reference = random.random() < REFERENCE_INCLUDE_PROB

    chosen: list[int] = []
    if include_reference:
        chosen.append(reference)
        need = layer_count - 1
    else:
        need = layer_count

    if need:
        chosen.extend(random.sample(others, need))
    random.shuffle(chosen)
    return chosen


def _place_region_dots(
    grid: np.ndarray,
    r0: int,
    r1: int,
    colors: list[int],
) -> bool:
    height, width = grid.shape
    assert 0 <= r0 < r1 <= height

    max_dots = min(DOTS_PER_COLOR_CAP, max(1, width // 4))
    for color in colors:
        dot_count = random.randint(1, max_dots)
        cols = random.sample(range(width), dot_count)
        for col in cols:
            rows = list(range(r0, r1))
            random.shuffle(rows)
            placed = False
            for row in rows:
                if int(grid[row, col]) == BG:
                    grid[row, col] = int(color)
                    placed = True
                    break
            if not placed:
                return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    stripe_rows, stripe_colors = _find_stripes(grid)
    bounds = _region_bounds(grid.shape[0], stripe_rows)
    if len(bounds) != len(stripe_colors) + 1:
        return False

    palette = set(stripe_colors)
    for r0, r1 in bounds:
        if r1 - r0 < REGION_HEIGHT_MIN:
            return False
        region = grid[r0:r1, :]
        colors = [int(v) for v in np.unique(region) if int(v) != BG]
        if not colors or len(colors) > (r1 - r0):
            return False
        for color in colors:
            cols = np.flatnonzero(np.any(region == color, axis=0))
            if cols.size != int(np.sum(region == color)):
                return False
        if not set(colors) <= palette:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    max_region_height = min(
        REGION_HEIGHT_CAP, (30 - stripe_count) // (stripe_count + 1)
    )
    if max_region_height < REGION_HEIGHT_MIN:
        return None

    region_heights = [
        random.randint(REGION_HEIGHT_MIN, max_region_height)
        for _ in range(stripe_count + 1)
    ]
    width = random.randint(*GRID_WIDTH_RANGE)
    height = sum(region_heights) + stripe_count
    if height > 30:
        return None

    stripe_colors = random.sample(COLOR_POOL, stripe_count)
    grid = np.full((height, width), BG, dtype=int)

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for idx, region_height in enumerate(region_heights):
        bounds.append((cursor, cursor + region_height))
        cursor += region_height
        if idx < stripe_count:
            grid[cursor, :] = int(stripe_colors[idx])
            cursor += 1

    for region_index, (r0, r1) in enumerate(bounds):
        max_layers = min(r1 - r0, stripe_count)
        reference = _reference_color(region_index, stripe_colors)
        colors = _choose_region_colors(stripe_colors, reference, max_layers)
        if not _place_region_dots(grid, r0, r1, colors):
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

COUNT_OPTIONS = ((2, 1), (2, 2), (3, 1), (3, 2))
MIN_GRID = 9
MAX_GRID = 24
MAX_SEGMENT = 12
GEN_RETRY_LIMIT = 300


def neighbors4(r: int, c: int):
    """4-neighborhood using the shared ARC (row, col) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components from a boolean mask as lists of (r, c) coords."""
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    out: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _top_seed_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(1, grid.shape[1] - 1) if int(grid[0, c]) == RED]


def _left_seed_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(1, grid.shape[0] - 1) if int(grid[r, 0]) == RED]


def _draw_rays(grid: np.ndarray, seed_rows: list[int], seed_cols: list[int]) -> None:
    for c in seed_cols:
        grid[1:-1, c] = RED
    for r in seed_rows:
        grid[r, 1:-1] = RED


def _inner_spans(limit: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for divider in sorted(dividers):
        if start <= divider - 1:
            spans.append((start, divider - 1))
        start = divider + 1
    if start <= limit - 2:
        spans.append((start, limit - 2))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend rays inward from top and left frame seeds, then recolor the smallest and largest enclosed rectangles.

    Input structure:
    - The grid has a one-pixel grey rectangular frame on the outer border and a black interior.
    - Some top-frame cells and some left-frame cells are replaced by red seed pixels.
    - No other foreground objects appear inside the frame before the rays are extended.

    Rule:
    - For each red seed on the top border, draw a red vertical ray straight down through the interior until the cell before the bottom frame.
    - For each red seed on the left border, draw a red horizontal ray straight right through the interior until the cell before the right frame.
    - The remaining 4-connected black cells inside the frame are the regions created by those dividers.
    - Recolor the smallest black region teal and the largest black region blue, leaving the grey frame and all red seeds/rays unchanged.

    Key insight:
    - The seeds only encode divider positions; the areas to compare are the black rectangles created after both vertical and horizontal rays are projected inward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seed_cols = _top_seed_cols(g)
    seed_rows = _left_seed_rows(g)
    _draw_rays(out, seed_rows, seed_cols)

    interior_black = out[1:-1, 1:-1] == BLACK
    regions_local = _components_mask(interior_black)
    if not regions_local:
        return out

    regions = []
    for comp in regions_local:
        global_comp = [(r + 1, c + 1) for r, c in comp]
        top_left = min(global_comp)
        regions.append((len(global_comp), top_left, global_comp))
    regions.sort(key=lambda item: (item[0], item[1]))

    for r, c in regions[0][2]:
        out[r, c] = TEAL
    for r, c in regions[-1][2]:
        out[r, c] = BLUE
    return out


def _products_unique(widths: tuple[int, ...], heights: tuple[int, ...]) -> bool:
    products = [w * h for w in widths for h in heights]
    return len(products) == len(set(products))


@lru_cache(None)
def _admissible_layouts(top_count: int, left_count: int) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    width_count = top_count + 1
    height_count = left_count + 1

    width_sets = []
    for widths in combinations(range(1, MAX_SEGMENT + 1), width_count):
        grid_w = sum(widths) + top_count + 2
        if MIN_GRID <= grid_w <= MAX_GRID:
            width_sets.append(widths)

    height_sets = []
    for heights in combinations(range(1, MAX_SEGMENT + 1), height_count):
        grid_h = sum(heights) + left_count + 2
        if MIN_GRID <= grid_h <= MAX_GRID:
            height_sets.append(heights)

    pairs = []
    for widths in width_sets:
        for heights in height_sets:
            if _products_unique(widths, heights):
                pairs.append((widths, heights))
    return tuple(pairs)


def _seed_positions_from_segments(segments: list[int]) -> list[int]:
    seeds: list[int] = []
    cursor = 1
    for size in segments[:-1]:
        cursor += size
        seeds.append(cursor)
        cursor += 1
    return seeds


def _render_input(widths: list[int], heights: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    top_count = len(widths) - 1
    left_count = len(heights) - 1
    h = sum(heights) + left_count + 2
    w = sum(widths) + top_count + 2

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    seed_cols = _seed_positions_from_segments(widths)
    seed_rows = _seed_positions_from_segments(heights)
    grid[0, seed_cols] = RED
    grid[seed_rows, 0] = RED
    return grid, seed_rows, seed_cols


def _render_expected(inp: np.ndarray, seed_rows: list[int], seed_cols: list[int]) -> np.ndarray:
    out = inp.copy()
    _draw_rays(out, seed_rows, seed_cols)

    row_spans = _inner_spans(out.shape[0], seed_rows)
    col_spans = _inner_spans(out.shape[1], seed_cols)
    regions = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            regions.append((area, (r0, c0), (r0, r1, c0, c1)))
    regions.sort(key=lambda item: (item[0], item[1]))

    min_region = regions[0][2]
    max_region = regions[-1][2]
    r0, r1, c0, c1 = min_region
    out[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    r0, r1, c0, c1 = max_region
    out[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    top_count, left_count = random.choice(COUNT_OPTIONS)
    layouts = _admissible_layouts(top_count, left_count)
    if not layouts:
        return None

    widths, heights = random.choice(layouts)
    widths = list(widths)
    heights = list(heights)
    random.shuffle(widths)
    random.shuffle(heights)

    inp, seed_rows, seed_cols = _render_input(widths, heights)
    expected = _render_expected(inp, seed_rows, seed_cols)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid seed-ray layout")

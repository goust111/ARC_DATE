from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, get_bg, paste
import random
from itertools import product


def _unique_largest_non_bg_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.flatten()))
    non_bg_counts = [(color, count) for color, count in counts.items() if color != int(bg)]
    assert non_bg_counts

    largest_count = max(count for _color, count in non_bg_counts)
    candidates = [color for color, count in non_bg_counts if count == largest_count]
    assert len(candidates) == 1
    return int(candidates[0])


def _mask_bbox(g: np.ndarray, mask_color: int) -> tuple[int, int, int, int]:
    mask = g == int(mask_color)
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1

    b = bbox(mask)
    assert b is not None
    return b


def _palette(g: np.ndarray, bg: int, mask_color: int, width: int) -> tuple[np.ndarray, int]:
    palette_mask = (g != int(bg)) & (g != int(mask_color))
    b = bbox(palette_mask)
    assert b is not None

    palette = crop_bbox(g, b)
    assert palette.shape == (3, int(width))
    assert bool(np.all(palette != int(bg)))
    assert bool(np.all(palette != int(mask_color)))
    assert bool(np.array_equal(palette[0], palette[1]))
    assert len(set(map(int, palette[2]))) == 1

    column_colors = palette[0].copy()
    hole_color = int(palette[2, 0])
    return column_colors, hole_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the large mask crop column-by-column using the separate three-row palette strip.

    Input structure:
    - The background is the most common color.
    - One large monochrome foreground object is the mask; its bbox is the output region and may contain background holes.
    - The remaining foreground is a 3-row palette rectangle with the same width as the mask bbox: two identical color rows and one uniform row.

    Rule:
    - Crop to the mask object's bbox.
    - For each output column, color mask pixels with the corresponding color from the repeated palette rows.
    - Replace background holes inside the crop with the palette's uniform third-row color.

    Key insight:
    The grey object is only a stencil; its columns select colors from the legend, while its holes expose the legend's fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mask_color = _unique_largest_non_bg_color(g, bg)

    b = _mask_bbox(g, mask_color)
    mask_crop = crop_bbox(g == int(mask_color), b)
    height, width = map(int, mask_crop.shape)

    column_colors, hole_color = _palette(g, bg, mask_color, width)
    out = np.tile(column_colors, (height, 1))
    out[~mask_crop] = hole_color
    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and object bounds.
BACKGROUND_COLOR, MASK_COLOR = 0, 5
COLOR_MIN, COLOR_MAX = 1, 9
GRID_SIDE_MIN, GRID_SIDE_MAX = 10, 18
MASK_SIDE_MIN, MASK_SIDE_MAX = 5, 10
PALETTE_HEIGHT = 3
GAP_MIN, GAP_MAX = 1, 3

# Palette run construction.
RUN_COUNT_OPTIONS = (4, 5)
RUN_LENGTH_MIN, RUN_LENGTH_MAX = 1, 3
COLUMN_COLOR_DISTINCT_MIN, COLUMN_COLOR_DISTINCT_MAX = 3, 5
UNUSED_COLOR_BIAS = 0.7

# Mask density and retry budget.
MASK_DENSITY_MIN, MASK_DENSITY_MAX = 0.60, 0.71
GENERATION_ATTEMPTS = 300


def _valid_run_lengths(width: int, n_runs: int) -> list[tuple[int, ...]]:
    length_range = range(RUN_LENGTH_MIN, RUN_LENGTH_MAX + 1)
    return [
        tuple(int(x) for x in lengths)
        for lengths in product(length_range, repeat=int(n_runs))
        if sum(lengths) == int(width)
    ]


def _run_colors(colors: list[int], n_runs: int) -> list[int]:
    seq: list[int] = []
    unused = set(colors)
    for i in range(int(n_runs)):
        slots_left = int(n_runs) - i
        choices = [c for c in colors if not seq or c != seq[-1]]
        if len(unused) >= slots_left:
            choices = [c for c in choices if c in unused]
        elif unused and random.random() < UNUSED_COLOR_BIAS:
            preferred = [c for c in choices if c in unused]
            if preferred:
                choices = preferred
        color = random.choice(choices)
        seq.append(int(color))
        unused.discard(int(color))
    return seq


def _palette_row(width: int, colors: list[int]) -> np.ndarray:
    valid_counts = [
        n_runs for n_runs in RUN_COUNT_OPTIONS if _valid_run_lengths(int(width), n_runs)
    ]
    n_runs = random.choice(valid_counts)
    valid_lengths = _valid_run_lengths(int(width), n_runs)
    lengths = random.choice(valid_lengths)
    n_distinct = random.randint(
        COLUMN_COLOR_DISTINCT_MIN, min(COLUMN_COLOR_DISTINCT_MAX, n_runs)
    )
    chosen = random.sample(colors, n_distinct)
    seq = _run_colors(chosen, n_runs)
    return np.repeat(seq, lengths).astype(int)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    """Grow a connected mask with full row/column span to the target area."""
    mask = np.zeros((int(height), int(width)), dtype=bool)
    steps = ["D"] * (int(height) - 1) + ["R"] * (int(width) - 1)
    random.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    if int(mask.sum()) > int(target_area):
        return None

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for nr, nc in _neighbors4(rr, cc, int(height), int(width)):
            if not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < int(target_area):
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)

    return mask


def _grid_choices(mask_h: int, mask_w: int, gap: int) -> list[tuple[int, int]]:
    """Grid sizes where the mask remains the unique largest non-bg object."""
    max_mask = int(np.floor(MASK_DENSITY_MAX * int(mask_h) * int(mask_w)))
    min_h = max(GRID_SIDE_MIN, int(mask_h) + PALETTE_HEIGHT + int(gap))
    min_w = max(GRID_SIDE_MIN, int(mask_w))
    choices: list[tuple[int, int]] = []
    for h in range(min_h, GRID_SIDE_MAX + 1):
        for w in range(min_w, GRID_SIDE_MAX + 1):
            if h * w - max_mask - PALETTE_HEIGHT * int(mask_w) > max_mask:
                choices.append((h, w))
    return choices


def _sample_dimensions() -> tuple[int, int, int, int, int] | None:
    """Sample mask size, layout gap, and compatible grid size."""
    mask_h = random.randint(MASK_SIDE_MIN, MASK_SIDE_MAX)
    mask_w = random.randint(MASK_SIDE_MIN, MASK_SIDE_MAX)
    gap = random.randint(GAP_MIN, GAP_MAX)
    grid_choices = _grid_choices(mask_h, mask_w, gap)
    if not grid_choices:
        return None
    grid_h, grid_w = random.choice(grid_choices)
    return mask_h, mask_w, gap, grid_h, grid_w


def _sample_palette(mask_w: int) -> tuple[np.ndarray, int]:
    """Choose hole color and the same-width column palette row."""
    available = [c for c in range(COLOR_MIN, COLOR_MAX + 1) if c != MASK_COLOR]
    hole_color = random.choice(available)
    column_pool = [c for c in available if c != hole_color]
    column_colors = _palette_row(mask_w, column_pool)
    return column_colors, int(hole_color)


def _compose_grid(grid_h, grid_w, gap, mask, column_colors, hole_color) -> np.ndarray:
    """Paste mask and palette into disjoint row bands of a blank grid."""
    mask_h, mask_w = map(int, mask.shape)
    hole_row = np.full(int(mask_w), int(hole_color), dtype=int)
    palette = np.vstack([column_colors, column_colors, hole_row])
    mask_sprite = np.where(mask, MASK_COLOR, BACKGROUND_COLOR)
    grid = np.zeros((int(grid_h), int(grid_w)), dtype=int)
    block_h = int(mask_h) + PALETTE_HEIGHT + int(gap)
    top = random.randint(0, int(grid_h) - block_h)
    palette_above = random.choice([True, False])
    if palette_above:
        palette_r = top
        mask_r = top + PALETTE_HEIGHT + int(gap)
    else:
        mask_r = top
        palette_r = top + int(mask_h) + int(gap)
    mask_c = random.randint(0, int(grid_w) - int(mask_w))
    palette_c = random.randint(0, int(grid_w) - int(mask_w))
    paste(grid, mask_sprite, mask_r, mask_c, transparent=BACKGROUND_COLOR)
    paste(grid, palette, palette_r, palette_c, transparent=None)
    return grid


def _gen_attempt() -> np.ndarray | None:
    dims = _sample_dimensions()
    if dims is None:
        return None
    mask_h, mask_w, gap, grid_h, grid_w = dims

    column_colors, hole_color = _sample_palette(mask_w)
    area = int(mask_h) * int(mask_w)
    min_area = int(np.ceil(MASK_DENSITY_MIN * area))
    max_area = int(np.floor(MASK_DENSITY_MAX * area))
    target_area = random.randint(min_area, max_area)
    mask = _connected_mask(mask_h, mask_w, target_area)
    if mask is None:
        return None

    output = np.tile(column_colors, (int(mask_h), 1))
    output[~mask] = int(hole_color)
    mask = output != int(hole_color)
    grid = _compose_grid(grid_h, grid_w, gap, mask, column_colors, hole_color)
    if np.array_equal(main(grid), output):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

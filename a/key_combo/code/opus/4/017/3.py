from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox, crop_bbox, paste
except Exception:
    class Color:
        BLACK = 0
        RED = 2

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, b):
        r0, c0, r1, c1 = b
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent=None):
        h, w = patch.shape
        view = dst[r0 : r0 + h, c0 : c0 + w]
        if transparent is None:
            view[:] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


BG = int(Color.BLACK)
OUTLINE = int(Color.RED)
PALETTE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 300
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (15, 22)
GAP_RANGE = (1, 3)
FILL_ROWS_RANGE = (4, 8)
PEAK_WIDTH_RANGE = (3, 7)


def _palette_row(grid: np.ndarray) -> np.ndarray:
    palette_mask = (grid != BG) & (grid != OUTLINE)
    b = bbox(palette_mask)
    assert b is not None, "expected a detached palette strip"
    palette = crop_bbox(grid, b)
    assert palette.shape[0] == 1, "palette must be a single row"
    assert bool(np.all((palette != BG) & (palette != OUTLINE)))
    return palette[0].copy()


def _filled_mask_from_half_outline(grid: np.ndarray) -> np.ndarray:
    outline_mask = grid == OUTLINE
    b = bbox(outline_mask)
    assert b is not None, "expected a red half-outline"
    _r0, _c0, _r1, axis_left = b

    filled = np.zeros_like(outline_mask, dtype=bool)
    for r in np.unique(np.argwhere(outline_mask)[:, 0]):
        row_cols = np.flatnonzero(outline_mask[int(r)])
        assert row_cols.size > 0
        inner_left = int(row_cols.max())
        right_boundary = 2 * int(axis_left) + 1 - inner_left
        assert right_boundary < grid.shape[1], "mirrored fill must stay inside the canvas"
        if inner_left + 1 < right_boundary:
            filled[int(r), inner_left + 1 : right_boundary] = True

    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the red half-outline, clear it, then recolor each filled column from the bottom palette strip.

    Input structure:
    - The grid uses black background, a single red half-outline in the upper region, and one detached non-black non-red palette strip on the bottom row.
    - The half-outline is the left side of a horizontally mirrored closed shape, with its global right edge immediately left of the vertical half-column symmetry axis.
    - The palette strip width matches the bbox width of the filled symmetric interior.

    Rule:
    - Find the red half-outline and use its global right edge to define the mirror axis between columns.
    - For each row containing red, take the red pixel nearest that axis, mirror it to obtain the hidden right boundary, and fill only the strict interior between those two boundaries.
    - Crop to the bbox of the filled region.
    - Recolor each filled pixel in crop column j with palette column j; keep all other crop cells black.

    Key insight:
    - The red object is not the final shape itself: its axis-nearest pixel in each row encodes the interior span, and the detached palette strip supplies the final color for each output column.
    """
    g = np.asarray(grid, dtype=int)
    filled = _filled_mask_from_half_outline(g)
    filled_box = bbox(filled)
    assert filled_box is not None, "expected a non-empty filled region"

    filled_crop = crop_bbox(filled, filled_box)
    palette = _palette_row(g)
    assert palette.shape[0] == filled_crop.shape[1], "palette width must match the filled crop width"

    out = np.full(filled_crop.shape, BG, dtype=int)
    column_canvas = np.tile(palette, (filled_crop.shape[0], 1))
    out[filled_crop] = column_canvas[filled_crop]
    return out


def _sample_half_widths(fill_rows: int, peak_width: int) -> list[int]:
    if random.random() < 0.2:
        return [peak_width] * fill_rows

    half_len = (fill_rows + 1) // 2
    left = sorted(random.choices(range(1, peak_width + 1), k=half_len))
    left[-1] = peak_width
    if all(width == peak_width for width in left):
        left[0] = max(1, peak_width - 1)
        left.sort()
        left[-1] = peak_width

    if fill_rows % 2 == 1:
        return left + left[-2::-1]
    return left + left[::-1]


def _minimal_outers(inners: list[int]) -> list[int]:
    outers = list(inners)
    for _ in range(2):
        for i in range(len(inners) - 1):
            shared = min(int(inners[i]), int(inners[i + 1]))
            outers[i] = min(int(outers[i]), shared)
            outers[i + 1] = min(int(outers[i + 1]), shared)
        for i in range(len(inners) - 2, -1, -1):
            shared = min(int(inners[i]), int(inners[i + 1]))
            outers[i] = min(int(outers[i]), shared)
            outers[i + 1] = min(int(outers[i + 1]), shared)
    return outers


def _sample_palette(width: int) -> np.ndarray:
    distinct = random.randint(3, min(5, width))
    colors = random.sample(PALETTE_COLORS, distinct)
    row: list[int] = []
    while len(row) < width:
        choices = colors if not row else [color for color in colors if color != row[-1]]
        color = random.choice(choices)
        run = random.randint(1, min(3, width - len(row)))
        row.extend([int(color)] * run)

    if len(set(row)) < 3:
        row[0] = colors[0]
        row[width // 2] = colors[1]
        row[-1] = colors[2]
    return np.array(row, dtype=int)


def _expected_output(half_widths: list[int], palette: np.ndarray) -> np.ndarray:
    fill_rows = len(half_widths)
    peak_width = max(half_widths)
    out = np.full((fill_rows, 2 * peak_width), BG, dtype=int)
    for r, half_width in enumerate(half_widths):
        left = peak_width - int(half_width)
        right = peak_width + int(half_width)
        out[r, left:right] = palette[left:right]
    return out


def _build_input(
    half_widths: list[int],
    palette: np.ndarray,
    grid_h: int,
    grid_w: int,
    contour_top: int,
    fill_left: int,
    palette_left: int,
) -> np.ndarray:
    peak_width = max(half_widths)
    axis_left = int(fill_left) + int(peak_width) - 1
    inners = [axis_left] + [axis_left - int(width) for width in half_widths] + [axis_left]
    outers = _minimal_outers(inners)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for dr, (outer_left, inner_left) in enumerate(zip(outers, inners)):
        row = int(contour_top) + int(dr)
        grid[row, int(outer_left) : int(inner_left) + 1] = OUTLINE

    paste(grid, palette[np.newaxis, :], grid_h - 1, palette_left)
    return grid


def _gen_attempt() -> np.ndarray | None:
    fill_rows = random.randint(*FILL_ROWS_RANGE)
    peak_width = random.randint(*PEAK_WIDTH_RANGE)
    half_widths = _sample_half_widths(fill_rows, peak_width)
    peak_width = max(half_widths)
    output_width = 2 * peak_width
    shape_height = fill_rows + 2
    gap = random.randint(*GAP_RANGE)

    min_h = max(GRID_H_RANGE[0], shape_height + gap + 1)
    if min_h > GRID_H_RANGE[1]:
        return None
    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    max_top = grid_h - shape_height - gap - 1
    if max_top < 0:
        return None
    contour_top = random.randint(0, max_top)

    min_w = max(GRID_W_RANGE[0], output_width + 3)
    if min_w > GRID_W_RANGE[1]:
        return None
    grid_w = random.randint(min_w, GRID_W_RANGE[1])

    if grid_w - output_width - 1 < 1:
        return None
    fill_left = random.randint(1, grid_w - output_width - 1)
    palette_left = random.randint(0, grid_w - output_width)

    palette = _sample_palette(output_width)
    expected = _expected_output(half_widths, palette)
    inp = _build_input(
        half_widths=half_widths,
        palette=palette,
        grid_h=grid_h,
        grid_w=grid_w,
        contour_top=contour_top,
        fill_left=fill_left,
        palette_left=palette_left,
    )

    # Structural checks: one 4-connected outline component, a valid bottom strip,
    # and an oracle output that matches the semantic design.
    outline_rows = np.unique(np.argwhere(inp == OUTLINE)[:, 0])
    if outline_rows.size != shape_height:
        return None
    if not np.all(np.diff(outline_rows) == 1):
        return None
    palette_row = inp[-1, palette_left : palette_left + output_width]
    if not np.array_equal(palette_row, palette):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key_combo puzzle input")

from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _regions_between(size: int, separator_lines: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for line in sorted(separator_lines):
        assert start < line < size - 1, "separator lines must split nonempty regions"
        regions.append((start, line - 1))
        start = line + 1
    assert start < size, "expected a trailing region after the separators"
    regions.append((start, size - 1))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the populated tile pattern into every cell of a separator grid.

    Input structure
    - The canvas has one background color and a single non-background separator color.
    - Solid separator rows and/or columns divide the canvas into equal-sized rectangular cells.
    - At least one cell contains a non-background pattern, and all populated cells contain the same pattern.

    Rule
    - Find the separator color from the solid full-width rows or full-height columns.
    - Use the separator lines to split the canvas into cells.
    - Copy the unique populated cell pattern into every cell, preserving the separator lines.

    Key insight
    The separators define a coarse tiled board; the visible foreground pattern is the missing tile to repeat everywhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    bg = get_bg(g)

    separator_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows or cols:
            separator_candidates.append((color, rows, cols))

    assert len(separator_candidates) == 1, "expected one solid separator color"
    separator, row_lines, col_lines = separator_candidates[0]

    row_regions = _regions_between(height, row_lines) if row_lines else [(0, height - 1)]
    col_regions = _regions_between(width, col_lines) if col_lines else [(0, width - 1)]
    assert len(row_regions) * len(col_regions) > 1, "expected the separators to define multiple cells"
    assert len({r1 - r0 + 1 for r0, r1 in row_regions}) == 1, "expected equal cell heights"
    assert len({c1 - c0 + 1 for c0, c1 in col_regions}) == 1, "expected equal cell widths"

    populated: list[np.ndarray] = []
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(block == separator), "separator color should only appear on separator lines"
            if np.any(block != bg):
                populated.append(block.copy())

    assert populated, "expected at least one populated cell pattern"
    template = populated[0]
    for block in populated[1:]:
        assert np.array_equal(block, template), "all populated cells must share the same pattern"

    out = g.copy()
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            out[r0 : r1 + 1, c0 : c1 + 1] = template
    return out

# --------------- generator (o5x) ---------------

# Canvas, separator, and layout knobs.
BG_COLOR = 0
SEPARATOR_COLOR = 4
CELL_LAYOUTS = [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
CELL_H_RANGE = (5, 9)
CELL_W_RANGE = (5, 9)

# Pattern palette, density, and retry knobs.
PATTERN_COLOR_POOL = [c for c in range(1, 10) if c != SEPARATOR_COLOR]
PATTERN_COLOR_COUNT_RANGE = (3, 4)
PATTERN_PIXEL_COUNT_RANGE = (8, 10)
GENERATION_RETRIES = 300


def _gen_pattern(cell_h: int, cell_w: int, colors: list[int], n_pixels: int) -> np.ndarray | None:
    pattern = np.zeros((cell_h, cell_w), dtype=int)
    anchor_rows = [r for r in range(cell_h) if not (cell_h % 2 and r == cell_h // 2)]
    anchor_cols = [c for c in range(cell_w) if not (cell_w % 2 and c == cell_w // 2)]
    ar = random.choice(anchor_rows)
    ac = random.choice(anchor_cols)
    blocked = {(ar, cell_w - 1 - ac), (cell_h - 1 - ar, ac)}

    coords = [(ar, ac)]
    row_counts = [0] * cell_h
    col_counts = [0] * cell_w
    row_counts[ar] = 1
    col_counts[ac] = 1

    available = [
        (r, c)
        for r in range(cell_h)
        for c in range(cell_w)
        if (r, c) != (ar, ac) and (r, c) not in blocked
    ]
    random.shuffle(available)
    for r, c in available:
        if len(coords) == n_pixels:
            break
        if row_counts[r] >= cell_w - 1 or col_counts[c] >= cell_h - 1:
            continue
        coords.append((r, c))
        row_counts[r] += 1
        col_counts[c] += 1

    if len(coords) != n_pixels:
        return None

    random.shuffle(coords)
    shuffled_colors = colors[:]
    random.shuffle(shuffled_colors)
    for i, (r, c) in enumerate(coords):
        pattern[r, c] = shuffled_colors[i] if i < len(shuffled_colors) else random.choice(colors)
    return pattern


def _render_output(
    cell_rows: int, cell_cols: int, cell_h: int, cell_w: int, pattern: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height = cell_rows * cell_h + cell_rows - 1
    width = cell_cols * cell_w + cell_cols - 1
    output = np.zeros((height, width), dtype=int)

    for rr in range(1, cell_rows):
        output[rr * cell_h + rr - 1, :] = SEPARATOR_COLOR
    for cc in range(1, cell_cols):
        output[:, cc * cell_w + cc - 1] = SEPARATOR_COLOR

    cell_starts = []
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            r0 = rr * (cell_h + 1)
            c0 = cc * (cell_w + 1)
            output[r0 : r0 + cell_h, c0 : c0 + cell_w] = pattern
            cell_starts.append((r0, c0))
    return output, cell_starts


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols = random.choice(CELL_LAYOUTS)
    cell_h = random.randint(*CELL_H_RANGE)
    cell_w = random.randint(*CELL_W_RANGE)
    n_colors = random.randint(*PATTERN_COLOR_COUNT_RANGE)
    colors = random.sample(PATTERN_COLOR_POOL, n_colors)
    n_pixels = random.randint(*PATTERN_PIXEL_COUNT_RANGE)

    pattern = _gen_pattern(cell_h, cell_w, colors, n_pixels)
    if pattern is None:
        return None

    output, cell_starts = _render_output(cell_rows, cell_cols, cell_h, cell_w, pattern)
    inp = output.copy()
    keep = random.choice(cell_starts)
    for r0, c0 in cell_starts:
        if (r0, c0) != keep:
            inp[r0 : r0 + cell_h, c0 : c0 + cell_w] = BG_COLOR

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c444b776 input")

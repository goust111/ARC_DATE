from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste
import itertools
import random


def _visible_colors(line: np.ndarray, bg: int) -> frozenset[int]:
    return frozenset(int(v) for v in line if int(v) != int(bg))


def _axis_bands(g: np.ndarray, axis: int, bg: int) -> list[tuple[int, int]]:
    signatures: list[tuple[int, frozenset[int]]] = []
    for idx in range(g.shape[axis]):
        line = g[idx, :] if axis == 0 else g[:, idx]
        colors = _visible_colors(line, bg)
        if colors:
            signatures.append((int(idx), colors))

    assert signatures, "expected foreground rows or columns"

    bands: list[tuple[int, int]] = []
    start = prev = signatures[0][0]
    current = signatures[0][1]
    for idx, colors in signatures[1:]:
        assert idx == prev + 1, "foreground rows and columns must be contiguous"
        if colors != current:
            bands.append((start, prev))
            start = int(idx)
            current = colors
        prev = int(idx)
    bands.append((start, prev))
    return bands


def _cell_color(g: np.ndarray, rows: tuple[int, int], cols: tuple[int, int], bg: int) -> int:
    r0, r1 = rows
    c0, c1 = cols
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    colors = np.unique(region[region != int(bg)])
    assert colors.size == 1, "each lattice cell must have one foreground color"
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the color of each noisy block in a row/column lattice and mirror the coarse grid horizontally.

    Input structure:
    - A background surrounds a rectangular lattice of monochrome foreground blocks.
    - Consecutive rows in a lattice band share the same set of visible foreground colors.
    - Consecutive columns in a lattice band share the same set of visible foreground colors.
    - Each row-band/column-band intersection has one foreground color, with possible background holes.

    Rule:
    - Group non-empty rows and columns into consecutive bands by their foreground color sets.
    - Replace each lattice cell by the single foreground color inside that row/column-band intersection.
    - Return the resulting coarse color grid with the column order reversed.

    Key insight:
    - Whole-row and whole-column color sets reveal the hidden lattice even when neighboring cells share a color or contain holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    row_bands = _axis_bands(g, axis=0, bg=bg)
    col_bands = _axis_bands(g, axis=1, bg=bg)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-cell block lattice"

    out = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for r_idx, row_band in enumerate(row_bands):
        for c_idx, col_band in enumerate(reversed(col_bands)):
            out[r_idx, c_idx] = _cell_color(g, row_band, col_band, bg)
    return out

# --------------- generator (att2) ---------------

# Coarse output colors and dimensions.
ROW_BAND_BOUNDS = (2, 4)
COL_BAND_BOUNDS = (2, 3)
COLOR_VALUES = range(1, 10)
MIN_DISTINCT_COLORS = 4
MAX_DISTINCT_COLORS = 8
ADJACENCY_REPAIR_ATTEMPTS = 30

# Expanded lattice dimensions.
LATTICE_HEIGHT_BOUNDS = (8, 16)
LATTICE_WIDTH_BOUNDS = (8, 14)
BAND_HEIGHT_BOUNDS = (3, 8)
BAND_WIDTH_BOUNDS = (3, 6)

# Canvas, holes, and retry limits.
CANVAS_HEIGHT_BOUNDS = (18, 22)
CANVAS_WIDTH_BOUNDS = (11, 19)
MIN_BG_PADDING = 1
BG_COLOR = 0
HOLE_MIN_DENOMINATOR = 20
HOLE_MAX_DENOMINATOR = 5
COLOR_COUNT_BINS = 10
GENERATION_ATTEMPTS = 300


def _line_color_set(a: np.ndarray, axis: int, idx: int) -> frozenset[int]:
    line = a[idx, :] if axis == 0 else a[:, idx]
    return frozenset(map(int, line))


def _adjacent_sets_ok(a: np.ndarray) -> bool:
    row_sets = [_line_color_set(a, axis=0, idx=r) for r in range(a.shape[0])]
    col_sets = [_line_color_set(a, axis=1, idx=c) for c in range(a.shape[1])]
    return all(row_sets[i] != row_sets[i - 1] for i in range(1, len(row_sets))) and all(
        col_sets[i] != col_sets[i - 1] for i in range(1, len(col_sets))
    )


def _nudge_line(a: np.ndarray, axis: int, idx: int, avoid: frozenset[int], palette: list[int]) -> None:
    if axis == 0:
        coords = [(idx, c) for c in range(a.shape[1])]
    else:
        coords = [(r, idx) for r in range(a.shape[0])]

    counts = {color: int(np.count_nonzero(a == color)) for color in palette}
    eligible = [(r, c) for r, c in coords if counts[int(a[r, c])] > 1]
    if not eligible:
        eligible = coords

    outside = [color for color in palette if color not in avoid]
    r, c = random.choice(eligible)
    if outside:
        a[r, c] = random.choice(outside)
        return

    line_colors = [int(a[rr, cc]) for rr, cc in coords if int(a[rr, cc]) != int(a[r, c])]
    if line_colors:
        a[r, c] = random.choice(line_colors)


def _make_output(rows: int, cols: int) -> np.ndarray | None:
    n_cells = rows * cols
    n_colors = random.randint(MIN_DISTINCT_COLORS, min(MAX_DISTINCT_COLORS, n_cells))
    palette = random.sample(COLOR_VALUES, n_colors)

    counts = {color: 1 for color in palette}
    for _ in range(n_cells - n_colors):
        fewest = min(counts.values())
        choices = [color for color, count in counts.items() if count == fewest]
        counts[random.choice(choices)] += 1

    vals: list[int] = []
    for color, count in counts.items():
        vals.extend([color] * count)
    random.shuffle(vals)
    out = np.array(vals, dtype=int).reshape(rows, cols)

    for _ in range(ADJACENCY_REPAIR_ATTEMPTS):
        if _adjacent_sets_ok(out) and len(set(map(int, out.flat))) == n_colors:
            return out
        for r in range(1, rows):
            if _line_color_set(out, axis=0, idx=r) == _line_color_set(out, axis=0, idx=r - 1):
                _nudge_line(out, axis=0, idx=r, avoid=_line_color_set(out, axis=0, idx=r - 1), palette=palette)
        for c in range(1, cols):
            if _line_color_set(out, axis=1, idx=c) == _line_color_set(out, axis=1, idx=c - 1):
                _nudge_line(out, axis=1, idx=c, avoid=_line_color_set(out, axis=1, idx=c - 1), palette=palette)
    return None


def _pick_sizes(n: int, total_lo: int, total_hi: int, part_lo: int, part_hi: int) -> list[int]:
    lo = max(total_lo, n * part_lo)
    hi = min(total_hi, n * part_hi)
    total = random.randint(lo, hi)
    choices = [p for p in itertools.product(range(part_lo, part_hi + 1), repeat=n) if sum(p) == total]
    return list(random.choice(choices))


def _hole_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    n_holes = random.randint((area + HOLE_MIN_DENOMINATOR - 1) // HOLE_MIN_DENOMINATOR, area // HOLE_MAX_DENOMINATOR)
    mask = np.zeros((height, width), dtype=bool)
    row_left = [width] * height
    col_left = [height] * width
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)

    placed = 0
    for r, c in coords:
        if placed == n_holes:
            return mask
        if row_left[r] > 1 and col_left[c] > 1:
            mask[r, c] = True
            row_left[r] -= 1
            col_left[c] -= 1
            placed += 1
    return mask if placed == n_holes else None


def _zero_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=COLOR_COUNT_BINS)
    return int(counts[BG_COLOR]) > int(counts[1:].max()) and get_bg(grid) == BG_COLOR


def _expand_output(out: np.ndarray) -> np.ndarray | None:
    rows, cols = out.shape
    heights = _pick_sizes(rows, *LATTICE_HEIGHT_BOUNDS, *BAND_HEIGHT_BOUNDS)
    widths = _pick_sizes(cols, *LATTICE_WIDTH_BOUNDS, *BAND_WIDTH_BOUNDS)
    lattice_h = sum(heights)
    lattice_w = sum(widths)

    min_grid_h = max(CANVAS_HEIGHT_BOUNDS[0], lattice_h + 2 * MIN_BG_PADDING)
    min_grid_w = max(CANVAS_WIDTH_BOUNDS[0], lattice_w + 2 * MIN_BG_PADDING)
    if min_grid_h > CANVAS_HEIGHT_BOUNDS[1] or min_grid_w > CANVAS_WIDTH_BOUNDS[1]:
        return None

    grid_h = random.randint(min_grid_h, CANVAS_HEIGHT_BOUNDS[1])
    grid_w = random.randint(min_grid_w, CANVAS_WIDTH_BOUNDS[1])
    top = random.randint(MIN_BG_PADDING, grid_h - lattice_h - MIN_BG_PADDING)
    left = random.randint(MIN_BG_PADDING, grid_w - lattice_w - MIN_BG_PADDING)

    lattice = np.full((lattice_h, lattice_w), BG_COLOR, dtype=int)
    input_colors = out[:, ::-1]
    r0 = 0
    for r, height in enumerate(heights):
        c0 = 0
        for c, width in enumerate(widths):
            cell = np.full((height, width), int(input_colors[r, c]), dtype=int)
            holes = _hole_mask(height, width)
            if holes is None:
                return None
            cell[holes] = BG_COLOR
            paste(lattice, cell, r0, c0)
            c0 += width
        r0 += height

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    paste(grid, lattice, top, left)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_BAND_BOUNDS)
    cols = random.randint(*COL_BAND_BOUNDS)
    out = _make_output(rows, cols)
    if out is None:
        return None

    grid = _expand_output(out)
    if grid is None or not _zero_is_dominant(grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 80214e03 input")

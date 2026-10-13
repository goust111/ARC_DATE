from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _visible_colors(line: np.ndarray, bg: int) -> frozenset[int]:
    return frozenset(int(v) for v in np.unique(line) if int(v) != bg)


def _axis_bands(grid: np.ndarray, axis: int, bg: int) -> list[tuple[int, int]]:
    """
    Split rows or columns into consecutive non-empty bands.

    A band ends whenever a full-background separator line appears or the visible
    foreground-color set changes.
    """
    bands: list[tuple[int, int]] = []
    start: int | None = None
    end: int | None = None
    current_sig: frozenset[int] | None = None

    for idx in range(grid.shape[axis]):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        sig = _visible_colors(line, bg)

        if not sig:
            if start is not None and end is not None:
                bands.append((start, end))
            start = end = None
            current_sig = None
            continue

        if start is None:
            start = end = idx
            current_sig = sig
            continue

        assert end is not None
        if sig != current_sig or idx != end + 1:
            bands.append((start, end))
            start = idx
            current_sig = sig
        end = idx

    if start is not None and end is not None:
        bands.append((start, end))

    assert bands, "expected at least one foreground band"
    return bands


def _cell_color(grid: np.ndarray, row_band: tuple[int, int], col_band: tuple[int, int], bg: int) -> int:
    r0, r1 = row_band
    c0, c1 = col_band
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = np.unique(cell[cell != bg])
    assert colors.size == 1, "each panel must contain exactly one foreground color"
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract one color per row-band/column-band panel and reverse the column order.

    Input structure:
    - The dominant color is background.
    - Background separator rows and columns divide the foreground into panels.
    - Each panel contains one foreground color, optionally with background holes.

    Rule:
    - Group non-empty rows and columns into consecutive bands.
    - Read the single foreground color from every band intersection.
    - Output the resulting coarse color grid with reversed column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    row_bands = _axis_bands(g, axis=0, bg=bg)
    col_bands = _axis_bands(g, axis=1, bg=bg)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-panel grid"

    out = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for r_idx, row_band in enumerate(row_bands):
        for c_idx, col_band in enumerate(reversed(col_bands)):
            out[r_idx, c_idx] = _cell_color(g, row_band, col_band, bg)
    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400

ROW_BAND_RANGE = (2, 4)
COL_BAND_RANGE = (2, 4)
BAND_HEIGHT_RANGE = (2, 5)
BAND_WIDTH_RANGE = (2, 5)
SEPARATOR_RANGE = (1, 2)
PADDING_RANGE = (0, 2)
GRID_MAX_SIZE = 30

PALETTE = tuple(range(10))
MIN_DISTINCT_COLORS = 2
MAX_DISTINCT_COLORS = 6


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    other_counts = np.delete(counts, bg)
    return int(counts[bg]) > int(other_counts.max()) and _get_bg(grid) == bg


def _make_output(rows: int, cols: int, colors: list[int]) -> np.ndarray:
    out = np.empty((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            choices = [
                color
                for color in colors
                if (r == 0 or color != int(out[r - 1, c])) and (c == 0 or color != int(out[r, c - 1]))
            ]
            assert choices, "adjacency-constrained color placement must remain possible"

            diagonal = int(out[r - 1, c - 1]) if r > 0 and c > 0 else None
            if diagonal is not None and diagonal in choices and random.random() < 0.4:
                out[r, c] = diagonal
            else:
                out[r, c] = random.choice(choices)

    assert len(set(map(int, out.flat))) >= 2, "output must use multiple colors"
    return out


def _hole_keep_mask(height: int, width: int) -> np.ndarray:
    """
    Return a foreground mask with optional holes while keeping every local row and
    column non-empty, so separator discovery cannot split a band incorrectly.
    """
    keep = np.ones((height, width), dtype=bool)
    if height == 1 or width == 1:
        return keep

    area = height * width
    max_holes = max(0, area // 3)
    target_holes = random.randint(0, max_holes)
    row_left = [width] * height
    col_left = [height] * width

    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    holes = 0
    for r, c in coords:
        if holes >= target_holes:
            break
        if row_left[r] > 1 and col_left[c] > 1:
            keep[r, c] = False
            row_left[r] -= 1
            col_left[c] -= 1
            holes += 1
    return keep


def _sample_sizes(count: int, part_range: tuple[int, int]) -> list[int]:
    return [random.randint(*part_range) for _ in range(count)]


def _sample_separators(count: int) -> list[int]:
    return [random.randint(*SEPARATOR_RANGE) for _ in range(max(0, count - 1))]


def _build_grid(out: np.ndarray, bg: int) -> np.ndarray | None:
    rows, cols = out.shape
    input_colors = out[:, ::-1]

    heights = _sample_sizes(rows, BAND_HEIGHT_RANGE)
    widths = _sample_sizes(cols, BAND_WIDTH_RANGE)
    row_gaps = _sample_separators(rows)
    col_gaps = _sample_separators(cols)
    top = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    right = random.randint(*PADDING_RANGE)

    grid_h = top + bottom + sum(heights) + sum(row_gaps)
    grid_w = left + right + sum(widths) + sum(col_gaps)
    if grid_h > GRID_MAX_SIZE or grid_w > GRID_MAX_SIZE:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    r0 = top
    for r_idx, height in enumerate(heights):
        c0 = left
        for c_idx, width in enumerate(widths):
            color = int(input_colors[r_idx, c_idx])
            keep = _hole_keep_mask(height, width)
            cell = np.full((height, width), bg, dtype=int)
            cell[keep] = color
            grid[r0 : r0 + height, c0 : c0 + width] = cell
            c0 += width
            if c_idx < cols - 1:
                c0 += col_gaps[c_idx]
        r0 += height
        if r_idx < rows - 1:
            r0 += row_gaps[r_idx]
    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*ROW_BAND_RANGE)
    cols = random.randint(*COL_BAND_RANGE)
    bg = random.choice(PALETTE)

    available = [color for color in PALETTE if color != bg]
    distinct = random.randint(MIN_DISTINCT_COLORS, min(MAX_DISTINCT_COLORS, rows * cols, len(available)))
    colors = random.sample(available, distinct)
    out = _make_output(rows, cols, colors)

    grid = _build_grid(out, bg)
    if grid is None or not _background_is_dominant(grid, bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _visible_colors(line: np.ndarray, bg: int) -> frozenset[int]:
    return frozenset(int(v) for v in line if int(v) != bg)


def _axis_bands(grid: np.ndarray, axis: int, bg: int) -> list[tuple[int, int]]:
    signatures: list[tuple[int, frozenset[int]]] = []
    for idx in range(grid.shape[axis]):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        colors = _visible_colors(line, bg)
        if colors:
            signatures.append((int(idx), colors))

    assert signatures, "expected foreground rows or columns"

    bands: list[tuple[int, int]] = []
    start = prev = signatures[0][0]
    current = signatures[0][1]
    for idx, colors in signatures[1:]:
        assert idx == prev + 1, "foreground rectangle must be contiguous along each axis"
        if colors != current:
            bands.append((start, prev))
            start = idx
            current = colors
        prev = idx
    bands.append((start, prev))
    return bands


def _cell_color(grid: np.ndarray, row_band: tuple[int, int], col_band: tuple[int, int], bg: int) -> int:
    r0, r1 = row_band
    c0, c1 = col_band
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = np.unique(cell[cell != bg])
    assert colors.size == 1, "each hidden lattice cell must contain one foreground color"
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the color of each noisy block in a row/column lattice and mirror the coarse grid horizontally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    row_bands = _axis_bands(g, axis=0, bg=bg)
    col_bands = _axis_bands(g, axis=1, bg=bg)
    assert len(row_bands) >= 2 and len(col_bands) >= 2

    out = np.empty((len(row_bands), len(col_bands)), dtype=int)
    for rr, row_band in enumerate(row_bands):
        for cc, col_band in enumerate(reversed(col_bands)):
            out[rr, cc] = _cell_color(g, row_band, col_band, bg)
    return out


COLOR_POOL = tuple(range(1, 10))
BG_COLOR = 0
ROW_COUNT_RANGE = (2, 5)
COL_COUNT_RANGE = (2, 5)
ROW_BAND_RANGE = (2, 6)
COL_BAND_RANGE = (2, 6)
PADDING_RANGE = (1, 4)
MAX_DISTINCT_COLORS = 7
REPAIR_ATTEMPTS = 50
GEN_ATTEMPTS = 500


def _row_set(arr: np.ndarray, idx: int) -> frozenset[int]:
    return frozenset(int(v) for v in arr[idx, :])


def _col_set(arr: np.ndarray, idx: int) -> frozenset[int]:
    return frozenset(int(v) for v in arr[:, idx])


def _adjacent_sets_ok(arr: np.ndarray) -> bool:
    return all(_row_set(arr, r) != _row_set(arr, r - 1) for r in range(1, arr.shape[0])) and all(
        _col_set(arr, c) != _col_set(arr, c - 1) for c in range(1, arr.shape[1])
    )


def _make_coarse_grid(rows: int, cols: int) -> np.ndarray | None:
    n_cells = rows * cols
    min_colors = 2 if n_cells < 3 else 3
    n_colors = random.randint(min_colors, min(MAX_DISTINCT_COLORS, n_cells))
    palette = random.sample(COLOR_POOL, n_colors)

    flat = [random.choice(palette) for _ in range(n_cells)]
    for i, color in enumerate(palette):
        flat[i] = color
    random.shuffle(flat)
    arr = np.array(flat, dtype=int).reshape(rows, cols)

    for _ in range(REPAIR_ATTEMPTS):
        if _adjacent_sets_ok(arr):
            return arr
        for r in range(1, rows):
            if _row_set(arr, r) == _row_set(arr, r - 1):
                c = random.randrange(cols)
                choices = [color for color in palette if color not in _row_set(arr, r - 1)]
                arr[r, c] = random.choice(choices or palette)
        for c in range(1, cols):
            if _col_set(arr, c) == _col_set(arr, c - 1):
                r = random.randrange(rows)
                choices = [color for color in palette if color not in _col_set(arr, c - 1)]
                arr[r, c] = random.choice(choices or palette)
    return None


def _sample_band_sizes(count: int, lo: int, hi: int) -> list[int]:
    return [random.randint(lo, hi) for _ in range(count)]


def _hole_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if height * width <= 3 or random.random() < 0.2:
        return mask

    max_holes = max(0, height * width - max(height, width))
    if max_holes == 0:
        return mask

    target = random.randint(1, max_holes)
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    row_left = [width] * height
    col_left = [height] * width
    placed = 0

    for r, c in coords:
        if placed == target:
            break
        if row_left[r] <= 1 or col_left[c] <= 1:
            continue
        mask[r, c] = True
        row_left[r] -= 1
        col_left[c] -= 1
        placed += 1
    return mask


def _expand_to_input(coarse_out: np.ndarray) -> np.ndarray:
    rows, cols = coarse_out.shape
    row_sizes = _sample_band_sizes(rows, *ROW_BAND_RANGE)
    col_sizes = _sample_band_sizes(cols, *COL_BAND_RANGE)
    lattice_h = sum(row_sizes)
    lattice_w = sum(col_sizes)

    lattice = np.full((lattice_h, lattice_w), BG_COLOR, dtype=int)
    input_coarse = coarse_out[:, ::-1]

    r0 = 0
    for rr, band_h in enumerate(row_sizes):
        c0 = 0
        for cc, band_w in enumerate(col_sizes):
            color = int(input_coarse[rr, cc])
            cell = np.full((band_h, band_w), color, dtype=int)
            cell[_hole_mask(band_h, band_w)] = BG_COLOR
            lattice[r0 : r0 + band_h, c0 : c0 + band_w] = cell
            c0 += band_w
        r0 += band_h

    pad_top = random.randint(*PADDING_RANGE)
    pad_bottom = random.randint(*PADDING_RANGE)
    pad_left = random.randint(*PADDING_RANGE)
    pad_right = random.randint(*PADDING_RANGE)
    grid = np.full((lattice_h + pad_top + pad_bottom, lattice_w + pad_left + pad_right), BG_COLOR, dtype=int)
    grid[pad_top : pad_top + lattice_h, pad_left : pad_left + lattice_w] = lattice
    return grid


def _valid_instance(grid: np.ndarray, coarse_out: np.ndarray) -> bool:
    try:
        pred = main(grid)
    except Exception:
        return False
    if not np.array_equal(pred, coarse_out):
        return False
    if _get_bg(grid) != BG_COLOR:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rows = random.randint(*ROW_COUNT_RANGE)
        cols = random.randint(*COL_COUNT_RANGE)
        coarse_out = _make_coarse_grid(rows, cols)
        if coarse_out is None:
            continue
        grid = _expand_to_input(coarse_out)
        if _valid_instance(grid, coarse_out):
            return grid
    raise RuntimeError("failed to generate a valid input")

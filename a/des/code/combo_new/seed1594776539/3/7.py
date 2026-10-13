import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9

DIRS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
POINTER_COLORS = (1, 2, 3, 4, 6, 7, 8)

TILE_SIZE_CHOICES = (5, 7, 9, 11)
TILE_SIZE_TO_COUNT_RANGE = {
    5: (2, 4),
    7: (2, 3),
    9: (2, 3),
    11: (2, 2),
}
GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _segments_from_dividers(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(dividers) + [length]
    segments: list[tuple[int, int]] = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left > 1:
            segments.append((left + 1, right - 1))
    return segments


def _tile_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = grid.shape
    row_dividers = [r for r in range(h) if np.all(grid[r] == GREY)]
    col_dividers = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert all(b - a > 1 for a, b in zip(row_dividers, row_dividers[1:])), "row dividers must be single-pixel lines"
    assert all(b - a > 1 for a, b in zip(col_dividers, col_dividers[1:])), "col dividers must be single-pixel lines"

    row_segments = _segments_from_dividers(h, row_dividers)
    col_segments = _segments_from_dividers(w, col_dividers)
    assert row_segments and col_segments, "expected at least one subregion"

    row_sizes = {r1 - r0 + 1 for r0, r1 in row_segments}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_segments}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "subregions must be equal-sized"
    tile_size = row_sizes.pop()
    assert tile_size == col_sizes.pop(), "subregions must be square"
    assert tile_size % 2 == 1 and tile_size >= 3, "subregions must have an odd side length"
    return row_segments, col_segments, tile_size


def _read_pointer(tile: np.ndarray) -> tuple[int, int, int, int]:
    size = tile.shape[0]
    center = size // 2
    assert tile.shape == (size, size), "tile must be square"
    assert int(tile[center, center]) == MAROON, "expected a maroon pivot at tile center"

    pointer_mask = (tile != BG) & (tile != GREY) & (tile != MAROON)
    pointer_cells = np.argwhere(pointer_mask)
    assert 1 <= int(pointer_cells.shape[0]) <= 3, "pointer length must be between 1 and 3"

    found: tuple[int, int, int, int] | None = None
    for dr, dc in DIRS:
        rr = center + dr
        cc = center + dc
        if not _in_bounds(tile.shape, rr, cc):
            continue
        color = int(tile[rr, cc])
        if color in (BG, GREY, MAROON):
            continue

        length = 0
        scan_r, scan_c = rr, cc
        while _in_bounds(tile.shape, scan_r, scan_c) and int(tile[scan_r, scan_c]) == color:
            length += 1
            scan_r += dr
            scan_c += dc

        assert found is None, "each tile must contain exactly one pointer"
        found = (dr, dc, color, length)

    assert found is not None, "failed to detect pointer direction"
    dr, dc, color, length = found

    expected_mask = np.zeros_like(pointer_mask)
    for step in range(1, length + 1):
        expected_mask[center + dr * step, center + dc * step] = True
    assert np.array_equal(pointer_mask, expected_mask), "pointer must be one contiguous straight segment"
    return dr, dc, color, length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each tile's central pointer by drawing an equal-length ray in the opposite direction.

    Input structure:
    - Black background fills a rectangular grid split by single-pixel grey divider lines into equal square subregions.
    - Each subregion has a maroon pivot pixel at its exact center.
    - From each pivot, one colored straight pointer extends in one of the 8 compass directions.
    - Pointer colors are neither black, grey, nor maroon, and their lengths are 1 to 3 pixels excluding the pivot.

    Rule:
    - Parse every square subregion independently.
    - Read the pointer's direction and length from the contiguous colored segment starting at the pivot.
    - Starting again from the pivot, draw the same color for the same number of steps in the exact opposite direction.
    - Only paint background pixels; stop early if the next step would hit a divider line or leave the subregion.

    Key insight:
    - The pointer encodes both attributes needed for the output: its orientation determines the reverse ray direction, and its pixel count determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_segments, col_segments, tile_size = _tile_slices(g)
    out = g.copy()
    center_offset = tile_size // 2

    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            dr, dc, color, length = _read_pointer(tile)
            pivot_r = r0 + center_offset
            pivot_c = c0 + center_offset

            for step in range(1, length + 1):
                rr = pivot_r - dr * step
                cc = pivot_c - dc * step
                if not _in_bounds(out.shape, rr, cc):
                    break
                if int(out[rr, cc]) != BG:
                    break
                out[rr, cc] = color

    return out


def _balanced_samples(values: tuple, count: int) -> list:
    pool: list = []
    while len(pool) < count:
        chunk = list(values)
        random.shuffle(chunk)
        pool.extend(chunk)
    result = pool[:count]
    random.shuffle(result)
    return result


def _choose_layout() -> tuple[int, int, int]:
    tile_size = random.choice(TILE_SIZE_CHOICES)
    lo, hi = TILE_SIZE_TO_COUNT_RANGE[tile_size]
    rows = random.randint(lo, hi)
    cols = random.randint(lo, hi)
    return rows, cols, tile_size


def _sample_specs(tile_size: int, tile_count: int) -> list[tuple[int, tuple[int, int], int]]:
    max_len = min(3, tile_size // 2)
    directions = _balanced_samples(DIRS, tile_count)
    lengths = _balanced_samples(tuple(range(1, max_len + 1)), tile_count)
    colors = _balanced_samples(POINTER_COLORS, tile_count)

    specs = [(int(color), (int(dr), int(dc)), int(length)) for color, (dr, dc), length in zip(colors, directions, lengths)]
    random.shuffle(specs)
    return specs


def _render_pair(rows: int, cols: int, tile_size: int, specs: list[tuple[int, tuple[int, int], int]]) -> tuple[np.ndarray, np.ndarray]:
    height = rows * tile_size + (rows - 1)
    width = cols * tile_size + (cols - 1)
    inp = np.full((height, width), BG, dtype=int)

    for r in range(1, rows):
        inp[r * (tile_size + 1) - 1, :] = GREY
    for c in range(1, cols):
        inp[:, c * (tile_size + 1) - 1] = GREY

    out = inp.copy()
    center = tile_size // 2

    for idx, spec in enumerate(specs):
        tile_r = idx // cols
        tile_c = idx % cols
        r0 = tile_r * (tile_size + 1)
        c0 = tile_c * (tile_size + 1)
        pivot_r = r0 + center
        pivot_c = c0 + center
        color, (dr, dc), length = spec

        inp[pivot_r, pivot_c] = MAROON
        out[pivot_r, pivot_c] = MAROON
        for step in range(1, length + 1):
            rr = pivot_r + dr * step
            cc = pivot_c + dc * step
            inp[rr, cc] = color
            out[rr, cc] = color

        for step in range(1, length + 1):
            rr = pivot_r - dr * step
            cc = pivot_c - dc * step
            if int(out[rr, cc]) != BG:
                break
            out[rr, cc] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rows, cols, tile_size = _choose_layout()
    specs = _sample_specs(tile_size, rows * cols)
    inp, expected = _render_pair(rows, cols, tile_size, specs)

    try:
        predicted = main(inp)
    except Exception:
        return None

    if not np.array_equal(predicted, expected):
        return None
    if predicted.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

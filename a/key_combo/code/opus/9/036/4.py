import random

import numpy as np


BACKGROUND = 0
MARKER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _swap_colors(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    mask_a = out == int(color_a)
    mask_b = out == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _odd_variant(motif: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    return _swap_colors(np.fliplr(motif), color_a, color_b)


def _parse_input(grid: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 1 < h <= 30 and 1 < w <= 30, "grid must fit ARC bounds"

    marker_rows = np.flatnonzero(g[:, 0] == MARKER)
    assert 2 <= marker_rows.size <= 4, "marker height K must be in [2, 4]"
    assert np.array_equal(marker_rows, np.arange(marker_rows.size)), "grey marker must be a top-left contiguous run"
    assert np.count_nonzero(g == MARKER) == marker_rows.size, "grey may only appear in the marker column"

    k = int(marker_rows.size)
    assert np.all(g[k:, 0] == BACKGROUND), "marker column below the motif must stay black"
    assert np.all(g[k:, :] == BACKGROUND), "rows below the motif must be a blank suffix"
    assert h - k >= 3 * k, "suffix must hold at least three full motif copies"

    motif = g[:k, 1:].copy()
    assert np.all(np.any(motif != BACKGROUND, axis=1)), "each motif row must contain visible pattern pixels"

    colors = sorted(int(c) for c in np.unique(motif) if int(c) != BACKGROUND)
    assert len(colors) == 2, "motif must use exactly two foreground colors"
    color_a, color_b = colors

    flipped = np.fliplr(motif)
    swapped = _swap_colors(motif, color_a, color_b)
    odd = _odd_variant(motif, color_a, color_b)
    assert not np.array_equal(flipped, motif), "motif must not be horizontally symmetric"
    assert not np.array_equal(swapped, motif), "color swap must change the motif"
    assert not np.array_equal(odd, motif), "odd-cycle transform must differ from the original motif"
    assert not np.array_equal(odd, flipped), "odd-cycle transform must require recoloring"
    assert not np.array_equal(odd, swapped), "odd-cycle transform must require horizontal reflection"

    return k, motif, (color_a, color_b)


def _build_output_from_motif(grid: np.ndarray, k: int, motif: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    h, w = map(int, grid.shape)
    out = np.zeros((h, w), dtype=int)
    out[:k, 1:] = motif

    odd_tile = _odd_variant(motif, colors[0], colors[1])
    write_r = k
    block_index = 0
    while write_r < h:
        tile = motif if block_index % 2 == 0 else odd_tile
        take = min(k, h - write_r)
        out[write_r : write_r + take, 1:] = tile[:take]
        write_r += take
        block_index += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the blank suffix by cyclically repeating the grey-marked motif, alternating odd repeats with a horizontal flip plus color swap.

    Input structure:
    - The grid has black background and a contiguous grey run in the top of column 0 that marks the motif height K.
    - Rows 0..K-1, columns 1..end contain a motif drawn with exactly two non-black, non-grey colors.
    - All rows below the motif are fully black, including column 0, so they form an empty suffix.

    Rule:
    - Extract the Kx(W-1) motif from rows 0..K-1 and columns 1..end, and treat the grey marker column as black in the output.
    - Keep the top motif rows in place as the first occurrence.
    - Fill the suffix downward in K-row blocks: the 2nd, 4th, 6th... occurrences use the original motif, while the 3rd, 5th, 7th... occurrences use the motif after horizontal reflection and swapping its two foreground colors.
    - If the last occurrence is truncated by the bottom edge, copy only its top rows.

    Key insight:
    - The grey run defines the motif height, and each repeat is controlled by a coupled parity rule: odd later cycles must apply both horizontal reflection and the two-color exchange together.
    """
    g = np.asarray(grid, dtype=int)
    k, motif, colors = _parse_input(g)
    return _build_output_from_motif(g, k, motif, colors)


def _sample_row_occupancy(width: int) -> list[int]:
    if width <= 4 or random.random() < 0.55:
        run_len = random.randint(2, min(width, 4))
        start = random.randint(0, width - run_len)
        cols = set(range(start, start + run_len))
        if width >= 6 and run_len < width and random.random() < 0.45:
            leftovers = [c for c in range(width) if c not in cols]
            cols.add(random.choice(leftovers))
        return sorted(cols)

    count = random.randint(2, min(width, 5))
    return sorted(random.sample(range(width), count))


def _make_motif(k: int, width: int, color_a: int, color_b: int) -> np.ndarray | None:
    motif = np.zeros((k, width), dtype=int)
    row_cols = [_sample_row_occupancy(width) for _ in range(k)]

    left_row = random.randrange(k)
    right_row = random.randrange(k)
    if 0 not in row_cols[left_row]:
        row_cols[left_row].append(0)
    if width - 1 not in row_cols[right_row]:
        row_cols[right_row].append(width - 1)

    for r in range(k):
        row_cols[r] = sorted(set(row_cols[r]))

    occupied = [(r, c) for r in range(k) for c in row_cols[r]]
    if len(occupied) < 2 * k + 1:
        return None

    min_secondary = max(2, len(occupied) // 4)
    max_secondary = min(len(occupied) - 1, max(min_secondary, (2 * len(occupied)) // 3))
    secondary_count = random.randint(min_secondary, max_secondary)
    secondary_cells = set(random.sample(occupied, secondary_count))

    for r, c in occupied:
        motif[r, c] = color_b if (r, c) in secondary_cells else color_a

    rows_a = {r for r, _c in np.argwhere(motif == color_a)}
    rows_b = {r for r, _c in np.argwhere(motif == color_b)}
    if len(rows_a) < 2 or len(rows_b) < 2:
        return None
    return motif


def _validate_output(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        k, motif, colors = _parse_input(inp)
    except AssertionError:
        return False

    if out.shape != inp.shape:
        return False
    if np.any(out == MARKER):
        return False
    if not np.all(out[:, 0] == BACKGROUND):
        return False
    if not np.array_equal(out[:k, 1:], motif):
        return False

    odd_tile = _odd_variant(motif, colors[0], colors[1])
    if np.array_equal(odd_tile, motif):
        return False

    h = inp.shape[0]
    write_r = k
    block_index = 0
    saw_original = False
    saw_odd = False
    while write_r < h:
        tile = motif if block_index % 2 == 0 else odd_tile
        take = min(k, h - write_r)
        if not np.array_equal(out[write_r : write_r + take, 1:], tile[:take]):
            return False
        if block_index % 2 == 0:
            saw_original = True
        else:
            saw_odd = True
        write_r += take
        block_index += 1

    if not (saw_original and saw_odd):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_input() -> np.ndarray | None:
    k = random.randint(2, 4)
    width = random.randint(8, 12)
    motif_width = width - 1
    full_suffix_blocks = random.randint(3, 5)
    partial_rows = random.randint(1, k - 1) if random.random() < 0.55 else 0
    height = k + full_suffix_blocks * k + partial_rows
    if max(height, width) > 30:
        return None

    color_a, color_b = random.sample(PALETTE, 2)
    motif = _make_motif(k, motif_width, color_a, color_b)
    if motif is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:k, 0] = MARKER
    grid[:k, 1:] = motif

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _validate_output(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-fill ARC input")

from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_MAX_ATTEMPTS = 400


def _separator_indices(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.count_nonzero(grid != BLACK, axis=1) == grid.shape[1])
    cols = np.flatnonzero(np.count_nonzero(grid != BLACK, axis=0) == grid.shape[0])
    assert rows.size >= 2 and cols.size >= 2, "expected divider rows and columns"
    return rows.astype(int), cols.astype(int)


def _cell_spans(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    bounds = [-1, *map(int, separators.tolist()), int(length)]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = {stop - start for start, stop in spans}
    assert min(sizes) > 0 and len(sizes) == 1, "expected equal-size cells"
    return spans


def _is_horizontally_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise a lattice, recover its true cell boundaries, and recolor each cell by horizontal mirror symmetry.

    Input structure:
    - Black is the background.
    - Complete blue divider rows and columns split the grid into a 3-by-3 or 4-by-4 lattice of equal cells.
    - Some divider pixels and some cell-interior pixels are overwritten by orange noise.
    - Each cell contains a red 4-connected pattern that does not touch that cell's boundary.

    Rule:
    - Identify true divider rows and columns as the rows/columns with no black pixels.
    - Erase every orange pixel to black, repaint divider rows and columns blue, and repaint divider intersections orange.
    - For each cell interior, test whether the red pattern equals its top-bottom reflection.
    - Recolor symmetric red patterns green and asymmetric red patterns yellow; leave black background unchanged.

    Key insight:
    - Orange pixels must be treated as removable noise before splitting the grid into cells, because noisy divider lines still remain detectable as complete non-black rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_rows, sep_cols = _separator_indices(g)
    row_spans = _cell_spans(g.shape[0], sep_rows)
    col_spans = _cell_spans(g.shape[1], sep_cols)

    cleaned = g.copy()
    cleaned[cleaned == ORANGE] = BLACK

    out = np.zeros_like(g)
    out[sep_rows, :] = BLUE
    out[:, sep_cols] = BLUE
    out[np.ix_(sep_rows, sep_cols)] = ORANGE

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = cleaned[r0:r1, c0:c1]
            red_mask = cell == RED
            if not red_mask.any():
                continue
            recolor = GREEN if _is_horizontally_symmetric(red_mask) else YELLOW
            patch = out[r0:r1, c0:c1]
            patch[red_mask] = recolor

    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enumerate_shape_pool(inner_size: int, symmetric: bool) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    total = inner_size * inner_size
    max_pixels = min(5, total)
    for bits in range(1 << total):
        count = bits.bit_count()
        if count < 2 or count > max_pixels:
            continue
        flat = np.array([(bits >> idx) & 1 for idx in range(total)], dtype=bool)
        mask = flat.reshape(inner_size, inner_size)
        if not _connected(mask):
            continue
        if _is_horizontally_symmetric(mask) == symmetric:
            pool.append(mask)
    assert pool, f"missing {'symmetric' if symmetric else 'asymmetric'} pool for {inner_size}"
    return pool


_SHAPE_POOLS = {
    2: {
        True: _enumerate_shape_pool(2, True),
        False: _enumerate_shape_pool(2, False),
    },
    3: {
        True: _enumerate_shape_pool(3, True),
        False: _enumerate_shape_pool(3, False),
    },
}


def _random_shape(cell_size: int, symmetric: bool) -> np.ndarray:
    inner_size = cell_size - 2
    mask = _SHAPE_POOLS[inner_size][symmetric][int(_RNG.integers(len(_SHAPE_POOLS[inner_size][symmetric])))]
    mask = mask.copy()
    if bool(_RNG.integers(0, 2)):
        mask = np.fliplr(mask)
    if bool(_RNG.integers(0, 2)):
        mask = np.flipud(mask)
    return mask


def _divider_positions(cell_size: int, cell_count: int) -> np.ndarray:
    step = cell_size + 1
    return (cell_size + np.arange(cell_count - 1) * step).astype(int)


def _cell_origin(index: int, cell_size: int) -> int:
    return int(index * (cell_size + 1))


def _render_clean_grid(cell_size: int, cell_count: int, cell_masks: list[list[np.ndarray]]) -> np.ndarray:
    side = cell_count * cell_size + (cell_count - 1)
    grid = np.zeros((side, side), dtype=int)
    sep_rows = _divider_positions(cell_size, cell_count)
    sep_cols = _divider_positions(cell_size, cell_count)
    grid[sep_rows, :] = BLUE
    grid[:, sep_cols] = BLUE

    for rr in range(cell_count):
        for cc in range(cell_count):
            mask = cell_masks[rr][cc]
            top = _cell_origin(rr, cell_size) + 1
            left = _cell_origin(cc, cell_size) + 1
            grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = RED
    return grid


def _render_expected_output(
    cell_size: int,
    cell_count: int,
    cell_masks: list[list[np.ndarray]],
    symmetry_flags: np.ndarray,
) -> np.ndarray:
    side = cell_count * cell_size + (cell_count - 1)
    out = np.zeros((side, side), dtype=int)
    sep_rows = _divider_positions(cell_size, cell_count)
    sep_cols = _divider_positions(cell_size, cell_count)
    out[sep_rows, :] = BLUE
    out[:, sep_cols] = BLUE
    out[np.ix_(sep_rows, sep_cols)] = ORANGE

    for rr in range(cell_count):
        for cc in range(cell_count):
            color = GREEN if bool(symmetry_flags[rr, cc]) else YELLOW
            mask = cell_masks[rr][cc]
            top = _cell_origin(rr, cell_size) + 1
            left = _cell_origin(cc, cell_size) + 1
            out[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = color
    return out


def _cell_index(pos: tuple[int, int], cell_size: int) -> tuple[int, int]:
    r, c = pos
    return int(r // (cell_size + 1)), int(c // (cell_size + 1))


def _sample_noise_positions(clean: np.ndarray, cell_size: int, symmetry_flags: np.ndarray) -> list[tuple[int, int]]:
    height, width = clean.shape
    sep_rows = set(map(int, _divider_positions(cell_size, symmetry_flags.shape[0]).tolist()))
    sep_cols = set(map(int, _divider_positions(cell_size, symmetry_flags.shape[1]).tolist()))

    line_candidates: list[tuple[int, int]] = []
    intersection_candidates: list[tuple[int, int]] = []
    symmetric_bg: list[tuple[int, int]] = []
    asymmetric_bg: list[tuple[int, int]] = []
    all_bg: list[tuple[int, int]] = []

    for r in range(height):
        for c in range(width):
            on_sep_row = r in sep_rows
            on_sep_col = c in sep_cols
            if on_sep_row and on_sep_col:
                intersection_candidates.append((r, c))
                continue
            if on_sep_row or on_sep_col:
                line_candidates.append((r, c))
                continue
            if clean[r, c] != BLACK:
                continue
            all_bg.append((r, c))
            cell_r, cell_c = _cell_index((r, c), cell_size)
            if bool(symmetry_flags[cell_r, cell_c]):
                symmetric_bg.append((r, c))
            else:
                asymmetric_bg.append((r, c))

    assert line_candidates and symmetric_bg and asymmetric_bg and all_bg

    chosen: list[tuple[int, int]] = []
    chosen.append(line_candidates[int(_RNG.integers(len(line_candidates)))])
    chosen.append(symmetric_bg[int(_RNG.integers(len(symmetric_bg)))])
    chosen.append(asymmetric_bg[int(_RNG.integers(len(asymmetric_bg)))])

    if intersection_candidates and bool(_RNG.integers(0, 2)):
        chosen.append(intersection_candidates[int(_RNG.integers(len(intersection_candidates)))])

    target_noise = int(_RNG.integers(5, 11))
    remaining_pool = [pos for pos in line_candidates + intersection_candidates + all_bg if pos not in set(chosen)]
    assert len(remaining_pool) >= target_noise - len(chosen)
    if target_noise > len(chosen):
        picks = _RNG.choice(len(remaining_pool), size=target_noise - len(chosen), replace=False)
        for idx in np.atleast_1d(picks):
            chosen.append(remaining_pool[int(idx)])
    return chosen


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    cell_count = int(_RNG.choice([3, 4]))
    cell_size = int(_RNG.choice([4, 5]))

    symmetry_flags = _RNG.integers(0, 2, size=(cell_count, cell_count)).astype(bool)
    if symmetry_flags.all() or (~symmetry_flags).all():
        flip_r = int(_RNG.integers(cell_count))
        flip_c = int(_RNG.integers(cell_count))
        symmetry_flags[flip_r, flip_c] = not bool(symmetry_flags[flip_r, flip_c])

    cell_masks: list[list[np.ndarray]] = []
    for rr in range(cell_count):
        row: list[np.ndarray] = []
        for cc in range(cell_count):
            row.append(_random_shape(cell_size, bool(symmetry_flags[rr, cc])))
        cell_masks.append(row)

    clean = _render_clean_grid(cell_size, cell_count, cell_masks)
    expected = _render_expected_output(cell_size, cell_count, cell_masks, symmetry_flags)
    inp = clean.copy()
    noise_positions = _sample_noise_positions(clean, cell_size, symmetry_flags)
    for r, c in noise_positions:
        inp[r, c] = ORANGE

    sep_rows = _divider_positions(cell_size, cell_count)
    sep_cols = _divider_positions(cell_size, cell_count)
    got_rows, got_cols = _separator_indices(inp)
    assert np.array_equal(got_rows, sep_rows)
    assert np.array_equal(got_cols, sep_cols)

    assert 5 <= len(noise_positions) <= 10
    assert any((r in sep_rows) ^ (c in sep_cols) for r, c in noise_positions)
    assert any((r not in sep_rows) and (c not in sep_cols) for r, c in noise_positions)
    assert any(
        (r not in sep_rows) and (c not in sep_cols) and bool(symmetry_flags[_cell_index((r, c), cell_size)])
        for r, c in noise_positions
    ), "need interior orange noise inside at least one symmetric cell"
    assert np.array_equal(main(inp), expected)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        try:
            inp, _ = _build_instance()
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")

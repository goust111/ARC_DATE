from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _visible_axis_period(grid: np.ndarray, axis: int) -> int:
    size = int(grid.shape[axis])
    best_period = None
    best_score = None

    for period in range(1, size):
        support = 0
        conflicts = 0
        if axis == 0:
            for r in range(size - period):
                row_a = grid[r]
                row_b = grid[r + period]
                mask = (row_a != BLACK) & (row_b != BLACK)
                if np.any(mask):
                    support += int(np.count_nonzero(mask))
                    conflicts += int(np.count_nonzero(row_a[mask] != row_b[mask]))
        else:
            for c in range(size - period):
                col_a = grid[:, c]
                col_b = grid[:, c + period]
                mask = (col_a != BLACK) & (col_b != BLACK)
                if np.any(mask):
                    support += int(np.count_nonzero(mask))
                    conflicts += int(np.count_nonzero(col_a[mask] != col_b[mask]))

        if conflicts == 0 and support > 0:
            score = (support, -period)
            if best_score is None or score > best_score:
                best_score = score
                best_period = period

    if best_period is None:
        raise AssertionError(f"failed to detect a visible period on axis {axis}")
    return int(best_period)


def _detect_periods(grid: np.ndarray) -> tuple[int, int]:
    return _visible_axis_period(grid, axis=0), _visible_axis_period(grid, axis=1)


def _orbit_coords(shape: tuple[int, int], r: int, c: int, row_period: int, col_period: int) -> list[tuple[int, int]]:
    rows, cols = shape
    r0 = r % row_period
    c0 = c % col_period
    return [(rr, cc) for rr in range(r0, rows, row_period) for cc in range(c0, cols, col_period)]


def _orbit_colors(grid: np.ndarray, r: int, c: int, row_period: int, col_period: int) -> set[int]:
    colors = {
        int(grid[rr, cc])
        for rr, cc in _orbit_coords(grid.shape, r, c, row_period, col_period)
        if int(grid[rr, cc]) != BLACK
    }
    return colors


def _infer_color(grid: np.ndarray, r: int, c: int, row_period: int, col_period: int) -> int | None:
    mirror_r = grid.shape[0] - 1 - r
    candidates = set()
    candidates |= _orbit_colors(grid, r, c, row_period, col_period)
    if int(grid[mirror_r, c]) != BLACK:
        candidates.add(int(grid[mirror_r, c]))
    candidates |= _orbit_colors(grid, mirror_r, c, row_period, col_period)

    if not candidates:
        return None
    if len(candidates) != 1:
        raise AssertionError(f"ambiguous repair colors at {(r, c)}: {sorted(candidates)}")
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore black occlusions by combining translational periodicity with top-bottom mirror symmetry.

    Input structure:
    - The grid is a 2D colored canvas whose true non-black pattern repeats with a fixed row period
      and a fixed column period.
    - The full pattern is also symmetric under reversing the row order.
    - Some cells have been replaced by black, including cases where an entire periodic orbit is hidden.

    Rule:
    - Detect the smallest visible row and column periods from the non-black cells.
    - For each black cell, first look for a unique non-black color in its periodic orbit.
    - If that orbit is fully black, use the mirrored cell; if the mirrored cell is also black, use the
      mirrored cell's periodic orbit. Repeat until every black cell is resolved.

    Key insight:
    - Periodicity recovers ordinary occlusions, while mirror symmetry unlocks orbits that were erased
      completely; some cells require both constraints in combination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a non-empty grid"

    row_period, col_period = _detect_periods(g)
    out = g.copy()

    for _ in range(out.size):
        if not np.any(out == BLACK):
            break
        changed = False
        for r, c in np.argwhere(out == BLACK):
            color = _infer_color(out, int(r), int(c), row_period, col_period)
            if color is not None:
                out[int(r), int(c)] = int(color)
                changed = True
        if not changed:
            break

    assert not np.any(out == BLACK), "some occlusions could not be restored"
    return out


def _exact_axis_period(grid: np.ndarray, axis: int) -> int:
    size = int(grid.shape[axis])
    for period in range(1, size):
        if axis == 0:
            if np.array_equal(grid[:-period, :], grid[period:, :]):
                return period
        else:
            if np.array_equal(grid[:, :-period], grid[:, period:]):
                return period
    return size


def _row_pattern_with_exact_period(col_period: int, palette: tuple[int, ...]) -> np.ndarray:
    while True:
        row = np.array([random.choice(palette) for _ in range(col_period)], dtype=int)
        if len(set(map(int, row))) >= 2 and _exact_axis_period(row.reshape(1, -1), axis=1) == col_period:
            return row


def _build_full_pattern() -> tuple[np.ndarray, int, int]:
    while True:
        row_period = random.choice((4, 6))
        col_period = random.choice((3, 4, 5))
        row_repeats = random.randint(2, min(4, 30 // row_period))
        col_repeats = random.randint(2, min(5, 30 // col_period))
        palette = tuple(random.sample(PALETTE, k=random.randint(3, 4)))

        half_rows = [_row_pattern_with_exact_period(col_period, palette) for _ in range(row_period // 2)]
        tile = np.array(half_rows + half_rows[::-1], dtype=int)
        full = np.tile(tile, (row_repeats, col_repeats))

        if len(set(map(int, np.unique(full)))) != len(palette):
            continue
        if not np.array_equal(full, np.flipud(full)):
            continue
        if _exact_axis_period(full, axis=0) != row_period:
            continue
        if _exact_axis_period(full, axis=1) != col_period:
            continue
        return full, row_period, col_period


def _orbit_cells_by_residue(shape: tuple[int, int], residue: tuple[int, int], row_period: int, col_period: int) -> list[tuple[int, int]]:
    rr, cc = residue
    rows, cols = shape
    return [(r, c) for r in range(rr, rows, row_period) for c in range(cc, cols, col_period)]


def _classify_black_cells(grid: np.ndarray, row_period: int, col_period: int) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0}
    for r, c in np.argwhere(grid == BLACK):
        rr = int(r)
        cc = int(c)
        own = _orbit_colors(grid, rr, cc, row_period, col_period)
        mirror_r = grid.shape[0] - 1 - rr
        if own:
            counts["A"] += 1
            continue
        if int(grid[mirror_r, cc]) != BLACK:
            counts["B"] += 1
            continue
        if _orbit_colors(grid, mirror_r, cc, row_period, col_period):
            counts["C"] += 1
            continue
        raise AssertionError(f"unrecoverable black cell at {(rr, cc)}")
    return counts


def _make_occluded_input(full: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    grid = full.copy()
    pair_choices = [
        ((rr, cc), (row_period - 1 - rr, cc))
        for rr in range(row_period // 2)
        for cc in range(col_period)
    ]
    random.shuffle(pair_choices)
    if len(pair_choices) < 3:
        return None

    pair_for_b = pair_choices[0]
    pair_for_c = pair_choices[1]
    used_residues = {pair_for_b[0], pair_for_b[1], pair_for_c[0], pair_for_c[1]}

    remaining = []
    for pair in pair_choices[2:]:
        for residue in pair:
            if residue not in used_residues:
                remaining.append(residue)
    if not remaining:
        return None

    a_residue = random.choice(remaining)
    a_cells = _orbit_cells_by_residue(grid.shape, a_residue, row_period, col_period)
    a_black_count = random.randint(1, len(a_cells) - 1)
    for r, c in random.sample(a_cells, k=a_black_count):
        grid[r, c] = BLACK

    b_hidden, _b_mirror = pair_for_b
    for r, c in _orbit_cells_by_residue(grid.shape, b_hidden, row_period, col_period):
        grid[r, c] = BLACK

    c_hidden, c_mirror = pair_for_c
    c_hidden_cells = _orbit_cells_by_residue(grid.shape, c_hidden, row_period, col_period)
    c_mirror_cells = _orbit_cells_by_residue(grid.shape, c_mirror, row_period, col_period)
    for r, c in c_hidden_cells:
        grid[r, c] = BLACK

    mirror_black_count = random.randint(1, len(c_mirror_cells) - 1)
    for r, c in random.sample(c_mirror_cells, k=mirror_black_count):
        grid[r, c] = BLACK

    return grid


def _valid_instance(inp: np.ndarray, out: np.ndarray, row_period: int, col_period: int) -> bool:
    if np.count_nonzero(inp == BLACK) == 0:
        return False
    if not np.array_equal(out, np.flipud(out)):
        return False
    if _exact_axis_period(out, axis=0) != row_period or _exact_axis_period(out, axis=1) != col_period:
        return False
    try:
        detected = _detect_periods(inp)
    except AssertionError:
        return False
    if detected != (row_period, col_period):
        return False
    if not np.array_equal(main(inp), out):
        return False
    counts = _classify_black_cells(inp, row_period, col_period)
    return counts["A"] > 0 and counts["B"] > 0 and counts["C"] > 0


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        full, row_period, col_period = _build_full_pattern()
        inp = _make_occluded_input(full, row_period, col_period)
        if inp is None:
            continue
        if _valid_instance(inp, full, row_period, col_period):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")

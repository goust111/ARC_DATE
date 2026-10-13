from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
BLUE = 1
GREY = 5
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_axes(grid: np.ndarray) -> tuple[int, int]:
    axis_rows = np.where(np.all(grid == GREY, axis=1))[0]
    axis_cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(axis_rows) == 1, "expected exactly one full grey row"
    assert len(axis_cols) == 1, "expected exactly one full grey column"
    return int(axis_rows[0]), int(axis_cols[0])


def _orbit_positions(r: int, c: int, axis_r: int, axis_c: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (r, c),
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    }
    return tuple(sorted(orbit))


def _reconstruct_expected(
    grid: np.ndarray, axis_r: int, axis_c: int
) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    h, w = grid.shape
    expected = np.full((h, w), BACKGROUND, dtype=int)
    expected[axis_r, :] = GREY
    expected[:, axis_c] = GREY

    defects: set[tuple[int, int]] = set()
    ambiguous_orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in range(h):
        for c in range(w):
            if r == axis_r or c == axis_c:
                continue
            orbit = _orbit_positions(r, c, axis_r, axis_c)
            if orbit in seen:
                continue
            seen.add(orbit)
            if any(not _in_bounds(grid.shape, rr, cc) for rr, cc in orbit):
                continue

            blue_count = sum(int(grid[rr, cc]) == BLUE for rr, cc in orbit)
            if blue_count == 2:
                ambiguous_orbits.append(orbit)
                continue

            expected_color = BLUE if blue_count >= 3 else BACKGROUND
            for rr, cc in orbit:
                expected[rr, cc] = expected_color
                if int(grid[rr, cc]) != expected_color:
                    defects.add((rr, cc))

    return expected, defects, ambiguous_orbits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect symmetry defects by reconstructing the intended pattern, then mark their surroundings.

    Input structure:
    - The grid has black background, one full grey row, and one full grey column forming the symmetry axes.
    - Away from the axes, blue pixels form a sparse pattern that should be symmetric across both axes.
    - A few off-axis blue cells are deleted to black, and a few off-axis black cells are changed to blue.

    Rule:
    - Find the full grey row and column.
    - For each off-axis four-cell mirror orbit, count how many positions are blue in the input.
    - If the orbit has at least three blue cells, all four positions should be blue; if it has at most one blue cell, all four should be black.
    - Any off-axis position whose actual color differs from that reconstructed expectation is a defect.
    - Return the input copy, but recolor every background cell in the 8-neighborhood of each defect to orange; defect cells and axis cells stay unchanged.

    Key insight:
    - The intended pattern is recovered by majority vote inside each four-way symmetry orbit, and only then can the true defect pixels be identified.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c = _find_axes(g)
    _expected, defects, ambiguous_orbits = _reconstruct_expected(g, axis_r, axis_c)
    assert not ambiguous_orbits, "symmetry orbits must not be split 2-vs-2"

    out = g.copy()
    for r, c in defects:
        for rr, cc in _neighbors8(r, c):
            if not _in_bounds(g.shape, rr, cc):
                continue
            if rr == axis_r or cc == axis_c:
                continue
            if (rr, cc) in defects:
                continue
            if int(g[rr, cc]) == BACKGROUND:
                out[rr, cc] = ORANGE

    return out


def _candidate_side(axis: int, limit: int) -> list[int]:
    return [pos for pos in range(axis) if 0 <= 2 * axis - pos < limit]


def _sample_seed_positions(row_count: int, col_count: int, target: int) -> set[tuple[int, int]] | None:
    cells = [(r, c) for r in range(row_count) for c in range(col_count)]
    for _ in range(80):
        used = {(random.randrange(row_count), random.randrange(col_count))}
        while len(used) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in used:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < row_count and 0 <= cc < col_count and (rr, cc) not in used:
                        frontier.append((rr, cc))
            pool = frontier if frontier and random.random() < 0.8 else cells
            choice = random.choice(pool)
            used.add(choice)
        if len({r for r, _ in used}) >= 2 and len({c for _, c in used}) >= 2:
            return used
    return None


def _orbit_from_seed(rows: list[int], cols: list[int], axis_r: int, axis_c: int, seed: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r = rows[seed[0]]
    c = cols[seed[1]]
    return _orbit_positions(r, c, axis_r, axis_c)


def _build_ideal_grid(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    blue_orbits: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[axis_r, :] = GREY
    grid[:, axis_c] = GREY
    for orbit in blue_orbits:
        for r, c in orbit:
            grid[r, c] = BLUE
    return grid


def _choose_defect_positions(
    orbit_choices: list[tuple[tuple[int, int], ...]],
    defect_count: int,
    existing: set[tuple[int, int]],
    min_gap: int = 3,
) -> list[tuple[tuple[int, int], ...], tuple[int, int]] | None:
    shuffled = orbit_choices.copy()
    random.shuffle(shuffled)
    chosen: list[tuple[tuple[int, int], ...], tuple[int, int]] = []
    blocked = set(existing)

    for orbit in shuffled:
        cells = list(orbit)
        random.shuffle(cells)
        for pos in cells:
            if all(max(abs(pos[0] - r), abs(pos[1] - c)) >= min_gap for r, c in blocked):
                chosen.append((orbit, pos))
                blocked.add(pos)
                break
        if len(chosen) == defect_count:
            return chosen
    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    axis_r = random.randint(4, h - 5)
    axis_c = random.randint(4, w - 5)

    rows = _candidate_side(axis_r, h)
    cols = _candidate_side(axis_c, w)
    if len(rows) < 4 or len(cols) < 4:
        return None

    seed_count = random.randint(4, min(7, len(rows) * len(cols) - 3))
    seeds = _sample_seed_positions(len(rows), len(cols), seed_count)
    if seeds is None:
        return None

    all_seed_cells = [(r, c) for r in range(len(rows)) for c in range(len(cols))]
    blue_seed_set = set(seeds)
    black_seed_set = [cell for cell in all_seed_cells if cell not in blue_seed_set]
    if len(black_seed_set) < 2:
        return None

    blue_orbits = [_orbit_from_seed(rows, cols, axis_r, axis_c, seed) for seed in sorted(blue_seed_set)]
    black_orbits = [_orbit_from_seed(rows, cols, axis_r, axis_c, seed) for seed in black_seed_set]

    ideal = _build_ideal_grid(h, w, axis_r, axis_c, blue_orbits)

    deletion_count = random.choice((2, 3))
    addition_count = random.choice((1, 2))
    if len(blue_orbits) <= deletion_count:
        return None

    deletion_choices = _choose_defect_positions(blue_orbits, deletion_count, set())
    if deletion_choices is None:
        return None
    addition_choices = _choose_defect_positions(
        black_orbits,
        addition_count,
        {pos for _orbit, pos in deletion_choices},
    )
    if addition_choices is None:
        return None

    noisy = ideal.copy()
    planted_defects: set[tuple[int, int]] = set()
    for _orbit, pos in deletion_choices:
        noisy[pos] = BACKGROUND
        planted_defects.add(pos)
    for _orbit, pos in addition_choices:
        noisy[pos] = BLUE
        planted_defects.add(pos)

    expected, detected_defects, ambiguous_orbits = _reconstruct_expected(noisy, axis_r, axis_c)
    if ambiguous_orbits:
        return None
    if not np.array_equal(expected, ideal):
        return None
    if detected_defects != planted_defects:
        return None

    actual_deletions = sum(int(ideal[r, c]) == BLUE and int(noisy[r, c]) == BACKGROUND for r, c in detected_defects)
    actual_additions = sum(int(ideal[r, c]) == BACKGROUND and int(noisy[r, c]) == BLUE for r, c in detected_defects)
    if actual_deletions not in (2, 3) or actual_additions not in (1, 2):
        return None

    out = main(noisy)
    if np.array_equal(out, noisy):
        return None
    if np.any(out[axis_r, :] == ORANGE) or np.any(out[:, axis_c] == ORANGE):
        return None
    if np.sum(out == ORANGE) < len(detected_defects):
        return None

    for r, c in detected_defects:
        if int(out[r, c]) != int(noisy[r, c]):
            return None

    return noisy


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-defect input")

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


MASK = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
GEN_SIZE_CHOICES = (5, 6, 7)
GEN_RUN_MIN = 1
GEN_RUN_MAX = 4
GEN_MAX_ATTEMPTS = 600


def _compatible_line(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all((a == b) | (a == MASK) | (b == MASK)))


def _merge_line(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    conflict = (out != MASK) & (b != MASK) & (out != b)
    assert not np.any(conflict), "compatible lines cannot disagree on visible colors"
    fill = (out == MASK) & (b != MASK)
    out[fill] = b[fill]
    return out


def _compress_axis(grid: np.ndarray, axis: int) -> np.ndarray:
    work = grid if axis == 0 else grid.T
    merged: list[np.ndarray] = []
    rep = work[0].copy()

    for idx in range(1, work.shape[0]):
        line = work[idx]
        if _compatible_line(rep, line):
            rep = _merge_line(rep, line)
        else:
            merged.append(rep)
            rep = line.copy()
    merged.append(rep)

    out = np.stack(merged, axis=0)
    return out if axis == 0 else out.T


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    return _compress_axis(_compress_axis(grid, axis=0), axis=1)


def _orbit_cells(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
        (c, r),
        (c, n - 1 - r),
        (n - 1 - c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(pts))


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            orbit = _orbit_cells(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _repair_compressed(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "compressed grid must be square"

    out = g.copy()
    n = g.shape[0]
    for r, c in zip(*np.where(g == MASK)):
        colors = {
            int(g[rr, cc])
            for rr, cc in _orbit_cells(n, int(r), int(c))
            if g[rr, cc] != MASK
        }
        assert len(colors) == 1, "each masked compressed cell needs one visible orbit color"
        out[r, c] = colors.pop()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse wildcard-equal stretch runs, then repair masked cells from D4 symmetry in the compressed square.

    Input structure:
    - The large grid is a row- and column-stretched version of a smaller square pattern.
    - Neighboring duplicated rows or columns occur in contiguous runs, and pink cells are masks.
    - Symmetry exists only after compression: the compressed square is D4-symmetric, while the large grid is not.

    Rule:
    - Scan top to bottom and merge each maximal run of adjacent rows that match when pink is treated as a wildcard.
    - On the row-compressed grid, scan left to right and merge each maximal run of adjacent wildcard-matching columns.
    - In the compressed square, replace every remaining pink cell by the unique visible color in its full D4 orbit.

    Key insight:
    - The masks hide values on a symmetry table that is only visible after removing the non-uniform row and column stretching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0, "input must be a non-empty 2-D grid"

    compressed = _compress_grid(g)
    return _repair_compressed(compressed)


def _has_adjacent_equal(grid: np.ndarray, axis: int) -> bool:
    work = grid if axis == 0 else grid.T
    return any(np.array_equal(work[i - 1], work[i]) for i in range(1, work.shape[0]))


def _has_adjacent_compatible(grid: np.ndarray, axis: int) -> bool:
    work = grid if axis == 0 else grid.T
    return any(_compatible_line(work[i - 1], work[i]) for i in range(1, work.shape[0]))


def _is_d4_symmetric(grid: np.ndarray) -> bool:
    n = grid.shape[0]
    if grid.shape[0] != grid.shape[1]:
        return False
    for r in range(n):
        for c in range(n):
            vals = {int(grid[rr, cc]) for rr, cc in _orbit_cells(n, r, c)}
            if len(vals) != 1:
                return False
    return True


def _expand(grid: np.ndarray, row_runs: Iterable[int], col_runs: Iterable[int]) -> np.ndarray:
    return np.repeat(np.repeat(grid, list(row_runs), axis=0), list(col_runs), axis=1)


def _random_runs(n: int) -> list[int]:
    for _ in range(200):
        runs = [random.randint(GEN_RUN_MIN, GEN_RUN_MAX) for _ in range(n)]
        if max(runs) == 1:
            continue
        if runs == runs[::-1]:
            continue
        return runs
    raise RuntimeError("failed to sample asymmetric run lengths")


def _build_solved_small(n: int) -> np.ndarray | None:
    orbits = _all_orbits(n)
    color_count = random.randint(3, min(6, len(PALETTE), len(orbits)))
    colors = random.sample(PALETTE, color_count)
    orbit_colors = colors + [random.choice(colors) for _ in range(len(orbits) - color_count)]
    random.shuffle(orbit_colors)

    out = np.zeros((n, n), dtype=int)
    for orbit, color in zip(orbits, orbit_colors):
        for r, c in orbit:
            out[r, c] = int(color)

    if len(set(map(int, out.flat))) < 3:
        return None
    if _has_adjacent_equal(out, axis=0) or _has_adjacent_equal(out, axis=1):
        return None
    if not _is_d4_symmetric(out):
        return None
    return out


def _mask_small(solved: np.ndarray) -> np.ndarray | None:
    n = solved.shape[0]
    orbits = [orbit for orbit in _all_orbits(n) if len(orbit) > 1]
    if len(orbits) < 2:
        return None

    masked = solved.copy()
    orbit_count = random.randint(2, min(5, len(orbits)))
    chosen_orbits = random.sample(orbits, orbit_count)
    masked_positions: set[tuple[int, int]] = set()

    for orbit in chosen_orbits:
        hide_count = random.randint(1, len(orbit) - 1)
        for cell in random.sample(list(orbit), hide_count):
            masked[cell] = MASK
            masked_positions.add(cell)

    if len(masked_positions) < 2:
        return None
    if _has_adjacent_compatible(masked, axis=0) or _has_adjacent_compatible(masked, axis=1):
        return None

    for orbit in _all_orbits(n):
        visible = {int(masked[r, c]) for r, c in orbit if masked[r, c] != MASK}
        if any(masked[r, c] == MASK for r, c in orbit):
            if len(visible) != 1:
                return None
    return masked


def _semantic_checks(
    solved: np.ndarray,
    masked_small: np.ndarray,
    row_runs: list[int],
    col_runs: list[int],
    inp: np.ndarray,
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if solved.shape[0] != solved.shape[1]:
        return False
    if row_runs == row_runs[::-1] or col_runs == col_runs[::-1]:
        return False
    if inp.shape[0] == inp.shape[1]:
        return False
    if not _is_d4_symmetric(solved):
        return False
    if not np.array_equal(_compress_grid(inp), masked_small):
        return False
    if not np.array_equal(main(inp), solved):
        return False

    expanded_solved = _expand(solved, row_runs, col_runs)
    if np.array_equal(expanded_solved, np.flipud(expanded_solved)):
        return False
    if np.array_equal(expanded_solved, np.fliplr(expanded_solved)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GEN_SIZE_CHOICES)
    solved = _build_solved_small(n)
    if solved is None:
        return None

    masked_small = _mask_small(solved)
    if masked_small is None:
        return None

    for _ in range(100):
        row_runs = _random_runs(n)
        col_runs = _random_runs(n)
        if sum(row_runs) > 30 or sum(col_runs) > 30:
            continue
        if sum(row_runs) == sum(col_runs):
            continue

        inp = _expand(masked_small, row_runs, col_runs)
        if _semantic_checks(solved, masked_small, row_runs, col_runs, inp):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-domain symmetry-repair input")

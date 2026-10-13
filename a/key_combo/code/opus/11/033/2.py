from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

SEED_COLORS = (BLUE, GREEN, ORANGE)
PERIOD = 3
GEN_RETRIES = 500


def _trigger_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == RED)]
    assert len(rows) == 1, "expected exactly one full-width red trigger bar"
    return int(rows[0])


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != RED))
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert seeds, "expected at least one seed pixel"
    seeds.sort()
    return seeds


def _bar_columns(seed_col: int, width: int) -> list[int]:
    return list(range(int(seed_col), int(width), PERIOD))


def _render_bars(
    shape: tuple[int, int], trigger_row: int, seeds: Iterable[tuple[int, int, int]]
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)
    out[trigger_row, :] = RED
    for seed_row, seed_col, color in seeds:
        out[: seed_row + 1, seed_col:w:PERIOD] = color
    return out


def _grey_halo_cells(
    base: np.ndarray,
    trigger_row: int,
    seeds: Iterable[tuple[int, int, int]],
    active_indices: set[int] | None = None,
) -> set[tuple[int, int]]:
    h, w = base.shape
    if active_indices is None:
        active_indices = {
            idx for idx, (seed_row, _seed_col, _color) in enumerate(seeds) if seed_row == trigger_row - 1
        }

    halo: set[tuple[int, int]] = set()
    for idx, (seed_row, seed_col, _color) in enumerate(seeds):
        if idx not in active_indices:
            continue
        for col in _bar_columns(seed_col, w):
            for rr in range(max(0, seed_row - 1), min(h, seed_row + 2)):
                for cc in range(max(0, col - 1), min(w, col + 2)):
                    if base[rr, cc] == BLACK:
                        halo.add((rr, cc))
    return halo


def _render_output(
    shape: tuple[int, int],
    trigger_row: int,
    seeds: list[tuple[int, int, int]],
    active_indices: set[int] | None = None,
) -> np.ndarray:
    out = _render_bars(shape, trigger_row, seeds)
    for rr, cc in _grey_halo_cells(out, trigger_row, seeds, active_indices):
        out[rr, cc] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw periodic vertical bars from multiple seeds, then add gray halos only for the seed adjacent to the trigger bar.

    Input structure:
    - The grid is a black background with one full-width red horizontal trigger bar near the bottom.
    - Two or three isolated seed pixels lie above the bar, using distinct colors chosen from blue, green, and orange.
    - Exactly one seed is immediately above the trigger bar; the other seeds are higher up.

    Rule:
    - For every seed, draw same-colored vertical bars in the seed column and every third column to its right, extending from the top row down to the seed row inclusive.
    - A seed is activated if and only if its row is exactly one row above the red trigger bar.
    - For each activated seed, expand every bottom endpoint of its bars into a Chebyshev-distance-1 halo, coloring only black cells gray while preserving all existing non-black pixels.

    Key insight:
    - The same seed simultaneously determines two independent things: the phase of its periodic bar family and whether that family gains halos, with activation decided only by row adjacency to the trigger bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    trigger_row = _trigger_row(g)
    seeds = _extract_seeds(g)
    return _render_output(g.shape, trigger_row, seeds)


def _sample_height_width() -> tuple[int, int]:
    return random.randint(13, 18), random.randint(16, 24)


def _sample_trigger_row(height: int) -> int:
    bottom_margin = random.choice((2, 3))
    return height - 1 - bottom_margin


def _sample_seed_residues(width: int, seed_count: int) -> list[int]:
    edge_mode = random.random() < 0.45
    if edge_mode and random.random() < 0.5:
        active_residue = 0
    elif edge_mode:
        active_residue = (width - 1) % PERIOD
    else:
        active_residue = random.randrange(PERIOD)

    others = [r for r in range(PERIOD) if r != active_residue]
    random.shuffle(others)
    return [active_residue] + others[: seed_count - 1]


def _candidate_seed_cols(width: int, residue: int, min_bars: int) -> list[int]:
    cols = []
    for col in range(residue, width, PERIOD):
        if ((width - 1 - col) // PERIOD) + 1 >= min_bars:
            cols.append(int(col))
    return cols


def _sample_seed_cols(width: int, residues: list[int]) -> list[int] | None:
    cols: list[int] = []
    for idx, residue in enumerate(residues):
        min_bars = 2 if idx == 0 else random.choice((1, 2))
        candidates = _candidate_seed_cols(width, residue, min_bars)
        if idx == 0 and residue == 0 and random.random() < 0.5 and 0 in candidates:
            cols.append(0)
            continue
        if not candidates:
            return None
        cols.append(int(random.choice(candidates)))
    return cols


def _sample_inactive_rows(trigger_row: int, inactive_count: int) -> list[int] | None:
    candidates = list(range(0, trigger_row - 2))
    if inactive_count > len(candidates):
        return None

    rows: list[int] = []
    if inactive_count >= 2 and random.random() < 0.35 and 0 in candidates:
        rows.append(0)
        candidates.remove(0)

    random.shuffle(candidates)
    rows.extend(candidates[: inactive_count - len(rows)])
    if inactive_count and max(rows) < 2:
        return None
    return rows


def _make_grid(
    height: int, width: int, trigger_row: int, seeds: list[tuple[int, int, int]]
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[trigger_row, :] = RED
    for row, col, color in seeds:
        grid[row, col] = color
    return grid


def _validate_input(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h <= 30 and w <= 30

    trigger_row = _trigger_row(grid)
    seeds = _extract_seeds(grid)
    assert 2 <= len(seeds) <= 3
    assert int(np.count_nonzero(grid == RED)) == w
    assert all(grid[r, c] in SEED_COLORS for r, c, _color in seeds)

    colors = [color for _r, _c, color in seeds]
    rows = [row for row, _c, _color in seeds]
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    assert len(set(rows)) == len(rows), "seed rows must be distinct"

    active = [idx for idx, (row, _c, _color) in enumerate(seeds) if row == trigger_row - 1]
    assert len(active) == 1, "exactly one seed must be adjacent to the trigger bar"
    for idx, (row, _c, _color) in enumerate(seeds):
        if idx not in active:
            assert row <= trigger_row - 3, "inactive seeds must be at least 3 rows above the bar"

    bar_sets = [set(_bar_columns(col, w)) for _row, col, _color in seeds]
    for i in range(len(bar_sets)):
        for j in range(i + 1, len(bar_sets)):
            assert bar_sets[i].isdisjoint(bar_sets[j]), "periodic bar columns must not overlap"

    out = main(grid)
    base = _render_bars(grid.shape, trigger_row, seeds)
    expected_halo = _grey_halo_cells(base, trigger_row, seeds)
    actual_halo = {tuple(map(int, rc)) for rc in np.argwhere(out == GREY)}
    assert actual_halo == expected_halo, "gray cells must match exactly the activated halos"
    assert expected_halo, "activated seed must produce at least one gray halo cell"

    no_halo = _render_output(grid.shape, trigger_row, seeds, active_indices=set())
    all_halo = _render_output(grid.shape, trigger_row, seeds, active_indices=set(range(len(seeds))))
    assert not np.array_equal(out, no_halo), "activation must change the result"
    assert not np.array_equal(out, all_halo), "inactive seeds must remain non-halo distractors"

    inactive_rows = [row for idx, (row, _c, _color) in enumerate(seeds) if idx not in active]
    assert inactive_rows and max(inactive_rows) >= 2, "need a visible inactive bar"
    active_seed = seeds[active[0]]
    assert len(_bar_columns(active_seed[1], w)) >= 2, "active seed should show periodicity"
    assert np.any(out[:trigger_row, :] == active_seed[2]), "bars must exist above the trigger row"


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_height_width()
    trigger_row = _sample_trigger_row(height)
    seed_count = random.choice((2, 3))

    residues = _sample_seed_residues(width, seed_count)
    cols = _sample_seed_cols(width, residues)
    if cols is None:
        return None

    active_row = trigger_row - 1
    inactive_rows = _sample_inactive_rows(trigger_row, seed_count - 1)
    if inactive_rows is None:
        return None

    colors = random.sample(list(SEED_COLORS), seed_count)
    seeds = [(active_row, cols[0], colors[0])]
    for row, col, color in zip(inactive_rows, cols[1:], colors[1:]):
        seeds.append((int(row), int(col), int(color)))

    grid = _make_grid(height, width, trigger_row, seeds)
    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-seed trigger-halo ARC input")

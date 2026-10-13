from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RNG = np.random.default_rng()

GRID_COUNTS = [3, 4]
CELL_SIZES = [3, 5]
MAX_ATTEMPTS = 600


def _partition_spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), size]
    spans = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert spans and all(start < stop for start, stop in spans)
    return spans


def _lattice_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_spans = _partition_spans(g.shape[0], row_seps)
    col_spans = _partition_spans(g.shape[1], col_seps)
    assert len(row_spans) == len(col_spans) and len(row_spans) in (3, 4)
    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1 and row_sizes == col_sizes
    return row_spans, col_spans


def _cell(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], r: int, c: int) -> np.ndarray:
    r0, r1 = row_spans[r]
    c0, c1 = col_spans[c]
    return grid[r0:r1, c0:c1]


def _pixel_cell_bounds(cell_size: int, r: int, c: int) -> tuple[int, int, int, int]:
    r0 = r * (cell_size + 1)
    c0 = c * (cell_size + 1)
    return r0, r0 + cell_size, c0, c0 + cell_size


def _is_asymmetric(dirs: Iterable[tuple[int, int]]) -> bool:
    base = frozenset(tuple(map(int, d)) for d in dirs)
    if not base:
        return False
    return frozenset((-dr, -dc) for dr, dc in base) != base


def _safe_cells(n: int, dirs: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    safe: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if all(0 <= r + dr < n and 0 <= c + dc < n for dr, dc in dirs):
                safe.add((r, c))
    return safe


def _target_cells(cell: tuple[int, int], dirs: tuple[tuple[int, int], ...], n: int) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (r + dr, c + dc)
        for dr, dc in dirs
        if 0 <= r + dr < n and 0 <= c + dc < n
    }


def _locate_exemplar_and_template(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    exemplar_cells = []
    for r in range(len(row_spans)):
        for c in range(len(col_spans)):
            view = _cell(grid, row_spans, col_spans, r, c)
            if np.any(view == BLUE):
                exemplar_cells.append((r, c, view))

    assert len(exemplar_cells) == 1, "expected exactly one exemplar cell"
    ex_r, ex_c, exemplar_view = exemplar_cells[0]
    assert int(np.sum(exemplar_view == BLUE)) == 1, "exemplar must contain exactly one blue pixel"

    dirs = []
    n = len(row_spans)
    for dr, dc in CARDINAL_DIRS:
        nr, nc = ex_r + dr, ex_c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        if np.any(_cell(grid, row_spans, col_spans, nr, nc) == RED):
            dirs.append((dr, dc))

    dirs = tuple(dirs)
    assert dirs and _is_asymmetric(dirs), "expected a non-symmetric exemplar template"
    return (ex_r, ex_c), dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the exemplar's red stamp only from cells containing exactly two orange markers.

    Input structure:
    - Grey separator rows and columns divide the grid into a 3x3 or 4x4 lattice of equal square cells.
    - One exemplar cell contains exactly one blue pixel, and some of its cardinally adjacent cells contain red pixels that indicate a directional template.
    - Every other ordinary cell is otherwise black or contains one to three orange marker pixels; separator lines stay solid grey.

    Rule:
    - Recover the lattice from the grey separators and find the unique exemplar cell.
    - Read the exemplar template as the set of cardinal neighbor directions whose cells contain red.
    - For every non-exemplar cell, count its orange pixels; if and only if the count is exactly two, fill each template-directed neighboring cell entirely red, discarding only out-of-bounds targets.
    - Keep the original separator lines, orange markers, and blue exemplar pixel unchanged on top of the stamped red cells.

    Key insight:
    - The red pattern near the blue exemplar is only a directional stencil; the actual gate is the orange-marker count, and only cells with exactly two markers are allowed to fire that stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans = _lattice_spans(g)
    (ex_r, ex_c), dirs = _locate_exemplar_and_template(g, row_spans, col_spans)
    n = len(row_spans)

    out = g.copy()
    for r in range(n):
        for c in range(n):
            if (r, c) == (ex_r, ex_c):
                continue
            orange_count = int(np.sum(_cell(g, row_spans, col_spans, r, c) == ORANGE))
            if orange_count != 2:
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                r0, r1 = row_spans[nr]
                c0, c1 = col_spans[nc]
                out[r0:r1, c0:c1] = RED

    out[g == GREY] = GREY
    out[g == ORANGE] = ORANGE
    out[g == BLUE] = BLUE
    return out


def _all_direction_sets() -> list[tuple[tuple[int, int], ...]]:
    direction_sets = []
    for size in range(1, 4):
        for subset in itertools.combinations(CARDINAL_DIRS, size):
            if _is_asymmetric(subset):
                direction_sets.append(tuple(subset))
    return direction_sets


ASYM_DIRECTION_SETS = _all_direction_sets()
DIRECTION_SETS_BY_GRID = {
    3: [dirs for dirs in ASYM_DIRECTION_SETS if len(dirs) == 2],
    4: [dirs for dirs in ASYM_DIRECTION_SETS if len(dirs) in (2, 3)],
}


def _choice(seq):
    return seq[int(RNG.integers(0, len(seq)))]


def _sample_orange_positions(cell_size: int, count: int) -> list[tuple[int, int]]:
    all_positions = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    picks = RNG.choice(len(all_positions), size=count, replace=False)
    return [all_positions[int(i)] for i in picks]


def _render_input(model: dict) -> np.ndarray:
    n = model["n"]
    cell_size = model["cell_size"]
    total = n * cell_size + (n - 1)
    grid = np.full((total, total), BLACK, dtype=int)

    for k in range(1, n):
        idx = k * cell_size + (k - 1)
        grid[idx, :] = GREY
        grid[:, idx] = GREY

    center = cell_size // 2
    counts = model["counts"]
    exemplar = model["exemplar"]
    template_cells = set(model["template_cells"])

    for r in range(n):
        for c in range(n):
            r0, r1, c0, c1 = _pixel_cell_bounds(cell_size, r, c)
            if (r, c) == exemplar:
                grid[r0 + center, c0 + center] = BLUE
                continue
            if (r, c) in template_cells:
                grid[r0 + center, c0 + center] = RED
                continue
            count = counts[r, c]
            for dr, dc in _sample_orange_positions(cell_size, int(count)):
                grid[r0 + dr, c0 + dc] = ORANGE

    return grid


def _render_expected(model: dict, input_grid: np.ndarray) -> np.ndarray:
    cell_size = model["cell_size"]
    out = input_grid.copy()
    for r, c in sorted(model["actual_targets"]):
        r0, r1, c0, c1 = _pixel_cell_bounds(cell_size, r, c)
        out[r0:r1, c0:c1] = RED
    out[input_grid == GREY] = GREY
    out[input_grid == ORANGE] = ORANGE
    out[input_grid == BLUE] = BLUE
    return out


def _union_targets(cells: Iterable[tuple[int, int]], dirs: tuple[tuple[int, int], ...], n: int) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for cell in cells:
        targets |= _target_cells(cell, dirs, n)
    return targets


def _validate_model(model: dict, input_grid: np.ndarray, expected_grid: np.ndarray) -> bool:
    n = model["n"]
    dirs = model["dirs"]
    exemplar = model["exemplar"]
    template_cells = set(model["template_cells"])
    safe = _safe_cells(n, dirs)
    ordinary = {
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) != exemplar and (r, c) not in template_cells
    }

    counts = model["counts"]
    two_cells = {cell for cell in ordinary if int(counts[cell]) == 2}
    one_cells = {cell for cell in ordinary if int(counts[cell]) == 1}
    three_cells = {cell for cell in ordinary if int(counts[cell]) == 3}
    zero_cells = {cell for cell in ordinary if int(counts[cell]) == 0}

    if len(two_cells) < 2 or not one_cells or not three_cells:
        return False
    if any(cell not in safe for cell in two_cells):
        return False
    if not _is_asymmetric(dirs):
        return False

    actual = _union_targets(two_cells, dirs, n)
    ge2 = _union_targets({cell for cell in ordinary if int(counts[cell]) in (2, 3)}, dirs, n)
    nonzero = _union_targets({cell for cell in ordinary if int(counts[cell]) in (1, 2, 3)}, dirs, n)
    allordinary = _union_targets(ordinary, dirs, n)
    one_only = _union_targets(one_cells, dirs, n)
    three_only = _union_targets(three_cells, dirs, n)

    if not actual:
        return False
    if exemplar in actual:
        return False
    if ge2 == actual or nonzero == actual:
        return False
    if zero_cells and allordinary == actual:
        return False
    if not (one_only - actual):
        return False
    if not (three_only - actual):
        return False
    if len(actual - template_cells) < 2:
        return False
    if len(actual) > (n * n) // 2 + 2:
        return False
    if np.array_equal(input_grid, expected_grid):
        return False
    if int(np.count_nonzero(input_grid != expected_grid)) < model["cell_size"] ** 2:
        return False

    return True


def _attempt_model() -> dict | None:
    n = _choice(GRID_COUNTS)
    dirs = _choice(DIRECTION_SETS_BY_GRID[n])
    cell_size = _choice(CELL_SIZES)
    safe = _safe_cells(n, dirs)

    exemplar_candidates = []
    for exemplar in safe:
        template_cells = {(exemplar[0] + dr, exemplar[1] + dc) for dr, dc in dirs}
        ordinary_safe = safe - {exemplar} - template_cells
        ordinary = {
            (r, c)
            for r in range(n)
            for c in range(n)
            if (r, c) != exemplar and (r, c) not in template_cells
        }
        if len(ordinary_safe) >= 2 and len(ordinary) >= 4:
            exemplar_candidates.append((exemplar, template_cells, ordinary_safe, ordinary))
    if not exemplar_candidates:
        return None

    exemplar, template_cells, ordinary_safe, ordinary = _choice(exemplar_candidates)

    safe_list = list(ordinary_safe)
    perm = RNG.permutation(len(safe_list))
    safe_list = [safe_list[int(i)] for i in perm]
    max_two = min(4, len(safe_list))
    if max_two < 2:
        return None

    num_two = int(RNG.integers(2, max_two + 1))
    two_cells = set(safe_list[:num_two])
    decoy_pool = list(ordinary - two_cells)
    if len(decoy_pool) < 2:
        return None
    decoy_pool = [decoy_pool[int(i)] for i in RNG.permutation(len(decoy_pool))]
    one_cell = decoy_pool[0]
    three_cell = decoy_pool[1]

    counts = np.zeros((n, n), dtype=int)
    counts[one_cell] = 1
    counts[three_cell] = 3
    for cell in two_cells:
        counts[cell] = 2

    for cell in ordinary - {one_cell, three_cell} - two_cells:
        counts[cell] = int(RNG.choice([0, 1, 3], p=[0.50, 0.28, 0.22]))

    actual_targets = _union_targets(two_cells, dirs, n)
    model = {
        "n": n,
        "cell_size": cell_size,
        "dirs": dirs,
        "exemplar": exemplar,
        "template_cells": tuple(sorted(template_cells)),
        "counts": counts,
        "actual_targets": actual_targets,
    }
    return model


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        model = _attempt_model()
        if model is None:
            continue
        input_grid = _render_input(model)
        expected_grid = _render_expected(model, input_grid)
        if not _validate_model(model, input_grid, expected_grid):
            continue
        actual_grid = main(input_grid)
        if np.array_equal(actual_grid, expected_grid):
            return input_grid
    raise RuntimeError("failed to generate a valid counting-gated stamp instance")

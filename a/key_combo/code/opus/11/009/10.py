from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


PINK = 6
GREY = 5
YELLOW = 4
RANK_COLORS = (1, 2, 3, 7)
CELL_GRID = 5
INNER = 3
MASK_SIZE = 3


def _cell_spans_from_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    if not separators:
        raise AssertionError("missing separator lines")
    if any(b - a != 4 for a, b in zip(separators, separators[1:])):
        raise AssertionError("separator lines must be spaced by 4")

    bounds: list[int] = []
    if separators[0] != 0:
        bounds.append(-1)
    bounds.extend(separators)
    if separators[-1] != length - 1:
        bounds.append(length)

    spans = [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:]) if a + 1 <= b - 1]
    if len(spans) != CELL_GRID or any(hi - lo + 1 != INNER for lo, hi in spans):
        raise AssertionError("could not infer a 5x5 grid of 3x3 cells")
    return spans


def _infer_cell_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    row_counts = np.count_nonzero(g == PINK, axis=1)
    col_counts = np.count_nonzero(g == PINK, axis=0)
    sep_rows = [int(r) for r, count in enumerate(row_counts) if count > g.shape[1] // 2]
    sep_cols = [int(c) for c, count in enumerate(col_counts) if count > g.shape[0] // 2]
    return _cell_spans_from_separators(g.shape[0], sep_rows), _cell_spans_from_separators(g.shape[1], sep_cols)


def _extract_cells(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> tuple[dict[tuple[int, ...], np.ndarray], dict[tuple[int, ...], int], list[list[tuple[int, ...]]]]:
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    counts: dict[tuple[int, ...], int] = Counter()
    cell_keys: list[list[tuple[int, ...]]] = []
    for r0, r1 in row_spans:
        row_keys: list[tuple[int, ...]] = []
        for c0, c1 in col_spans:
            arr = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
            key = tuple(int(x) for x in arr.ravel())
            patterns.setdefault(key, arr.copy())
            counts[key] += 1
            row_keys.append(key)
        cell_keys.append(row_keys)
    return patterns, counts, cell_keys


def _sorted_new_positions(existing: set[tuple[int, int]], new_positions: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(
        new_positions,
        key=lambda pos: (
            min(abs(pos[0] - src[0]) + abs(pos[1] - src[1]) for src in existing),
            pos[0],
            pos[1],
        ),
    )


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    row_spans, col_spans = _infer_cell_grid(g)
    patterns, counts, cell_keys = _extract_cells(g, row_spans, col_spans)
    if len(counts) != 2:
        raise AssertionError("expected exactly two cell patterns")

    majority_key = max(counts, key=counts.get)
    minority_key = min(counts, key=counts.get)
    majority = patterns[majority_key]
    minority = patterns[minority_key]

    if not np.all(majority == GREY):
        raise AssertionError("majority cells must be solid grey")
    if not set(map(int, np.unique(minority))).issubset({GREY, YELLOW}):
        raise AssertionError("minority blueprint must use only grey and yellow")

    mask = minority == YELLOW
    active_count = int(mask.sum())
    if not (3 <= active_count <= 5):
        raise AssertionError("blueprint must contain 3-5 yellow pixels")

    existing = {
        (r, c)
        for r in range(CELL_GRID)
        for c in range(CELL_GRID)
        if cell_keys[r][c] == minority_key
    }
    if not (1 <= len(existing) <= 2):
        raise AssertionError("input must contain 1-2 existing minority cells")

    active_local = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
    for top in range(CELL_GRID - MASK_SIZE + 1):
        for left in range(CELL_GRID - MASK_SIZE + 1):
            active_global = {(top + r, left + c) for r, c in active_local}
            if existing.issubset(active_global):
                candidates.append((top, left, active_global))
    if len(candidates) != 1:
        raise AssertionError("blueprint placement must be uniquely determined")

    top, left, active_global = candidates[0]
    new_positions = active_global - existing
    if not (2 <= len(new_positions) <= 4):
        raise AssertionError("must have 2-4 new minority positions")

    ordered_new = _sorted_new_positions(existing, new_positions)
    return {
        "grid": g,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "majority": majority,
        "minority": minority,
        "mask": mask,
        "offset": (top, left),
        "existing": existing,
        "active_global": active_global,
        "new_positions": ordered_new,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the minority-cell blueprint onto the cell grid, then rank the newly revealed positions to choose their fill colors.

    Input structure:
    - A 5x5 array of 3x3 cells is divided by magenta separator lines.
    - Most cells are solid grey; 1-2 minority cells share one different 3x3 grey/yellow pattern.
    - The yellow pixels inside that minority pattern are a 3x3 blueprint to be located within the 5x5 cell array.

    Rule:
    - Infer the 5x5 cell layout and identify the majority and minority cell patterns.
    - Interpret the yellow pixels of the minority cell as active blueprint coordinates and find the unique 3x3 placement whose active cell positions contain all existing minority cells.
    - Every active blueprint position that is currently still a majority cell becomes a new minority cell.
    - Rank those new positions by Manhattan distance to the nearest existing minority cell, breaking ties by row then column; recolor the minority pattern's yellow pixels as blue, red, green, then orange by that rank while keeping its grey pixels unchanged.
    - Existing minority cells stay yellow, all other cells and separator lines stay unchanged.

    Key insight:
    - The minority cell serves two roles at once: its yellow pixels both encode the large-scale target positions and define the small-scale shape that must be copied into each new cell, while the final colors come from a separate distance-based ranking.
    """
    info = _analyze(grid)
    out = np.asarray(grid, dtype=int).copy()
    minority = np.asarray(info["minority"], dtype=int)
    row_spans = info["row_spans"]
    col_spans = info["col_spans"]

    for rank, (cell_r, cell_c) in enumerate(info["new_positions"]):
        patch = minority.copy()
        patch[patch == YELLOW] = RANK_COLORS[rank]
        r0, r1 = row_spans[cell_r]
        c0, c1 = col_spans[cell_c]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _make_case(mask: np.ndarray, offset: tuple[int, int], existing: tuple[tuple[int, int], ...]) -> dict[str, object] | None:
    active_local = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    active_global = {(offset[0] + r, offset[1] + c) for r, c in active_local}
    existing_set = set(existing)
    if not existing_set.issubset(active_global):
        return None
    if not (1 <= len(existing_set) <= 2):
        return None

    candidates = []
    for top in range(CELL_GRID - MASK_SIZE + 1):
        for left in range(CELL_GRID - MASK_SIZE + 1):
            probe = {(top + r, left + c) for r, c in active_local}
            if existing_set.issubset(probe):
                candidates.append((top, left))
    if len(candidates) != 1:
        return None

    new_positions = active_global - existing_set
    if not (2 <= len(new_positions) <= 4):
        return None

    ordered_new = _sorted_new_positions(existing_set, new_positions)
    distances = [
        min(abs(pos[0] - src[0]) + abs(pos[1] - src[1]) for src in existing_set)
        for pos in ordered_new
    ]
    return {
        "mask": mask.copy(),
        "offset": offset,
        "existing": tuple(sorted(existing_set)),
        "new_count": len(ordered_new),
        "existing_count": len(existing_set),
        "has_tie": len(distances) != len(set(distances)),
    }


def _build_case_bank() -> list[dict[str, object]]:
    bank: list[dict[str, object]] = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        active_count = bits.bit_count()
        if active_count < 3 or active_count > 5:
            continue
        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        for index in range(MASK_SIZE * MASK_SIZE):
            if (bits >> index) & 1:
                mask[index // MASK_SIZE, index % MASK_SIZE] = True
        local_positions = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        for top in range(CELL_GRID - MASK_SIZE + 1):
            for left in range(CELL_GRID - MASK_SIZE + 1):
                active_global = [(top + r, left + c) for r, c in local_positions]
                for existing_count in (1, 2):
                    if active_count - existing_count < 2:
                        continue
                    for sample in combinations(active_global, existing_count):
                        case = _make_case(mask, (top, left), sample)
                        if case is not None:
                            bank.append(case)
    return bank


CASE_BANK = _build_case_bank()


def _choose_case() -> dict[str, object]:
    want_tie = random.random() < 0.5
    want_orange = random.random() < 0.35
    want_existing = 1 if random.random() < 0.5 else 2

    filters = [
        lambda case: case["has_tie"] == want_tie and (case["new_count"] == 4) == want_orange and case["existing_count"] == want_existing,
        lambda case: case["has_tie"] == want_tie and (case["new_count"] == 4) == want_orange,
        lambda case: case["has_tie"] == want_tie,
        lambda case: True,
    ]
    for keep in filters:
        pool = [case for case in CASE_BANK if keep(case)]
        if pool:
            return random.choice(pool)
    raise AssertionError("empty case bank")


def _make_layout(with_border: bool) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    size = 21 if with_border else 19
    grid = np.full((size, size), GREY, dtype=int)
    if with_border:
        separators = [0, 4, 8, 12, 16, 20]
        spans = [(1, 3), (5, 7), (9, 11), (13, 15), (17, 19)]
    else:
        separators = [3, 7, 11, 15]
        spans = [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18)]
    grid[separators, :] = PINK
    grid[:, separators] = PINK
    return grid, spans, spans


def _validate_generated_input(grid: np.ndarray) -> None:
    info = _analyze(grid)
    out = main(grid)
    expected_colors = {GREY, PINK, YELLOW}
    for rank, (cell_r, cell_c) in enumerate(info["new_positions"]):
        r0, r1 = info["row_spans"][cell_r]
        c0, c1 = info["col_spans"][cell_c]
        cell = out[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.asarray(info["minority"]).copy()
        expected[expected == YELLOW] = RANK_COLORS[rank]
        if not np.array_equal(cell, expected):
            raise AssertionError("rank-colored output cell does not match the minority pattern")
        expected_colors.add(RANK_COLORS[rank])

    for cell_r, cell_c in info["existing"]:
        r0, r1 = info["row_spans"][cell_r]
        c0, c1 = info["col_spans"][cell_c]
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], info["minority"]):
            raise AssertionError("existing minority cells must remain yellow")

    if np.array_equal(grid, out):
        raise AssertionError("input must differ from output")
    if not set(map(int, np.unique(out))).issubset(expected_colors):
        raise AssertionError("unexpected output colors")


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _choose_case()
        grid, row_spans, col_spans = _make_layout(with_border=random.random() < 0.5)
        minority = np.full((INNER, INNER), GREY, dtype=int)
        minority[np.asarray(case["mask"], dtype=bool)] = YELLOW

        for cell_r, cell_c in case["existing"]:
            r0, r1 = row_spans[cell_r]
            c0, c1 = col_spans[cell_c]
            grid[r0 : r1 + 1, c0 : c1 + 1] = minority

        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")

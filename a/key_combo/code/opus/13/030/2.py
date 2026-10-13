from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PIVOT = 1
SOURCE = 6

ANGLE_BY_COUNT = {1: 1, 2: 2, 3: 3}  # CCW quarter-turns
COLOR_BY_COUNT = {1: 2, 2: 3, 3: 4}

# --------------- generator config ---------------

GRID_ROWS_RANGE = (13, 18)
GRID_COLS_RANGE = (13, 18)
PIVOT_MARGIN = 2
PIVOT_COUNT_VALUES = (1, 2, 3)
SOURCE_DISTANCES = (2, 3, 4)
MIN_INPUT_CHEBYSHEV = 2
MIN_OUTPUT_CHEBYSHEV = 2
PIVOT_MIN_CHEBYSHEV = 4
GENERATION_TRIES = 500

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


Coord = tuple[int, int]


def _coords(mask: np.ndarray) -> list[Coord]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _well_separated(cell: Coord, others: set[Coord] | list[Coord], min_gap: int) -> bool:
    return all(_chebyshev(cell, other) >= min_gap for other in others)


def _rotate_offset(dr: int, dc: int, quarter_turns: int) -> Coord:
    if quarter_turns == 1:
        return -dc, dr
    if quarter_turns == 2:
        return -dr, -dc
    if quarter_turns == 3:
        return dc, -dr
    raise ValueError(f"unsupported quarter_turns: {quarter_turns}")


def _source_to_pivot_map(sources: list[Coord], pivots: list[Coord]) -> tuple[dict[Coord, Coord], dict[Coord, int]]:
    mapping: dict[Coord, Coord] = {}
    counts = {pivot: 0 for pivot in pivots}
    for source in sources:
        aligned = [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
        assert len(aligned) == 1, "each source must align with exactly one pivot"
        pivot = aligned[0]
        mapping[source] = pivot
        counts[pivot] += 1
    return mapping, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each pivot's aligned source pixels, then rotate every source by the count-selected angle and recolor by the same count.

    Input structure:
    - The grid is a black background containing isolated blue pivot pixels and magenta source pixels.
    - Every magenta source lies in the same row or the same column as exactly one blue pivot.
    - A pivot may own 1, 2, or 3 magenta sources, and all sources owned by one pivot share that pivot.

    Rule:
    - For each magenta source, find its unique aligned blue pivot.
    - Count how many sources belong to each pivot.
    - Remove all magenta source pixels.
    - Rotate each source around its pivot, preserving offset length, with the pivot's shared count deciding the CCW angle:
      count 1 -> 90 degrees, count 2 -> 180 degrees, count 3 -> 270 degrees.
    - Draw the rotated in-bounds target using the color selected by the same count:
      count 1 -> red, count 2 -> green, count 3 -> yellow.
    - Keep all blue pivots unchanged.

    Key insight:
    - The same per-pivot source count controls two things at once: the common rotation angle for that pivot's whole group and the output color of every rotated source in that group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, PIVOT, SOURCE}, "unexpected colors"

    sources = _coords(g == SOURCE)
    pivots = _coords(g == PIVOT)
    assert pivots and sources, "expected pivots and sources"

    source_to_pivot, counts = _source_to_pivot_map(sources, pivots)

    out = g.copy()
    out[g == SOURCE] = BACKGROUND

    for source, pivot in source_to_pivot.items():
        k = counts[pivot]
        assert k in ANGLE_BY_COUNT and k in COLOR_BY_COUNT, "unsupported source count"

        dr = int(source[0] - pivot[0])
        dc = int(source[1] - pivot[1])
        rr, rc = _rotate_offset(dr, dc, ANGLE_BY_COUNT[k])
        target = (int(pivot[0] + rr), int(pivot[1] + rc))

        if _in_bounds(g.shape, *target):
            assert out[target] == BACKGROUND, "rotated targets must not collide"
            out[target] = COLOR_BY_COUNT[k]

    return out


def _sample_pivots(shape: tuple[int, int]) -> list[Coord] | None:
    h, w = shape
    rows = list(range(PIVOT_MARGIN, h - PIVOT_MARGIN))
    cols = list(range(PIVOT_MARGIN, w - PIVOT_MARGIN))
    if len(rows) < 3 or len(cols) < 3:
        return None

    cells = [(r, c) for r in rows for c in cols]
    random.shuffle(cells)
    pivots: list[Coord] = []
    for cell in cells:
        if any(cell[0] == pivot[0] or cell[1] == pivot[1] for pivot in pivots):
            continue
        if not _well_separated(cell, pivots, PIVOT_MIN_CHEBYSHEV):
            continue
        pivots.append(cell)
        if len(pivots) == 3:
            return pivots
    return None


def _expected_output(shape: tuple[int, int], pivot_counts: dict[Coord, int], assignments: list[tuple[Coord, Coord]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for pivot in pivot_counts:
        out[pivot] = PIVOT
    for source, pivot in assignments:
        k = pivot_counts[pivot]
        dr = int(source[0] - pivot[0])
        dc = int(source[1] - pivot[1])
        rr, rc = _rotate_offset(dr, dc, ANGLE_BY_COUNT[k])
        target = (int(pivot[0] + rr), int(pivot[1] + rc))
        assert _in_bounds(shape, *target)
        out[target] = COLOR_BY_COUNT[k]
    return out


def _candidate_placements(
    shape: tuple[int, int],
    pivots: list[Coord],
    pivot: Coord,
    pivot_count: int,
    used_sources: list[Coord],
    used_targets: set[Coord],
) -> list[tuple[Coord, Coord]]:
    used_inputs = set(pivots) | set(used_sources)
    candidates: list[tuple[Coord, Coord]] = []

    for distance in SOURCE_DISTANCES:
        for dr0, dc0 in DIRS4:
            dr = dr0 * distance
            dc = dc0 * distance
            source = (int(pivot[0] + dr), int(pivot[1] + dc))
            if not _in_bounds(shape, *source):
                continue
            if source in used_inputs or source in used_targets:
                continue
            if not _well_separated(source, used_inputs, MIN_INPUT_CHEBYSHEV):
                continue

            aligned = [other for other in pivots if other[0] == source[0] or other[1] == source[1]]
            if len(aligned) != 1 or aligned[0] != pivot:
                continue

            rr, rc = _rotate_offset(dr, dc, ANGLE_BY_COUNT[pivot_count])
            target = (int(pivot[0] + rr), int(pivot[1] + rc))
            if not _in_bounds(shape, *target):
                continue
            if target in used_inputs or target in used_targets:
                continue
            if not _well_separated(target, set(pivots) | used_targets, MIN_OUTPUT_CHEBYSHEV):
                continue

            candidates.append((source, target))

    random.shuffle(candidates)
    return candidates


def _place_sources(
    shape: tuple[int, int],
    pivots: list[Coord],
    pivot_counts: dict[Coord, int],
) -> list[tuple[Coord, Coord]] | None:
    slot_order: list[Coord] = []
    for pivot in sorted(pivots, key=lambda p: (-pivot_counts[p], p[0], p[1])):
        slot_order.extend([pivot] * pivot_counts[pivot])

    assignments: list[tuple[Coord, Coord]] = []
    used_sources: list[Coord] = []
    used_targets: set[Coord] = set()

    def search(i: int) -> bool:
        if i == len(slot_order):
            return True

        pivot = slot_order[i]
        pivot_count = pivot_counts[pivot]
        for source, target in _candidate_placements(shape, pivots, pivot, pivot_count, used_sources, used_targets):
            assignments.append((source, pivot))
            used_sources.append(source)
            used_targets.add(target)
            if search(i + 1):
                return True
            used_targets.remove(target)
            used_sources.pop()
            assignments.pop()
        return False

    if not search(0):
        return None
    return assignments


def _orientation_mix(assignments: list[tuple[Coord, Coord]]) -> bool:
    has_same_row = False
    has_same_col = False
    for source, pivot in assignments:
        if source[0] == pivot[0]:
            has_same_row = True
        if source[1] == pivot[1]:
            has_same_col = True
    return has_same_row and has_same_col


def _build_input(shape: tuple[int, int], pivots: list[Coord], assignments: list[tuple[Coord, Coord]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for pivot in pivots:
        grid[pivot] = PIVOT
    for source, _pivot in assignments:
        grid[source] = SOURCE
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1]),
        random.randint(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1]),
    )

    pivots = _sample_pivots(shape)
    if pivots is None:
        return None

    counts = list(PIVOT_COUNT_VALUES)
    random.shuffle(counts)
    pivot_counts = {pivot: count for pivot, count in zip(pivots, counts)}

    assignments = _place_sources(shape, pivots, pivot_counts)
    if assignments is None or not _orientation_mix(assignments):
        return None

    grid = _build_input(shape, pivots, assignments)
    expected = _expected_output(shape, pivot_counts, assignments)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

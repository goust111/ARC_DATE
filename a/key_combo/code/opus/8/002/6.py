from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORANGE_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _select_pivot(source: tuple[int, int], pivots: list[tuple[int, int]]) -> tuple[int, int]:
    sr, sc = source
    row_matches = [pivot for pivot in pivots if pivot[0] == sr]
    if row_matches:
        return min(row_matches, key=lambda pivot: abs(pivot[1] - sc))

    col_matches = [pivot for pivot in pivots if pivot[1] == sc]
    if col_matches:
        return min(col_matches, key=lambda pivot: abs(pivot[0] - sr))

    raise ValueError("each source must align with a pivot")


def _rotate_ccw_90(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _trace_rays(
    shape: tuple[int, int],
    orange_specs: list[tuple[tuple[int, int], int]],
    blocked: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    h, w = shape
    counts: dict[tuple[int, int], int] = {}

    for (r0, c0), step in orange_specs:
        for dr, dc in DIRS4:
            k = 1
            while True:
                r = r0 + dr * step * k
                c = c0 + dc * step * k
                if not (0 <= r < h and 0 <= c < w):
                    break
                pos = (r, c)
                if pos not in blocked:
                    counts[pos] = counts.get(pos, 0) + 1
                k += 1

    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each magenta source around its aligned blue pivot, then emit distance-stepped rays from the rotated orange point.

    Input structure:
    - A black background contains 2-3 blue pivot pixels and 3-4 magenta source pixels.
    - Every magenta source shares a row or a column with exactly one intended blue pivot.
    - Source-to-pivot distances vary across the set and stay small enough for the rotated target to remain in bounds.

    Rule:
    - For each magenta source, pair it with the blue pivot in the same row; if none exists, use the blue pivot in the same column.
    - Rotate the source offset around that pivot by 90 degrees counterclockwise, erase the magenta source, and place an orange pixel at the rotated target.
    - Let D be the source's Manhattan distance to its pivot. From the orange target, mark green pixels every D cells along the four cardinal directions until the grid edge.
    - Orange targets and blue pivots keep priority over rays, and any ray cell reached by two or more different orange sources becomes red.

    Key insight:
    - The pivot rotation determines both where each orange point lands and the step size that controls the later sparse ray pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("input must be a 2D grid")

    out = g.copy()
    pivots = [tuple(map(int, coord)) for coord in np.argwhere(g == BLUE)]
    sources = [tuple(map(int, coord)) for coord in np.argwhere(g == PINK)]

    orange_specs: list[tuple[tuple[int, int], int]] = []
    orange_cells: set[tuple[int, int]] = set()

    for source in sources:
        pivot = _select_pivot(source, pivots)
        sr, sc = source
        pr, pc = pivot
        dr, dc = sr - pr, sc - pc
        rr, rc = _rotate_ccw_90(dr, dc)
        target = (pr + rr, pc + rc)
        step = abs(dr) + abs(dc)
        if step <= 0:
            raise ValueError("source and pivot must differ")
        if not (0 <= target[0] < g.shape[0] and 0 <= target[1] < g.shape[1]):
            raise ValueError("rotated target must remain in bounds")

        out[sr, sc] = BLACK
        orange_specs.append((target, step))
        orange_cells.add(target)

    for r, c in orange_cells:
        out[r, c] = ORANGE

    blocked = set(pivots) | orange_cells
    ray_counts = _trace_rays(g.shape, orange_specs, blocked)
    for (r, c), count in ray_counts.items():
        out[r, c] = RED if count >= 2 else GREEN

    return out


def _inverse_source_offset(orange_dr: int, orange_dc: int) -> tuple[int, int]:
    return orange_dc, -orange_dr


def _choose_pivots(h: int, w: int, pivot_count: int) -> list[tuple[int, int]] | None:
    row_pool = list(range(3, h - 3))
    col_pool = list(range(3, w - 3))
    if len(row_pool) < pivot_count or len(col_pool) < pivot_count:
        return None

    for _ in range(80):
        rows = random.sample(row_pool, pivot_count)
        cols = random.sample(col_pool, pivot_count)
        pivots = list(zip(rows, cols))
        ok = True
        for i, (r1, c1) in enumerate(pivots):
            for r2, c2 in pivots[i + 1 :]:
                if abs(r1 - r2) < 2 or abs(c1 - c2) < 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return pivots
    return None


def _make_assignments(pivot_count: int, source_count: int) -> list[int]:
    assignments = list(range(pivot_count))
    while len(assignments) < source_count:
        assignments.append(random.randrange(pivot_count))
    random.shuffle(assignments)
    return assignments


def _choose_distances(source_count: int) -> list[int]:
    while True:
        distances = [random.randint(2, 4) for _ in range(source_count)]
        if len(set(distances)) >= 2:
            return distances


def _local_candidates(
    pivot_idx: int,
    distance: int,
    pivots: list[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    h, w = shape
    pr, pc = pivots[pivot_idx]
    pivot_rows = {r for r, _ in pivots}
    pivot_cols = {c for _, c in pivots}
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for orange_dr, orange_dc in ORANGE_DIRS:
        orange = (pr + orange_dr * distance, pc + orange_dc * distance)
        source_dr, source_dc = _inverse_source_offset(orange_dr * distance, orange_dc * distance)
        source = (pr + source_dr, pc + source_dc)

        if not (0 <= orange[0] < h and 0 <= orange[1] < w):
            continue
        if not (0 <= source[0] < h and 0 <= source[1] < w):
            continue
        if orange in pivots or source in pivots:
            continue

        sr, sc = source
        if sr == pr:
            if sc in pivot_cols:
                continue
        elif sc == pc:
            if sr in pivot_rows:
                continue
        else:
            continue

        candidates.append((source, orange, distance))

    random.shuffle(candidates)
    return candidates


def _input_cells_clear(candidate: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    cr, cc = candidate
    for r, c in occupied:
        if max(abs(cr - r), abs(cc - c)) <= 1:
            return False
    return True


def _search_specs(
    specs: list[tuple[int, int]],
    pivots: list[tuple[int, int]],
    shape: tuple[int, int],
    idx: int,
    placed: list[tuple[tuple[int, int], tuple[int, int], int, int]],
    used_sources: set[tuple[int, int]],
    used_oranges: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]] | None:
    if idx == len(specs):
        return placed.copy()

    pivot_idx, distance = specs[idx]
    occupied_inputs = set(pivots) | used_sources
    for source, orange, step in _local_candidates(pivot_idx, distance, pivots, shape):
        if source in used_sources or orange in used_oranges:
            continue
        if source == orange or source in used_oranges or orange in used_sources:
            continue
        if not _input_cells_clear(source, occupied_inputs):
            continue

        placed.append((source, orange, pivot_idx, step))
        used_sources.add(source)
        used_oranges.add(orange)
        result = _search_specs(specs, pivots, shape, idx + 1, placed, used_sources, used_oranges)
        if result is not None:
            return result
        placed.pop()
        used_sources.remove(source)
        used_oranges.remove(orange)

    return None


def _build_grid(
    shape: tuple[int, int],
    pivots: list[tuple[int, int]],
    placements: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in pivots:
        grid[r, c] = BLUE
    for source, _orange, _pivot_idx, _step in placements:
        grid[source] = PINK
    return grid


def _validate_layout(
    grid: np.ndarray,
    pivots: list[tuple[int, int]],
    placements: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> bool:
    expected_pairs = {source: pivots[pivot_idx] for source, _orange, pivot_idx, _step in placements}
    for source, pivot in expected_pairs.items():
        if _select_pivot(source, pivots) != pivot:
            return False

    try:
        out = main(grid)
    except Exception:
        return False

    if np.any(grid == BLACK) and not np.any(out == ORANGE):
        return False

    density = np.count_nonzero(out) / out.size
    return density <= 0.45


def _attempt_generate(require_overlap: bool | None) -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    pivot_count = random.randint(2, 3)
    source_count = random.randint(3, 4)
    pivots = _choose_pivots(h, w, pivot_count)
    if pivots is None:
        return None

    assignments = _make_assignments(pivot_count, source_count)
    distances = _choose_distances(source_count)
    specs = list(zip(assignments, distances))
    random.shuffle(specs)

    placements = _search_specs(specs, pivots, (h, w), 0, [], set(), set())
    if placements is None:
        return None

    grid = _build_grid((h, w), pivots, placements)
    if not _validate_layout(grid, pivots, placements):
        return None

    out = main(grid)
    has_overlap = bool(np.any(out == RED))
    if require_overlap is True and not has_overlap:
        return None
    if require_overlap is False and has_overlap:
        return None
    return grid


def generate_input() -> np.ndarray:
    overlap_modes = [True, False, None]
    for _ in range(400):
        grid = _attempt_generate(random.choice(overlap_modes))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

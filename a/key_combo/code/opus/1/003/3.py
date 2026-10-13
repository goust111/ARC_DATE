from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

SIDE_CHOICES = (11, 13, 15, 17, 19)
COLOR_COUNT_CHOICES = (3, 4)
COUNT_VALUES = (1, 2, 3, 4, 5)
TIE_CASE_PROB = 0.65
NON_RING_BIAS = 0.75
GENERATION_RETRIES = 400

_NEIGHBOR_OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
)


def _max_layers(shape: tuple[int, int]) -> int:
    h, w = shape
    return min((h - 1) // 4 + 1, (w - 1) // 4 + 1)


def _ring_cells(shape: tuple[int, int], layer: int) -> set[tuple[int, int]]:
    h, w = shape
    offset = 2 * layer
    r0, c0 = offset, offset
    r1, c1 = h - 1 - offset, w - 1 - offset
    assert r0 <= r1 and c0 <= c1

    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _paint_ring(grid: np.ndarray, color: int, layer: int) -> None:
    for r, c in _ring_cells(grid.shape, layer):
        grid[r, c] = int(color)


def _color_stats(grid: np.ndarray) -> list[tuple[int, int, int]]:
    stats: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG_COLOR:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        count = int(coords.shape[0])
        top_row = int(coords[:, 0].min())
        stats.append((color, count, top_row))
    return stats


def _rank_colors(grid: np.ndarray) -> list[int]:
    stats = _color_stats(grid)
    assert stats, "expected at least one marker color"
    rank_keys = [(count, top_row) for _color, count, top_row in stats]
    assert len(set(rank_keys)) == len(rank_keys), "ranking keys must be unique"
    stats.sort(key=lambda item: (-item[1], item[2]))
    return [color for color, _count, _top_row in stats]


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    ranked_colors = _rank_colors(g)
    assert len(ranked_colors) <= _max_layers(g.shape), "too many colors for the available nested layers"

    out = np.full_like(g, BG_COLOR)
    for layer, color in enumerate(ranked_colors):
        _paint_ring(out, color, layer)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank scattered marker colors by population, then draw centered nested rectangular borders in that order.

    Input structure:
    - The background is black.
    - The foreground consists of isolated single-pixel markers in 3 or 4 colors on an odd-sized grid.
    - The color populations determine the frame order; if two colors have the same population, the one whose topmost marker is higher ranks first.

    Rule:
    - Count how many markers each non-black color has.
    - Sort colors by descending count, breaking ties by the smallest row index among that color's markers.
    - Draw one-cell-thick rectangular rings from the outside inward, leaving a one-cell black gap between consecutive rings.
    - Erase all original markers unless a ring is drawn on that cell, in which case the ring color remains.

    Key insight:
    - Marker positions do not tell you where to draw; they only encode a global color ranking that becomes the outer-to-inner nesting order of the final frame structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _expected_output(g)


def _sample_rank_counts(color_count: int) -> list[int]:
    if color_count > len(COUNT_VALUES):
        raise ValueError("count palette is too small")

    if random.random() >= TIE_CASE_PROB:
        return sorted(random.sample(COUNT_VALUES, color_count), reverse=True)

    for _ in range(100):
        pair_index = random.randint(0, color_count - 2)
        duplicate = random.choice(COUNT_VALUES)
        higher_pool = [value for value in COUNT_VALUES if value > duplicate]
        lower_pool = [value for value in COUNT_VALUES if value < duplicate]
        need_higher = pair_index
        need_lower = color_count - pair_index - 2
        if len(higher_pool) < need_higher or len(lower_pool) < need_lower:
            continue

        higher = sorted(random.sample(higher_pool, need_higher), reverse=True)
        lower = sorted(random.sample(lower_pool, need_lower), reverse=True)
        return higher + [duplicate, duplicate] + lower

    return sorted(random.sample(COUNT_VALUES, color_count), reverse=True)


def _is_adjacent(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for dr, dc in _NEIGHBOR_OFFSETS:
        if (r + dr, c + dc) in occupied:
            return True
    return False


def _choose_spaced_cells(
    candidates: list[tuple[int, int]],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    random.shuffle(candidates)
    for cell in candidates:
        if _is_adjacent(cell, occupied | set(chosen)):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _available_cells(
    side: int,
    occupied: set[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(side):
        if r in used_rows:
            continue
        for c in range(side):
            if c in used_cols:
                continue
            cell = (r, c)
            if cell in occupied or _is_adjacent(cell, occupied):
                continue
            cells.append(cell)
    return cells


def _build_grid(
    side: int,
    color_counts: dict[int, int],
    fixed_jobs: list[tuple[int, tuple[int, int]]],
    ring_union: set[tuple[int, int]],
) -> np.ndarray | None:
    placements = {color: [] for color in color_counts}
    used_rows = {color: set() for color in color_counts}
    used_cols = {color: set() for color in color_counts}
    occupied: set[tuple[int, int]] = set()

    def place(color: int, cell: tuple[int, int]) -> bool:
        r, c = cell
        if cell in occupied or _is_adjacent(cell, occupied):
            return False
        if r in used_rows[color] or c in used_cols[color]:
            return False
        placements[color].append(cell)
        used_rows[color].add(r)
        used_cols[color].add(c)
        occupied.add(cell)
        return True

    for color, cell in fixed_jobs:
        if not place(color, cell):
            return None

    colors = list(color_counts)
    random.shuffle(colors)
    colors.sort(key=lambda color: color_counts[color] - len(placements[color]), reverse=True)

    for color in colors:
        remaining = color_counts[color] - len(placements[color])
        for _ in range(remaining):
            candidates = _available_cells(side, occupied, used_rows[color], used_cols[color])
            if not candidates:
                return None

            non_ring = [cell for cell in candidates if cell not in ring_union]
            pool = non_ring if non_ring and random.random() < NON_RING_BIAS else candidates
            if not pool:
                pool = candidates
            if not pool:
                return None

            cell = random.choice(pool)
            if not place(color, cell):
                return None

    grid = np.full((side, side), BG_COLOR, dtype=int)
    for color, cells in placements.items():
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _input_contract_holds(grid: np.ndarray, expect_tie: bool) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG_COLOR]
    if not 3 <= len(colors) <= 4:
        return False

    stats = _color_stats(grid)
    keys = [(count, top_row) for _color, count, top_row in stats]
    if len(set(keys)) != len(keys):
        return False

    counts = [count for _color, count, _top_row in stats]
    duplicated_count_exists = len(set(counts)) < len(counts)
    if duplicated_count_exists != expect_tie:
        return False

    marker_cells = list(map(tuple, np.argwhere(grid != BG_COLOR)))
    occupied = set(marker_cells)
    for color in colors:
        coords = np.argwhere(grid == color)
        rows = [int(r) for r, _c in coords]
        cols = [int(c) for _r, c in coords]
        if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
            return False
    for cell in marker_cells:
        r, c = cell
        for dr, dc in _NEIGHBOR_OFFSETS:
            if dr == 0 and dc == 0:
                continue
            if (r + dr, c + dc) in occupied:
                return False

    ring_union = set()
    for layer in range(len(colors)):
        ring_union |= _ring_cells(grid.shape, layer)
    on_ring = any(cell in ring_union for cell in marker_cells)
    off_ring = any(cell not in ring_union for cell in marker_cells)
    return on_ring and off_ring


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDE_CHOICES)
    max_colors = min(4, _max_layers((side, side)))
    color_count = random.choice([count for count in COLOR_COUNT_CHOICES if count <= max_colors])
    colors = random.sample(PALETTE, color_count)

    rank_counts = _sample_rank_counts(color_count)
    expect_tie = len(set(rank_counts)) < len(rank_counts)
    shuffled_counts = rank_counts[:]
    random.shuffle(shuffled_counts)
    color_counts = dict(zip(colors, shuffled_counts))

    ring_union: set[tuple[int, int]] = set()
    for layer in range(color_count):
        ring_union |= _ring_cells((side, side), layer)
    non_ring = [
        (r, c)
        for r in range(side)
        for c in range(side)
        if (r, c) not in ring_union
    ]
    if not non_ring:
        return None

    occupied: set[tuple[int, int]] = set()
    ring_job_count = 1 + int(color_count > 1 and random.random() < 0.5)
    ring_specials = _choose_spaced_cells(list(ring_union), ring_job_count, occupied)
    if ring_specials is None:
        return None
    occupied.update(ring_specials)

    inside_special = _choose_spaced_cells(non_ring[:], 1, occupied)
    if inside_special is None:
        return None
    occupied.update(inside_special)

    fixed_colors = random.sample(colors, ring_job_count + 1)
    fixed_jobs = [
        (fixed_colors[idx], cell)
        for idx, cell in enumerate(ring_specials)
    ]
    fixed_jobs.append((fixed_colors[-1], inside_special[0]))

    grid = _build_grid(side, color_counts, fixed_jobs, ring_union)
    if grid is None:
        return None
    if not _input_contract_holds(grid, expect_tie):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, _expected_output(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame counting input")

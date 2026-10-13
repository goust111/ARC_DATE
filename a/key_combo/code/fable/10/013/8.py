from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
BACKGROUND_POOL = [0, 2, 3, 4, 5, 6, 7, 8, 9]
SOURCE_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (10, 19)
GRID_WIDTH_RANGE = (10, 19)
SOURCE_COUNT_CHOICES = [2, 3, 4]
SOURCE_COUNT_WEIGHTS = [0.25, 0.35, 0.40]
MAX_GENERATION_ATTEMPTS = 600


def _dominant_background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= r < height and 0 <= c < width


def _extract_sources(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg = _dominant_background(g)
    assert bg != BLUE, "blue must stay available for generated intersections"

    coords = np.argwhere(g != bg)
    assert 2 <= coords.shape[0] <= 4, "expected two to four source pixels"

    colors = [int(g[r, c]) for r, c in coords]
    assert BLUE not in colors, "source colors must not use blue"
    assert len(set(colors)) == len(colors), "source colors must be distinct"

    sources = []
    seen_diffs = set()
    seen_sums = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != bg:
                raise AssertionError("source pixels must be isolated singletons")

        diff = rr - cc
        summ = rr + cc
        assert diff not in seen_diffs, "two sources cannot share an NW-SE diagonal"
        assert summ not in seen_sums, "two sources cannot share an NE-SW diagonal"
        seen_diffs.add(diff)
        seen_sums.add(summ)
        sources.append((rr, cc, int(g[rr, cc])))

    return bg, sources


def _blue_cells_from_pairwise(
    shape: tuple[int, int], sources: list[tuple[int, int, int]]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], set[int]]]:
    blue_cells: set[tuple[int, int]] = set()
    contributors: dict[tuple[int, int], set[int]] = {}

    for idx_a in range(len(sources)):
        ra, ca, _ = sources[idx_a]
        da = ra - ca
        sa = ra + ca
        pa = (ra + ca) % 2
        for idx_b in range(idx_a + 1, len(sources)):
            rb, cb, _ = sources[idx_b]
            pb = (rb + cb) % 2
            if pa != pb:
                continue

            db = rb - cb
            sb = rb + cb
            candidates = [
                ((da + sb) // 2, (sb - da) // 2),
                ((db + sa) // 2, (sa - db) // 2),
            ]
            for cell in candidates:
                if not _in_bounds(shape, cell[0], cell[1]):
                    continue
                if cell == (ra, ca) or cell == (rb, cb):
                    continue
                blue_cells.add(cell)
                contributors.setdefault(cell, set()).update({idx_a, idx_b})

    return blue_cells, contributors


def _visible_blue_by_scan(shape: tuple[int, int], sources: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    source_positions = {(r, c) for r, c, _ in sources}
    blue_cells: set[tuple[int, int]] = set()
    height, width = shape
    for r in range(height):
        for c in range(width):
            if (r, c) in source_positions:
                continue
            hit_count = 0
            for sr, sc, _ in sources:
                if abs(r - sr) == abs(c - sc):
                    hit_count += 1
            if hit_count >= 2:
                blue_cells.add((r, c))
    return blue_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the intersections of the invisible diagonal rays emitted by the isolated source pixels.

    Input structure:
    - The grid has one uniform dominant background color.
    - There are two to four isolated single-pixel sources, each with a different non-blue color.
    - No two sources share either diagonal direction, so no full diagonal line is duplicated.

    Rule:
    - Imagine every source pixel extending four diagonal rays to the grid boundary.
    - For each non-source cell, count how many different sources have a ray passing through it.
    - Color the cell blue if at least two different sources reach it; keep source pixels and all other cells unchanged.

    Key insight:
    - The visible blue pixels are not drawn rays; they are only the cells where invisible diagonal lines from different sources intersect.
    """
    g = np.asarray(grid, dtype=int)
    bg, sources = _extract_sources(g)
    out = g.copy()
    source_positions = {(r, c) for r, c, _ in sources}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if (r, c) in source_positions:
                continue
            hit_count = 0
            for sr, sc, _ in sources:
                if abs(r - sr) == abs(c - sc):
                    hit_count += 1
            if hit_count >= 2:
                out[r, c] = BLUE
            else:
                out[r, c] = bg

    return out


def _parity_pattern(source_count: int) -> list[int]:
    base = random.randint(0, 1)
    if source_count == 2:
        return [base, base]
    if source_count == 3:
        return [base, base, base]
    if random.random() < 0.6:
        return [base, base, 1 - base, 1 - base]
    return [base, base, base, base]


def _candidate_cells(
    shape: tuple[int, int],
    parity: int,
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = shape
    diffs = {r - c for r, c in chosen}
    sums = {r + c for r, c in chosen}
    cells = []
    for r in range(height):
        for c in range(width):
            if (r + c) % 2 != parity:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            if (r - c) in diffs or (r + c) in sums:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _sample_positions(shape: tuple[int, int], parities: list[int]) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    order = list(range(len(parities)))
    random.shuffle(order)
    placed: list[tuple[int, int] | None] = [None] * len(parities)

    for idx in order:
        cells = _candidate_cells(shape, parities[idx], chosen)
        if not cells:
            return None
        cell = cells[0]
        chosen.append(cell)
        placed[idx] = cell

    return [cell for cell in placed if cell is not None]


def _render_input(
    shape: tuple[int, int],
    bg: int,
    positions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color
    return grid


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    bg, sources = _extract_sources(grid)
    blue_cells, contributors = _blue_cells_from_pairwise(grid.shape, sources)
    if blue_cells != _visible_blue_by_scan(grid.shape, sources):
        return False

    if not blue_cells or len(blue_cells) < 2 or len(blue_cells) > 8:
        return False

    source_positions = {(r, c) for r, c, _ in sources}
    per_source_hits = Counter()
    for source_ids in contributors.values():
        if len(source_ids) > 2:
            return False
        for idx in source_ids:
            per_source_hits[idx] += 1
    if any(per_source_hits[idx] == 0 for idx in range(len(sources))):
        return False

    parities = [(r + c) % 2 for r, c, _ in sources]
    parity_counts = Counter(parities)
    if any(count == 1 for count in parity_counts.values()):
        return False

    if len(sources) == 4 and len(parity_counts) == 2:
        same_pairs = sum(count * (count - 1) // 2 for count in parity_counts.values())
        diff_pairs = parity_counts[0] * parity_counts[1]
        if same_pairs == 0 or diff_pairs == 0:
            return False

    single_ray_cells = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) in source_positions or (r, c) in blue_cells:
                continue
            hit_count = sum(abs(r - sr) == abs(c - sc) for sr, sc, _ in sources)
            if hit_count == 1:
                single_ray_cells += 1
    if single_ray_cells <= len(blue_cells):
        return False

    blue_positions = set(map(tuple, np.argwhere(expected == BLUE)))
    if blue_positions != blue_cells:
        return False
    if any(int(expected[r, c]) != int(grid[r, c]) for r, c in source_positions):
        return False
    if np.any((expected != bg) & (expected != BLUE) & (grid == bg)):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    shape = (height, width)

    source_count = random.choices(SOURCE_COUNT_CHOICES, weights=SOURCE_COUNT_WEIGHTS, k=1)[0]
    parities = _parity_pattern(source_count)
    positions = _sample_positions(shape, parities)
    if positions is None:
        return None

    bg = random.choice(BACKGROUND_POOL)
    color_pool = [color for color in SOURCE_COLOR_POOL if color not in {bg, BLUE}]
    if len(color_pool) < source_count:
        return None
    colors = random.sample(color_pool, source_count)
    random.shuffle(colors)

    grid = _render_input(shape, bg, positions, colors)
    try:
        expected = np.full(shape, bg, dtype=int)
        for (r, c), color in zip(positions, colors):
            expected[r, c] = color
        blue_cells, _contributors = _blue_cells_from_pairwise(shape, [(r, c, color) for (r, c), color in zip(positions, colors)])
        for r, c in blue_cells:
            expected[r, c] = BLUE
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-diagonal-intersection puzzle")

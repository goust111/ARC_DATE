from __future__ import annotations

import random

import numpy as np


GRID_SIZE = 9
TILE_SIZE = 3
PALETTE = (1, 2, 3, 4)
COUNT_PATTERNS = (
    (1, 2, 6),
    (1, 3, 5),
    (2, 3, 4),
    (1, 2, 3, 3),
)
MAX_HOLES = 24
GEN_ATTEMPTS = 400


def _candidate_periods(length: int) -> list[int]:
    return [p for p in range(1, length + 1) if length % p == 0]


def _tile_for_period(grid: np.ndarray, period: tuple[int, int]) -> np.ndarray | None:
    pr, pc = period
    tile = np.full((pr, pc), -1, dtype=int)
    for r, c in np.argwhere(grid != 0):
        rr = int(r % pr)
        cc = int(c % pc)
        color = int(grid[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_min_period_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in _candidate_periods(h):
        for pc in _candidate_periods(w):
            tile = _tile_for_period(grid, (pr, pc))
            if tile is not None:
                candidates.append((pr * pc, pr, pc, tile))
    assert candidates, "no valid periodic tile found"
    min_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1, "expected a unique smallest period"
    return best[0]


def _repair_grid(grid: np.ndarray, tile: np.ndarray) -> np.ndarray:
    out = grid.copy()
    th, tw = tile.shape
    for r, c in np.argwhere(out == 0):
        out[r, c] = tile[int(r % th), int(c % tw)]
    return out


def _least_frequent_color(tile: np.ndarray) -> int:
    colors, counts = np.unique(tile, return_counts=True)
    least = int(counts.min())
    least_colors = colors[counts == least]
    assert least_colors.size == 1, "expected a unique least frequent color"
    return int(least_colors[0])


def _stamp_output(tile: np.ndarray, marker_color: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.zeros((th * th, tw * tw), dtype=int)
    for tr, tc in np.argwhere(tile == marker_color):
        r0 = int(tr) * th
        c0 = int(tc) * tw
        out[r0 : r0 + th, c0 : c0 + tw] = tile
    return out


def _tile_color_counts(tile: np.ndarray) -> dict[int, int]:
    colors, counts = np.unique(tile, return_counts=True)
    return {int(color): int(count) for color, count in zip(colors, counts)}


def _full_grid_from_tile(tile: np.ndarray) -> np.ndarray:
    return np.tile(tile, (GRID_SIZE // TILE_SIZE, GRID_SIZE // TILE_SIZE))


def _has_smaller_period(tile: np.ndarray) -> bool:
    full = _full_grid_from_tile(tile)
    for pr in (1, TILE_SIZE):
        for pc in (1, TILE_SIZE):
            if pr == TILE_SIZE and pc == TILE_SIZE:
                continue
            if _tile_for_period(full, (pr, pc)) is not None:
                return True
    return False


def _sorted_colors_by_tile_count(tile: np.ndarray) -> list[int]:
    counts = _tile_color_counts(tile)
    ordered = sorted(counts.items(), key=lambda item: item[1])
    return [color for color, _count in ordered]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the hidden 3x3 periodic tile, then stamp it into blocks selected by the repaired tile's least frequent color.

    Input structure:
    - The input is a 9x9 grid built from a hidden 3x3 periodic tile repeated 3x3 times.
    - Several cells have been replaced by black, but the remaining non-black cells are still consistent with one unique smallest 3x3 period.
    - The repaired 3x3 tile uses 3-4 non-black colors and has one uniquely least frequent color.

    Rule:
    - Infer the unique smallest periodic tile from the non-black cells.
    - Replace every black cell by the color from the matching tile residue to recover the full periodic pattern.
    - Find the uniquely least frequent color inside the repaired 3x3 tile.
    - Create a black 9x9 output and, for each least-frequency tile cell, copy the whole 3x3 tile into the corresponding 3x3 output block.

    Key insight:
    - The black cells hide frequency evidence, so the rare-color mask is only visible after periodic repair; that repaired mask then selects where the whole tile must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"

    pr, pc, tile = _infer_min_period_tile(g)
    assert (pr, pc) == (TILE_SIZE, TILE_SIZE), "expected a unique 3x3 minimal period"
    repaired = _repair_grid(g, tile)
    marker_color = _least_frequent_color(tile)

    expected_repaired = _full_grid_from_tile(tile)
    assert np.array_equal(repaired, expected_repaired), "repair must recreate the periodic grid"
    return _stamp_output(tile, marker_color)


def _sample_tile() -> np.ndarray | None:
    for _ in range(120):
        pattern = random.choice(COUNT_PATTERNS)
        colors = random.sample(PALETTE, len(pattern))
        cells: list[int] = []
        for color, count in zip(colors, pattern):
            cells.extend([color] * count)
        random.shuffle(cells)
        tile = np.array(cells, dtype=int).reshape(TILE_SIZE, TILE_SIZE)
        if not _has_smaller_period(tile):
            return tile
    return None


def _positions_for_color(tile: np.ndarray, color: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tr, tc in np.argwhere(tile == color):
        key = (int(tr), int(tc))
        groups[key] = [
            (r, c)
            for r in range(int(tr), GRID_SIZE, TILE_SIZE)
            for c in range(int(tc), GRID_SIZE, TILE_SIZE)
        ]
    return groups


def _erase_color_instances(grid: np.ndarray, tile: np.ndarray, color: int, n_remove: int) -> None:
    groups = _positions_for_color(tile, color)
    capacity = sum(len(positions) - 1 for positions in groups.values())
    assert n_remove <= capacity, "cannot erase that many cells without hiding a residue completely"

    for positions in groups.values():
        random.shuffle(positions)

    removed = 0
    while removed < n_remove:
        eligible = [positions for positions in groups.values() if len(positions) > 1]
        assert eligible, "ran out of erasable positions"
        positions = random.choice(eligible)
        r, c = positions.pop()
        grid[r, c] = 0
        removed += 1


def _visible_unique_least_color(grid: np.ndarray) -> int | None:
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    if colors.size == 0:
        return None
    least = int(counts.min())
    least_colors = colors[counts == least]
    if least_colors.size != 1:
        return None
    return int(least_colors[0])


def _expected_output(tile: np.ndarray) -> np.ndarray:
    return _stamp_output(tile, _least_frequent_color(tile))


def _gen_attempt() -> np.ndarray | None:
    tile = _sample_tile()
    if tile is None:
        return None

    tile_counts = _tile_color_counts(tile)
    ordered_colors = _sorted_colors_by_tile_count(tile)
    rare_color = ordered_colors[0]
    decoy_color = ordered_colors[1]

    rare_full = tile_counts[rare_color] * (GRID_SIZE // TILE_SIZE) ** 2
    decoy_full = tile_counts[decoy_color] * (GRID_SIZE // TILE_SIZE) ** 2
    min_decoy_visible = tile_counts[decoy_color]
    max_margin = min(4, rare_full - min_decoy_visible)
    if max_margin < 1:
        return None
    margin = random.randint(1, max_margin)
    decoy_visible_goal = rare_full - margin
    decoy_remove = decoy_full - decoy_visible_goal
    if decoy_remove > MAX_HOLES:
        return None

    grid = _full_grid_from_tile(tile)
    _erase_color_instances(grid, tile, decoy_color, decoy_remove)

    remaining_holes = MAX_HOLES - decoy_remove
    extra_candidates = ordered_colors[2:]
    if extra_candidates and remaining_holes > 0:
        extra_color = random.choice(extra_candidates)
        extra_full = tile_counts[extra_color] * (GRID_SIZE // TILE_SIZE) ** 2
        extra_floor = rare_full
        extra_cap = min(
            remaining_holes,
            extra_full - tile_counts[extra_color],
            extra_full - extra_floor,
            3,
        )
        if extra_cap > 0:
            _erase_color_instances(grid, tile, extra_color, random.randint(0, extra_cap))

    visible_least = _visible_unique_least_color(grid)
    if visible_least != decoy_color:
        return None
    if int(np.sum(grid == 0)) > MAX_HOLES:
        return None

    expected = _expected_output(tile)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
TEAL = 8
MAX_ATTEMPTS = 800
LAYOUT_CHOICES = ((1, 5), (5, 1), (3, 3))


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _extract_tiles(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, RED}, "input must use black and red only"

    row_runs = _active_runs(np.any(g != BG, axis=1))
    col_runs = _active_runs(np.any(g != BG, axis=0))
    assert row_runs and col_runs, "expected a separated tile array"

    tile_hs = {r1 - r0 + 1 for r0, r1 in row_runs}
    tile_ws = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(tile_hs) == 1 and len(tile_ws) == 1, "all tiles must share one size"
    tile_h = next(iter(tile_hs))
    tile_w = next(iter(tile_ws))
    assert tile_h >= 2 and tile_w >= 2, "tile size must be at least 2x2"

    windows: list[tuple[int, int, int, int]] = []
    tiles: list[np.ndarray] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = tile == RED
            assert np.any(mask), "tile must be non-empty"
            assert not np.all(mask), "tile must not be full"
            windows.append((r0, r1, c0, c1))
            tiles.append(mask.copy())

    assert len(tiles) % 2 == 1 and len(tiles) >= 3, "must have an odd number of tiles"
    return tiles, windows


def _is_partner(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    comp = ~mask_a
    return bool(
        np.array_equal(mask_b, comp) or np.array_equal(mask_b, np.rot90(comp, 2))
    )


def _needs_rotation(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    comp = ~mask_a
    return bool(
        not np.array_equal(mask_b, comp) and np.array_equal(mask_b, np.rot90(comp, 2))
    )


def _partner_graph(masks: list[np.ndarray]) -> list[list[int]]:
    graph = [[] for _ in masks]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _is_partner(masks[i], masks[j]):
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _mixed_mask(mask: np.ndarray) -> bool:
    return bool(
        np.all(mask.any(axis=1))
        and np.all((~mask).any(axis=1))
        and np.all(mask.any(axis=0))
        and np.all((~mask).any(axis=0))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique unpaired mask tile under complementary pairing with optional 180-degree rotation and complete its missing partner.

    Input structure:
    - A black background is split by full blank rows and columns into an array of equal-size rectangular tiles.
    - Each tile contains a red mask only, and every tile is non-empty and not fully filled.
    - All tiles can be paired by complementary union, except for exactly one leftover tile.

    Rule:
    - Two tiles are partners exactly when one tile, kept as is or rotated by 180 degrees, has a red-mask complement that matches the other tile cell by cell.
    - Build all such tile pairs and identify the unique tile with no partner.
    - Output one tile-sized rectangle where the leftover tile's original red cells stay red and every other cell is colored teal.

    Key insight:
    - The decisive relation is not equality but "disjoint union fills the whole tile", and some valid partners appear only after a 180-degree rotation before testing complementarity.
    """
    masks, _windows = _extract_tiles(grid)
    graph = _partner_graph(masks)
    degrees = [len(neighbors) for neighbors in graph]

    leftovers = [idx for idx, degree in enumerate(degrees) if degree == 0]
    assert len(leftovers) == 1, "expected exactly one unpaired tile"
    assert all(degree == 1 for idx, degree in enumerate(degrees) if idx != leftovers[0]), (
        "every non-leftover tile must have exactly one partner"
    )

    leftover = masks[leftovers[0]]
    out = np.full(leftover.shape, TEAL, dtype=int)
    out[leftover] = RED
    return out


def _random_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    cells = [(r, c) for r in range(height) for c in range(width)]
    min_count = max(height + width - 2, area // 3)
    max_count = min(area - max(height, width), (2 * area) // 3)
    for _ in range(200):
        count = random.randint(min_count, max_count)
        chosen = set(random.sample(cells, count))
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if _mixed_mask(mask):
            return mask
    raise RuntimeError("failed to sample a mixed tile mask")


def _make_pair(mask: np.ndarray, rotated: bool) -> tuple[np.ndarray, np.ndarray]:
    partner = np.rot90(~mask, 2) if rotated else ~mask
    return mask.copy(), partner.copy()


def _build_grid(
    masks: list[np.ndarray], layout: tuple[int, int], gap: int, margins: tuple[int, int]
) -> np.ndarray:
    rows, cols = layout
    tile_h, tile_w = masks[0].shape
    top, left = margins
    grid_h = top * 2 + rows * tile_h + (rows - 1) * gap
    grid_w = left * 2 + cols * tile_w + (cols - 1) * gap
    grid = np.zeros((grid_h, grid_w), dtype=int)
    shuffled = masks[:]
    random.shuffle(shuffled)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = top + rr * (tile_h + gap)
            c0 = left + cc * (tile_w + gap)
            patch = grid[r0 : r0 + tile_h, c0 : c0 + tile_w]
            patch[shuffled[idx]] = RED
            idx += 1
    return grid


def _expected_output(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, TEAL, dtype=int)
    out[mask] = RED
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        masks, _windows = _extract_tiles(grid)
    except AssertionError:
        return False

    graph = _partner_graph(masks)
    degrees = [len(neighbors) for neighbors in graph]
    leftovers = [idx for idx, degree in enumerate(degrees) if degree == 0]
    if len(leftovers) != 1:
        return False
    if any(degree != 1 for idx, degree in enumerate(degrees) if idx != leftovers[0]):
        return False

    pair_edges = [(i, neighbors[0]) for i, neighbors in enumerate(graph) if len(neighbors) == 1 and i < neighbors[0]]
    if not pair_edges:
        return False
    if not any(_needs_rotation(masks[i], masks[j]) for i, j in pair_edges):
        return False
    if len(pair_edges) > 1 and all(_needs_rotation(masks[i], masks[j]) for i, j in pair_edges):
        return False

    counts = [int(mask.sum()) for mask in masks]
    unmatched_count = counts[leftovers[0]]
    if Counter(counts)[unmatched_count] < 2:
        return False
    if unmatched_count == min(counts) or unmatched_count == max(counts):
        return False

    expected = _expected_output(masks[leftovers[0]])
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(set(map(int, np.unique(actual)))) != 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(LAYOUT_CHOICES)
    rows, cols = layout
    tile_count = rows * cols
    pair_count = (tile_count - 1) // 2
    tile_h = random.randint(3, 6)
    tile_w = random.randint(3, 6)
    gap = random.randint(1, 2)
    margins = (random.randint(0, 2), random.randint(0, 2))

    if (
        margins[0] * 2 + rows * tile_h + (rows - 1) * gap > 30
        or margins[1] * 2 + cols * tile_w + (cols - 1) * gap > 30
    ):
        return None

    unmatched = _random_mask(tile_h, tile_w)
    masks = [unmatched]

    rotated_slots = {0}
    if pair_count >= 3 and random.random() < 0.5:
        rotated_slots.add(random.randrange(pair_count))

    for pair_idx in range(pair_count):
        for _ in range(100):
            base = _random_mask(tile_h, tile_w)
            if pair_idx in rotated_slots and np.array_equal(base, np.rot90(base, 2)):
                continue
            first, second = _make_pair(base, rotated=pair_idx in rotated_slots)
            masks.extend((first, second))
            break
        else:
            return None

    if len(masks) != tile_count:
        return None

    grid = _build_grid(masks, layout, gap, margins)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-pairing ARC input")

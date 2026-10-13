from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_guide_line(grid: np.ndarray) -> tuple[str, int, int]:
    n, m = grid.shape
    assert n == m and n > 0, "expected a non-empty square tile"

    solid_rows = [
        r
        for r in range(n)
        if len(set(map(int, grid[r, :]))) == 1 and int(grid[r, 0]) not in (BLACK, RED)
    ]
    solid_cols = [
        c
        for c in range(m)
        if len(set(map(int, grid[:, c]))) == 1 and int(grid[0, c]) not in (BLACK, RED)
    ]
    assert len(solid_rows) + len(solid_cols) == 1, "expected exactly one non-red guide line"

    if solid_rows:
        r = solid_rows[0]
        return "row", r, int(grid[r, 0])
    c = solid_cols[0]
    return "col", c, int(grid[0, c])


def _label_non_wall_regions(grid: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    next_id = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] == RED or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if grid[nr, nc] == RED or labels[nr, nc] != 0:
                        continue
                    labels[nr, nc] = next_id
                    queue.append((nr, nc))
    return labels, next_id


def _build_region_graph(labels: np.ndarray, wall_mask: np.ndarray) -> dict[int, set[int]]:
    graph = {
        rid: set() for rid in range(1, int(labels.max()) + 1)
    }
    h, w = labels.shape
    for r in range(h):
        for c in range(w):
            if not wall_mask[r, c]:
                continue
            adjacent = set()
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(labels.shape, nr, nc) and labels[nr, nc] > 0:
                    adjacent.add(int(labels[nr, nc]))
            for a, b in combinations(sorted(adjacent), 2):
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _guide_anchor_regions(labels: np.ndarray, orientation: str, index: int) -> set[int]:
    if orientation == "row":
        anchors = {int(v) for v in labels[index, :] if int(v) > 0}
    else:
        anchors = {int(v) for v in labels[:, index] if int(v) > 0}
    assert anchors, "guide line must pass through at least one region"
    return anchors


def _bipartite_parity(
    graph: dict[int, set[int]], anchors: set[int], n_regions: int
) -> dict[int, int]:
    parity = {rid: 0 for rid in anchors}
    queue = deque(sorted(anchors))
    while queue:
        rid = queue.popleft()
        for other in graph[rid]:
            want = 1 - parity[rid]
            if other in parity:
                assert parity[other] == want, "region adjacency graph must be bipartite"
            else:
                parity[other] = want
                queue.append(other)

    if n_regions == 1:
        return {1: 0}
    assert len(parity) == n_regions, "all regions must be connected through wall adjacencies"
    return parity


def _erase_opposite_partition(grid: np.ndarray) -> tuple[np.ndarray, str, int]:
    orientation, index, _guide_color = _find_guide_line(grid)
    labels, n_regions = _label_non_wall_regions(grid)
    assert 2 <= n_regions <= 4, "expected 2-4 wall-separated regions"

    graph = _build_region_graph(labels, grid == RED)
    anchors = _guide_anchor_regions(labels, orientation, index)
    parity = _bipartite_parity(graph, anchors, n_regions)

    out = grid.copy()
    for rid, part in parity.items():
        if part == 1:
            out[labels == rid] = BLACK
    return out, orientation, index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the input tile along the guide band, then erase every wall-adjacent opposite region class inside each copy.

    Input structure:
    - The input is a small square tile with exactly one full non-black non-red guide row or guide column.
    - Red pixels are one-cell-thick walls that split the non-red area into 2-4 four-connected regions.
    - Non-wall cells are colored patches; the guide line lies in the region class that must be preserved.

    Rule:
    - Detect the unique solid guide row or column and use its index to choose one block-row or block-column in an n-by-n output canvas of n-by-n blocks.
    - Within the tile, label non-wall regions, connect regions that touch opposite sides of the same red wall, and bipartition that region graph.
    - Keep the partition containing the guide line, erase the opposite partition to black, preserve red walls, and copy this processed tile into every block of the selected output band.

    Key insight:
    - The same guide line controls two decisions at once: where the tile is repeated in the enlarged canvas, and which side of the wall-induced bipartite region structure survives inside each repeated copy.
    """
    tile = np.asarray(grid, dtype=int)
    assert tile.ndim == 2, "expected a 2D grid"
    h, w = tile.shape
    assert h == w and h <= 30, "expected a square ARC tile"

    processed, orientation, index = _erase_opposite_partition(tile)
    n = h
    out = np.zeros((n * n, n * n), dtype=int)
    if orientation == "row":
        band = np.tile(processed, (1, n))
        out[index * n : (index + 1) * n, :] = band
    else:
        band = np.tile(processed, (n, 1))
        out[:, index * n : (index + 1) * n] = band
    return out


def _line_candidates(n: int, count: int) -> list[tuple[int, ...]]:
    interior = list(range(1, n - 1))
    combos = []
    for combo in combinations(interior, count):
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1)):
            combos.append(tuple(int(v) for v in combo))
    return combos


def _intervals_from_walls(n: int, walls: tuple[int, ...]) -> list[tuple[int, int]]:
    bounds = (-1, *walls, n)
    intervals = []
    for i in range(len(bounds) - 1):
        lo = bounds[i] + 1
        hi = bounds[i + 1] - 1
        if lo <= hi:
            intervals.append((lo, hi))
    return intervals


def _pick_structure() -> tuple[int, str, tuple[int, ...], list[tuple[int, int]], int, int]:
    n = random.choice((4, 5))
    orientation = random.choice(("row", "col"))

    if n == 4:
        wall_count = 1
    else:
        wall_count = random.choice((1, 2))

    walls = random.choice(_line_candidates(n, wall_count))
    intervals = _intervals_from_walls(n, walls)
    guide_region = random.randrange(len(intervals))
    guide_lo, guide_hi = intervals[guide_region]
    guide_index = random.randint(guide_lo, guide_hi)
    return n, orientation, walls, intervals, guide_region, guide_index


def _sample_region_colors(region_count: int, guide_region: int) -> tuple[list[int], list[int]]:
    colors = random.sample(PALETTE, 2 * region_count)
    mains = colors[:region_count]
    alts = colors[region_count:]
    mains[guide_region] = colors[guide_region]
    assert all(m != a for m, a in zip(mains, alts))
    return mains, alts


def _row_pattern(width: int, main: int, alt: int, phase: int) -> np.ndarray:
    vals = np.empty(width, dtype=int)
    for c in range(width):
        vals[c] = main if (c + phase) % 2 == 0 else alt
    return vals


def _col_pattern(height: int, main: int, alt: int, phase: int) -> np.ndarray:
    vals = np.empty(height, dtype=int)
    for r in range(height):
        vals[r] = main if (r + phase) % 2 == 0 else alt
    return vals


def _fill_row_oriented(
    n: int,
    walls: tuple[int, ...],
    intervals: list[tuple[int, int]],
    guide_region: int,
    guide_row: int,
    mains: list[int],
    alts: list[int],
) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    region_of_row = {}
    for rid, (lo, hi) in enumerate(intervals):
        for r in range(lo, hi + 1):
            region_of_row[r] = rid

    for wall in walls:
        grid[wall, :] = RED

    for r in range(n):
        if r in walls:
            continue
        rid = region_of_row[r]
        main_color = mains[rid]
        alt_color = alts[rid]
        if r == guide_row:
            grid[r, :] = main_color
            continue
        phase = (r + rid + random.randint(0, 1)) % 2
        grid[r, :] = _row_pattern(n, main_color, alt_color, phase)
    return grid


def _fill_col_oriented(
    n: int,
    walls: tuple[int, ...],
    intervals: list[tuple[int, int]],
    guide_region: int,
    guide_col: int,
    mains: list[int],
    alts: list[int],
) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    region_of_col = {}
    for rid, (lo, hi) in enumerate(intervals):
        for c in range(lo, hi + 1):
            region_of_col[c] = rid

    for wall in walls:
        grid[:, wall] = RED

    for c in range(n):
        if c in walls:
            continue
        rid = region_of_col[c]
        main_color = mains[rid]
        alt_color = alts[rid]
        if c == guide_col:
            grid[:, c] = main_color
            continue
        phase = (c + rid + random.randint(0, 1)) % 2
        grid[:, c] = _col_pattern(n, main_color, alt_color, phase)
    return grid


def _non_red_solid_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [
        r
        for r in range(grid.shape[0])
        if len(set(map(int, grid[r, :]))) == 1 and int(grid[r, 0]) not in (BLACK, RED)
    ]
    cols = [
        c
        for c in range(grid.shape[1])
        if len(set(map(int, grid[:, c]))) == 1 and int(grid[0, c]) not in (BLACK, RED)
    ]
    return rows, cols


def _valid_input(grid: np.ndarray) -> bool:
    rows, cols = _non_red_solid_lines(grid)
    if len(rows) + len(cols) != 1:
        return False

    try:
        processed, orientation, index = _erase_opposite_partition(grid)
    except AssertionError:
        return False

    if orientation == "row":
        if len(rows) != 1 or rows[0] != index:
            return False
    else:
        if len(cols) != 1 or cols[0] != index:
            return False

    if np.array_equal(processed, grid):
        return False
    if BLACK not in set(map(int, processed.flat)):
        return False
    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n, orientation, walls, intervals, guide_region, guide_index = _pick_structure()
    mains, alts = _sample_region_colors(len(intervals), guide_region)

    if orientation == "row":
        grid = _fill_row_oriented(n, walls, intervals, guide_region, guide_index, mains, alts)
    else:
        grid = _fill_col_oriented(n, walls, intervals, guide_region, guide_index, mains, alts)

    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

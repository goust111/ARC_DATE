from __future__ import annotations

from collections import deque
from itertools import permutations
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
CYAN = 8

TILE = 3
STEP = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 500
TREE_RETRIES = 300


def _tile_layout(length: int) -> tuple[list[int], set[int]]:
    assert (length + 1) % STEP == 0, "grid side must be 4k-1 for 3x3 tiles with 1px separators"
    count = (length + 1) // STEP
    starts = [idx * STEP for idx in range(count)]
    seps = {start + TILE for start in starts[:-1]}
    return starts, seps


def _neighbors(cell: tuple[int, int], rows: int, cols: int):
    r, c = cell
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(1, len(path) - 1):
        r0, c0 = path[idx - 1]
        r1, c1 = path[idx]
        r2, c2 = path[idx + 1]
        if (r1 - r0, c1 - c0) != (r2 - r1, c2 - c1):
            turns += 1
    return turns


def _invert_tile(tile: np.ndarray) -> np.ndarray:
    return np.where(tile == CYAN, BLACK, CYAN).astype(int)


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_starts, sep_rows = _tile_layout(int(g.shape[0]))
    col_starts, sep_cols = _tile_layout(int(g.shape[1]))

    for r in sep_rows:
        assert np.all(g[r, :] == BLACK), "separator rows must be black"
    for c in sep_cols:
        assert np.all(g[:, c] == BLACK), "separator columns must be black"

    tile_rows = len(row_starts)
    tile_cols = len(col_starts)
    original_tiles: dict[tuple[int, int], np.ndarray] = {}
    transformed_tiles: dict[tuple[int, int], np.ndarray] = {}
    passable = np.zeros((tile_rows, tile_cols), dtype=bool)
    start = None
    end = None

    for tr, r0 in enumerate(row_starts):
        for tc, c0 in enumerate(col_starts):
            tile = g[r0 : r0 + TILE, c0 : c0 + TILE].copy()
            original_tiles[(tr, tc)] = tile

            if np.all(tile == RED):
                assert start is None, "expected one start tile"
                start = (tr, tc)
                transformed_tiles[(tr, tc)] = tile
                passable[tr, tc] = True
                continue
            if np.all(tile == GREEN):
                assert end is None, "expected one end tile"
                end = (tr, tc)
                transformed_tiles[(tr, tc)] = tile
                passable[tr, tc] = True
                continue

            assert set(map(int, np.unique(tile))) <= {BLACK, CYAN}, "ordinary tiles must use only black and cyan"
            inverted = _invert_tile(tile)
            transformed_tiles[(tr, tc)] = inverted
            passable[tr, tc] = bool(np.any(inverted == BLACK))

    assert start is not None and end is not None and start != end, "expected unique start and end tiles"
    return g, row_starts, col_starts, original_tiles, transformed_tiles, passable, start, end


def _bfs_dist_counts(passable: np.ndarray, start: tuple[int, int]):
    rows, cols = passable.shape
    dist = -np.ones((rows, cols), dtype=int)
    counts = np.zeros((rows, cols), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    q = deque([start])
    dist[start] = 0
    counts[start] = 1

    while q:
        cell = q.popleft()
        r, c = cell
        for nb in _neighbors(cell, rows, cols):
            nr, nc = nb
            if not passable[nr, nc]:
                continue
            nd = int(dist[r, c] + 1)
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                counts[nr, nc] = int(counts[r, c])
                parent[nb] = cell
                q.append(nb)
            elif dist[nr, nc] == nd:
                counts[nr, nc] += int(counts[r, c])
    return dist, counts, parent


def _reconstruct_unique_shortest_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]):
    rows, cols = passable.shape
    dist, counts, parent = _bfs_dist_counts(passable, start)
    assert dist[end] >= 0, "start and end must be connected"

    if counts[end] == 1:
        path = [end]
        cur = end
        while cur != start:
            preds = [
                nb
                for nb in _neighbors(cur, rows, cols)
                if passable[nb] and dist[nb] == dist[cur] - 1 and counts[nb] > 0
            ]
            assert len(preds) == 1, "unique shortest path should have a unique predecessor chain"
            cur = preds[0]
            path.append(cur)
        path.reverse()
        return path, dist, counts

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, dist, counts


def _build_output(
    base_grid: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    transformed_tiles: dict[tuple[int, int], np.ndarray],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros_like(base_grid)
    for tr, r0 in enumerate(row_starts):
        for tc, c0 in enumerate(col_starts):
            out[r0 : r0 + TILE, c0 : c0 + TILE] = transformed_tiles[(tr, tc)]

    for tr, tc in path[1:-1]:
        r0 = row_starts[tr]
        c0 = col_starts[tc]
        tile = out[r0 : r0 + TILE, c0 : c0 + TILE].copy()
        tile[tile == BLACK] = YELLOW
        out[r0 : r0 + TILE, c0 : c0 + TILE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each ordinary tile, treat the inversion result as a wall-vs-channel test, then mark the unique tile path.

    Input structure:
    - The grid is partitioned into 3x3 tiles by 1-pixel black separator rows and columns.
    - One tile is solid red (start) and one tile is solid green (end).
    - Every other tile contains only black and cyan cells.

    Rule:
    - Keep the red and green tiles unchanged, but invert each ordinary tile by swapping cyan and black.
    - After inversion, a tile that is fully cyan is a wall, while a tile that still contains at least one black cell is a passable channel.
    - Find the shortest 4-neighbor path between the start and end in tile coordinates.
    - On the path, recolor black cells inside each intermediate channel tile to yellow; leave cyan cells, separators, start, and end unchanged.

    Key insight:
    - The inversion is only an intermediate test: tiles that looked emptiest in the input can become walls, while tiles with even a little cyan become traversable after inversion.
    """
    g, row_starts, col_starts, _original_tiles, transformed_tiles, passable, start, end = _analyze_grid(grid)
    path, _dist, _counts = _reconstruct_unique_shortest_path(passable, start, end)
    return _build_output(g, row_starts, col_starts, transformed_tiles, path)


def _make_sparse_patterns() -> list[np.ndarray]:
    patterns = []
    for perm in permutations(range(TILE)):
        tile = np.zeros((TILE, TILE), dtype=int)
        for r, c in enumerate(perm):
            tile[r, c] = CYAN
        patterns.append(tile)
    return patterns


def _make_mixed_patterns() -> list[np.ndarray]:
    raw = _make_sparse_patterns() + [
        np.array([[CYAN, BLACK, CYAN], [BLACK, CYAN, BLACK], [CYAN, BLACK, CYAN]], dtype=int),
        np.array([[CYAN, CYAN, CYAN], [BLACK, CYAN, BLACK], [BLACK, CYAN, BLACK]], dtype=int),
        np.array([[CYAN, BLACK, BLACK], [CYAN, CYAN, BLACK], [BLACK, CYAN, CYAN]], dtype=int),
        np.array([[CYAN, CYAN, BLACK], [CYAN, BLACK, CYAN], [BLACK, CYAN, CYAN]], dtype=int),
        np.array([[CYAN, CYAN, CYAN], [CYAN, BLACK, CYAN], [CYAN, CYAN, CYAN]], dtype=int),
    ]
    unique: dict[bytes, np.ndarray] = {}
    for tile in raw:
        unique[tile.tobytes()] = tile
    return [tile.copy() for tile in unique.values()]


SPARSE_PATTERNS = _make_sparse_patterns()
MIXED_PATTERNS = _make_mixed_patterns()
FULL_CYAN_TILE = np.full((TILE, TILE), CYAN, dtype=int)
WALL_TILE = np.zeros((TILE, TILE), dtype=int)


def _random_pattern(patterns: list[np.ndarray]) -> np.ndarray:
    tile = random.choice(patterns).copy()
    if random.random() < 0.5:
        tile = np.fliplr(tile)
    return np.rot90(tile, k=random.randrange(4)).copy()


def _occupied_degree(cell: tuple[int, int], occupied: set[tuple[int, int]], rows: int, cols: int) -> int:
    return sum((nb in occupied) for nb in _neighbors(cell, rows, cols))


def _farthest_in_tree(
    occupied: set[tuple[int, int]], rows: int, cols: int, start: tuple[int, int]
) -> tuple[tuple[int, int], dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    q = deque([start])
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cell = q.popleft()
        for nb in _neighbors(cell, rows, cols):
            if nb in occupied and nb not in dist:
                dist[nb] = dist[cell] + 1
                parent[nb] = cell
                q.append(nb)
    farthest = max(dist, key=lambda cell: (dist[cell], cell[0], cell[1]))
    return farthest, dist, parent


def _diameter_path(occupied: set[tuple[int, int]], rows: int, cols: int) -> list[tuple[int, int]]:
    first = next(iter(occupied))
    a, _dist0, _parent0 = _farthest_in_tree(occupied, rows, cols, first)
    b, _dist1, parent = _farthest_in_tree(occupied, rows, cols, a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _grow_passable_tree(tile_rows: int, tile_cols: int):
    total = tile_rows * tile_cols
    target_min = min(total - 1, tile_rows + tile_cols + 2)
    target_max = min(total - 1, tile_rows + tile_cols + 4)
    if target_max < target_min:
        target_min = target_max
    target = random.randint(target_min, target_max)

    for _ in range(TREE_RETRIES):
        occupied = {(random.randrange(tile_rows), random.randrange(tile_cols))}
        while len(occupied) < target:
            seen_rows = {r for r, _ in occupied}
            seen_cols = {c for _, c in occupied}
            candidates = []
            weights = []

            for r in range(tile_rows):
                for c in range(tile_cols):
                    cell = (r, c)
                    if cell in occupied:
                        continue
                    attach = [nb for nb in _neighbors(cell, tile_rows, tile_cols) if nb in occupied]
                    if len(attach) != 1:
                        continue

                    weight = 1
                    if r not in seen_rows:
                        weight += 7
                    if c not in seen_cols:
                        weight += 7

                    anchor = attach[0]
                    anchor_degree = _occupied_degree(anchor, occupied, tile_rows, tile_cols)
                    if anchor_degree == 1:
                        weight += 2
                    else:
                        weight += 4

                    candidates.append(cell)
                    weights.append(weight)

            if not candidates:
                break
            occupied.add(random.choices(candidates, weights=weights, k=1)[0])

        if len(occupied) != target:
            continue
        if len({r for r, _ in occupied}) != tile_rows:
            continue
        if len({c for _, c in occupied}) != tile_cols:
            continue

        path = _diameter_path(occupied, tile_rows, tile_cols)
        off_path = occupied - set(path)
        if len(path) < max(5, min(tile_rows, tile_cols) + 2):
            continue
        if len(off_path) < 2:
            continue
        if _count_turns(path) < 1:
            continue
        return occupied, path

    return None


def _render_input_grid(
    tile_rows: int,
    tile_cols: int,
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
) -> np.ndarray:
    height = tile_rows * STEP - 1
    width = tile_cols * STEP - 1
    grid = np.zeros((height, width), dtype=int)

    start = path[0]
    end = path[-1]
    path_set = set(path)
    off_path = [cell for cell in occupied if cell not in path_set]
    internal_path = path[1:-1]
    assert len(off_path) >= 2 and internal_path

    off_full_node = random.choice(off_path)
    remaining_off_path = [cell for cell in off_path if cell != off_full_node]
    off_mixed_node = random.choice(remaining_off_path)
    path_mixed_node = random.choice(internal_path)

    assigned: dict[tuple[int, int], np.ndarray] = {
        off_full_node: FULL_CYAN_TILE.copy(),
        off_mixed_node: _random_pattern(MIXED_PATTERNS),
        path_mixed_node: _random_pattern(SPARSE_PATTERNS),
    }

    optional_full_nodes = [
        cell
        for cell in occupied
        if cell not in {start, end, off_full_node, off_mixed_node, path_mixed_node}
    ]
    random.shuffle(optional_full_nodes)
    for cell in optional_full_nodes[: random.randint(0, min(1, len(optional_full_nodes)))]:
        assigned[cell] = FULL_CYAN_TILE.copy()

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            cell = (tr, tc)
            r0 = tr * STEP
            c0 = tc * STEP
            if cell == start:
                tile = np.full((TILE, TILE), RED, dtype=int)
            elif cell == end:
                tile = np.full((TILE, TILE), GREEN, dtype=int)
            elif cell in occupied:
                tile = assigned.get(cell)
                if tile is None:
                    tile = _random_pattern(MIXED_PATTERNS)
            else:
                tile = WALL_TILE
            grid[r0 : r0 + TILE, c0 : c0 + TILE] = tile

    return grid


def _validate_semantics(grid: np.ndarray) -> np.ndarray:
    g, row_starts, col_starts, original_tiles, transformed_tiles, passable, start, end = _analyze_grid(grid)
    path, dist, counts = _reconstruct_unique_shortest_path(passable, start, end)
    path_set = set(path)

    assert counts[end] == 1, "generated instance must have a unique shortest path"
    assert len(path) >= 5, "path should be long enough to be visible"
    assert _count_turns(path) >= 1, "path should not degenerate to a straight segment"

    ordinary_cells = [(tr, tc) for tr in range(passable.shape[0]) for tc in range(passable.shape[1]) if (tr, tc) not in {start, end}]
    walls = [cell for cell in ordinary_cells if not passable[cell]]
    off_path_channels = [cell for cell in ordinary_cells if passable[cell] and cell not in path_set]
    full_cyan_channels = [cell for cell in ordinary_cells if np.all(original_tiles[cell] == CYAN)]
    sparse_channels = [
        cell
        for cell in ordinary_cells
        if passable[cell] and not np.all(original_tiles[cell] == CYAN) and int(np.count_nonzero(original_tiles[cell] == CYAN)) <= 5
    ]

    assert walls, "need at least one true wall tile"
    assert len(off_path_channels) >= 2, "need non-path channels to rule out marking all channels"
    assert any(cell in off_path_channels for cell in full_cyan_channels), "need an off-path full-cyan channel"
    assert any(cell in off_path_channels for cell in sparse_channels), "need an off-path sparse channel"
    assert any(
        cell in path_set
        and cell not in {start, end}
        and np.any(transformed_tiles[cell] == BLACK)
        and np.any(transformed_tiles[cell] == CYAN)
        for cell in sparse_channels
    ), "need a mixed internal path tile so the output shows yellow plus cyan"

    sep_rows = {start_idx + TILE for start_idx in row_starts[:-1]}
    sep_cols = {start_idx + TILE for start_idx in col_starts[:-1]}
    for r in range(g.shape[0]):
        if r not in sep_rows:
            assert np.any(g[r, :] != BLACK), "non-separator rows must not be fully black"
    for c in range(g.shape[1]):
        if c not in sep_cols:
            assert np.any(g[:, c] != BLACK), "non-separator columns must not be fully black"

    out = _build_output(g, row_starts, col_starts, transformed_tiles, path)
    assert np.any(out == YELLOW), "output should visibly mark the path"

    for cell in off_path_channels:
        r0 = row_starts[cell[0]]
        c0 = col_starts[cell[1]]
        assert not np.any(out[r0 : r0 + TILE, c0 : c0 + TILE] == YELLOW), "off-path channels must stay unmarked"

    assert np.array_equal(main(g), out), "solver and semantic construction must agree"
    assert dist[end] == len(path) - 1, "path length must match BFS distance"
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        tile_rows = random.randint(3, 6)
        tile_cols = random.randint(3, 6)
        grown = _grow_passable_tree(tile_rows, tile_cols)
        if grown is None:
            continue
        occupied, path = grown
        grid = _render_input_grid(tile_rows, tile_cols, occupied, path)
        try:
            return _validate_semantics(grid)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid tile-inversion pathfinding instance")

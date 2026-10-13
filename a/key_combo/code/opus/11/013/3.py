from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

TILE_ROWS = 3
TILE_COLS = 3
TILE_SIZE_CHOICES = (3, 4)
BLUE_COUNT_CHOICES = (1, 2, 3, 4, 5, 6)
GENERATE_ATTEMPTS = 600

NODES = [(r, c) for r in range(TILE_ROWS) for c in range(TILE_COLS)]
NODE_TO_INDEX = {node: idx for idx, node in enumerate(NODES)}
DIAGONAL_EDGES = []
for i, (r0, c0) in enumerate(NODES):
    for j, (r1, c1) in enumerate(NODES[i + 1 :], i + 1):
        if abs(r0 - r1) == abs(c0 - c1):
            DIAGONAL_EDGES.append((i, j))
LONG_EDGES = {
    edge
    for edge in DIAGONAL_EDGES
    if max(abs(NODES[edge[0]][0] - NODES[edge[1]][0]), abs(NODES[edge[0]][1] - NODES[edge[1]][1])) == 2
}
PAIR_PATTERNS = [
    pair_set
    for pair_set in combinations(DIAGONAL_EDGES, 2)
    if len({pair_set[0][0], pair_set[0][1], pair_set[1][0], pair_set[1][1]}) == 4
]
SHORT_PATTERNS = [pattern for pattern in PAIR_PATTERNS if all(edge not in LONG_EDGES for edge in pattern)]
LONG_PATTERNS = [pattern for pattern in PAIR_PATTERNS if any(edge in LONG_EDGES for edge in pattern)]


def _separator_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _bands_from_separators(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    prev = -1
    for stop in [*separators, limit]:
        if stop - prev > 1:
            bands.append((prev + 1, stop - 1))
        prev = stop
    return bands


def _layout_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _bands_from_separators(_separator_indices(grid, GREEN, axis=0), grid.shape[0])
    col_bands = _bands_from_separators(_separator_indices(grid, GREEN, axis=1), grid.shape[1])
    assert len(row_bands) == TILE_ROWS and len(col_bands) == TILE_COLS, "expected a 3x3 tiled layout"
    return row_bands, col_bands


def _tile_center(band_r: tuple[int, int], band_c: tuple[int, int]) -> tuple[int, int]:
    return ((band_r[0] + band_r[1]) // 2, (band_c[0] + band_c[1]) // 2)


def _tile_bounds(row_bands, col_bands, node_idx: int) -> tuple[int, int, int, int]:
    tr, tc = NODES[node_idx]
    r0, r1 = row_bands[tr]
    c0, c1 = col_bands[tc]
    return r0, r1, c0, c1


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert abs(dr) == abs(dc)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(a[0] + sr * k, a[1] + sc * k) for k in range(1, abs(dr))]


def _invert_tiles(grid: np.ndarray, row_bands, col_bands) -> tuple[np.ndarray, dict[int, int]]:
    out = grid.copy()
    counts: dict[int, int] = {}
    for idx in range(len(NODES)):
        r0, r1, c0, c1 = _tile_bounds(row_bands, col_bands, idx)
        tile = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((tile == BLACK) | (tile == BLUE)), "tiles may only contain black or blue before inversion"
        inverted = np.where(tile == BLACK, RED, BLACK)
        out[r0 : r1 + 1, c0 : c1 + 1] = inverted
        counts[idx] = int(np.count_nonzero(inverted == RED))
    return out, counts


def _paint_equal_diagonals(out: np.ndarray, row_bands, col_bands, counts: dict[int, int]) -> None:
    centers = {
        idx: _tile_center(row_bands[NODES[idx][0]], col_bands[NODES[idx][1]])
        for idx in range(len(NODES))
    }
    for i, j in DIAGONAL_EDGES:
        if counts[i] != counts[j]:
            continue
        for r, c in _path_cells(centers[i], centers[j]):
            if out[r, c] in (BLACK, GREEN):
                out[r, c] = ORANGE


def _oracle(grid: np.ndarray) -> np.ndarray:
    row_bands, col_bands = _layout_from_grid(grid)
    out, counts = _invert_tiles(grid, row_bands, col_bands)
    _paint_equal_diagonals(out, row_bands, col_bands, counts)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each blue/black tile and draw orange diagonal links between tiles whose inverted red counts match.

    Input structure:
    - The grid has black background, full green separator rows and columns, and a 3x3 array of tile regions.
    - Each tile contains only black and blue cells; separator rows and columns stay entirely green.
    - A tile pair is geometrically relevant when their tile positions lie on the same 45-degree diagonal of the 3x3 layout.

    Rule:
    - Keep every green separator row and separator column in place.
    - Inside each tile, independently invert occupancy: black cells become red, blue cells become black.
    - Count the red cells in every inverted tile.
    - For every diagonal tile pair with equal red counts, follow the straight diagonal path between the two tile-center cells and recolor every path cell that is still black or green to orange.

    Key insight:
    - The connector decision is based on per-tile counts after local inversion, so the useful comparison is not the original blue pattern but the red occupancy that the inversion creates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _oracle(g)


def _sample_pair_pattern(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if rng.random() < 0.4:
        return rng.choice(LONG_PATTERNS)
    return rng.choice(SHORT_PATTERNS)


def _mandatory_endpoint_cells(row_bands, col_bands, pattern) -> dict[int, set[tuple[int, int]]]:
    mandatory = {idx: set() for idx in range(len(NODES))}
    centers = {
        idx: _tile_center(row_bands[NODES[idx][0]], col_bands[NODES[idx][1]])
        for idx in range(len(NODES))
    }
    for edge in pattern:
        for endpoint in edge:
            r0, r1, c0, c1 = _tile_bounds(row_bands, col_bands, endpoint)
            local_box = (r0, r1, c0, c1)
            for r, c in _path_cells(centers[edge[0]], centers[edge[1]]):
                if local_box[0] <= r <= local_box[1] and local_box[2] <= c <= local_box[3]:
                    mandatory[endpoint].add((r, c))
    return mandatory


def _target_components(pattern) -> list[list[int]]:
    parent = list(range(len(NODES)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pattern:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for idx in range(len(NODES)):
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def _assign_blue_counts(rng: random.Random, pattern, mandatory_counts: dict[int, int]) -> dict[int, int] | None:
    target_edges = {tuple(sorted(edge)) for edge in pattern}
    groups = _target_components(pattern)
    node_to_group = {}
    for gi, group in enumerate(groups):
        for node in group:
            node_to_group[node] = gi

    neighbors = {gi: set() for gi in range(len(groups))}
    mins = {gi: max(1, *(mandatory_counts[node] for node in group)) for gi, group in enumerate(groups)}
    for a, b in DIAGONAL_EDGES:
        ga = node_to_group[a]
        gb = node_to_group[b]
        if ga == gb:
            if tuple(sorted((a, b))) not in target_edges:
                return None
            continue
        neighbors[ga].add(gb)
        neighbors[gb].add(ga)

    order = sorted(range(len(groups)), key=lambda gi: (-len(neighbors[gi]), -len(groups[gi]), mins[gi]))
    assignment: dict[int, int] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        gi = order[pos]
        choices = [value for value in BLUE_COUNT_CHOICES if value >= mins[gi]]
        rng.shuffle(choices)
        for value in choices:
            if any(assignment.get(other) == value for other in neighbors[gi]):
                continue
            assignment[gi] = value
            if backtrack(pos + 1):
                return True
            del assignment[gi]
        return False

    if not backtrack(0):
        return None

    node_values = {}
    for gi, group in enumerate(groups):
        for node in group:
            node_values[node] = assignment[gi]
    return node_values


def _build_layout(tile_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    side = TILE_ROWS * tile_size + (TILE_ROWS + 1)
    grid = np.full((side, side), BLACK, dtype=int)
    separators = [0, tile_size + 1, 2 * tile_size + 2, 3 * tile_size + 3]
    grid[separators, :] = GREEN
    grid[:, separators] = GREEN
    row_bands = [(separators[i] + 1, separators[i + 1] - 1) for i in range(TILE_ROWS)]
    col_bands = [(separators[i] + 1, separators[i + 1] - 1) for i in range(TILE_COLS)]
    return grid, row_bands, col_bands


def _paint_tile_blues(grid: np.ndarray, row_bands, col_bands, node_idx: int, blue_count: int,
                      mandatory: set[tuple[int, int]], rng: random.Random) -> bool:
    r0, r1, c0, c1 = _tile_bounds(row_bands, col_bands, node_idx)
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if len(mandatory) > blue_count:
        return False
    extras = [cell for cell in cells if cell not in mandatory]
    chosen = set(mandatory)
    needed = blue_count - len(chosen)
    if needed > len(extras):
        return False
    if needed:
        chosen.update(rng.sample(extras, needed))
    for r, c in chosen:
        grid[r, c] = BLUE
    return True


def _count_target_pairs(node_values: dict[int, int]) -> int:
    total = 0
    for a, b in DIAGONAL_EDGES:
        if node_values[a] == node_values[b]:
            total += 1
    return total


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_size = rng.choice(TILE_SIZE_CHOICES)
    pattern = _sample_pair_pattern(rng)
    grid, row_bands, col_bands = _build_layout(tile_size)
    mandatory = _mandatory_endpoint_cells(row_bands, col_bands, pattern)
    mandatory_counts = {idx: len(cells) for idx, cells in mandatory.items()}
    node_values = _assign_blue_counts(rng, pattern, mandatory_counts)
    if node_values is None or _count_target_pairs(node_values) != len(pattern):
        return None

    order = list(range(len(NODES)))
    rng.shuffle(order)
    for node_idx in order:
        if not _paint_tile_blues(
            grid,
            row_bands,
            col_bands,
            node_idx,
            node_values[node_idx],
            mandatory[node_idx],
            rng,
        ):
            return None

    oracle = grid.copy()
    for idx in range(len(NODES)):
        r0, r1, c0, c1 = _tile_bounds(row_bands, col_bands, idx)
        tile = grid[r0 : r1 + 1, c0 : c1 + 1]
        count = int(np.count_nonzero(tile == BLUE))
        if not (1 <= count <= 6):
            return None
    expected = _oracle(grid)
    orange_count = int(np.count_nonzero(expected == ORANGE))
    if orange_count < 2:
        return None
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled inversion connector input")

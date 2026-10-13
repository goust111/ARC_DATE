from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7
BLOCKS = 3
DEFAULT_TILE = 5
GEN_ATTEMPTS = 2000


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _tile_size(shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    assert h == w and h % BLOCKS == 0
    tile = h // BLOCKS
    assert tile >= 2
    return tile


def _obstacle_map(grid: np.ndarray, tile: int) -> np.ndarray:
    obstacles = np.zeros((BLOCKS, BLOCKS), dtype=bool)
    for br in range(BLOCKS):
        for bc in range(BLOCKS):
            block = grid[br * tile : (br + 1) * tile, bc * tile : (bc + 1) * tile]
            obstacles[br, bc] = np.any(block == ORANGE)
    return obstacles


def _segment_endpoints_and_dirs(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert 2 <= len(coords) <= 3, "expected one short green diagonal segment"
    coord_set = set(coords)

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        adjacent = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in coord_set:
                    adjacent.append(nxt)
        neighbors[(r, c)] = adjacent

    endpoints = [coord for coord, adj in neighbors.items() if len(adj) == 1]
    assert len(endpoints) == 2, "segment must have exactly two endpoints"

    rays: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for endpoint in endpoints:
        nr, nc = neighbors[endpoint][0]
        direction = (endpoint[0] - nr, endpoint[1] - nc)
        assert abs(direction[0]) == 1 and abs(direction[1]) == 1
        rays.append((endpoint, direction))
    return rays


def _trace_ray(
    start: tuple[int, int],
    direction: tuple[int, int],
    obstacles: np.ndarray,
    shape: tuple[int, int],
    tile: int,
) -> dict | None:
    r, c = map(int, start)
    dr, dc = map(int, direction)
    path: list[tuple[int, int]] = []
    reflections: list[str] = []
    seen: set[tuple[int, int, int, int]] = set()

    for _ in range(shape[0] * shape[1] * 8):
        state = (r, c, dr, dc)
        if state in seen:
            return None
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            if r in (0, shape[0] - 1) or c in (0, shape[1] - 1):
                return {"path": path, "terminal": (r, c), "reflections": reflections}
            return None

        cur_br, cur_bc = r // tile, c // tile
        nxt_br, nxt_bc = nr // tile, nc // tile
        if obstacles[nxt_br, nxt_bc]:
            row_cross = nxt_br != cur_br
            col_cross = nxt_bc != cur_bc
            if row_cross and col_cross:
                dr, dc = -dr, -dc
                reflections.append("corner")
            elif row_cross:
                dr = -dr
                reflections.append("horizontal")
            elif col_cross:
                dc = -dc
                reflections.append("vertical")
            else:
                return None
            continue

        r, c = nr, nc
        path.append((r, c))
        if r in (0, shape[0] - 1) or c in (0, shape[1] - 1):
            return {"path": path, "terminal": (r, c), "reflections": reflections}

    return None


def _trace_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    tile = _tile_size(g.shape)
    obstacles = _obstacle_map(g, tile)

    rays = _segment_endpoints_and_dirs(g)
    traces = []
    for start, direction in rays:
        start_block = (start[0] // tile, start[1] // tile)
        assert not obstacles[start_block], "green segment must lie in an empty coarse block"
        trace = _trace_ray(start, direction, obstacles, g.shape, tile)
        assert trace is not None, "ray should exit instead of looping"
        traces.append(trace)

    return {"tile": tile, "obstacles": obstacles, "traces": traces}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the orange 3x3 tile mosaic as invisible reflectors and extend the green diagonal ray paths.

    Input structure:
    - The grid is a black square partitioned into a 3x3 lattice of equal square blocks.
    - Some blocks are fully orange, encoding a coarse obstacle map, and one empty block contains a short green diagonal segment.
    - A single red border pixel marks one ray terminal for reference but is not itself part of the obstacle map.

    Rule:
    - Compress the orange occupancy into a 3x3 boolean obstacle layout by marking any block containing orange as blocked.
    - Take the two endpoints of the green diagonal segment and shoot one outward diagonal ray from each endpoint.
    - Whenever the next step would enter a blocked coarse block, reflect the ray by flipping the row component, the column component, or both depending on whether the crossing is through a horizontal side, a vertical side, or a block corner; then continue until the ray reaches the grid edge.
    - Paint every black pixel visited by either ray green, while preserving the original green segment, the orange blocks, and the red marker.

    Key insight:
    - The large orange squares are not direct output objects; they must first be compressed into a hidden coarse reflector map that governs the fine-scale diagonal ray motion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    scene = _trace_scene(g)
    out = g.copy()
    for trace in scene["traces"]:
        for r, c in trace["path"]:
            if out[r, c] == BG:
                out[r, c] = GREEN
    return out


def _segment_candidates(block_r: int, block_c: int, tile: int, length: int, slash: bool) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    r0 = block_r * tile
    c0 = block_c * tile

    if slash:
        for lr in range(tile - length + 1):
            for lc in range(length - 1, tile):
                cells = [(r0 + lr + i, c0 + lc - i) for i in range(length)]
                candidates.append(cells)
    else:
        for lr in range(tile - length + 1):
            for lc in range(tile - length + 1):
                cells = [(r0 + lr + i, c0 + lc + i) for i in range(length)]
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _profile_ok(profile: str, kinds: list[str]) -> bool:
    if profile == "side":
        return len(kinds) >= 1 and all(kind != "corner" for kind in kinds)
    if profile == "corner":
        return "corner" in kinds
    if profile == "mixed":
        return len(kinds) >= 2 and len(set(kinds)) >= 2
    raise ValueError(profile)


def _gen_attempt(profile: str) -> np.ndarray | None:
    tile = DEFAULT_TILE
    size = tile * BLOCKS
    obstacle_count = random.randint(2, 4)

    occ = np.zeros((BLOCKS, BLOCKS), dtype=bool)
    filled = random.sample(range(BLOCKS * BLOCKS), obstacle_count)
    for idx in filled:
        occ[idx // BLOCKS, idx % BLOCKS] = True

    empty_blocks = [(r, c) for r in range(BLOCKS) for c in range(BLOCKS) if not occ[r, c]]
    if len(empty_blocks) < 2:
        return None

    start_block = random.choice(empty_blocks)
    length = random.choice([2, 3])
    slash = random.choice([False, True])

    for segment in _segment_candidates(start_block[0], start_block[1], tile, length, slash):
        grid = np.full((size, size), BG, dtype=int)
        for br in range(BLOCKS):
            for bc in range(BLOCKS):
                if occ[br, bc]:
                    grid[br * tile : (br + 1) * tile, bc * tile : (bc + 1) * tile] = ORANGE

        for r, c in segment:
            grid[r, c] = GREEN

        try:
            scene = _trace_scene(grid)
        except AssertionError:
            continue

        traces = scene["traces"]
        terminals = [trace["terminal"] for trace in traces]
        if len(set(terminals)) != 2:
            continue

        kinds = [kind for trace in traces for kind in trace["reflections"]]
        total_reflections = len(kinds)
        total_path = sum(len(trace["path"]) for trace in traces)
        if not (1 <= total_reflections <= 3):
            continue
        if total_path < 6:
            continue
        if not _profile_ok(profile, kinds):
            continue

        red_terminal = random.choice(terminals)
        if grid[red_terminal] != BG:
            continue
        grid[red_terminal] = RED

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    profile = random.choices(["side", "corner", "mixed"], weights=[3, 3, 2], k=1)[0]
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7

MASK_SIZE = 3
BLOCK_SCALE = 3
LATTICE_SIZE = MASK_SIZE * BLOCK_SCALE

GRID_CHOICES = (17, 19, 21, 23)
SEED_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    cells = set(coords)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _seed_library() -> tuple[np.ndarray, ...]:
    seeds: list[np.ndarray] = []
    cells = [(r, c) for r in range(MASK_SIZE) for c in range(MASK_SIZE)]
    for size in range(SEED_COUNT_RANGE[0], SEED_COUNT_RANGE[1] + 1):
        for combo in combinations(cells, size):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
            for r, c in combo:
                mask[r, c] = BLUE
            if _is_connected(mask == BLUE):
                seeds.append(mask)
    return tuple(seeds)


SEED_LIBRARY = _seed_library()


def _central_origin(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return h // 2 - LATTICE_SIZE // 2, w // 2 - LATTICE_SIZE // 2


def _paint_scaled_blocks(grid: np.ndarray, seed: np.ndarray) -> None:
    top, left = _central_origin(grid.shape)
    for r, c in np.argwhere(seed == BLUE):
        rr = top + BLOCK_SCALE * int(r)
        cc = left + BLOCK_SCALE * int(c)
        grid[rr : rr + BLOCK_SCALE, cc : cc + BLOCK_SCALE] = BLUE


def _edge_direction(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    directions: list[tuple[int, int]] = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))
    assert len(directions) == 1
    return directions[0]


def _wall_openings(grid: np.ndarray, wall_pos: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = wall_pos
    dr, dc = direction
    h, w = grid.shape
    if dr != 0:
        lo = hi = c
        while lo - 1 >= 0 and grid[r, lo - 1] == BLUE:
            lo -= 1
        while hi + 1 < w and grid[r, hi + 1] == BLUE:
            hi += 1
        return [(r, lo - 1), (r, hi + 1)]

    lo = hi = r
    while lo - 1 >= 0 and grid[lo - 1, c] == BLUE:
        lo -= 1
    while hi + 1 < h and grid[hi + 1, c] == BLUE:
        hi += 1
    return [(lo - 1, c), (hi + 1, c)]


def _trace_rays(
    grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]
) -> tuple[set[tuple[int, int]], list[tuple[int, int, str]], dict[str, int]]:
    orange: set[tuple[int, int]] = {start}
    reserved: set[tuple[int, int]] = {start}
    queue: list[tuple[int, int]] = [start]
    seen: set[tuple[int, int]] = set()
    terminal_kind: dict[tuple[int, int], str] = {}
    wall_hits = 0

    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        if (r, c) in seen:
            continue
        seen.add((r, c))
        orange.add((r, c))

        cr, cc = r, c
        while True:
            nr, nc = cr + direction[0], cc + direction[1]
            if not _in_bounds(grid.shape, nr, nc):
                terminal_kind[(cr, cc)] = terminal_kind.get((cr, cc), "boundary")
                break
            if grid[nr, nc] != BLUE:
                orange.add((nr, nc))
                cr, cc = nr, nc
                continue

            wall_hits += 1
            openings: list[tuple[int, int]] = []
            for rr, cc2 in _wall_openings(grid, (nr, nc), direction):
                if (
                    _in_bounds(grid.shape, rr, cc2)
                    and grid[rr, cc2] == BLACK
                    and (rr, cc2) not in reserved
                ):
                    openings.append((rr, cc2))
            if openings:
                for opening in sorted(openings):
                    reserved.add(opening)
                    queue.append(opening)
            else:
                terminal_kind[(cr, cc)] = "closed"
            break

    terminals = [(r, c, kind) for (r, c), kind in sorted(terminal_kind.items())]
    stats = {
        "wall_hits": wall_hits,
        "branch_count": max(0, len(reserved) - 1),
        "terminal_count": len(terminals),
        "closed_terminals": sum(kind == "closed" for _, _, kind in terminals),
        "boundary_terminals": sum(kind == "boundary" for _, _, kind in terminals),
        "orange_count": len(orange),
    }
    return orange, terminals, stats


def _render_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(grid, dtype=int, copy=True)
    seed = out[:MASK_SIZE, :MASK_SIZE]
    _paint_scaled_blocks(out, seed)

    marker_cells = np.argwhere(out == ORANGE)
    assert len(marker_cells) == 1
    start = tuple(map(int, marker_cells[0]))
    direction = _edge_direction(start, out.shape)

    orange, terminals, stats = _trace_rays(out, start, direction)
    for r, c in orange:
        out[r, c] = ORANGE
    for r, c, _kind in terminals:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(out.shape, nr, nc) and out[nr, nc] == BLACK:
                out[nr, nc] = RED
    stats["red_count"] = int(np.count_nonzero(out == RED))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Build a central scaled blue block maze, trace inward orange rays through its openings, and mark ray dead ends in red.

    Input structure
    - The grid is a black background with a 3x3 blue seed mask in the upper-left corner.
    - The seed contains 3-5 blue pixels forming one 4-connected shape.
    - Exactly one orange marker lies on a single outer edge, away from the seed.

    Rule
    - Read the 3x3 seed and reproduce it as a centered 9x9 lattice where each blue seed pixel becomes a solid 3x3 blue block.
    - From the orange edge marker, move inward perpendicular to that edge, painting every traversed black cell orange.
    - When the next step would hit a blue wall, inspect the full contiguous wall segment perpendicular to travel; if either segment-end neighbor is still black, launch a new inward branch from that opening.
    - Any ray that stops at the boundary or at a wall with no unused openings is terminal; paint its 4-neighbor black cells red without overwriting blue or orange pixels.

    Key insight
    - The tiny 3x3 seed simultaneously defines the enlarged obstacle layout and the ray-branch topology, so the same local mask controls both maze geometry and terminal markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert g.shape[0] >= 13 and g.shape[1] >= 13

    seed = g[:MASK_SIZE, :MASK_SIZE]
    blue_count = int(np.count_nonzero(seed == BLUE))
    assert SEED_COUNT_RANGE[0] <= blue_count <= SEED_COUNT_RANGE[1]
    assert _is_connected(seed == BLUE)

    marker_cells = np.argwhere(g == ORANGE)
    assert len(marker_cells) == 1
    start = tuple(map(int, marker_cells[0]))
    _edge_direction(start, g.shape)

    out, _stats = _render_output(g)
    return out


def _marker_too_close_to_seed(pos: tuple[int, int]) -> bool:
    r, c = pos
    for sr in range(MASK_SIZE):
        for sc in range(MASK_SIZE):
            if max(abs(r - sr), abs(c - sc)) <= 1:
                return True
    return False


def _edge_candidates(height: int, width: int) -> list[tuple[str, tuple[int, int]]]:
    candidates: list[tuple[str, tuple[int, int]]] = []
    for c in range(1, width - 1):
        for edge, pos in (("top", (0, c)), ("bottom", (height - 1, c))):
            if not _marker_too_close_to_seed(pos):
                candidates.append((edge, pos))
    for r in range(1, height - 1):
        for edge, pos in (("left", (r, 0)), ("right", (r, width - 1))):
            if not _marker_too_close_to_seed(pos):
                candidates.append((edge, pos))
    return candidates


def _candidate_mode(stats: dict[str, int]) -> str:
    if stats["closed_terminals"] > 0:
        return "closed"
    if stats["terminal_count"] >= 3 or stats["branch_count"] >= 3:
        return "branchy"
    return "open"


def _find_marker_options(seed: np.ndarray, height: int, width: int) -> list[dict[str, object]]:
    base = np.zeros((height, width), dtype=int)
    base[:MASK_SIZE, :MASK_SIZE] = seed
    _paint_scaled_blocks(base, seed)

    options: list[dict[str, object]] = []
    for edge, pos in _edge_candidates(height, width):
        probe = base.copy()
        probe[pos] = ORANGE
        try:
            _out, stats = _render_output(probe)
        except AssertionError:
            continue
        if stats["wall_hits"] < 1 or stats["branch_count"] < 1:
            continue
        if stats["orange_count"] < 4 or stats["red_count"] < 1:
            continue
        option = {
            "edge": edge,
            "position": pos,
            "mode": _candidate_mode(stats),
            "stats": stats,
        }
        options.append(option)
    return options


def _choose_option(options: list[dict[str, object]]) -> dict[str, object]:
    desired = random.choices(["closed", "branchy", "open"], weights=[3, 4, 2], k=1)[0]
    bucket = [option for option in options if option["mode"] == desired]
    if not bucket:
        bucket = [option for option in options if option["mode"] != "open"]
    if not bucket:
        bucket = options
    return random.choice(bucket)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.choice(GRID_CHOICES)
        width = random.choice(GRID_CHOICES)
        seed = np.array(random.choice(SEED_LIBRARY), copy=True)

        options = _find_marker_options(seed, height, width)
        if not options:
            continue

        chosen = _choose_option(options)
        grid = np.zeros((height, width), dtype=int)
        grid[:MASK_SIZE, :MASK_SIZE] = seed
        mr, mc = chosen["position"]
        grid[mr, mc] = ORANGE

        try:
            out, stats = _render_output(grid)
        except AssertionError:
            continue
        if stats["wall_hits"] < 1 or stats["branch_count"] < 1 or stats["red_count"] < 1:
            continue
        if not np.array_equal(main(grid), out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for scaled ray-branch navigation")

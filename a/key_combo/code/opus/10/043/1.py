from __future__ import annotations

from collections import deque, defaultdict
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2

GEN_ATTEMPTS = 400
CHAIN_LAYOUT = ((1, 3, 4), (7, 3, 4), (7, 10, 4), (13, 10, 4))
DECOY_LAYOUTS = ((1, 16, 4), (7, 16, 4), (13, 16, 4), (16, 3, 4))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _blue_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(grid == BLUE):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in seen:
            continue
        stack = [(rr, cc)]
        seen.add((rr, cc))
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            cells.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLUE and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _block_ports(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    ports = []
    for rr, cc in ((rm, c0 - 1), (rm, c1 + 1), (r0 - 1, cm), (r1 + 1, cm)):
        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BLUE:
            ports.append((rr, cc))
    return ports


def _find_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for coords in _blue_components(grid):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == BG) | (patch == BLUE)), "blocks may only contain blue and black"
        assert np.all(patch[0, :] == BLUE)
        assert np.all(patch[-1, :] == BLUE)
        assert np.all(patch[:, 0] == BLUE)
        assert np.all(patch[:, -1] == BLUE)

        holes = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1) if grid[rr, cc] == BG]
        assert holes, "every block must contain at least one interior hole"

        blue_count = int(np.count_nonzero(patch == BLUE))
        assert blue_count + len(holes) == patch.size, "block bbox must be a rectangle with interior holes only"

        ports = _block_ports(grid, (r0, c0, r1, c1))
        assert len(ports) >= 2, "each traversable block needs at least two midpoint approach cells"
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "holes": sorted(holes),
                "capacity": len(holes),
                "ports": ports,
            }
        )

    blocks.sort(key=lambda block: block["bbox"])
    return blocks


def _marker_positions(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    left = [(int(r), int(c)) for r, c in np.argwhere(grid == RED) if int(c) == 0]
    right = [(int(r), int(c)) for r, c in np.argwhere(grid == RED) if int(c) == w - 1]
    assert len(left) == 1 and len(right) == 1, "expected one left-edge marker and one right-edge marker"
    return left[0], right[0]


def _state_search(
    grid: np.ndarray,
    cap_override: tuple[int, ...] | None = None,
    *,
    require_unique: bool,
) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start, end = _marker_positions(g)
    blocks = _find_blocks(g)

    if cap_override is None:
        caps = tuple(int(block["capacity"]) for block in blocks)
    else:
        caps = tuple(int(v) for v in cap_override)
        assert len(caps) == len(blocks)

    passable = g != BLUE
    port_to_blocks: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, block in enumerate(blocks):
        for port in block["ports"]:
            port_to_blocks[port].append(i)

    start_state = (start, tuple(0 for _ in blocks))
    dist = {start_state: 0}
    count = {start_state: 1}
    parent: dict[tuple, tuple] = {}
    queue: deque[tuple] = deque([start_state])
    goal_dist: int | None = None
    goal_states: list[tuple] = []

    while queue:
        state = queue.popleft()
        pos, used = state
        cur_dist = dist[state]
        if goal_dist is not None and cur_dist > goal_dist:
            break
        if pos == end:
            goal_dist = cur_dist
            goal_states.append(state)
            continue

        r, c = pos
        next_states: list[tuple] = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc]:
                next_states.append(((nr, nc), used))

        for block_index in port_to_blocks.get(pos, []):
            if used[block_index] >= caps[block_index]:
                continue
            for port in blocks[block_index]["ports"]:
                if port == pos:
                    continue
                next_used = list(used)
                next_used[block_index] += 1
                next_states.append((port, tuple(next_used)))

        for next_state in next_states:
            next_dist = cur_dist + 1
            if next_state not in dist:
                dist[next_state] = next_dist
                count[next_state] = count[state]
                parent[next_state] = state
                queue.append(next_state)
            elif dist[next_state] == next_dist:
                count[next_state] = min(2, count[next_state] + count[state])
                parent.pop(next_state, None)

    shortest_count = 0
    path_states: list[tuple] = []
    used_counts: tuple[int, ...] | None = None
    if goal_dist is not None:
        shortest_count = min(2, sum(count[state] for state in goal_states if dist[state] == goal_dist))
        if shortest_count == 1:
            winners = [state for state in goal_states if dist[state] == goal_dist and count[state] == 1]
            assert len(winners) == 1
            cur = winners[0]
            used_counts = cur[1]
            path_states = [cur]
            while cur != start_state:
                cur = parent[cur]
                path_states.append(cur)
            path_states.reverse()

    if require_unique:
        assert goal_dist is not None, "no valid route exists"
        assert shortest_count == 1, "route must be uniquely determined"

    return {
        "blocks": blocks,
        "goal_dist": goal_dist,
        "shortest_count": shortest_count,
        "path_states": path_states,
        "used_counts": used_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique shortest capacity-gated route between the edge markers and mark every gate use.

    Input structure:
    - The background is black.
    - Blue foreground components are disconnected solid rectangles with one or more
      strictly interior black holes.
    - A single red marker lies on the left edge and another red marker lies on the right edge.

    Rule:
    - Treat each blue rectangle as a gate whose capacity equals its number of interior holes.
    - The route may walk orthogonally on non-blue cells, and from a midpoint-adjacent
      approach cell of a gate it may cross to another midpoint-adjacent side of the same
      gate, consuming one unit of that gate's remaining capacity.
    - Among all valid left-to-right routes, use the unique shortest one.
    - Paint every visited non-blue route cell red; for each use of a gate, fill that gate's
      next hole in row-major order with red.

    Key insight:
    - The hole count is not decoration: it is a reusable traversal budget, so routing must
      track remaining gate uses instead of treating the blue rectangles as static walls.
    """
    info = _state_search(grid, require_unique=True)
    out = np.asarray(grid, dtype=int).copy()

    for pos, _used in info["path_states"]:
        r, c = pos
        if out[r, c] != BLUE:
            out[r, c] = RED

    assert info["used_counts"] is not None
    for block, used in zip(info["blocks"], info["used_counts"]):
        assert used <= block["capacity"]
        for r, c in block["holes"][:used]:
            out[r, c] = RED

    return out


def _sample_holes(rng: random.Random, r0: int, c0: int, size: int, count: int) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(r0 + 1, r0 + size - 1) for c in range(c0 + 1, c0 + size - 1)]
    return sorted(rng.sample(interior, count))


def _candidate_grid(rng: random.Random) -> np.ndarray:
    h = rng.randint(18, 23)
    w = rng.randint(20, 24)
    vertical_offset = rng.randint(0, h - 17)

    blocks = []
    for r, c, size in CHAIN_LAYOUT:
        hole_count = rng.randint(1, 3)
        blocks.append((r + vertical_offset, c, size, hole_count))

    if rng.random() < 0.7:
        choices = [(r + vertical_offset, c, size) for r, c, size in DECOY_LAYOUTS if r + vertical_offset + size <= h]
        if choices:
            r, c, size = rng.choice(choices)
            if c + size <= w:
                blocks.append((r, c, size, rng.randint(1, 3)))

    grid = np.zeros((h, w), dtype=int)
    for r, c, size, hole_count in blocks:
        grid[r : r + size, c : c + size] = BLUE
        for rr, cc in _sample_holes(rng, r, c, size, hole_count):
            grid[rr, cc] = BG

    grid[CHAIN_LAYOUT[0][0] + vertical_offset + 2, 0] = RED
    grid[CHAIN_LAYOUT[-1][0] + vertical_offset + 2, w - 1] = RED

    if rng.random() < 0.5:
        grid = np.flipud(grid).copy()
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        info = _state_search(grid, require_unique=True)
    except AssertionError:
        return False

    capacities = [block["capacity"] for block in info["blocks"]]
    if len(info["blocks"]) < 4 or len(info["blocks"]) > 5:
        return False
    if max(capacities) < 2 or len(set(capacities)) < 2:
        return False

    used_counts = info["used_counts"]
    assert used_counts is not None
    used_blocks = [i for i, used in enumerate(used_counts) if used > 0]
    if len(used_blocks) != 4 or sum(used_counts) != 4:
        return False
    if info["goal_dist"] is None or info["goal_dist"] < 16:
        return False

    for index in used_blocks:
        capped = list(capacities)
        capped[index] = 0
        alt = _state_search(grid, tuple(capped), require_unique=False)
        if alt["goal_dist"] is not None and alt["goal_dist"] <= info["goal_dist"]:
            return False

    out = main(grid)
    painted_bg = int(np.count_nonzero((out == RED) & (grid == BG)))
    if painted_bg < 10:
        return False

    expected_holes = 0
    for block, used in zip(info["blocks"], used_counts):
        filled = set(block["holes"][:used])
        unfilled = set(block["holes"][used:])
        if any(out[r, c] != RED for r, c in filled):
            return False
        if any(out[r, c] == RED for r, c in unfilled):
            return False
        expected_holes += used

    if expected_holes != 4:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _candidate_grid(rng)
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid hole-capacity routing instance")

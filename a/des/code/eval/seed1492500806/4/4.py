from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

SOURCE_POOL = (BLUE, RED, GREEN, MAGENTA, CYAN, BROWN)
MIN_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _ordered_cycle_from_mask(mask: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]]:
    height, width = mask.shape
    points = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert points, "expected a non-empty pipe"

    neighbor_map: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for r, c in points:
        neighbors = [nbr for nbr in _neighbors4(r, c, height, width) if nbr in points]
        assert len(neighbors) == 2, "pipe must be a single-pixel simple cycle"
        neighbors.sort()
        neighbor_map[(r, c)] = (neighbors[0], neighbors[1])

    start = min(points)
    cycle = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        n0, n1 = neighbor_map[cur]
        nxt = n0 if prev is None or n1 == prev else n1
        if nxt == start:
            assert len(visited) == len(points), "pipe must be one connected loop"
            break
        assert nxt not in visited, "pipe cannot self-intersect or branch"
        cycle.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    return cycle, neighbor_map


def _left_of_step(cell: tuple[int, int], nxt: tuple[int, int]) -> tuple[int, int]:
    dr = int(nxt[0] - cell[0])
    dc = int(nxt[1] - cell[1])
    return (int(cell[0] - dc), int(cell[1] + dr))


def _decode_valves(
    grid: np.ndarray,
    cycle: list[tuple[int, int]],
    neighbor_map: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    height, width = grid.shape
    yellow_points = {tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)}
    valve_dirs: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    for cell in cycle:
        if int(grid[cell]) != ORANGE:
            continue

        nbr0, nbr1 = neighbor_map[cell]
        straight = (nbr0[0] == nbr1[0] == cell[0]) or (nbr0[1] == nbr1[1] == cell[1])
        assert straight, "generator uses valves only on straight segments"

        adjacent_yellow = [nbr for nbr in _neighbors4(cell[0], cell[1], height, width) if nbr in yellow_points]
        assert len(adjacent_yellow) == 1, "each valve must have exactly one adjacent arrow"
        arrow = adjacent_yellow[0]

        matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for incoming, outgoing in ((nbr0, nbr1), (nbr1, nbr0)):
            if _left_of_step(cell, outgoing) == arrow:
                matches.append((incoming, outgoing))
        assert len(matches) == 1, "arrow must define a unique valve direction"
        valve_dirs[cell] = matches[0]

    return valve_dirs


def _step_allowed(
    src: tuple[int, int],
    dst: tuple[int, int],
    valve_dirs: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    if src in valve_dirs and dst != valve_dirs[src][1]:
        return False
    if dst in valve_dirs and src != valve_dirs[dst][0]:
        return False
    return True


def _solve(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], set[int]], dict[tuple[int, int], int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    pipe_mask = (g != BLACK) & (g != YELLOW)
    cycle, neighbor_map = _ordered_cycle_from_mask(pipe_mask)

    sources = {
        cell: int(g[cell])
        for cell in cycle
        if int(g[cell]) not in (GREY, ORANGE)
    }
    assert sources, "expected at least one source"
    source_cells = set(sources)
    valve_cells = {cell for cell in cycle if int(g[cell]) == ORANGE}
    valve_dirs = _decode_valves(g, cycle, neighbor_map)
    assert valve_cells == set(valve_dirs), "every valve must decode cleanly"

    reach: dict[tuple[int, int], set[int]] = {cell: set() for cell in cycle}
    for source_cell, color in sources.items():
        queue = deque([source_cell])
        seen = {source_cell}
        while queue:
            cell = queue.popleft()
            reach[cell].add(color)
            for nbr in neighbor_map[cell]:
                if not _step_allowed(cell, nbr, valve_dirs):
                    continue
                if nbr in seen:
                    continue
                if nbr in source_cells and nbr != source_cell:
                    continue
                seen.add(nbr)
                queue.append(nbr)

    out = g.copy()
    for cell in cycle:
        if cell in source_cells or cell in valve_cells:
            continue
        colors = reach[cell]
        out[cell] = next(iter(colors)) if len(colors) == 1 else GREY

    return out, reach, sources, valve_cells


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)[0]


def _boundary_from_region(region: np.ndarray) -> np.ndarray:
    height, width = region.shape
    boundary = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        r = int(r)
        c = int(c)
        for rr, cc in _neighbors4(r, c, height, width):
            if not bool(region[rr, cc]):
                boundary[r, c] = True
                break
        else:
            if r == 0 or c == 0 or r == height - 1 or c == width - 1:
                boundary[r, c] = True
    return boundary


def _random_region(height: int, width: int) -> np.ndarray | None:
    region = np.zeros((height, width), dtype=bool)

    top = random.randint(3, height - 8)
    bottom = random.randint(top + 4, height - 4)
    left = random.randint(3, width - 8)
    right = random.randint(left + 4, width - 4)
    region[top : bottom + 1, left : right + 1] = True

    available: list[str] = []
    if top >= 2 and right - left >= 4:
        available.append("top")
    if bottom <= height - 3 and right - left >= 4:
        available.append("bottom")
    if left >= 2 and bottom - top >= 4:
        available.append("left")
    if right <= width - 3 and bottom - top >= 4:
        available.append("right")

    bump_count = random.choices([0, 1, 2, 3], weights=[1, 3, 3, 1], k=1)[0]
    random.shuffle(available)
    for side in available[: min(bump_count, len(available))]:
        if side == "top":
            depth = random.randint(1, min(3, top - 1))
            c0 = random.randint(left + 1, right - 3)
            c1 = random.randint(c0 + 1, right - 1)
            region[top - depth : top, c0 : c1 + 1] = True
        elif side == "bottom":
            depth = random.randint(1, min(3, height - 2 - bottom))
            c0 = random.randint(left + 1, right - 3)
            c1 = random.randint(c0 + 1, right - 1)
            region[bottom + 1 : bottom + depth + 1, c0 : c1 + 1] = True
        elif side == "left":
            depth = random.randint(1, min(3, left - 1))
            r0 = random.randint(top + 1, bottom - 3)
            r1 = random.randint(r0 + 1, bottom - 1)
            region[r0 : r1 + 1, left - depth : left] = True
        else:
            depth = random.randint(1, min(3, width - 2 - right))
            r0 = random.randint(top + 1, bottom - 3)
            r1 = random.randint(r0 + 1, bottom - 1)
            region[r0 : r1 + 1, right + 1 : right + depth + 1] = True

    boundary = _boundary_from_region(region)
    try:
        cycle, _ = _ordered_cycle_from_mask(boundary)
    except AssertionError:
        return None
    if len(cycle) < 16:
        return None
    return boundary


def _valve_arrow_for_direction(
    cell: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
    pipe_mask: np.ndarray,
) -> tuple[int, int] | None:
    del incoming
    arrow = _left_of_step(cell, outgoing)
    height, width = pipe_mask.shape
    r, c = arrow
    if not (0 <= r < height and 0 <= c < width):
        return None
    if bool(pipe_mask[r, c]):
        return None
    return arrow


def _valve_candidates(
    cycle: list[tuple[int, int]],
    pipe_mask: np.ndarray,
) -> dict[int, dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    candidates: dict[int, dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {}
    n = len(cycle)
    for idx, cell in enumerate(cycle):
        prev_cell = cycle[(idx - 1) % n]
        next_cell = cycle[(idx + 1) % n]
        straight = (prev_cell[0] == next_cell[0] == cell[0]) or (prev_cell[1] == next_cell[1] == cell[1])
        if not straight:
            continue
        forward_arrow = _valve_arrow_for_direction(cell, prev_cell, next_cell, pipe_mask)
        backward_arrow = _valve_arrow_for_direction(cell, next_cell, prev_cell, pipe_mask)
        if forward_arrow is None or backward_arrow is None:
            continue
        candidates[idx] = {
            "forward": (prev_cell, next_cell, forward_arrow),
            "backward": (next_cell, prev_cell, backward_arrow),
        }
    return candidates


def _random_extra_split(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _arc_cells(start_idx: int, length: int, cycle_len: int) -> list[int]:
    return [((start_idx + step + 1) % cycle_len) for step in range(length)]


def _place_arc_valves(
    arc_indices: list[int],
    arc_type: str,
    candidates: dict[int, dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]],
) -> list[tuple[int, str]]:
    if arc_type == "conflict":
        return []

    if arc_type == "single":
        usable = [idx for pos, idx in enumerate(arc_indices) if 1 <= pos <= len(arc_indices) - 2 and idx in candidates]
        if not usable:
            return []
        return [(random.choice(usable), random.choice(["forward", "backward"]))]

    feasible_pairs: list[tuple[int, int]] = []
    for left_pos, left_idx in enumerate(arc_indices):
        if left_pos < 1 or left_idx not in candidates:
            continue
        for right_pos in range(left_pos + 2, len(arc_indices) - 1):
            right_idx = arc_indices[right_pos]
            if right_idx not in candidates:
                continue
            feasible_pairs.append((left_idx, right_idx))
    if not feasible_pairs:
        return []
    left_idx, right_idx = random.choice(feasible_pairs)
    return [(left_idx, "backward"), (right_idx, "forward")]


def _build_candidate_grid() -> np.ndarray | None:
    height = random.randint(12, 22)
    width = random.randint(12, 22)
    boundary = _random_region(height, width)
    if boundary is None:
        return None

    cycle, _ = _ordered_cycle_from_mask(boundary)
    n = len(cycle)
    candidates = _valve_candidates(cycle, boundary)

    source_count = random.choice([3, 4])
    arc_types = ["conflict", "single", "double"]
    if source_count == 4:
        arc_types.append(random.choice(["conflict", "single", "double"]))
    random.shuffle(arc_types)

    min_gap = {"conflict": 2, "single": 4, "double": 6}
    base_gaps = [min_gap[t] for t in arc_types]
    extra = n - source_count - sum(base_gaps)
    if extra < 0:
        return None

    gaps = [base + add for base, add in zip(base_gaps, _random_extra_split(extra, source_count))]
    start = random.randrange(n)
    source_indices: list[int] = []
    cur = start
    for gap in gaps:
        source_indices.append(cur)
        cur = (cur + gap + 1) % n

    valve_specs: list[tuple[int, str]] = []
    for source_idx, gap, arc_type in zip(source_indices, gaps, arc_types):
        arc = _arc_cells(source_idx, gap, n)
        placed = _place_arc_valves(arc, arc_type, candidates)
        if arc_type != "conflict" and not placed:
            return None
        valve_specs.extend(placed)

    valve_indices = [idx for idx, _direction in valve_specs]
    if len(valve_indices) != len(set(valve_indices)):
        return None
    if any(idx in source_indices for idx in valve_indices):
        return None

    colors = random.sample(SOURCE_POOL, source_count)
    grid = np.full((height, width), BLACK, dtype=int)
    for cell in cycle:
        grid[cell] = GREY

    for idx, color in zip(source_indices, colors):
        grid[cycle[idx]] = color

    used_arrows: set[tuple[int, int]] = set()
    for idx, direction in valve_specs:
        incoming, outgoing, arrow = candidates[idx][direction]
        del incoming, outgoing
        if arrow in used_arrows:
            return None
        used_arrows.add(arrow)
        grid[cycle[idx]] = ORANGE
        grid[arrow] = YELLOW

    for idx in valve_indices:
        cell = cycle[idx]
        adjacent_yellow = [nbr for nbr in _neighbors4(cell[0], cell[1], height, width) if int(grid[nbr]) == YELLOW]
        if len(adjacent_yellow) != 1:
            return None

    out, reach, sources, valves = _solve(grid)
    del out, sources, valves

    unique_count = 0
    conflict_count = 0
    unreached_count = 0
    for cell in cycle:
        if int(grid[cell]) != GREY:
            continue
        count = len(reach[cell])
        if count == 0:
            unreached_count += 1
        elif count == 1:
            unique_count += 1
        else:
            conflict_count += 1

    if unique_count == 0 or conflict_count == 0 or unreached_count == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MIN_GEN_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

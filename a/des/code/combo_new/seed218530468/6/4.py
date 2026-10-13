from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6
SOURCE_COLORS = (BLUE, RED, GREEN)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}

MIN_GRID_INTERSECTIONS = 3
MAX_GRID_INTERSECTIONS = 5
MIN_GAP = 3
MAX_GAP = 5
MIN_STUB = 1
MAX_STUB = 3
MIN_PAD = 1
MAX_PAD = 3
MIN_ROUTE_VERTICES = 5
MAX_ROUTE_VERTICES = 10
GEN_ATTEMPTS = 400
EXTRA_GATE_PROB = 0.35


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _neg(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return -d[1], d[0]


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return d[1], -d[0]


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def _step_direction(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return b[0] - a[0], b[1] - a[1]


def _cells_on_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment must be axis-aligned")


def _turn_type(incoming: tuple[int, int], outgoing: tuple[int, int]) -> str:
    if outgoing == incoming:
        return "S"
    if outgoing == _left(incoming):
        return "L"
    if outgoing == _right(incoming):
        return "R"
    raise ValueError("route cannot reverse or jump")


def _channel_neighbors(
    grid: np.ndarray, pos: tuple[int, int], source: tuple[int, int]
) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nxt = pos[0] + dr, pos[1] + dc
        if not _in_bounds(grid.shape, nxt):
            continue
        if nxt == source or int(grid[nxt]) == GREY:
            neighbors.append(nxt)
    return neighbors


def _gate_offset(grid: np.ndarray, pos: tuple[int, int]) -> tuple[int, int] | None:
    found: list[tuple[int, int]] = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            gate = pos[0] + dr, pos[1] + dc
            if _in_bounds(grid.shape, gate) and int(grid[gate]) == MAGENTA:
                found.append((dr, dc))
    if not found:
        return None
    return found[0]


def _trace_path(grid: np.ndarray) -> tuple[tuple[int, int], int, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    source_points = np.argwhere(np.isin(g, SOURCE_COLORS))
    assert len(source_points) == 1, "expected exactly one source"
    source = tuple(map(int, source_points[0]))
    source_color = int(g[source])

    start_neighbors = _channel_neighbors(g, source, source)
    start_neighbors = [pos for pos in start_neighbors if int(g[pos]) == GREY]
    assert len(start_neighbors) == 1, "source must sit on a dead end"

    prev = source
    cur = start_neighbors[0]
    incoming = _step_direction(prev, cur)
    visited: set[tuple[int, int]] = set()
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        state = (cur, incoming)
        assert state not in seen_states, "signal path must terminate"
        seen_states.add(state)

        if int(g[cur]) == GREY:
            visited.add(cur)

        neighbors = _channel_neighbors(g, cur, source)
        candidates = [pos for pos in neighbors if pos != prev]
        if not candidates:
            break

        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            by_dir = {_step_direction(cur, pos): pos for pos in candidates}
            desired = incoming
            gate = _gate_offset(g, cur)
            if gate is not None:
                if gate == _add(incoming, _left(incoming)):
                    desired = _left(incoming)
                elif gate == _add(incoming, _right(incoming)):
                    desired = _right(incoming)

            if desired not in by_dir:
                for fallback in (incoming, _left(incoming), _right(incoming)):
                    if fallback in by_dir:
                        desired = fallback
                        break
                else:
                    desired = next(iter(by_dir))
            nxt = by_dir[desired]

        prev, cur = cur, nxt
        incoming = _step_direction(prev, cur)

    return source, source_color, visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the grey channel cells that a signal actually traverses from its colored source.

    Input structure:
    - Black background surrounds a single-pixel-wide orthogonal grey channel network that stays away from the outer border.
    - One dead-end endpoint is replaced by a single red, blue, or green source pixel.
    - Some four-way grey crossings have one magenta gate pixel on exactly one diagonal cell around the crossing.

    Rule:
    - Start from the source and move into the unique adjacent grey channel cell.
    - Along ordinary channel cells, continue along the only forward path; at ungated crossings, go straight.
    - At a gated crossing, interpret the gate relative to the current travel direction: front-left means turn left, front-right means turn right, and either rear diagonal means keep going straight.
    - Stop at the first dead-end endpoint and recolor every grey cell visited along the way to the source color, while leaving the source, magenta gates, and black background unchanged.

    Key insight:
    - The same magenta diagonal can mean different actions depending on the signal's incoming direction, so the path must be simulated directionally rather than read in global coordinates.
    """
    g = np.asarray(grid, dtype=int)
    source, source_color, visited = _trace_path(g)

    out = g.copy()
    for r, c in visited:
        if (r, c) != source and int(out[r, c]) == GREY:
            out[r, c] = source_color
    return out


def _boundary_sides(vertex: tuple[int, int], m: int, n: int) -> list[str]:
    sides: list[str] = []
    if vertex[0] == 0:
        sides.append("top")
    if vertex[0] == m - 1:
        sides.append("bottom")
    if vertex[1] == 0:
        sides.append("left")
    if vertex[1] == n - 1:
        sides.append("right")
    return sides


def _route_requirements(cats: list[str]) -> bool:
    counts = Counter(cats)
    return counts["S"] >= 2 and counts["L"] >= 1 and counts["R"] >= 1


def _can_still_meet(cats: list[str], remaining: int) -> bool:
    counts = Counter(cats)
    need = max(0, 2 - counts["S"]) + max(0, 1 - counts["L"]) + max(0, 1 - counts["R"])
    return need <= remaining


def _route_search(
    m: int, n: int, rng: random.Random
) -> tuple[list[tuple[int, int]], str, str, list[str]] | None:
    max_vertices = min(MAX_ROUTE_VERTICES, m * n)
    starts = [
        ((r, c), side)
        for r in range(m)
        for c in range(n)
        for side in _boundary_sides((r, c), m, n)
    ]
    rng.shuffle(starts)

    def dfs(
        cur: tuple[int, int],
        incoming: tuple[int, int],
        path: list[tuple[int, int]],
        visited: set[tuple[int, int]],
        cats: list[str],
    ) -> tuple[list[tuple[int, int]], str, list[str]] | None:
        if len(path) >= MIN_ROUTE_VERTICES:
            exits = _boundary_sides(cur, m, n)
            rng.shuffle(exits)
            for exit_side in exits:
                out_dir = SIDE_TO_DIR[exit_side]
                if out_dir == _neg(incoming):
                    continue
                all_cats = cats + [_turn_type(incoming, out_dir)]
                if _route_requirements(all_cats):
                    return path.copy(), exit_side, all_cats

        if len(path) >= max_vertices:
            return None

        remaining_after_move = (max_vertices - len(path) - 1) + 1
        moves: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for dr, dc in DIRS:
            nxt = cur[0] + dr, cur[1] + dc
            if 0 <= nxt[0] < m and 0 <= nxt[1] < n and nxt not in visited:
                moves.append(((dr, dc), nxt))
        rng.shuffle(moves)

        for out_dir, nxt in moves:
            new_cats = cats + [_turn_type(incoming, out_dir)]
            if not _can_still_meet(new_cats, remaining_after_move):
                continue
            visited.add(nxt)
            path.append(nxt)
            found = dfs(nxt, out_dir, path, visited, new_cats)
            if found is not None:
                return found
            path.pop()
            visited.remove(nxt)
        return None

    for start, start_side in starts:
        incoming = _neg(SIDE_TO_DIR[start_side])
        found = dfs(start, incoming, [start], {start}, [])
        if found is not None:
            path, exit_side, cats = found
            return path, start_side, exit_side, cats
    return None


def _sample_axis(
    count: int, rng: random.Random
) -> tuple[int, list[int], int, int]:
    pad_before = rng.randint(MIN_PAD, MAX_PAD)
    pad_after = rng.randint(MIN_PAD, MAX_PAD)
    stub_before = rng.randint(MIN_STUB, MAX_STUB)
    stub_after = rng.randint(MIN_STUB, MAX_STUB)
    gaps = [rng.randint(MIN_GAP, MAX_GAP) for _ in range(count - 1)]

    first = pad_before + stub_before
    positions = [first]
    for gap in gaps:
        positions.append(positions[-1] + gap)

    start = positions[0] - stub_before
    end = positions[-1] + stub_after
    size = end + pad_after + 1
    return size, positions, start, end


def _decisions_for_path(
    path: list[tuple[int, int]], start_side: str, exit_side: str
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    incoming = _neg(SIDE_TO_DIR[start_side])
    for idx, vertex in enumerate(path):
        if idx < len(path) - 1:
            outgoing = _step_direction(vertex, path[idx + 1])
        else:
            outgoing = SIDE_TO_DIR[exit_side]
        decisions.append(
            {
                "vertex": vertex,
                "incoming": incoming,
                "outgoing": outgoing,
                "category": _turn_type(incoming, outgoing),
            }
        )
        incoming = outgoing
    return decisions


def _assign_gates(
    decisions: list[dict[str, object]], rng: random.Random
) -> list[tuple[int, int] | None]:
    straight_ids = [idx for idx, item in enumerate(decisions) if item["category"] == "S"]
    assert len(straight_ids) >= 2
    gated_straight = rng.choice(straight_ids)
    ungated_straight = rng.choice([idx for idx in straight_ids if idx != gated_straight])

    result: list[tuple[int, int] | None] = []
    for idx, item in enumerate(decisions):
        incoming = item["incoming"]
        category = item["category"]
        if category == "L":
            result.append(_add(incoming, _left(incoming)))
        elif category == "R":
            result.append(_add(incoming, _right(incoming)))
        elif idx == ungated_straight:
            result.append(None)
        else:
            rear_turn = rng.choice([_left(incoming), _right(incoming)])
            result.append(_add(_neg(incoming), rear_turn))
    return result


def _vertex_to_cell(
    vertex: tuple[int, int], rows: list[int], cols: list[int]
) -> tuple[int, int]:
    return rows[vertex[0]], cols[vertex[1]]


def _endpoint_cell(
    vertex: tuple[int, int],
    side: str,
    rows: list[int],
    cols: list[int],
    top_end: int,
    bottom_end: int,
    left_end: int,
    right_end: int,
) -> tuple[int, int]:
    vr, vc = _vertex_to_cell(vertex, rows, cols)
    if side == "top":
        return top_end, vc
    if side == "bottom":
        return bottom_end, vc
    if side == "left":
        return vr, left_end
    return vr, right_end


def _planned_path_cells(
    path: list[tuple[int, int]],
    start_side: str,
    exit_side: str,
    rows: list[int],
    cols: list[int],
    top_end: int,
    bottom_end: int,
    left_end: int,
    right_end: int,
) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    source = _endpoint_cell(path[0], start_side, rows, cols, top_end, bottom_end, left_end, right_end)
    start = _vertex_to_cell(path[0], rows, cols)
    for cell in _cells_on_segment(source, start)[1:]:
        visited.add(cell)

    for a, b in zip(path, path[1:]):
        start_cell = _vertex_to_cell(a, rows, cols)
        end_cell = _vertex_to_cell(b, rows, cols)
        for cell in _cells_on_segment(start_cell, end_cell)[1:]:
            visited.add(cell)

    finish = _vertex_to_cell(path[-1], rows, cols)
    sink = _endpoint_cell(path[-1], exit_side, rows, cols, top_end, bottom_end, left_end, right_end)
    for cell in _cells_on_segment(finish, sink)[1:]:
        visited.add(cell)

    return visited


def _build_scene(
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    m = rng.randint(MIN_GRID_INTERSECTIONS, MAX_GRID_INTERSECTIONS)
    n = rng.randint(MIN_GRID_INTERSECTIONS, MAX_GRID_INTERSECTIONS)

    route = _route_search(m, n, rng)
    if route is None:
        return None
    path, start_side, exit_side, cats = route

    height, rows, top_end, bottom_end = _sample_axis(m, rng)
    width, cols, left_end, right_end = _sample_axis(n, rng)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for r in rows:
        grid[r, left_end : right_end + 1] = GREY
    for c in cols:
        grid[top_end : bottom_end + 1, c] = GREY

    decisions = _decisions_for_path(path, start_side, exit_side)
    gate_offsets = _assign_gates(decisions, rng)
    used_gates: set[tuple[int, int]] = set()
    for item, offset in zip(decisions, gate_offsets):
        if offset is None:
            continue
        vr, vc = _vertex_to_cell(item["vertex"], rows, cols)
        gate = vr + offset[0], vc + offset[1]
        if not _in_bounds(grid.shape, gate) or int(grid[gate]) != BLACK:
            return None
        grid[gate] = MAGENTA
        used_gates.add(gate)

    path_vertices = {tuple(v) for v in path}
    for i in range(m):
        for j in range(n):
            vertex = (i, j)
            if vertex in path_vertices or rng.random() >= EXTRA_GATE_PROB:
                continue
            vr, vc = _vertex_to_cell(vertex, rows, cols)
            offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            rng.shuffle(offsets)
            for dr, dc in offsets:
                gate = vr + dr, vc + dc
                if _in_bounds(grid.shape, gate) and int(grid[gate]) == BLACK and gate not in used_gates:
                    grid[gate] = MAGENTA
                    used_gates.add(gate)
                    break

    source = _endpoint_cell(path[0], start_side, rows, cols, top_end, bottom_end, left_end, right_end)
    source_color = rng.choice(SOURCE_COLORS)
    grid[source] = source_color

    traversed = _planned_path_cells(path, start_side, exit_side, rows, cols, top_end, bottom_end, left_end, right_end)
    expected = grid.copy()
    for r, c in traversed:
        expected[r, c] = source_color

    if not _route_requirements(cats):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        built = _build_scene(rng)
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid signal-routing ARC input")

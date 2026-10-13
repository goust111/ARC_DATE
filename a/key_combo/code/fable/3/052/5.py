from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TURN_LEFT = "L"
TURN_RIGHT = "R"
ALL_COLORS = list(range(10))
MAX_ATTEMPTS = 400
END_SHAPES = [(2, 2), (2, 3), (3, 2)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _turn_dir(dir_idx: int, turn: str) -> int:
    if turn == TURN_LEFT:
        return (dir_idx - 1) % 4
    if turn == TURN_RIGHT:
        return (dir_idx + 1) % 4
    raise ValueError(f"unknown turn {turn}")


def _relative_turn(incoming: tuple[int, int], outgoing: tuple[int, int]) -> str:
    idx = DIRS.index(incoming)
    if DIRS[_turn_dir(idx, TURN_LEFT)] == outgoing:
        return TURN_LEFT
    if DIRS[_turn_dir(idx, TURN_RIGHT)] == outgoing:
        return TURN_RIGHT
    raise AssertionError("corner is not a 90-degree turn")


def _minimal_period(seq: list[int]) -> list[int]:
    for period in range(1, len(seq) + 1):
        ok = True
        for i, value in enumerate(seq):
            if value != seq[i % period]:
                ok = False
                break
        if ok:
            return seq[:period]
    return seq


def _order_path(path_cells: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    degrees = {
        cell: sum(1 for nb in _neighbors4(cell) if nb in path_cells)
        for cell in path_cells
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    edge_endpoints = [cell for cell in endpoints if _touches_edge(cell, shape)]
    assert len(edge_endpoints) == 1, "partial path must start from a unique edge endpoint"
    start = edge_endpoints[0]

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [nb for nb in _neighbors4(cur) if nb in path_cells and nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be a simple chain"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(path_cells), "path traversal missed cells"
    return ordered


def _extract_trail_and_obstacles(
    trail_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    degrees = {
        cell: sum(1 for nb in _neighbors4(cell) if nb in trail_cells)
        for cell in trail_cells
    }
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    obstacle_leaves = set()
    for leaf in leaves:
        neighbor = next(nb for nb in _neighbors4(leaf) if nb in trail_cells)
        if degrees[neighbor] == 3:
            obstacle_leaves.add(leaf)

    endpoints = [leaf for leaf in leaves if leaf not in obstacle_leaves]
    assert len(endpoints) == 2, "trail must have exactly two non-obstacle endpoints"

    edge_endpoints = [cell for cell in endpoints if _touches_edge(cell, shape)]
    assert len(edge_endpoints) == 1, "partial trail must start from a unique edge endpoint"
    start = edge_endpoints[0]
    goal = endpoints[0] if endpoints[1] == start else endpoints[1]

    parent = {start: None}
    queue = [start]
    while queue and goal not in parent:
        cur = queue.pop(0)
        for nb in _neighbors4(cur):
            if nb in trail_cells and nb not in parent:
                parent[nb] = cur
                queue.append(nb)

    assert goal in parent, "failed to recover main trail"
    ordered: list[tuple[int, int]] = []
    cur = goal
    while cur is not None:
        ordered.append(cur)
        cur = parent[cur]
    ordered.reverse()
    return ordered, obstacle_leaves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the color cycle and obstacle-turn rule from a partial trail, then continue the trail until it stops.

    Input structure:
    - A dominant background contains one partial multicolor 4-connected path whose start touches the grid edge.
    - Two obstacle colors appear only as isolated single pixels; each color has already caused at least one visible turn in the partial trail.
    - A separate monochrome terminal block is present elsewhere and is not part of the trail.

    Rule:
    - Identify the edge-starting trail, order its cells from the edge endpoint to the break, and read the smallest repeating color cycle from that ordered sequence.
    - At each visible corner, the obstacle lies one cell straight ahead from the incoming direction; infer from these corners which obstacle color means left turn and which means right turn.
    - Continue from the break with the current direction and next cycle phase: move one cell at a time, turn in place when the forward cell is an obstacle, and stop before entering the terminal block or leaving the grid.

    Key insight:
    - The input does not show the rule explicitly; the already-drawn trail is a log that simultaneously reveals the color period, the current phase, and the obstacle-color-to-turn mapping needed for the continuation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    fg_mask = g != bg
    comps = _connected_components(fg_mask)
    assert comps, "expected foreground structure"

    trail_comp = None
    remaining_comps: list[list[tuple[int, int]]] = []
    for comp in comps:
        if len(comp) > 1 and any(_touches_edge(cell, g.shape) for cell in comp):
            if trail_comp is None or len(comp) > len(trail_comp):
                if trail_comp is not None:
                    remaining_comps.append(trail_comp)
                trail_comp = comp
            else:
                remaining_comps.append(comp)
        else:
            remaining_comps.append(comp)
    assert trail_comp is not None, "failed to find the edge-starting trail"

    trail_set = set(trail_comp)
    ordered, corner_obstacles = _extract_trail_and_obstacles(trail_set, g.shape)
    assert len(ordered) >= 2
    color_seq = [int(g[r, c]) for r, c in ordered]
    cycle = _minimal_period(color_seq)
    assert cycle, "failed to infer a color cycle"

    obstacle_cells: dict[tuple[int, int], int] = {}
    end_cells: set[tuple[int, int]] = set()
    for cell in corner_obstacles:
        obstacle_cells[cell] = int(g[cell])
    for comp in remaining_comps:
        if len(comp) == 1:
            cell = comp[0]
            obstacle_cells[cell] = int(g[cell])
        else:
            end_cells.update(comp)
    assert end_cells, "expected a terminal block"

    turn_by_color: dict[int, str] = {}
    for i in range(1, len(ordered) - 1):
        prev = ordered[i - 1]
        cur = ordered[i]
        nxt = ordered[i + 1]
        incoming = (cur[0] - prev[0], cur[1] - prev[1])
        outgoing = (nxt[0] - cur[0], nxt[1] - cur[1])
        if incoming == outgoing:
            continue
        obstacle = (cur[0] + incoming[0], cur[1] + incoming[1])
        assert obstacle in obstacle_cells, "missing obstacle at visible corner"
        turn = _relative_turn(incoming, outgoing)
        color = obstacle_cells[obstacle]
        if color in turn_by_color:
            assert turn_by_color[color] == turn, "obstacle color must map to a unique turn"
        turn_by_color[color] = turn

    out = g.copy()
    phase = len(ordered) % len(cycle)
    cur = ordered[-1]
    prev = ordered[-2]
    dir_vec = (cur[0] - prev[0], cur[1] - prev[1])
    dir_idx = DIRS.index(dir_vec)
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (cur[0], cur[1], dir_idx, phase)
        if state in seen_states:
            break
        seen_states.add(state)

        dr, dc = DIRS[dir_idx]
        forward = (cur[0] + dr, cur[1] + dc)
        if not _in_bounds(g.shape, forward):
            break
        if forward in end_cells:
            break
        if forward in obstacle_cells:
            turn = turn_by_color[int(obstacle_cells[forward])]
            dir_idx = _turn_dir(dir_idx, turn)
            dr, dc = DIRS[dir_idx]
            forward = (cur[0] + dr, cur[1] + dc)
            if not _in_bounds(g.shape, forward):
                break
            if forward in end_cells or forward in obstacle_cells:
                break

        cur = forward
        out[cur] = cycle[phase]
        phase = (phase + 1) % len(cycle)

    return out


def _distance_to_boundary(cell: tuple[int, int], dir_idx: int, shape: tuple[int, int]) -> int:
    r, c = cell
    h, w = shape
    if dir_idx == 0:
        return r
    if dir_idx == 1:
        return w - 1 - c
    if dir_idx == 2:
        return h - 1 - r
    return c


def _is_path_step_valid(
    cell: tuple[int, int],
    prev: tuple[int, int],
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(shape, cell):
        return False
    if cell in path_set or cell in object_set:
        return False

    path_neighbors = {nb for nb in _neighbors4(cell) if nb in path_set}
    if path_neighbors != {prev}:
        return False
    if any(nb in object_set for nb in _neighbors4(cell)):
        return False
    return True


def _trace_segment(
    curr: tuple[int, int],
    dir_idx: int,
    length: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    temp_path = set(path_set)
    prev = curr
    dr, dc = DIRS[dir_idx]
    for _ in range(length):
        nxt = (prev[0] + dr, prev[1] + dc)
        if not _is_path_step_valid(nxt, prev, shape, temp_path, object_set):
            return None
        cells.append(nxt)
        temp_path.add(nxt)
        prev = nxt
    return cells


def _is_singleton_object_valid(
    cell: tuple[int, int],
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
    allowed_path_neighbor: tuple[int, int] | None,
) -> bool:
    if not _in_bounds(shape, cell):
        return False
    if cell in path_set or cell in object_set:
        return False

    path_neighbors = {nb for nb in _neighbors4(cell) if nb in path_set}
    if allowed_path_neighbor is None:
        if path_neighbors:
            return False
    elif path_neighbors != {allowed_path_neighbor}:
        return False

    if any(nb in object_set for nb in _neighbors4(cell)):
        return False
    return True


def _edge_start(shape: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return (0, random.randint(2, w - 3)), 2
    if side == "bottom":
        return (h - 1, random.randint(2, w - 3)), 0
    if side == "left":
        return (random.randint(2, h - 3), 0), 1
    return (random.randint(2, h - 3), w - 1), 3


def _terminal_block_ahead(
    last: tuple[int, int],
    dir_idx: int,
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = shape
    r, c = last
    candidates: list[list[tuple[int, int]]] = []
    dims = END_SHAPES[:]
    random.shuffle(dims)

    for bh, bw in dims:
        if dir_idx == 0:
            top = r - bh
            for offset in range(-bw + 1, 1):
                left = c + offset
                cells = [(rr, cc) for rr in range(top, top + bh) for cc in range(left, left + bw)]
                candidates.append(cells)
        elif dir_idx == 1:
            left = c + 1
            for offset in range(-bh + 1, 1):
                top = r + offset
                cells = [(rr, cc) for rr in range(top, top + bh) for cc in range(left, left + bw)]
                candidates.append(cells)
        elif dir_idx == 2:
            top = r + 1
            for offset in range(-bw + 1, 1):
                left = c + offset
                cells = [(rr, cc) for rr in range(top, top + bh) for cc in range(left, left + bw)]
                candidates.append(cells)
        else:
            left = c - bw
            for offset in range(-bh + 1, 1):
                top = r + offset
                cells = [(rr, cc) for rr in range(top, top + bh) for cc in range(left, left + bw)]
                candidates.append(cells)

    random.shuffle(candidates)
    valid: list[list[tuple[int, int]]] = []
    for cells in candidates:
        cell_set = set(cells)
        if any(not _in_bounds(shape, cell) for cell in cell_set):
            continue
        if any(_touches_edge(cell, shape) for cell in cell_set):
            continue
        if cell_set & path_set or cell_set & object_set:
            continue
        ok = True
        for cell in cell_set:
            path_neighbors = {nb for nb in _neighbors4(cell) if nb in path_set}
            if path_neighbors - {last}:
                ok = False
                break
            if any(nb in object_set for nb in _neighbors4(cell)):
                ok = False
                break
        if ok:
            valid.append(list(cell_set))
    return valid


def _place_loose_terminal_block(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = shape
    dims = END_SHAPES[:]
    random.shuffle(dims)
    for bh, bw in dims:
        tops = list(range(1, h - bh))
        lefts = list(range(1, w - bw))
        random.shuffle(tops)
        random.shuffle(lefts)
        for top in tops:
            for left in lefts:
                cells = [(rr, cc) for rr in range(top, top + bh) for cc in range(left, left + bw)]
                cell_set = set(cells)
                if cell_set & path_set or cell_set & object_set:
                    continue
                if any(any(nb in path_set or nb in object_set for nb in _neighbors4(cell)) for cell in cell_set):
                    continue
                return cells
    return None


def _place_extra_obstacles(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    object_set: set[tuple[int, int]],
    obstacle_colors: tuple[int, int],
) -> dict[tuple[int, int], int]:
    h, w = shape
    count = random.randint(0, 3)
    extras: dict[tuple[int, int], int] = {}
    positions = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(positions)
    blocked = set(object_set)
    for cell in positions:
        if len(extras) >= count:
            break
        if cell in path_set or cell in blocked:
            continue
        if any(nb in path_set or nb in blocked for nb in _neighbors4(cell)):
            continue
        color = random.choice(obstacle_colors)
        extras[cell] = color
        blocked.add(cell)
    return extras


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    route: list[tuple[int, int]],
    visible_len: int,
    cycle: list[int],
    phase0: int,
    obstacles: dict[tuple[int, int], int],
    end_block: list[tuple[int, int]],
    end_color: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for cell, color in obstacles.items():
        grid[cell] = color
    for cell in end_block:
        grid[cell] = end_color
    for idx, cell in enumerate(route[:visible_len]):
        grid[cell] = cycle[(phase0 + idx) % len(cycle)]
    return grid


def _build_route_attempt(
    shape: tuple[int, int],
    turns: list[str],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[tuple[int, int]], list[str]] | None:
    start, dir_idx0 = _edge_start(shape)
    route = [start]
    path_set = {start}
    object_set: set[tuple[int, int]] = set()
    obstacle_order: list[tuple[int, int]] = []
    corner_indices: list[int] = []

    def try_finish(
        curr: tuple[int, int],
        dir_idx: int,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[tuple[int, int]], list[str]] | None:
        modes = ["end", "boundary"]
        random.shuffle(modes)
        for mode in modes:
            dist = _distance_to_boundary(curr, dir_idx, shape)
            if mode == "boundary":
                if dist < 2:
                    continue
                seg = _trace_segment(curr, dir_idx, dist, shape, path_set, object_set)
                if seg is None:
                    continue
                added = set(seg)
                loose_end = _place_loose_terminal_block(shape, path_set | added, object_set)
                if loose_end is None:
                    continue
                return route + seg, obstacle_order[:], corner_indices[:], loose_end, turns[:]
            else:
                max_len = min(6, dist - 1)
                if max_len < 2:
                    continue
                lengths = list(range(2, max_len + 1))
                random.shuffle(lengths)
                for length in lengths:
                    seg = _trace_segment(curr, dir_idx, length, shape, path_set, object_set)
                    if seg is None:
                        continue
                    added = set(seg)
                    last = seg[-1]
                    candidates = _terminal_block_ahead(last, dir_idx, shape, path_set | added, object_set)
                    if not candidates:
                        continue
                    return route + seg, obstacle_order[:], corner_indices[:], candidates[0], turns[:]
        return None

    def dfs(curr: tuple[int, int], dir_idx: int, turn_idx: int):
        if turn_idx == len(turns):
            return try_finish(curr, dir_idx)

        dist = _distance_to_boundary(curr, dir_idx, shape)
        max_len = min(5, dist - 1)
        if max_len < 2:
            return None

        lengths = list(range(2, max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            seg = _trace_segment(curr, dir_idx, length, shape, path_set, object_set)
            if seg is None:
                continue
            corner = seg[-1]
            dr, dc = DIRS[dir_idx]
            obstacle = (corner[0] + dr, corner[1] + dc)
            next_dir = _turn_dir(dir_idx, turns[turn_idx])
            next_step = (corner[0] + DIRS[next_dir][0], corner[1] + DIRS[next_dir][1])
            temp_path = path_set | set(seg)
            if not _is_singleton_object_valid(obstacle, shape, temp_path, object_set, corner):
                continue
            if not _is_path_step_valid(next_step, corner, shape, temp_path, object_set | {obstacle}):
                continue

            route.extend(seg)
            path_set.update(seg)
            object_set.add(obstacle)
            obstacle_order.append(obstacle)
            corner_indices.append(len(route) - 1)
            result = dfs(corner, next_dir, turn_idx + 1)
            if result is not None:
                return result

            corner_indices.pop()
            obstacle_order.pop()
            object_set.remove(obstacle)
            for _ in seg:
                path_set.remove(route.pop())
        return None

    return dfs(start, dir_idx0, 0)


def _turn_sequence() -> list[str]:
    n_turns = random.randint(4, 6)
    first = random.choice([TURN_LEFT, TURN_RIGHT])
    second = TURN_RIGHT if first == TURN_LEFT else TURN_LEFT
    turns = [first, second]
    for _ in range(n_turns - 2):
        turns.append(random.choice([TURN_LEFT, TURN_RIGHT]))
    return turns


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 28), random.randint(18, 28))
    bg = random.choice(ALL_COLORS)

    cycle_len = random.choice([3, 3, 4, 4, 5])
    needed = cycle_len + 3
    palette = random.sample([c for c in ALL_COLORS if c != bg], needed)
    cycle = palette[:cycle_len]
    obstacle_colors = (palette[cycle_len], palette[cycle_len + 1])
    end_color = palette[cycle_len + 2]

    left_color, right_color = random.sample(list(obstacle_colors), 2)
    turn_color = {TURN_LEFT: left_color, TURN_RIGHT: right_color}

    turns = _turn_sequence()
    route_data = _build_route_attempt(shape, turns)
    if route_data is None:
        return None
    route, obstacle_order, corner_indices, end_block, turns = route_data

    if len(route) < 16:
        return None

    obstacle_map = {cell: turn_color[turn] for cell, turn in zip(obstacle_order, turns)}
    if len(obstacle_map) != len(obstacle_order):
        return None

    covered_turn_idx = None
    seen_turns: set[str] = set()
    for idx, cell in enumerate(obstacle_order):
        seen_turns.add(TURN_LEFT if obstacle_map[cell] == left_color else TURN_RIGHT)
        if len(seen_turns) == 2:
            covered_turn_idx = idx
            break
    assert covered_turn_idx is not None

    prefix_min = max(2 * cycle_len, corner_indices[covered_turn_idx] + 3)
    prefix_max = len(route) - 4
    blocked_lengths = {corner_idx + 1 for corner_idx in corner_indices} | {
        corner_idx + 2 for corner_idx in corner_indices
    }
    candidates = [
        visible_len
        for visible_len in range(prefix_min, prefix_max + 1)
        if visible_len not in blocked_lengths
    ]
    if not candidates:
        return None
    prefix_len = random.choice(candidates)

    full_path = set(route)
    objects_taken = set(obstacle_map) | set(end_block)
    extras = _place_extra_obstacles(shape, full_path, objects_taken, obstacle_colors)
    obstacle_map.update(extras)

    phase0 = random.randrange(len(cycle))
    inp = _render_grid(shape, bg, route, prefix_len, cycle, phase0, obstacle_map, end_block, end_color)
    expected = _render_grid(shape, bg, route, len(route), cycle, phase0, obstacle_map, end_block, end_color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    fg_counts = Counter(map(int, inp.ravel()))
    if fg_counts[bg] <= max(count for color, count in fg_counts.items() if color != bg):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")

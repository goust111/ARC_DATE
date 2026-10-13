from __future__ import annotations

import random

import numpy as np


WHITE = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

WALL_COLORS = (BLUE, RED, GREEN)
TURN_TAGS = ("L", "S", "R")
TURN_TO_COLOR = {"L": BLUE, "S": GREEN, "R": RED}
COLOR_TO_TURN = {BLUE: "L", GREEN: "S", RED: "R"}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
PATH_ATTEMPTS = 200
DFS_BRANCHING_LIMIT = 2000


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(pos: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _rot_left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (-int(dc), int(dr))


def _rot_right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (int(dc), -int(dr))


def _find_unique(g: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(g == int(color))
    assert pts.shape[0] == 1, f"expected one pixel of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _relative_turn(
    incoming: tuple[int, int], outgoing: tuple[int, int]
) -> str:
    if outgoing == incoming:
        return "S"
    if outgoing == _rot_left(incoming):
        return "L"
    if outgoing == _rot_right(incoming):
        return "R"
    raise AssertionError("path makes an invalid reversal")


def _ray_to_wall(
    g: np.ndarray,
    passable: np.ndarray,
    pos: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    r, c = pos
    dr, dc = direction
    dist = 0
    while True:
        r += int(dr)
        c += int(dc)
        dist += 1
        assert _in_bounds(g.shape, r, c), "maze must be enclosed by wall pixels"
        if not bool(passable[r, c]):
            return int(dist), int(g[r, c])


def _junction_turn(
    g: np.ndarray,
    passable: np.ndarray,
    pos: tuple[int, int],
    heading: tuple[int, int],
) -> str:
    probes = [
        ("L", _rot_left(heading), 0),
        ("S", heading, 1),
        ("R", _rot_right(heading), 2),
    ]
    best: tuple[tuple[int, int], str] | None = None
    for _label, direction, tie_rank in probes:
        dist, color = _ray_to_wall(g, passable, pos, direction)
        cand = (int(dist), int(tie_rank))
        if best is None or cand < best[0]:
            assert color in COLOR_TO_TURN, "instruction wall must be blue, red, or green"
            best = (cand, COLOR_TO_TURN[color])
    assert best is not None
    return best[1]


def _solve_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    passable = np.isin(g, (WHITE, YELLOW, ORANGE))
    start = _find_unique(g, YELLOW)
    end = _find_unique(g, ORANGE)

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    seen_states: set[tuple[tuple[int, int], tuple[int, int] | None]] = set()

    for _ in range(int(g.size * 4)):
        if cur == end:
            return path

        neighbors = [p for p in _neighbors4(cur, g.shape) if bool(passable[p])]
        if prev is None:
            assert len(neighbors) == 1, "start must have exactly one outgoing corridor"
            nxt = neighbors[0]
            heading = None
        else:
            heading = (int(cur[0] - prev[0]), int(cur[1] - prev[1]))
            options = [p for p in neighbors if p != prev]
            assert options, "path reached a dead end before the target"
            if len(options) == 1:
                nxt = options[0]
            else:
                turn = _junction_turn(g, passable, cur, heading)
                move_dir = {
                    "L": _rot_left(heading),
                    "S": heading,
                    "R": _rot_right(heading),
                }[turn]
                nxt = (int(cur[0] + move_dir[0]), int(cur[1] + move_dir[1]))
                assert nxt in options, "wall instruction points outside the available corridor"

        state = (cur, heading)
        assert state not in seen_states, "walk looped before reaching the target"
        seen_states.add(state)

        prev, cur = cur, nxt
        path.append(cur)

    raise AssertionError("walk did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Navigate a 1-cell-wide maze by reading nearby wall colors at junctions.

    Input structure:
    - White corridor cells form the walkable maze.
    - Blue, red, and green wall pixels surround the corridors.
    - A single yellow pixel is the start and a single orange pixel is the target.

    Rule:
    - Walk from the yellow start through the corridor.
    - At a junction with multiple forward options, look toward the left, front, and
      right rays until each first hits a wall pixel. The nearest wall color decides:
      blue -> turn left, red -> turn right, green -> go straight.
    - Paint every visited corridor cell, including the target, yellow.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r, c in _solve_path(g):
        out[int(r), int(c)] = YELLOW
    return out


def _node_to_cell(node: tuple[int, int]) -> tuple[int, int]:
    r, c = node
    return int(2 * r + 1), int(2 * c + 1)


def _cells_for_node_path(nodes: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for idx, node in enumerate(nodes):
        cr, cc = _node_to_cell(node)
        cells.add((cr, cc))
        if idx == 0:
            continue
        pr, pc = _node_to_cell(nodes[idx - 1])
        cells.add(((int(pr) + int(cr)) // 2, (int(pc) + int(cc)) // 2))
    return cells


def _node_neighbors(
    node: tuple[int, int], rows: int, cols: int
) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _reachable_unvisited(
    start: tuple[int, int],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> int:
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in _node_neighbors(cur, rows, cols):
            if nxt in seen or nxt in used:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return int(len(seen))


def _dfs_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    rows: int,
    cols: int,
    target_len: int,
    budget: list[int],
) -> bool:
    if len(path) == target_len:
        return True
    budget[0] -= 1
    if budget[0] <= 0:
        return False

    cur = path[-1]
    prev_dir = None
    if len(path) >= 2:
        prev = path[-2]
        prev_dir = (int(cur[0] - prev[0]), int(cur[1] - prev[1]))

    candidates: list[tuple[int, int, float, tuple[int, int]]] = []
    for nxt in _node_neighbors(cur, rows, cols):
        if nxt in used:
            continue
        move = (int(nxt[0] - cur[0]), int(nxt[1] - cur[1]))
        straight_bias = 1 if prev_dir is not None and move == prev_dir else 0
        exits = sum(1 for p in _node_neighbors(nxt, rows, cols) if p not in used and p != cur)
        candidates.append((straight_bias, exits, random.random(), nxt))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))

    for _straight, _exits, _rand, nxt in candidates:
        used.add(nxt)
        path.append(nxt)
        remaining = int(target_len - len(path))
        if remaining <= int(_reachable_unvisited(nxt, used, rows, cols)) - 1:
            if _dfs_path(path, used, rows, cols, target_len, budget):
                return True
        path.pop()
        used.remove(nxt)
    return False


def _build_main_path(
    rows: int, cols: int, target_len: int
) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}
        budget = [DFS_BRANCHING_LIMIT]
        if _dfs_path(path, used, rows, cols, target_len, budget):
            return path
    return None


def _max_branch_len(
    node: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> int:
    dr, dc = direction
    cur = node
    length = 0
    while True:
        cur = (int(cur[0] + dr), int(cur[1] + dc))
        if not (0 <= cur[0] < rows and 0 <= cur[1] < cols):
            break
        if cur in occupied:
            break
        length += 1
    return int(length)


def _try_add_branch(
    index: int,
    main_nodes: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> dict | None:
    node = main_nodes[index]
    incoming = (
        int(node[0] - main_nodes[index - 1][0]),
        int(node[1] - main_nodes[index - 1][1]),
    )
    outgoing = (
        int(main_nodes[index + 1][0] - node[0]),
        int(main_nodes[index + 1][1] - node[1]),
    )
    turn = _relative_turn(incoming, outgoing)
    rel_dirs = {"L": _rot_left(incoming), "S": incoming, "R": _rot_right(incoming)}
    branch_rels = [tag for tag in TURN_TAGS if tag != turn]
    random.shuffle(branch_rels)

    for branch_rel in branch_rels:
        branch_dir = rel_dirs[branch_rel]
        max_len = _max_branch_len(node, branch_dir, occupied, rows, cols)
        if max_len <= 0:
            continue
        branch_len = random.randint(1, min(3, max_len))
        branch_nodes: list[tuple[int, int]] = []
        cur = node
        for _ in range(branch_len):
            cur = (int(cur[0] + branch_dir[0]), int(cur[1] + branch_dir[1]))
            branch_nodes.append(cur)

        blocked_rel = next(tag for tag in TURN_TAGS if tag not in (turn, branch_rel))
        return {
            "index": int(index),
            "turn": turn,
            "blocked_dir": rel_dirs[blocked_rel],
            "branch_nodes": branch_nodes,
        }
    return None


def _desired_turns(count: int) -> list[str]:
    offset = random.randrange(len(TURN_TAGS))
    seq = [TURN_TAGS[(offset + i) % len(TURN_TAGS)] for i in range(count)]
    random.shuffle(seq)
    return seq


def _gen_attempt() -> np.ndarray | None:
    node_rows = random.randint(5, 9)
    node_cols = random.randint(5, 9)
    junction_count = random.randint(1, 4)
    max_path_len = min(16, node_rows * node_cols // 2 + 4)
    min_path_len = max(7, junction_count + 5)
    if min_path_len > max_path_len:
        return None

    desired_turns = _desired_turns(junction_count)
    main_len = random.randint(min_path_len, max_path_len)
    main_nodes = _build_main_path(node_rows, node_cols, main_len)
    if main_nodes is None:
        return None

    turn_by_index = {
        idx: _relative_turn(
            (
                int(main_nodes[idx][0] - main_nodes[idx - 1][0]),
                int(main_nodes[idx][1] - main_nodes[idx - 1][1]),
            ),
            (
                int(main_nodes[idx + 1][0] - main_nodes[idx][0]),
                int(main_nodes[idx + 1][1] - main_nodes[idx][1]),
            ),
        )
        for idx in range(1, len(main_nodes) - 1)
    }

    occupied = set(main_nodes)
    branch_records: list[dict] = []
    chosen_indices: set[int] = set()
    for desired in desired_turns:
        candidates = [
            idx
            for idx, tag in turn_by_index.items()
            if tag == desired and all(abs(int(idx) - int(prev)) >= 2 for prev in chosen_indices)
        ]
        random.shuffle(candidates)
        record = None
        for idx in candidates:
            attempt = _try_add_branch(idx, main_nodes, occupied, node_rows, node_cols)
            if attempt is not None:
                record = attempt
                break
        if record is None:
            return None
        branch_records.append(record)
        chosen_indices.add(int(record["index"]))
        occupied.update(record["branch_nodes"])

    height = int(2 * node_rows + 1)
    width = int(2 * node_cols + 1)
    g = np.random.choice(np.array(WALL_COLORS, dtype=int), size=(height, width))

    main_cells = _cells_for_node_path(main_nodes)
    all_corridor_cells = set(main_cells)
    for record in branch_records:
        branch_nodes = [main_nodes[int(record["index"])]] + list(record["branch_nodes"])
        all_corridor_cells.update(_cells_for_node_path(branch_nodes))

    for r, c in all_corridor_cells:
        g[int(r), int(c)] = WHITE

    start = _node_to_cell(main_nodes[0])
    end = _node_to_cell(main_nodes[-1])
    g[int(start[0]), int(start[1])] = YELLOW
    g[int(end[0]), int(end[1])] = ORANGE

    for record in branch_records:
        junction = _node_to_cell(main_nodes[int(record["index"])])
        dr, dc = record["blocked_dir"]
        wall_r = int(junction[0] + dr)
        wall_c = int(junction[1] + dc)
        assert not bool(np.isin(g[wall_r, wall_c], (WHITE, YELLOW, ORANGE)))
        g[wall_r, wall_c] = TURN_TO_COLOR[str(record["turn"])]

    passable = np.isin(g, (WHITE, YELLOW, ORANGE))
    start_neighbors = sum(1 for p in _neighbors4(start, g.shape) if bool(passable[p]))
    end_neighbors = sum(1 for p in _neighbors4(end, g.shape) if bool(passable[p]))
    if start_neighbors != 1 or end_neighbors != 1:
        return None

    corridor_degrees = [sum(1 for p in _neighbors4(cell, g.shape) if bool(passable[p])) for cell in all_corridor_cells]
    if any(int(deg) > 3 for deg in corridor_degrees):
        return None
    if sum(1 for deg in corridor_degrees if int(deg) == 3) != len(branch_records):
        return None

    expected = np.zeros_like(g, dtype=bool)
    for r, c in main_cells:
        expected[int(r), int(c)] = True

    try:
        solved = main(g)
    except AssertionError:
        return None
    if not bool(np.array_equal(solved == YELLOW, expected)):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-guided maze input")

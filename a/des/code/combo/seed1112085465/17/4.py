import random

import numpy as np


BG = 8
WALL = 5
PATH = 2

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_NAMES = ("up", "right", "down", "left")

LEGEND_COLORS = (0, 1, 3, 4, 6, 7, 9)
LEGEND_RADIUS_MAX = 4
LEGEND_CENTER_OFFSET = 5
LEGEND_BLOCK_CELLS = 5

CELL_ROWS_MIN = 10
CELL_ROWS_MAX = 13
CELL_COLS_MIN = 10
CELL_COLS_MAX = 13
ROUTE_ATTEMPTS = 180
GEN_ATTEMPTS = 400


def _move(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for name, vec in DIRS.items():
        if vec == (dr, dc):
            return name
    raise AssertionError(f"non-cardinal step: {a} -> {b}")


def _legend_center(corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if corner == "top_left":
        return LEGEND_CENTER_OFFSET, LEGEND_CENTER_OFFSET
    if corner == "top_right":
        return LEGEND_CENTER_OFFSET, w - 1 - LEGEND_CENTER_OFFSET
    if corner == "bottom_left":
        return h - 1 - LEGEND_CENTER_OFFSET, LEGEND_CENTER_OFFSET
    if corner == "bottom_right":
        return h - 1 - LEGEND_CENTER_OFFSET, w - 1 - LEGEND_CENTER_OFFSET
    raise AssertionError(f"bad corner {corner}")


def _priority_from_legend(grid: np.ndarray) -> list[str]:
    coords = np.argwhere((grid != BG) & (grid != WALL))
    assert coords.shape == (4, 2), "legend must contain exactly four colored pixels"

    candidates: list[tuple[tuple[int, int], dict[str, int]]] = []
    h, w = grid.shape
    for cr in range(h):
        for cc in range(w):
            distances: dict[str, int] = {}
            ok = True
            for r, c in coords:
                if r == cr and c > cc:
                    name = "right"
                    dist = int(c - cc)
                elif r == cr and c < cc:
                    name = "left"
                    dist = int(cc - c)
                elif c == cc and r < cr:
                    name = "up"
                    dist = int(cr - r)
                elif c == cc and r > cr:
                    name = "down"
                    dist = int(r - cr)
                else:
                    ok = False
                    break
                if dist <= 0 or name in distances:
                    ok = False
                    break
                distances[name] = dist
            if ok and set(distances) == set(DIR_NAMES) and len(set(distances.values())) == 4:
                candidates.append(((cr, cc), distances))

    assert len(candidates) == 1, "legend geometry must define a unique center"
    distances = candidates[0][1]
    return sorted(DIR_NAMES, key=lambda name: distances[name])


def _border_gaps(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    gaps: set[tuple[int, int]] = set()
    for c in range(w):
        if int(grid[0, c]) == BG:
            gaps.add((0, c))
        if int(grid[h - 1, c]) == BG:
            gaps.add((h - 1, c))
    for r in range(h):
        if int(grid[r, 0]) == BG:
            gaps.add((r, 0))
        if int(grid[r, w - 1]) == BG:
            gaps.add((r, w - 1))
    gaps_list = sorted(gaps)
    assert len(gaps_list) == 2, "maze must expose exactly one entrance and one exit"
    return gaps_list


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    priority: list[str],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    current = start
    previous: tuple[int, int] | None = None
    seen: set[tuple[tuple[int, int], tuple[int, int] | None]] = set()
    path = [start]

    for _ in range(h * w * 4):
        if current == goal:
            return path

        state = (current, previous)
        if state in seen:
            return None
        seen.add(state)

        next_cell = None
        for name in priority:
            dr, dc = DIRS[name]
            nr, nc = current[0] + dr, current[1] + dc
            if not _in_bounds((h, w), nr, nc):
                continue
            if int(grid[nr, nc]) != BG:
                continue
            if previous is not None and (nr, nc) == previous:
                continue
            next_cell = (nr, nc)
            break

        if next_cell is None:
            return None

        previous, current = current, next_cell
        path.append(current)

    return None


def _solve_path(grid: np.ndarray) -> list[tuple[int, int]]:
    priority = _priority_from_legend(grid)
    gaps = _border_gaps(grid)

    solutions = []
    for idx, start in enumerate(gaps):
        goal = gaps[1 - idx]
        path = _trace_path(grid, start, goal, priority)
        if path is not None:
            solutions.append(path)

    assert len(solutions) == 1, "exactly one gap must act as the entrance"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a corner legend that ranks the four directions, then follow the
    deterministic non-backtracking walk through the grey-walled maze and paint
    the visited corridor red from entrance to exit.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r, c in _solve_path(g):
        out[r, c] = PATH
    return out


def _blocked_cells(rows: int, cols: int, corner: str) -> set[tuple[int, int]]:
    r0 = 0 if "top" in corner else rows - LEGEND_BLOCK_CELLS
    c0 = 0 if "left" in corner else cols - LEGEND_BLOCK_CELLS
    blocked = set()
    for r in range(r0, r0 + LEGEND_BLOCK_CELLS):
        for c in range(c0, c0 + LEGEND_BLOCK_CELLS):
            blocked.add((r, c))
    return blocked


def _border_side(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = cell
    sides = []
    if r == 0:
        sides.append("top")
    if r == rows - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == cols - 1:
        sides.append("right")
    return sides


def _side_cells(rows: int, cols: int, side: str, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    if side == "top":
        for c in range(1, cols - 1):
            cell = (0, c)
            if cell not in blocked:
                cells.append(cell)
    elif side == "bottom":
        for c in range(1, cols - 1):
            cell = (rows - 1, c)
            if cell not in blocked:
                cells.append(cell)
    elif side == "left":
        for r in range(1, rows - 1):
            cell = (r, 0)
            if cell not in blocked:
                cells.append(cell)
    elif side == "right":
        for r in range(1, rows - 1):
            cell = (r, cols - 1)
            if cell not in blocked:
                cells.append(cell)
    else:
        raise AssertionError(f"bad side {side}")
    return cells


def _free_neighbors(
    cell: tuple[int, int],
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> list[tuple[str, tuple[int, int]]]:
    out = []
    for name in DIR_NAMES:
        nxt = _move(cell, name)
        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
            continue
        if nxt in blocked or nxt in visited:
            continue
        out.append((name, nxt))
    return out


def _random_route(
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], str, str] | None:
    available_sides = [side for side in ("top", "right", "bottom", "left") if _side_cells(rows, cols, side, blocked)]
    if len(available_sides) < 2:
        return None

    start_side = rng.choice(available_sides)
    start = rng.choice(_side_cells(rows, cols, start_side, blocked))
    min_len = rng.randint(max(10, (rows + cols) // 2), max(12, rows + cols + 1))
    max_len = rows * cols // 2 + 6

    for _ in range(ROUTE_ATTEMPTS):
        path = [start]
        visited = {start}
        last_dir: str | None = None

        while len(path) <= max_len:
            current = path[-1]
            current_sides = [side for side in _border_side(current, rows, cols) if side != start_side]
            if len(path) >= min_len and current_sides:
                moves = [_direction_between(a, b) for a, b in zip(path, path[1:])]
                turn_count = sum(moves[i] != moves[i - 1] for i in range(1, len(moves)))
                if len(set(moves)) >= 3 and turn_count >= 2:
                    return path, start_side, rng.choice(current_sides)
                break

            candidates = []
            for name, nxt in _free_neighbors(current, rows, cols, blocked, visited):
                sides = [side for side in _border_side(nxt, rows, cols) if side != start_side]
                is_finish = len(path) + 1 >= min_len and bool(sides)
                if sides and not is_finish:
                    continue

                onward = 0
                for _, nn in _free_neighbors(nxt, rows, cols, blocked, visited | {nxt}):
                    if nn != current:
                        onward += 1
                if onward == 0 and not is_finish:
                    continue

                score = rng.random()
                if last_dir is not None and name != last_dir:
                    score += 0.2
                score += 0.05 * onward
                if is_finish:
                    score += 0.3
                candidates.append((score, name, nxt))

            if not candidates:
                break

            candidates.sort(reverse=True)
            _, name, nxt = candidates[0]
            path.append(nxt)
            visited.add(nxt)
            last_dir = name

    return None


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _max_straight_branch_length(
    start: tuple[int, int],
    direction: str,
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    cap: int = 3,
) -> int:
    cur = start
    length = 0
    for _ in range(cap):
        nxt = _move(cur, direction)
        if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
            break
        if nxt in blocked or nxt in occupied:
            break
        length += 1
        cur = nxt
    return length


def _add_straight_branch(
    attachment: tuple[int, int],
    direction: str,
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    rng: random.Random,
) -> bool:
    max_len = _max_straight_branch_length(attachment, direction, rows, cols, blocked, occupied)
    if max_len == 0:
        return False

    length = rng.randint(1, max_len)
    prev = attachment
    for _ in range(length):
        nxt = _move(prev, direction)
        occupied.add(nxt)
        edges.add(_edge(prev, nxt))
        prev = nxt
    return True


def _route_graph(
    route: list[tuple[int, int]],
    priority: list[str],
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]] | None:
    rank = {name: idx for idx, name in enumerate(priority)}
    occupied = set(route)
    edges = {_edge(a, b) for a, b in zip(route, route[1:])}

    used_attachments: set[int] = set()
    asym_candidates: list[tuple[int, str]] = []
    extra_candidates: list[tuple[int, str]] = []

    for idx in range(1, len(route) - 1):
        cell = route[idx]
        back_dir = _direction_between(cell, route[idx - 1])
        forward_dir = _direction_between(cell, route[idx + 1])
        for direction in DIR_NAMES:
            if direction in (back_dir, forward_dir):
                continue
            if _max_straight_branch_length(cell, direction, rows, cols, blocked, occupied, cap=1) == 0:
                continue
            if rank[forward_dir] < rank[direction]:
                extra_candidates.append((idx, direction))
            if rank[forward_dir] < rank[direction] < rank[back_dir]:
                asym_candidates.append((idx, direction))

    if not asym_candidates:
        return None

    rng.shuffle(asym_candidates)
    asym_added = False
    for idx, direction in asym_candidates:
        if _add_straight_branch(route[idx], direction, rows, cols, blocked, occupied, edges, rng):
            used_attachments.add(idx)
            asym_added = True
            break
    if not asym_added:
        return None

    rng.shuffle(extra_candidates)
    target_extra = rng.randint(1, 3)
    extra_added = 0
    for idx, direction in extra_candidates:
        if extra_added >= target_extra:
            break
        if idx in used_attachments:
            continue
        if _add_straight_branch(route[idx], direction, rows, cols, blocked, occupied, edges, rng):
            used_attachments.add(idx)
            extra_added += 1

    return occupied, edges


def _actual_cell(cell: tuple[int, int]) -> tuple[int, int]:
    return 2 * cell[0] + 1, 2 * cell[1] + 1


def _gap_cell(cell: tuple[int, int], side: str, shape: tuple[int, int]) -> tuple[int, int]:
    ar, ac = _actual_cell(cell)
    h, w = shape
    if side == "top":
        return 0, ac
    if side == "bottom":
        return h - 1, ac
    if side == "left":
        return ar, 0
    if side == "right":
        return ar, w - 1
    raise AssertionError(f"bad side {side}")


def _route_actual_path(
    route: list[tuple[int, int]],
    start_side: str,
    end_side: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    actual = [_gap_cell(route[0], start_side, shape)]
    prev_actual = _actual_cell(route[0])
    actual.append(prev_actual)

    for cell in route[1:]:
        cur_actual = _actual_cell(cell)
        mid = ((prev_actual[0] + cur_actual[0]) // 2, (prev_actual[1] + cur_actual[1]) // 2)
        actual.extend([mid, cur_actual])
        prev_actual = cur_actual

    actual.append(_gap_cell(route[-1], end_side, shape))
    return actual


def _carve_grid(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    start_gap: tuple[int, int],
    end_gap: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, WALL, dtype=int)
    for cell in occupied:
        r, c = _actual_cell(cell)
        grid[r, c] = BG
    for a, b in edges:
        ra, ca = _actual_cell(a)
        rb, cb = _actual_cell(b)
        grid[(ra + rb) // 2, (ca + cb) // 2] = BG
    grid[start_gap] = BG
    grid[end_gap] = BG
    return grid


def _place_legend(grid: np.ndarray, corner: str, priority: list[str], rng: random.Random) -> None:
    center = _legend_center(corner, grid.shape)
    colors = rng.sample(list(LEGEND_COLORS), 4)
    color_for_direction = {direction: color for direction, color in zip(rng.sample(list(priority), len(priority)), colors)}

    for rank, direction in enumerate(priority, start=1):
        dr, dc = DIRS[direction]
        rr = center[0] + dr * rank
        cc = center[1] + dc * rank
        grid[rr, cc] = color_for_direction[direction]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(CELL_ROWS_MIN, CELL_ROWS_MAX)
    cols = rng.randint(CELL_COLS_MIN, CELL_COLS_MAX)
    shape = (2 * rows + 1, 2 * cols + 1)

    corner = rng.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    blocked = _blocked_cells(rows, cols, corner)

    priority = list(DIR_NAMES)
    rng.shuffle(priority)

    route_data = _random_route(rows, cols, blocked, rng)
    if route_data is None:
        return None
    route, start_side, end_side = route_data

    graph = _route_graph(route, priority, rows, cols, blocked, rng)
    if graph is None:
        return None
    occupied, edges = graph

    start_gap = _gap_cell(route[0], start_side, shape)
    end_gap = _gap_cell(route[-1], end_side, shape)
    grid = _carve_grid(shape, occupied, edges, start_gap, end_gap)
    _place_legend(grid, corner, priority, rng)

    expected_path = _route_actual_path(route, start_side, end_side, shape)
    try:
        if _priority_from_legend(grid) != priority:
            return None
        if _solve_path(grid) != expected_path:
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

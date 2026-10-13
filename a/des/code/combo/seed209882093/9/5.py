import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
SIDE_ORDER = ("U", "R", "D", "L")
OPPOSITE = {"U": "D", "R": "L", "D": "U", "L": "R"}
LEFT_OF = {"U": "L", "R": "U", "D": "R", "L": "D"}
RIGHT_OF = {v: k for k, v in LEFT_OF.items()}

GREY = 5
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _step(point: tuple[int, int], side: str, scale: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[side]
    return point[0] + dr * scale, point[1] + dc * scale


def _neg(vec: tuple[int, int]) -> tuple[int, int]:
    return -vec[0], -vec[1]


def _port_position(center: tuple[int, int], side: str) -> tuple[int, int]:
    return _step(center, side, 1)


def _turn_kind(in_heading: tuple[int, int], out_heading: tuple[int, int]) -> str:
    if out_heading == in_heading:
        return "straight"
    if out_heading == DIRS[LEFT_OF[_side_from_vec(in_heading)]]:
        return "left"
    if out_heading == DIRS[RIGHT_OF[_side_from_vec(in_heading)]]:
        return "right"
    raise ValueError("u-turns are not valid for this puzzle")


def _side_from_vec(vec: tuple[int, int]) -> str:
    for side, delta in DIRS.items():
        if delta == vec:
            return side
    raise ValueError(f"unknown vector {vec}")


def _trace_points(moves: list[str]) -> list[tuple[int, int]]:
    points = [(0, 0)]
    cur = (0, 0)
    for side in moves:
        cur = _step(cur, side)
        points.append(cur)
    return points


def _moves_close(moves: list[str]) -> bool:
    end = _trace_points(moves)[-1]
    return end == (0, 0)


def _fill_enclosed(canvas: np.ndarray, bg: int, fill_color: int) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not seen[r, c] and int(out[r, c]) == bg:
            seen[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for side in SIDE_ORDER:
            nr, nc = _step((r, c), side)
            push(nr, nc)

    out[(out == bg) & ~seen] = fill_color
    return out


def _render_moves(
    moves: list[str],
    bg: int,
    path_color: int,
    fill_color: int,
) -> np.ndarray:
    points = _trace_points(moves)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    shifted = [(r - r0, c - c0) for r, c in points]
    for r, c in shifted:
        out[r, c] = path_color

    if shifted[0] == shifted[-1]:
        out = _fill_enclosed(out, bg, fill_color)
    return out


def _detect_blocks(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    candidates: list[dict] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            corners = [
                int(g[r - 1, c - 1]),
                int(g[r - 1, c + 1]),
                int(g[r + 1, c - 1]),
                int(g[r + 1, c + 1]),
            ]
            if len(set(corners)) != 1:
                continue
            border_color = corners[0]
            side_vals = {side: int(g[r + dr, c + dc]) for side, (dr, dc) in DIRS.items()}
            ports = [side for side in SIDE_ORDER if side_vals[side] != border_color]
            if len(ports) != 2:
                continue
            if len({side_vals[side] for side in ports}) != 1:
                continue
            if any(side_vals[side] != border_color for side in SIDE_ORDER if side not in ports):
                continue
            path_color = side_vals[ports[0]]
            center_color = int(g[r, c])
            if path_color == bg or center_color in (border_color, path_color):
                continue
            candidates.append(
                {
                    "center": (r, c),
                    "border": border_color,
                    "path": path_color,
                    "center_color": center_color,
                    "ports": tuple(ports),
                }
            )

    if not candidates:
        raise ValueError("no valid 3x3 program blocks found")

    pair = Counter((b["border"], b["path"]) for b in candidates).most_common(1)[0][0]
    blocks = [b for b in candidates if (b["border"], b["path"]) == pair]
    centers = [b["center"] for b in blocks]
    if len(set(centers)) != len(centers):
        raise ValueError("duplicate block centers")
    return blocks, bg


def _ports_link(
    grid: np.ndarray,
    bg: int,
    center_a: tuple[int, int],
    side_a: str,
    center_b: tuple[int, int],
    side_b: str,
) -> bool:
    if OPPOSITE[side_a] != side_b:
        return False

    r1, c1 = _port_position(center_a, side_a)
    r2, c2 = _port_position(center_b, side_b)
    dr, dc = DIRS[side_a]

    if dr != 0:
        if c1 != c2:
            return False
        delta = r2 - r1
        if delta not in (dr, 2 * dr):
            return False
        if abs(delta) == 2 and int(grid[r1 + dr, c1]) != bg:
            return False
        return True

    if r1 != r2:
        return False
    delta = c2 - c1
    if delta not in (dc, 2 * dc):
        return False
    if abs(delta) == 2 and int(grid[r1, c1 + dc]) != bg:
        return False
    return True


def _decode_program(grid: np.ndarray) -> tuple[int, int, int, list[str]]:
    g = np.asarray(grid, dtype=int)
    blocks, bg = _detect_blocks(g)
    if len(blocks) < 2:
        raise ValueError("need at least two program blocks")

    path_colors = {b["path"] for b in blocks}
    if len(path_colors) != 1:
        raise ValueError("all ports must share one color")
    path_color = next(iter(path_colors))

    start_candidates = [i for i, b in enumerate(blocks) if b["center_color"] != bg]
    if len(start_candidates) != 1:
        raise ValueError("start block must be uniquely marked")
    start = start_candidates[0]
    fill_color = blocks[start]["center_color"]

    for i, block in enumerate(blocks):
        if i != start and block["center_color"] != bg:
            raise ValueError("only the start block may have a non-background center")

    adj: list[dict[int, tuple[str, str]]] = [dict() for _ in blocks]
    used_ports: set[tuple[int, str]] = set()

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            for side_i in blocks[i]["ports"]:
                for side_j in blocks[j]["ports"]:
                    if not _ports_link(g, bg, blocks[i]["center"], side_i, blocks[j]["center"], side_j):
                        continue
                    if (i, side_i) in used_ports or (j, side_j) in used_ports:
                        raise ValueError("ambiguous port linkage")
                    used_ports.add((i, side_i))
                    used_ports.add((j, side_j))
                    adj[i][j] = (side_i, side_j)
                    adj[j][i] = (side_j, side_i)

    degrees = [len(nbrs) for nbrs in adj]
    if degrees[start] != 1:
        raise ValueError("start block must be a chain endpoint")
    if any(d not in (1, 2) for d in degrees):
        raise ValueError("chain degrees must be 1 or 2")
    if sum(d == 1 for d in degrees) != 2:
        raise ValueError("chain must have exactly two endpoints")
    if sum(degrees) != 2 * (len(blocks) - 1):
        raise ValueError("graph is not a simple chain")

    order = [start]
    prev = -1
    while True:
        cur = order[-1]
        nxts = [n for n in adj[cur] if n != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("branch detected")
        nxt = nxts[0]
        order.append(nxt)
        prev = cur

    if len(order) != len(blocks) or len(set(order)) != len(blocks):
        raise ValueError("chain traversal did not cover all blocks")

    first_link_side = next(iter(adj[start].values()))[0]
    first_free = [side for side in blocks[start]["ports"] if side != first_link_side]
    if len(first_free) != 1:
        raise ValueError("start block must have exactly one free port")

    moves: list[str] = []
    current_heading = _neg(DIRS[first_free[0]])
    prev = -1

    for pos, idx in enumerate(order):
        if pos == 0:
            out_side = first_link_side
        else:
            in_side = adj[idx][prev][0]
            if current_heading != _neg(DIRS[in_side]):
                raise ValueError("incoming direction mismatch")
            if pos == len(order) - 1:
                free = [side for side in blocks[idx]["ports"] if side != in_side]
                if len(free) != 1:
                    raise ValueError("last block must have exactly one free port")
                out_side = free[0]
            else:
                nxt = order[pos + 1]
                out_side = adj[idx][nxt][0]

        next_heading = DIRS[out_side]
        _turn_kind(current_heading, next_heading)
        moves.append(out_side)
        current_heading = next_heading
        prev = idx

    return bg, path_color, fill_color, moves


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, path_color, fill_color, moves = _decode_program(g)
    return _render_moves(moves, bg, path_color, fill_color)


def _count_turns(moves: list[str]) -> int:
    return sum(moves[i] != moves[i - 1] for i in range(1, len(moves)))


def _search_open_walk(rng: random.Random, n_moves: int) -> list[str] | None:
    visited = {(0, 0)}
    moves: list[str] = []

    def dfs(pos: tuple[int, int], prev_side: str | None) -> bool:
        if len(moves) == n_moves:
            return True

        choices = list(SIDE_ORDER)
        rng.shuffle(choices)
        for side in choices:
            if prev_side is not None and side == OPPOSITE[prev_side]:
                continue
            nxt = _step(pos, side)
            if nxt in visited:
                continue
            visited.add(nxt)
            moves.append(side)
            if dfs(nxt, side):
                return True
            moves.pop()
            visited.remove(nxt)
        return False

    if dfs((0, 0), None) and _count_turns(moves) >= 1:
        return moves.copy()
    return None


def _search_closed_walk(rng: random.Random, n_moves: int) -> list[str] | None:
    if n_moves % 2 != 0:
        return None

    start = (0, 0)
    visited = {start}
    moves: list[str] = []

    def dfs(pos: tuple[int, int], prev_side: str | None) -> bool:
        step_no = len(moves)
        if step_no == n_moves:
            return pos == start

        choices = list(SIDE_ORDER)
        rng.shuffle(choices)
        for side in choices:
            if prev_side is not None and side == OPPOSITE[prev_side]:
                continue
            nxt = _step(pos, side)

            if step_no == n_moves - 1:
                if nxt != start:
                    continue
            else:
                if nxt == start or nxt in visited:
                    continue
                remaining = n_moves - (step_no + 1)
                dist = abs(nxt[0] - start[0]) + abs(nxt[1] - start[1])
                if dist > remaining or (remaining - dist) % 2 != 0:
                    continue

            moves.append(side)
            added = nxt != start
            if added:
                visited.add(nxt)
            if dfs(nxt, side):
                return True
            if added:
                visited.remove(nxt)
            moves.pop()
        return False

    if dfs(start, None) and _count_turns(moves) >= 3:
        return moves.copy()
    return None


def _program_ports(start_heading: str, moves: list[str]) -> list[tuple[str, str]]:
    ports: list[tuple[str, str]] = []
    for i, out_side in enumerate(moves):
        in_side = OPPOSITE[start_heading] if i == 0 else OPPOSITE[moves[i - 1]]
        if in_side == out_side:
            raise ValueError("a block cannot use the same side twice")
        ports.append((in_side, out_side))
    return ports


def _actual_centers(rng: random.Random, moves: list[str]) -> list[tuple[int, int]]:
    centers = [(0, 0)]
    cur = (0, 0)
    for side in moves[:-1]:
        cur = _step(cur, side, rng.choice((3, 4)))
        centers.append(cur)
    return centers


def _paint_block(
    grid: np.ndarray,
    center: tuple[int, int],
    ports: tuple[str, str],
    border_color: int,
    path_color: int,
    center_color: int,
    bg: int,
) -> bool:
    r, c = center
    if r - 1 < 0 or c - 1 < 0 or r + 1 >= grid.shape[0] or c + 1 >= grid.shape[1]:
        return False
    if np.any(grid[r - 1 : r + 2, c - 1 : c + 2] != bg):
        return False

    grid[r - 1 : r + 2, c - 1 : c + 2] = border_color
    grid[r, c] = center_color
    for side in ports:
        pr, pc = _port_position(center, side)
        grid[pr, pc] = path_color
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    closed = rng.random() < 0.45
    if closed:
        n_moves = rng.choice([8, 10, 12, 14])
        moves = _search_closed_walk(rng, n_moves)
    else:
        n_moves = rng.randint(5, 13)
        moves = _search_open_walk(rng, n_moves)
    if moves is None:
        return None

    valid_starts = [moves[0], LEFT_OF[moves[0]], RIGHT_OF[moves[0]]]
    if closed:
        valid_starts = [side for side in valid_starts if side != moves[-1]]
    if not valid_starts:
        return None
    start_heading = rng.choice(valid_starts)

    try:
        ports = _program_ports(start_heading, moves)
    except ValueError:
        return None

    centers = _actual_centers(rng, moves)
    rows = [r for r, _ in centers]
    cols = [c for _, c in centers]
    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    rmin = min(r - 1 for r in rows)
    rmax = max(r + 1 for r in rows)
    cmin = min(c - 1 for c in cols)
    cmax = max(c + 1 for c in cols)

    h = rmax - rmin + 1 + top + bottom
    w = cmax - cmin + 1 + left + right
    while h * w <= 14 * len(moves) + 8:
        if h <= w:
            h += 1
        else:
            w += 1

    shift_r = top - rmin
    shift_c = left - cmin
    centers = [(r + shift_r, c + shift_c) for r, c in centers]

    bg_choices = [color for color in range(10) if color != GREY]
    bg = rng.choice(bg_choices)
    remaining = [color for color in range(10) if color not in (bg, GREY)]
    path_color, fill_color = rng.sample(remaining, 2)

    grid = np.full((h, w), bg, dtype=int)
    for i, (center, block_ports) in enumerate(zip(centers, ports)):
        center_color = fill_color if i == 0 else bg
        if not _paint_block(grid, center, block_ports, GREY, path_color, center_color, bg):
            return None

    if _bg_color(grid) != bg:
        return None

    expected = _render_moves(moves, bg, path_color, fill_color)
    if closed and not np.any(expected == fill_color):
        return None
    if not closed and np.any(expected == fill_color):
        return None

    try:
        bg2, path2, fill2, moves2 = _decode_program(grid)
    except Exception:
        return None

    if (bg2, path2, fill2) != (bg, path_color, fill_color):
        return None
    if moves2 != moves:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retries")

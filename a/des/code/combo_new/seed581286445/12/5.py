from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# --------------- solver helpers ---------------


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_bg(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _wall_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    repeated = [color for color, cnt in counts.items() if color != bg and cnt > 1]
    assert repeated, "expected a repeated wall color"
    repeated.sort(key=lambda color: (-counts[color], color))
    return int(repeated[0])


def _border_sequence(g: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    pts = [
        (r, c, int(g[r, c]))
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and int(g[r, c]) not in (bg, wall)
    ]
    assert pts, "expected a border color sequence"
    rows = {r for r, _, _ in pts}
    cols = {c for _, c, _ in pts}
    if len(rows) == 1:
        return sorted(pts, key=lambda item: item[1])
    assert len(cols) == 1, "border sequence must be a single row or a single column"
    return sorted(pts, key=lambda item: item[0])


def _room_boxes(open_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = open_mask.shape
    roomy = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(open_mask[r : r + 2, c : c + 2].all()):
                roomy[r : r + 2, c : c + 2] = True

    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(roomy):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if r1 - r0 + 1 <= 1 or c1 - c0 + 1 <= 1:
            continue
        if not bool(open_mask[r0 : r1 + 1, c0 : c1 + 1].all()):
            continue
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        boxes.append((r0, c0, r1, c1))

    boxes.sort()
    return boxes


def _component_mask_from(start: tuple[int, int], open_mask: np.ndarray) -> np.ndarray:
    h, w = open_mask.shape
    comp = np.zeros((h, w), dtype=bool)
    sr, sc = start
    assert bool(open_mask[sr, sc])
    q = deque([(sr, sc)])
    comp[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and bool(open_mask[nr, nc]) and not bool(comp[nr, nc]):
                comp[nr, nc] = True
                q.append((nr, nc))
    return comp


def _room_id_grid(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    room_ids = np.full(shape, -1, dtype=int)
    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        room_ids[r0 : r1 + 1, c0 : c1 + 1] = idx
    return room_ids


def _room_exits(
    room_id: int,
    box: tuple[int, int, int, int],
    component_mask: np.ndarray,
    room_ids: np.ndarray,
) -> list[tuple[int, int]]:
    exits: set[tuple[int, int]] = set()
    h, w = component_mask.shape
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(h, w, nr, nc):
                    continue
                if bool(component_mask[nr, nc]) and int(room_ids[nr, nc]) == -1:
                    exits.add((nr, nc))
    ordered = sorted(exits)
    assert ordered, "each path room must have at least one exit"
    return ordered


def _rot_cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _rot_ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _traverse_from_exit(
    room_id: int,
    exit_cell: tuple[int, int],
    component_mask: np.ndarray,
    room_ids: np.ndarray,
) -> tuple[int, tuple[int, int]]:
    h, w = component_mask.shape
    adj_room_cells = []
    er, ec = exit_cell
    for dr, dc in DIRS4:
        nr, nc = er + dr, ec + dc
        if _in_bounds(h, w, nr, nc) and int(room_ids[nr, nc]) == room_id:
            adj_room_cells.append((nr, nc))
    assert adj_room_cells, "exit must touch its room"

    prev = adj_room_cells[0]
    cur = exit_cell
    direction = (cur[0] - prev[0], cur[1] - prev[1])
    for _ in range(h * w * 4):
        next_rooms = {
            int(room_ids[nr, nc])
            for dr, dc in DIRS4
            for nr, nc in [(cur[0] + dr, cur[1] + dc)]
            if _in_bounds(h, w, nr, nc) and int(room_ids[nr, nc]) >= 0 and int(room_ids[nr, nc]) != room_id
        }
        if next_rooms:
            assert len(next_rooms) == 1, "corridor step should reach one next room"
            return next(iter(next_rooms)), cur

        corridor_dirs = []
        for dr, dc in DIRS4:
            nr, nc = cur[0] + dr, cur[1] + dc
            if _in_bounds(h, w, nr, nc) and bool(component_mask[nr, nc]) and int(room_ids[nr, nc]) == -1:
                corridor_dirs.append((dr, dc))
        assert corridor_dirs, "corridor traversal hit a dead end"

        back = (-direction[0], -direction[1])
        straight = direction
        right = _rot_cw(direction)
        left = _rot_ccw(direction)

        if len(corridor_dirs) <= 2:
            next_dir = next(d for d in corridor_dirs if d != back)
        elif len(corridor_dirs) == 3:
            for candidate in (right, straight, left):
                if candidate in corridor_dirs and candidate != back:
                    next_dir = candidate
                    break
            else:
                raise AssertionError("T junction has no valid outgoing direction")
        else:
            for candidate in (straight, right, left):
                if candidate in corridor_dirs and candidate != back:
                    next_dir = candidate
                    break
            else:
                raise AssertionError("cross junction has no straight continuation")

        cur = (cur[0] + next_dir[0], cur[1] + next_dir[1])
        direction = next_dir

    raise AssertionError("corridor traversal exceeded guard")


def _choose_departure(
    room_id: int,
    arrival_exit: tuple[int, int] | None,
    previous_room: int | None,
    room_exits: list[list[tuple[int, int]]],
    component_mask: np.ndarray,
    room_ids: np.ndarray,
) -> tuple[tuple[int, int], int, tuple[int, int]]:
    candidates = list(room_exits[room_id])
    if arrival_exit is not None:
        candidates = [cell for cell in candidates if cell != arrival_exit]
    assert candidates, "room must have an exit to continue"

    results: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for exit_cell in candidates:
        next_room, arrival = _traverse_from_exit(room_id, exit_cell, component_mask, room_ids)
        results.append((exit_cell, next_room, arrival))

    if previous_room is not None:
        non_back = [item for item in results if item[1] != previous_room]
        if len(non_back) == 1:
            return non_back[0]
        if non_back:
            results = non_back

    if len(results) == 1:
        return results[0]
    if len({item[1] for item in results}) == 1:
        return results[0]
    raise AssertionError("ambiguous room departure")


def _fill_box(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow maze corridors from the marked start room and fill each reached room with the next border-sequence color.

    Input structure:
    - Black background surrounds a grey one-pixel-thick maze whose open area contains
      one-pixel-wide corridors and larger rectangular rooms.
    - Exactly one room contains a single non-grey, non-black marker pixel giving the
      start room and its fill color.
    - On one grid edge, a contiguous row or column of distinct colored pixels forms
      the instruction sequence read left-to-right or top-to-bottom.

    Rule:
    - Fill the start room with the marker color.
    - Then consume the border sequence one color at a time. From the current room,
      leave through its corridor opening, walk until the next room, and at each
      corridor junction choose the direction by local shape: on a T junction turn
      right relative to the entering direction; on a four-way cross continue straight.
    - Fill each newly reached room with the current instruction color while leaving
      walls, corridors, and the outer sequence unchanged.

    Key insight:
    - The colored border pixels encode only the room-fill order; the actual route is
      determined by geometric maze parsing plus local junction-shape turning rules.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_bg(g)
    wall = _wall_color(g, bg)
    seq_pts = _border_sequence(g, bg, wall)
    seq_colors = [color for _, _, color in seq_pts]
    seq_pos = {(r, c) for r, c, _ in seq_pts}

    special_cells = [
        (r, c, int(g[r, c]))
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in (bg, wall) and (r, c) not in seq_pos
    ]
    assert len(special_cells) == 1, "expected exactly one start marker"
    start_r, start_c, start_color = special_cells[0]

    open_mask = g != wall
    boxes = _room_boxes(open_mask)
    assert boxes, "expected rectangular rooms"
    room_ids = _room_id_grid(g.shape, boxes)
    start_room = int(room_ids[start_r, start_c])
    assert start_room >= 0, "start marker must sit inside a room"

    component_mask = _component_mask_from((start_r, start_c), open_mask)
    room_exits = [_room_exits(idx, box, component_mask, room_ids) for idx, box in enumerate(boxes)]

    out = g.copy()
    _fill_box(out, boxes[start_room], start_color)

    current_room = start_room
    previous_room: int | None = None
    arrival_exit: tuple[int, int] | None = None
    for color in seq_colors:
        _, next_room, next_arrival = _choose_departure(
            current_room,
            arrival_exit,
            previous_room,
            room_exits,
            component_mask,
            room_ids,
        )
        _fill_box(out, boxes[next_room], color)
        previous_room, current_room, arrival_exit = current_room, next_room, next_arrival

    return out


# --------------- generator helpers ---------------

GEN_RETRIES = 400
PITCH = 6
NODE_MARGIN = 3


def _sample_spans(size: int) -> tuple[int, int]:
    if size == 2:
        return random.choice([(0, 1), (1, 0)])
    if size == 3:
        return (1, 1)
    assert size == 4
    return random.choice([(1, 2), (2, 1)])


def _neighbors(node: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = node
    return [(nr, nc) for dr, dc in DIRS4 for nr, nc in [(r + dr, c + dc)] if 0 <= nr < rows and 0 <= nc < cols]


def _gen_path(rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        cur = path[-1]
        opts = [n for n in _neighbors(cur, rows, cols) if n not in used]
        random.shuffle(opts)
        opts.sort(key=lambda n: sum(1 for m in _neighbors(n, rows, cols) if m not in used), reverse=True)
        for nxt in opts:
            used.add(nxt)
            path.append(nxt)
            if dfs(path, used):
                return True
            path.pop()
            used.remove(nxt)
        return False

    for _ in range(80):
        start = random.choice(cells)
        path = [start]
        used = {start}
        if dfs(path, used):
            return path
    return None


def _turn_kind(
    prev_node: tuple[int, int],
    cur_node: tuple[int, int],
    next_node: tuple[int, int],
) -> str:
    d1 = (cur_node[0] - prev_node[0], cur_node[1] - prev_node[1])
    d2 = (next_node[0] - cur_node[0], next_node[1] - cur_node[1])
    if d2 == d1:
        return "straight"
    if d2 == _rot_cw(d1):
        return "right"
    return "left"


def _node_to_coord(maze_top: int, maze_left: int, node: tuple[int, int]) -> tuple[int, int]:
    return (maze_top + NODE_MARGIN + node[0] * PITCH, maze_left + NODE_MARGIN + node[1] * PITCH)


def _carve_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = BG
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = BG


def _carve_stub(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int]) -> None:
    r, c = center
    dr, dc = direction
    grid[r + dr, c + dc] = BG


def _room_box(center: tuple[int, int]) -> tuple[int, int, int, int]:
    height = random.randint(2, 4)
    width = random.randint(2, 4)
    up, down = _sample_spans(height)
    left, right = _sample_spans(width)
    return (center[0] - up, center[1] - left, center[0] + down, center[1] + right)


def _expected_from_boxes(
    inp: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    room_ids_in_order: list[int],
    colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for room_id, color in zip(room_ids_in_order, colors, strict=False):
        _fill_box(out, boxes[room_id], color)
    return out


def _gen_palette(
    grid: np.ndarray,
    colors: list[int],
    side: str,
) -> None:
    h, w = grid.shape
    n = len(colors)
    if side in ("top", "bottom"):
        start = random.randint(1, w - n - 1)
        row = 0 if side == "top" else h - 1
        for offset, color in enumerate(colors):
            grid[row, start + offset] = color
    else:
        start = random.randint(1, h - n - 1)
        col = 0 if side == "left" else w - 1
        for offset, color in enumerate(colors):
            grid[start + offset, col] = color


def _gen_attempt() -> np.ndarray | None:
    required_mode = random.choice(["T", "X"])
    coarse_rows, coarse_cols = random.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
    max_rooms = 6 if coarse_rows * coarse_cols >= 12 else 5
    room_count = random.randint(3, max_rooms)

    gap_choices = [0, 1, 1, 2]
    gaps = [random.choice(gap_choices) for _ in range(room_count - 1)]
    if all(g == 0 for g in gaps):
        gaps[random.randrange(len(gaps))] = 1

    room_path_indices = [0]
    for gap in gaps:
        room_path_indices.append(room_path_indices[-1] + gap + 1)
    path_len = room_path_indices[-1] + 1
    if path_len > coarse_rows * coarse_cols:
        return None

    path = _gen_path(coarse_rows, coarse_cols, path_len)
    if path is None:
        return None

    room_index_set = set(room_path_indices)
    eligible_t: list[int] = []
    eligible_x: list[int] = []
    for idx in range(1, path_len - 1):
        if idx in room_index_set:
            continue
        kind = _turn_kind(path[idx - 1], path[idx], path[idx + 1])
        if kind == "right":
            eligible_t.append(idx)
        elif kind == "straight":
            eligible_x.append(idx)

    if required_mode == "T" and not eligible_t:
        return None
    if required_mode == "X" and not eligible_x:
        return None

    maze_h = 2 * NODE_MARGIN + (coarse_rows - 1) * PITCH + 1
    maze_w = 2 * NODE_MARGIN + (coarse_cols - 1) * PITCH + 1
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    palette_side = random.choice(["top", "bottom", "left", "right"])
    if palette_side == "top":
        top_margin += 2
    elif palette_side == "bottom":
        bottom_margin += 2
    elif palette_side == "left":
        left_margin += 2
    else:
        right_margin += 2

    h = top_margin + maze_h + bottom_margin
    w = left_margin + maze_w + right_margin
    if h > 30 or w > 30:
        return None

    maze_top = top_margin
    maze_left = left_margin
    grid = np.full((h, w), BG, dtype=int)
    grid[maze_top : maze_top + maze_h, maze_left : maze_left + maze_w] = WALL

    centers = [_node_to_coord(maze_top, maze_left, node) for node in path]
    room_boxes_by_path: dict[int, tuple[int, int, int, int]] = {}
    for idx in room_path_indices:
        room_boxes_by_path[idx] = _room_box(centers[idx])

    boxes = [room_boxes_by_path[idx] for idx in room_path_indices]
    for r0, c0, r1, c1 in boxes:
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG

    for a, b in zip(centers, centers[1:], strict=False):
        _carve_line(grid, a, b)

    decorated: set[int] = set()
    if required_mode == "T":
        decorated.add(random.choice(eligible_t))
    else:
        decorated.add(random.choice(eligible_x))

    for idx in eligible_t:
        if idx not in decorated and random.random() < 0.3:
            decorated.add(idx)
    for idx in eligible_x:
        if idx not in decorated and random.random() < 0.3:
            decorated.add(idx)

    for idx in decorated:
        prev_node, cur_node, next_node = path[idx - 1], path[idx], path[idx + 1]
        enter = (cur_node[0] - prev_node[0], cur_node[1] - prev_node[1])
        kind = _turn_kind(prev_node, cur_node, next_node)
        center = centers[idx]
        if kind == "right":
            _carve_stub(grid, center, enter)
        elif kind == "straight":
            _carve_stub(grid, center, _rot_cw(enter))
            _carve_stub(grid, center, _rot_ccw(enter))

    colors = random.sample(COLOR_POOL, room_count)
    start_color = colors[0]
    sequence_colors = colors[1:]

    start_box = boxes[0]
    sr = random.randint(start_box[0], start_box[2])
    sc = random.randint(start_box[1], start_box[3])
    grid[sr, sc] = start_color

    _gen_palette(grid, sequence_colors, palette_side)
    expected = _expected_from_boxes(grid, boxes, list(range(room_count)), colors)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-room ARC input")

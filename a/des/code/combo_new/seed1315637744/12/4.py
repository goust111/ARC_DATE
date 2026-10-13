import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _wall_lines(grid: np.ndarray, axis: int) -> list[int]:
    lines: list[int] = []
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        vals = set(map(int, np.unique(line)))
        if vals.issubset({BLACK, BLUE}) and BLUE in vals:
            lines.append(idx)
    assert lines and lines[0] == 0 and lines[-1] == length - 1, "expected blue outer walls"
    assert all(lines[i + 1] - lines[i] >= 2 for i in range(len(lines) - 1)), "rooms must have positive size"
    return lines


def _spans(lines: list[int]) -> list[tuple[int, int]]:
    spans = [(lines[i] + 1, lines[i + 1] - 1) for i in range(len(lines) - 1)]
    assert spans and all(lo <= hi for lo, hi in spans)
    return spans


def _nearest_side(coords: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h, w = shape
    margins = {
        "up": min(rows),
        "down": h - 1 - max(rows),
        "left": min(cols),
        "right": w - 1 - max(cols),
    }
    best = min(margins.values())
    sides = [side for side, dist in margins.items() if dist == best]
    assert len(sides) == 1, "object must have a unique nearest wall"
    return sides[0]


def _extract_rooms(
    grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> dict[tuple[int, int], dict]:
    rooms: dict[tuple[int, int], dict] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, counts = np.unique(patch, return_counts=True)
            colors = list(map(int, vals))
            assert colors and all(color not in {BLACK, BLUE} for color in colors), "room interior colors are invalid"
            base = int(vals[int(np.argmax(counts))])
            mask = patch != base
            obj = None
            if np.any(mask):
                comps = _components(mask)
                assert len(comps) == 1, "each room may contain at most one anomaly object"
                comp = comps[0]
                assert 2 <= len(comp) <= 4, "object must have 2-4 cells"
                obj_colors = {int(patch[r, c]) for r, c in comp}
                assert len(obj_colors) == 1, "object must be monochrome"
                color = obj_colors.pop()
                assert color != base
                obj = {
                    "color": color,
                    "coords_rel": comp,
                    "coords_abs": [(r0 + r, c0 + c) for r, c in comp],
                }
            rooms[(ri, ci)] = {
                "base": base,
                "shape": patch.shape,
                "bbox": (r0, c0, r1, c1),
                "object": obj,
            }
    return rooms


def _gap_graph(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> dict[tuple[int, int], list[dict]]:
    edges = {(ri, ci): [] for ri in range(len(row_spans)) for ci in range(len(col_spans))}
    gap_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLACK)}
    seen_gaps: set[tuple[int, int]] = set()
    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)

    for r, c in gap_cells:
        assert (r in wall_row_set) ^ (c in wall_col_set), "gap must lie on exactly one wall segment"

    for wr_idx in range(1, len(wall_rows) - 1):
        wr = wall_rows[wr_idx]
        for ci, (c0, c1) in enumerate(col_spans):
            blacks = [c for c in range(c0, c1 + 1) if int(grid[wr, c]) == BLACK]
            assert len(blacks) <= 1, "wall segment may have at most one gap"
            if not blacks:
                continue
            c = blacks[0]
            top = (wr_idx - 1, ci)
            bottom = (wr_idx, ci)
            edges[top].append({"neighbor": bottom, "side": "down", "gap": (wr, c)})
            edges[bottom].append({"neighbor": top, "side": "up", "gap": (wr, c)})
            seen_gaps.add((wr, c))

    for wc_idx in range(1, len(wall_cols) - 1):
        wc = wall_cols[wc_idx]
        for ri, (r0, r1) in enumerate(row_spans):
            blacks = [r for r in range(r0, r1 + 1) if int(grid[r, wc]) == BLACK]
            assert len(blacks) <= 1, "wall segment may have at most one gap"
            if not blacks:
                continue
            r = blacks[0]
            left = (ri, wc_idx - 1)
            right = (ri, wc_idx)
            edges[left].append({"neighbor": right, "side": "right", "gap": (r, wc)})
            edges[right].append({"neighbor": left, "side": "left", "gap": (r, wc)})
            seen_gaps.add((r, wc))

    assert seen_gaps == gap_cells, "all black cells must be valid gaps"
    return edges


def _verify_route(
    start: tuple[int, int], rooms: dict[tuple[int, int], dict], edges: dict[tuple[int, int], list[dict]]
) -> None:
    obj = rooms[start]["object"]
    assert obj is not None
    forward = list(edges[start])
    assert len(forward) == 1, "each object room must have exactly one gap"
    assert forward[0]["side"] == _nearest_side(obj["coords_rel"], rooms[start]["shape"])

    color = int(obj["color"])
    visited = {start}
    prev = start
    current = forward[0]["neighbor"]

    while True:
        assert current not in visited, "gap graph must not loop before reaching a match"
        visited.add(current)
        if int(rooms[current]["base"]) == color:
            return
        next_edges = [edge for edge in edges[current] if edge["neighbor"] != prev]
        assert len(next_edges) == 1, "a non-matching transit room must have one onward gap"
        prev, current = current, next_edges[0]["neighbor"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear each misplaced room object after verifying that it escapes through wall gaps to a room with the matching floor color.

    Input structure:
    - Blue one-cell-thick wall rows and wall columns partition the grid into rectangular rooms.
    - Room interiors are filled mostly by one non-blue floor color, with some walls containing single black gap cells.
    - Some rooms contain one small 4-connected monochrome object of size 2-4 whose color differs from that room's floor color.
    - Each object room has exactly one gap on its nearest wall, and non-matching transit rooms along the route have onward gaps.

    Rule:
    - Detect the rectangular room layout from the blue walls and identify each room's dominant non-blue floor color.
    - In every room, treat the unique small non-floor component as the escaping object.
    - Follow the room-to-room chain induced by black wall gaps, starting from the object's unique gap and continuing until a room whose floor color matches the object color is reached.
    - Restore the object's source cells to that source room's floor color; the eventual destination room already has the matching color, so the object visually disappears there and the black gaps remain unchanged.

    Key insight:
    The visible output change is only the removal of anomalous room-local objects, but that removal is justified by a hidden routing process through black wall gaps until the object reaches a room whose floor color matches its own color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    wall_rows = _wall_lines(g, axis=0)
    wall_cols = _wall_lines(g, axis=1)
    row_spans = _spans(wall_rows)
    col_spans = _spans(wall_cols)
    rooms = _extract_rooms(g, row_spans, col_spans)
    edges = _gap_graph(g, wall_rows, wall_cols, row_spans, col_spans)

    out = g.copy()
    for room_id, room in rooms.items():
        obj = room["object"]
        if obj is None:
            continue
        _verify_route(room_id, rooms, edges)
        for r, c in obj["coords_abs"]:
            out[r, c] = int(room["base"])
    return out


GEN_RETRIES = 300
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (4, 6)
MAX_OBJECTS = 3
MAX_CHAIN_LEN = 4
SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
)


def _neighbors(room: tuple[int, int], room_rows: int, room_cols: int) -> list[tuple[int, int]]:
    r, c = room
    nbrs: list[tuple[int, int]] = []
    if r > 0:
        nbrs.append((r - 1, c))
    if r + 1 < room_rows:
        nbrs.append((r + 1, c))
    if c > 0:
        nbrs.append((r, c - 1))
    if c + 1 < room_cols:
        nbrs.append((r, c + 1))
    random.shuffle(nbrs)
    return nbrs


def _path_with_steps(
    start: tuple[int, int],
    steps: int,
    blocked: set[tuple[int, int]],
    room_rows: int,
    room_cols: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    seen = set(blocked)
    seen.add(start)

    def dfs(current: tuple[int, int], remaining: int) -> list[tuple[int, int]] | None:
        if remaining == 0:
            return list(path)
        for nxt in _neighbors(current, room_rows, room_cols):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            found = dfs(nxt, remaining - 1)
            if found is not None:
                return found
            path.pop()
            seen.remove(nxt)
        return None

    return dfs(start, steps)


def _sample_path(
    room_rows: int, room_cols: int, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(room_rows) for c in range(room_cols) if (r, c) not in blocked]
    if len(free) < 2:
        return None

    max_steps = min(MAX_CHAIN_LEN, len(free) - 1)
    lengths = list(range(1, max_steps + 1))
    random.shuffle(lengths)
    if max_steps >= 2:
        if random.random() < 0.5:
            lengths.remove(1)
            lengths.insert(0, 1)
        else:
            long_len = random.randint(2, max_steps)
            lengths.remove(long_len)
            lengths.insert(0, long_len)

    random.shuffle(free)
    for steps in lengths:
        for start in free:
            path = _path_with_steps(start, steps, blocked, room_rows, room_cols)
            if path is not None:
                return path
    return None


def _step_side(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1:
        return "up"
    if dr == 1:
        return "down"
    if dc == -1:
        return "left"
    if dc == 1:
        return "right"
    raise AssertionError("rooms must be adjacent")


def _choose_sizes(count: int) -> list[int]:
    lo, hi = ROOM_SIZE_RANGE
    return [random.randint(lo, hi) for _ in range(count)]


def _wall_positions(sizes: list[int]) -> list[int]:
    pos = 0
    walls = [0]
    for size in sizes:
        pos += size + 1
        walls.append(pos)
    return walls


def _place_object(room_h: int, room_w: int, side: str) -> list[tuple[int, int]]:
    shapes = list(SHAPES)
    random.shuffle(shapes)
    for shape in shapes:
        sh, sw = shape.shape
        if side in {"up", "down"}:
            if room_h - sh < 1 or room_w - sw < 2:
                continue
            row0 = 0 if side == "up" else room_h - sh
            col0 = random.randint(1, room_w - sw - 1)
        else:
            if room_w - sw < 1 or room_h - sh < 2:
                continue
            row0 = random.randint(1, room_h - sh - 1)
            col0 = 0 if side == "left" else room_w - sw
        coords = [(row0 + r, col0 + c) for r in range(sh) for c in range(sw) if int(shape[r, c]) == 1]
        if _nearest_side(coords, (room_h, room_w)) == side:
            return coords
    raise AssertionError("failed to place a valid object")


def _random_gap_for_edge(
    a: tuple[int, int],
    b: tuple[int, int],
    wall_rows: list[int],
    wall_cols: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    if a[0] != b[0]:
        ci = a[1]
        wr = wall_rows[max(a[0], b[0])]
        c0, c1 = col_spans[ci]
        return wr, random.randint(c0, c1)
    ri = a[0]
    wc = wall_cols[max(a[1], b[1])]
    r0, r1 = row_spans[ri]
    return random.randint(r0, r1), wc


def _source_gap(
    room_id: tuple[int, int],
    next_room: tuple[int, int],
    coords_rel: list[tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    side = _step_side(room_id, next_room)
    r0, c0 = row_spans[room_id[0]][0], col_spans[room_id[1]][0]
    abs_coords = [(r0 + r, c0 + c) for r, c in coords_rel]
    if side == "up":
        return wall_rows[room_id[0]], random.choice([c for _, c in abs_coords])
    if side == "down":
        return wall_rows[room_id[0] + 1], random.choice([c for _, c in abs_coords])
    if side == "left":
        return random.choice([r for r, _ in abs_coords]), wall_cols[room_id[1]]
    return random.choice([r for r, _ in abs_coords]), wall_cols[room_id[1] + 1]


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)
    heights = _choose_sizes(room_rows)
    widths = _choose_sizes(room_cols)
    wall_rows = _wall_positions(heights)
    wall_cols = _wall_positions(widths)
    h = wall_rows[-1] + 1
    w = wall_cols[-1] + 1
    if h > 30 or w > 30:
        return None

    row_spans = _spans(wall_rows)
    col_spans = _spans(wall_cols)
    total_rooms = room_rows * room_cols
    max_objects = min(MAX_OBJECTS, max(1, total_rooms // 3))
    object_count = random.randint(1, max_objects)

    paths: list[list[tuple[int, int]]] = []
    used_rooms: set[tuple[int, int]] = set()
    for _ in range(object_count):
        path = _sample_path(room_rows, room_cols, used_rooms)
        if path is None:
            return None
        paths.append(path)
        used_rooms.update(path)

    bases = {
        (ri, ci): random.choice(COLOR_POOL) for ri in range(room_rows) for ci in range(room_cols)
    }
    objects: dict[tuple[int, int], dict] = {}

    for path in paths:
        obj_color = random.choice(COLOR_POOL)
        dest = path[-1]
        bases[dest] = obj_color
        for room in path[:-1]:
            choices = [color for color in COLOR_POOL if color != obj_color]
            bases[room] = random.choice(choices)

        source = path[0]
        first_step = path[1]
        side = _step_side(source, first_step)
        room_h = row_spans[source[0]][1] - row_spans[source[0]][0] + 1
        room_w = col_spans[source[1]][1] - col_spans[source[1]][0] + 1
        coords_rel = _place_object(room_h, room_w, side)
        objects[source] = {"color": obj_color, "coords_rel": coords_rel}

    gap_map: dict[frozenset[tuple[int, int]], tuple[int, int]] = {}
    for path in paths:
        source = path[0]
        first_edge = frozenset((source, path[1]))
        gap_map[first_edge] = _source_gap(
            source,
            path[1],
            objects[source]["coords_rel"],
            wall_rows,
            wall_cols,
            row_spans,
            col_spans,
        )
        for a, b in zip(path[1:], path[2:]):
            edge = frozenset((a, b))
            gap_map[edge] = _random_gap_for_edge(a, b, wall_rows, wall_cols, row_spans, col_spans)

    grid = np.full((h, w), BLUE, dtype=int)
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(bases[(ri, ci)])

    for gap in gap_map.values():
        grid[gap] = BLACK

    expected = grid.copy()
    for room, obj in objects.items():
        rr0, cc0 = row_spans[room[0]][0], col_spans[room[1]][0]
        for dr, dc in obj["coords_rel"]:
            ar, ac = rr0 + dr, cc0 + dc
            grid[ar, ac] = int(obj["color"])
            expected[ar, ac] = int(bases[room])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-routing puzzle")

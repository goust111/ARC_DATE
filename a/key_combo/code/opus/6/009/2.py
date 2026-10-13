import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
YELLOW = 4
SNAKE = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def bbox_of(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def region_info(grid: np.ndarray) -> list[dict]:
    non_wall = grid != WALL
    infos = []
    for comp in components(non_wall):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        comp_set = set(comp)
        wall_cells = set()
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == WALL:
                    wall_cells.add((nr, nc))
        infos.append(
            {
                "cells": comp,
                "cell_set": comp_set,
                "wall_set": wall_cells,
                "bbox": bbox_of(comp),
            }
        )
    infos.sort(key=lambda reg: reg["bbox"])
    return infos


def snake_path(grid: np.ndarray, endpoint_mode: str = "min") -> list[tuple[int, int]]:
    pts = [tuple(map(int, xy)) for xy in np.argwhere(grid == SNAKE)]
    assert pts, "snake path missing"
    snake = set(pts)
    deg = {p: sum((nr, nc) in snake for nr, nc in neighbors4(*p)) for p in snake}
    endpoints = [p for p, d in deg.items() if d == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    start = min(endpoints) if endpoint_mode == "min" else max(endpoints)

    out = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        for nr, nc in neighbors4(*cur):
            if (nr, nc) in snake and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break
        assert len(nxts) == 1, "snake must be a simple chain"
        prev, cur = cur, nxts[0]
        out.append(cur)
    assert len(out) == len(pts), "snake traversal must cover whole chain"
    return out


def segments_from_path(path: list[tuple[int, int]]) -> list[dict]:
    assert len(path) >= 2
    segments = []
    i = 0
    while i < len(path) - 1:
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        j = i + 1
        while j < len(path) - 1:
            ndr = path[j + 1][0] - path[j][0]
            ndc = path[j + 1][1] - path[j][1]
            if (ndr, ndc) != (dr, dc):
                break
            j += 1
        segments.append(
            {
                "start_index": i,
                "cells": path[i : j + 1],
                "direction": (dr, dc),
            }
        )
        i = j
    return segments


def projection(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def analyze_regions(grid: np.ndarray, endpoint_mode: str = "min", segment_pick: str = "first") -> tuple[list[dict], list[tuple[int, int]], list[dict]]:
    regions = region_info(grid)
    path = snake_path(grid, endpoint_mode=endpoint_mode)
    segs = segments_from_path(path)
    for reg in regions:
        touching = []
        for seg in segs:
            if any(any((nr, nc) in reg["wall_set"] for nr, nc in neighbors4(r, c)) for r, c in seg["cells"]):
                touching.append(seg)
        assert touching, "every region must touch at least one snake segment"
        chosen = touching[0] if segment_pick == "first" else touching[-1]
        reg["gravity"] = chosen["direction"]
    return regions, path, segs


def solve_variant(grid: np.ndarray, endpoint_mode: str = "min", segment_pick: str = "first", return_meta: bool = False):
    g = np.array(grid, dtype=int, copy=True)
    regions, path, segs = analyze_regions(g, endpoint_mode=endpoint_mode, segment_pick=segment_pick)
    out = g.copy()
    meta_regions = []
    for reg in regions:
        movers = [p for p in reg["cells"] if int(g[p]) == YELLOW]
        if not movers:
            meta_regions.append({"bbox": reg["bbox"], "gravity": reg["gravity"], "input": [], "output": []})
            continue
        for p in movers:
            out[p] = BG
        dr, dc = reg["gravity"]
        stopped = set()
        ordered = sorted(movers, key=lambda p: (projection(p, (dr, dc)), p[0], p[1]), reverse=True)
        for r, c in ordered:
            nr, nc = r, c
            while (nr + dr, nc + dc) in reg["cell_set"] and (nr + dr, nc + dc) not in stopped:
                nr += dr
                nc += dc
            stopped.add((nr, nc))
        for p in stopped:
            out[p] = YELLOW
        meta_regions.append(
            {
                "bbox": reg["bbox"],
                "gravity": reg["gravity"],
                "input": sorted(movers),
                "output": sorted(stopped),
            }
        )
    if return_meta:
        return out, {"regions": meta_regions, "path": path, "segments": segs}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside blue-walled rooms using the traversal direction of the first wall-adjacent snake segment.

    Input structure:
    - Background is black. Blue pixels form several disjoint rectangular closed rooms, each with a black/yellow interior.
    - Grey pixels form a 4-connected one-cell-wide snake in the corridors outside the rooms, with exactly two endpoints.
    - Each room contains one to three yellow pixels, and the snake runs alongside at least one wall of every room.

    Rule:
    - Find the snake endpoints and choose the lexicographically smaller one as the traversal start.
    - Traverse the whole snake, split it into maximal straight segments, and keep each segment's traversal direction.
    - For each enclosed room, find the first snake segment whose cells are orthogonally adjacent to that room's blue wall.
    - Use that segment direction as gravity for the room, then slide all yellow pixels inside the room in that direction.
    - Move pixels from nearest-to-far-wall toward farthest-from-far-wall, stopping each one at the far wall or against an already stopped yellow pixel.
    - Keep walls and snake cells unchanged.

    Key insight:
    - The snake is not a picture to copy: its ordered local travel direction encodes a per-room gravity vector, so endpoint choice, traversal order, and first wall-adjacent segment all matter.
    """
    return solve_variant(grid)


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def place_rooms(rng: random.Random, h: int, w: int, n_rooms: int) -> list[dict] | None:
    rooms = []
    for rid in range(n_rooms):
        placed = False
        for _ in range(300):
            inner_h = rng.randint(3, 5)
            inner_w = rng.randint(3, 5)
            oh = inner_h + 2
            ow = inner_w + 2
            r0 = rng.randint(2, h - oh - 3)
            c0 = rng.randint(2, w - ow - 3)
            box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if any(boxes_overlap(expand_box(box, 1), expand_box(other["box"], 1)) for other in rooms):
                continue
            room = {
                "id": rid,
                "box": box,
                "inner_box": (r0 + 1, c0 + 1, r0 + oh - 2, c0 + ow - 2),
            }
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None
    rooms.sort(key=lambda rm: rm["box"])
    for idx, room in enumerate(rooms):
        room["id"] = idx
    return rooms


def paint_rooms(grid: np.ndarray, rooms: list[dict]) -> None:
    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL


def room_blocked_masks(shape: tuple[int, int], rooms: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    blocked = np.zeros(shape, dtype=bool)
    wall_adjacent = np.zeros(shape, dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
        for r in range(r0, r1 + 1):
            for c in (c0, c1):
                for nr, nc in neighbors4(r, c):
                    if in_bounds(shape, nr, nc) and not blocked[nr, nc]:
                        wall_adjacent[nr, nc] = True
        for c in range(c0, c1 + 1):
            for r in (r0, r1):
                for nr, nc in neighbors4(r, c):
                    if in_bounds(shape, nr, nc) and not blocked[nr, nc]:
                        wall_adjacent[nr, nc] = True
    return blocked, wall_adjacent


def make_anchor_choices(room: dict, used_cells: set[tuple[int, int]]) -> list[dict]:
    r0, c0, r1, c1 = room["box"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    choices = []

    def add(side: str, direction: tuple[int, int], fixed: int, starts: list[int], vertical: bool):
        for length in range(2, min(4, inner_h if vertical else inner_w) + 1):
            limit = len(starts) - length + 1
            for off in range(limit):
                span = starts[off : off + length]
                if vertical:
                    cells = [(rr, fixed) for rr in span]
                else:
                    cells = [(fixed, cc) for cc in span]
                if direction in [(-1, 0), (0, -1)]:
                    cells = list(reversed(cells))
                cell_set = set(cells)
                if cell_set & used_cells:
                    continue
                choices.append(
                    {
                        "room_id": room["id"],
                        "side": side,
                        "direction": direction,
                        "cells": cells,
                        "cell_set": cell_set,
                    }
                )

    add("top", (0, 1), r0 - 1, list(range(c0 + 1, c1)), False)
    add("top", (0, -1), r0 - 1, list(range(c0 + 1, c1)), False)
    add("bottom", (0, 1), r1 + 1, list(range(c0 + 1, c1)), False)
    add("bottom", (0, -1), r1 + 1, list(range(c0 + 1, c1)), False)
    add("left", (1, 0), c0 - 1, list(range(r0 + 1, r1)), True)
    add("left", (-1, 0), c0 - 1, list(range(r0 + 1, r1)), True)
    add("right", (1, 0), c1 + 1, list(range(r0 + 1, r1)), True)
    add("right", (-1, 0), c1 + 1, list(range(r0 + 1, r1)), True)
    return choices


def choose_primary_anchors(rng: random.Random, rooms: list[dict]) -> list[dict] | None:
    for _ in range(120):
        used = set()
        anchors = []
        for room in rooms:
            choices = make_anchor_choices(room, used)
            if not choices:
                break
            anchor = rng.choice(choices)
            anchors.append(anchor)
            used |= anchor["cell_set"]
        if len(anchors) != len(rooms):
            continue
        dirs = {a["direction"] for a in anchors}
        if len(dirs) < 2:
            continue
        if not any(d[0] != 0 for d in dirs) or not any(d[1] != 0 for d in dirs):
            continue
        return anchors
    return None


def choose_decoy_anchor(rng: random.Random, rooms: list[dict], primaries: list[dict]) -> dict | None:
    primary_by_room = {a["room_id"]: a for a in primaries}
    used = set()
    for a in primaries:
        used |= a["cell_set"]
    room_order = list(range(len(rooms)))
    rng.shuffle(room_order)
    for rid in room_order:
        choices = []
        for cand in make_anchor_choices(rooms[rid], used):
            if cand["direction"] != primary_by_room[rid]["direction"]:
                choices.append(cand)
        if choices:
            return rng.choice(choices)
    return None


def bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    safe: np.ndarray,
    blocked_used: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nr, nc in neighbors4(*cur):
            nxt = (nr, nc)
            if not in_bounds(shape, nr, nc):
                continue
            if nxt in prev:
                continue
            if nxt != goal and not safe[nr, nc]:
                continue
            if nxt in forbidden:
                continue
            if nxt in blocked_used:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def path_has_valid_degrees(path: list[tuple[int, int]]) -> bool:
    pset = set(path)
    if len(pset) != len(path):
        return False
    degrees = {p: sum((nr, nc) in pset for nr, nc in neighbors4(*p)) for p in pset}
    degs = sorted(degrees.values())
    return degs.count(1) == 2 and all(d in (1, 2) for d in degs)


def build_snake(rng: random.Random, shape: tuple[int, int], rooms: list[dict], anchor_sequence: list[dict]) -> list[tuple[int, int]] | None:
    blocked, wall_adjacent = room_blocked_masks(shape, rooms)
    safe = (~blocked) & (~wall_adjacent)
    all_anchor_cells = set()
    for anchor in anchor_sequence:
        all_anchor_cells |= anchor["cell_set"]

    safe_cells = [tuple(map(int, xy)) for xy in np.argwhere(safe)]
    safe_cells.sort()
    if len(safe_cells) < 2:
        return None

    start_choices = safe_cells[: min(12, len(safe_cells))]

    for start in start_choices:
        path = [start]
        used = {start}
        cur = start
        ok = True
        for idx, anchor in enumerate(anchor_sequence):
            future = set()
            for later in anchor_sequence[idx + 1 :]:
                future |= later["cell_set"]
            route = bfs_path(shape, cur, anchor["cells"][0], safe, used, future | (all_anchor_cells - anchor["cell_set"]))
            if route is None:
                ok = False
                break
            for cell in route[1:]:
                path.append(cell)
                used.add(cell)
            for cell in anchor["cells"][1:]:
                if cell in used:
                    ok = False
                    break
                path.append(cell)
                used.add(cell)
            if not ok:
                break
            cur = anchor["cells"][-1]
        if not ok:
            continue

        end_choices = [cell for cell in reversed(safe_cells) if cell not in used and cell > start]
        for end in end_choices[:80]:
            route = bfs_path(shape, cur, end, safe, used, all_anchor_cells)
            if route is None:
                continue
            for cell in route[1:]:
                path.append(cell)
                used.add(cell)
            if path_has_valid_degrees(path):
                return path
            del path[len(path) - (len(route) - 1) :]
            used.difference_update(route[1:])
    return None


def choose_yellows_for_room(
    rng: random.Random,
    room: dict,
    gravity: tuple[int, int],
    want_stack: bool,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["inner_box"]
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if want_stack:
        count = rng.randint(2, 3)
        if gravity[0] != 0:
            c = rng.randint(c0, c1)
            rows = list(range(r0, r1 + 1))
            rng.shuffle(rows)
            return sorted((rows[i], c) for i in range(count))
        r = rng.randint(r0, r1)
        cols = list(range(c0, c1 + 1))
        rng.shuffle(cols)
        return sorted((r, cols[i]) for i in range(count))
    count = rng.randint(1, 3)
    return sorted(rng.sample(cells, count))


def paint_yellows(rng: random.Random, grid: np.ndarray, rooms: list[dict], primary_by_room: dict[int, dict]) -> None:
    stack_room = rng.randrange(len(rooms))
    for room in rooms:
        gravity = primary_by_room[room["id"]]["direction"]
        picks = choose_yellows_for_room(rng, room, gravity, want_stack=room["id"] == stack_room)
        for r, c in picks:
            grid[r, c] = YELLOW


def validate_instance(grid: np.ndarray) -> bool:
    base, meta = solve_variant(grid, return_meta=True)
    if np.array_equal(base, grid):
        return False

    if len(meta["regions"]) != 2:
        return False
    dirs = {reg["gravity"] for reg in meta["regions"]}
    if len(dirs) != 2:
        return False
    if not any(d[0] != 0 for d in dirs) or not any(d[1] != 0 for d in dirs):
        return False

    reverse = solve_variant(grid, endpoint_mode="max")
    if np.array_equal(reverse, base):
        return False

    last_pick = solve_variant(grid, segment_pick="last")
    if np.array_equal(last_pick, base):
        return False

    if not any(
        len(reg["input"]) >= 2
        and (
            len({c for _, c in reg["input"]}) == 1
            or len({r for r, _ in reg["input"]}) == 1
        )
        and reg["input"] != reg["output"]
        for reg in meta["regions"]
    ):
        return False

    pts = [tuple(map(int, xy)) for xy in np.argwhere(grid == SNAKE)]
    pset = set(pts)
    if len([p for p in pset if sum((nr, nc) in pset for nr, nc in neighbors4(*p)) == 1]) != 2:
        return False
    return True


def append_line(path: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r, c = path[-1]
    er, ec = end
    if r != er and c != ec:
        raise ValueError("append_line requires axis-aligned move")
    dr = 0 if r == er else (1 if er > r else -1)
    dc = 0 if c == ec else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def transform_grid(grid: np.ndarray, variant: str) -> np.ndarray:
    if variant == "id":
        return grid
    if variant == "rot90":
        return np.rot90(grid, 1)
    if variant == "rot180":
        return np.rot90(grid, 2)
    if variant == "rot270":
        return np.rot90(grid, 3)
    if variant == "flipud":
        return np.flipud(grid)
    if variant == "fliplr":
        return np.fliplr(grid)
    if variant == "transpose":
        return grid.T.copy()
    if variant == "anti":
        return np.fliplr(np.rot90(grid, 1))
    raise ValueError(variant)


def canonical_rooms(rng: random.Random, h: int, w: int) -> tuple[dict, dict] | None:
    for _ in range(200):
        a_inner_h = rng.randint(3, 5)
        a_inner_w = rng.randint(3, 5)
        b_inner_h = rng.randint(3, 5)
        b_inner_w = rng.randint(3, 5)
        a_oh, a_ow = a_inner_h + 2, a_inner_w + 2
        b_oh, b_ow = b_inner_h + 2, b_inner_w + 2

        a_r0 = rng.randint(3, 6)
        a_c0 = rng.randint(3, 6)
        a_r1 = a_r0 + a_oh - 1
        a_c1 = a_c0 + a_ow - 1

        b_r0_min = a_r1 + 4
        b_r0_max = h - b_oh - 3
        b_c0_min = max(a_c1 + 4, w // 2 - 1)
        b_c0_max = w - b_ow - 3
        if b_r0_min > b_r0_max or b_c0_min > b_c0_max:
            continue
        b_r0 = rng.randint(b_r0_min, b_r0_max)
        b_c0 = rng.randint(b_c0_min, b_c0_max)
        b_r1 = b_r0 + b_oh - 1
        b_c1 = b_c0 + b_ow - 1

        if a_c0 - 1 <= 0:
            continue
        room_a = {"id": 0, "box": (a_r0, a_c0, a_r1, a_c1), "inner_box": (a_r0 + 1, a_c0 + 1, a_r1 - 1, a_c1 - 1)}
        room_b = {"id": 1, "box": (b_r0, b_c0, b_r1, b_c1), "inner_box": (b_r0 + 1, b_c0 + 1, b_r1 - 1, b_c1 - 1)}
        return room_a, room_b
    return None


def canonical_snake(room_a: dict, room_b: dict) -> list[tuple[int, int]]:
    a_r0, a_c0, _a_r1, _a_c1 = room_a["box"]
    _b_r0, _b_c0, b_r1, b_c1 = room_b["box"]
    path = [(0, 0)]
    append_line(path, (a_r0 - 1, 0))
    append_line(path, (a_r0 - 1, b_c1 + 1))
    append_line(path, (b_r1 + 1, b_c1 + 1))
    append_line(path, (b_r1 + 1, a_c0 - 1))
    append_line(path, (a_r0 + 1, a_c0 - 1))
    return path


def paint_canonical_yellows(rng: random.Random, grid: np.ndarray, room_a: dict, room_b: dict) -> None:
    stack_room = rng.choice([0, 1])

    def room_cells(room: dict) -> list[tuple[int, int]]:
        r0, c0, r1, c1 = room["inner_box"]
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    if stack_room == 0:
        r0, c0, r1, c1 = room_a["inner_box"]
        row = rng.randint(r0, r1)
        cols = list(range(c0, c1 + 1))
        rng.shuffle(cols)
        for c in cols[: rng.randint(2, 3)]:
            grid[row, c] = YELLOW
    else:
        picks = rng.sample(room_cells(room_a), rng.randint(1, 3))
        for r, c in picks:
            grid[r, c] = YELLOW

    if stack_room == 1:
        r0, c0, r1, c1 = room_b["inner_box"]
        col = rng.randint(c0, c1)
        rows = list(range(r0, r1 + 1))
        rng.shuffle(rows)
        for r in rows[: rng.randint(2, 3)]:
            grid[r, col] = YELLOW
    else:
        picks = rng.sample(room_cells(room_b), rng.randint(1, 3))
        for r, c in picks:
            grid[r, c] = YELLOW


def build_instance(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    rooms = canonical_rooms(rng, h, w)
    if rooms is None:
        return None
    room_a, room_b = rooms
    snake = canonical_snake(room_a, room_b)
    if not path_has_valid_degrees(snake):
        return None

    grid = np.zeros((h, w), dtype=int)
    paint_rooms(grid, [room_a, room_b])
    for r, c in snake:
        grid[r, c] = SNAKE
    paint_canonical_yellows(rng, grid, room_a, room_b)

    grid = transform_grid(
        grid,
        rng.choice(["id", "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "anti"]),
    )
    if not validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        rng = random.Random()
        grid = build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-gravity ARC instance")

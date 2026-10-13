from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREEN = 3
GRAY = 5
MAROON = 9
TOWER_COLORS = (1, 2, 4, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
ROOM_COUNT_RANGE = (1, 3)
ROOM_INNER_H_RANGE = (2, 5)
ROOM_INNER_W_RANGE = (2, 6)
ROOM_MARGIN = 3
ROOM_GAP = 2
TOWER_COUNT_RANGE = (1, 3)
SCENE_RETRIES = 400
ROOM_RETRIES = 120
ROUTE_RETRIES = 240


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("segment endpoints must share a row or a column")


def _segment_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    if ar == br:
        assert ac != bc
        return 0, 1 if bc > ac else -1
    if ac == bc:
        assert ar != br
        return 1 if br > ar else -1, 0
    raise AssertionError("segment endpoints must share a row or a column")


def _segment_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _mirror_kind(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    turn_map = {
        ((0, 1), (1, 0)): "\\",
        ((1, 0), (0, 1)): "\\",
        ((0, -1), (-1, 0)): "\\",
        ((-1, 0), (0, -1)): "\\",
        ((0, 1), (-1, 0)): "/",
        ((-1, 0), (0, 1)): "/",
        ((0, -1), (1, 0)): "/",
        ((1, 0), (0, -1)): "/",
    }
    key = (in_dir, out_dir)
    assert key in turn_map, "expected a right-angle turn"
    return turn_map[key]


def _reflect(direction: tuple[int, int], mirror: str) -> tuple[int, int]:
    dr, dc = direction
    if mirror == "\\":
        return dc, dr
    assert mirror == "/"
    return -dc, -dr


def _touching_boxes(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _find_rooms(
    g: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int, int, int], int], dict[tuple[int, int], int]]:
    rooms: list[dict[str, object]] = []
    entrance_map: dict[tuple[int, int, int, int], int] = {}
    interior_lookup: dict[tuple[int, int], int] = {}

    for comp in _components(g == GREEN, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "green room frame must enclose an interior"

        expected = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))

        comp_set = set(comp)
        assert comp_set == expected, "green pixels must form a one-pixel-thick rectangle frame"

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        room_index = len(rooms)
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": interior,
            }
        )

        for cell in interior:
            assert cell not in interior_lookup, "room interiors must not overlap"
            interior_lookup[cell] = room_index

        for r in range(r0 + 1, r1):
            key = (r, c0, 0, 1)
            assert key not in entrance_map
            entrance_map[key] = room_index
            key = (r, c1, 0, -1)
            assert key not in entrance_map
            entrance_map[key] = room_index
        for c in range(c0 + 1, c1):
            key = (r0, c, 1, 0)
            assert key not in entrance_map
            entrance_map[key] = room_index
            key = (r1, c, -1, 0)
            assert key not in entrance_map
            entrance_map[key] = room_index

    return rooms, entrance_map, interior_lookup


def _find_mirror_kinds(g: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = g.shape
    mirrors: dict[tuple[int, int], str] = {}
    for r, c in map(tuple, np.argwhere(g == GRAY)):
        r = int(r)
        c = int(c)
        backslash = False
        slash = False
        for dr, dc in [(-1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GRAY:
                backslash = True
        for dr, dc in [(-1, 1), (1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GRAY:
                slash = True
        assert backslash ^ slash, "each gray mirror pixel must belong to exactly one diagonal orientation"
        mirrors[(r, c)] = "\\" if backslash else "/"
    return mirrors


def _find_towers(g: np.ndarray) -> list[dict[str, object]]:
    special = {BG, GREEN, GRAY, MAROON}
    towers: list[dict[str, object]] = []
    for r, c in map(tuple, np.argwhere(~np.isin(g, list(special)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        maroon_neighbors = []
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == MAROON:
                maroon_neighbors.append((nr, nc))
        assert len(maroon_neighbors) == 1, "each signal tower must have exactly one maroon direction marker"
        mr, mc = maroon_neighbors[0]
        towers.append(
            {
                "pos": (r, c),
                "dir": (mr - r, mc - c),
                "color": color,
            }
        )
    towers.sort(key=lambda item: item["pos"])
    return towers


def _fill_room(out: np.ndarray, room: dict[str, object], color: int) -> None:
    for r, c in room["interior"]:
        if int(out[r, c]) == BG:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    rooms, entrance_map, interior_lookup = _find_rooms(g)
    mirror_kinds = _find_mirror_kinds(g)
    towers = _find_towers(g)

    out = g.copy()
    out[out == MAROON] = BG

    for tower in towers:
        r, c = tower["pos"]
        dr, dc = tower["dir"]
        color = int(tower["color"])

        seen_states = {(r, c, dr, dc)}
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if int(g[nr, nc]) == GREEN:
                room_index = entrance_map.get((nr, nc, dr, dc))
                if room_index is not None:
                    _fill_room(out, rooms[room_index], color)
                break

            r, c = nr, nc
            cell = int(g[r, c])

            if cell == BG or cell == MAROON:
                out[r, c] = color

            room_index = interior_lookup.get((r, c))
            if room_index is not None:
                _fill_room(out, rooms[room_index], color)

            if cell == GRAY:
                dr, dc = _reflect((dr, dc), mirror_kinds[(r, c)])

            state = (r, c, dr, dc)
            if state in seen_states:
                break
            seen_states.add(state)

    return out


def _room_bbox_cells(room: dict[str, int]) -> set[tuple[int, int]]:
    cells = set()
    for r in range(room["r0"], room["r1"] + 1):
        for c in range(room["c0"], room["c1"] + 1):
            cells.add((r, c))
    return cells


def _place_rooms(h: int, w: int) -> list[dict[str, int]] | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    rooms: list[dict[str, int]] = []

    for _ in range(ROOM_RETRIES):
        if len(rooms) == room_count:
            return rooms

        inner_h = random.randint(*ROOM_INNER_H_RANGE)
        inner_w = random.randint(*ROOM_INNER_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        max_r0 = h - outer_h - ROOM_MARGIN
        max_c0 = w - outer_w - ROOM_MARGIN
        if max_r0 < ROOM_MARGIN or max_c0 < ROOM_MARGIN:
            return None

        r0 = random.randint(ROOM_MARGIN, max_r0)
        c0 = random.randint(ROOM_MARGIN, max_c0)
        room = {
            "r0": r0,
            "c0": c0,
            "r1": r0 + outer_h - 1,
            "c1": c0 + outer_w - 1,
        }

        if any(
            _touching_boxes(
                (room["r0"], room["c0"], room["r1"], room["c1"]),
                (other["r0"], other["c0"], other["r1"], other["c1"]),
                ROOM_GAP,
            )
            for other in rooms
        ):
            continue
        rooms.append(room)

    return None


def _pick_room_target(
    room: dict[str, int],
    used_sides: set[tuple[int, str]],
    room_index: int,
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    choices = ["left", "right", "top", "bottom"]
    random.shuffle(choices)
    r0, c0, r1, c1 = room["r0"], room["c0"], room["r1"], room["c1"]

    for side in choices:
        if (room_index, side) in used_sides:
            continue
        if side == "left":
            row = random.randint(r0 + 1, r1 - 1)
            return (row, c0 - 1), (0, 1), side
        if side == "right":
            row = random.randint(r0 + 1, r1 - 1)
            return (row, c1 + 1), (0, -1), side
        if side == "top":
            col = random.randint(c0 + 1, c1 - 1)
            return (r0 - 1, col), (1, 0), side
        col = random.randint(c0 + 1, c1 - 1)
        return (r1 + 1, col), (-1, 0), side
    return None


def _pick_boundary_target(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    side = random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        return (random.randint(1, h - 2), 0), (0, -1)
    if side == "right":
        return (random.randint(1, h - 2), w - 1), (0, 1)
    if side == "top":
        return (0, random.randint(1, w - 2)), (-1, 0)
    return (h - 1, random.randint(1, w - 2)), (1, 0)


def _trace_route(
    points: list[tuple[int, int]],
    final_dir: tuple[int, int],
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    assert len(points) >= 2
    assert _segment_dir(points[-2], points[-1]) == final_dir

    dirs: list[tuple[int, int]] = []
    all_cells: set[tuple[int, int]] = set()
    ordered_path: list[tuple[int, int]] = []

    for i in range(len(points) - 1):
        if _segment_len(points[i], points[i + 1]) < 2:
            return None
        seg_dir = _segment_dir(points[i], points[i + 1])
        dirs.append(seg_dir)
        seg = _segment_cells(points[i], points[i + 1])
        if i > 0:
            seg = seg[1:]
        for cell in seg:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                return None
            if cell in blocked:
                return None
            if cell in all_cells:
                return None
            all_cells.add(cell)
            ordered_path.append(cell)

    bends = points[1:-1]
    for i, bend in enumerate(bends):
        if dirs[i] == dirs[i + 1] or dirs[i] == (-dirs[i + 1][0], -dirs[i + 1][1]):
            return None
        assert bend in all_cells

    return all_cells, bends, dirs


def _all_gray_valid(gray_cells: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in gray_cells:
        backslash = False
        slash = False
        for dr, dc in [(-1, -1), (1, 1)]:
            if (r + dr, c + dc) in gray_cells:
                backslash = True
        for dr, dc in [(-1, 1), (1, -1)]:
            if (r + dr, c + dc) in gray_cells:
                slash = True
        if backslash == slash:
            return False
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in gray_cells:
                return False
    return True


def _choose_mirror_extras(
    bends: list[tuple[int, int]],
    dirs: list[tuple[int, int]],
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    route_cells: set[tuple[int, int]],
    existing_gray: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    bend_cells = set(bends)
    extras: set[tuple[int, int]] = set()

    for i, bend in enumerate(bends):
        mirror = _mirror_kind(dirs[i], dirs[i + 1])
        if mirror == "\\":
            candidates = [(bend[0] - 1, bend[1] - 1), (bend[0] + 1, bend[1] + 1)]
        else:
            candidates = [(bend[0] - 1, bend[1] + 1), (bend[0] + 1, bend[1] - 1)]

        valid = []
        for cell in candidates:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                continue
            if cell in blocked or cell in route_cells or cell in extras:
                continue
            valid.append(cell)
        if not valid:
            return None

        random.shuffle(valid)
        chosen = {valid[0]}
        if len(valid) == 2 and random.random() < 0.45:
            chosen.add(valid[1])
        extras |= chosen

    all_gray = existing_gray | bend_cells | extras
    if not _all_gray_valid(all_gray, h, w):
        return None
    return bend_cells, extras


def _sample_route_points(
    h: int,
    w: int,
    end: tuple[int, int],
    final_dir: tuple[int, int],
) -> list[tuple[int, int]] | None:
    er, ec = end
    horizontal = final_dir[0] == 0
    if horizontal:
        pattern = random.choices(["H0", "VH1", "HVH2"], weights=[1, 2, 4])[0]
        if pattern == "H0":
            if final_dir == (0, 1):
                candidates = list(range(0, ec - 1))
            else:
                candidates = list(range(ec + 2, w))
            if not candidates:
                return None
            sc = random.choice(candidates)
            return [(er, sc), end]

        if pattern == "VH1":
            if final_dir == (0, 1):
                bend_cols = list(range(1, ec - 1))
            else:
                bend_cols = list(range(ec + 2, w - 1))
            if not bend_cols:
                return None
            bc = random.choice(bend_cols)
            bend = (er, bc)
            start_rows = [r for r in range(h) if abs(r - er) >= 2]
            if not start_rows:
                return None
            sr = random.choice(start_rows)
            return [(sr, bc), bend, end]

        if final_dir == (0, 1):
            mid_cols = list(range(2, ec - 1))
        else:
            mid_cols = list(range(ec + 2, w - 2))
        if not mid_cols:
            return None
        mid_c = random.choice(mid_cols)
        start_rows = [r for r in range(h) if abs(r - er) >= 2]
        if not start_rows:
            return None
        sr = random.choice(start_rows)
        start_cols = [c for c in range(w) if abs(c - mid_c) >= 2]
        if not start_cols:
            return None
        sc = random.choice(start_cols)
        return [(sr, sc), (sr, mid_c), (er, mid_c), end]

    pattern = random.choices(["V0", "HV1", "VHV2"], weights=[1, 2, 4])[0]
    if pattern == "V0":
        if final_dir == (1, 0):
            candidates = list(range(0, er - 1))
        else:
            candidates = list(range(er + 2, h))
        if not candidates:
            return None
        sr = random.choice(candidates)
        return [(sr, ec), end]

    if pattern == "HV1":
        if final_dir == (1, 0):
            bend_rows = list(range(1, er - 1))
        else:
            bend_rows = list(range(er + 2, h - 1))
        if not bend_rows:
            return None
        br = random.choice(bend_rows)
        bend = (br, ec)
        start_cols = [c for c in range(w) if abs(c - ec) >= 2]
        if not start_cols:
            return None
        sc = random.choice(start_cols)
        return [(br, sc), bend, end]

    if final_dir == (1, 0):
        mid_rows = list(range(2, er - 1))
    else:
        mid_rows = list(range(er + 2, h - 2))
    if not mid_rows:
        return None
    mid_r = random.choice(mid_rows)
    start_cols = [c for c in range(w) if abs(c - ec) >= 2]
    if not start_cols:
        return None
    sc = random.choice(start_cols)
    start_rows = [r for r in range(h) if abs(r - mid_r) >= 2]
    if not start_rows:
        return None
    sr = random.choice(start_rows)
    return [(sr, sc), (mid_r, sc), (mid_r, ec), end]


def _plan_tower_route(
    h: int,
    w: int,
    end: tuple[int, int],
    final_dir: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing_gray: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(ROUTE_RETRIES):
        points = _sample_route_points(h, w, end, final_dir)
        if points is None:
            continue

        traced = _trace_route(points, final_dir, h, w, blocked)
        if traced is None:
            continue
        route_cells, bends, dirs = traced

        first_dir = dirs[0]
        start = points[0]
        marker = (start[0] + first_dir[0], start[1] + first_dir[1])
        if marker not in route_cells:
            continue

        extras_info = _choose_mirror_extras(
            bends,
            dirs,
            h,
            w,
            blocked,
            route_cells,
            existing_gray,
        )
        if extras_info is None:
            continue
        bend_cells, extra_gray = extras_info

        return {
            "start": start,
            "marker": marker,
            "first_dir": first_dir,
            "route_cells": route_cells,
            "bend_cells": bend_cells,
            "extra_gray": extra_gray,
            "points": points,
        }
    return None


def _build_scene() -> tuple[np.ndarray, dict[str, object]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rooms = _place_rooms(h, w)
    if rooms is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    for room in rooms:
        for r in range(room["r0"], room["r1"] + 1):
            grid[r, room["c0"]] = GREEN
            grid[r, room["c1"]] = GREEN
        for c in range(room["c0"], room["c1"] + 1):
            grid[room["r0"], c] = GREEN
            grid[room["r1"], c] = GREEN
        blocked |= _room_bbox_cells(room)

    max_towers = min(TOWER_COUNT_RANGE[1], len(TOWER_COLORS), len(rooms) + 1)
    num_towers = random.randint(TOWER_COUNT_RANGE[0], max_towers)
    num_fill = random.randint(1, min(num_towers, len(rooms)))
    room_indices = random.sample(range(len(rooms)), num_fill)

    targets: list[dict[str, object]] = []
    used_room_sides: set[tuple[int, str]] = set()
    for room_index in room_indices:
        picked = _pick_room_target(rooms[room_index], used_room_sides, room_index)
        if picked is None:
            return None
        end, final_dir, side = picked
        used_room_sides.add((room_index, side))
        targets.append(
            {
                "kind": "room",
                "room_index": room_index,
                "end": end,
                "final_dir": final_dir,
            }
        )

    for _ in range(num_towers - num_fill):
        end, final_dir = _pick_boundary_target(h, w)
        targets.append(
            {
                "kind": "boundary",
                "end": end,
                "final_dir": final_dir,
            }
        )
    random.shuffle(targets)

    colors = random.sample(TOWER_COLORS, num_towers)
    gray_cells: set[tuple[int, int]] = set()
    plans: list[dict[str, object]] = []

    for target, color in zip(targets, colors, strict=False):
        plan = _plan_tower_route(h, w, target["end"], target["final_dir"], blocked, gray_cells)
        if plan is None:
            return None

        start = plan["start"]
        marker = plan["marker"]
        if int(grid[start]) != BG or int(grid[marker]) != BG:
            return None

        grid[start] = color
        grid[marker] = MAROON
        for r, c in plan["bend_cells"] | plan["extra_gray"]:
            grid[r, c] = GRAY

        blocked |= set(plan["route_cells"])
        blocked |= set(plan["extra_gray"])
        gray_cells |= set(plan["bend_cells"]) | set(plan["extra_gray"])

        plans.append(
            {
                **plan,
                **target,
                "color": color,
            }
        )

    if not gray_cells:
        return None

    meta = {
        "rooms": rooms,
        "plans": plans,
    }
    return grid, meta


def _valid_scene(inp: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    plans = meta["plans"]
    rooms = meta["rooms"]
    filled_rooms = 0
    for plan in plans:
        if plan["kind"] != "room":
            continue
        room = rooms[plan["room_index"]]
        interior = out[room["r0"] + 1 : room["r1"], room["c0"] + 1 : room["c1"]]
        if interior.size == 0 or not np.all(interior == int(plan["color"])):
            return False
        filled_rooms += 1

    if filled_rooms == 0:
        return False
    if np.any(out == MAROON):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(SCENE_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        inp, meta = built
        if _valid_scene(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid mirror-room signal puzzle")

from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
WALL = 9
MARKER_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)
SIDES = ("top", "bottom", "left", "right")
OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _infer_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.flat))
    bg = int(max(counts, key=lambda color: (counts[color], -color)))
    others = [color for color in counts if color != bg]
    assert others, "expected at least one non-background color"
    wall = int(max(others, key=lambda color: (counts[color], -color)))
    return bg, wall


def _door_side(r: int, c: int, bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _path_cells_in_room(
    bbox: tuple[int, int, int, int], door_pos: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    side = _door_side(door_pos[0], door_pos[1], bbox)
    assert side is not None
    odr, odc = OUTWARD[side]
    dr = -odr
    dc = -odc
    cells: list[tuple[int, int]] = []
    for step in range(length):
        rr = int(door_pos[0] + dr * step)
        cc = int(door_pos[1] + dc * step)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            break
        if step > 0 and (rr == r0 or rr == r1 or cc == c0 or cc == c1):
            break
        cells.append((rr, cc))
    return cells


def _validate_room_candidate(
    grid: np.ndarray,
    comp_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        return None
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected_bg: set[tuple[int, int]] = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                return None
            expected_bg.add((r, c))

    doors: list[dict[str, object]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if not on_border:
                continue
            is_corner = r in (r0, r1) and c in (c0, c1)
            val = int(grid[r, c])
            if is_corner:
                if val != wall:
                    return None
                continue
            if val == wall:
                continue
            if val != bg:
                return None
            side = _door_side(r, c, bbox)
            if side is None:
                return None
            odr, odc = OUTWARD[side]
            irr = r - odr
            icc = c - odc
            orr = r + odr
            occ = c + odc
            if not (0 <= orr < h and 0 <= occ < w):
                return None
            if int(grid[irr, icc]) != bg:
                return None
            if int(grid[orr, occ]) == bg:
                return None
            doors.append(
                {
                    "pos": (r, c),
                    "side": side,
                    "outward": (odr, odc),
                    "inward": (-odr, -odc),
                    "outside": (orr, occ),
                }
            )
            expected_bg.add((r, c))

    if not 1 <= len(doors) <= 3:
        return None
    door_positions = [door["pos"] for door in doors]
    for i, (r1d, c1d) in enumerate(door_positions):
        for r2d, c2d in door_positions[i + 1 :]:
            if abs(int(r1d) - int(r2d)) + abs(int(c1d) - int(c2d)) == 1:
                return None
    if expected_bg != comp_set:
        return None

    return {"bbox": bbox, "doors": doors}


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict[str, object]]:
    h, w = grid.shape
    rooms: list[dict[str, object]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for comp in _components(grid == bg):
        if _touches_border(comp, h, w):
            continue
        comp_set = {(int(r), int(c)) for r, c in comp}
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        min_r = min(rows)
        max_r = max(rows)
        min_c = min(cols)
        max_c = max(cols)
        candidates: list[dict[str, object]] = []
        for r0 in {min_r - 1, min_r}:
            for r1 in {max_r, max_r + 1}:
                for c0 in {min_c - 1, min_c}:
                    for c1 in {max_c, max_c + 1}:
                        cand = _validate_room_candidate(grid, comp_set, (r0, c0, r1, c1), bg, wall)
                        if cand is not None:
                            candidates.append(cand)
        unique: dict[tuple[int, int, int, int], dict[str, object]] = {}
        for cand in candidates:
            unique[cand["bbox"]] = cand
        if len(unique) != 1:
            continue
        room = next(iter(unique.values()))
        bbox = room["bbox"]
        if bbox in seen_bboxes:
            continue
        seen_bboxes.add(bbox)
        rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _trace_markers(
    grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int], bg: int, wall: int
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    dr, dc = direction
    if dr != 0:
        perpendicular = ((0, -1), (0, 1))
    else:
        perpendicular = ((-1, 0), (1, 0))

    markers: list[tuple[int, int, int]] = []
    r, c = start
    while 0 <= r < h and 0 <= c < w:
        val = int(grid[r, c])
        if val == bg:
            break
        if val != wall:
            markers.append((r, c, val))
        has_branch = False
        for pdr, pdc in perpendicular:
            nr = r + pdr
            nc = c + pdc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                has_branch = True
                break
        if has_branch:
            break
        r += dr
        c += dc
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull corridor markers through each door and paint matching straight paths back into the rooms.

    Input structure:
    - bg = black. One non-background structural color forms thin rectangular room frames and 1-pixel corridors.
    - Each room is a rectangle with 1-3 single-cell door gaps on its border; the interior is otherwise all bg.
    - Straight corridor segments outside the doors contain single colored marker pixels, with one marker color per segment.

    Rule:
    - Detect every rectangular room and count its door gaps; call that count D for the room.
    - For each door, follow the corridor straight outward from the door and collect the nearest D markers on that segment
      (or all available markers if fewer than D exist); use their shared marker color as the fill color and erase the
      collected markers from the corridor.
    - Starting at the door cell itself, extend a straight line of that color length D back into the room, stopping early
      if the next step would hit the opposite wall.

    Key insight:
    - The room's door count is the single hidden control value: it determines both how many corridor markers are absorbed
      and how far the corresponding colored path grows inside the room.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall = _infer_colors(g)
    rooms = _find_rooms(g, bg, wall)

    work = g.copy()
    out = g.copy()
    for room in rooms:
        bbox = room["bbox"]
        doors = room["doors"]
        depth = int(len(doors))
        for door in doors:
            markers = _trace_markers(work, door["outside"], door["outward"], bg, wall)
            if not markers:
                continue
            picked = markers[: min(depth, len(markers))]
            color = int(picked[0][2])
            for rr, cc, _ in picked:
                work[rr, cc] = bg
                out[rr, cc] = bg
            for rr, cc in _path_cells_in_room(bbox, door["pos"], depth):
                out[rr, cc] = color
    return out


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _room_walls(
    bbox: tuple[int, int, int, int], door_positions: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    walls: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        walls.add((r0, c))
        walls.add((r1, c))
    for r in range(r0 + 1, r1):
        walls.add((r, c0))
        walls.add((r, c1))
    walls.difference_update(door_positions)
    return walls


def _sample_marker_count(rng: np.random.Generator, depth: int, force_short: bool | None) -> int:
    if depth == 1:
        return int(rng.integers(1, 4))
    if force_short is None:
        short = bool(rng.integers(0, 2))
    else:
        short = bool(force_short)
    if short:
        return int(rng.integers(1, depth))
    return int(rng.integers(depth, depth + 3))


def _sample_room_spec(
    rng: np.random.Generator,
    h: int,
    w: int,
    forced_depth: int | None = None,
) -> dict[str, object] | None:
    depth = int(forced_depth if forced_depth is not None else rng.integers(1, 4))
    sides = list(rng.choice(np.array(SIDES, dtype=object), size=depth, replace=False))

    stop_branch = depth == 3 and bool(rng.integers(0, 2))
    if stop_branch:
        stop_side = str(sides[int(rng.integers(0, len(sides)))])
        if stop_side in ("top", "bottom"):
            room_h = 3
            room_w = int(rng.integers(4, 9))
        else:
            room_h = int(rng.integers(4, 9))
            room_w = 3
    else:
        room_h = int(rng.integers(4, 9))
        room_w = int(rng.integers(4, 9))

    if room_h > h - 2 or room_w > w - 2:
        return None

    doors: list[dict[str, object]] = []
    path_sets: list[set[tuple[int, int]]] = []
    saw_short_marker = False
    saw_extra_marker = False
    saw_short_path = False
    forced_short_door = depth > 1 and bool(rng.integers(0, 2))

    for side_index, side in enumerate(sides):
        if side in ("top", "bottom"):
            options = list(range(1, room_w - 1))
            offset = int(rng.choice(np.array(options)))
            if side == "top":
                door_local = (0, offset)
            else:
                door_local = (room_h - 1, offset)
        else:
            options = list(range(1, room_h - 1))
            offset = int(rng.choice(np.array(options)))
            if side == "left":
                door_local = (offset, 0)
            else:
                door_local = (offset, room_w - 1)

        path_local = _path_cells_in_room((0, 0, room_h - 1, room_w - 1), door_local, depth)
        if len(path_local) < depth:
            saw_short_path = True
        path_set = set(path_local)
        if any(path_set & other for other in path_sets):
            return None
        path_sets.append(path_set)

        marker_count = _sample_marker_count(
            rng,
            depth,
            force_short=True if forced_short_door and side_index == 0 else None,
        )
        if marker_count < depth:
            saw_short_marker = True
        if marker_count > depth:
            saw_extra_marker = True
        corridor_len = int(rng.integers(max(2, marker_count), max(2, marker_count) + 4))
        marker_steps = sorted(
            int(v) for v in rng.choice(np.arange(1, corridor_len + 1), size=marker_count, replace=False)
        )
        doors.append(
            {
                "side": str(side),
                "door_local": door_local,
                "path_local": path_local,
                "corridor_len": corridor_len,
                "marker_steps": marker_steps,
            }
        )

    if stop_branch and not saw_short_path:
        return None
    if not stop_branch and any(len(door["path_local"]) < depth for door in doors):
        return None
    if depth > 1 and not (saw_short_marker or saw_extra_marker):
        return None

    return {
        "depth": depth,
        "room_h": room_h,
        "room_w": room_w,
        "doors": doors,
    }


def _place_room(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    spec: dict[str, object],
) -> dict[str, object] | None:
    h, w = grid.shape
    room_h = int(spec["room_h"])
    room_w = int(spec["room_w"])
    doors = spec["doors"]

    max_up = max((int(door["corridor_len"]) for door in doors if door["side"] == "top"), default=0)
    max_down = max((int(door["corridor_len"]) for door in doors if door["side"] == "bottom"), default=0)
    max_left = max((int(door["corridor_len"]) for door in doors if door["side"] == "left"), default=0)
    max_right = max((int(door["corridor_len"]) for door in doors if door["side"] == "right"), default=0)

    min_r0 = max_up
    max_r0 = h - room_h - max_down
    min_c0 = max_left
    max_c0 = w - room_w - max_right
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None

    color_order = [int(v) for v in rng.permutation(MARKER_PALETTE)]
    while len(color_order) < len(doors):
        color_order.extend(int(v) for v in rng.permutation(MARKER_PALETTE))

    for _ in range(100):
        r0 = int(rng.integers(min_r0, max_r0 + 1))
        c0 = int(rng.integers(min_c0, max_c0 + 1))
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)

        absolute_doors: list[dict[str, object]] = []
        door_positions: set[tuple[int, int]] = set()
        corridor_cells: set[tuple[int, int]] = set()
        marker_cells: dict[tuple[int, int], int] = {}
        for idx, door in enumerate(doors):
            lr, lc = door["door_local"]
            door_pos = (r0 + int(lr), c0 + int(lc))
            door_positions.add(door_pos)
            side = str(door["side"])
            odr, odc = OUTWARD[side]
            path_abs = [(r0 + rr, c0 + cc) for rr, cc in door["path_local"]]
            corr_abs: list[tuple[int, int]] = []
            for step in range(1, int(door["corridor_len"]) + 1):
                rr = int(door_pos[0] + odr * step)
                cc = int(door_pos[1] + odc * step)
                corr_abs.append((rr, cc))
            if len(set(corr_abs)) != len(corr_abs):
                return None
            corridor_cells.update(corr_abs)
            color = color_order[idx]
            for step in door["marker_steps"]:
                marker_cells[corr_abs[int(step) - 1]] = color
            absolute_doors.append(
                {
                    "pos": door_pos,
                    "side": side,
                    "outward": (odr, odc),
                    "inward": (-odr, -odc),
                    "corridor_abs": corr_abs,
                    "marker_steps": list(door["marker_steps"]),
                    "color": color,
                    "path_abs": path_abs,
                }
            )

        wall_cells = _room_walls(bbox, door_positions)
        structure_cells = set(wall_cells) | corridor_cells
        reserve = _expand_cells(structure_cells | {(r, c) for r in range(r0, r0 + room_h) for c in range(c0, c0 + room_w)}, h, w)
        if reserve & blocked:
            continue

        for rr, cc in wall_cells:
            grid[rr, cc] = WALL
        for rr, cc in corridor_cells:
            grid[rr, cc] = WALL
        for (rr, cc), color in marker_cells.items():
            grid[rr, cc] = color

        blocked.update(reserve)
        return {"bbox": bbox, "doors": absolute_doors}
    return None


def _expected_from_spec(grid: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        depth = int(len(room["doors"]))
        for door in room["doors"]:
            color = int(door["color"])
            picked_steps = door["marker_steps"][: min(depth, len(door["marker_steps"]))]
            for step in picked_steps:
                rr, cc = door["corridor_abs"][int(step) - 1]
                out[rr, cc] = BG
            for rr, cc in door["path_abs"]:
                out[rr, cc] = color
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    target_rooms = int(rng.integers(1, 4))
    forced_depth = int(rng.integers(1, 4))

    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    placed_rooms: list[dict[str, object]] = []

    for room_index in range(target_rooms):
        spec = None
        for _ in range(200):
            spec = _sample_room_spec(rng, h, w, forced_depth if room_index == 0 else None)
            if spec is not None:
                break
        if spec is None:
            return None

        placed = None
        for _ in range(200):
            placed = _place_room(rng, grid, blocked, spec)
            if placed is not None:
                break
        if placed is None:
            return None
        placed_rooms.append(placed)

    expected = _expected_from_spec(grid, placed_rooms)

    detected = _find_rooms(grid, BG, WALL)
    if len(detected) != len(placed_rooms):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(100):
        h = int(rng.integers(22, 31))
        w = int(rng.integers(22, 31))
        room_h = int(rng.integers(5, 9))
        room_w = int(rng.integers(5, 9))
        r0 = int(rng.integers(5, h - room_h - 4))
        c0 = int(rng.integers(5, w - room_w - 4))
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)

        depth = int(rng.integers(1, 4))
        sides = [str(v) for v in rng.choice(np.array(SIDES, dtype=object), size=depth, replace=False)]
        door_positions: set[tuple[int, int]] = set()
        door_specs: list[dict[str, object]] = []
        for side in sides:
            if side in ("top", "bottom"):
                offset = int(rng.integers(1, room_w - 1))
                door = (r0 if side == "top" else r0 + room_h - 1, c0 + offset)
            else:
                offset = int(rng.integers(1, room_h - 1))
                door = (r0 + offset, c0 if side == "left" else c0 + room_w - 1)
            if door in door_positions:
                break
            odr, odc = OUTWARD[side]
            max_len = 4
            if not (0 <= door[0] + odr * max_len < h and 0 <= door[1] + odc * max_len < w):
                break
            door_positions.add(door)
            door_specs.append({"side": side, "pos": door, "outward": (odr, odc)})
        else:
            grid = np.full((h, w), BG, dtype=int)
            for rr, cc in _room_walls(bbox, door_positions):
                grid[rr, cc] = WALL

            colors = [int(v) for v in rng.permutation(MARKER_PALETTE)]
            for idx, door in enumerate(door_specs):
                dr, dc = door["outward"]
                rr, cc = door["pos"]
                corridor_len = int(rng.integers(2, 5))
                marker_count = int(rng.integers(1, 4 if depth == 1 else depth + 2))
                marker_count = min(marker_count, corridor_len)
                marker_steps = sorted(
                    int(v) for v in rng.choice(np.arange(1, corridor_len + 1), size=marker_count, replace=False)
                )
                marker_color = colors[idx % len(colors)]
                for step in range(1, corridor_len + 1):
                    cell = (int(rr + dr * step), int(cc + dc * step))
                    grid[cell] = marker_color if step in marker_steps else WALL

            if len(_find_rooms(grid, BG, WALL)) == 1:
                return grid
    raise RuntimeError("failed to generate a valid room-corridor instance")

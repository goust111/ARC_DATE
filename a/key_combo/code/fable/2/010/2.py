from __future__ import annotations

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def _gap_side(room: dict, gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = room["bbox"]
    r, c = gap
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    if r == r0:
        return "top"
    return "bottom"


def _room_from_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != WALL for r, c in corners):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != BG):
        return None

    gaps: list[tuple[int, int]] = []
    for r, c in _perimeter_cells(bbox):
        v = int(grid[r, c])
        if v == WALL:
            continue
        if v == BG and (r, c) not in corners:
            gaps.append((r, c))
            continue
        return None

    if not (1 <= len(gaps) <= 2):
        return None

    return {
        "bbox": bbox,
        "gaps": tuple(gaps),
        "gap_set": set(gaps),
        "interior_slice": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
    }


def _detect_rooms(grid: np.ndarray) -> list[dict]:
    wall_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == WALL)]
    wall_set = set(wall_coords)
    rooms: dict[tuple[int, int, int, int], dict] = {}
    for r0, c0 in wall_coords:
        for r1, c1 in wall_coords:
            if r1 <= r0 + 2 or c1 <= c0 + 2:
                continue
            if (r0, c1) not in wall_set or (r1, c0) not in wall_set:
                continue
            bbox = (r0, c0, r1, c1)
            room = _room_from_bbox(grid, bbox)
            if room is not None:
                rooms[bbox] = room
    return sorted(rooms.values(), key=lambda room: room["bbox"])


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int] | None:
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return (1, 0)
    if r == h - 1 and 0 < c < w - 1:
        return (-1, 0)
    if c == 0 and 0 < r < h - 1:
        return (0, 1)
    if c == w - 1 and 0 < r < h - 1:
        return (0, -1)
    return None


def _is_entry_gap(room: dict, gap: tuple[int, int], dr: int, dc: int) -> bool:
    side = _gap_side(room, gap)
    return (
        (dr, dc) == (0, 1)
        and side == "left"
        or (dr, dc) == (0, -1)
        and side == "right"
        or (dr, dc) == (1, 0)
        and side == "top"
        or (dr, dc) == (-1, 0)
        and side == "bottom"
    )


def _opposite_wall_cell(room: dict, gap: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    r0, c0, r1, c1 = room["bbox"]
    r, c = gap
    if (dr, dc) == (0, 1):
        return (r, c1)
    if (dr, dc) == (0, -1):
        return (r, c0)
    if (dr, dc) == (1, 0):
        return (r1, c)
    return (r0, c)


def _trace_marker(
    grid: np.ndarray,
    rooms: list[dict],
    gap_to_room: dict[tuple[int, int], int],
    marker: tuple[int, int],
) -> dict | None:
    drdc = _marker_direction(grid.shape, marker[0], marker[1])
    if drdc is None:
        return None
    dr, dc = drdc
    r, c = marker[0] + dr, marker[1] + dc
    passed: list[int] = []

    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value == WALL:
            return None
        if value != BG:
            return None

        gap = (r, c)
        room_idx = gap_to_room.get(gap)
        if room_idx is not None:
            room = rooms[room_idx]
            if _is_entry_gap(room, gap, dr, dc):
                exit_gap = _opposite_wall_cell(room, gap, dr, dc)
                if exit_gap in room["gap_set"] and exit_gap != gap:
                    passed.append(room_idx)
                    r, c = exit_gap[0] + dr, exit_gap[1] + dc
                    continue
                return {
                    "marker": marker,
                    "color": int(grid[marker]),
                    "room_idx": room_idx,
                    "entry_gap": gap,
                    "passed_rooms": tuple(passed),
                }
        r += dr
        c += dc

    return None


def _solve(grid: np.ndarray, *, return_debug: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _detect_rooms(g)
    gap_to_room: dict[tuple[int, int], int] = {}
    for idx, room in enumerate(rooms):
        for gap in room["gaps"]:
            gap_to_room[gap] = idx

    markers: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if int(g[r, c]) not in MARKER_COLORS:
                continue
            if _marker_direction(g.shape, r, c) is None:
                continue
            markers.append((r, c))
    markers.sort()

    raw_events = []
    for marker in markers:
        event = _trace_marker(g, rooms, gap_to_room, marker)
        if event is not None:
            raw_events.append(event)

    chosen_events: list[dict] = []
    used_rooms: set[int] = set()
    for event in raw_events:
        if event["room_idx"] in used_rooms:
            continue
        used_rooms.add(event["room_idx"])
        chosen_events.append(event)

    out = g.copy()
    for event in chosen_events:
        room = rooms[event["room_idx"]]
        rs, cs = room["interior_slice"]
        fill_mask = out[rs, cs] == BG
        out[rs, cs][fill_mask] = event["color"]
        er, ec = event["entry_gap"]
        out[er, ec] = WALL
        mr, mc = event["marker"]
        out[mr, mc] = BG

    if return_debug:
        return out, {"rooms": rooms, "markers": markers, "events": chosen_events, "raw_events": raw_events}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each edge marker straight through room doors so only the ray trapped inside a room captures it.

    Input structure:
    - Black background with several gray one-cell-thick hollow rectangles acting as rooms.
    - Each room has one or two one-cell wall gaps.
    - Colored singleton markers sit on the outer border and point inward along their row or column.

    Rule:
    - Every border marker emits a straight ray inward.
    - If the ray meets a gray wall cell before entering a room through a gap, nothing changes and that marker stays.
    - If the ray enters a room through a gap and can leave through an aligned opposite gap, the room stays hollow and the ray keeps going.
    - If the ray enters a room but the opposite wall blocks it, fill that room's interior background with the marker color, recolor only the entry gap back to gray, and erase the firing marker.

    Key insight:
    - Doors matter only along the actual sightline: rooms that the ray merely passes through are decoys, and only the room where the straight ray terminates gets captured.
    """
    return _solve(grid)


def _draw_room(grid: np.ndarray, bbox: tuple[int, int, int, int], gaps: list[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    for r, c in gaps:
        grid[r, c] = BG


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _aligned_chain_gaps(
    bbox: tuple[int, int, int, int],
    ray_coord: int,
    orientation: str,
    side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "h":
        if side == "left":
            return [(ray_coord, c0), (ray_coord, c1)]
        return [(ray_coord, c1), (ray_coord, c0)]
    if side == "top":
        return [(r0, ray_coord), (r1, ray_coord)]
    return [(r1, ray_coord), (r0, ray_coord)]


def _capture_room_gaps(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    ray_coord: int,
    orientation: str,
    side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "h":
        entry = (ray_coord, c0) if side == "left" else (ray_coord, c1)
        opposite = (ray_coord, c1) if side == "left" else (ray_coord, c0)
    else:
        entry = (r0, ray_coord) if side == "top" else (r1, ray_coord)
        opposite = (r1, ray_coord) if side == "top" else (r0, ray_coord)

    if rng.random() < 0.5:
        return [entry]

    candidates = [
        pt
        for pt in _perimeter_cells(bbox)
        if pt != entry
        and pt != opposite
        and pt not in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    ]
    second = candidates[int(rng.integers(0, len(candidates)))]
    return [entry, second]


def _place_distractor_room(
    rng: np.random.Generator,
    grid: np.ndarray,
    used_boxes: list[tuple[int, int, int, int]],
    reserved_rows: set[int],
    reserved_cols: set[int],
) -> bool:
    h, w = grid.shape
    for _ in range(80):
        room_h = int(rng.integers(4, 7))
        room_w = int(rng.integers(4, 7))
        r0 = int(rng.integers(2, h - room_h - 1))
        c0 = int(rng.integers(2, w - room_w - 1))
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        if any(not _boxes_far(bbox, other, pad=1) for other in used_boxes):
            continue
        if any(r0 <= rr <= bbox[2] for rr in reserved_rows):
            continue
        if any(c0 <= cc <= bbox[3] for cc in reserved_cols):
            continue

        sides = []
        for r in range(r0 + 1, bbox[2]):
            sides.append((r, c0))
            sides.append((r, bbox[3]))
        for c in range(c0 + 1, bbox[3]):
            sides.append((r0, c))
            sides.append((bbox[2], c))
        gap_count = 1 if rng.random() < 0.45 else 2
        idxs = rng.choice(len(sides), size=gap_count, replace=False)
        gaps = [sides[int(i)] for i in np.atleast_1d(idxs)]
        _draw_room(grid, bbox, gaps)
        used_boxes.append(bbox)
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "h" if rng.random() < 0.5 else "v"
    pass_count = 1 if rng.random() < 0.55 else 2
    n_rooms = pass_count + 1

    if orientation == "h":
        h = int(rng.integers(16, 29))
        w_min = 19 if n_rooms == 2 else 24
        w = int(rng.integers(w_min, 31))
        grid = np.zeros((h, w), dtype=int)
        side = "left" if rng.random() < 0.5 else "right"
        ray_r = int(rng.integers(3, h - 3))
        room_heights = [int(rng.integers(4, 7)) for _ in range(n_rooms)]
        room_widths = [int(rng.integers(4, 7)) for _ in range(n_rooms)]
        gaps_between = [int(rng.integers(2, 5)) for _ in range(n_rooms - 1)]
        lead = int(rng.integers(3, 6))
        trail = int(rng.integers(2, 5))
        need = sum(room_widths) + sum(gaps_between) + lead + trail
        if need >= w - 1:
            return None

        tops = []
        for room_h in room_heights:
            lo = max(2, ray_r - (room_h - 2))
            hi = min(h - room_h - 2, ray_r - 1)
            if lo > hi:
                return None
            tops.append(int(rng.integers(lo, hi + 1)))

        boxes: list[tuple[int, int, int, int]] = []
        cursor = lead
        if side == "left":
            for idx in range(n_rooms):
                r0 = tops[idx]
                c0 = cursor
                bbox = (r0, c0, r0 + room_heights[idx] - 1, c0 + room_widths[idx] - 1)
                boxes.append(bbox)
                cursor = bbox[3] + 1 + (gaps_between[idx] if idx < len(gaps_between) else 0)
        else:
            cursor = w - 1 - lead
            for idx in range(n_rooms):
                r0 = tops[idx]
                c1 = cursor
                bbox = (r0, c1 - room_widths[idx] + 1, r0 + room_heights[idx] - 1, c1)
                boxes.append(bbox)
                cursor = bbox[1] - 1 - (gaps_between[idx] if idx < len(gaps_between) else 0)

        marker_rows = set()
        for bbox in boxes[:-1]:
            _draw_room(grid, bbox, _aligned_chain_gaps(bbox, ray_r, "h", side))
        _draw_room(grid, boxes[-1], _capture_room_gaps(rng, boxes[-1], ray_r, "h", side))

        first_box = boxes[0]
        candidate_rows = [r for r in range(first_box[0], first_box[2] + 1) if r != ray_r]
        if not candidate_rows:
            return None
        block_r = candidate_rows[int(rng.integers(0, len(candidate_rows)))]
        marker_rows.update({ray_r, block_r})

        colors = list(rng.choice(MARKER_COLORS, size=2, replace=False))
        if side == "left":
            grid[ray_r, 0] = colors[0]
            grid[block_r, 0] = colors[1]
        else:
            grid[ray_r, w - 1] = colors[0]
            grid[block_r, w - 1] = colors[1]

        used_boxes = boxes[:]
        _place_distractor_room(rng, grid, used_boxes, reserved_rows=marker_rows, reserved_cols=set())
    else:
        h_min = 19 if n_rooms == 2 else 24
        h = int(rng.integers(h_min, 31))
        w = int(rng.integers(16, 29))
        grid = np.zeros((h, w), dtype=int)
        side = "top" if rng.random() < 0.5 else "bottom"
        ray_c = int(rng.integers(3, w - 3))
        room_heights = [int(rng.integers(4, 7)) for _ in range(n_rooms)]
        room_widths = [int(rng.integers(4, 7)) for _ in range(n_rooms)]
        gaps_between = [int(rng.integers(2, 5)) for _ in range(n_rooms - 1)]
        lead = int(rng.integers(3, 6))
        trail = int(rng.integers(2, 5))
        need = sum(room_heights) + sum(gaps_between) + lead + trail
        if need >= h - 1:
            return None

        lefts = []
        for room_w in room_widths:
            lo = max(2, ray_c - (room_w - 2))
            hi = min(w - room_w - 2, ray_c - 1)
            if lo > hi:
                return None
            lefts.append(int(rng.integers(lo, hi + 1)))

        boxes = []
        if side == "top":
            cursor = lead
            for idx in range(n_rooms):
                c0 = lefts[idx]
                bbox = (cursor, c0, cursor + room_heights[idx] - 1, c0 + room_widths[idx] - 1)
                boxes.append(bbox)
                cursor = bbox[2] + 1 + (gaps_between[idx] if idx < len(gaps_between) else 0)
        else:
            cursor = h - 1 - lead
            for idx in range(n_rooms):
                c0 = lefts[idx]
                bbox = (cursor - room_heights[idx] + 1, c0, cursor, c0 + room_widths[idx] - 1)
                boxes.append(bbox)
                cursor = bbox[0] - 1 - (gaps_between[idx] if idx < len(gaps_between) else 0)

        marker_cols = set()
        for bbox in boxes[:-1]:
            _draw_room(grid, bbox, _aligned_chain_gaps(bbox, ray_c, "v", side))
        _draw_room(grid, boxes[-1], _capture_room_gaps(rng, boxes[-1], ray_c, "v", side))

        first_box = boxes[0]
        candidate_cols = [c for c in range(first_box[1], first_box[3] + 1) if c != ray_c]
        if not candidate_cols:
            return None
        block_c = candidate_cols[int(rng.integers(0, len(candidate_cols)))]
        marker_cols.update({ray_c, block_c})

        colors = list(rng.choice(MARKER_COLORS, size=2, replace=False))
        if side == "top":
            grid[0, ray_c] = colors[0]
            grid[0, block_c] = colors[1]
        else:
            grid[h - 1, ray_c] = colors[0]
            grid[h - 1, block_c] = colors[1]

        used_boxes = boxes[:]
        _place_distractor_room(rng, grid, used_boxes, reserved_rows=set(), reserved_cols=marker_cols)

    out, debug = _solve(grid, return_debug=True)
    events = debug["events"]
    if len(events) < 1:
        return None
    if len(debug["markers"]) <= len(events):
        return None
    if not any(len(event["passed_rooms"]) >= 1 for event in events):
        return None
    if len(debug["rooms"]) <= len({event["room_idx"] for event in events}):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-ray capture input")

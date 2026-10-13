import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
GREY = 5
MARKER_PALETTE = (2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _can_place(cells: set[tuple[int, int]], reserved: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells:
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    expanded = _expand_cells(cells, h, w, pad=1)
    return expanded.isdisjoint(reserved)


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components(grid == WALL):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.any((patch == WALL) & (~border)):
            continue
        if not (
            int(patch[0, 0]) == WALL
            and int(patch[0, -1]) == WALL
            and int(patch[-1, 0]) == WALL
            and int(patch[-1, -1]) == WALL
        ):
            continue

        border_size = int(np.count_nonzero(border))
        if int(np.count_nonzero(patch == WALL)) < border_size - 1:
            continue

        interior = patch[1:-1, 1:-1]
        marker_colors = set(map(int, np.unique(interior))) - {BG, WALL, GREY}
        if len(marker_colors) != 1:
            continue
        marker_color = int(next(iter(marker_colors)))
        marker_count = int(np.count_nonzero(interior == marker_color))
        if marker_count <= 0:
            continue

        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "marker_color": marker_color,
                "marker_count": marker_count,
            }
        )
    return rooms


def _channel_touches_room(room: dict, channel: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = room["bbox"]
    for r, c in channel:
        if r0 < r < r1 and c0 < c < c1:
            return True
        if r0 <= r <= r1 and c0 <= c <= c1:
            on_vertical_wall = c in (c0, c1) and r0 < r < r1
            on_horizontal_wall = r in (r0, r1) and c0 < c < c1
            if on_vertical_wall or on_horizontal_wall:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find room pairs connected by grey channels and let the color flow from the
    room with more marker pixels into the room with fewer marker pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = _find_rooms(g)
    channels: list[dict] = []
    for coords in _components(g == GREY):
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if len(rows) != 1 and len(cols) != 1:
            continue
        linked = [idx for idx, room in enumerate(rooms) if _channel_touches_room(room, coords)]
        if len(linked) != 2:
            continue
        channels.append({"coords": coords, "rooms": tuple(linked)})

    for channel in channels:
        left = rooms[channel["rooms"][0]]
        right = rooms[channel["rooms"][1]]
        if int(left["marker_count"]) == int(right["marker_count"]):
            continue

        if int(left["marker_count"]) > int(right["marker_count"]):
            source, target = left, right
        else:
            source, target = right, left
        source_color = int(source["marker_color"])

        for r, c in channel["coords"]:
            out[r, c] = source_color

        r0, c0, r1, c1 = target["bbox"]
        target_patch = out[r0 + 1 : r1, c0 + 1 : c1]
        target_patch[target_patch == BG] = source_color

    return out


def _build_paired_structure(
    rng: random.Random,
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    orientation = rng.choice(("horizontal", "vertical"))

    inner_h1 = rng.randint(2, 5)
    inner_w1 = rng.randint(2, 6)
    inner_h2 = rng.randint(2, 5)
    inner_w2 = rng.randint(2, 6)
    room_h1, room_w1 = inner_h1 + 2, inner_w1 + 2
    room_h2, room_w2 = inner_h2 + 2, inner_w2 + 2
    gap = rng.randint(2, 6)
    margin = 1

    for _ in range(120):
        if orientation == "horizontal":
            total_w = room_w1 + gap + room_w2
            if total_w > w - 2 * margin:
                return None
            top1 = rng.randint(margin, h - margin - room_h1)
            top2 = rng.randint(margin, h - margin - room_h2)
            left1 = rng.randint(margin, w - margin - total_w)
            left2 = left1 + room_w1 + gap
            overlap_lo = max(top1 + 1, top2 + 1)
            overlap_hi = min(top1 + room_h1 - 2, top2 + room_h2 - 2)
            if overlap_lo > overlap_hi:
                continue
            row = rng.randint(overlap_lo, overlap_hi)
            start_col = left1 + room_w1 - 2
            end_col = left2 + 1
            channel = [(row, c) for c in range(start_col, end_col + 1)]
            room1 = {
                "bbox": (top1, left1, top1 + room_h1 - 1, left1 + room_w1 - 1),
                "blocked": {(row, left1 + room_w1 - 2)},
                "inner_shape": (inner_h1, inner_w1),
            }
            room2 = {
                "bbox": (top2, left2, top2 + room_h2 - 1, left2 + room_w2 - 1),
                "blocked": {(row, left2 + 1)},
                "inner_shape": (inner_h2, inner_w2),
            }
        else:
            total_h = room_h1 + gap + room_h2
            if total_h > h - 2 * margin:
                return None
            top1 = rng.randint(margin, h - margin - total_h)
            top2 = top1 + room_h1 + gap
            left1 = rng.randint(margin, w - margin - room_w1)
            left2 = rng.randint(margin, w - margin - room_w2)
            overlap_lo = max(left1 + 1, left2 + 1)
            overlap_hi = min(left1 + room_w1 - 2, left2 + room_w2 - 2)
            if overlap_lo > overlap_hi:
                continue
            col = rng.randint(overlap_lo, overlap_hi)
            start_row = top1 + room_h1 - 2
            end_row = top2 + 1
            channel = [(r, col) for r in range(start_row, end_row + 1)]
            room1 = {
                "bbox": (top1, left1, top1 + room_h1 - 1, left1 + room_w1 - 1),
                "blocked": {(top1 + room_h1 - 2, col)},
                "inner_shape": (inner_h1, inner_w1),
            }
            room2 = {
                "bbox": (top2, left2, top2 + room_h2 - 1, left2 + room_w2 - 1),
                "blocked": {(top2 + 1, col)},
                "inner_shape": (inner_h2, inner_w2),
            }

        cells = _rect_cells(*room1["bbox"]) | _rect_cells(*room2["bbox"]) | set(channel)
        if not _can_place(cells, reserved, h, w):
            continue
        return [room1, room2], channel

    return None


def _build_isolated_room(
    rng: random.Random,
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> dict | None:
    margin = 1
    for _ in range(120):
        inner_h = rng.randint(2, 5)
        inner_w = rng.randint(2, 6)
        room_h, room_w = inner_h + 2, inner_w + 2
        top = rng.randint(margin, h - margin - room_h)
        left = rng.randint(margin, w - margin - room_w)
        room = {
            "bbox": (top, left, top + room_h - 1, left + room_w - 1),
            "blocked": set(),
            "inner_shape": (inner_h, inner_w),
        }
        cells = _rect_cells(*room["bbox"])
        if _can_place(cells, reserved, h, w):
            return room
    return None


def _room_inner_cells(room: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    blocked = set(room["blocked"])
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) not in blocked:
                cells.append((r, c))
    return cells


def _choose_counts(rng: random.Random, room1: dict, room2: dict, scenario: str) -> tuple[int, int]:
    max1 = max(1, len(_room_inner_cells(room1)) - 1)
    max2 = max(1, len(_room_inner_cells(room2)) - 1)
    if scenario == "tie":
        count = rng.randint(1, min(max1, max2))
        return count, count
    if scenario == "first":
        high = rng.randint(2, max1)
        low = rng.randint(1, min(max2, high - 1))
        return high, low
    high = rng.randint(2, max2)
    low = rng.randint(1, min(max1, high - 1))
    return low, high


def _paint_room_walls(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for pair in pairs:
        room1 = pair["rooms"][0]
        room2 = pair["rooms"][1]
        if int(room1["count"]) == int(room2["count"]):
            continue
        if int(room1["count"]) > int(room2["count"]):
            source, target = room1, room2
        else:
            source, target = room2, room1
        source_color = int(source["color"])
        for r, c in pair["channel"]:
            out[r, c] = source_color
        r0, c0, r1, c1 = target["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) == BG:
                    out[r, c] = source_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    pair_count = rng.choice((1, 2))
    isolated_count = rng.choice((1, 2)) if pair_count == 1 else rng.choice((0, 1))

    scenarios = [rng.choice(("first", "second", "tie")) for _ in range(pair_count)]
    if all(s == "tie" for s in scenarios):
        scenarios[rng.randrange(pair_count)] = rng.choice(("first", "second"))

    pairs: list[dict] = []
    rooms: list[dict] = []
    for scenario in scenarios:
        built = _build_paired_structure(rng, h, w, reserved)
        if built is None:
            return None
        pair_rooms, channel = built
        pair_record = {"rooms": pair_rooms, "channel": channel, "scenario": scenario}
        pairs.append(pair_record)
        rooms.extend(pair_rooms)
        reserved |= _rect_cells(*pair_rooms[0]["bbox"])
        reserved |= _rect_cells(*pair_rooms[1]["bbox"])
        reserved |= set(channel)

    for _ in range(isolated_count):
        room = _build_isolated_room(rng, h, w, reserved)
        if room is None:
            return None
        rooms.append(room)
        reserved |= _rect_cells(*room["bbox"])

    if len(rooms) > len(MARKER_PALETTE):
        return None

    colors = list(rng.sample(MARKER_PALETTE, len(rooms)))
    for room, color in zip(rooms, colors):
        room["color"] = int(color)

    for pair in pairs:
        room1 = pair["rooms"][0]
        room2 = pair["rooms"][1]
        count1, count2 = _choose_counts(rng, room1, room2, pair["scenario"])
        room1["count"] = int(count1)
        room2["count"] = int(count2)

    for room in rooms:
        if "count" in room:
            continue
        max_count = max(1, len(_room_inner_cells(room)) - 1)
        room["count"] = int(rng.randint(1, max_count))

    for room in rooms:
        _paint_room_walls(grid, room)

    for pair in pairs:
        for r, c in pair["channel"]:
            grid[r, c] = GREY

    for room in rooms:
        available = _room_inner_cells(room)
        if len(available) < int(room["count"]):
            return None
        chosen = rng.sample(available, int(room["count"]))
        room["markers"] = set(chosen)
        for r, c in chosen:
            grid[r, c] = int(room["color"])

    expected = _expected_output(grid, pairs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

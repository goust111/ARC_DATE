import random
from collections import deque

import numpy as np


BG = 0
RED = 2
WALL = 5
DOOR_COLORS = (1, 3, 4, 6, 7, 8, 9)

GRID_MIN = 24
GRID_MAX = 30
PAIR_THICKNESS_RANGE = (6, 7)
PAIR_LENGTH_RANGE = (6, 8)
SINGLE_SIZE_RANGE = (6, 7)
PAIR_GAP_RANGE = (3, 6)
GEN_RETRIES = 400

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _door_side(r0: int, c0: int, r1: int, c1: int, door_r: int, door_c: int) -> str | None:
    if door_r in (r0, r1) and door_c in (c0, c1):
        return None
    if door_r == r0:
        return "top"
    if door_r == r1:
        return "bottom"
    if door_c == c0:
        return "left"
    if door_c == c1:
        return "right"
    return None


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    for comp in _components_mask(g == WALL):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        boundary = []
        for r in range(r0, r1 + 1):
            boundary.append((r, c0))
            boundary.append((r, c1))
        for c in range(c0 + 1, c1):
            boundary.append((r0, c))
            boundary.append((r1, c))

        door: tuple[int, int, int] | None = None
        valid = True
        for r, c in boundary:
            val = int(g[r, c])
            if val == WALL:
                continue
            if val == BG or door is not None:
                valid = False
                break
            door = (r, c, val)
        if not valid or door is None:
            continue

        sub = g[r0 + 1 : r1, c0 + 1 : c1]
        if sub.size == 0 or np.any(sub == WALL):
            continue

        door_r, door_c, door_color = door
        side = _door_side(r0, c0, r1, c1, door_r, door_c)
        if side is None:
            continue

        red_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) == RED]
        if not red_cells:
            continue
        red_components = _components_mask(sub == RED)
        if len(red_components) != 1:
            continue
        item_cells = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in red_components[0]]

        key_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) not in (BG, RED)]
        if len(key_cells) != 1:
            continue
        key_r, key_c = key_cells[0]
        key_color = int(g[key_r, key_c])

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "door_pos": (door_r, door_c),
                "door_color": int(door_color),
                "side": side,
                "item_cells": item_cells,
                "key_pos": (key_r, key_c),
                "key_color": key_color,
            }
        )

    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _open_order(rooms: list[dict]) -> list[int]:
    door_to_room = {room["door_color"]: idx for idx, room in enumerate(rooms)}
    key_targets = {room["key_color"] for room in rooms if room["key_color"] in door_to_room}
    starts = [idx for idx, room in enumerate(rooms) if room["door_color"] not in key_targets]
    starts.sort(key=lambda idx: rooms[idx]["bbox"])

    order: list[int] = []
    opened: set[int] = set()
    q = deque(starts)
    while q:
        idx = q.popleft()
        if idx in opened:
            continue
        opened.add(idx)
        order.append(idx)
        nxt = door_to_room.get(rooms[idx]["key_color"])
        if nxt is not None and nxt not in opened:
            q.append(nxt)

    for idx in range(len(rooms)):
        if idx in opened:
            continue
        q = deque([idx])
        while q:
            cur = q.popleft()
            if cur in opened:
                continue
            opened.add(cur)
            order.append(cur)
            nxt = door_to_room.get(rooms[cur]["key_color"])
            if nxt is not None and nxt not in opened:
                q.append(nxt)
    return order


def _push_item(
    cells: list[tuple[int, int]],
    side: str,
    wall_mask: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = wall_mask.shape
    dr, dc = DIRS[side]
    cur = list(cells)
    while True:
        nxt: list[tuple[int, int]] = []
        for r, c in cur:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or bool(wall_mask[nr, nc]):
                return cur
            nxt.append((nr, nc))
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each keyed room door and rigidly push the red item out through that doorway.

    Input structure:
    - The background is black.
    - Grey rectangular wall rings define rooms; each ring has exactly one colored boundary pixel acting as a door.
    - Every room interior contains one single-pixel key whose color matches some room's door, plus one connected red item.

    Rule:
    - Identify every room and match each interior key color to the unique door of the same color.
    - Start from any room whose door color is not mentioned by another room's key, then follow the released key-to-door chain until all rooms are opened.
    - Remove every key and door, and translate each red item in the direction of its own door until the next step would hit the grid edge or a grey wall.

    Key insight:
    - The color relation is only for deciding which room unlocks next; the actual motion is set by the geometric side of each room's doorway.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_mask = g == WALL
    rooms = _find_rooms(g)
    order = _open_order(rooms)

    for room in rooms:
        dr, dc = room["door_pos"]
        kr, kc = room["key_pos"]
        out[dr, dc] = BG
        out[kr, kc] = BG
        for r, c in room["item_cells"]:
            out[r, c] = BG

    for idx in order:
        room = rooms[idx]
        final_cells = _push_item(room["item_cells"], room["side"], wall_mask)
        for r, c in final_cells:
            out[r, c] = RED
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_pair(rng: random.Random, h: int, w: int) -> tuple[list[dict], list[str]] | None:
    orientation = rng.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        room_h = rng.randint(*PAIR_THICKNESS_RANGE)
        w1 = rng.randint(*PAIR_LENGTH_RANGE)
        w2 = rng.randint(*PAIR_LENGTH_RANGE)
        gap_max = min(PAIR_GAP_RANGE[1], w - w1 - w2 - 4)
        if gap_max < PAIR_GAP_RANGE[0]:
            return None
        gap = rng.randint(PAIR_GAP_RANGE[0], gap_max)
        total_w = w1 + gap + w2
        if rng.random() < 0.5:
            r0_lo, r0_hi = 2, 4
            free_corners = ["bl", "br"]
        else:
            r0_lo, r0_hi = max(2, h - room_h - 4), h - room_h - 2
            free_corners = ["tl", "tr"]
        if r0_lo > r0_hi:
            return None
        c0_hi = w - total_w - 2
        if c0_hi < 2:
            return None
        r0 = rng.randint(r0_lo, r0_hi)
        c0 = rng.randint(2, c0_hi)
        left = (r0, c0, r0 + room_h - 1, c0 + w1 - 1)
        right = (r0, c0 + w1 + gap, r0 + room_h - 1, c0 + w1 + gap + w2 - 1)
        door_rows = rng.sample(list(range(r0 + 1, r0 + room_h - 1)), 2)
        return (
            [
                {"bbox": left, "side": "right", "lane": door_rows[0], "travel_limit": gap},
                {"bbox": right, "side": "left", "lane": door_rows[1], "travel_limit": gap},
            ],
            free_corners,
        )

    room_w = rng.randint(*PAIR_THICKNESS_RANGE)
    h1 = rng.randint(*PAIR_LENGTH_RANGE)
    h2 = rng.randint(*PAIR_LENGTH_RANGE)
    gap_max = min(PAIR_GAP_RANGE[1], h - h1 - h2 - 4)
    if gap_max < PAIR_GAP_RANGE[0]:
        return None
    gap = rng.randint(PAIR_GAP_RANGE[0], gap_max)
    total_h = h1 + gap + h2
    if rng.random() < 0.5:
        c0_lo, c0_hi = 2, 4
        free_corners = ["tr", "br"]
    else:
        c0_lo, c0_hi = max(2, w - room_w - 4), w - room_w - 2
        free_corners = ["tl", "bl"]
    if c0_lo > c0_hi:
        return None
    r0_hi = h - total_h - 2
    if r0_hi < 2:
        return None
    c0 = rng.randint(c0_lo, c0_hi)
    r0 = rng.randint(2, r0_hi)
    top = (r0, c0, r0 + h1 - 1, c0 + room_w - 1)
    bottom = (r0 + h1 + gap, c0, r0 + h1 + gap + h2 - 1, c0 + room_w - 1)
    door_cols = rng.sample(list(range(c0 + 1, c0 + room_w - 1)), 2)
    return (
        [
            {"bbox": top, "side": "bottom", "lane": door_cols[0], "travel_limit": gap},
            {"bbox": bottom, "side": "top", "lane": door_cols[1], "travel_limit": gap},
        ],
        free_corners,
    )


def _sample_single(rng: random.Random, h: int, w: int, corner: str) -> dict | None:
    room_h = rng.randint(*SINGLE_SIZE_RANGE)
    room_w = rng.randint(*SINGLE_SIZE_RANGE)

    if corner[0] == "t":
        r0_lo, r0_hi = 2, 4
    else:
        r0_lo, r0_hi = max(2, h - room_h - 4), h - room_h - 2
    if corner[1] == "l":
        c0_lo, c0_hi = 2, 4
    else:
        c0_lo, c0_hi = max(2, w - room_w - 4), w - room_w - 2
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = rng.randint(r0_lo, r0_hi)
    c0 = rng.randint(c0_lo, c0_hi)
    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    side_options = {
        "tl": ["top", "left"],
        "tr": ["top", "right"],
        "bl": ["bottom", "left"],
        "br": ["bottom", "right"],
    }
    side = rng.choice(side_options[corner])
    if side in ("top", "bottom"):
        lane = rng.randint(c0 + 1, c1 - 1)
        travel_limit = r0 if side == "top" else h - 1 - r1
    else:
        lane = rng.randint(r0 + 1, r1 - 1)
        travel_limit = c0 if side == "left" else w - 1 - c1

    if travel_limit < 2:
        return None
    return {"bbox": (r0, c0, r1, c1), "side": side, "lane": lane, "travel_limit": travel_limit}


def _assign_chain_colors(rng: random.Random, count: int) -> tuple[list[int], list[int]]:
    door_colors = rng.sample(list(DOOR_COLORS), count)
    order = list(range(count))
    rng.shuffle(order)
    key_colors = [BG] * count
    for pos in range(count - 1):
        key_colors[order[pos]] = door_colors[order[pos + 1]]
    back_target_choices = order[1:-1] if count > 3 else [order[1]]
    key_colors[order[-1]] = door_colors[rng.choice(back_target_choices)]
    return door_colors, key_colors


def _door_cell(spec: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = spec["bbox"]
    lane = spec["lane"]
    side = spec["side"]
    if side == "top":
        return r0, lane
    if side == "bottom":
        return r1, lane
    if side == "left":
        return lane, c0
    return lane, c1


def _item_cells(rng: random.Random, spec: dict) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = spec["bbox"]
    side = spec["side"]
    travel_limit = int(spec["travel_limit"])
    if side in ("left", "right"):
        max_len = min(4, travel_limit, c1 - c0 - 1)
        if max_len < 2:
            return None
        length = rng.randint(2, max_len)
        row = int(spec["lane"])
        left = rng.randint(c0 + 1, c1 - length)
        return [(row, left + offset) for offset in range(length)]

    max_len = min(4, travel_limit, r1 - r0 - 1)
    if max_len < 2:
        return None
    length = rng.randint(2, max_len)
    col = int(spec["lane"])
    top = rng.randint(r0 + 1, r1 - length)
    return [(top + offset, col) for offset in range(length)]


def _key_cell(rng: random.Random, spec: dict, item_cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = spec["bbox"]
    choices = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if (r, c) not in item_cells]
    if not choices:
        return None
    return rng.choice(choices)


def _draw_room(grid: np.ndarray, spec: dict, door_color: int, key_color: int, key_pos: tuple[int, int], item_cells: list[tuple[int, int]]):
    r0, c0, r1, c1 = spec["bbox"]
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    dr, dc = _door_cell(spec)
    grid[dr, dc] = int(door_color)
    kr, kc = key_pos
    grid[kr, kc] = int(key_color)
    for r, c in item_cells:
        grid[r, c] = RED


def _build_grid(rng: random.Random, h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    door_colors, key_colors = _assign_chain_colors(rng, len(specs))
    grid = np.full((h, w), BG, dtype=int)

    for idx, spec in enumerate(specs):
        items = _item_cells(rng, spec)
        if items is None:
            return None
        key_pos = _key_cell(rng, spec, set(items))
        if key_pos is None:
            return None
        _draw_room(grid, spec, door_colors[idx], key_colors[idx], key_pos, items)

    parsed = _find_rooms(grid)
    if len(parsed) != len(specs):
        return None
    door_set = {room["door_color"] for room in parsed}
    if len(door_set) != len(specs):
        return None
    starts = [room for room in parsed if room["door_color"] not in {other["key_color"] for other in parsed}]
    if len(starts) != 1:
        return None
    _ = main(grid)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    pair_result = _sample_pair(rng, h, w)
    if pair_result is None:
        return None
    specs, free_corners = pair_result

    room_count = rng.choice([3, 4])
    corner_choices = rng.sample(free_corners, room_count - 2)
    for corner in corner_choices:
        spec = _sample_single(rng, h, w, corner)
        if spec is None:
            return None
        if any(_boxes_intersect(spec["bbox"], other["bbox"], pad=1) for other in specs):
            return None
        specs.append(spec)

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _boxes_intersect(specs[i]["bbox"], specs[j]["bbox"], pad=1):
                return None

    return _build_grid(rng, h, w, specs)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")

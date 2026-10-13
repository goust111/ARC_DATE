from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    comp_id = -np.ones((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(comp_id[r, c]) != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            comp_id[r, c] = idx
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(comp_id[nr, nc]) == -1:
                        comp_id[nr, nc] = idx
                        q.append((nr, nc))
            comps.append(pts)
    return comps, comp_id


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _corner_bg(g: np.ndarray) -> int:
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _wall_color(g: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(g[g != bg], return_counts=True)
    assert vals.size > 0, "expected at least one non-background color"
    return int(vals[np.argmax(cnts)])


def _roomish_mask(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    roomish = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(passable[r : r + 2, c : c + 2].all()):
                roomish[r : r + 2, c : c + 2] = True
    return roomish


def _bfs_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if start == goal:
        return [start]

    h, w = mask.shape
    assert 0 <= start[0] < h and 0 <= start[1] < w and bool(mask[start]), "invalid bfs start"
    assert 0 <= goal[0] < h and 0 <= goal[1] < w and bool(mask[goal]), "invalid bfs goal"

    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not bool(mask[nxt]) or nxt in prev:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)

    assert goal in prev, "corridor path must exist"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _reading_key(room: dict) -> tuple[int, int, int, int]:
    return room["bbox"]


def _find_room_door(
    g: np.ndarray,
    bg: int,
    wall: int,
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = g.shape
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(g[r0 - 1, c]) not in (bg, wall):
                candidates.append(((r0 - 1, c), (-1, 0)))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if int(g[r1 + 1, c]) not in (bg, wall):
                candidates.append(((r1 + 1, c), (1, 0)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(g[r, c0 - 1]) not in (bg, wall):
                candidates.append(((r, c0 - 1), (0, -1)))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if int(g[r, c1 + 1]) not in (bg, wall):
                candidates.append(((r, c1 + 1), (0, 1)))

    assert len(candidates) == 1, "each room must expose exactly one colored door"
    door_pos, direction = candidates[0]
    start = (door_pos[0] + direction[0], door_pos[1] + direction[1])
    assert 0 <= start[0] < h and 0 <= start[1] < w, "corridor cell beyond door must be in bounds"
    assert int(g[start]) == bg, "corridor must continue from the door into a black cell"
    return door_pos, start


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color corridor routes from each room's key to the matching same-colored door.

    Input structure:
    - The background is black and grey orthogonal walls carve out rectangular rooms
      plus one-cell-wide black corridors.
    - Every room contains exactly one colored key pixel in its interior.
    - Every room wall has exactly one colored door pixel at a one-cell gap; door
      colors are a derangement of the key colors.

    Rule:
    - Match each key to the door of the same color in another room.
    - Identify the black corridor cell just outside each source room's door and the
      black corridor cell just outside the matched destination door.
    - Process rooms in reading order of their room bounding boxes; for each room,
      fill the unique corridor path between those two black cells with the room's
      key color, letting later rooms overwrite earlier colors on shared corridor cells.
    - Replace each key pixel by black and keep every door pixel and wall pixel unchanged.

    Key insight:
    - The task combines two inferences: keys and doors pair by color across rooms,
      while the wall layout defines the unique corridor topology; room detection comes
      from finding the rectangular open areas rather than treating all black pixels alike.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _corner_bg(g)
    wall = _wall_color(g, bg)
    passable = g != wall
    roomish = _roomish_mask(passable)
    comps, comp_id = _components(roomish)

    special = [(int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != wall))]
    assert special, "expected colored keys and doors"

    key_cells = [pos for pos in special if bool(roomish[pos])]
    door_cells = [pos for pos in special if not bool(roomish[pos])]
    assert key_cells and door_cells, "expected both keys and doors"

    rooms_by_comp: dict[int, dict] = {}
    for r, c in key_cells:
        idx = int(comp_id[r, c])
        assert idx >= 0, "every key must lie inside a room interior"
        rooms_by_comp.setdefault(idx, {"coords": comps[idx], "keys": []})
        rooms_by_comp[idx]["keys"].append((r, c))

    rooms: list[dict] = []
    room_mask = np.zeros(g.shape, dtype=bool)
    for idx, info in rooms_by_comp.items():
        coords = info["coords"]
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        keys = info["keys"]
        assert len(keys) == 1, "each room must contain exactly one key"
        key_pos = keys[0]
        key_color = int(g[key_pos])
        door_pos, corridor_start = _find_room_door(g, bg, wall, bbox)
        door_color = int(g[door_pos])
        assert key_color != door_color, "a room's key color must differ from its own door color"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] != wall), "room bbox must be fully open"
        room_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        rooms.append(
            {
                "comp": idx,
                "bbox": bbox,
                "key_pos": key_pos,
                "key_color": key_color,
                "door_pos": door_pos,
                "door_color": door_color,
                "corridor_start": corridor_start,
            }
        )

    rooms.sort(key=_reading_key)
    key_lookup = {room["key_color"]: room for room in rooms}
    door_lookup = {room["door_color"]: room for room in rooms}
    assert set(key_lookup) == set(door_lookup), "key and door colors must match bijectively"

    corridor_mask = (g == bg) & (~room_mask)
    out = g.copy()
    for room in rooms:
        target = door_lookup[room["key_color"]]
        path = _bfs_path(corridor_mask, room["corridor_start"], target["corridor_start"])
        for r, c in path:
            out[r, c] = room["key_color"]
    for room in rooms:
        kr, kc = room["key_pos"]
        out[kr, kc] = bg
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _random_derangement(n: int) -> list[int]:
    assert n >= 2
    order = list(range(n))
    while True:
        random.shuffle(order)
        if all(i != order[i] for i in range(n)):
            return order


def _sample_presence(branch_count: int) -> list[tuple[bool, bool]]:
    while True:
        layout: list[tuple[bool, bool]] = []
        total = 0
        top_total = 0
        bottom_total = 0
        both_total = 0
        for _ in range(branch_count):
            top = random.random() < 0.8
            bottom = random.random() < 0.8
            if not top and not bottom:
                if random.random() < 0.5:
                    top = True
                else:
                    bottom = True
            layout.append((top, bottom))
            total += int(top) + int(bottom)
            top_total += int(top)
            bottom_total += int(bottom)
            both_total += int(top and bottom)
        if 4 <= total <= 6 and top_total > 0 and bottom_total > 0 and both_total > 0:
            return layout


def _transform_grid(g: np.ndarray) -> np.ndarray:
    out = g.copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k).copy()
    return out


def _generate_candidate() -> np.ndarray | None:
    branch_count = random.choice([2, 3])
    presence = _sample_presence(branch_count)

    branches: list[dict] = []
    room_count = 0
    for top_present, bottom_present in presence:
        top_h = random.randint(3, 5) if top_present else 0
        top_w = random.randint(3, 5) if top_present else 0
        bottom_h = random.randint(3, 5) if bottom_present else 0
        bottom_w = random.randint(3, 5) if bottom_present else 0
        widths = [w for w in (top_w, bottom_w) if w > 0]
        door_offset = random.randint(0, min(widths) - 1)
        top_stem = random.randint(1, 4) if top_present else 0
        bottom_stem = random.randint(1, 4) if bottom_present else 0
        max_w = max(widths)
        branches.append(
            {
                "top_present": top_present,
                "bottom_present": bottom_present,
                "top_h": top_h,
                "top_w": top_w,
                "bottom_h": bottom_h,
                "bottom_w": bottom_w,
                "door_offset": door_offset,
                "top_stem": top_stem,
                "bottom_stem": bottom_stem,
                "left_extent": 1 + door_offset,
                "right_extent": max_w - door_offset,
                "span": max_w + 2,
            }
        )
        room_count += int(top_present) + int(bottom_present)

    top_need = max((b["top_h"] + b["top_stem"] + 1) for b in branches if b["top_present"])
    bottom_need = max((b["bottom_h"] + b["bottom_stem"] + 1) for b in branches if b["bottom_present"])
    trunk_row = top_need + 1
    block_h = trunk_row + bottom_need + 2

    base_side = 1
    base_gap = 2
    min_w = 2 * base_side + sum(int(b["span"]) for b in branches) + base_gap * (branch_count - 1)
    if block_h > 28 or min_w > 28:
        return None

    width_slack = min(28 - min_w, 4)
    extra = random.randint(0, max(0, width_slack))
    extras = _random_composition(extra, branch_count + 1)

    pos = base_side + extras[0]
    for idx, branch in enumerate(branches):
        branch["outer_left"] = pos
        branch["x"] = pos + int(branch["left_extent"])
        pos += int(branch["span"])
        if idx + 1 < branch_count:
            pos += base_gap + extras[idx + 1]
    block_w = pos + base_side + extras[-1]
    if block_w > 28:
        return None

    block = np.full((block_h, block_w), WALL, dtype=int)
    rooms: list[dict] = []
    branch_xs = [int(branch["x"]) for branch in branches]
    left_stub = random.randint(0, 2)
    right_stub = random.randint(0, 2)
    trunk_c0 = max(0, min(branch_xs) - left_stub)
    trunk_c1 = min(block_w - 1, max(branch_xs) + right_stub)
    block[trunk_row, trunk_c0 : trunk_c1 + 1] = BG

    for branch in branches:
        x = int(branch["x"])
        offset = int(branch["door_offset"])

        if branch["top_present"]:
            outer_left = x - (1 + offset)
            outer_top = trunk_row - int(branch["top_stem"]) - int(branch["top_h"]) - 1
            h = int(branch["top_h"])
            w = int(branch["top_w"])
            block[outer_top + 1 : outer_top + 1 + h, outer_left + 1 : outer_left + 1 + w] = BG
            door_r = outer_top + h + 1
            block[door_r + 1 : trunk_row + 1, x] = BG
            rooms.append(
                {
                    "bbox": (outer_top + 1, outer_left + 1, outer_top + h, outer_left + w),
                    "door_pos": (door_r, x),
                }
            )

        if branch["bottom_present"]:
            outer_left = x - (1 + offset)
            outer_top = trunk_row + int(branch["bottom_stem"])
            h = int(branch["bottom_h"])
            w = int(branch["bottom_w"])
            block[outer_top + 1 : outer_top + 1 + h, outer_left + 1 : outer_left + 1 + w] = BG
            door_r = outer_top
            block[trunk_row:door_r, x] = BG
            rooms.append(
                {
                    "bbox": (outer_top + 1, outer_left + 1, outer_top + h, outer_left + w),
                    "door_pos": (door_r, x),
                }
            )

    if len(rooms) != room_count:
        return None

    key_colors = random.sample(COLOR_POOL, room_count)
    perm = _random_derangement(room_count)

    for idx, room in enumerate(rooms):
        room["key_color"] = key_colors[idx]
        room["door_color"] = key_colors[perm[idx]]
        r0, c0, r1, c1 = room["bbox"]
        key_r = random.randint(r0, r1)
        key_c = random.randint(c0, c1)
        room["key_pos"] = (key_r, key_c)
        block[key_r, key_c] = room["key_color"]
        dr, dc = room["door_pos"]
        block[dr, dc] = room["door_color"]

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    final_h = block_h + top_margin + bottom_margin
    final_w = block_w + left_margin + right_margin
    if final_h > 30 or final_w > 30:
        return None

    grid = np.full((final_h, final_w), BG, dtype=int)
    grid[top_margin : top_margin + block_h, left_margin : left_margin + block_w] = block
    grid = _transform_grid(grid)

    try:
        out = main(grid)
    except Exception:
        return None

    fill_mask = (grid == BG) & (out != BG)
    fill_colors = {int(v) for v in out[fill_mask].tolist()}
    if int(fill_mask.sum()) < room_count or len(fill_colors) < 2:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door corridor puzzle")

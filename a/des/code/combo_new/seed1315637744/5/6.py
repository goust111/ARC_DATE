from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = tuple(range(10))
DOOR_COLORS = tuple(color for color in PALETTE if color not in {BG, WALL})
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _door_orientation(g: np.ndarray, r: int, c: int, wall: int) -> str | None:
    if not (0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1):
        return None
    vertical = (
        int(g[r - 1, c]) == wall
        and int(g[r + 1, c]) == wall
        and int(g[r, c - 1]) != wall
        and int(g[r, c + 1]) != wall
    )
    horizontal = (
        int(g[r, c - 1]) == wall
        and int(g[r, c + 1]) == wall
        and int(g[r - 1, c]) != wall
        and int(g[r + 1, c]) != wall
    )
    if vertical == horizontal:
        return None
    return "vertical" if vertical else "horizontal"


def _label_rooms(barrier: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    h, w = barrier.shape
    passable = ~barrier
    comp_ids = -np.ones((h, w), dtype=int)
    rooms: dict[int, tuple[int, int, int, int]] = {}

    for cid, cells in enumerate(_components(passable)):
        for r, c in cells:
            comp_ids[r, c] = cid
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        rooms[cid] = (r0, c0, r1, c1)
    return comp_ids, rooms


def _extract_content(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    return [
        (int(rr), int(cc), int(patch[rr, cc]))
        for rr, cc in np.argwhere(patch != BG)
    ]


def _paint_content(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    content: list[tuple[int, int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    for dr, dc, color in content:
        assert 0 <= dr < h and 0 <= dc < w, "swapped content must fit the destination room"
        out[r0 + dr, c0 + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair rooms by matching door colors and swap the contents inside each paired room.

    Input structure:
    - The background is black.
    - Grey one-pixel-thick wall rectangles enclose several rectangular rooms.
    - Some room walls contain a single non-grey, non-black door pixel; each door color appears exactly twice.
    - Each room interior may contain a small non-black pattern or may be empty.

    Rule:
    - Treat the colored wall pixels as doors and identify the unique enclosed rectangular room adjacent to each one.
    - For each door color, pair the two rooms whose walls contain that color.
    - Swap the interior non-black content of each paired room, preserving every pixel's offset from the room interior's top-left corner.
    - Keep the walls, doors, room positions, and any room without a door unchanged.

    Key insight:
    - The colored pixels on the walls are not room contents; they are structural markers that define which enclosed rooms exchange their interior patterns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g != BG), "grid must contain walls or room contents"

    non_bg = g[g != BG]
    wall_vals, wall_counts = np.unique(non_bg, return_counts=True)
    wall = int(wall_vals[np.argmax(wall_counts)])

    h, w = g.shape
    door_mask = np.zeros((h, w), dtype=bool)
    door_cells: list[tuple[int, int, str, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color in {BG, wall}:
                continue
            orientation = _door_orientation(g, r, c, wall)
            if orientation is None:
                continue
            door_mask[r, c] = True
            door_cells.append((r, c, orientation, color))

    barrier = (g == wall) | door_mask
    comp_ids, rooms = _label_rooms(barrier)

    color_to_rooms: dict[int, list[int]] = defaultdict(list)
    room_to_colors: dict[int, set[int]] = defaultdict(set)
    for r, c, orientation, color in door_cells:
        neighbor_coords = (
            ((r, c - 1), (r, c + 1))
            if orientation == "vertical"
            else ((r - 1, c), (r + 1, c))
        )
        adjacent_rooms = []
        for nr, nc in neighbor_coords:
            if barrier[nr, nc]:
                continue
            room_id = int(comp_ids[nr, nc])
            if room_id in rooms:
                adjacent_rooms.append(room_id)
        adjacent_rooms = sorted(set(adjacent_rooms))
        assert len(adjacent_rooms) == 1, "each door must border exactly one enclosed room"
        room_id = adjacent_rooms[0]
        color_to_rooms[color].append(room_id)
        room_to_colors[room_id].add(color)

    for colors in room_to_colors.values():
        assert len(colors) <= 1, "rooms should not belong to multiple door pairings"

    paired_rooms: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for room_ids in color_to_rooms.values():
        unique_ids = list(dict.fromkeys(room_ids))
        assert len(room_ids) == 2 and len(unique_ids) == 2, "each door color must pair exactly two rooms"
        a, b = unique_ids
        assert a not in paired_rooms and b not in paired_rooms, "room pairings must be disjoint"
        paired_rooms.add(a)
        paired_rooms.add(b)
        pairs.append((a, b))

    room_contents = {room_id: _extract_content(g, bbox) for room_id, bbox in rooms.items()}
    out = g.copy()
    for room_id in paired_rooms:
        r0, c0, r1, c1 = rooms[room_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for room_a, room_b in pairs:
        _paint_content(out, rooms[room_a], room_contents[room_b])
        _paint_content(out, rooms[room_b], room_contents[room_a])

    return out


# --------------- generator ---------------


def _sample_pattern(limit_h: int, limit_w: int, palette: list[int]) -> list[tuple[int, int, int]]:
    size = random.randint(1, min(4, limit_h * limit_w))
    seed = (random.randrange(limit_h), random.randrange(limit_w))
    cells = {seed}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < limit_h and 0 <= nc < limit_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    color = random.choice(palette)
    return [(r, c, color) for r, c in sorted(cells)]


def _sample_door(room: dict) -> None:
    side = random.choice(("top", "bottom", "left", "right"))
    room["door_side"] = side
    room["door_offset"] = random.randrange(room["w"] if side in {"top", "bottom"} else room["h"])


def _valid_positions(
    blocked: np.ndarray, room_h: int, room_w: int
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    patch_h = room_h + 2
    patch_w = room_w + 2
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - patch_h):
        for c0 in range(1, w - patch_w):
            if blocked[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_rooms(rooms: list[dict], shape: tuple[int, int]) -> bool:
    blocked = np.zeros(shape, dtype=bool)
    order = sorted(
        range(len(rooms)),
        key=lambda idx: (rooms[idx]["h"] + 2) * (rooms[idx]["w"] + 2),
        reverse=True,
    )
    for idx in order:
        room = rooms[idx]
        positions = _valid_positions(blocked, room["h"], room["w"])
        if not positions:
            return False
        top, left = random.choice(positions)
        room["top"] = top
        room["left"] = left
        patch_h = room["h"] + 2
        patch_w = room["w"] + 2
        blocked[top - 1 : top + patch_h + 1, left - 1 : left + patch_w + 1] = True
    return True


def _draw_room(grid: np.ndarray, room: dict, content: list[tuple[int, int, int]]) -> None:
    top = room["top"]
    left = room["left"]
    h = room["h"]
    w = room["w"]
    grid[top : top + h + 2, left] = WALL
    grid[top : top + h + 2, left + w + 1] = WALL
    grid[top, left : left + w + 2] = WALL
    grid[top + h + 1, left : left + w + 2] = WALL

    if room["door_color"] is not None:
        side = room["door_side"]
        offset = room["door_offset"]
        if side == "top":
            grid[top, left + 1 + offset] = room["door_color"]
        elif side == "bottom":
            grid[top + h + 1, left + 1 + offset] = room["door_color"]
        elif side == "left":
            grid[top + 1 + offset, left] = room["door_color"]
        else:
            grid[top + 1 + offset, left + w + 1] = room["door_color"]

    for dr, dc, color in content:
        grid[top + 1 + dr, left + 1 + dc] = color


def _render_grid(shape: tuple[int, int], rooms: list[dict], contents: list[list[tuple[int, int, int]]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for room, content in zip(rooms, contents):
        _draw_room(grid, room, content)
    return grid


def _expected_output(shape: tuple[int, int], rooms: list[dict]) -> np.ndarray:
    pair_to_ids: dict[int, list[int]] = defaultdict(list)
    contents = [list(room["content"]) for room in rooms]
    for idx, room in enumerate(rooms):
        if room["pair_id"] is not None:
            pair_to_ids[room["pair_id"]].append(idx)
    for idxs in pair_to_ids.values():
        assert len(idxs) == 2
        a, b = idxs
        contents[a], contents[b] = contents[b], contents[a]
    return _render_grid(shape, rooms, contents)


def _build_rooms() -> list[dict]:
    pair_count = random.choice((1, 2, 3))
    max_doorless = min(2, 6 - 2 * pair_count)
    doorless_count = random.randint(0, max_doorless)
    door_colors = random.sample(list(DOOR_COLORS), pair_count)
    rooms: list[dict] = []

    for pair_id, door_color in enumerate(door_colors):
        for _ in range(2):
            room = {
                "h": random.randint(3, 6),
                "w": random.randint(3, 6),
                "pair_id": pair_id,
                "door_color": door_color,
                "content": [],
            }
            _sample_door(room)
            rooms.append(room)

    for _ in range(doorless_count):
        rooms.append(
            {
                "h": random.randint(3, 6),
                "w": random.randint(3, 6),
                "pair_id": None,
                "door_color": None,
                "content": [],
            }
        )

    random.shuffle(rooms)

    content_palette = [color for color in PALETTE if color not in {BG, WALL, *door_colors}]
    pair_to_ids: dict[int, list[int]] = defaultdict(list)
    for idx, room in enumerate(rooms):
        if room["pair_id"] is not None:
            pair_to_ids[room["pair_id"]].append(idx)

    for idxs in pair_to_ids.values():
        a, b = idxs
        common_h = min(rooms[a]["h"], rooms[b]["h"])
        common_w = min(rooms[a]["w"], rooms[b]["w"])
        mode = random.choice(("both", "left_only", "right_only"))
        if mode in {"both", "left_only"}:
            rooms[a]["content"] = _sample_pattern(common_h, common_w, content_palette)
        if mode in {"both", "right_only"}:
            rooms[b]["content"] = _sample_pattern(common_h, common_w, content_palette)
        if mode == "both":
            for _ in range(20):
                if rooms[a]["content"] != rooms[b]["content"]:
                    break
                rooms[b]["content"] = _sample_pattern(common_h, common_w, content_palette)
            if rooms[a]["content"] == rooms[b]["content"]:
                dr, dc, color = rooms[b]["content"][0]
                alternatives = [c for c in content_palette if c != color]
                rooms[b]["content"][0] = (dr, dc, random.choice(alternatives))

    for room in rooms:
        if room["pair_id"] is None and random.random() < 0.6:
            room["content"] = _sample_pattern(room["h"], room["w"], content_palette)

    return rooms


def _sample_shape(total_rooms: int) -> tuple[int, int]:
    base = max(18, 12 + 2 * total_rooms)
    return random.randint(base, 30), random.randint(base, 30)


def _gen_attempt() -> np.ndarray | None:
    rooms = _build_rooms()
    shape = _sample_shape(len(rooms))
    if not _place_rooms(rooms, shape):
        return None

    inp = _render_grid(shape, rooms, [list(room["content"]) for room in rooms])
    out = _expected_output(shape, rooms)

    if np.array_equal(inp, out):
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-swapping puzzle instance")

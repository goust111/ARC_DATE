import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS4)}
SIDE_TO_INWARD = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}


# --------------- solver ---------------


def _most_common_color(values) -> int:
    counts = Counter(int(v) for v in values)
    return max(counts, key=lambda color: (counts[color], -color))


def _get_bg(grid: np.ndarray) -> int:
    return _most_common_color(grid.ravel())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _border_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 3 and w >= 3
    return {
        "top": _most_common_color(grid[0, 1 : w - 1]),
        "bottom": _most_common_color(grid[h - 1, 1 : w - 1]),
        "left": _most_common_color(grid[1 : h - 1, 0]),
        "right": _most_common_color(grid[1 : h - 1, w - 1]),
    }


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _spiral_path(
    interior: set[tuple[int, int]],
    start: tuple[int, int],
    side: str,
) -> list[tuple[int, int]]:
    if start not in interior:
        return []
    path = [start]
    visited = {start}
    pos = start
    dir_idx = DIR_TO_INDEX[SIDE_TO_INWARD[side]]
    while len(path) < len(interior):
        moved = False
        for turn in range(4):
            cand_idx = (dir_idx + turn) % 4
            dr, dc = DIRS4[cand_idx]
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in interior and nxt not in visited:
                pos = nxt
                dir_idx = cand_idx
                visited.add(nxt)
                path.append(nxt)
                moved = True
                break
        if not moved:
            break
    return path


def _room_from_component(
    grid: np.ndarray,
    bg: int,
    coords: np.ndarray,
) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    if r0 <= 0 or c0 <= 0 or r1 >= grid.shape[0] - 1 or c1 >= grid.shape[1] - 1:
        return None
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    actual = {(int(r), int(c)) for r, c in coords}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if actual - expected:
        return None

    missing = expected - actual
    if len(missing) != 1 or len(actual) != len(expected) - 1:
        return None
    gap = next(iter(missing))
    side = _gap_side(gap, r0, c0, r1, c1)
    if side is None:
        return None

    interior = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == bg
    }
    if len(interior) != (r1 - r0 - 1) * (c1 - c0 - 1):
        return None

    inward = SIDE_TO_INWARD[side]
    start = (gap[0] + inward[0], gap[1] + inward[1])
    path = _spiral_path(interior, start, side)
    if len(path) != len(interior):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "interior": interior,
        "path": path,
    }


def _find_rooms(grid: np.ndarray, bg: int, stripes: dict[str, int]) -> list[dict]:
    forbidden = {bg, *stripes.values()}
    rooms = []
    for color in sorted(int(c) for c in np.unique(grid)):
        if color in forbidden:
            continue
        for comp in _components(grid == color):
            room = _room_from_component(grid, bg, comp)
            if room is not None:
                rooms.append(room)
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each room interior from the border stripe selected by its wall gap,
    then trace a clockwise spiral from the gap into the room until the path
    length matches the room's interior area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    stripes = _border_colors(g)
    rooms = _find_rooms(g, bg, stripes)

    out = g.copy()
    for room in rooms:
        fill_color = stripes[room["side"]]
        for r, c in room["path"]:
            out[r, c] = fill_color
    return out


# --------------- generator ---------------


GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
ROOM_H_RANGE = (4, 8)
ROOM_W_RANGE = (4, 8)
MAX_ROOMS = 5
ROOM_PAD = 1
ROOM_PLACE_TRIES = 80
GEN_RETRIES = 400
SIDES = ["top", "right", "bottom", "left"]


def _gap_for_bbox(r0: int, c0: int, r1: int, c1: int, side: str, anchor: int) -> tuple[int, int]:
    if side == "top":
        return (r0, c0 + 1 if anchor == 0 else c1 - 1)
    if side == "bottom":
        return (r1, c0 + 1 if anchor == 0 else c1 - 1)
    if side == "left":
        return (r0 + 1 if anchor == 0 else r1 - 1, c0)
    return (r0 + 1 if anchor == 0 else r1 - 1, c1)


def _build_room_record(r0: int, c0: int, h: int, w: int, side: str, anchor: int) -> dict | None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    gap = _gap_for_bbox(r0, c0, r1, c1, side, anchor)
    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    inward = SIDE_TO_INWARD[side]
    start = (gap[0] + inward[0], gap[1] + inward[1])
    path = _spiral_path(interior, start, side)
    if len(path) != len(interior):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "interior": interior,
        "path": path,
    }


def _draw_room(grid: np.ndarray, room: dict, wall_color: int) -> None:
    r0, c0, r1, c1 = room["bbox"]
    gap = room["gap"]
    for c in range(c0, c1 + 1):
        if (r0, c) != gap:
            grid[r0, c] = wall_color
        if (r1, c) != gap:
            grid[r1, c] = wall_color
    for r in range(r0, r1 + 1):
        if (r, c0) != gap:
            grid[r, c0] = wall_color
        if (r, c1) != gap:
            grid[r, c1] = wall_color


def _render_expected(grid: np.ndarray, rooms: list[dict], stripes: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        fill_color = stripes[room["side"]]
        for r, c in room["path"]:
            out[r, c] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    colors = rng.sample(range(10), 6)
    bg, wall_color, top_color, bottom_color, left_color, right_color = colors
    stripes = {
        "top": top_color,
        "bottom": bottom_color,
        "left": left_color,
        "right": right_color,
    }
    if len(set(stripes.values())) != 4:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[0, 1:-1] = top_color
    grid[h - 1, 1:-1] = bottom_color
    grid[1:-1, 0] = left_color
    grid[1:-1, w - 1] = right_color

    occ = np.zeros((h, w), dtype=bool)
    occ[0, :] = True
    occ[h - 1, :] = True
    occ[:, 0] = True
    occ[:, w - 1] = True

    inner_area = max(1, (h - 4) * (w - 4))
    max_rooms = max(2, min(MAX_ROOMS, inner_area // 42))
    n_rooms = rng.randint(2, max_rooms)

    sides = SIDES[:]
    rng.shuffle(sides)
    room_sides = [sides[i % 4] for i in range(n_rooms)]
    rng.shuffle(room_sides)

    rooms = []
    for side in room_sides:
        placed = False
        for _ in range(ROOM_PLACE_TRIES):
            room_h = rng.randint(*ROOM_H_RANGE)
            room_w = rng.randint(*ROOM_W_RANGE)
            if h - room_h - 2 < 2 or w - room_w - 2 < 2:
                continue

            r0 = rng.randint(2, h - room_h - 2)
            c0 = rng.randint(2, w - room_w - 2)
            r1, c1 = r0 + room_h - 1, c0 + room_w - 1
            sr0, sc0 = max(0, r0 - ROOM_PAD), max(0, c0 - ROOM_PAD)
            sr1, sc1 = min(h, r1 + ROOM_PAD + 1), min(w, c1 + ROOM_PAD + 1)
            if occ[sr0:sr1, sc0:sc1].any():
                continue

            anchor = rng.randrange(2)
            room = _build_room_record(r0, c0, room_h, room_w, side, anchor)
            if room is None:
                continue

            _draw_room(grid, room, wall_color)
            occ[sr0:sr1, sc0:sc1] = True
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(counts[c] for c in counts if c != bg):
        return None

    found_rooms = _find_rooms(grid, bg, stripes)
    if len(found_rooms) != len(rooms):
        return None

    expected = _render_expected(grid, rooms, stripes)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")

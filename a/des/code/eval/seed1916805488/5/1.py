import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLORS = list(range(10))

GRID_MIN = 14
GRID_MAX = 24
ROOM_COUNT_MIN = 1
ROOM_COUNT_MAX = 4
ROOM_H_MIN = 6
ROOM_H_MAX = 9
ROOM_W_MIN = 6
ROOM_W_MAX = 9
ROOM_GAP = 2
PLACE_TRIES = 80
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _room_components(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    rooms = []
    for coords in _components_mask(grid == wall):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            raise AssertionError("wall component too small to be a room")
        if coords.shape[0] != 2 * (hh + ww) - 4:
            raise AssertionError("wall component must be a 1-pixel rectangular ring")
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == wall):
            raise AssertionError("room border must be solid wall color")
        if np.any(patch[1:-1, 1:-1] == wall):
            raise AssertionError("room interior must not contain wall color")
        rooms.append((r0, c0, r1, c1))
    return rooms


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        try:
            rooms = _room_components(grid, color)
        except AssertionError:
            continue
        if rooms:
            candidates.append((color, rooms))
    assert len(candidates) == 1, "expected exactly one wall color"
    return int(candidates[0][0])


def _marker_specs_for_room(
    grid: np.ndarray, room: tuple[int, int, int, int], bg: int, wall: int
) -> list[tuple[int, int, int, int, int]]:
    r0, c0, r1, c1 = room
    specs = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color in (bg, wall):
                continue
            touches = []
            if r == r0 + 1:
                touches.append((-1, 0, r0, c))
            if r == r1 - 1:
                touches.append((1, 0, r1, c))
            if c == c0 + 1:
                touches.append((0, -1, r, c0))
            if c == c1 - 1:
                touches.append((0, 1, r, c1))
            assert len(touches) == 1, "marker must be adjacent to exactly one wall"
            dr, dc, hole_r, hole_c = touches[0]
            specs.append((r, c, color, dr, dc, hole_r, hole_c))
    colors = {spec[2] for spec in specs}
    assert 1 <= len(specs) <= 3, "each room must contain 1-3 markers"
    assert len(colors) == len(specs), "markers inside a room must have distinct colors"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each room marker's adjacent wall, open that wall, and extend a colored corridor to the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    wall = _find_wall_color(g, bg)

    rooms = _room_components(g, wall)
    assert rooms, "expected at least one room"
    rooms.sort(key=lambda box: (box[1], box[0]))

    out = g.copy()
    for room in rooms:
        specs = _marker_specs_for_room(g, room, bg, wall)
        specs.sort(key=lambda item: (item[0], item[1]))
        for _, _, color, dr, dc, hole_r, hole_c in specs:
            out[hole_r, hole_c] = color
            cr, cc = hole_r + dr, hole_c + dc
            while 0 <= cr < g.shape[0] and 0 <= cc < g.shape[1]:
                if out[cr, cc] == bg:
                    out[cr, cc] = color
                cr += dr
                cc += dc

    return out


def _clear_bbox(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - ROOM_GAP)
    cc0 = max(0, c0 - ROOM_GAP)
    rr1 = min(mask.shape[0] - 1, r1 + ROOM_GAP)
    cc1 = min(mask.shape[1] - 1, c1 + ROOM_GAP)
    return not np.any(mask[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _paint_room(grid: np.ndarray, wall: int, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall


def _sample_markers(
    rng: random.Random,
    room: tuple[int, int, int, int],
    colors: list[int],
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = room
    candidates = {
        "N": [(r0 + 1, c) for c in range(c0 + 2, c1 - 1)],
        "S": [(r1 - 1, c) for c in range(c0 + 2, c1 - 1)],
        "W": [(r, c0 + 1) for r in range(r0 + 2, r1 - 1)],
        "E": [(r, c1 - 1) for r in range(r0 + 2, r1 - 1)],
    }
    n_markers = rng.randint(1, 3)
    marker_colors = rng.sample(colors, n_markers)
    dirs = [rng.choice(["N", "S", "W", "E"]) for _ in range(n_markers)]
    used = set()
    markers = []
    for color, side in zip(marker_colors, dirs):
        spots = [pos for pos in candidates[side] if pos not in used]
        if not spots:
            raise ValueError("insufficient marker slots")
        pos = rng.choice(spots)
        used.add(pos)
        markers.append((pos[0], pos[1], color))
    return markers


def _marker_ray(
    marker: tuple[int, int, int], room: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c, _ = marker
    r0, c0, r1, c1 = room
    if r == r0 + 1:
        return (r0, c), (-1, 0)
    if r == r1 - 1:
        return (r1, c), (1, 0)
    if c == c0 + 1:
        return (r, c0), (0, -1)
    if c == c1 - 1:
        return (r, c1), (0, 1)
    raise AssertionError("marker is not adjacent to a wall")


def _valid_layout(
    grid: np.ndarray,
    bg: int,
    rooms: list[tuple[int, int, int, int]],
    markers_by_room: list[list[tuple[int, int, int]]],
) -> bool:
    h, w = grid.shape
    occupied = grid != bg
    for room, markers in zip(rooms, markers_by_room):
        for marker in markers:
            (hole_r, hole_c), (dr, dc) = _marker_ray(marker, room)
            cr, cc = hole_r + dr, hole_c + dc
            while 0 <= cr < h and 0 <= cc < w:
                if occupied[cr, cc]:
                    return False
                cr += dr
                cc += dc
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice(COLORS)
    remaining = [c for c in COLORS if c != bg]
    wall = rng.choice(remaining)
    marker_palette = [c for c in remaining if c != wall]
    if len(marker_palette) < 3:
        return None

    grid = np.full((h, w), bg, dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    target_rooms = rng.randint(ROOM_COUNT_MIN, ROOM_COUNT_MAX)
    rooms = []
    for _ in range(target_rooms):
        placed = False
        for _ in range(PLACE_TRIES):
            rh = rng.randint(ROOM_H_MIN, min(ROOM_H_MAX, h - 2))
            rw = rng.randint(ROOM_W_MIN, min(ROOM_W_MAX, w - 2))
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            room = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if not _clear_bbox(mask, *room):
                continue
            rooms.append(room)
            mask[r0 : r0 + rh, c0 : c0 + rw] = True
            placed = True
            break
        if not placed:
            break
    if not rooms:
        return None

    markers_by_room: list[list[tuple[int, int, int]]] = []
    for room in rooms:
        _paint_room(grid, wall, room)
        try:
            markers = _sample_markers(rng, room, marker_palette)
        except ValueError:
            return None
        for r, c, color in markers:
            grid[r, c] = color
        markers_by_room.append(markers)

    if not _valid_layout(grid, bg, rooms, markers_by_room):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

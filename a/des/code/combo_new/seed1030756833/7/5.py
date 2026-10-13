from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

DIRECTIONS = ("up", "down", "left", "right")
STAMP_COLORS = (1, 4, 5, 6, 7, 8, 9)
GRID_SIZE_RANGE = (18, 30)
ROOM_COUNT_RANGE = (1, 4)
FRAME_H_RANGE = (6, 11)
FRAME_W_RANGE = (6, 11)
MAX_STAMP_SIDE = 5
MAX_STAMP_AREA = 9
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or not bool(mask[r0, c0]):
                continue
            seen[r0, c0] = True
            q: deque[tuple[int, int]] = deque([(r0, c0)])
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(pts)
    return comps


def _room_from_wall_component(g: np.ndarray, pts: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(pts)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    pset = set(pts)
    if pset != border:
        return None

    markers = [(r, c) for r, c in pts if int(g[r, c]) == GREEN]
    if len(markers) != 1:
        return None
    mr, mc = markers[0]

    if mr == r0 and c0 < mc < c1:
        direction = "up"
    elif mr == r1 and c0 < mc < c1:
        direction = "down"
    elif mc == c0 and r0 < mr < r1:
        direction = "left"
    elif mc == c1 and r0 < mr < r1:
        direction = "right"
    else:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "marker": (int(mr), int(mc)),
        "direction": direction,
    }


def _find_rooms(g: np.ndarray) -> list[dict]:
    wall_mask = (g == RED) | (g == GREEN)
    rooms: list[dict] = []
    for pts in _components(wall_mask):
        room = _room_from_wall_component(g, pts)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _target_origin(
    room_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    patch_shape: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    mr, mc = marker
    ph, pw = patch_shape

    if direction == "up":
        return r0 - ph, mc - pw // 2
    if direction == "down":
        return r1 + 1, mc - pw // 2
    if direction == "left":
        return mr - ph // 2, c0 - pw
    if direction == "right":
        return mr - ph // 2, c1 + 1
    raise AssertionError(direction)


def _extract_stamp(g: np.ndarray, room: dict) -> dict:
    r0, c0, r1, c1 = room["bbox"]
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    stamp_mask = interior != BLACK
    assert not np.any(interior == RED), "room interiors must not contain red wall pixels"
    assert np.any(stamp_mask), "each room must contain a stamp"

    comps = _components(stamp_mask)
    assert len(comps) == 1, "each room must contain one connected stamp"
    sr0, sc0, sr1, sc1 = _bbox(comps[0])

    mask = stamp_mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    return {
        "clear_slice": (r0 + 1, c0 + 1, stamp_mask),
        "mask": mask,
        "patch": patch,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each room's internal sprite outside the room in the direction marked by its green wall pixel.

    Input structure:
    - The background is black.
    - Red one-pixel-thick rectangular walls enclose several rooms, and exactly one non-corner wall pixel of each room is green instead of red.
    - Inside every room is one small connected non-black, non-red stamp sprite that stays at least one cell away from the wall.
    - The exterior around each marked wall has enough empty space for the moved stamp.

    Rule:
    - For each room, read the wall containing the green marker: top means up, bottom means down, left means left, and right means right.
    - Clear the stamp from the room interior while preserving the red wall and green marker.
    - Copy the stamp outside that wall so its bbox edge nearest the wall is one cell beyond the wall.
    - Align the stamp center on the perpendicular axis with the green marker's row or column.

    Key insight:
    The green pixel is an exit marker: it gives both the outward direction and the alignment axis for moving the enclosed stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    rooms = _find_rooms(g)
    assert rooms, "expected at least one marked red room"

    moves: list[dict] = []
    for room in rooms:
        stamp = _extract_stamp(g, room)
        patch = stamp["patch"]
        mask = stamp["mask"]
        tr, tc = _target_origin(room["bbox"], room["marker"], patch.shape, room["direction"])
        assert 0 <= tr and 0 <= tc and tr + patch.shape[0] <= h and tc + patch.shape[1] <= w
        stamp["target"] = (int(tr), int(tc))
        moves.append(stamp)

    out = g.copy()
    for move in moves:
        r0, c0, mask = move["clear_slice"]
        view = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        view[mask] = BLACK

    for move in moves:
        tr, tc = move["target"]
        patch = move["patch"]
        mask = move["mask"]
        target = out[tr : tr + patch.shape[0], tc : tc + patch.shape[1]]
        target[mask] = patch[mask]

    return out


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cell = random.choice(frontier)
        cells.add(cell)
        mask[cell] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_stamp_mask(max_h: int, max_w: int, direction: str) -> np.ndarray:
    max_h = max(1, min(MAX_STAMP_SIDE, int(max_h)))
    max_w = max(1, min(MAX_STAMP_SIDE, int(max_w)))

    for _ in range(80):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        area = random.randint(1, min(MAX_STAMP_AREA, h * w))
        mask = _crop_mask(_grow_mask(h, w, area))
        ph, pw = mask.shape
        if direction in ("up", "down") and pw % 2 == 0:
            continue
        if direction in ("left", "right") and ph % 2 == 0:
            continue
        return mask

    return np.ones((1, 1), dtype=bool)


def _target_bbox(
    room_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    patch_shape: tuple[int, int],
    direction: str,
) -> tuple[int, int, int, int]:
    tr, tc = _target_origin(room_bbox, marker, patch_shape, direction)
    ph, pw = patch_shape
    return tr, tc, tr + ph - 1, tc + pw - 1


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _clear_against_reserved(
    reserved: np.ndarray, cells: list[tuple[int, int]], pad: int = 1
) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if bool(reserved[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved[r, c] = True


def _draw_room(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    stamp_mask: np.ndarray,
    stamp_color: int,
) -> None:
    r0, c0, r1, c1 = room_bbox
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    mr, mc = marker
    grid[mr, mc] = GREEN

    ph, pw = stamp_mask.shape
    sr = random.randint(r0 + 2, r1 - ph - 1)
    sc = random.randint(c0 + 2, c1 - pw - 1)
    grid[sr : sr + ph, sc : sc + pw][stamp_mask] = int(stamp_color)


def _marker_for_direction(
    room_bbox: tuple[int, int, int, int], direction: str, patch_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = room_bbox
    ph, pw = patch_shape

    if direction in ("up", "down"):
        lo = c0 + 1 + pw // 2
        hi = c1 - 1 - pw // 2
        col = random.randint(lo, hi)
        row = r0 if direction == "up" else r1
        return row, col

    lo = r0 + 1 + ph // 2
    hi = r1 - 1 - ph // 2
    row = random.randint(lo, hi)
    col = c0 if direction == "left" else c1
    return row, col


def _try_place_room(grid: np.ndarray, reserved: np.ndarray, direction: str) -> bool:
    h, w = grid.shape

    for _ in range(PLACEMENT_ATTEMPTS):
        fh = random.randint(*FRAME_H_RANGE)
        fw = random.randint(*FRAME_W_RANGE)
        if fh >= h or fw >= w:
            continue

        stamp_mask = _make_stamp_mask(fh - 4, fw - 4, direction)
        ph, pw = stamp_mask.shape
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        room_bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        marker = _marker_for_direction(room_bbox, direction, stamp_mask.shape)
        tb = _target_bbox(room_bbox, marker, stamp_mask.shape, direction)
        tr0, tc0, tr1, tc1 = tb
        if tr0 < 0 or tc0 < 0 or tr1 >= h or tc1 >= w:
            continue

        candidate = _bbox_cells(room_bbox) + _bbox_cells(tb)
        if not _clear_against_reserved(reserved, candidate, pad=1):
            continue

        _draw_room(grid, room_bbox, marker, stamp_mask, random.choice(STAMP_COLORS))
        _mark_reserved(reserved, candidate)
        return True

    return False


def _direction_sequence(count: int) -> list[str]:
    dirs = list(DIRECTIONS)
    random.shuffle(dirs)
    return dirs[:count]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    room_count = random.randint(*ROOM_COUNT_RANGE)
    for direction in _direction_sequence(room_count):
        if not _try_place_room(grid, reserved, direction):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-room input")

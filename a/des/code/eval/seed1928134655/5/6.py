from __future__ import annotations

from collections import deque
from typing import NamedTuple
import random

import numpy as np


BG = 0
WALL = 1
ARROW = 2
PALETTE = (3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")


class Room(NamedTuple):
    frame: tuple[int, int, int, int]
    interior: tuple[int, int, int, int]


class GenRoom(NamedTuple):
    top: int
    left: int
    height: int
    width: int
    side: str

    @property
    def frame(self) -> tuple[int, int, int, int]:
        return (self.top, self.left, self.top + self.height - 1, self.left + self.width - 1)

    @property
    def interior(self) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.frame
        return (r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    @property
    def expanded(self) -> tuple[int, int, int, int]:
        r0, c0, r1, c1 = self.frame
        return (r0 - 2, c0 - 2, r1 + 2, c1 + 2)


SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
)


GENERATE_ATTEMPTS = 300
ROOM_LAYOUT_ATTEMPTS = 120
ROOM_CONTENT_ATTEMPTS = 160


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) != color:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_mask_ok(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _find_rooms(g: np.ndarray) -> list[Room]:
    rooms: list[Room] = []
    for comp in _components_of_color(g, WALL):
        assert _frame_mask_ok(comp), "blue structures must be closed rectangular room frames"
        r0, c0, r1, c1 = _bbox(comp)
        rooms.append(Room(frame=(r0, c0, r1, c1), interior=(r0 + 1, c0 + 1, r1 - 1, c1 - 1)))
    assert rooms, "expected at least one room"
    return rooms


def _arrow_cells(frame: tuple[int, int, int, int], side: str) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = frame
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return ((r0 - 1, mid_c), (r0 - 2, mid_c - 1), (r0 - 2, mid_c + 1))
    if side == "bottom":
        return ((r1 + 1, mid_c), (r1 + 2, mid_c - 1), (r1 + 2, mid_c + 1))
    if side == "left":
        return ((mid_r, c0 - 1), (mid_r - 1, c0 - 2), (mid_r + 1, c0 - 2))
    assert side == "right"
    return ((mid_r, c1 + 1), (mid_r - 1, c1 + 2), (mid_r + 1, c1 + 2))


def _detect_arrow(g: np.ndarray, room: Room) -> tuple[str, tuple[tuple[int, int], ...]]:
    h, w = g.shape
    matches: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for side in SIDES:
        cells = _arrow_cells(room.frame, side)
        if all(0 <= r < h and 0 <= c < w and int(g[r, c]) == ARROW for r, c in cells):
            matches.append((side, cells))
    assert len(matches) == 1, "each room must have exactly one exterior red arrow"
    return matches[0]


def _corner_cells(interior: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = interior
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _find_marker(g: np.ndarray, interior: tuple[int, int, int, int]) -> tuple[tuple[int, int], int]:
    candidates = [(r, c) for r, c in _corner_cells(interior) if int(g[r, c]) not in {BG, WALL, ARROW}]
    assert len(candidates) == 1, "expected exactly one corner color marker per room"
    marker = candidates[0]
    color = int(g[marker])
    for nr, nc in _neighbors4(*marker):
        r0, c0, r1, c1 = interior
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            assert int(g[nr, nc]) != color, "corner marker must be a singleton"
    return marker, color


def _room_objects(
    g: np.ndarray, interior: tuple[int, int, int, int], marker: tuple[int, int]
) -> list[tuple[int, list[tuple[int, int]]]]:
    r0, c0, r1, c1 = interior
    visited: set[tuple[int, int]] = set()
    objects: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) == marker:
                continue
            color = int(g[r, c])
            if color in {BG, WALL, ARROW} or (r, c) in visited:
                continue
            visited.add((r, c))
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in visited and int(g[nr, nc]) == color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            assert 2 <= len(coords) <= 4, "furniture objects must be 2-4 connected pixels"
            objects.append((color, coords))
    assert objects, "each room must contain at least one furniture object"
    return objects


def _push_info(
    coords: list[tuple[int, int]], side: str, interior: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = interior
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if side == "top":
        return 1, 0, r1 - max(rows)
    if side == "bottom":
        return -1, 0, min(rows) - r0
    if side == "left":
        return 0, 1, c1 - max(cols)
    assert side == "right"
    return 0, -1, min(cols) - c0


def _plan_room(
    g: np.ndarray, interior: tuple[int, int, int, int], marker: tuple[int, int], side: str
) -> tuple[dict[tuple[int, int], int], bool]:
    planned: dict[tuple[int, int], int] = {}
    any_move = False
    _marker, target_color = _find_marker(g, interior)
    assert _marker == marker
    for _color, coords in _room_objects(g, interior, marker):
        dr, dc, shift = _push_info(coords, side, interior)
        any_move = any_move or shift > 0
        for r, c in coords:
            nr = r + dr * shift
            nc = c + dc * shift
            prior = planned.get((nr, nc))
            assert prior is None, "pushed furniture objects must not overlap"
            planned[(nr, nc)] = target_color
    return planned, any_move


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push every furniture object inside each blue room toward the wall opposite its exterior red arrow,
    then recolor those objects using the room's corner floor marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    rooms = _find_rooms(g)
    all_arrow_pixels = {(int(r), int(c)) for r, c in np.argwhere(g == ARROW)}
    matched_arrow_pixels: set[tuple[int, int]] = set()
    out = g.copy()

    for room in rooms:
        side, arrow_cells = _detect_arrow(g, room)
        matched_arrow_pixels.update(arrow_cells)
        marker, _target_color = _find_marker(g, room.interior)
        planned, _ = _plan_room(g, room.interior, marker, side)

        ir0, ic0, ir1, ic1 = room.interior
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
        for (r, c), color in planned.items():
            out[r, c] = color
        for r, c in arrow_cells:
            out[r, c] = BG

    assert matched_arrow_pixels == all_arrow_pixels, "all red pixels must belong to room arrows"
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_room_count() -> int:
    return random.choices((1, 2, 3, 4), weights=(2, 3, 2, 1), k=1)[0]


def _sample_grid_shape(room_count: int) -> tuple[int, int]:
    if room_count == 1:
        low = 18
    elif room_count == 2:
        low = 20
    elif room_count == 3:
        low = 23
    else:
        low = 26
    return random.randint(low, 30), random.randint(low, 30)


def _sample_room_size(room_count: int) -> tuple[int, int]:
    sizes = (7, 9) if room_count >= 3 else (7, 9, 11)
    return random.choice(sizes), random.choice(sizes)


def _place_rooms(h: int, w: int, room_count: int) -> list[GenRoom] | None:
    rooms: list[GenRoom] = []
    for _ in range(room_count):
        placed = False
        for _ in range(ROOM_LAYOUT_ATTEMPTS):
            rh, rw = _sample_room_size(room_count)
            if h < rh + 4 or w < rw + 4:
                continue
            top = random.randint(2, h - rh - 2)
            left = random.randint(2, w - rw - 2)
            side = random.choice(SIDES)
            room = GenRoom(top, left, rh, rw, side)
            if any(_boxes_overlap(room.expanded, other.expanded) for other in rooms):
                continue
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None
    return rooms


def _local_corners(ih: int, iw: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, iw - 1), (ih - 1, 0), (ih - 1, iw - 1))


def _target_touching(cells: list[tuple[int, int]], side: str, ih: int, iw: int) -> bool:
    if side == "top":
        return any(r == ih - 1 for r, _ in cells)
    if side == "bottom":
        return any(r == 0 for r, _ in cells)
    if side == "left":
        return any(c == iw - 1 for _, c in cells)
    return any(c == 0 for _, c in cells)


def _shape_cells(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(shape):
        cells.append((top + int(r), left + int(c)))
    return cells


def _too_close(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for orow, ocol in occupied:
            if max(abs(r - orow), abs(c - ocol)) <= 1:
                return True
    return False


def _sample_object_count(ih: int, iw: int) -> int:
    if ih <= 5 or iw <= 5:
        return random.choices((1, 2), weights=(3, 1), k=1)[0]
    return random.choices((1, 2, 3), weights=(2, 2, 1), k=1)[0]


def _build_room_patch(ih: int, iw: int, side: str) -> np.ndarray | None:
    corners = _local_corners(ih, iw)
    forbidden_corners = set(corners)
    for _ in range(ROOM_CONTENT_ATTEMPTS):
        patch = np.full((ih, iw), BG, dtype=int)
        marker = random.choice(corners)
        marker_color = random.choice(PALETTE)
        patch[marker] = marker_color
        occupied = {marker}

        obj_count = _sample_object_count(ih, iw)
        obj_palette = random.sample([c for c in PALETTE if c != marker_color], obj_count)
        success = True
        for obj_color in obj_palette:
            candidates: list[list[tuple[int, int]]] = []
            for shape in SHAPES:
                sh, sw = shape.shape
                if sh > ih or sw > iw:
                    continue
                for top in range(ih - sh + 1):
                    for left in range(iw - sw + 1):
                        cells = _shape_cells(shape, top, left)
                        if any(cell in forbidden_corners for cell in cells):
                            continue
                        if _target_touching(cells, side, ih, iw):
                            continue
                        if _too_close(cells, occupied):
                            continue
                        candidates.append(cells)
            if not candidates:
                success = False
                break
            chosen = random.choice(candidates)
            for r, c in chosen:
                patch[r, c] = obj_color
            occupied.update(chosen)

        if not success:
            continue

        marker_global = marker
        try:
            planned, any_move = _plan_room(
                patch,
                (0, 0, ih - 1, iw - 1),
                marker_global,
                side,
            )
        except AssertionError:
            continue
        if not any_move or not planned:
            continue
        return patch
    return None


def _draw_room(grid: np.ndarray, room: GenRoom, patch: np.ndarray) -> None:
    r0, c0, r1, c1 = room.frame
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL

    ir0, ic0, ir1, ic1 = room.interior
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = patch
    for r, c in _arrow_cells(room.frame, room.side):
        grid[r, c] = ARROW


def _gen_attempt() -> np.ndarray | None:
    room_count = _sample_room_count()
    h, w = _sample_grid_shape(room_count)
    rooms = _place_rooms(h, w, room_count)
    if rooms is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for room in rooms:
        ir0, ic0, ir1, ic1 = room.interior
        patch = _build_room_patch(ir1 - ir0 + 1, ic1 - ic0 + 1, room.side)
        if patch is None:
            return None
        _draw_room(grid, room, patch)

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-pushing ARC input")

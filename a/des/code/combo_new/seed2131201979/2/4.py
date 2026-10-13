from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
ARROW_SHAPES = {
    "up": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)}),
    "left": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)}),
}
ARROW_DIMS = {
    direction: (
        max(r for r, _ in cells) + 1,
        max(c for _, c in cells) + 1,
    )
    for direction, cells in ARROW_SHAPES.items()
}
MAX_GEN_ATTEMPTS = 400
MAX_PAIR_ATTEMPTS = 300


@dataclass
class Room:
    r0: int
    c0: int
    h: int
    w: int
    direction: str
    arrow_r: int
    arrow_c: int
    color: int = 0

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_components(grid: np.ndarray, grey: int) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == grey):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        expected = 2 * (h + w) - 4
        if len(pts) != expected:
            continue
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in pts):
            continue
        rooms.append((r0, c0, r1, c1))
    rooms.sort()
    return rooms


def _infer_grey(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    if not colors:
        return GREY
    if GREY in colors:
        return GREY
    return max(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _detect_arrow(
    grid: np.ndarray, room_bbox: tuple[int, int, int, int], bg: int, grey: int
) -> tuple[str, int, list[tuple[int, int]], int, int]:
    r0, c0, r1, c1 = room_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != bg) & (interior != grey)
    pts_local = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts_local:
        raise ValueError("room has no arrow")
    colors = Counter(int(interior[r, c]) for r, c in pts_local)
    color = colors.most_common(1)[0][0]
    arrow_pts = [(r, c) for r, c in pts_local if int(interior[r, c]) == color]
    ar0, ac0, ar1, ac1 = _bbox(arrow_pts)
    norm = frozenset((r - ar0, c - ac0) for r, c in arrow_pts)
    for direction, shape in ARROW_SHAPES.items():
        if norm == shape:
            abs_points = [(r0 + 1 + r, c0 + 1 + c) for r, c in arrow_pts]
            if direction in ("up", "down"):
                axis_row = -1
                axis_col = c0 + 1 + ac0 + 1
            else:
                axis_row = r0 + 1 + ar0 + 1
                axis_col = -1
            return direction, color, abs_points, axis_row, axis_col
    raise ValueError("unrecognized arrow geometry")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each arrow-marked room on the indicated side and send a same-colored path outward.

    Input structure:
    - The background is black.
    - Grey pixels form several non-overlapping hollow rectangular rooms with wall thickness 1.
    - Inside each room there is exactly one colored arrow made from a 1x3 or 3x1 shaft plus a 3-pixel V-shaped head.

    Rule:
    - Detect every room and read the direction of its interior arrow.
    - Remove the non-corner part of the wall on the side pointed to by that arrow.
    - Erase the arrow itself, then draw a 1-pixel-wide straight path in the arrow color starting from the opening and extending outward until the grid edge or another grey wall blocks it.

    Key insight:
    - The arrow's geometry gives the exit direction, while its color gives the path color; the rest of the transformation is a room-by-room projection from that exit to the outside.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    grey = _infer_grey(g, bg)
    rooms = _frame_components(g, grey)
    if not rooms:
        return g.copy()

    room_info = []
    for room_bbox in rooms:
        direction, color, arrow_pts, axis_row, axis_col = _detect_arrow(g, room_bbox, bg, grey)
        room_info.append((room_bbox, direction, color, arrow_pts, axis_row, axis_col))

    out = g.copy()
    for (r0, c0, r1, c1), direction, _, arrow_pts, _, _ in room_info:
        for r, c in arrow_pts:
            out[r, c] = bg
        if direction == "up":
            out[r0, c0 + 1 : c1] = bg
        elif direction == "down":
            out[r1, c0 + 1 : c1] = bg
        elif direction == "left":
            out[r0 + 1 : r1, c0] = bg
        else:
            out[r0 + 1 : r1, c1] = bg

    h, w = out.shape
    for (r0, c0, r1, c1), direction, color, _, axis_row, axis_col in room_info:
        if direction == "up":
            r, c = r0, axis_col
        elif direction == "down":
            r, c = r1, axis_col
        elif direction == "left":
            r, c = axis_row, c0
        else:
            r, c = axis_row, c1
        dr, dc = DIRS[direction]
        while 0 <= r < h and 0 <= c < w and out[r, c] != grey:
            out[r, c] = color
            r += dr
            c += dc

    return out


def _room_too_close(candidate: tuple[int, int, int, int], rooms: list[Room], gap: int = 2) -> bool:
    r0, c0, r1, c1 = candidate
    for room in rooms:
        rr0, cc0, rr1, cc1 = room.bbox
        separated = (
            r1 + gap < rr0
            or rr1 + gap < r0
            or c1 + gap < cc0
            or cc1 + gap < c0
        )
        if not separated:
            return True
    return False


def _bbox_contains_point(bbox: tuple[int, int, int, int], pt: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = pt
    return r0 <= r <= r1 and c0 <= c <= c1


def _room_hits_reserved(candidate: tuple[int, int, int, int], reserved: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in reserved)


def _source_path_cells(room: Room) -> list[tuple[int, int]]:
    if room.direction == "up":
        axis = room.arrow_c + 1
        return [(r, axis) for r in range(room.r0, -1, -1)]
    if room.direction == "down":
        axis = room.arrow_c + 1
        return [(r, axis) for r in range(room.r1, 1000)]
    if room.direction == "left":
        axis = room.arrow_r + 1
        return [(axis, c) for c in range(room.c0, -1, -1)]
    axis = room.arrow_r + 1
    return [(axis, c) for c in range(room.c1, 1000)]


def _path_to_boundary(room: Room, height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    if room.direction == "up":
        axis = room.arrow_c + 1
        for r in range(room.r0, -1, -1):
            cells.append((r, axis))
    elif room.direction == "down":
        axis = room.arrow_c + 1
        for r in range(room.r1, height):
            cells.append((r, axis))
    elif room.direction == "left":
        axis = room.arrow_r + 1
        for c in range(room.c0, -1, -1):
            cells.append((axis, c))
    else:
        axis = room.arrow_r + 1
        for c in range(room.c1, width):
            cells.append((axis, c))
    return cells


def _segment_clear(
    cells: list[tuple[int, int]],
    rooms: list[Room],
    reserved: set[tuple[int, int]],
    ignore: tuple[tuple[int, int, int, int], ...] = (),
) -> bool:
    ignored = set(ignore)
    for cell in cells:
        if cell in reserved:
            return False
        for room in rooms:
            if room.bbox in ignored:
                continue
            if _bbox_contains_point(room.bbox, cell):
                return False
    return True


def _boundary_arrow_candidates(
    bbox: tuple[int, int, int, int],
    rooms: list[Room],
    reserved: set[tuple[int, int]],
    height: int,
    width: int,
    forbidden: set[str],
) -> list[tuple[str, int, int, list[tuple[int, int]]]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    candidates: list[tuple[str, int, int, list[tuple[int, int]]]] = []
    for direction in ("up", "down", "left", "right"):
        if direction in forbidden:
            continue
        ah, aw = ARROW_DIMS[direction]
        top_max = h - ah - 1
        left_max = w - aw - 1
        if top_max < 1 or left_max < 1:
            continue
        for top in range(1, top_max + 1):
            for left in range(1, left_max + 1):
                room = Room(r0, c0, h, w, direction, r0 + top, c0 + left)
                cells = _path_to_boundary(room, height, width)
                if len(cells) < 3:
                    continue
                if not _segment_clear(cells[1:], rooms, reserved, ignore=(bbox,)):
                    continue
                candidates.append((direction, r0 + top, c0 + left, cells))
    return candidates


def _assign_boundary_arrow(
    bbox: tuple[int, int, int, int],
    rooms: list[Room],
    reserved: set[tuple[int, int]],
    height: int,
    width: int,
    rng: np.random.Generator,
    forbidden: set[str] | None = None,
) -> tuple[Room, list[tuple[int, int]]] | None:
    candidates = _boundary_arrow_candidates(
        bbox,
        rooms,
        reserved,
        height,
        width,
        forbidden or set(),
    )
    if not candidates:
        return None
    direction, arrow_r, arrow_c, cells = candidates[int(rng.integers(len(candidates)))]
    r0, c0, r1, c1 = bbox
    room = Room(r0, c0, r1 - r0 + 1, c1 - c0 + 1, direction, arrow_r, arrow_c)
    return room, cells


def _place_obstacle_pair(
    rooms: list[Room],
    reserved: set[tuple[int, int]],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[list[Room], set[tuple[int, int]]] | None:
    for _ in range(MAX_PAIR_ATTEMPTS):
        direction = str(rng.choice(np.array(["up", "down", "left", "right"], dtype=object)))
        hs = int(rng.integers(6, 10))
        ws = int(rng.integers(6, 10))
        ht = int(rng.integers(6, 10))
        wt = int(rng.integers(6, 10))
        gap = int(rng.integers(2, 7))

        if direction in ("left", "right"):
            lane = int(rng.integers(2, height - 2))
            src_rows = [
                r0
                for r0 in range(0, height - hs + 1)
                if r0 + 2 <= lane <= r0 + hs - 3
            ]
            tgt_rows = [
                r0
                for r0 in range(0, height - ht + 1)
                if r0 + 1 <= lane <= r0 + ht - 2
            ]
            if not src_rows or not tgt_rows:
                continue
            r0s = int(rng.choice(np.array(src_rows, dtype=int)))
            r0t = int(rng.choice(np.array(tgt_rows, dtype=int)))
            total_w = ws + gap + wt
            if total_w > width:
                continue
            base = int(rng.integers(0, width - total_w + 1))
            if direction == "right":
                c0s, c0t = base, base + ws + gap
            else:
                c0t, c0s = base, base + wt + gap
            top_local = lane - r0s - 1
            left_choices = list(range(1, ws - 4))
            if not left_choices:
                continue
            arrow_r = r0s + top_local
            arrow_c = c0s + int(rng.choice(np.array(left_choices, dtype=int)))
            source = Room(r0s, c0s, hs, ws, direction, arrow_r, arrow_c)
            target_bbox = (r0t, c0t, r0t + ht - 1, c0t + wt - 1)
            if direction == "right":
                source_path = [(lane, c) for c in range(source.c1, c0t)]
            else:
                source_path = [(lane, c) for c in range(source.c0, c0t + wt - 1, -1)]
        else:
            lane = int(rng.integers(2, width - 2))
            src_cols = [
                c0
                for c0 in range(0, width - ws + 1)
                if c0 + 2 <= lane <= c0 + ws - 3
            ]
            tgt_cols = [
                c0
                for c0 in range(0, width - wt + 1)
                if c0 + 1 <= lane <= c0 + wt - 2
            ]
            if not src_cols or not tgt_cols:
                continue
            c0s = int(rng.choice(np.array(src_cols, dtype=int)))
            c0t = int(rng.choice(np.array(tgt_cols, dtype=int)))
            total_h = hs + gap + ht
            if total_h > height:
                continue
            base = int(rng.integers(0, height - total_h + 1))
            if direction == "down":
                r0s, r0t = base, base + hs + gap
            else:
                r0t, r0s = base, base + ht + gap
            left_local = lane - c0s - 1
            top_choices = list(range(1, hs - 4))
            if not top_choices:
                continue
            arrow_r = r0s + int(rng.choice(np.array(top_choices, dtype=int)))
            arrow_c = c0s + left_local
            source = Room(r0s, c0s, hs, ws, direction, arrow_r, arrow_c)
            target_bbox = (r0t, c0t, r0t + ht - 1, c0t + wt - 1)
            if direction == "down":
                source_path = [(r, lane) for r in range(source.r1, r0t)]
            else:
                source_path = [(r, lane) for r in range(source.r0, r0t + ht - 1, -1)]

        if len(source_path) < 3:
            continue
        if _room_too_close(source.bbox, rooms) or _room_hits_reserved(source.bbox, reserved):
            continue
        if _room_too_close(target_bbox, rooms + [source]) or _room_hits_reserved(target_bbox, reserved):
            continue
        if not _segment_clear(source_path[1:], rooms, reserved):
            continue

        temp_rooms = rooms + [source]
        boundary_assignment = _assign_boundary_arrow(
            target_bbox,
            temp_rooms,
            reserved | set(source_path),
            height,
            width,
            rng,
            forbidden={OPPOSITE[direction]},
        )
        if boundary_assignment is None:
            continue
        target, target_path = boundary_assignment
        return [source, target], set(source_path) | set(target_path)

    return None


def _draw_room(grid: np.ndarray, room: Room) -> None:
    grid[room.r0, room.c0 : room.c1 + 1] = GREY
    grid[room.r1, room.c0 : room.c1 + 1] = GREY
    grid[room.r0 : room.r1 + 1, room.c0] = GREY
    grid[room.r0 : room.r1 + 1, room.c1] = GREY
    for dr, dc in ARROW_SHAPES[room.direction]:
        grid[room.arrow_r + dr, room.arrow_c + dc] = room.color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(np.array([1, 2], dtype=int)))
    min_size = 18 if pair_count == 1 else 24
    height = int(rng.integers(min_size, 31))
    width = int(rng.integers(min_size, 31))
    rooms: list[Room] = []
    reserved: set[tuple[int, int]] = set()

    for _ in range(pair_count):
        placed = _place_obstacle_pair(rooms, reserved, height, width, rng)
        if placed is None:
            return None
        new_rooms, new_reserved = placed
        rooms.extend(new_rooms)
        reserved |= new_reserved

    if len(rooms) > len(PALETTE):
        return None
    colors = rng.choice(PALETTE, size=len(rooms), replace=False)
    for room, color in zip(rooms, colors):
        room.color = int(color)

    grid = np.full((height, width), BG, dtype=int)
    for room in rooms:
        _draw_room(grid, room)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-arrow grid")

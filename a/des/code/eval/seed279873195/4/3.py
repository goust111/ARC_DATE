import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
FURNITURE_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
ROOM_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
ROOM_H_RANGE = (7, 11)
ROOM_W_RANGE = (7, 12)
WINDOW_WIDTHS = (1, 2)
MAX_GENERATION_ATTEMPTS = 400
ROOM_PLACEMENT_ATTEMPTS = 120
FURNITURE_PLACEMENT_ATTEMPTS = 80


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _room_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    wall_mask = (grid == GREY) | (grid == YELLOW)
    rooms: list[dict] = []

    for coords in _connected_components(wall_mask):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        height, width = sub.shape
        if height < 3 or width < 3:
            continue

        border = _room_border_mask(height, width)
        border_colors = (sub == GREY) | (sub == YELLOW)
        if not np.array_equal(border_colors, border):
            continue

        windows = np.argwhere(sub == YELLOW)
        if not (1 <= len(windows) <= 2):
            continue

        side_names = []
        if np.any(windows[:, 0] == 0):
            side_names.append("top")
        if np.any(windows[:, 0] == height - 1):
            side_names.append("bottom")
        if np.any(windows[:, 1] == 0):
            side_names.append("left")
        if np.any(windows[:, 1] == width - 1):
            side_names.append("right")
        if len(side_names) != 1:
            continue

        side = side_names[0]
        if side in ("top", "bottom"):
            line = sorted(int(c) for c in windows[:, 1])
            if line[0] == 0 or line[-1] == width - 1:
                continue
            if line[-1] - line[0] + 1 != len(line):
                continue
            rays = [c0 + c for c in line]
        else:
            line = sorted(int(r) for r in windows[:, 0])
            if line[0] == 0 or line[-1] == height - 1:
                continue
            if line[-1] - line[0] + 1 != len(line):
                continue
            rays = [r0 + r for r in line]

        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "side": side,
                "rays": rays,
            }
        )

    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return rooms


def _cast_room_light(original: np.ndarray, out: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    side = room["side"]
    rays = room["rays"]

    if side == "top":
        for c in rays:
            blocked = False
            for r in range(r0 + 1, r1):
                value = int(original[r, c])
                if value == GREY:
                    break
                if value == BLACK:
                    if not blocked:
                        out[r, c] = YELLOW
                else:
                    blocked = True
    elif side == "bottom":
        for c in rays:
            blocked = False
            for r in range(r1 - 1, r0, -1):
                value = int(original[r, c])
                if value == GREY:
                    break
                if value == BLACK:
                    if not blocked:
                        out[r, c] = YELLOW
                else:
                    blocked = True
    elif side == "left":
        for r in rays:
            blocked = False
            for c in range(c0 + 1, c1):
                value = int(original[r, c])
                if value == GREY:
                    break
                if value == BLACK:
                    if not blocked:
                        out[r, c] = YELLOW
                else:
                    blocked = True
    else:
        for r in rays:
            blocked = False
            for c in range(c1 - 1, c0, -1):
                value = int(original[r, c])
                if value == GREY:
                    break
                if value == BLACK:
                    if not blocked:
                        out[r, c] = YELLOW
                else:
                    blocked = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a light direction from each room's wall window, then cast straight
    yellow rays into the room until furniture blocks them or the opposite wall
    is reached.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    assert rooms, "expected at least one rectangular room"

    out = g.copy()
    for room in rooms:
        _cast_room_light(g, out, room)
    return out


def _mark_room_footprint(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    h_total, w_total = occupied.shape
    occupied[max(0, r0 - 1) : min(h_total, r0 + h + 1), max(0, c0 - 1) : min(w_total, c0 + w + 1)] = True


def _touches_beam(side: str, rr: int, cc: int, size: int, beam_lines: list[int]) -> bool:
    if side in ("top", "bottom"):
        return any(col in beam_lines for col in range(cc, cc + size))
    return any(row in beam_lines for row in range(rr, rr + size))


def _inside_beam(side: str, rr: int, cc: int, size: int, beam_lines: list[int]) -> bool:
    if side in ("top", "bottom"):
        return all(col in beam_lines for col in range(cc, cc + size))
    return all(row in beam_lines for row in range(rr, rr + size))


def _can_place_square(occupied: np.ndarray, rr: int, cc: int, size: int) -> bool:
    ih, iw = occupied.shape
    r0 = max(0, rr - 1)
    r1 = min(ih, rr + size + 1)
    c0 = max(0, cc - 1)
    c1 = min(iw, cc + size + 1)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_square(patch: np.ndarray, occupied: np.ndarray, rr: int, cc: int, size: int, color: int) -> None:
    patch[1 + rr : 1 + rr + size, 1 + cc : 1 + cc + size] = color
    occupied[rr : rr + size, cc : cc + size] = True


def _window_data(side: str, height: int, width: int, rng: random.Random) -> tuple[int, int, list[int]]:
    if side in ("top", "bottom"):
        interior_span = width - 2
    else:
        interior_span = height - 2

    window_width = rng.choice(WINDOW_WIDTHS)
    window_width = min(window_width, interior_span)
    start = rng.randint(0, interior_span - window_width)
    beam_lines = list(range(start, start + window_width))
    return start, window_width, beam_lines


def _place_blocker(
    patch: np.ndarray,
    occupied: np.ndarray,
    side: str,
    beam_lines: list[int],
    rng: random.Random,
) -> bool:
    ih, iw = occupied.shape
    max_size = min(2, len(beam_lines))
    sizes = [1] if max_size == 1 else [1, 2]
    candidates: list[tuple[int, int, int]] = []

    for size in sizes:
        for rr in range(1, ih - size):
            for cc in range(1, iw - size):
                if not _can_place_square(occupied, rr, cc, size):
                    continue
                if not _inside_beam(side, rr, cc, size, beam_lines):
                    continue
                candidates.append((rr, cc, size))

    if not candidates:
        return False

    rr, cc, size = rng.choice(candidates)
    color = rng.choice(FURNITURE_COLORS)
    _place_square(patch, occupied, rr, cc, size, color)
    return True


def _place_extra_furniture(
    patch: np.ndarray,
    occupied: np.ndarray,
    side: str,
    beam_lines: list[int],
    count: int,
    rng: random.Random,
) -> None:
    ih, iw = occupied.shape
    for _ in range(count):
        for _ in range(FURNITURE_PLACEMENT_ATTEMPTS):
            size = rng.choice((1, 2))
            if ih - size <= 1 or iw - size <= 1:
                break
            rr = rng.randint(1, ih - size - 1)
            cc = rng.randint(1, iw - size - 1)
            if _touches_beam(side, rr, cc, size, beam_lines):
                continue
            if not _can_place_square(occupied, rr, cc, size):
                continue
            color = rng.choice(FURNITURE_COLORS)
            _place_square(patch, occupied, rr, cc, size, color)
            break


def _make_room(height: int, width: int, side: str, blocked: bool, rng: random.Random) -> np.ndarray | None:
    patch = np.full((height, width), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    start, window_width, beam_lines = _window_data(side, height, width, rng)
    if side == "top":
        patch[0, 1 + start : 1 + start + window_width] = YELLOW
    elif side == "bottom":
        patch[-1, 1 + start : 1 + start + window_width] = YELLOW
    elif side == "left":
        patch[1 + start : 1 + start + window_width, 0] = YELLOW
    else:
        patch[1 + start : 1 + start + window_width, -1] = YELLOW

    occupied = np.zeros((height - 2, width - 2), dtype=bool)
    pieces = 0

    if blocked:
        if not _place_blocker(patch, occupied, side, beam_lines, rng):
            return None
        pieces += 1

    extra_count = rng.randint(0, 2)
    if not blocked:
        extra_count = max(1, extra_count + 1)
    _place_extra_furniture(patch, occupied, side, beam_lines, extra_count, rng)
    pieces += int(np.count_nonzero(occupied))

    if pieces == 0:
        return None
    return patch


def _blocked_flags(num_rooms: int, rng: random.Random) -> list[bool]:
    if num_rooms == 1:
        return [rng.choice((False, True))]
    flags = [False, True]
    while len(flags) < num_rooms:
        flags.append(rng.choice((False, True)))
    rng.shuffle(flags)
    return flags


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    num_rooms = rng.choice(ROOM_COUNT_CHOICES)
    directions = list(rng.sample(("top", "bottom", "left", "right"), k=num_rooms))
    blocked_flags = _blocked_flags(num_rooms, rng)

    for side, blocked in zip(directions, blocked_flags):
        placed = False
        for _ in range(ROOM_PLACEMENT_ATTEMPTS):
            room_h = rng.randint(*ROOM_H_RANGE)
            room_w = rng.randint(*ROOM_W_RANGE)
            if room_h + 2 > height or room_w + 2 > width:
                continue

            room = _make_room(room_h, room_w, side, blocked, rng)
            if room is None:
                continue

            r0 = rng.randint(1, height - room_h - 1)
            c0 = rng.randint(1, width - room_w - 1)
            if np.any(occupied[r0 - 1 : r0 + room_h + 1, c0 - 1 : c0 + room_w + 1]):
                continue

            grid[r0 : r0 + room_h, c0 : c0 + room_w] = room
            _mark_room_footprint(occupied, r0, c0, room_h, room_w)
            placed = True
            break
        if not placed:
            return None

    rooms = _extract_rooms(grid)
    if len(rooms) != num_rooms:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    if np.count_nonzero(out == YELLOW) <= np.count_nonzero(grid == YELLOW):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-lighting puzzle input")

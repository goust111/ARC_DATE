from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")


def _grey_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(grid == GREY):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == GREY:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rr = [p[0] for p in cells]
        cc = [p[1] for p in cells]
        boxes.append((min(rr), max(rr), min(cc), max(cc)))
    boxes.sort()
    return boxes


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    values = sorted(values)
    result = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            result.append((start, prev))
            start = prev = value
    result.append((start, prev))
    return result


def _find_door(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> tuple[str, int, int] | None:
    side_values = {
        "top": [c for c in range(c0 + 1, c1) if grid[r0, c] != GREY],
        "bottom": [c for c in range(c0 + 1, c1) if grid[r1, c] != GREY],
        "left": [r for r in range(r0 + 1, r1) if grid[r, c0] != GREY],
        "right": [r for r in range(r0 + 1, r1) if grid[r, c1] != GREY],
    }
    candidates: list[tuple[str, int, int]] = []
    for side, values in side_values.items():
        for start, end in _runs(values):
            candidates.append((side, start, end))
    if len(candidates) != 1:
        return None
    side, start, end = candidates[0]
    if not 1 <= end - start + 1 <= 3:
        return None
    return side, start, end


def _inside_color(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int | None:
    if r1 <= r0 + 1 or c1 <= c0 + 1:
        return None
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    colored = np.argwhere((inside != BLACK) & (inside != GREY))
    if len(colored) != 1:
        return None
    rr, cc = colored[0]
    return int(inside[int(rr), int(cc)])


def _detect_rooms(grid: np.ndarray) -> list[dict[str, int | str]]:
    rooms: list[dict[str, int | str]] = []
    for r0, r1, c0, c1 in _grey_components(grid):
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        door = _find_door(grid, r0, r1, c0, c1)
        color = _inside_color(grid, r0, r1, c0, c1)
        if door is None or color is None:
            continue
        side, start, end = door
        rooms.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "side": side,
                "start": start,
                "end": end,
                "color": color,
            }
        )
    rooms.sort(key=lambda room: (int(room["r0"]), int(room["c0"])))
    return rooms


def _draw_corridor(out: np.ndarray, source: np.ndarray, room: dict[str, int | str]) -> None:
    rows, cols = source.shape
    side = str(room["side"])
    start = int(room["start"])
    end = int(room["end"])
    color = int(room["color"])
    r0, r1, c0, c1 = int(room["r0"]), int(room["r1"]), int(room["c0"]), int(room["c1"])

    if side == "top":
        stripe = np.arange(start, end + 1)
        r = r0 - 1
        while r >= 0 and not np.any(source[r, stripe] == GREY):
            out[r, stripe] = color
            r -= 1
    elif side == "bottom":
        stripe = np.arange(start, end + 1)
        r = r1 + 1
        while r < rows and not np.any(source[r, stripe] == GREY):
            out[r, stripe] = color
            r += 1
    elif side == "left":
        stripe = np.arange(start, end + 1)
        c = c0 - 1
        while c >= 0 and not np.any(source[stripe, c] == GREY):
            out[stripe, c] = color
            c -= 1
    else:
        stripe = np.arange(start, end + 1)
        c = c1 + 1
        while c < cols and not np.any(source[stripe, c] == GREY):
            out[stripe, c] = color
            c += 1


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each room door into an outside corridor colored by that room's inner marker.

    Input structure:
    A black grid contains several hollow grey rectangular room walls. Each room has one
    contiguous gap of width one to three on a wall, and exactly one non-black, non-grey
    colored pixel somewhere inside the room.

    Rule:
    Detect every grey room rectangle, locate its single wall gap, read the unique inner
    colored pixel, and draw a straight outside corridor from the gap. The corridor has
    the same width as the gap, uses the inner pixel's color, and extends in the door's
    outward direction until its next front would hit a grey wall or leave the grid.
    Existing room walls and interiors are preserved.

    Key insight:
    The door only gives the corridor position and direction; the corridor color must be
    inferred from the isolated colored pixel hidden inside the same room.
    """
    source = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    for room in _detect_rooms(source):
        _draw_corridor(out, source, room)
    return out


def _opposite(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]


def _capacity(r0: int, r1: int, c0: int, c1: int, side: str) -> int:
    return (c1 - c0 - 1) if side in ("top", "bottom") else (r1 - r0 - 1)


def _make_room(
    r0: int,
    c0: int,
    height: int,
    width: int,
    side: str | None = None,
    start: int | None = None,
    door_width: int | None = None,
    forbidden_side: str | None = None,
) -> dict[str, int | str]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    choices = [item for item in SIDES if item != forbidden_side]
    if side is None:
        side = random.choice(choices)
    cap = _capacity(r0, r1, c0, c1, side)
    if door_width is None:
        door_width = random.randint(1, min(3, cap))
    if start is None:
        low = c0 + 1 if side in ("top", "bottom") else r0 + 1
        high = (c1 - door_width) if side in ("top", "bottom") else (r1 - door_width)
        start = random.randint(low, high)
    return {
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "side": side,
        "start": start,
        "end": start + door_width - 1,
        "color": random.choice(COLORS),
    }


def _boxes_too_close(a: dict[str, int | str], b: dict[str, int | str], margin: int = 1) -> bool:
    return not (
        int(a["r1"]) + margin < int(b["r0"])
        or int(b["r1"]) + margin < int(a["r0"])
        or int(a["c1"]) + margin < int(b["c0"])
        or int(b["c1"]) + margin < int(a["c0"])
    )


def _render_rooms(rows: int, cols: int, rooms: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for room in rooms:
        r0, r1, c0, c1 = int(room["r0"]), int(room["r1"]), int(room["c0"]), int(room["c1"])
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        side = str(room["side"])
        start = int(room["start"])
        end = int(room["end"])
        if side == "top":
            grid[r0, start : end + 1] = BLACK
        elif side == "bottom":
            grid[r1, start : end + 1] = BLACK
        elif side == "left":
            grid[start : end + 1, c0] = BLACK
        else:
            grid[start : end + 1, c1] = BLACK
    for room in rooms:
        r0, r1, c0, c1 = int(room["r0"]), int(room["r1"]), int(room["c0"]), int(room["c1"])
        rr = random.randint(r0 + 1, r1 - 1)
        cc = random.randint(c0 + 1, c1 - 1)
        grid[rr, cc] = int(room["color"])
    return grid


def _corridor_cells(grid: np.ndarray, room: dict[str, int | str]) -> list[tuple[int, int]]:
    rows, cols = grid.shape
    side = str(room["side"])
    start = int(room["start"])
    end = int(room["end"])
    r0, r1, c0, c1 = int(room["r0"]), int(room["r1"]), int(room["c0"]), int(room["c1"])
    cells: list[tuple[int, int]] = []
    if side == "top":
        r = r0 - 1
        stripe = list(range(start, end + 1))
        while r >= 0 and not any(grid[r, c] == GREY for c in stripe):
            cells.extend((r, c) for c in stripe)
            r -= 1
    elif side == "bottom":
        r = r1 + 1
        stripe = list(range(start, end + 1))
        while r < rows and not any(grid[r, c] == GREY for c in stripe):
            cells.extend((r, c) for c in stripe)
            r += 1
    elif side == "left":
        c = c0 - 1
        stripe = list(range(start, end + 1))
        while c >= 0 and not any(grid[r, c] == GREY for r in stripe):
            cells.extend((r, c) for r in stripe)
            c -= 1
    else:
        c = c1 + 1
        stripe = list(range(start, end + 1))
        while c < cols and not any(grid[r, c] == GREY for r in stripe):
            cells.extend((r, c) for r in stripe)
            c += 1
    return cells


def _cell_in_box(cell: tuple[int, int], room: dict[str, int | str]) -> bool:
    r, c = cell
    return int(room["r0"]) <= r <= int(room["r1"]) and int(room["c0"]) <= c <= int(room["c1"])


def _valid_generation(grid: np.ndarray, rooms: list[dict[str, int | str]]) -> bool:
    if len(_detect_rooms(grid)) != len(rooms):
        return False
    used: set[tuple[int, int]] = set()
    for i, room in enumerate(rooms):
        cells = _corridor_cells(grid, room)
        if not cells:
            return False
        for cell in cells:
            if grid[cell] != BLACK:
                return False
            for j, other in enumerate(rooms):
                if i != j and _cell_in_box(cell, other):
                    return False
            if cell in used:
                return False
        used.update(cells)
    return True


def _add_random_room(rows: int, cols: int, rooms: list[dict[str, int | str]]) -> bool:
    for _ in range(120):
        height = random.randint(4, 8)
        width = random.randint(4, 8)
        if rows - height - 1 < 1 or cols - width - 1 < 1:
            continue
        r0 = random.randint(1, rows - height - 1)
        c0 = random.randint(1, cols - width - 1)
        room = _make_room(r0, c0, height, width)
        if all(not _boxes_too_close(room, other) for other in rooms):
            rooms.append(room)
            return True
    return False


def _overlap_starts(a0: int, a1: int, size: int, limit: int) -> list[int]:
    starts = []
    for b0 in range(1, limit - size):
        b1 = b0 + size - 1
        if min(a1, b1 - 1) >= max(a0, b0 + 1):
            starts.append(b0)
    return starts


def _build_hitting_pair(rows: int, cols: int) -> list[dict[str, int | str]] | None:
    for _ in range(120):
        side = random.choice(SIDES)
        sh, sw = random.randint(5, 8), random.randint(5, 8)
        th, tw = random.randint(5, 8), random.randint(5, 8)
        gap = random.randint(2, 6)
        if side in ("left", "right"):
            if cols < sw + gap + tw + 2 or rows < max(sh, th) + 2:
                continue
            sr0 = random.randint(1, rows - sh - 1)
            sr1 = sr0 + sh - 1
            possible = _overlap_starts(sr0 + 1, sr1 - 1, th, rows)
            if not possible:
                continue
            tr0 = random.choice(possible)
            tr1 = tr0 + th - 1
            ov0 = max(sr0 + 1, tr0 + 1)
            ov1 = min(sr1 - 1, tr1 - 1)
            door_width = random.randint(1, min(3, ov1 - ov0 + 1))
            start = random.randint(ov0, ov1 - door_width + 1)
            total = sw + gap + tw
            if side == "right":
                sc0 = random.randint(1, cols - total - 1)
                tc0 = sc0 + sw + gap
                source = _make_room(sr0, sc0, sh, sw, side="right", start=start, door_width=door_width)
                target = _make_room(tr0, tc0, th, tw, forbidden_side="left")
            else:
                tc0 = random.randint(1, cols - total - 1)
                sc0 = tc0 + tw + gap
                source = _make_room(sr0, sc0, sh, sw, side="left", start=start, door_width=door_width)
                target = _make_room(tr0, tc0, th, tw, forbidden_side="right")
        else:
            if rows < sh + gap + th + 2 or cols < max(sw, tw) + 2:
                continue
            sc0 = random.randint(1, cols - sw - 1)
            sc1 = sc0 + sw - 1
            possible = _overlap_starts(sc0 + 1, sc1 - 1, tw, cols)
            if not possible:
                continue
            tc0 = random.choice(possible)
            tc1 = tc0 + tw - 1
            ov0 = max(sc0 + 1, tc0 + 1)
            ov1 = min(sc1 - 1, tc1 - 1)
            door_width = random.randint(1, min(3, ov1 - ov0 + 1))
            start = random.randint(ov0, ov1 - door_width + 1)
            total = sh + gap + th
            if side == "bottom":
                sr0 = random.randint(1, rows - total - 1)
                tr0 = sr0 + sh + gap
                source = _make_room(sr0, sc0, sh, sw, side="bottom", start=start, door_width=door_width)
                target = _make_room(tr0, tc0, th, tw, forbidden_side="top")
            else:
                tr0 = random.randint(1, rows - total - 1)
                sr0 = tr0 + th + gap
                source = _make_room(sr0, sc0, sh, sw, side="top", start=start, door_width=door_width)
                target = _make_room(tr0, tc0, th, tw, forbidden_side="bottom")
        if not _boxes_too_close(source, target):
            return [source, target]
    return None


def _random_room_set(rows: int, cols: int, count: int) -> list[dict[str, int | str]] | None:
    rooms: list[dict[str, int | str]] = []
    for _ in range(count):
        if not _add_random_room(rows, cols, rooms):
            return None
    return rooms


def generate_input() -> np.ndarray:
    for _ in range(250):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        target_count = random.randint(2, 4)
        if random.random() < 0.65:
            rooms = _build_hitting_pair(rows, cols)
            if rooms is None:
                continue
            while len(rooms) < target_count and _add_random_room(rows, cols, rooms):
                pass
        else:
            rooms = _random_room_set(rows, cols, target_count)
            if rooms is None:
                continue
        grid = _render_rooms(rows, cols, rooms)
        if _valid_generation(grid, rooms):
            return grid
    rooms = _build_hitting_pair(24, 24)
    if rooms is None:
        rooms = [
            _make_room(3, 2, 6, 6, side="right", start=5, door_width=2),
            _make_room(2, 13, 7, 6, forbidden_side="left"),
        ]
    grid = _render_rooms(24, 24, rooms)
    return grid

from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7
ARROW_COLORS = (RED, GREEN, ORANGE, MAGENTA)
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _isolated_nonwall_positions(line: np.ndarray) -> bool:
    idx = np.flatnonzero(line != WALL)
    return idx.size == 0 or bool(np.all(np.diff(idx) > 1))


def _parse_wall_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = g.shape
    row_lines = [r for r in range(h) if _isolated_nonwall_positions(g[r, :])]
    col_lines = [c for c in range(w) if _isolated_nonwall_positions(g[:, c])]
    assert row_lines and col_lines
    assert row_lines[0] == 0 and row_lines[-1] == h - 1
    assert col_lines[0] == 0 and col_lines[-1] == w - 1
    assert all(b - a >= 4 for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a >= 4 for a, b in zip(col_lines, col_lines[1:]))
    return row_lines, col_lines


def _validate_arrow(room: np.ndarray) -> None:
    present = []
    for color in ARROW_COLORS:
        coords = np.argwhere(room == color)
        if coords.size == 0:
            continue
        assert coords.shape[0] == 3
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert r1 - r0 == 1 and c1 - c0 == 1
        present.append(int(color))
    assert len(present) == 1


def _room_door(
    g: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> tuple[str, int, int]:
    doors: list[tuple[str, int, int]] = []

    top = np.flatnonzero(g[r0 - 1, c0 : c1 + 1] != WALL)
    if top.size:
        assert top.size == 1
        doors.append(("up", r0 - 1, c0 + int(top[0])))

    bottom = np.flatnonzero(g[r1 + 1, c0 : c1 + 1] != WALL)
    if bottom.size:
        assert bottom.size == 1
        doors.append(("down", r1 + 1, c0 + int(bottom[0])))

    left = np.flatnonzero(g[r0 : r1 + 1, c0 - 1] != WALL)
    if left.size:
        assert left.size == 1
        doors.append(("left", r0 + int(left[0]), c0 - 1))

    right = np.flatnonzero(g[r0 : r1 + 1, c1 + 1] != WALL)
    if right.size:
        assert right.size == 1
        doors.append(("right", r0 + int(right[0]), c1 + 1))

    assert len(doors) == 1
    return doors[0]


def _extract_rooms(g: np.ndarray) -> list[dict]:
    row_lines, col_lines = _parse_wall_lines(g)
    rooms: list[dict] = []
    for ri, (rt, rb) in enumerate(zip(row_lines, row_lines[1:])):
        for ci, (cl, cr) in enumerate(zip(col_lines, col_lines[1:])):
            r0, r1 = rt + 1, rb - 1
            c0, c1 = cl + 1, cr - 1
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            assert room.shape[0] >= 3 and room.shape[1] >= 3
            assert not np.any(room == WALL)
            _validate_arrow(room)
            side, door_r, door_c = _room_door(g, r0, r1, c0, c1)
            greys = [tuple(map(int, rc)) for rc in np.argwhere(room == GREY) + np.array([r0, c0])]
            rooms.append(
                {
                    "grid_pos": (ri, ci),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "side": side,
                    "door": (door_r, door_c),
                    "greys": greys,
                }
            )
    assert rooms
    return rooms


def _distance_to_door(room: dict, rc: tuple[int, int]) -> int:
    r, c = rc
    side = room["side"]
    door_r, door_c = room["door"]
    if side == "up":
        return int(r - door_r)
    if side == "down":
        return int(door_r - r)
    if side == "left":
        return int(c - door_c)
    return int(door_c - c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's grey pixels through its unique doorway into the neighboring room.

    The blue walls form a rectangular room grid. Each room contains one 3-pixel L arrow and
    zero to three grey pixels. The room's unique doorway is the lone hole on one of its four
    walls; all grey pixels in that room move in that doorway direction, pass through the hole,
    and keep sliding in the adjacent room until they hit a wall, a fixed arrow, or a grey
    pixel that has already stopped there. Pixels closer to the doorway move first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    h, w = g.shape
    out = g.copy()
    out[out == GREY] = BG

    for room in rooms:
        if not room["greys"]:
            continue
        side = room["side"]
        door_r, door_c = room["door"]
        dr, dc = SIDE_TO_DIR[side]
        ordered = sorted(room["greys"], key=lambda rc: _distance_to_door(room, rc))
        entry_r = door_r + dr
        entry_c = door_c + dc
        assert 0 <= entry_r < h and 0 <= entry_c < w

        for _src in ordered:
            r = entry_r
            c = entry_c
            assert out[r, c] == BG
            while 0 <= r + dr < h and 0 <= c + dc < w and out[r + dr, c + dc] == BG:
                r += dr
                c += dc
            out[r, c] = GREY

    return out


# --------------- generator ---------------

ROOM_ROW_CHOICES = (2, 3, 4)
ROOM_COL_CHOICES = (2, 3, 4)
ROOM_SIZE_RANGE = (3, 6)
GREY_COUNT_MAX = 3
GEN_ATTEMPTS = 400


def _safe_door_offsets(length: int) -> list[int]:
    assert length >= 3
    if length == 3:
        return [0, 2]
    return list(range(length))


def _perfect_matching(rows: int, cols: int, rng: random.Random) -> dict[tuple[int, int], tuple[int, int]] | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    used: set[tuple[int, int]] = set()
    pairs: dict[tuple[int, int], tuple[int, int]] = {}

    def backtrack() -> bool:
        if len(used) == rows * cols:
            return True
        cell = next(rc for rc in cells if rc not in used)
        r, c = cell
        nbrs = []
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr = r + dr
            nc = c + dc
            nbr = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nbr not in used:
                nbrs.append(nbr)
        rng.shuffle(nbrs)
        for nbr in nbrs:
            used.add(cell)
            used.add(nbr)
            pairs[cell] = nbr
            pairs[nbr] = cell
            if backtrack():
                return True
            used.remove(cell)
            used.remove(nbr)
            del pairs[cell]
            del pairs[nbr]
        return False

    return pairs if backtrack() else None


def _arrow_mask(side: str, rng: random.Random) -> np.ndarray:
    missing_choices = {
        "up": [(0, 0), (0, 1)],
        "down": [(1, 0), (1, 1)],
        "left": [(0, 0), (1, 0)],
        "right": [(0, 1), (1, 1)],
    }
    mask = np.ones((2, 2), dtype=bool)
    mr, mc = rng.choice(missing_choices[side])
    mask[mr, mc] = False
    return mask


def _door_and_side_by_room(
    rows: int,
    cols: int,
    heights: list[int],
    widths: list[int],
    wall_rows: list[int],
    wall_cols: list[int],
    pairing: dict[tuple[int, int], tuple[int, int]],
    rng: random.Random,
) -> dict[tuple[int, int], tuple[str, int, int]]:
    info: dict[tuple[int, int], tuple[str, int, int]] = {}
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for cell, other in pairing.items():
        edge = tuple(sorted((cell, other)))
        if edge in seen:
            continue
        seen.add(edge)
        (r1, c1), (r2, c2) = edge
        if r1 == r2:
            row = r1
            left = (row, min(c1, c2))
            right = (row, max(c1, c2))
            offset = rng.choice(_safe_door_offsets(heights[row]))
            door_r = wall_rows[row] + 1 + offset
            door_c = wall_cols[left[1] + 1]
            info[left] = ("right", door_r, door_c)
            info[right] = ("left", door_r, door_c)
        else:
            col = c1
            top = (min(r1, r2), col)
            bottom = (max(r1, r2), col)
            offset = rng.choice(_safe_door_offsets(widths[col]))
            door_r = wall_rows[top[0] + 1]
            door_c = wall_cols[col] + 1 + offset
            info[top] = ("down", door_r, door_c)
            info[bottom] = ("up", door_r, door_c)

    assert len(info) == rows * cols
    return info


def _room_arrow_starts(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    side: str,
    door_r: int,
    door_c: int,
) -> list[tuple[int, int]]:
    starts = []
    if side in ("left", "right"):
        blocked = door_r - r0
        valid_rows = [rr for rr in range(r0, r1) if blocked not in (rr - r0, rr - r0 + 1)]
        valid_cols = list(range(c0, c1))
    else:
        blocked = door_c - c0
        valid_rows = list(range(r0, r1))
        valid_cols = [cc for cc in range(c0, c1) if blocked not in (cc - c0, cc - c0 + 1)]

    for rr in valid_rows:
        for cc in valid_cols:
            starts.append((rr, cc))
    return starts


def _line_cells(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    side: str,
    door_r: int,
    door_c: int,
) -> list[tuple[int, int]]:
    if side in ("left", "right"):
        return [(door_r, c) for c in range(c0, c1 + 1)]
    return [(r, door_c) for r in range(r0, r1 + 1)]


def _sample_case() -> np.ndarray | None:
    rng = random.Random()
    row_col_choices = [(r, c) for r in ROOM_ROW_CHOICES for c in ROOM_COL_CHOICES if (r * c) % 2 == 0]
    room_rows, room_cols = rng.choice(row_col_choices)
    heights = [rng.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(room_rows)]
    widths = [rng.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(room_cols)]
    wall_rows = [0]
    wall_cols = [0]
    for h in heights:
        wall_rows.append(wall_rows[-1] + h + 1)
    for w in widths:
        wall_cols.append(wall_cols[-1] + w + 1)

    grid = np.full((wall_rows[-1] + 1, wall_cols[-1] + 1), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    pairing = _perfect_matching(room_rows, room_cols, rng)
    if pairing is None:
        return None

    door_info = _door_and_side_by_room(
        room_rows,
        room_cols,
        heights,
        widths,
        wall_rows,
        wall_cols,
        pairing,
        rng,
    )

    for side, door_r, door_c in door_info.values():
        grid[door_r, door_c] = BG

    room_cells = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    rng.shuffle(room_cells)
    grey_plan: dict[tuple[int, int], list[tuple[int, int]]] = {}
    nonzero_rooms = 0

    for rr, cc in room_cells:
        r0 = wall_rows[rr] + 1
        r1 = wall_rows[rr + 1] - 1
        c0 = wall_cols[cc] + 1
        c1 = wall_cols[cc + 1] - 1
        side, door_r, door_c = door_info[(rr, cc)]

        starts = _room_arrow_starts(r0, r1, c0, c1, side, door_r, door_c)
        if not starts:
            return None
        ar0, ac0 = rng.choice(starts)
        arrow_color = rng.choice(ARROW_COLORS)
        arrow_mask = _arrow_mask(side, rng)
        arrow_cells = []
        for dr in range(2):
            for dc in range(2):
                if arrow_mask[dr, dc]:
                    arrow_cells.append((ar0 + dr, ac0 + dc))
                    grid[ar0 + dr, ac0 + dc] = arrow_color

        line = _line_cells(r0, r1, c0, c1, side, door_r, door_c)
        free_line = [cell for cell in line if cell not in set(arrow_cells)]
        if not free_line:
            return None
        grey_count = rng.randint(0, min(GREY_COUNT_MAX, len(free_line)))
        if grey_count:
            grey_plan[(rr, cc)] = rng.sample(free_line, grey_count)
            nonzero_rooms += 1
        else:
            grey_plan[(rr, cc)] = []

    if nonzero_rooms == 0:
        rr, cc = rng.choice(room_cells)
        r0 = wall_rows[rr] + 1
        r1 = wall_rows[rr + 1] - 1
        c0 = wall_cols[cc] + 1
        c1 = wall_cols[cc + 1] - 1
        side, door_r, door_c = door_info[(rr, cc)]
        line = _line_cells(r0, r1, c0, c1, side, door_r, door_c)
        free_line = [cell for cell in line if grid[cell] == BG]
        if not free_line:
            return None
        grey_plan[(rr, cc)] = [rng.choice(free_line)]

    for coords in grey_plan.values():
        for r, c in coords:
            grid[r, c] = GREY

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid room-push puzzle input")

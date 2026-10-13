import random
from collections import deque

import numpy as np


BLACK = 0
DEFAULT_WALL = 5
PALETTE_LEN = 4
ORIENTATIONS = ("top", "bottom", "left", "right")

# --------------- solver helpers ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _wall_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    if DEFAULT_WALL in vals:
        return DEFAULT_WALL
    non_black = [(int(v), int(cnt)) for v, cnt in zip(vals, counts) if int(v) != BLACK]
    assert non_black, "expected at least one non-black color"
    return max(non_black, key=lambda item: item[1])[0]


def _find_palette(grid: np.ndarray, wall: int) -> tuple[list[int], np.ndarray]:
    colored = (grid != BLACK) & (grid != wall)
    comps = _components(colored)
    assert comps, "expected a color palette component"

    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r0 != r1:
            continue
        if len(comp) != PALETTE_LEN or c1 - c0 + 1 != PALETTE_LEN:
            continue
        colors = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
        dist = min(
            r0 + c0,
            r0 + (w - 1 - c1),
            (h - 1 - r0) + c0,
            (h - 1 - r0) + (w - 1 - c1),
        )
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        candidates.append((dist, r0, c0, colors, mask))

    assert candidates, "failed to locate a 1x4 corner palette"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, colors, mask = candidates[0]
    return colors, mask


def _has_black_or_outside(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return True
        if int(grid[r, c]) == BLACK:
            return True
    return False


def _is_horizontal_door(grid: np.ndarray, r: int, c: int, wall: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK:
        return False
    if c - 1 < 0 or c + 1 >= w:
        return False
    if int(grid[r, c - 1]) != wall or int(grid[r, c + 1]) != wall:
        return False
    if not _has_black_or_outside(grid, [(r - 1, c), (r + 1, c)]):
        return False
    return not _is_vertical_gap_pattern(grid, r, c, wall)


def _is_vertical_gap_pattern(grid: np.ndarray, r: int, c: int, wall: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK:
        return False
    if r - 1 < 0 or r + 1 >= h:
        return False
    return int(grid[r - 1, c]) == wall and int(grid[r + 1, c]) == wall


def _is_vertical_door(grid: np.ndarray, r: int, c: int, wall: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK:
        return False
    if r - 1 < 0 or r + 1 >= h:
        return False
    if int(grid[r - 1, c]) != wall or int(grid[r + 1, c]) != wall:
        return False
    if not _has_black_or_outside(grid, [(r, c - 1), (r, c + 1)]):
        return False
    if c - 1 >= 0 and c + 1 < w and int(grid[r, c - 1]) == wall and int(grid[r, c + 1]) == wall:
        return False
    return True


def _closed_wall_grid(grid: np.ndarray, wall: int) -> np.ndarray:
    closed = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            horiz = _is_horizontal_door(grid, r, c, wall)
            vert = _is_vertical_door(grid, r, c, wall)
            if horiz ^ vert:
                closed[r, c] = wall
    return closed


def _room_bboxes(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    closed = _closed_wall_grid(grid, wall)
    h, w = grid.shape
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components(closed == BLACK):
        if _touches_edge(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) != area:
            continue
        if not np.all(closed[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
            continue
        if r0 - 1 < 0 or c0 - 1 < 0 or r1 + 1 >= h or c1 + 1 >= w:
            continue
        frame_ok = (
            np.all(closed[r0 - 1, c0 - 1 : c1 + 2] == wall)
            and np.all(closed[r1 + 1, c0 - 1 : c1 + 2] == wall)
            and np.all(closed[r0 - 1 : r1 + 2, c0 - 1] == wall)
            and np.all(closed[r0 - 1 : r1 + 2, c1 + 1] == wall)
        )
        if frame_ok:
            rooms.append((r0, c0, r1, c1))
    assert rooms, "expected at least one enclosed room"
    return rooms


def _doors_for_room(grid: np.ndarray, bbox: tuple[int, int, int, int], wall: int) -> list[tuple[int, int, str]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    doors: list[tuple[int, int, str]] = []

    top_r = r0 - 1
    if top_r >= 0:
        for c in range(c0, c1 + 1):
            if _is_horizontal_door(grid, top_r, c, wall):
                doors.append((top_r, c, "top"))

    bottom_r = r1 + 1
    if bottom_r < h:
        for c in range(c0, c1 + 1):
            if _is_horizontal_door(grid, bottom_r, c, wall):
                doors.append((bottom_r, c, "bottom"))

    left_c = c0 - 1
    if left_c >= 0:
        for r in range(r0, r1 + 1):
            if _is_vertical_door(grid, r, left_c, wall):
                doors.append((r, left_c, "left"))

    right_c = c1 + 1
    if right_c < w:
        for r in range(r0, r1 + 1):
            if _is_vertical_door(grid, r, right_c, wall):
                doors.append((r, right_c, "right"))

    doors.sort(key=lambda item: (item[0], item[1], item[2]))
    return doors


def _fill_room(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, orientation: str) -> None:
    r0, c0, r1, c1 = bbox
    if orientation == "top":
        for offset, r in enumerate(range(r0, r1 + 1)):
            if offset % 2 == 0:
                out[r, c0 : c1 + 1] = color
            else:
                out[r, c0 : c1 + 1] = BLACK
    elif orientation == "bottom":
        for offset, r in enumerate(range(r1, r0 - 1, -1)):
            if offset % 2 == 0:
                out[r, c0 : c1 + 1] = color
            else:
                out[r, c0 : c1 + 1] = BLACK
    elif orientation == "left":
        for offset, c in enumerate(range(c0, c1 + 1)):
            if offset % 2 == 0:
                out[r0 : r1 + 1, c] = color
            else:
                out[r0 : r1 + 1, c] = BLACK
    else:
        for offset, c in enumerate(range(c1, c0 - 1, -1)):
            if offset % 2 == 0:
                out[r0 : r1 + 1, c] = color
            else:
                out[r0 : r1 + 1, c] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled rectangular room with a striped gradient whose color depends on door count and whose direction depends on the first door.

    Input structure:
    - Black is the background and also the base color inside all rooms.
    - Grey wall pixels form axis-aligned 1-pixel-thick rectangular room borders, with some 1-pixel black gaps acting as doors.
    - Each room is a black rectangle surrounded by grey walls, and the room has between one and four door gaps on its border.
    - A separate 1x4 horizontal color pool sits in a corner area and maps door counts 1..4 from left to right.

    Rule:
    - Read the corner palette left-to-right so its first, second, third, and fourth colors correspond to rooms with 1, 2, 3, and 4 doors.
    - Recover each room by conceptually closing every 1-pixel door gap in the grey wall network, then find enclosed black rectangular room interiors.
    - For each room, count its original door gaps and locate the first door by topmost-then-leftmost order.
    - Fill the room interior with alternating colored and black stripes, starting from the side adjacent to that first door: top door means top-to-bottom, bottom door means bottom-to-top, left door means left-to-right, and right door means right-to-left.

    Key insight:
    - The problem splits into two independent inferences on the same room: door count selects the palette color, while the earliest door position selects the stripe direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    wall = _wall_color(g)
    palette, _palette_mask = _find_palette(g, wall)
    rooms = _room_bboxes(g, wall)

    out = g.copy()
    for room in rooms:
        doors = _doors_for_room(g, room, wall)
        assert 1 <= len(doors) <= PALETTE_LEN, "room door count must be in 1..4"
        fill = int(palette[len(doors) - 1])
        orientation = doors[0][2]
        _fill_room(out, room, fill, orientation)
    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
ROOM_COUNT_RANGE = (2, 4)
ROOM_INNER_MIN = 2
ROOM_INNER_MAX = 7
ROOM_BUFFER = 2
PALETTE_BUFFER = 2
GEN_RETRIES = 400
PLACE_RETRIES = 200
PALETTE_COLORS = tuple(c for c in range(1, 10) if c != DEFAULT_WALL)


def _choose_nonadjacent_positions(length: int, count: int, rng: random.Random) -> list[int] | None:
    cells = list(range(length))
    for _ in range(100):
        picks = sorted(rng.sample(cells, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    for start in (rng.randrange(2), 1 - (rng.randrange(2))):
        picks = list(range(start, length, 2))
        if len(picks) >= count:
            return picks[:count]
    for start in (0, 1):
        picks = list(range(start, length, 2))
        if len(picks) >= count:
            return picks[:count]
    return None


def _room_spec(door_count: int, orientation: str, rng: random.Random) -> dict[str, object] | None:
    if orientation in ("top", "bottom"):
        min_w = 2 * door_count - 1
        if min_w > ROOM_INNER_MAX:
            return None
        inner_h = rng.randint(ROOM_INNER_MIN, ROOM_INNER_MAX)
        inner_w = rng.randint(min_w, ROOM_INNER_MAX)
        door_positions = _choose_nonadjacent_positions(inner_w, door_count, rng)
    else:
        min_h = 2 * door_count - 1
        if min_h > ROOM_INNER_MAX:
            return None
        inner_h = rng.randint(min_h, ROOM_INNER_MAX)
        inner_w = rng.randint(ROOM_INNER_MIN, ROOM_INNER_MAX)
        door_positions = _choose_nonadjacent_positions(inner_h, door_count, rng)
    if door_positions is None:
        return None
    return {
        "door_count": door_count,
        "orientation": orientation,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "door_positions": door_positions,
    }


def _palette_anchor(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - PALETTE_LEN
    if corner == "bl":
        return h - 1, 0
    return h - 1, w - PALETTE_LEN


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
        return False
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _draw_room(grid: np.ndarray, top: int, left: int, spec: dict[str, object], wall: int) -> None:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    bottom = top + inner_h + 1
    right = left + inner_w + 1

    grid[top, left : right + 1] = wall
    grid[bottom, left : right + 1] = wall
    grid[top : bottom + 1, left] = wall
    grid[top : bottom + 1, right] = wall

    orientation = str(spec["orientation"])
    for pos in spec["door_positions"]:
        idx = int(pos)
        if orientation == "top":
            grid[top, left + 1 + idx] = BLACK
        elif orientation == "bottom":
            grid[bottom, left + 1 + idx] = BLACK
        elif orientation == "left":
            grid[top + 1 + idx, left] = BLACK
        else:
            grid[top + 1 + idx, right] = BLACK


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    counts = rng.sample([1, 2, 3, 4], room_count)
    orientations = rng.sample(list(ORIENTATIONS), room_count)

    specs: list[dict[str, object]] = []
    for door_count, orientation in zip(counts, orientations):
        spec = _room_spec(door_count, orientation, rng)
        if spec is None:
            return None
        specs.append(spec)

    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    palette_colors = rng.sample(PALETTE_COLORS, PALETTE_LEN)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    pr, pc = _palette_anchor(h, w, corner)
    grid[pr, pc : pc + PALETTE_LEN] = palette_colors
    if not _reserve(blocked, pr, pc, pr, pc + PALETTE_LEN - 1, PALETTE_BUFFER):
        return None

    placed: list[tuple[int, int, dict[str, object]]] = []
    for spec in specs:
        outer_h = int(spec["inner_h"]) + 2
        outer_w = int(spec["inner_w"]) + 2
        if outer_h + 2 >= h or outer_w + 2 >= w:
            return None

        success = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(1, h - outer_h - 1)
            left = rng.randint(1, w - outer_w - 1)
            local = blocked.copy()
            if not _reserve(local, top, left, top + outer_h - 1, left + outer_w - 1, ROOM_BUFFER):
                continue
            blocked = local
            placed.append((top, left, spec))
            success = True
            break
        if not success:
            return None

    wall = DEFAULT_WALL
    for top, left, spec in placed:
        _draw_room(grid, top, left, spec, wall)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door puzzle")

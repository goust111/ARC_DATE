import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
REFLECT = {
    "\\": {
        (1, 0): (0, 1),
        (0, 1): (1, 0),
        (-1, 0): (0, -1),
        (0, -1): (-1, 0),
    },
    "/": {
        (1, 0): (0, -1),
        (0, -1): (1, 0),
        (-1, 0): (0, 1),
        (0, 1): (-1, 0),
    },
}
GEN_ATTEMPTS = 400
GRID_SIZE_RANGE = (16, 31)
ROOM_COUNT_RANGE = (1, 5)
ROOM_HEIGHT_RANGE = (6, 13)
ROOM_WIDTH_RANGE = (6, 13)
ROOM_GAP = 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _room_border_components(grid: np.ndarray) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components4(grid != BG):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue

        on_perimeter = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
        if not np.all(on_perimeter):
            continue

        expected_size = 2 * (height + width) - 4
        if len(coords) != expected_size:
            continue

        perim = {
            (r0, c) for c in range(c0, c1 + 1)
        } | {
            (r1, c) for c in range(c0, c1 + 1)
        } | {
            (r, c0) for r in range(r0, r1 + 1)
        } | {
            (r, c1) for r in range(r0, r1 + 1)
        }
        if set(map(tuple, coords.tolist())) != perim:
            continue

        colors = [int(grid[r, c]) for r, c in coords]
        if any(color == BG for color in colors):
            continue
        entrance = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) != GREY]
        if len(entrance) != 1:
            continue
        if int(grid[entrance[0]]) == BG:
            continue
        rooms.append({"bbox": (r0, c0, r1, c1), "entrance": entrance[0]})
    return rooms


def _mirror_info(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, int, set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) != BG
    ]
    assert 2 <= len(cells) <= 4, "mirror must contain 2..4 pixels"
    colors = {int(grid[r, c]) for r, c in cells}
    assert len(colors) == 1, "mirror must have a single color"
    color = next(iter(colors))
    assert color != GREY, "mirror color must differ from grey border"

    if len({r - c for r, c in cells}) == 1:
        orient = "\\"
        ordered = sorted(cells)
        for (r, c), (nr, nc) in zip(ordered, ordered[1:]):
            assert nr == r + 1 and nc == c + 1, "mirror must be contiguous"
    else:
        assert len({r + c for r, c in cells}) == 1, "mirror must be diagonal"
        orient = "/"
        ordered = sorted(cells)
        for (r, c), (nr, nc) in zip(ordered, ordered[1:]):
            assert nr == r + 1 and nc == c - 1, "mirror must be contiguous"
    return orient, int(color), set(cells)


def _entrance_dir(entrance: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = entrance
    r0, c0, r1, c1 = bbox
    on_top = r == r0
    on_bottom = r == r1
    on_left = c == c0
    on_right = c == c1
    assert on_top + on_bottom + on_left + on_right == 1, "entrance must lie on exactly one wall"
    if on_top:
        return (1, 0)
    if on_bottom:
        return (-1, 0)
    if on_left:
        return (0, 1)
    return (0, -1)


def _trace_exit(
    bbox: tuple[int, int, int, int],
    entrance: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    orient: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    dr, dc = _entrance_dir(entrance, bbox)
    r, c = entrance
    while True:
        r += dr
        c += dc
        assert r0 < r < r1 and c0 < c < c1, "ray must hit the mirror before leaving the room"
        if (r, c) in mirror_cells:
            break

    dr, dc = REFLECT[orient][(dr, dc)]
    while True:
        r += dr
        c += dc
        if r in (r0, r1) or c in (c0, c1):
            return (r, c)
        assert r0 < r < r1 and c0 < c < c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each grey-framed room, trace the beam from the colored entrance to the
    diagonal mirror, reflect it by 90 degrees, and recolor the exit wall cell
    with the mirror color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rooms = _room_border_components(g)
    assert rooms, "expected at least one room"

    for room in rooms:
        bbox = room["bbox"]
        entrance = room["entrance"]
        orient, mirror_color, mirror_cells = _mirror_info(g, bbox)
        exit_r, exit_c = _trace_exit(bbox, entrance, mirror_cells, orient)
        assert int(out[exit_r, exit_c]) == GREY, "exit must be a grey border cell"
        out[exit_r, exit_c] = mirror_color
    return out


def _sample_room_spec(rng: random.Random, combo: tuple[str, tuple[int, int]]) -> dict:
    orient, incoming = combo
    room_h = rng.randint(*ROOM_HEIGHT_RANGE)
    room_w = rng.randint(*ROOM_WIDTH_RANGE)
    max_len = min(4, room_h - 2, room_w - 2)
    length = rng.randint(2, max_len)

    if orient == "\\":
        start_r = rng.randint(1, room_h - length - 1)
        start_c = rng.randint(1, room_w - length - 1)
        mirror = [(start_r + i, start_c + i) for i in range(length)]
    else:
        start_r = rng.randint(1, room_h - length - 1)
        start_c = rng.randint(length, room_w - 2)
        mirror = [(start_r + i, start_c - i) for i in range(length)]

    hit_r, hit_c = mirror[rng.randrange(length)]
    if incoming == (1, 0):
        entrance = (0, hit_c)
    elif incoming == (-1, 0):
        entrance = (room_h - 1, hit_c)
    elif incoming == (0, 1):
        entrance = (hit_r, 0)
    else:
        entrance = (hit_r, room_w - 1)

    mirror_color, entrance_color = rng.sample(PALETTE, 2)
    room = np.full((room_h, room_w), BG, dtype=int)
    room[0, :] = GREY
    room[-1, :] = GREY
    room[:, 0] = GREY
    room[:, -1] = GREY
    for r, c in mirror:
        room[r, c] = mirror_color
    room[entrance] = entrance_color
    return {"grid": room, "combo": combo}


def _can_place(used: np.ndarray, top: int, left: int, room_h: int, room_w: int) -> bool:
    r0 = max(0, top - ROOM_GAP)
    c0 = max(0, left - ROOM_GAP)
    r1 = min(used.shape[0], top + room_h + ROOM_GAP)
    c1 = min(used.shape[1], left + room_w + ROOM_GAP)
    return not used[r0:r1, c0:c1].any()


def _place_room(grid: np.ndarray, used: np.ndarray, room: np.ndarray, top: int, left: int) -> None:
    room_h, room_w = room.shape
    mask = room != BG
    sub = grid[top : top + room_h, left : left + room_w]
    sub[mask] = room[mask]
    used[top : top + room_h, left : left + room_w] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_SIZE_RANGE)
    grid_w = rng.randint(*GRID_SIZE_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    used = np.zeros((grid_h, grid_w), dtype=bool)

    combos = [(orient, direction) for orient in ("\\", "/") for direction in DIRS4]
    rng.shuffle(combos)
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    if room_count > len(combos):
        room_count = len(combos)
    rooms = [_sample_room_spec(rng, combos[i]) for i in range(room_count)]
    rng.shuffle(rooms)

    for room in rooms:
        room_grid = room["grid"]
        room_h, room_w = room_grid.shape
        if room_h > grid_h or room_w > grid_w:
            return None
        placed = False
        for _ in range(120):
            top = rng.randint(0, grid_h - room_h)
            left = rng.randint(0, grid_w - room_w)
            if _can_place(used, top, left, room_h, room_w):
                _place_room(grid, used, room_grid, top, left)
                placed = True
                break
        if not placed:
            return None

    parsed = _room_border_components(grid)
    if len(parsed) != room_count:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

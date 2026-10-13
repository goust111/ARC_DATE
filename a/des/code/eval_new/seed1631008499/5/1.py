import random
from typing import Iterable

import numpy as np


BG = 0
LOCK_COLORS = (1, 4, 6, 7, 8, 9)
KEY = 2
PAYLOAD = 3
WALL = 5

DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate90(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return _normalize((c, -r) for r, c in shape)


def _rotations(shape: Iterable[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    cur = _normalize(shape)
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rotate90(cur)
    return out


def _canonical(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_rotations(shape))


def _iter_perimeter(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _slide_until_blocked(
    out: np.ndarray,
    payload: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    h, w = out.shape
    cur = list(payload)
    while True:
        nxt = [(r + dr, c + dc) for r, c in cur]
        if any(not (0 <= nr < h and 0 <= nc < w) or int(out[nr, nc]) != BG for nr, nc in nxt):
            return cur
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the shape-matched door of each walled room and slide its green payload out through that door.

    Input structure:
    - bg is black, with grey rectangular room walls and exactly one colored lock pixel replacing one wall cell per room.
    - Each enclosed room interior contains one red connected key shape and one or more green payload pixels.
    - On the outer grid edge, each lock color also appears as a same-colored template component whose shape matches one room key up to rotation.

    Rule:
    - Find each enclosed room interior as a passable component of black/red/green cells that does not touch the grid border.
    - Extract the room's red key shape, match it against the edge templates under 90-degree rotations, and use the matched template color as the room's lock color.
    - Erase the perimeter lock pixel of that color, then translate all green pixels from that room together in the door's direction until the next step would hit the grid boundary or any non-black obstacle.
    - Leave the red key and all unrelated structures unchanged.

    Key insight:
    - The red key does not identify a direction directly; it identifies a color by shape matching against the edge templates, and that color tells you which door to open, whose wall position then supplies the movement direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    template_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    extra_colors = [int(c) for c in np.unique(g) if int(c) not in (BG, KEY, PAYLOAD, WALL)]
    for color in extra_colors:
        edge_comps = [comp for comp in _components(g == color) if _touches_border(comp, h, w)]
        if not edge_comps:
            continue
        template_shape = _canonical(edge_comps[0])
        template_by_shape[template_shape] = color

    passable = (g == BG) | (g == KEY) | (g == PAYLOAD)
    rooms = [comp for comp in _components(passable) if not _touches_border(comp, h, w) and any(int(g[r, c]) == KEY for r, c in comp)]

    for room in rooms:
        ir0, ic0, ir1, ic1 = _bbox(room)
        if ir0 == 0 or ic0 == 0 or ir1 == h - 1 or ic1 == w - 1:
            continue

        key_cells = [(r, c) for r, c in room if int(g[r, c]) == KEY]
        key_shape = _canonical(key_cells)
        if key_shape not in template_by_shape:
            continue
        lock_color = template_by_shape[key_shape]

        pr0, pc0, pr1, pc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        lock_cells = [(r, c) for r, c in _iter_perimeter(pr0, pc0, pr1, pc1) if int(g[r, c]) == lock_color]
        if len(lock_cells) != 1:
            continue
        lr, lc = lock_cells[0]

        if lr == pr0:
            dr, dc = DIR_VECTORS["up"]
        elif lr == pr1:
            dr, dc = DIR_VECTORS["down"]
        elif lc == pc0:
            dr, dc = DIR_VECTORS["left"]
        else:
            dr, dc = DIR_VECTORS["right"]

        payload = [(r, c) for r, c in room if int(g[r, c]) == PAYLOAD]
        if not payload:
            continue

        out[lr, lc] = BG
        for r, c in payload:
            out[r, c] = BG
        shifted = _slide_until_blocked(out, payload, dr, dc)
        for r, c in shifted:
            out[r, c] = PAYLOAD

    return out


def _mark_blocked(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    for r, c in cells:
        blocked[r, c] = True


def _expand_box(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> list[tuple[int, int]]:
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(h - 1, r1 + pad)
    ec1 = min(w - 1, c1 + pad)
    return [(r, c) for r in range(er0, er1 + 1) for c in range(ec0, ec1 + 1)]


def _choice_rot(rng: random.Random, shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return rng.choice(_rotations(shape))


def _room_band(
    direction: str,
    outer_h: int,
    outer_w: int,
    H: int,
    W: int,
) -> tuple[range, range] | None:
    if direction == "up":
        row_range = range(max(2, H // 2), H - outer_h - 1)
        col_range = range(1, W - outer_w)
    elif direction == "down":
        row_range = range(1, max(2, H // 2 - outer_h + 1))
        col_range = range(1, W - outer_w)
    elif direction == "left":
        row_range = range(1, H - outer_h)
        col_range = range(max(2, W // 2), W - outer_w - 1)
    else:
        row_range = range(1, H - outer_h)
        col_range = range(1, max(2, W // 2 - outer_w + 1))
    if not row_range or not col_range:
        return None
    return row_range, col_range


def _place_room(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    direction: str,
    shape: tuple[tuple[int, int], ...],
    lock_color: int,
) -> dict | None:
    H, W = grid.shape
    for _ in range(200):
        key_shape = _choice_rot(rng, shape)
        sh0, sw0, sh1, sw1 = _bbox(key_shape)
        key_h = sh1 - sh0 + 1
        key_w = sw1 - sw0 + 1
        inner_h = rng.randint(max(5, key_h + 1), max(5, min(7, key_h + 3)))
        inner_w = rng.randint(max(5, key_w + 1), max(5, min(7, key_w + 3)))
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        bands = _room_band(direction, outer_h, outer_w, H, W)
        if bands is None:
            continue
        row_range, col_range = bands
        r0 = rng.choice(list(row_range))
        c0 = rng.choice(list(col_range))
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        lane = rng.randrange(inner_w if direction in ("up", "down") else inner_h)
        if direction == "up":
            door = (r0, c0 + 1 + lane)
            corridor = [(r, door[1]) for r in range(0, r0)]
        elif direction == "down":
            door = (r1, c0 + 1 + lane)
            corridor = [(r, door[1]) for r in range(r1 + 1, H)]
        elif direction == "left":
            door = (r0 + 1 + lane, c0)
            corridor = [(door[0], c) for c in range(0, c0)]
        else:
            door = (r0 + 1 + lane, c1)
            corridor = [(door[0], c) for c in range(c1 + 1, W)]

        room_zone = _expand_box(r0, c0, r1, c1, H, W, pad=1)
        if any(blocked[r, c] for r, c in room_zone):
            continue
        if any(blocked[r, c] for r, c in corridor):
            continue

        key_cells_rel: list[tuple[int, int]] | None = None
        for _ in range(80):
            key_r = rng.randint(0, inner_h - key_h)
            key_c = rng.randint(0, inner_w - key_w)
            trial = [(key_r + r, key_c + c) for r, c in key_shape]
            if direction in ("left", "right"):
                if any(r == lane for r, _ in trial):
                    continue
            else:
                if any(c == lane for _, c in trial):
                    continue
            key_cells_rel = trial
            break
        if key_cells_rel is None:
            continue

        line_len = inner_w if direction in ("left", "right") else inner_h
        count = rng.randint(1, min(3, line_len))
        payload_idx = sorted(rng.sample(range(line_len), count))
        if direction == "left":
            payload_rel = [(lane, idx) for idx in payload_idx]
        elif direction == "right":
            payload_rel = [(lane, idx) for idx in payload_idx]
        elif direction == "up":
            payload_rel = [(idx, lane) for idx in payload_idx]
        else:
            payload_rel = [(idx, lane) for idx in payload_idx]

        key_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in key_cells_rel]
        payload_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in payload_rel]

        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[door] = lock_color
        for r, c in key_abs:
            grid[r, c] = KEY
        for r, c in payload_abs:
            grid[r, c] = PAYLOAD

        _mark_blocked(blocked, room_zone)
        _mark_blocked(blocked, corridor)
        return {
            "direction": direction,
            "lock_color": lock_color,
            "key_shape": _canonical(shape),
            "payload_abs": payload_abs,
            "door": door,
        }
    return None


def _place_template(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> bool:
    H, W = grid.shape
    for _ in range(300):
        oriented = _choice_rot(rng, shape)
        th0, tw0, th1, tw1 = _bbox(oriented)
        th = th1 - th0 + 1
        tw = tw1 - tw0 + 1
        edge = rng.choice(("top", "bottom", "left", "right"))
        if edge == "top":
            r0 = 0
            c0 = rng.randint(0, W - tw)
        elif edge == "bottom":
            r0 = H - th
            c0 = rng.randint(0, W - tw)
        elif edge == "left":
            r0 = rng.randint(0, H - th)
            c0 = 0
        else:
            r0 = rng.randint(0, H - th)
            c0 = W - tw

        cells = [(r0 + r, c0 + c) for r, c in oriented]
        zone = _expand_box(r0, c0, r0 + th - 1, c0 + tw - 1, H, W, pad=1)
        if any(blocked[r, c] for r, c in zone):
            continue
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, zone)
        return True
    return False


def _shift_to_edge(payload: list[tuple[int, int]], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    if direction == "up":
        delta = min(r for r, _ in payload)
        return [(r - delta, c) for r, c in payload]
    if direction == "down":
        delta = H - 1 - max(r for r, _ in payload)
        return [(r + delta, c) for r, c in payload]
    if direction == "left":
        delta = min(c for _, c in payload)
        return [(r, c - delta) for r, c in payload]
    delta = W - 1 - max(c for _, c in payload)
    return [(r, c + delta) for r, c in payload]


def _expected_output(inp: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        out[room["door"]] = BG
        for r, c in room["payload_abs"]:
            out[r, c] = BG
    for room in rooms:
        for r, c in _shift_to_edge(room["payload_abs"], room["direction"], inp.shape):
            out[r, c] = PAYLOAD
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(1, 3)
    if room_count == 1:
        H = rng.randint(20, 28)
        W = rng.randint(20, 28)
    elif room_count == 2:
        H = rng.randint(24, 30)
        W = rng.randint(24, 30)
    else:
        H = rng.randint(27, 30)
        W = rng.randint(27, 30)
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    directions = rng.sample(tuple(DIR_VECTORS), room_count)
    shapes = rng.sample(SHAPES, room_count)
    colors = rng.sample(LOCK_COLORS, room_count)

    rooms: list[dict] = []
    for direction, shape, color in zip(directions, shapes, colors):
        room = _place_room(rng, grid, blocked, direction, shape, color)
        if room is None:
            return None
        rooms.append(room)

    for room in rooms:
        if not _place_template(rng, grid, blocked, room["key_shape"], room["lock_color"]):
            return None

    expected = _expected_output(grid, rooms)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")

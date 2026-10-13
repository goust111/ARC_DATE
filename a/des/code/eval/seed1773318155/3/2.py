import random
from collections import deque

import numpy as np


BG = 0
RED = 2
WALL = 5
STRIPE_PALETTE = [1, 3, 4, 6, 7, 8, 9]

MAX_GRID_H = 30
MAX_GRID_W = 30
MAX_ROOM_INNER = 4
GEN_RETRIES = 400

DIM_OPTIONS = [
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]
DIM_WEIGHTS = [1] * 12 + [3, 3, 3]
SIDES = ("up", "down", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_stripe(grid: np.ndarray) -> list[int]:
    candidates = [
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    ]
    best_line = None
    best_score = -1
    for _name, line in candidates:
        score = int(sum(int(v) not in (BG, RED, WALL) for v in line))
        if score > best_score:
            best_score = score
            best_line = line
    if best_line is None or best_score <= 0:
        return []

    colors: list[int] = []
    prev = None
    for value in map(int, best_line):
        if value == prev:
            continue
        prev = value
        if value not in (BG, RED, WALL):
            colors.append(value)
    return colors


def _room_from_component(grid: np.ndarray, comp: list[tuple[int, int]]):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    cells = set(comp)
    if not cells.issubset(border):
        return None
    missing = list(border - cells)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        door = "up"
    elif gap_r == r1:
        door = "down"
    elif gap_c == c0:
        door = "left"
    else:
        door = "right"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    has_red = bool((inner == RED).any())
    return {
        "bbox": (r0, r1, c0, c1),
        "door": door,
        "has_red": has_red,
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components(grid == WALL):
        room = _room_from_component(grid, comp)
        if room is not None:
            rooms.append(room)
    return rooms


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _next_room_index(rooms: list[dict], idx: int):
    r0, r1, c0, c1 = rooms[idx]["bbox"]
    door = rooms[idx]["door"]
    best = None
    best_key = None
    for j, other in enumerate(rooms):
        if j == idx:
            continue
        or0, or1, oc0, oc1 = other["bbox"]
        if door == "right":
            gap = oc0 - c1
            align = _interval_overlap(r0 + 1, r1 - 1, or0 + 1, or1 - 1)
        elif door == "left":
            gap = c0 - oc1
            align = _interval_overlap(r0 + 1, r1 - 1, or0 + 1, or1 - 1)
        elif door == "down":
            gap = or0 - r1
            align = _interval_overlap(c0 + 1, c1 - 1, oc0 + 1, oc1 - 1)
        else:
            gap = r0 - or1
            align = _interval_overlap(c0 + 1, c1 - 1, oc0 + 1, oc1 - 1)
        if gap <= 0 or align <= 0:
            continue
        key = (gap, -align, or0, oc0)
        if best_key is None or key < best_key:
            best_key = key
            best = j
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the room path from each room's door direction, then fill rooms
    in stripe order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    stripe_colors = _read_stripe(g)
    rooms = _find_rooms(g)
    if not stripe_colors or not rooms:
        return out

    start = next((i for i, room in enumerate(rooms) if room["has_red"]), None)
    if start is None:
        return out

    path: list[int] = []
    seen: set[int] = set()
    cur = start
    while cur is not None and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = _next_room_index(rooms, cur)

    for room_idx, color in zip(path, stripe_colors):
        r0, r1, c0, c1 = rooms[room_idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(color)

    return out


def _sample_sizes(rng: random.Random, parts: int, limit: int) -> list[int] | None:
    hi = min(MAX_ROOM_INNER, limit)
    if limit < parts or hi < 1:
        return None
    total = rng.randint(parts, min(parts * hi, limit))
    sizes = []
    remain = total
    for i in range(parts):
        left = parts - i - 1
        lo = max(1, remain - left * hi)
        hi_i = min(hi, remain - left)
        if lo > hi_i:
            return None
        sizes.append(rng.randint(lo, hi_i))
        remain -= sizes[-1]
    rng.shuffle(sizes)
    return sizes


def _segment_lengths(rng: random.Random, total: int, parts: int) -> list[int]:
    remain = total
    lengths = []
    for i in range(parts):
        left = parts - i - 1
        lo = 1
        hi = remain - left
        lengths.append(rng.randint(lo, hi))
        remain -= lengths[-1]
    rng.shuffle(lengths)
    return lengths


def _row_snake(rows: int, cols: int) -> list[tuple[int, int]]:
    path = []
    for r in range(rows):
        cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cols_iter:
            path.append((r, c))
    return path


def _col_snake(rows: int, cols: int) -> list[tuple[int, int]]:
    path = []
    for c in range(cols):
        rows_iter = range(rows) if c % 2 == 0 else range(rows - 1, -1, -1)
        for r in rows_iter:
            path.append((r, c))
    return path


def _transform_path(
    path: list[tuple[int, int]],
    rows: int,
    cols: int,
    flip_v: bool,
    flip_h: bool,
) -> list[tuple[int, int]]:
    out = []
    for r, c in path:
        nr = rows - 1 - r if flip_v else r
        nc = cols - 1 - c if flip_h else c
        out.append((nr, nc))
    return out


def _door_from_step(a: tuple[int, int], b: tuple[int, int]) -> str:
    ar, ac = a
    br, bc = b
    if br == ar and bc == ac + 1:
        return "right"
    if br == ar and bc == ac - 1:
        return "left"
    if bc == ac and br == ar + 1:
        return "down"
    return "up"


def _outer_sides(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = cell
    sides = []
    if r == 0:
        sides.append("up")
    if r == rows - 1:
        sides.append("down")
    if c == 0:
        sides.append("left")
    if c == cols - 1:
        sides.append("right")
    return sides


def _room_positions(
    row_heights: list[int],
    col_widths: list[int],
    gap_y: int,
    gap_x: int,
    top: int,
    left: int,
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes = {}
    y = top
    for r, inner_h in enumerate(row_heights):
        x = left
        for c, inner_w in enumerate(col_widths):
            boxes[(r, c)] = (y, y + inner_h + 1, x, x + inner_w + 1)
            x += inner_w + 2 + gap_x
        y += inner_h + 2 + gap_y
    return boxes


def _paint_room(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    door: str,
    rng: random.Random,
) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL

    if door in ("up", "down"):
        gap_c = rng.randint(c0 + 1, c1 - 1)
        gap_r = r0 if door == "up" else r1
    else:
        gap_r = rng.randint(r0 + 1, r1 - 1)
        gap_c = c0 if door == "left" else c1
    grid[gap_r, gap_c] = BG


def _paint_stripe(
    grid: np.ndarray,
    side: str,
    colors: list[int],
    rng: random.Random,
) -> None:
    if side in ("top", "bottom"):
        lengths = _segment_lengths(rng, grid.shape[1], len(colors))
        row = 0 if side == "top" else grid.shape[0] - 1
        c = 0
        for color, length in zip(colors, lengths):
            grid[row, c : c + length] = int(color)
            c += length
    else:
        lengths = _segment_lengths(rng, grid.shape[0], len(colors))
        col = 0 if side == "left" else grid.shape[1] - 1
        r = 0
        for color, length in zip(colors, lengths):
            grid[r : r + length, col] = int(color)
            r += length


def _generate_case(rng: random.Random):
    rows, cols = rng.choices(DIM_OPTIONS, weights=DIM_WEIGHTS, k=1)[0]
    room_count = rows * cols
    stripe_side = rng.choice(("top", "bottom", "left", "right"))

    gap_y = 1 if rows >= 4 else rng.randint(1, 2)
    gap_x = 1 if cols >= 4 else rng.randint(1, 2)

    base_top = 2 if stripe_side == "top" else 1
    base_bottom = 2 if stripe_side == "bottom" else 1
    base_left = 2 if stripe_side == "left" else 1
    base_right = 2 if stripe_side == "right" else 1

    fixed_h = rows * 2 + (rows - 1) * gap_y + base_top + base_bottom
    fixed_w = cols * 2 + (cols - 1) * gap_x + base_left + base_right
    room_heights = _sample_sizes(rng, rows, MAX_GRID_H - fixed_h)
    room_widths = _sample_sizes(rng, cols, MAX_GRID_W - fixed_w)
    if room_heights is None or room_widths is None:
        return None

    arr_h = sum(room_heights) + rows * 2 + (rows - 1) * gap_y
    arr_w = sum(room_widths) + cols * 2 + (cols - 1) * gap_x

    min_h = base_top + arr_h + base_bottom
    min_w = base_left + arr_w + base_right
    if min_h > MAX_GRID_H or min_w > MAX_GRID_W:
        return None

    extra_h = rng.randint(0, min(6, MAX_GRID_H - min_h))
    extra_w = rng.randint(0, min(6, MAX_GRID_W - min_w))
    extra_top = rng.randint(0, extra_h)
    extra_left = rng.randint(0, extra_w)
    extra_bottom = extra_h - extra_top
    extra_right = extra_w - extra_left

    top = base_top + extra_top
    left = base_left + extra_left
    height = min_h + extra_h
    width = min_w + extra_w
    stripe_len = width if stripe_side in ("top", "bottom") else height
    if stripe_len < room_count:
        return None

    path = _row_snake(rows, cols) if rng.random() < 0.5 else _col_snake(rows, cols)
    path = _transform_path(path, rows, cols, flip_v=bool(rng.getrandbits(1)), flip_h=bool(rng.getrandbits(1)))

    doors = {}
    for i, cell in enumerate(path):
        if i + 1 < len(path):
            doors[cell] = _door_from_step(cell, path[i + 1])
        else:
            doors[cell] = rng.choice(_outer_sides(cell, rows, cols))

    stripe_colors = rng.sample(STRIPE_PALETTE, room_count)
    boxes = _room_positions(room_heights, room_widths, gap_y, gap_x, top, left)

    inp = np.full((height, width), BG, dtype=int)
    _paint_stripe(inp, stripe_side, stripe_colors, rng)

    for cell in boxes:
        _paint_room(inp, boxes[cell], doors[cell], rng)

    sr0, sr1, sc0, sc1 = boxes[path[0]]
    red_r = rng.randint(sr0 + 1, sr1 - 1)
    red_c = rng.randint(sc0 + 1, sc1 - 1)
    inp[red_r, red_c] = RED

    out = inp.copy()
    for color, cell in zip(stripe_colors, path):
        r0, r1, c0, c1 = boxes[cell]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(color)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

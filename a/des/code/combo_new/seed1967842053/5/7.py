import random

import numpy as np


BG = 0
WALL = 5

ARROWS = {
    "up": ((-1, 0), ((1, -1), (1, 1))),
    "down": ((1, 0), ((-1, -1), (-1, 1))),
    "left": ((0, -1), ((-1, 1), (1, 1))),
    "right": ((0, 1), ((-1, -1), (1, -1))),
}

COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(r, c, direction):
    _, wings = ARROWS[direction]
    return [(r, c), (r + wings[0][0], c + wings[0][1]), (r + wings[1][0], c + wings[1][1])]


def _isolated(grid, cells, color):
    own = set(cells)
    h, w = grid.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) not in own and grid[rr, cc] == color:
                    return False
    return True


def _find_arrows(grid):
    h, w = grid.shape
    arrows = []
    used = set()
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, WALL):
                continue
            for direction in ARROWS:
                cells = _arrow_cells(r, c, direction)
                key = frozenset(cells)
                if key in used:
                    continue
                if all(_in_bounds(grid.shape, rr, cc) and grid[rr, cc] == color for rr, cc in cells):
                    if _isolated(grid, cells, color):
                        arrows.append((color, r, c, direction, cells))
                        used.add(key)
    return arrows


def _first_wall(grid, r, c, direction):
    (dr, dc), _ = ARROWS[direction]
    rr, cc = r + dr, c + dc
    while _in_bounds(grid.shape, rr, cc):
        if grid[rr, cc] == WALL:
            return rr, cc
        rr += dr
        cc += dc
    return None


def _line_offsets(length):
    return range(-((length - 1) // 2), length // 2 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Project colored arrow markers onto the grey walls they point at.

    Input structure:
    - The background is black, with one or more straight grey wall segments that are horizontal or vertical.
    - Each colored marker is a three-pixel arrow: a protruding head and two symmetric rear wings.
    - Every arrow has its own color, and same-colored single pixels between that arrow and its target wall are ammo.
    Rule:
    - For each arrow, trace from its head in the pointing direction to the nearest grey wall pixel.
    - Count all same-colored non-arrow pixels as the ammo count N, erase the arrow and ammo, and keep the walls.
    - Recolor N pixels of the hit wall with the arrow color, centered on the hit point along the wall direction;
      even lengths include the hit point and extend one extra cell toward the positive row or column side.
    Key insight:
    - The arrow direction selects the wall and hit point, while the count of matching loose pixels becomes the
      length of the colored projection drawn along that wall.
    """
    src = np.asarray(grid)
    out = src.copy()
    arrows = _find_arrows(src)

    for color, _, _, _, _ in arrows:
        out[src == color] = BG

    for color, r, c, direction, _ in arrows:
        hit = _first_wall(src, r, c, direction)
        if hit is None:
            continue
        length = int(np.count_nonzero(src == color)) - 3
        if length <= 0:
            continue

        if direction in ("up", "down"):
            axis = (0, 1)
        else:
            axis = (1, 0)

        for offset in _line_offsets(length):
            rr = hit[0] + axis[0] * offset
            cc = hit[1] + axis[1] * offset
            if _in_bounds(src.shape, rr, cc) and src[rr, cc] == WALL:
                out[rr, cc] = color

    return out


def _projection_cells(hit, direction, length):
    axis = (0, 1) if direction in ("up", "down") else (1, 0)
    return [(hit[0] + axis[0] * off, hit[1] + axis[1] * off) for off in _line_offsets(length)]


def _between_ray_cells(head, hit):
    r, c = head
    hr, hc = hit
    dr = (hr > r) - (hr < r)
    dc = (hc > c) - (hc < c)
    cells = []
    r += dr
    c += dc
    while (r, c) != (hr, hc):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _empty(grid, cells):
    return all(_in_bounds(grid.shape, r, c) and grid[r, c] == BG for r, c in cells)


def _far_from(cells, forbidden, distance=1):
    for r, c in cells:
        for rr, cc in forbidden:
            if max(abs(r - rr), abs(c - cc)) <= distance:
                return False
    return True


def _make_candidate(rng, h, w, color):
    direction = rng.choice(tuple(ARROWS))
    length = rng.randint(1, 6)
    left_need = (length - 1) // 2
    right_need = length // 2

    if direction in ("up", "down"):
        if h < 11 or w < left_need + right_need + 5:
            return None
        extra_left = rng.randint(1, 3)
        extra_right = rng.randint(1, 3)
        min_c = max(1, left_need + extra_left)
        max_c = min(w - 2, w - 1 - right_need - extra_right)
        if min_c > max_c:
            return None
        hit_c = rng.randint(min_c, max_c)
        wall_start = hit_c - left_need - extra_left
        wall_end = hit_c + right_need + extra_right
        max_gap = min(10, h - 3)
        if max_gap < 5:
            return None
        gap = rng.randint(5, max_gap)

        if direction == "up":
            max_wall_r = h - gap - 2
            if max_wall_r < 0:
                return None
            wall_r = rng.randint(0, max_wall_r)
            head = (wall_r + gap, hit_c)
        else:
            min_wall_r = gap + 1
            if min_wall_r >= h:
                return None
            wall_r = rng.randint(min_wall_r, h - 1)
            head = (wall_r - gap, hit_c)

        hit = (wall_r, hit_c)
        wall = [(wall_r, c) for c in range(wall_start, wall_end + 1)]
        lane_cols = range(max(wall_start, hit_c - 2), min(wall_end, hit_c + 2) + 1)
        lo = min(head[0], wall_r) + 1
        hi = max(head[0], wall_r)
        ammo_pool = [(r, c) for r in range(lo, hi) for c in lane_cols]
    else:
        if w < 11 or h < left_need + right_need + 5:
            return None
        extra_left = rng.randint(1, 3)
        extra_right = rng.randint(1, 3)
        min_r = max(1, left_need + extra_left)
        max_r = min(h - 2, h - 1 - right_need - extra_right)
        if min_r > max_r:
            return None
        hit_r = rng.randint(min_r, max_r)
        wall_start = hit_r - left_need - extra_left
        wall_end = hit_r + right_need + extra_right
        max_gap = min(10, w - 3)
        if max_gap < 5:
            return None
        gap = rng.randint(5, max_gap)

        if direction == "left":
            max_wall_c = w - gap - 2
            if max_wall_c < 0:
                return None
            wall_c = rng.randint(0, max_wall_c)
            head = (hit_r, wall_c + gap)
        else:
            min_wall_c = gap + 1
            if min_wall_c >= w:
                return None
            wall_c = rng.randint(min_wall_c, w - 1)
            head = (hit_r, wall_c - gap)

        hit = (hit_r, wall_c)
        wall = [(r, wall_c) for r in range(wall_start, wall_end + 1)]
        lane_rows = range(max(wall_start, hit_r - 2), min(wall_end, hit_r + 2) + 1)
        lo = min(head[1], wall_c) + 1
        hi = max(head[1], wall_c)
        ammo_pool = [(r, c) for r in lane_rows for c in range(lo, hi)]

    arrow = _arrow_cells(head[0], head[1], direction)
    blocked = set(arrow) | set(wall)
    ammo_pool = [p for p in ammo_pool if p not in blocked and _far_from([p], arrow, 1)]
    rng.shuffle(ammo_pool)
    if len(ammo_pool) < length:
        return None

    return {
        "color": color,
        "direction": direction,
        "length": length,
        "head": head,
        "hit": hit,
        "arrow": arrow,
        "wall": wall,
        "ammo": ammo_pool[:length],
    }


def _new_wall_blocks_old_arrow(candidate, metas):
    new_wall = set(candidate["wall"])
    for meta in metas:
        if new_wall.intersection(_between_ray_cells(meta["head"], meta["hit"])):
            return True
    return False


def _candidate_fits(grid, candidate, metas):
    if candidate is None:
        return False
    occupied = candidate["arrow"] + candidate["wall"] + candidate["ammo"]
    if not _empty(grid, occupied):
        return False
    if _new_wall_blocks_old_arrow(candidate, metas):
        return False
    for cell in _between_ray_cells(candidate["head"], candidate["hit"]):
        if grid[cell] == WALL:
            return False
    for r, c in _projection_cells(candidate["hit"], candidate["direction"], candidate["length"]):
        if not _in_bounds(grid.shape, r, c) or (r, c) not in set(candidate["wall"]):
            return False
    return True


def _place(grid, candidate):
    for r, c in candidate["wall"]:
        grid[r, c] = WALL
    for r, c in candidate["arrow"] + candidate["ammo"]:
        grid[r, c] = candidate["color"]


def _valid_generated(grid, metas):
    arrows = _find_arrows(grid)
    if len(arrows) != len(metas):
        return False
    by_color = {m["color"]: m for m in metas}
    if len(by_color) != len(metas):
        return False
    for color, r, c, direction, _ in arrows:
        meta = by_color.get(color)
        if meta is None:
            return False
        if meta["head"] != (r, c) or meta["direction"] != direction:
            return False
        if _first_wall(grid, r, c, direction) != meta["hit"]:
            return False
        if int(np.count_nonzero(grid == color)) - 3 != meta["length"]:
            return False
        for rr, cc in _projection_cells(meta["hit"], direction, meta["length"]):
            if not _in_bounds(grid.shape, rr, cc) or grid[rr, cc] != WALL:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
        grid = np.zeros((h, w), dtype=np.int8)
        count = rng.randint(1, 4)
        colors = rng.sample(COLORS, count)
        metas = []

        ok = True
        for color in colors:
            placed = False
            for _ in range(250):
                candidate = _make_candidate(rng, h, w, color)
                if _candidate_fits(grid, candidate, metas):
                    _place(grid, candidate)
                    metas.append(candidate)
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok and _valid_generated(grid, metas):
            return grid

    return np.zeros((16, 16), dtype=np.int8)

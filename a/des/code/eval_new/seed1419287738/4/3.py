import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
SIGNAL_COLORS = [RED, GREEN, BLUE, YELLOW]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _grey_receivers(grid):
    h, w = grid.shape
    seen = set()
    receivers = []
    for r, c in zip(*np.where(grid == GREY)):
        if (r, c) in seen:
            continue
        q = deque([(int(r), int(c))])
        seen.add((int(r), int(c)))
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if (nr, nc) not in seen and grid[nr, nc] == GREY:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        if len(cells) == 3:
            rows = {x for x, _ in cells}
            cols = {y for _, y in cells}
            if len(rows) == 1 or len(cols) == 1:
                cells = sorted(cells)
                mid = cells[1]
                rid = len(receivers)
                receivers.append({"cells": set(cells), "mid": mid})
    cell_to_receiver = {}
    for i, rec in enumerate(receivers):
        for cell in rec["cells"]:
            cell_to_receiver[cell] = i
    return receivers, cell_to_receiver


def _towers(grid):
    h, w = grid.shape
    towers = []
    seen = set()
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r:r + 2, c:c + 2]
            if np.count_nonzero(block == AZURE) != 3:
                continue
            signal_cells = []
            for rr in range(2):
                for cc in range(2):
                    val = int(block[rr, cc])
                    if val in SIGNAL_COLORS:
                        signal_cells.append((rr, cc, val))
            if len(signal_cells) != 1:
                continue
            key = (r, c)
            if key in seen:
                continue
            sr, sc, color = signal_cells[0]
            dirs = []
            dirs.append((-1, 0) if sr == 0 else (1, 0))
            dirs.append((0, -1) if sc == 0 else (0, 1))
            towers.append({"signal": (r + sr, c + sc), "color": color, "dirs": dirs})
            seen.add(key)
    return towers


def _filter_color(color, filt):
    if filt == ORANGE:
        return {RED: YELLOW, BLUE: GREEN}.get(color, color)
    if filt == MAGENTA:
        return {YELLOW: RED, GREEN: BLUE}.get(color, color)
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """Cast filtered signal rays from azure towers into grey receivers.

    Input structure:
    The black grid contains 2x2 azure signal towers, isolated orange or magenta
    filter pixels, and grey 1x3 or 3x1 receiver bars.  One colored pixel inside
    each tower marks both the signal color and the pair of orthogonal directions.

    Rule:
    Each tower emits two straight rays from its colored cell.  Rays paint black
    cells in their current signal color, pass through filters while changing
    color, and stop at the first non-black non-filter object.  When a ray touches
    a grey receiver, the receiver's center pixel is colored by the arriving
    signal.  If several colors reach the same receiver, choose the color with the
    fewest total receiver hits in the whole grid.

    Key insight:
    The tower corner encodes direction, filters update the ray color in sequence,
    and receiver conflicts are resolved by a global least-received color count.
    """
    out = np.array(grid, copy=True)
    receivers, receiver_by_cell = _grey_receivers(grid)
    hits = defaultdict(list)

    for tower in _towers(grid):
        sr, sc = tower["signal"]
        for dr, dc in tower["dirs"]:
            color = tower["color"]
            r, c = sr + dr, sc + dc
            while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                val = int(grid[r, c])
                if val in (ORANGE, MAGENTA):
                    color = _filter_color(color, val)
                    r += dr
                    c += dc
                    continue
                if (r, c) in receiver_by_cell:
                    hits[receiver_by_cell[(r, c)]].append(color)
                    break
                if val != BG:
                    break
                out[r, c] = color
                r += dr
                c += dc

    global_counts = defaultdict(int)
    for colors in hits.values():
        for color in colors:
            global_counts[color] += 1
    for rid, colors in hits.items():
        if not colors:
            continue
        chosen = min(colors, key=lambda color: (global_counts[color], color))
        mr, mc = receivers[rid]["mid"]
        out[mr, mc] = chosen
    return out


def _place_tower(grid, top, left, signal_pos, color):
    cells = [(top + r, left + c) for r in range(2) for c in range(2)]
    if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[r, c] != BG for r, c in cells):
        return False
    for r, c in cells:
        grid[r, c] = AZURE
    sr, sc = signal_pos
    grid[top + sr, left + sc] = color
    return True


def _place_receiver(grid, orientation, center):
    r, c = center
    cells = [(r, c - 1), (r, c), (r, c + 1)] if orientation == "h" else [(r - 1, c), (r, c), (r + 1, c)]
    if any(not (1 <= rr < grid.shape[0] - 1 and 1 <= cc < grid.shape[1] - 1) or grid[rr, cc] != BG for rr, cc in cells):
        return None
    for rr, cc in cells:
        grid[rr, cc] = GREY
    return cells


def _place_filter(grid, r, c, color):
    if 1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1 and grid[r, c] == BG:
        grid[r, c] = color
        return True
    return False


def _single_tower_case():
    h = random.randint(16, 26)
    w = random.randint(16, 26)
    grid = np.zeros((h, w), dtype=int)
    top = random.randint(6, h - 8)
    left = random.randint(6, w - 8)
    signal_pos = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    color = random.choice(SIGNAL_COLORS)
    if not _place_tower(grid, top, left, signal_pos, color):
        return None
    sr, sc = top + signal_pos[0], left + signal_pos[1]
    dirs = [((-1, 0) if signal_pos[0] == 0 else (1, 0)), ((0, -1) if signal_pos[1] == 0 else (0, 1))]
    for dr, dc in dirs:
        dist = random.randint(4, 7)
        rr, cc = sr + dr * dist, sc + dc * dist
        orientation = "h" if dr else "v"
        if _place_receiver(grid, orientation, (rr, cc)) is None:
            return None
        if random.random() < 0.8:
            fd = random.randint(2, dist - 2)
            _place_filter(grid, sr + dr * fd, sc + dc * fd, random.choice([ORANGE, MAGENTA]))
    return grid


def _multi_hit_case():
    h = random.randint(17, 25)
    w = random.randint(19, 27)
    grid = np.zeros((h, w), dtype=int)
    row = random.randint(6, h - 7)
    col = random.randint(9, w - 10)
    if _place_receiver(grid, "h", (row, col)) is None:
        return None
    left_tower_left = col - random.randint(7, 9)
    right_tower_left = col + random.randint(7, 9)
    if not _place_tower(grid, row, left_tower_left, (0, 1), random.choice(SIGNAL_COLORS)):
        return None
    if not _place_tower(grid, row, right_tower_left, (0, 0), random.choice(SIGNAL_COLORS)):
        return None
    _place_filter(grid, row, left_tower_left + random.randint(3, 5), random.choice([ORANGE, MAGENTA]))
    _place_filter(grid, row, right_tower_left - random.randint(2, 4), random.choice([ORANGE, MAGENTA]))
    extra_col = random.randint(4, w - 5)
    if _place_receiver(grid, "v", (random.randint(4, h - 5), extra_col)) is None:
        pass
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _multi_hit_case() if random.random() < 0.35 else _single_tower_case()
        if grid is None:
            continue
        out = main(grid)
        if not np.array_equal(out, grid):
            return grid
    raise RuntimeError("failed to generate a valid filtered-signal puzzle")

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5


def _grey_components(grid):
    seen = np.zeros(grid.shape, dtype=bool)
    rows, cols = grid.shape
    for sr, sc in np.argwhere(grid == GREY):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == GREY:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        yield cells


def _is_rect_frame(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if set(cells) != expected:
        return None
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored concentric inner borders in each grey room.

    Input structure:
    The grid has a black background with several separate one-pixel-wide grey
    rectangular frames.  Each frame encloses a room whose interior contains one
    to four single-pixel seeds; all seeds inside one room have the same non-grey
    color.

    Rule:
    Keep every grey frame unchanged.  For each room, count its interior seed
    pixels N and use the seed color to draw N one-pixel-thick concentric
    rectangular rings starting from the inner wall and moving inward.  The
    original seed pixels are cleared first, so only the ring pattern remains.
    If the room is too small for N complete layers, fill as many inward layers
    as fit, which may fill the entire interior.

    Key insight:
    The seed color is the paint color, while the number of seeds encodes how
    many inward border layers the room should receive.
    """
    out = np.array(grid, copy=True)
    for cells in _grey_components(grid):
        box = _is_rect_frame(cells)
        if box is None:
            continue
        r0, r1, c0, c1 = box
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BLACK) & (interior != GREY)
        seed_count = int(seed_mask.sum())
        if seed_count == 0:
            continue
        seed_values = interior[seed_mask]
        color = int(seed_values[0])

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        max_layers = (min(inner_h, inner_w) + 1) // 2
        layers = min(seed_count, max_layers)
        for layer in range(layers):
            top = r0 + 1 + layer
            bottom = r1 - 1 - layer
            left = c0 + 1 + layer
            right = c1 - 1 - layer
            if top > bottom or left > right:
                break
            out[top, left : right + 1] = color
            out[bottom, left : right + 1] = color
            out[top : bottom + 1, left] = color
            out[top : bottom + 1, right] = color
    return out


def _can_place(occupied, r0, r1, c0, c1):
    rows, cols = occupied.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(rows - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(cols - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _draw_frame(grid, r0, r1, c0, c1):
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    grid = np.zeros((rows, cols), dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    target_rooms = random.randint(2, 5)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    made = 0
    attempts = 0
    while made < target_rooms and attempts < 300:
        attempts += 1
        if random.random() < 0.35:
            inner_h = random.choice([1, 2])
            inner_w = random.randint(max(4, inner_h), 9)
            if random.random() < 0.5:
                inner_h, inner_w = inner_w, inner_h
        else:
            inner_h = random.randint(2, 7)
            inner_w = random.randint(2, 7)

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        if frame_h > rows - 2 or frame_w > cols - 2:
            continue
        r0 = random.randint(1, rows - frame_h - 1)
        c0 = random.randint(1, cols - frame_w - 1)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        if not _can_place(occupied, r0, r1, c0, c1):
            continue

        capacity = inner_h * inner_w
        max_seeds = min(4, capacity)
        seed_count = random.randint(1, max_seeds)
        if max_seeds >= 4 and random.random() < 0.25:
            seed_count = 4
        color = random.choice(palette)

        _draw_frame(grid, r0, r1, c0, c1)
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        positions = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in random.sample(positions, seed_count):
            grid[r, c] = color
        made += 1

    if made == 0:
        r0, c0, inner_h, inner_w = 2, 2, 4, 4
        r1, c1 = r0 + inner_h + 1, c0 + inner_w + 1
        _draw_frame(grid, r0, r1, c0, c1)
        color = random.choice(palette)
        for r, c in random.sample([(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)], 2):
            grid[r, c] = color

    return grid

import random
import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}


def _arrow_offsets(direction):
    dr, dc = DIRS[direction]
    if direction in ("R", "L"):
        return [(0, 0), (-1, -dc), (0, -dc), (1, -dc), (0, -2 * dc)]
    return [(0, 0), (-dr, -1), (-dr, 0), (-dr, 1), (-2 * dr, 0)]


def _arrow_cells(tip, direction):
    tr, tc = tip
    return [(tr + r, tc + c) for r, c in _arrow_offsets(direction)]


def _centerline_cells(tip, direction):
    tr, tc = tip
    dr, dc = DIRS[direction]
    return [(tr, tc), (tr - dr, tc - dc), (tr - 2 * dr, tc - 2 * dc)]


def _trail_line(old_tip, new_tip, direction):
    dr, dc = DIRS[direction]
    p = (old_tip[0] - 2 * dr, old_tip[1] - 2 * dc)
    cells = []
    while True:
        cells.append(p)
        if p == new_tip:
            return cells
        p = (p[0] + dr, p[1] + dc)


def _fits(rows, cols, tip, direction):
    return all(0 <= r < rows and 0 <= c < cols for r, c in _arrow_cells(tip, direction))


def _orientation_from_cells(cells):
    s = set(cells)
    for direction in DIRS:
        for tip in cells:
            if set(_arrow_cells(tip, direction)) == s:
                return tip, direction
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored arrow toward its matching target and paint its center trail grey.

    Input structure:
    The grid has a black background, several five-pixel colored arrows, and one
    single-pixel target of the same color for each arrow.  Each arrow is made of
    a tip, a two-pixel center shaft behind the tip, and two symmetric wing
    pixels beside the shaft pixel next to the tip.  Arrows point up, down, left,
    or right, and their same-colored target lies straight ahead in the same row
    or column.

    Rule:
    For every color, identify the arrow and its isolated target.  Translate the
    arrow in its pointing direction until the tip is immediately before the
    target.  Draw the moved arrow in its original color, keep the target pixel
    unchanged, and fill only the swept centerline cells between the old and new
    arrow positions with grey.

    Key insight:
    The task requires recognizing the five-pixel arrow orientation, pairing each
    arrow with the same-colored point, and applying a straight-line motion rule
    whose visible residue is the arrow shaft's center trail.
    """
    grid = np.array(grid, copy=True)
    rows, cols = grid.shape
    out = np.zeros_like(grid)

    for color in [int(x) for x in np.unique(grid) if x not in (BLACK, GREY)]:
        cells = list(zip(*np.where(grid == color)))
        if len(cells) < 6:
            continue

        arrow_tip = None
        arrow_dir = None
        arrow_set = None
        for combo in _combinations5(cells):
            tip, direction = _orientation_from_cells(combo)
            if direction is not None:
                remaining = [p for p in cells if p not in combo]
                if len(remaining) == 1:
                    tr, tc = tip
                    rr, cc = remaining[0]
                    if (direction in ("R", "L") and rr == tr) or (
                        direction in ("U", "D") and cc == tc
                    ):
                        dr, dc = DIRS[direction]
                        if (rr - tr) * dr + (cc - tc) * dc > 0:
                            arrow_tip = tip
                            arrow_dir = direction
                            arrow_set = set(combo)
                            break
        if arrow_dir is None:
            for r, c in cells:
                out[r, c] = color
            continue

        target = [p for p in cells if p not in arrow_set][0]
        dr, dc = DIRS[arrow_dir]
        new_tip = (target[0] - dr, target[1] - dc)
        new_arrow = set(_arrow_cells(new_tip, arrow_dir))

        for p in _trail_line(arrow_tip, new_tip, arrow_dir):
            if p not in new_arrow and p != target:
                out[p] = GREY

        for r, c in new_arrow:
            out[r, c] = color
        out[target] = color

    return out


def _combinations5(items):
    n = len(items)
    for a in range(n - 4):
        for b in range(a + 1, n - 3):
            for c in range(b + 1, n - 2):
                for d in range(c + 1, n - 1):
                    for e in range(d + 1, n):
                        yield [items[a], items[b], items[c], items[d], items[e]]


def _paint(grid, cells, color):
    for r, c in cells:
        grid[r, c] = color


def _clear(grid, cells):
    return all(grid[r, c] == BLACK for r, c in cells)


def _candidate(rows, cols, direction):
    dr, dc = DIRS[direction]
    if direction == "R":
        r = random.randint(1, rows - 2)
        tip_c = random.randint(2, cols - 4)
        gap = random.randint(2, min(8, cols - tip_c - 2))
        return (r, tip_c), (r, tip_c + gap)
    if direction == "L":
        r = random.randint(1, rows - 2)
        tip_c = random.randint(3, cols - 3)
        gap = random.randint(2, min(8, tip_c - 1))
        return (r, tip_c), (r, tip_c - gap)
    if direction == "D":
        c = random.randint(1, cols - 2)
        tip_r = random.randint(2, rows - 4)
        gap = random.randint(2, min(8, rows - tip_r - 2))
        return (tip_r, c), (tip_r + gap, c)
    c = random.randint(1, cols - 2)
    tip_r = random.randint(3, rows - 3)
    gap = random.randint(2, min(8, tip_r - 1))
    return (tip_r, c), (tip_r - gap, c)


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    grid = np.zeros((rows, cols), dtype=int)
    count = random.randint(2, min(5, len(COLORS)))
    colors = random.sample(COLORS, count)
    directions = ["R", "L", "D", "U"]
    random.shuffle(directions)
    directions = [directions[i % 4] for i in range(count)]

    placed = 0
    for color, direction in zip(colors, directions):
        for _ in range(300):
            tip, target = _candidate(rows, cols, direction)
            arrow = _arrow_cells(tip, direction)
            dr, dc = DIRS[direction]
            new_tip = (target[0] - dr, target[1] - dc)
            new_arrow = _arrow_cells(new_tip, direction)
            buffer = set(arrow + new_arrow + _trail_line(tip, new_tip, direction) + [target])
            for r, c in list(buffer):
                for nr in range(max(0, r - 1), min(rows, r + 2)):
                    for nc in range(max(0, c - 1), min(cols, c + 2)):
                        buffer.add((nr, nc))
            if _fits(rows, cols, tip, direction) and _fits(rows, cols, new_tip, direction):
                if all(0 <= r < rows and 0 <= c < cols for r, c in buffer):
                    if _clear(grid, buffer):
                        _paint(grid, arrow, color)
                        grid[target] = color
                        placed += 1
                        break

    if placed < 2:
        return generate_input()
    return grid

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
ARROW_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _blue_components(grid):
    h, w = grid.shape
    comp_id = -np.ones((h, w), dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLUE or comp_id[r, c] >= 0:
                continue
            idx = len(comps)
            pixels = []
            q = deque([(r, c)])
            comp_id[r, c] = idx
            while q:
                cr, cc = q.popleft()
                pixels.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if grid[nr, nc] == BLUE and comp_id[nr, nc] < 0:
                        comp_id[nr, nc] = idx
                        q.append((nr, nc))
            comps.append(pixels)
    return comp_id, comps


def _perpendicular(dr, dc):
    if dr == 0:
        return 1, 0
    return 0, 1


def _infer_arrow(component):
    cells = set(component)
    for r, c in cells:
        for dr, dc in DIRECTIONS:
            pr, pc = _perpendicular(dr, dc)
            head = {(r, c), (r + pr, c + pc), (r - pr, c - pc)}
            if not head.issubset(cells):
                continue
            tail = []
            k = 1
            while (r - k * dr, c - k * dc) in cells:
                tail.append((r - k * dr, c - k * dc))
                k += 1
            if len(tail) < 2:
                continue
            if head | set(tail) == cells:
                return (r, c), (dr, dc)
    return None


def _arrow_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    arrows = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, BLUE) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            component = []
            while q:
                cr, cc = q.popleft()
                component.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arrow = _infer_arrow(component)
            if arrow is not None:
                tip, direction = arrow
                arrows.append((color, tip, direction))
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each blue rectangle target with the color of the arrow pointing at it.

    Input structure:
    A black background contains several filled blue rectangles, each 2x2 or 3x3, and several small one-color arrow components. Each arrow color is distinct from black and blue. An arrow is a one-pixel-wide horizontal or vertical tail of length at least two plus a three-pixel head: the front tip pixel and the two pixels immediately perpendicular to it.

    Rule:
    Preserve every arrow pixel. For each detected arrow, cast a ray from the tip in the arrow direction. The first blue connected rectangle touched by that ray is recolored entirely to the arrow color. Blue rectangles not reached by any arrow remain blue.

    Key insight:
    The arrow direction is encoded by local geometry, not by color identity: identify the head-and-tail pattern, then use the resulting direction as a visible pointer from arrow to target rectangle.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    h, w = arr.shape
    comp_id, blue_comps = _blue_components(arr)

    for color, tip, (dr, dc) in _arrow_components(arr):
        r, c = tip[0] + dr, tip[1] + dc
        while 0 <= r < h and 0 <= c < w:
            idx = comp_id[r, c]
            if idx >= 0:
                for pr, pc in blue_comps[int(idx)]:
                    out[pr, pc] = color
                break
            r += dr
            c += dc
    return out


def _arrow_cells(tip, direction, tail_len):
    r, c = tip
    dr, dc = direction
    pr, pc = _perpendicular(dr, dc)
    cells = {(r, c), (r + pr, c + pc), (r - pr, c - pc)}
    for k in range(1, tail_len + 1):
        cells.add((r - k * dr, c - k * dc))
    return cells


def _rect_cells(top, left, size):
    return {(r, c) for r in range(top, top + size) for c in range(left, left + size)}


def _in_bounds(cells, h, w):
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _fits_with_margin(cells, occupied, h, w):
    if not _in_bounds(cells, h, w):
        return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _cell_on_ray(cell, tip, direction):
    r, c = cell
    tr, tc = tip
    dr, dc = direction
    if dr == 0:
        return r == tr and (c - tc) * dc > 0
    return c == tc and (r - tr) * dr > 0


def _target_for_arrow(tip, direction, size, gap, offset):
    r, c = tip
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return r - offset, c + gap
    if (dr, dc) == (0, -1):
        return r - offset, c - gap - size + 1
    if (dr, dc) == (1, 0):
        return r + gap, c - offset
    return r - gap - size + 1, c - offset


def _make_candidate(h, w, direction):
    tail_len = random.randint(2, 5)
    size = random.choice((2, 3))
    gap = random.randint(3, 8)
    offset = random.randrange(size)
    dr, dc = direction

    if dr == 0:
        r = random.randint(1, h - 2)
        if dc == 1:
            lo = tail_len
            hi = w - size - gap
        else:
            lo = size + gap - 1
            hi = w - tail_len - 1
        if lo > hi:
            return None
        c = random.randint(lo, hi)
    else:
        c = random.randint(1, w - 2)
        if dr == 1:
            lo = tail_len
            hi = h - size - gap
        else:
            lo = size + gap - 1
            hi = h - tail_len - 1
        if lo > hi:
            return None
        r = random.randint(lo, hi)

    top, left = _target_for_arrow((r, c), direction, size, gap, offset)
    arrow = _arrow_cells((r, c), direction, tail_len)
    target = _rect_cells(top, left, size)
    return (r, c), direction, arrow, target


def _place_arrow_pair(h, w, occupied, blue_cells, records, color, direction):
    for _ in range(300):
        cand = _make_candidate(h, w, direction)
        if cand is None:
            continue
        tip, direc, arrow, target = cand
        if not _fits_with_margin(arrow | target, occupied, h, w):
            continue
        if any(_cell_on_ray(cell, old_tip, old_dir) for old_tip, old_dir in records for cell in arrow | target):
            continue
        if any(_cell_on_ray(cell, tip, direc) for cell in blue_cells):
            continue
        occupied.update(arrow)
        occupied.update(target)
        blue_cells.update(target)
        records.append((tip, direc))
        return arrow, target, color
    return None


def _place_extra_blue(h, w, occupied, records):
    for _ in range(300):
        size = random.choice((2, 3))
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        rect = _rect_cells(top, left, size)
        if not _fits_with_margin(rect, occupied, h, w):
            continue
        if any(_cell_on_ray(cell, tip, direction) for tip, direction in records for cell in rect):
            continue
        occupied.update(rect)
        return rect
    return None


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        n_arrows = random.randint(2, 4)
        n_extra = random.randint(0, 3)
        colors = random.sample(ARROW_COLORS, n_arrows)
        dirs = [random.choice(DIRECTIONS) for _ in range(n_arrows)]
        random.shuffle(dirs)

        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        blue_cells = set()
        records = []
        placed = []

        ok = True
        for color, direction in zip(colors, dirs):
            result = _place_arrow_pair(h, w, occupied, blue_cells, records, color, direction)
            if result is None:
                ok = False
                break
            placed.append(result)
        if not ok:
            continue

        extras = []
        for _j in range(n_extra):
            rect = _place_extra_blue(h, w, occupied, records)
            if rect is not None:
                extras.append(rect)

        for arrow, target, color in placed:
            for r, c in target:
                grid[r, c] = BLUE
            for r, c in arrow:
                grid[r, c] = color
        for rect in extras:
            for r, c in rect:
                grid[r, c] = BLUE
        return grid

    grid = np.zeros((24, 24), dtype=int)
    examples = [
        ((8, 5), (0, 1), 3, 2, (7, 14), 2),
        ((15, 18), (0, -1), 4, 3, (14, 6), 4),
        ((5, 16), (1, 0), 3, 2, (13, 15), 6),
    ]
    for tip, direction, tail_len, size, target_pos, color in examples:
        for r, c in _rect_cells(target_pos[0], target_pos[1], size):
            grid[r, c] = BLUE
        for r, c in _arrow_cells(tip, direction, tail_len):
            grid[r, c] = color
    for r, c in _rect_cells(19, 3, 2):
        grid[r, c] = BLUE
    return grid

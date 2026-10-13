import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _grey_template(grid):
    h, w = grid.shape
    seen = set()
    comps = []
    for r, c in zip(*np.where(grid == GREY)):
        start = (int(r), int(c))
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if (nr, nc) not in seen and grid[nr, nc] == GREY:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        comps.append(cells)
    if not comps:
        return None
    cells = max(comps, key=len)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _rotated(mask, direction):
    if direction == "right":
        return mask
    if direction == "down":
        return np.rot90(mask, -1)
    if direction == "left":
        return np.rot90(mask, 2)
    return np.rot90(mask, 1)


def _direction(a, b):
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if abs(dc) >= abs(dr):
        return "right" if dc >= 0 else "left"
    return "down" if dr >= 0 else "up"


def _stamp(out, mask, center, color):
    h, w = mask.shape
    r0 = int(round(center[0] - (h - 1) / 2))
    c0 = int(round(center[1] - (w - 1) / 2))
    for rr, cc in zip(*np.where(mask)):
        r, c = r0 + int(rr), c0 + int(cc)
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a rotated grey template at the midpoint of each same-color anchor pair.

    Input structure:
    A black grid has one connected grey asymmetric template near a corner and several
    non-grey colors that each appear as exactly two anchor pixels.

    Rule:
    Same-color anchors are paired.  Reading-order first anchor to second anchor gives
    the dominant direction: right, down, left, or up.  The grey template's default
    orientation is right; it is rotated to that direction and stamped at the visual
    midpoint of the anchor pair, recolored to the anchor color.  Original anchors and
    the grey template remain in place.

    Key insight:
    The pair's spatial direction controls template rotation, while the pair color
    supplies the stamp color.
    """
    out = np.array(grid, copy=True)
    mask = _grey_template(grid)
    if mask is None:
        return out
    by_color = defaultdict(list)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = int(grid[r, c])
            if val not in (BG, GREY):
                by_color[val].append((r, c))
    for color, pts in by_color.items():
        if len(pts) != 2:
            continue
        pts = sorted(pts)
        direction = _direction(pts[0], pts[1])
        stamp = _rotated(mask, direction)
        center = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
        _stamp(out, stamp, center, color)
    return out


TEMPLATES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
]


def _can_write(grid, cells):
    return all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == BG for r, c in cells)


def _reserve_stamp(mask, center):
    h, w = mask.shape
    r0 = int(round(center[0] - (h - 1) / 2))
    c0 = int(round(center[1] - (w - 1) / 2))
    return {(r0 + int(r), c0 + int(c)) for r, c in zip(*np.where(mask))}


def generate_input() -> np.ndarray:
    for _ in range(1000):
        h = random.randint(17, 28)
        w = random.randint(17, 28)
        grid = np.zeros((h, w), dtype=int)
        tmpl = random.choice(TEMPLATES)
        tr, tc = random.choice([(1, 1), (1, w - tmpl.shape[1] - 2), (h - tmpl.shape[0] - 2, 1)])
        for r, c in zip(*np.where(tmpl)):
            grid[tr + int(r), tc + int(c)] = GREY

        colors = random.sample(COLORS, random.randint(3, 5))
        occupied = set(zip(*np.where(grid != BG)))
        ok = True
        for color in colors:
            placed = False
            for _ in range(200):
                direction = random.choice(["right", "down", "left"])
                stamp_mask = _rotated(tmpl, direction)
                center_r = random.randint(5, h - 6)
                center_c = random.randint(5, w - 6)
                dist = random.randint(3, 6)
                if direction == "right":
                    a, b = (center_r, center_c - dist), (center_r, center_c + dist)
                elif direction == "left":
                    a, b = (center_r - 1, center_c + dist), (center_r + 1, center_c - dist)
                else:
                    a, b = (center_r - dist, center_c), (center_r + dist, center_c)
                stamp_cells = _reserve_stamp(stamp_mask, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
                cells = {a, b} | stamp_cells
                if any(r < 1 or r >= h - 1 or c < 1 or c >= w - 1 for r, c in cells):
                    continue
                if occupied & ({a, b} | stamp_cells):
                    continue
                grid[a] = color
                grid[b] = color
                occupied |= {a, b} | stamp_cells
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid anchor-template puzzle")

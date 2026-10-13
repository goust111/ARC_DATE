from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _perimeter(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.extend([(r0, c), (r1, c)])
    for r in range(r0 + 1, r1):
        cells.extend([(r, c0), (r, c1)])
    return cells


def _mask(cells):
    r0, c0, r1, c1 = _bbox(cells)
    m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        m[r - r0, c - c0] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each room's stamp onto grey ink pixels with quadrant-dependent flips.

    Input structure: Blue rectangular room walls enclose black interiors.  Each
    room has one colored asymmetric stamp and one to three grey ink pixels.
    Rule: For every ink pixel, compare its position to the stamp center.  Left-up
    copies are unchanged, right-up copies are horizontally flipped, left-down
    copies are vertically flipped, and right-down copies are flipped both ways.
    The ink pixel is the copy center; copies are clipped to the room interior.
    Key insight: The grey ink pixel's quadrant relative to the stamp, not its
    color or distance, selects the mirror transform.
    """
    out = grid.copy()
    for wall in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(wall)
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
        sub = grid[interior]
        stamp_comps = _components((sub != BLACK) & (sub != GREY) & (sub != BLUE))
        if not stamp_comps:
            continue
        stamp = [(r0 + 1 + r, c0 + 1 + c) for r, c in max(stamp_comps, key=len)]
        mask = _mask(stamp)
        sr0, sc0, sr1, sc1 = _bbox(stamp)
        center_r, center_c = (sr0 + sr1) / 2, (sc0 + sc1) / 2
        color = int(grid[stamp[0]])
        for ir, ic in [(r0 + 1 + r, c0 + 1 + c) for r, c in map(tuple, np.argwhere(sub == GREY))]:
            m = mask.copy()
            if ic > center_c:
                m = np.fliplr(m)
            if ir > center_r:
                m = np.flipud(m)
            mh, mw = m.shape
            top, left = ir - mh // 2, ic - mw // 2
            for rr, cc in np.argwhere(m):
                tr, tc = top + int(rr), left + int(cc)
                if r0 < tr < r1 and c0 < tc < c1:
                    out[tr, tc] = color
    return out


def _draw_room(grid, r0, c0, h, w):
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r, c in _perimeter(r0, c0, r1, c1):
        grid[r, c] = BLUE


def _stamp(grid, mask, top, left, color):
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = color


def generate_input() -> np.ndarray:
    grid = np.zeros((30, 30), dtype=int)
    rooms = [(2, 2), (2, 17), (17, 2), (17, 17)]
    n = random.randint(2, 4)
    colors = random.sample([2, 3, 4, 6, 7, 8, 9], n)
    stamp = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)
    for i in range(n):
        r0, c0 = rooms[i]
        _draw_room(grid, r0, c0, 11, 11)
        sr, sc = r0 + 4, c0 + 4
        _stamp(grid, stamp, sr, sc, colors[i])
        candidates = [(r0 + 2, c0 + 2), (r0 + 2, c0 + 8), (r0 + 8, c0 + 2), (r0 + 8, c0 + 8)]
        for ir, ic in random.sample(candidates, random.randint(1, 3)):
            grid[ir, ic] = GREY
    return grid

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
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


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter(r0, c0, r1, c1):
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            pts.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            pts.append((r, c0))
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each grey frame border with the legend style chosen by the interior vote.

    Input structure:
    - Above a bottom legend area are disjoint hollow grey rectangles on black.
    - Each frame interior contains only red and blue vote pixels on black.
    - The bottom two rows act as a style legend: one row encodes solid grey border
      style for one vote color, and the other row encodes alternating grey/black
      dashed border style for the other.

    Rule:
    - Decode which vote color maps to solid and which maps to dashed.
    - For each frame, count red and blue interior pixels and choose the strict winner.
    - Clear the entire interior to black.
    - Redraw the border in the winner's style; the dashed style alternates grey and
      black around the frame perimeter starting with grey at the top-left corner.
    - Remove the legend rows from the output.

    Key insight:
    - The legend controls border geometry, while the interior colors act as a hidden
      plurality vote selecting which style survives after the interior evidence is
      erased.
    """
    legend = grid[-2:]
    row0 = legend[0]
    row1 = legend[1]
    solid_color = RED if np.count_nonzero(row0 == GREY) >= np.count_nonzero(row1 == GREY) else BLUE
    dashed_color = BLUE if solid_color == RED else RED

    out = grid[:-2].copy()
    for comp in _components(out == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        red_n = int(np.count_nonzero(inner == RED))
        blue_n = int(np.count_nonzero(inner == BLUE))
        winner = RED if red_n > blue_n else BLUE
        inner[:] = BLACK
        pts = _perimeter(r0, c0, r1, c1)
        for idx, (r, c) in enumerate(pts):
            if winner == solid_color:
                out[r, c] = GREY
            else:
                out[r, c] = GREY if idx % 2 == 0 else BLACK
    return out


def generate_input() -> np.ndarray:
    while True:
        n = random.randint(2, 4)
        frames = []
        cur_c = 1
        max_h = 0
        for _ in range(n):
            ih = random.randint(3, 5)
            iw = random.randint(4, 6)
            fh, fw = ih + 2, iw + 2
            frames.append((ih, iw, fh, fw))
            cur_c += fw + random.randint(1, 3)
            max_h = max(max_h, fh)
        h = max_h + 4
        w = max(cur_c + 1, 14)
        if h > 30 or w > 30:
            continue
        grid = np.zeros((h, w), dtype=int)
        c0 = 1
        for ih, iw, fh, fw in frames:
            r0 = random.randint(0, max_h - fh)
            grid[r0 : r0 + fh, c0 : c0 + fw] = GREY
            grid[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw] = BLACK
            area = ih * iw
            total = random.randint(max(2, area // 3), max(3, area - 1))
            red_n = random.randint(1, total - 1)
            blue_n = total - red_n
            if red_n == blue_n:
                red_n += 1
            cells = random.sample([(r, c) for r in range(r0 + 1, r0 + 1 + ih) for c in range(c0 + 1, c0 + 1 + iw)], red_n + blue_n)
            for r, c in cells[:red_n]:
                grid[r, c] = RED
            for r, c in cells[red_n:red_n + blue_n]:
                grid[r, c] = BLUE
            c0 += fw + random.randint(1, 3)
        grid[-2, 0] = RED
        grid[-2, 1:] = GREY
        for c in range(1, w):
            grid[-1, c] = GREY if c % 2 else BLACK
        grid[-1, 0] = BLUE
        return grid

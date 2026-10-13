import random
from collections import deque

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_of_color(grid, color):
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    out = []
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
            out.append(cells)
    return out


def _center_and_dir(cells):
    s = set(cells)
    for r, c in cells:
        nbrs = {(nr, nc) for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if (nr, nc) in s}
        if len(nbrs) == 3:
            center = (r, c)
            missing = None
            for dr, dc, name in ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")):
                if (r + dr, c + dc) not in nbrs:
                    missing = name
                    break
            direction = {"left": "right", "right": "left", "up": "down", "down": "up"}[missing]
            return center, direction
    return cells[0], "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each same-color T-arrow pair until the horizontal and vertical rays meet.

    Input structure:
    - Every active color appears as exactly two 4-cell T-shapes.
    - One T points horizontally and the other vertically, so their forward rays
      intersect at a unique cell.

    Rule:
    - Decode each T's facing direction from the side missing around its degree-3 center.
    - From each center, extend a monochrome ray in the facing direction until the two
      rays of the same color intersect.
    - Keep the original T-shapes intact and color the intersection as part of the path.

    Key insight:
    - The T's opening is not the arrowhead; the opposite side is. Once that hidden
      direction is decoded, the paired rays reveal a guaranteed L-shaped connection.
    """
    out = grid.copy()
    for color in [int(v) for v in np.unique(grid) if v != BLACK]:
        comps = _components_of_color(grid, color)
        if len(comps) != 2:
            continue
        info = [_center_and_dir(comp) for comp in comps]
        horiz = next((x for x in info if x[1] in ("left", "right")), None)
        vert = next((x for x in info if x[1] in ("up", "down")), None)
        if horiz is None or vert is None:
            b0 = comps[0]
            b1 = comps[1]
            rows0 = [r for r, _ in b0]
            cols0 = [c for _, c in b0]
            rows1 = [r for r, _ in b1]
            cols1 = [c for _, c in b1]
            c0 = (min(rows0) + max(rows0)) / 2, (min(cols0) + max(cols0)) / 2
            c1 = (min(rows1) + max(rows1)) / 2, (min(cols1) + max(cols1)) / 2
            if (max(cols0) - min(cols0)) >= (max(rows0) - min(rows0)):
                horiz = (info[0][0], "right" if c1[1] > c0[1] else "left")
                vert = (info[1][0], "down" if c0[0] > c1[0] else "up")
            else:
                horiz = (info[1][0], "right" if c0[1] > c1[1] else "left")
                vert = (info[0][0], "down" if c1[0] > c0[0] else "up")
        hr, hc = horiz[0]
        vr, vc = vert[0]
        if horiz[1] == "right":
            cols = range(hc, vc + 1)
        else:
            cols = range(hc, vc - 1, -1)
        for c in cols:
            out[hr, c] = color
        if vert[1] == "down":
            rows = range(vr, hr + 1)
        else:
            rows = range(vr, hr - 1, -1)
        for r in rows:
            out[r, vc] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample(COLORS, random.randint(2, 4)):
        ir = random.randint(5, h - 6)
        ic = random.randint(5, w - 6)
        hr = ir
        hc = random.randint(1, ic - 2)
        vr = random.randint(1, ir - 2)
        vc = ic
        for center, direction in [((hr, hc), "right"), ((vr, vc), "down")]:
            r, c = center
            grid[r, c] = color
            if direction == "right":
                for dr, dc in [(0, -1), (-1, 0), (1, 0)]:
                    grid[r + dr, c + dc] = color
            else:
                for dr, dc in [(1, 0), (0, -1), (0, 1)]:
                    grid[r + dr, c + dc] = color
    return grid

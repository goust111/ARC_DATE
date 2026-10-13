import random
from collections import deque

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(grid, color):
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each small rectangle toward its same-color segment as a solid strip.

    Input structure:
    - Each active color has one long one-pixel-wide segment and one or more detached
      small solid rectangles of the same color.
    - Rectangles lie entirely on one side of their matching segment.

    Rule:
    - For every rectangle, find the unique segment of the same color.
    - Determine whether the rectangle must project vertically toward a horizontal
      segment or horizontally toward a vertical segment.
    - Fill the strip between the rectangle face nearest the segment and the segment,
      preserving the rectangle and the segment.

    Key insight:
    - Color identifies the segment partner, but geometry chooses the projection axis
      and span; the strip inherits its width from the rectangle, not from the segment.
    """
    out = grid.copy()
    for color in [int(v) for v in np.unique(grid) if v != BLACK]:
        comps = _components(grid, color)
        if len(comps) < 2:
            continue
        seg = max(comps, key=len)
        others = [c for c in comps if c is not seg]
        sr0, sc0, sr1, sc1 = _bbox(seg)
        horizontal = sr0 == sr1
        for obj in others:
            r0, c0, r1, c1 = _bbox(obj)
            if horizontal:
                if r1 < sr0:
                    out[r1 + 1 : sr0, c0 : c1 + 1] = color
                else:
                    out[sr0 + 1 : r0, c0 : c1 + 1] = color
            else:
                if c1 < sc0:
                    out[r0 : r1 + 1, c1 + 1 : sc0] = color
                else:
                    out[r0 : r1 + 1, sc0 + 1 : c0] = color
    return out


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for color in random.sample(COLORS, random.randint(2, 4)):
        if random.random() < 0.5:
            r = random.randint(4, h - 5)
            c0 = random.randint(1, w - 8)
            c1 = random.randint(c0 + 4, min(w - 2, c0 + 7))
            grid[r, c0 : c1 + 1] = color
            for _ in range(random.randint(1, 2)):
                oh, ow = random.randint(1, 2), random.randint(2, 3)
                above = random.random() < 0.5
                rr = random.randint(1, r - oh - 1) if above else random.randint(r + 2, h - oh - 1)
                cc = random.randint(c0, c1 - ow + 1)
                grid[rr : rr + oh, cc : cc + ow] = color
        else:
            c = random.randint(4, w - 5)
            r0 = random.randint(1, h - 8)
            r1 = random.randint(r0 + 4, min(h - 2, r0 + 7))
            grid[r0 : r1 + 1, c] = color
            for _ in range(random.randint(1, 2)):
                oh, ow = random.randint(2, 3), random.randint(1, 2)
                left = random.random() < 0.5
                rr = random.randint(r0, r1 - oh + 1)
                cc = random.randint(1, c - ow - 1) if left else random.randint(c + 2, w - ow - 1)
                grid[rr : rr + oh, cc : cc + ow] = color
    return grid

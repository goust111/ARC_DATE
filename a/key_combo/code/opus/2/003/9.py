import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                            seen[xx, yy] = True
                            q.append((xx, yy))
                comps.append(comp)
    return comps


def _draw_frame(out, r0, c0, side, color):
    r1, c1 = r0 + side - 1, c0 + side - 1
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    return (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    gray = np.argwhere(grid == 5)
    gr0, gc0 = gray.min(axis=0)
    gr1, gc1 = gray.max(axis=0)
    orange = np.argwhere(grid == 7)
    oside = 1
    scale = max(gr1 - gr0 + 1, gc1 - gc0 + 1) // oside
    anchors = []
    for color in [int(c) for c in np.unique(grid) if c not in (0, 5, 7)]:
        for comp in _components(grid == color):
            pts = np.array(comp)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and r1 - r0 == c1 - c0:
                side = int(r1 - r0 + 1)
                anchors.append((side, color, (r0 + r1) / 2, (c0 + c1) / 2))
    anchors.sort(reverse=True)
    frames = []
    for side, color, cr, cc in anchors:
        fside = side * scale
        r0 = int(round(cr - (fside - 1) / 2))
        c0 = int(round(cc - (fside - 1) / 2))
        frames.append((side, color, *_draw_frame(out, r0, c0, fside, color)))
    frames.sort(reverse=True)
    higher_interiors = []
    for i, (_, color, r0, c0, r1, c1) in enumerate(frames):
        if i == len(frames) - 1:
            higher_interiors.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1))
            continue
        fill = frames[i + 1][1]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                inside_higher = any(a <= r <= b and x <= c <= y for a, x, b, y in higher_interiors)
                if not inside_higher and out[r, c] == 0:
                    out[r, c] = fill
        higher_interiors.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1))
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=int)
    grid[0, 0] = 7
    grid[0, 2:6] = grid[3, 2:6] = 5
    grid[0:4, 2] = grid[0:4, 5] = 5
    colors = random.sample([1, 2, 3, 4, 6, 8], 3)
    sizes = random.sample([1, 2, 3], 3)
    centers = [(8, 8), (8, 10), (10, 8), (10, 10)]
    random.shuffle(centers)
    for color, side, (cr, cc) in zip(colors, sizes, centers):
        r0 = int(cr - (side - 1) / 2)
        c0 = int(cc - (side - 1) / 2)
        grid[r0 : r0 + side, c0 : c0 + side] = color
    return grid

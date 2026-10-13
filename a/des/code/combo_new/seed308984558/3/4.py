import random
from collections import deque

import numpy as np


BG = 0
BASE = 9


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
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _holes(cells):
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(bg.shape[0]):
        for c in range(bg.shape[1]):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touch = False
            while q:
                x, y = q.popleft()
                touch |= x in (0, bg.shape[0] - 1) or y in (0, bg.shape[1] - 1)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < bg.shape[0] and 0 <= ny < bg.shape[1] and bg[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            if not touch:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the objects by enclosed-window count and repack them above the foundation line.

    Input structure:
    - A black background contains several disconnected monochrome objects and one full-width maroon baseline.
    - Objects may contain enclosed background windows.
    - The objects do not touch the baseline initially.

    Rule:
    - Count each object's enclosed windows, breaking ties by smaller area first.
    - Clear all original objects.
    - Place the sorted objects left-to-right with one blank column between bounding boxes, resting directly above the baseline.

    Key insight:
    - The ordering depends on detecting truly enclosed background regions inside each object rather than merely counting visible gaps.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    base_row = int(np.where(np.all(g == BASE, axis=1))[0][0])
    objs = []
    for color in [v for v in np.unique(g) if v not in (BG, BASE)]:
        for cells in _components(g == int(color)):
            objs.append((cells, int(color)))
            for r, c in cells:
                out[r, c] = BG
    objs.sort(key=lambda item: (_holes(item[0]), len(item[0])))
    cur = 0
    for cells, color in objs:
        r0, c0, r1, c1 = _bbox(cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        rr = base_row - mask.shape[0]
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r, c]:
                    out[rr + r, cur + c] = color
        cur += mask.shape[1] + 1
    return out


SHAPES = [
    np.array([[1, 1], [1, 0]]),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    base_row = h - 2
    g[base_row, :] = BASE
    for color in random.sample([1, 2, 3, 4, 6, 7, 8], random.randint(2, 4)):
        shape = random.choice(SHAPES)
        sh, sw = shape.shape
        for _ in range(200):
            r0 = random.randint(0, base_row - sh - 2)
            c0 = random.randint(0, w - sw)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]):
                continue
            g[r0 : r0 + sh, c0 : c0 + sw][shape == 1] = color
            break
    return g

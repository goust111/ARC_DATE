import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray):
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(pts)
    return parts


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each closed frame using the legend color indexed by its island count."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    legend = [int(g[r, w - 2]) for r in range(0, h, 2) if w >= 2 and int(g[r, w - 2]) != 0]
    frame_mask = (g == 1)
    out = g.copy()
    for pts in _components(frame_mask):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if c1 >= w - 3:
            continue
        inner = (g[r0 + 1 : r1, c0 + 1 : c1] == 2)
        count = len(_components(inner))
        color = legend[min(count, len(legend) - 1)]
        for r, c in pts:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(20, 28)
    g = np.zeros((h, w), dtype=int)
    legend = random.sample([3, 4, 5, 6, 7, 8, 9], 4)
    for i, color in enumerate(legend):
        r = 2 * i
        g[r, w - 2 : w] = color
    boxes = []
    for _ in range(random.randint(2, 4)):
        for _ in range(100):
            fh, fw = random.randint(4, 6), random.randint(4, 7)
            r0 = random.randint(1, h - fh - 2)
            c0 = random.randint(1, w - fw - 5)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(box[2] + 1 < b[0] or b[2] + 1 < box[0] or box[3] + 1 < b[1] or b[3] + 1 < box[1] for b in boxes):
                boxes.append(box)
                break
    for r0, c0, r1, c1 in boxes:
        g[r0 : r1 + 1, c0] = 1
        g[r0 : r1 + 1, c1] = 1
        g[r0, c0 : c1 + 1] = 1
        g[r1, c0 : c1 + 1] = 1
        islands = random.randint(0, 3)
        placed = []
        for _ in range(islands):
            ih, iw = random.randint(1, 2), random.randint(1, 2)
            for _ in range(50):
                rr = random.randint(r0 + 1, r1 - ih)
                cc = random.randint(c0 + 1, c1 - iw)
                rect = (rr, cc, rr + ih - 1, cc + iw - 1)
                if all(rect[2] < p[0] or p[2] < rect[0] or rect[3] < p[1] or p[3] < rect[1] for p in placed):
                    placed.append(rect)
                    g[rr : rr + ih, cc : cc + iw] = 2
                    break
    return g

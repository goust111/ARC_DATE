import random
from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the unique template pattern into every congruent grey frame of the same size."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    mask = g == 5
    seen = np.zeros_like(mask, dtype=bool)
    groups = {}
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
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        rs, cs = zip(*pts)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        key = (r1 - r0 + 1, c1 - c0 + 1)
        groups.setdefault(key, []).append((r0, r1, c0, c1))
    for key, boxes in groups.items():
        template = None
        for r0, r1, c0, c1 in boxes:
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if np.any(inner != 0):
                template = inner.copy()
        if template is None:
            continue
        for r0, r1, c0, c1 in boxes:
            if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != 0):
                continue
            out[r0 + 1 : r1, c0 + 1 : c1] = template
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    for dims in random.sample([(5, 6), (6, 5), (5, 5)], 2):
        boxes = []
        for _ in range(random.randint(2, 3)):
            for _ in range(100):
                h, w = dims
                r0 = random.randint(1, g.shape[0] - h - 1)
                c0 = random.randint(1, g.shape[1] - w - 1)
                if np.all(g[max(0, r0 - 1) : min(g.shape[0], r0 + h + 1), max(0, c0 - 1) : min(g.shape[1], c0 + w + 1)] == 0):
                    g[r0 : r0 + h, c0] = 5
                    g[r0 : r0 + h, c0 + w - 1] = 5
                    g[r0, c0 : c0 + w] = 5
                    g[r0 + h - 1, c0 : c0 + w] = 5
                    boxes.append((r0, c0, h, w))
                    break
        if boxes:
            r0, c0, h, w = random.choice(boxes)
            color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
            g[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = 0
            g[r0 + 1, c0 + 1 : c0 + w - 2] = color
            g[r0 + h - 2, c0 + 2] = color
    return g

import random
from collections import defaultdict

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble same-color fragments into rectangles, then stack the rectangles by area."""
    g = np.array(grid, dtype=int)
    rects = []
    for color in sorted(set(map(int, g.ravel())) - {0}):
        rs, cs = np.where(g == color)
        if len(rs) == 0:
            continue
        # Generator only emits slices sharing one common span.
        comps = []
        mask = g == color
        seen = np.zeros_like(mask, dtype=bool)
        for r, c in zip(*np.where(mask)):
            if seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
        hs = [max(r for r, _ in c) - min(r for r, _ in c) + 1 for c in comps]
        ws = [max(c for _, c in cpt) - min(c for _, c in cpt) + 1 for cpt in comps]
        if len(set(hs)) == 1:
            h, w = hs[0], sum(ws)
        else:
            h, w = sum(hs), ws[0]
        rects.append((h * w, np.full((h, w), color, dtype=int)))
    rects.sort(key=lambda x: x[0])
    width = max(r.shape[1] for _, r in rects)
    height = sum(r.shape[0] for _, r in rects)
    out = np.zeros((height, width), dtype=int)
    r0 = 0
    for _, rect in rects:
        out[r0 : r0 + rect.shape[0], : rect.shape[1]] = rect
        r0 += rect.shape[0]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    for color in random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4)):
        h, w = random.randint(2, 4), random.randint(3, 6)
        vertical = random.choice([True, False])
        dim = h if vertical else w
        parts = random.randint(2, min(3, dim))
        cuts = sorted(random.sample(range(1, dim), parts - 1))
        spans = [0] + cuts + [h if vertical else w]
        rects = []
        for a, b in zip(spans, spans[1:]):
            rects.append((b - a, w) if vertical else (h, b - a))
        for ph, pw in rects:
            for _ in range(100):
                r0 = random.randint(0, g.shape[0] - ph)
                c0 = random.randint(0, g.shape[1] - pw)
                if np.all(g[max(0, r0 - 1) : min(g.shape[0], r0 + ph + 1), max(0, c0 - 1) : min(g.shape[1], c0 + pw + 1)] == 0):
                    g[r0 : r0 + ph, c0 : c0 + pw] = color
                    break
    return g

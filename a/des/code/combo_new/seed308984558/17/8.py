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


def _norm(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    r0, c0 = min(rs), min(cs)
    return sorted((r - r0, c - c0) for r, c in pts)


def _clear(path, occ):
    return all(0 <= r < occ.shape[0] and 0 <= c < occ.shape[1] and not occ[r, c] for r, c in path)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect same-color object pairs with a red or blue orthogonal path by shape match."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    occ = g != 0
    occ[-2:, :2] = False
    for color in sorted(set(map(int, g.ravel())) - {0, 2, 1}):
        comps = _components(g == color)
        if len(comps) != 2:
            continue
        a, b = comps
        line_color = 2 if _norm(a) == _norm(b) else 1
        best = None
        for p in a:
            for q in b:
                path1 = [(p[0], c) for c in range(min(p[1], q[1]), max(p[1], q[1]) + 1)] + [(r, q[1]) for r in range(min(p[0], q[0]), max(p[0], q[0]) + 1)]
                path2 = [(r, p[1]) for r in range(min(p[0], q[0]), max(p[0], q[0]) + 1)] + [(q[0], c) for c in range(min(p[1], q[1]), max(p[1], q[1]) + 1)]
                cand = None
                if _clear(path1[1:-1], occ):
                    cand = path1
                elif _clear(path2[1:-1], occ):
                    cand = path2
                if cand is not None and (best is None or len(cand) < len(best)):
                    best = cand
        if best:
            for r, c in best[1:-1]:
                out[r, c] = line_color
                occ[r, c] = True
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((random.randint(18, 24), random.randint(18, 24)), dtype=int)
    g[-2, 0] = 2
    g[-1, 0] = 1
    shapes = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2)],
    ]
    for color in random.sample([3, 4, 5, 6, 7, 8, 9], random.randint(2, 3)):
        same = random.choice([True, False])
        s1 = random.choice(shapes)
        s2 = s1 if same else random.choice([s for s in shapes if s != s1])
        for pts, box in zip((s1, s2), [(1, 1), (1, g.shape[1] - 6)]):
            r0 = random.randint(box[0], g.shape[0] - 6)
            c0 = random.randint(box[1], max(box[1], min(g.shape[1] - 6, box[1] + 1)))
            for dr, dc in pts:
                g[r0 + dr, c0 + dc] = color
    return g

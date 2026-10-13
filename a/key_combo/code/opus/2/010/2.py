from collections import defaultdict, deque
import random

import numpy as np


RAY_COLORS = {1, 3, 4, 7}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    markers = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if int(g[r, c]) in RAY_COLORS]
    out = g.copy()
    srcs = defaultdict(set)
    for r, c, col in markers:
        srcs[(r, c)].add(col)
    for r, c, col in markers:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if g[rr, cc] == 5:
                    break
                srcs[(rr, cc)].add(col)
                rr += dr
                cc += dc
    for (r, c), colors in srcs.items():
        if g[r, c] == 5:
            continue
        if len(colors) >= 2:
            out[r, c] = 2
        elif g[r, c] == 0 or int(g[r, c]) in RAY_COLORS:
            out[r, c] = next(iter(colors))

    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if out[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            edge = False
            boundary_sources = set()
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                edge |= rr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if out[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif out[nr, nc] != 0 and out[nr, nc] != 5:
                        boundary_sources |= srcs.get((nr, nc), {int(out[nr, nc])} if int(out[nr, nc]) in RAY_COLORS else set())
            if not edge and boundary_sources:
                fill = next(iter(boundary_sources)) if len(boundary_sources) == 1 else 2
                for rr, cc in comp:
                    out[rr, cc] = fill
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((15, 15), dtype=int)
    g[2, 2:13] = 5
    g[12, 2:13] = 5
    g[2:13, 2] = 5
    if random.random() < 0.65:
        g[2:13, 12] = 5
        g[random.randint(5, 9), 12] = 0
    g[5, 4:11] = 5
    g[9, 4:11] = 5
    marks = [(7, 4), (4, 7), (10, 10), (7, 11)]
    cols = random.sample([1, 3, 4, 7], random.randint(2, 4))
    random.shuffle(marks)
    used_rows, used_cols = set(), set()
    for (r, c), col in zip(marks, cols):
        if r not in used_rows and c not in used_cols and g[r, c] == 0:
            g[r, c] = col
            used_rows.add(r)
            used_cols.add(c)
    return g

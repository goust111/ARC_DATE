import random
import numpy as np


DIRS = {2: (0, 1), 1: (1, 0), 3: (0, -1)}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sources = [(r, c, int(g[r, c])) for r in range(h) for c in range(w) if int(g[r, c]) in DIRS]
    claims = {}
    paths = {}
    for r, c, col in sources:
        dr, dc = DIRS[col]
        path = []
        nr, nc = r + dr, c + dc
        last = None
        while 0 <= nr < h and 0 <= nc < w:
            if g[nr, nc] == 5:
                break
            last = (nr, nc)
            path.append((nr, nc))
            nr += dr; nc += dc
        if last is None:
            continue
        dist = abs(last[0] - r) + abs(last[1] - c)
        key = (dist, r, c)
        if last not in claims or key < claims[last][0]:
            claims[last] = (key, (r, c, col))
            paths[(r, c, col)] = path
    out = g.copy()
    for r, c, _ in sources:
        out[r, c] = 0
    for end, (_, src) in claims.items():
        for p in paths[src]:
            out[p] = 5
        out[end] = src[2]
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((16, 18), dtype=int)
    anchors = [(3, 12), (9, 5), (11, 14), (5, 3)]
    random.shuffle(anchors)
    for top, left in anchors[:random.randint(3, 4)]:
        g[top:top + 2, left:left + 2] = 5
    candidates = [
        (3, 1, 2), (4, 1, 2), (9, 1, 2), (12, 1, 2),
        (1, 5, 1), (1, 12, 1), (1, 14, 1), (2, 3, 1),
        (3, 16, 3), (11, 16, 3), (5, 16, 3), (9, 16, 3),
    ]
    random.shuffle(candidates)
    count = random.randint(5, 8)
    placed = 0
    for r, c, col in candidates:
        if placed >= count:
            break
        dr, dc = DIRS[col]
        nr, nc = r + dr, c + dc
        sees = False
        while 0 <= nr < 16 and 0 <= nc < 18:
            if g[nr, nc] == 5:
                sees = True
                break
            nr += dr; nc += dc
        if sees and g[r, c] == 0:
            g[r, c] = col
            placed += 1
    return g

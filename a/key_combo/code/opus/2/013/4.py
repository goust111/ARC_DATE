import numpy as np
from collections import deque

RNG = np.random.default_rng(404)


def main(grid: np.ndarray) -> np.ndarray:
    """Task 4: flood fill maze corridor using palette cycle.

Input:
    A maze in black with cyan walls, palette row on last row and seed 9.
Output:
    Reachable corridor recolored by palette sequence indexed by BFS distance.

Reasoning:
    Read contiguous palette segment from bottom row.
    BFS from seed over cells != wall and recolor with P[dist mod k].

Complexity:
    O(H*W).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bottom = g[h - 1]
    nz = np.where(bottom != 0)[0]
    if nz.size == 0:
        return out
    start, end = int(nz.min()), int(nz.max())
    seq = [int(v) for v in bottom[start : end + 1] if v != 0]
    if not seq:
        return out
    k = len(seq)

    seeds = list(zip(*np.where(g == 9)))
    if not seeds:
        return out
    sr, sc = seeds[0]

    allowed = (g != 8)
    dist = -np.ones((h, w), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h - 1 and 0 <= nc < w and allowed[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    for r in range(h):
        for c in range(w):
            if dist[r, c] >= 0:
                out[r, c] = seq[dist[r, c] % k]

    return out


def _is_connected_maze(g: np.ndarray) -> bool:
    h, w = g.shape
    seeds = list(zip(*np.where(g == 9)))
    if not seeds:
        return False
    sr, sc = seeds[0]
    if g[sr, sc] == 8:
        return False
    cells = [(r, c) for r in range(h - 1) for c in range(w) if g[r, c] != 8]
    if not cells:
        return False
    vis = np.zeros((h, w), dtype=bool)
    dq = deque([(sr, sc)])
    vis[sr, sc] = True
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h - 1 and 0 <= nc < w and not vis[nr, nc] and g[nr, nc] != 8:
                vis[nr, nc] = True
                dq.append((nr, nc))
    return all(vis[r, c] for r, c in cells)


def generate_input() -> np.ndarray:
    h, w = int(RNG.integers(13, 18)), int(RNG.integers(14, 21))

    for _ in range(2000):
        g = np.zeros((h, w), dtype=int)

        # thin internal walls (one-cell-thick barriers with guaranteed opening)
        for _ in range(int(RNG.integers(2, 6))):
            col = int(RNG.integers(1, w - 1))
            gap = int(RNG.integers(0, h - 1))
            for rr in range(h - 1):
                if rr != gap:
                    g[rr, col] = 8

        for _ in range(int(RNG.integers(1, 4))):
            row = int(RNG.integers(1, h - 2))
            gap = int(RNG.integers(0, w))
            for cc in range(w):
                if cc != gap:
                    g[row, cc] = 8

        # keep a clear separator row above palette
        g[h - 2, :] = 8

        # add a few random isolated pillars to diversify mazes
        for _ in range(int(RNG.integers(0, 6))):
            r = int(RNG.integers(1, h - 3))
            c = int(RNG.integers(0, w))
            if g[r, c] != 8:
                g[r, c] = 8

        k = int(RNG.integers(3, 5))
        start = 0
        seq = [int(v) for v in RNG.choice([2, 3, 4], size=k, replace=True)]
        g[h - 1, start : start + k] = seq

        # place seed on a navigable cell
        zero = [(r, c) for r in range(h - 2) for c in range(w) if g[r, c] == 0]
        if not zero:
            continue
        r0, c0 = zero[int(RNG.integers(len(zero)))]
        g[r0, c0] = 9

        if _is_connected_maze(g):
            return g

    # fallback connected layout
    g = np.zeros((h, w), dtype=int)
    for r in range(h - 3):
        g[r, 7] = 8
    g[h - 2, :] = 8
    k = 3
    g[h - 1, 0:k] = [2, 3, 4]
    g[0, 0] = 9
    return g

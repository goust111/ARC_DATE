import numpy as np
from collections import deque

BG = 0
BLUE = 1
RED = 2


def _flood_fill(mask: np.ndarray, start_r: int, start_c: int, vis: np.ndarray):
    h, w = mask.shape
    q = deque([(start_r, start_c)])
    comp = [(start_r, start_c)]
    vis[start_r, start_c] = True
    touches_border = (start_r == 0 or start_c == 0 or start_r == h - 1 or start_c == w - 1)

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                vis[nr, nc] = True
                q.append((nr, nc))
                comp.append((nr, nc))
                if nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1:
                    touches_border = True
    return comp, touches_border


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape

    out = g.copy()

    # Draw 4-direction cross for each blue pixel.
    centers = list(zip(*np.where(g == BLUE)))
    for r, c in centers:
        out[r, :] = BLUE
        out[:, c] = BLUE

    # Fill black components that are not connected to border.
    black = out == BG
    vis = np.zeros_like(g, dtype=bool)
    for r in range(h):
        for c in range(w):
            if black[r, c] and not vis[r, c]:
                comp, touches = _flood_fill(black, r, c, vis)
                if not touches:
                    for rr, cc in comp:
                        out[rr, cc] = RED

    return out


def _random_points(rng: np.random.Generator, h: int, w: int, n: int):
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(cells)
    selected = []
    used_rows = set()
    used_cols = set()

    for r, c in cells:
        if r in used_rows or c in used_cols:
            continue
        ok = True
        for sr, sc in selected:
            if abs(sr - r) == 1 and abs(sc - c) == 0:
                ok = False
                break
            if abs(sc - c) == 1 and abs(sr - r) == 0:
                ok = False
                break
            if abs(sr - r) + abs(sc - c) == 1:
                ok = False
                break
            if sr == r and sc == c:
                ok = False
                break
        if ok:
            selected.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            if len(selected) == n:
                return selected
    return selected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 13
    grid = np.zeros((h, w), dtype=np.int64)
    n = int(rng.integers(2, 5))
    pts = _random_points(rng, h, w, n)
    while len(pts) < n:
        pts = pts + _random_points(rng, h, w, n - len(pts))

    for r, c in pts[:n]:
        grid[r, c] = BLUE

    return grid

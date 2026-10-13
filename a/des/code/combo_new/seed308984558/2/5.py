import random
from collections import deque

import numpy as np


BG = 0


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored cross whose arm lengths are encoded by the host object's row and column spans through the seed.

    Input structure:
    - A black background contains several monochrome host objects.
    - Each host encloses one singleton seed pixel of a different color.
    - The seed is surrounded by host pixels and does not belong to the host color.

    Rule:
    - For each seed, identify the host color from its surrounding neighbors.
    - Count host pixels in the seed row and seed column within that host object.
    - Draw a cross in the seed color extending that many cells left/right and up/down, stopping before other foreground objects.

    Key insight:
    - The seed's color only paints the result; the arm lengths come from measuring the host object's row and column thickness through the seed.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if g[r, c] == BG:
                continue
            if np.count_nonzero(g == g[r, c]) != 1:
                continue
            neigh = [int(g[r + 1, c]), int(g[r - 1, c]), int(g[r, c + 1]), int(g[r, c - 1])]
            host = max(set(v for v in neigh if v != BG and v != int(g[r, c])), key=neigh.count)
            comp = None
            for cells in _components(g == host):
                if sum((abs(x - r) + abs(y - c) == 1) for x, y in cells) >= 3:
                    comp = set(cells)
                    break
            if comp is None:
                continue
            hcount = sum((r, y) in comp for y in range(g.shape[1]))
            vcount = sum((x, c) in comp for x in range(g.shape[0]))
            for dc in range(-hcount, hcount + 1):
                cc = c + dc
                if 0 <= cc < g.shape[1] and (g[r, cc] == BG or (r, cc) == (r, c)):
                    out[r, cc] = g[r, c]
            for dr in range(-vcount, vcount + 1):
                rr = r + dr
                if 0 <= rr < g.shape[0] and (g[rr, c] == BG or (rr, c) == (r, c)):
                    out[rr, c] = g[r, c]
    return out


HOSTS = [np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])]


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 26), random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    host_colors = random.sample([1, 2, 3, 4, 6, 7], random.randint(2, 4))
    seed_colors = random.sample([v for v in [1, 2, 3, 4, 6, 7, 8, 9] if v not in host_colors], len(host_colors))
    for hcol, scol in zip(host_colors, seed_colors):
        shape = random.choice(HOSTS)
        sh, sw = shape.shape
        for _ in range(200):
            r0 = random.randint(1, h - sh - 2)
            c0 = random.randint(1, w - sw - 2)
            if np.any(g[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]):
                continue
            g[r0 : r0 + sh, c0 : c0 + sw] = hcol
            sr, sc = r0 + sh // 2, c0 + sw // 2
            g[sr, sc] = scol
            break
    return g

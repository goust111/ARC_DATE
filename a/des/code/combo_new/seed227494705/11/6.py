import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _endpoints(comp, color, grid):
    pts = set(comp)
    ends = []
    for r, c in comp:
        neigh = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) in pts]
        if len(neigh) == 1:
            nr, nc = neigh[0]
            ends.append(((r, c), (r - nr, c - nc)))
    return ends


def _ahead(p, d, q):
    return (q[0] - p[0]) * d[0] + (q[1] - p[1]) * d[1] >= 0


def _draw(out, a, b, color):
    r, c = a
    tr, tc = b
    dr = 0 if r == tr else (1 if tr > r else -1)
    dc = 0 if c == tc else (1 if tc > c else -1)
    while (r, c) != (tr, tc):
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color
        r += dr
        c += dc
    if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
        out[tr, tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Bridge each pair of same-colored path fragments by extending their open endpoints.

    Input structure: black background with pairs of same-colored short path
    fragments and grey obstacle pixels.
    Rule: find endpoint directions from each endpoint's sole same-colored
    neighbor, choose the two forward rays of a color pair that intersect, and
    draw the straight or L-shaped same-color bridge through that intersection,
    overwriting grey obstacles if encountered.
    Key insight: endpoint orientation, not shortest distance, determines the
    bridge's corner or direct connection.
    """
    out = grid.copy()
    for color in [int(x) for x in np.unique(grid) if x not in (0, 5)]:
        comps = _components(grid == color)
        if len(comps) != 2:
            continue
        ends1 = _endpoints(comps[0], color, grid)
        ends2 = _endpoints(comps[1], color, grid)
        choice = None
        for (p1, d1) in ends1:
            for (p2, d2) in ends2:
                if p1[0] == p2[0] and d1[0] == d2[0] == 0 and _ahead(p1, d1, p2) and _ahead(p2, d2, p1):
                    choice = (p1, p2, p2)
                elif p1[1] == p2[1] and d1[1] == d2[1] == 0 and _ahead(p1, d1, p2) and _ahead(p2, d2, p1):
                    choice = (p1, p2, p2)
                elif d1[0] == 0 and d2[1] == 0:
                    inter = (p2[0], p1[1])
                    if _ahead(p1, d1, inter) and _ahead(p2, d2, inter):
                        choice = (p1, inter, p2)
                elif d1[1] == 0 and d2[0] == 0:
                    inter = (p1[0], p2[1])
                    if _ahead(p1, d1, inter) and _ahead(p2, d2, inter):
                        choice = (p1, inter, p2)
                if choice:
                    break
            if choice:
                break
        if choice:
            a, corner, b = choice
            _draw(out, a, corner, color)
            _draw(out, corner, b, color)
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(1, 3))
    used = []
    for color in colors:
        for _ in range(100):
            inter = (random.randint(6, 17), random.randint(6, 17))
            dirs = random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 2)
            if dirs[0][0] == dirs[1][0] and dirs[0][1] == dirs[1][1]:
                continue
            ok = True
            segs = []
            for d in dirs:
                length = random.randint(2, 4)
                end = (inter[0] - d[0] * random.randint(3, 5), inter[1] - d[1] * random.randint(3, 5))
                pts = [(end[0] - d[0] * i, end[1] - d[1] * i) for i in range(length)]
                if not all(1 <= r < 23 and 1 <= c < 23 and grid[r, c] == 0 for r, c in pts):
                    ok = False
                segs.append(pts)
            if ok:
                for pts in segs:
                    for r, c in pts:
                        grid[r, c] = color
                used.append(inter)
                break
    for _ in range(random.randint(4, 8)):
        r, c = random.randint(1, 22), random.randint(1, 22)
        if grid[r, c] == 0:
            grid[r, c] = 5
    return grid

import numpy as np
from collections import Counter

RNG = np.random.default_rng(606)


def _components(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = True
            comp = []
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Task 6: constrained crosses inside nested square rings.

Input:
    Nested gray rings with one colored seed in each depth region.
Output:
    Cross segments from each seed stopped by inner wall, intersections recolored.

Reasoning:
    Identify concentric ring boxes, assign depth by area order, derive seed depth from
    containment, cast 4-dir lines, then apply intersection-color based on depth diff.

Complexity:
    O(H*W + seeds * H * W).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    ring_comps = _components(g == 5)
    rings = []
    for comp in ring_comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        top, bottom = min(rs), max(rs)
        left, right = min(cs), max(cs)
        rings.append((top, bottom, left, right))

    if not rings:
        return out

    rings = sorted(rings, key=lambda x: (x[1] - x[0] + 1) * (x[3] - x[2] + 1), reverse=True)
    n = len(rings)

    seeds = []
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val in (1, 2, 3, 4):
                # region depth
                depth = n + 1
                for i, (rt, rb, cl, cr) in enumerate(rings):
                    inside = rt < r < rb and cl < c < cr
                    if not inside:
                        break
                    if i == n - 1 or not (rings[i + 1][0] < r < rings[i + 1][1] and rings[i + 1][2] < c < rings[i + 1][3]):
                        depth = i + 1
                        break
                    depth = i + 1
                seeds.append((r, c, val, depth))

    # draw each cross and collect overlaps
    hit = {}
    for sr, sc, color, depth in seeds:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = sr + dr, sc + dc
            while 0 <= r < h and 0 <= c < w and g[r, c] != 5:
                hit.setdefault((r, c), []).append((depth, color))
                r += dr
                c += dc

    for (r, c), lst in hit.items():
        if len(lst) == 1:
            out[r, c] = lst[0][1]
        else:
            depths = [x[0] for x in lst]
            d = abs(max(depths) - min(depths))
            if d == 1:
                out[r, c] = 7
            elif d == 2:
                out[r, c] = 6
            else:
                out[r, c] = 9

    return out


def generate_input() -> np.ndarray:
    h = 17
    w = 17
    g = np.zeros((h, w), dtype=int)

    n = int(RNG.integers(2, 4))
    outer = 11 if n == 2 else 15
    step = 2

    rings = []
    start = (h - outer) // 2
    for i in range(n):
        s = outer - 2 * i
        top = start + i * 2
        left = start + i * 2
        bottom = top + s - 1
        right = left + s - 1
        rings.append((top, bottom, left, right))
        g[top, left : right + 1] = 5
        g[bottom, left : right + 1] = 5
        g[top : bottom + 1, left] = 5
        g[top : bottom + 1, right] = 5

    used_rows = set()
    used_cols = set()
    seed_colors = [1, 2, 3]
    for idx, (top, bottom, left, right) in enumerate(rings + [((rings[-1][0] + 1, rings[-1][1] - 1, rings[-1][2] + 1, rings[-1][3] - 1))]):
        if idx >= len(seed_colors):
            break
        if idx < len(rings):
            cand_rows = list(range(top + 1, (rings[idx + 1][0] if idx + 1 < len(rings) else bottom)))
            cand_cols = list(range(left + 1, (rings[idx + 1][2] if idx + 1 < len(rings) else right)))
        else:
            cand_rows = list(range(top, bottom + 1))
            cand_cols = list(range(left, right + 1))

        cand_rows = [r for r in cand_rows if r not in used_rows]
        cand_cols = [c for c in cand_cols if c not in used_cols]
        if not cand_rows:
            cand_rows = [top]
        if not cand_cols:
            cand_cols = [left]

        sr = int(RNG.choice(cand_rows)) if cand_rows else top
        sc = int(RNG.choice(cand_cols)) if cand_cols else left
        g[sr, sc] = seed_colors[idx]
        used_rows.add(sr)
        used_cols.add(sc)

    return g

import numpy as np
from collections import deque, defaultdict


def _components(g):
    H, W = g.shape
    rid = np.full((H, W), -1, dtype=int)
    comps = {}
    cid = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if g[r, c] == 6 or rid[r, c] != -1:
                continue
            q = deque([(r, c)])
            rid[r, c] = cid
            comps[cid] = []
            while q:
                y, x = q.popleft()
                comps[cid].append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and g[ny, nx] != 6 and rid[ny, nx] == -1:
                        rid[ny, nx] = cid
                        q.append((ny, nx))
            cid += 1
    return rid, comps


def main(grid):
    """
    1) Find wall-separated components ignoring top/bottom bars and walls.
    2) Bipartite-color regions across wall adjacency.
    3) Recolor non-wall black cells with gray/green region colors.
    4) For each control, extend toward nearer horizontal edge by one with region color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    rid, comps = _components(g)
    if not comps:
        return out

    # build adjacency graph
    graph = defaultdict(set)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 6:
                continue
            neigh = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and rid[nr, nc] != -1:
                    neigh.append(rid[nr, nc])
            for i in range(len(neigh)):
                for j in range(i + 1, len(neigh)):
                    a, b = neigh[i], neigh[j]
                    if a != b:
                        graph[a].add(b)
                        graph[b].add(a)

    start = rid[H - 2, 1]
    if start == -1:
        start = 0
    region_color = {start: 5}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in region_color:
                region_color[v] = 4 if region_color[u] == 5 else 5
                q.append(v)

    for cid, cells in comps.items():
        col = region_color.get(cid, 5)
        for r, c in cells:
            if g[r, c] == 0:
                out[r, c] = col

    for y, x in np.argwhere(g == 3):
        y = int(y)
        x = int(x)
        d_up = y
        d_down = (H - 1) - y
        ty, tx = (y - 1, x) if d_up < d_down else (y + 1, x)
        if 0 <= ty < H and 0 <= tx < W and g[ty, tx] != 6:
            out[ty, tx] = region_color.get(int(rid[y, x]), 5)
        out[y, x] = 3

    return out


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    H = 15
    W = 15

    g = np.zeros((H, W), dtype=int)
    g[0, :] = 1
    g[H - 1, :] = 2

    v = int(variant)
    hr = 4 + (v % 7)
    vc = 4 + ((v * 2) % 7)
    g[hr, :] = 6
    g[:, vc] = 6

    rid, comps = _components(g)
    region_keys = sorted(comps.keys())
    if len(region_keys) < 3:
        return g.astype(int)

    regions = [
        (2, 2),
        (2, W - 3),
        (H - 3, 2),
        (H - 3, W - 3),
    ]

    n_ctrl = 2 + (v % 3)
    n_ctrl = min(n_ctrl, len(region_keys))
    shift = v % len(region_keys)
    ordered = [regions[(shift + i) % len(regions)] for i in range(len(regions))]
    for y, x in ordered[:n_ctrl]:
        if g[y, x] != 0:
            # very rare fallback, shift one cell to stay inside same intended region
            cand = (y, x + 1) if x + 1 < W - 1 and g[y, x + 1] == 0 else (y + 1, x)
            y, x = cand
        g[y, x] = 3

    return g.astype(int)

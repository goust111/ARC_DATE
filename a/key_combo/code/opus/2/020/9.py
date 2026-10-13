import numpy as np
from collections import deque, defaultdict


def _regions(g):
    H, W = g.shape
    rid = np.full((H, W), -1, dtype=int)
    comps = {}
    cid = 0
    for r in range(H):
        for c in range(W):
            if g[r, c] == 2 or rid[r, c] != -1:
                continue
            q = deque([(r, c)])
            rid[r, c] = cid
            comps[cid] = []
            while q:
                y, x = q.popleft()
                comps[cid].append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and g[ny, nx] != 2 and rid[ny, nx] == -1:
                        rid[ny, nx] = cid
                        q.append((ny, nx))
            cid += 1
    return rid, comps


def main(grid):
    """
    1) Bipartite color regions separated by red walls.
    2) Compute ray covered rows/cols from blue controls.
    3) For each background pixel choose: gray/green background recolor and orange/pink if covered.
    4) Controls become yellow(4).
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    rid, comps = _regions(g)
    if not comps:
        return out

    # region graph via wall adjacency
    adj = defaultdict(set)
    for r in range(H):
        for c in range(W):
            if g[r, c] != 2:
                continue
            nbs = []
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dy, c + dx
                if 0 <= nr < H and 0 <= nc < W and rid[nr, nc] != -1:
                    nbs.append(rid[nr, nc])
            for i in range(len(nbs)):
                for j in range(i + 1, len(nbs)):
                    a, b = nbs[i], nbs[j]
                    if a != b:
                        adj[a].add(b)
                        adj[b].add(a)

    start = rid[H - 2, 0]
    if start == -1:
        start = 0
    color = {start: 5}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in color:
                color[v] = 3 if color[u] == 5 else 5
                q.append(v)

    control = np.argwhere(g == 1)
    cov_rows = set(int(r) for r, c in control)
    cov_cols = set(int(c) for r, c in control)

    for r in range(H):
        for c in range(W):
            if g[r, c] == 2:
                continue
            if g[r, c] == 1:
                out[r, c] = 4
                continue
            covered = (r in cov_rows) or (c in cov_cols)
            sid = int(rid[r, c])
            s = color.get(sid, 5)
            if covered:
                out[r, c] = 7 if s == 5 else 6
            else:
                out[r, c] = s

    return out


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    v = int(variant)
    H = 15
    W = 15

    g = np.zeros((H, W), dtype=int)
    hr = 4 + ((v * 3) % 8)
    vc = 4 + ((v * 5) % 7)

    g[hr, 1:W - 1] = 2
    g[1:H - 1, vc] = 2

    controls = [
        (2, 2),
        (2, W - 3),
        (H - 3, 2),
        (H - 3, W - 3),
    ]
    n_ctrl = 2 + (v % 3)
    shift = v % 4
    for y, x in controls[shift:shift + n_ctrl]:
        g[y, x] = 1
    if n_ctrl > (4 - shift):
        for y, x in controls[: (n_ctrl - (4 - shift))]:
            g[y, x] = 1

    return g.astype(int)

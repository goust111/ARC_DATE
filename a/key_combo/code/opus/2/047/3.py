import numpy as np
from collections import deque


def _components(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    out = np.array(g, dtype=np.int64)

    colors = [c for c in np.unique(g) if c != 0]
    coords_by_color = {c: np.argwhere(g == c) for c in colors}

    # draw horizontal/vertical pair lines first (horizontal first), then overwrite with vertical
    for c in colors:
        pts = coords_by_color[c]
        if len(pts) < 2:
            continue
        (r0, c0), (r1, c1) = map(tuple, pts[:2])
        if r0 == r1:
            cc0, cc1 = sorted((c0, c1))
            out[r0, cc0 : cc1 + 1] = c
        elif c0 == c1:
            rr0, rr1 = sorted((r0, r1))
            out[rr0 : rr1 + 1, c0] = c

    for c in colors:
        pts = coords_by_color[c]
        if len(pts) < 2:
            continue
        (r0, c0), (r1, c1) = map(tuple, pts[:2])
        if c0 == c1:
            rr0, rr1 = sorted((r0, r1))
            out[rr0 : rr1 + 1, c0] = c

    # enclosed regions are black components not connected to border
    black_mask = out == 0
    vis = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if 0 <= c < w and black_mask[r, c] and not vis[r, c]:
                q.append((r, c))
                vis[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and black_mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
    for r in range(h):
        for c in range(w):
            if not (out[r, c] == 0 and not vis[r, c]):
                continue
            # component flood
            comp = []
            qq = deque([(r, c)])
            vis[r, c] = True
            while qq:
                cr, cc = qq.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0 and not vis[nr, nc]:
                        vis[nr, nc] = True
                        qq.append((nr, nc))

            comp_arr = np.array(comp, dtype=int)
            rmin, cmin = comp_arr.min(axis=0)
            rmax, cmax = comp_arr.max(axis=0)
            left_colors = []
            for rr, cc in comp:
                if cc - 1 >= 0 and out[rr, cc - 1] != 0:
                    left_colors.append(int(out[rr, cc - 1]))
            if not left_colors:
                continue
            fill_color = left_colors[0]
            out[comp_arr[:, 0], comp_arr[:, 1]] = fill_color

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 11
    w = 11
    out = np.zeros((h, w), dtype=np.int64)

    color_pool = [1, 2, 3, 4]
    h1 = int(rng.integers(2, 6))
    h2 = h1 + int(rng.integers(3, 6))
    if h2 > h - 3:
        h2 = h - 3
        h1 = h2 - 3
    v1 = int(rng.integers(3, 6))
    v2 = int(rng.integers(6, 9))
    if v2 >= w - 2:
        v2 = w - 2
        v1 = max(3, v2 - 3)

    # color1: horizontal segment, color2: vertical segment (crossing)
    out[h1, 1] = color_pool[0]
    out[h1, w - 2] = color_pool[0]
    out[1, v1] = color_pool[1]
    out[h - 2, v1] = color_pool[1]

    # color3: another horizontal, color4: another vertical
    out[h2, 2] = color_pool[2]
    out[h2, w - 3] = color_pool[2]
    out[2, v2] = color_pool[3]
    out[h - 3, v2] = color_pool[3]

    return out


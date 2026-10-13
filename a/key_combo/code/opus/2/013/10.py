import numpy as np

RNG = np.random.default_rng(1010)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            st = [(r, c)]
            vis[r, c] = True
            comp = []
            while st:
                x, y = st.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        st.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Task 10: extrapolate frame sequence and compress to nested rings.

Input:
    Multiple gray-separated panels each with blue sprite.
Output:
    Square nested rings representing area ranking over extrapolated sequence.

Reasoning:
    Parse equal-width panels and sprite areas, extrapolate next frame by drift,
    rank frame areas to colors, then draw nested borders.

Complexity:
    O(H*W + N^2).
    """

    g = np.array(grid, dtype=int)
    H, W = g.shape

    sep = [c for c in range(W) if np.all(g[:, c] == 5)]
    segs = []
    start = 0
    for c in range(W):
        if c in sep:
            if start <= c - 1:
                segs.append((start, c - 1))
            start = c + 1
    if start < W:
        segs.append((start, W - 1))

    panels = []
    for s, e in segs:
        sub = g[:, s : e + 1]
        pts = list(zip(*np.where(sub == 1)))
        if not pts:
            continue
        top = min(r for r, _ in pts)
        panels.append((s, e, top, pts))

    if not panels:
        return np.zeros((1, 1), dtype=int)

    N = len(panels)
    tops = [p[2] for p in panels]
    if N >= 2:
        dr = tops[1] - tops[0]
    else:
        dr = 0

    base_s, base_e, base_top, base_pts = panels[0]
    pw = base_e - base_s + 1

    sprite = [(r - base_top, c) for r, c in base_pts]
    base_next_top = tops[-1] + dr

    areas = []
    for _, _, _, pts in panels:
        areas.append(len(pts))

    # extrapolated frame
    next_cnt = 0
    for dr0, dc0 in sprite:
        nr = base_next_top + dr0
        nc = dc0
        if 0 <= nc < pw and 0 <= nr < H:
            next_cnt += 1
    areas.append(next_cnt)

    colors = [2, 3, 4, 6, 7]
    order = sorted(range(len(areas)), key=lambda i: (-areas[i], i))
    color_by_frame = {idx: colors[k] for k, idx in enumerate(order)}

    S = 2 * (N + 1) - 1
    out = np.zeros((S, S), dtype=int)

    for layer in range(N + 1):
        c = color_by_frame[layer]
        for x in range(layer, S - layer):
            out[layer, x] = c
            out[S - 1 - layer, x] = c
            out[x, layer] = c
            out[x, S - 1 - layer] = c

    return out


def generate_input() -> np.ndarray:
    H = 12
    panel_n = int(RNG.choice([3, 4]))
    pw = 4
    sep = 1

    base_w = panel_n * pw + max(0, panel_n - 1) * sep
    W = 20
    g = np.zeros((H, W), dtype=int)

    base_shape = []
    n_pts = int(RNG.integers(3, 6))
    while len(base_shape) < n_pts:
        if not base_shape:
            base_shape.append((0, 1))
        else:
            r0, c0 = base_shape[int(RNG.integers(len(base_shape)))]
            nr, nc = r0 + int(RNG.choice([-1, 0, 1])), c0 + int(RNG.choice([-1, 0, 1]))
            if abs(nr - r0) + abs(nc - c0) != 1:
                continue
            if 0 <= nr < H and 0 <= nc < pw:
                if (nr, nc) not in base_shape:
                    base_shape.append((nr, nc))

    base_top = int(RNG.integers(1, H - 4))
    dr = int(RNG.choice([-2, -1, 1, 2]))

    for i in range(panel_n):
        s = i * (pw + sep)
        if i > 0 and s - 1 < W:
            g[:, s - 1] = 5

        for r_rel, c_rel in base_shape:
            rr = base_top + i * dr + r_rel
            cc = s + c_rel
            if 0 <= rr < H and cc < W:
                g[rr, cc] = 1

    return g

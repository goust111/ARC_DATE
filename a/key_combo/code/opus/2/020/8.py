import numpy as np
from collections import deque


def _components_mask(mask):
    h, w = mask.shape
    vis = np.full((h, w), -1, dtype=int)
    comps = {}
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c] != -1:
                continue
            q = deque([(r, c)])
            vis[r, c] = cid
            comps[cid] = []
            while q:
                y, x = q.popleft()
                comps[cid].append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and vis[ny, nx] == -1:
                        vis[ny, nx] = cid
                        q.append((ny, nx))
            cid += 1
    return vis, comps


def _find_frames(g):
    out = []
    for color in [c for c in np.unique(g) if c not in (0, 3, 5)]:
        ys, xs = np.where(g == color)
        if len(ys) == 0:
            continue
        r1, r2 = int(ys.min()), int(ys.max())
        c1, c2 = int(xs.min()), int(xs.max())
        if r2 - r1 < 2 or c2 - c1 < 2:
            continue
        if not (np.all(g[r1, c1 : c2 + 1] == color) and
                np.all(g[r2, c1 : c2 + 1] == color) and
                np.all(g[r1 : r2 + 1, c1] == color) and
                np.all(g[r1 : r2 + 1, c2] == color)):
            continue
        out.append((r1, r2, c1, c2, color))
    return out


def main(grid):
    """
    1) Detect each gray hollow frame and inside gray marker count N.
    2) Find inner green rectangle, derive stepping direction from nearest corner.
    3) Place N green stamps; reverse direction when hitting frame border.
    4) Leave borders and other areas unchanged.
    """
    g = np.array(grid, dtype=int).copy()

    for r1, r2, c1, c2, _ in _find_frames(g):
        inner = g[r1 + 1 : r2, c1 + 1 : c2]
        if inner.size == 0:
            continue
        N = int(np.sum(inner == 5))
        if N == 0:
            continue

        # find green rectangle component
        vis, comps = _components_mask(inner == 3)
        if not comps:
            continue
        c0 = max(comps.keys(), key=lambda k: len(comps[k]))
        cells = comps[c0]
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        sr1, sr2 = min(rs), max(rs)
        sc1, sc2 = min(cs), max(cs)
        gh = sr2 - sr1 + 1
        gw = sc2 - sc1 + 1

        ih, iw = inner.shape
        # start near corner from component top-left
        gr = sr1
        gc = sc1
        corners = [
            (0, 0),
            (0, iw - gw),
            (ih - gh, 0),
            (ih - gh, iw - gw),
        ]
        d = []
        for tr, tc in corners:
            d.append(abs(tr - gr) + abs(tc - gc))
        idx = int(np.argmin(d))
        if idx == 0:
            dr, dc = gh, gw
        elif idx == 1:
            dr, dc = gh, -gw
        elif idx == 2:
            dr, dc = -gh, gw
        else:
            dr, dc = -gh, -gw

        r0 = int(gr)
        c0 = int(gc)

        # ensure global coordinates by offset
        R0 = r1 + 1
        C0 = c1 + 1
        for _ in range(N):
            tr = r0 + dr
            tc = c0 + dc
            if tr < 0 or tc < 0 or tr + gh > ih or tc + gw > iw:
                if tr < 0 or tr + gh > ih:
                    dr *= -1
                if tc < 0 or tc + gw > iw:
                    dc *= -1
                tr = r0 + dr
                tc = c0 + dc
            r0, c0 = tr, tc
            g[R0 + r0 : R0 + r0 + gh, C0 + c0 : C0 + c0 + gw] = 3

    return g


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 29 + 53)
    H = 22
    W = 22
    g = np.zeros((H, W), dtype=int)

    n_frames = int(rng.integers(2, 4))
    colors = [1, 2, 3, 4, 6, 9]
    rng.shuffle(colors)
    boxes = []

    for idx in range(n_frames):
        ih = int(rng.integers(8, 12))
        iw = int(rng.integers(8, 12))
        h2 = ih + 2
        w2 = iw + 2
        color = colors[idx]
        for _ in range(200):
            r1 = int(rng.integers(0, H - h2))
            c1 = int(rng.integers(0, W - w2))
            r2 = r1 + h2 - 1
            c2 = c1 + w2 - 1
            if any(not (r2 + 2 <= ar1 or ar2 + 2 <= r1 or c2 + 2 <= ac1 or ac2 + 2 <= c1)
                   for ar1, ar2, ac1, ac2 in boxes):
                continue

            g[r1 : r2 + 1, c1 : c2 + 1] = 0
            g[r1, c1 : c2 + 1] = color
            g[r2, c1 : c2 + 1] = color
            g[r1 : r2 + 1, c1] = color
            g[r1 : r2 + 1, c2] = color

            N = (idx + 2)
            # markers unique counts
            cand = [(r1 + 1 + y, c1 + 1 + x) for y in range(1, ih - 1) for x in range(1, iw - 1)]
            rng.shuffle(cand)
            for p in cand[:N]:
                g[p] = 5

            # green rectangle near corner
            hrect = int(rng.integers(2, 4))
            wrect = int(rng.integers(2, 4))
            corner = int(rng.integers(0, 4))
            if corner == 0:
                sr = r1 + 1
                sc = c1 + 1
            elif corner == 1:
                sr = r1 + 1
                sc = c2 - wrect
            elif corner == 2:
                sr = r2 - hrect
                sc = c1 + 1
            else:
                sr = r2 - hrect
                sc = c2 - wrect
            g[sr : sr + hrect, sc : sc + wrect] = 3

            boxes.append((r1, r2, c1, c2))
            break

    return g

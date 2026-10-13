import numpy as np
from collections import deque


def _rectangles(g):
    rects = []
    for color in [c for c in np.unique(g) if c not in (0, 2, 7)]:
        ys, xs = np.where(g == color)
        if len(ys) == 0:
            continue
        r1, r2 = int(ys.min()), int(ys.max())
        c1, c2 = int(xs.min()), int(xs.max())
        if r2 - r1 < 4 or c2 - c1 < 4:
            continue
        if not (np.all(g[r1, c1 : c2 + 1] == color) and
                np.all(g[r2, c1 : c2 + 1] == color) and
                np.all(g[r1 : r2 + 1, c1] == color) and
                np.all(g[r1 : r2 + 1, c2] == color)):
            continue
        rects.append((r1, r2, c1, c2, color))
    return rects


def _red_components(sub):
    h, w = sub.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if sub[y, x] != 2 or vis[y, x]:
                continue
            q = deque([(y, x)])
            vis[y, x] = True
            comp = []
            while q:
                yy, xx = q.popleft()
                comp.append((yy, xx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and sub[ny, nx] == 2 and not vis[ny, nx]:
                        vis[ny, nx] = True
                        q.append((ny, nx))
            comps.append(comp)
    return comps


def main(grid):
    """
    1) Count ray lines (rows+cols) from orange control markers through every rectangle.
    2) Choose rectangle with maximal count K.
    3) Keep only red components of area exactly K; recolor others to fill color.
    4) Return cropped selected rectangle.
    """
    g = np.array(grid, dtype=int)
    rects = _rectangles(g)
    if not rects:
        return g

    control_pos = np.argwhere(g == 7)
    control_rows = sorted(set(int(r) for r, _ in control_pos)) if len(control_pos) else []
    control_cols = sorted(set(int(c) for _, c in control_pos)) if len(control_pos) else []

    best_k = -1
    best = None
    for r1, r2, c1, c2, color in rects:
        hr = [r for r in control_rows if r1 <= r <= r2]
        hc = [c for c in control_cols if c1 <= c <= c2]
        k = len(hr) + len(hc)
        if k > best_k:
            best_k = k
            best = (r1, r2, c1, c2, color)

    if best is None or best_k < 1:
        return g

    r1, r2, c1, c2, color = best
    sub = g[r1 : r2 + 1, c1 : c2 + 1].copy()
    for comp in _red_components(sub):
        if len(comp) != best_k:
            for y, x in comp:
                sub[y, x] = color
    return sub


def _place_rectangles(g, rng):
    H, W = g.shape
    colors = [1, 3, 4, 6, 9]
    rng.shuffle(colors)
    n = int(rng.integers(3, 5))
    used = np.zeros((H, W), dtype=bool)
    rects = []

    for i in range(n):
        color = colors[i]
        for _ in range(120):
            h = int(rng.integers(5, 8))
            w = int(rng.integers(5, 8))
            r1 = int(rng.integers(1, H - h - 1))
            c1 = int(rng.integers(1, W - w - 1))
            r2 = r1 + h - 1
            c2 = c1 + w - 1
            if used[r1 - 1 : r2 + 2, c1 - 1 : c2 + 2].any():
                continue
            g[r1 : r2 + 1, c1 : c2 + 1] = color
            used[r1 : r2 + 1, c1 : c2 + 1] = True
            rects.append((r1, r2, c1, c2, color))
            break
    return rects


def _paint_rect_component(grid, top, left, h, w, cells, color):
    for y, x in cells:
        grid[top + y, left + x] = color


def _random_path_cells(h, w, area, rng):
    cells = [(0, 0)]
    used = {(0, 0)}
    y, x = 0, 0
    for _ in range(area - 1):
        options = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in used:
                options.append((ny, nx))
        if not options:
            break
        y, x = options[int(rng.integers(0, len(options)))]
        used.add((y, x))
        cells.append((y, x))
    return cells


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 17 + 27)
    H, W = 22, 22

    for _ in range(200):
        g = np.zeros((H, W), dtype=int)
        rects = _place_rectangles(g, rng)
        if len(rects) < 3:
            continue

        # controls
        for _ in range(3):
            for _ in range(100):
                y = int(rng.integers(1, H - 1))
                x = int(rng.integers(1, W - 1))
                if g[y, x] == 0:
                    g[y, x] = 7
                    break

        control_pos = np.argwhere(g == 7)
        rows = sorted(set(int(r) for r, _ in control_pos))
        cols = sorted(set(int(c) for _, c in control_pos))

        counts = []
        for r1, r2, c1, c2, _ in rects:
            hr = len([r for r in rows if r1 <= r <= r2])
            hc = len([c for c in cols if c1 <= c <= c2])
            counts.append(hr + hc)

        if len(set(counts)) != len(rects):
            continue
        k = int(max(counts))
        if k < 1 or k > 6:
            continue

        max_i = int(np.argmax(counts))
        r1, r2, c1, c2, col = rects[max_i]
        h = r2 - r1 + 1
        w = c2 - c1 + 1

        # clear red then place for each rect
        for (rr1, rr2, cc1, cc2, ccol) in rects:
            g[rr1 : rr2 + 1, cc1 : cc2 + 1] = ccol

        rr1, rr2, cc1, cc2, ccol = rects[max_i]
        h = rr2 - rr1 + 1
        w = cc2 - cc1 + 1

        # one component area==k
        a1 = k
        cells = _random_path_cells(h - 2, w - 2, a1, rng)
        for y, x in cells:
            g[rr1 + 1 + y, cc1 + 1 + x] = 2
        # one component different area
        a2 = k + 1 if k < 6 else k - 1
        cells2 = _random_path_cells(h - 2, w - 2, min(a2, h * w), rng)
        for y, x in cells2:
            if g[rr1 + 1 + y, cc1 + 1 + x] == 0:
                g[rr1 + 1 + y, cc1 + 1 + x] = 2

        # ensure each rectangle has red pixels
        for ri, (yy1, yy2, xx1, xx2, ccol) in enumerate(rects):
            h2 = yy2 - yy1 + 1
            w2 = xx2 - xx1 + 1
            if np.all(g[yy1 + 1 : yy2, xx1 + 1 : xx2] != 2):
                r = int(rng.integers(yy1 + 1, yy2))
                c = int(rng.integers(xx1 + 1, xx2))
                g[r, c] = 2

        return g.astype(int)

    # fallback
    return np.zeros((22, 22), dtype=int)

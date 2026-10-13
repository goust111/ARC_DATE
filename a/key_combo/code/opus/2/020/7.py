import numpy as np
from collections import deque


def _components(g):
    h, w = g.shape
    vis = np.full((h, w), -1, dtype=int)
    comps = {}
    cid = 0
    for r in range(h):
        for c in range(w):
            if g[r, c] == 0 or g[r, c] == 5 or vis[r, c] != -1:
                continue
            q = deque([(r, c)])
            vis[r, c] = cid
            comps[cid] = []
            while q:
                y, x = q.popleft()
                comps[cid].append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and g[ny, nx] != 0 and g[ny, nx] != 5 and vis[ny, nx] == -1:
                        vis[ny, nx] = cid
                        q.append((ny, nx))
            cid += 1
    return vis, comps


def _find_frames(g):
    out = []
    for color in [c for c in np.unique(g) if c not in (0, 5, 3)]:
        ys, xs = np.where(g == color)
        if len(ys) == 0:
            continue
        r1, r2 = int(ys.min()), int(ys.max())
        c1, c2 = int(xs.min()), int(xs.max())
        if r2 - r1 < 3 or c2 - c1 < 3:
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
    1) Count components touching the outside of each gray frame.
    2) Pick frame with maximum count.
    3) Mirror its interior horizontally.
    4) Repaint black halves: left half with right-side sprite color, right half with left-side color.
    """
    g = np.array(grid, dtype=int).copy()
    vis, comps = _components(g)
    frames = _find_frames(g)
    if not frames:
        return g

    best = None
    best_cnt = -1
    for frame in frames:
        r1, r2, c1, c2, _ = frame
        touch = set()
        # outside neighbors from four edges
        for c in range(c1 + 1, c2):
            for rr, cc in ((r1 - 1, c), (r2 + 1, c)):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] != 0 and g[rr, cc] != 5:
                    cid = vis[rr, cc]
                    if cid != -1:
                        touch.add(cid)
        for r in range(r1 + 1, r2):
            for rr, cc in ((r, c1 - 1), (r, c2 + 1)):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] != 0 and g[rr, cc] != 5:
                    cid = vis[rr, cc]
                    if cid != -1:
                        touch.add(cid)
        if len(touch) > best_cnt:
            best_cnt = len(touch)
            best = frame

    if best is None:
        return g

    r1, r2, c1, c2, _ = best
    inner = g[r1 + 1 : r2, c1 + 1 : c2]
    if inner.size == 0:
        return g
    h, w = inner.shape

    # side sprite colors
    left_colors = []
    right_colors = []
    for r in range(r1 + 1, r2):
        cl = c1 - 1
        cr = c2 + 1
        if 0 <= cl < g.shape[1] and g[r, cl] not in (0, 5):
            left_colors.append(int(g[r, cl]))
        if 0 <= cr < g.shape[1] and g[r, cr] not in (0, 5):
            right_colors.append(int(g[r, cr]))
    for c in range(c1 + 1, c2):
        ut = r1 - 1
        dt = r2 + 1
        if 0 <= ut < g.shape[0] and g[ut, c] not in (0, 5):
            left_colors.append(int(g[ut, c]))
        if 0 <= dt < g.shape[0] and g[dt, c] not in (0, 5):
            right_colors.append(int(g[dt, c]))
    c_left = left_colors[0] if left_colors else 3
    c_right = right_colors[0] if right_colors else 2

    mirror = np.zeros_like(inner)
    for y in range(h):
        for x in range(w):
            if inner[y, x] != 0:
                mirror[y, w - 1 - x] = inner[y, x]

    res = mirror.copy()
    mid = (w - 1) // 2
    for y in range(h):
        for x in range(w):
            if res[y, x] == 0:
                if x <= mid:
                    res[y, x] = c_right
                else:
                    res[y, x] = c_left

    return res


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 13 + 41)
    H = 20
    W = 20
    g = np.zeros((H, W), dtype=int)

    n_frames = int(rng.integers(2, 4))
    colors = [1, 2, 3, 4, 6, 9]
    rng.shuffle(colors)
    placed = []

    for idx in range(n_frames):
        hi = int(rng.integers(6, 9))
        wi = int(rng.integers(4, 7))
        h2 = hi + 2
        w2 = wi + 2
        color = colors[idx]
        for _ in range(200):
            r1 = int(rng.integers(0, H - h2))
            c1 = int(rng.integers(0, W - w2))
            r2 = r1 + h2 - 1
            c2 = c1 + w2 - 1
            if any(not (r2 + 2 <= pr1 or pr2 + 2 <= r1 or c2 + 2 <= pc1 or pc2 + 2 <= c1) for pr1, pr2, pc1, pc2, _ in placed):
                continue
            g[r1 : r2 + 1, c1 : c2 + 1] = 0
            g[r1, c1 : c2 + 1] = color
            g[r2, c1 : c2 + 1] = color
            g[r1 : r2 + 1, c1] = color
            g[r1 : r2 + 1, c2] = color
            # interior pattern
            for _ in range(6):
                pr = int(rng.integers(r1 + 1, r2))
                pc = int(rng.integers(c1 + 1, c2))
                g[pr, pc] = int(rng.choice([1, 2, 3]))

            # place outside sprites count for this frame
            target_count = idx + 1
            candidates = []
            for r in range(r1 - 1, r2 + 2):
                for c in [c1 - 1, c2 + 1]:
                    if 0 <= r < H and 0 <= c < W and g[r, c] == 0:
                        candidates.append((r, c))
            for c in range(c1, c2 + 1):
                for r in [r1 - 1, r2 + 1]:
                    if 0 <= r < H and 0 <= c < W and g[r, c] == 0:
                        candidates.append((r, c))
            rng.shuffle(candidates)
            for j in range(min(target_count, len(candidates))):
                g[candidates[j]] = int(rng.choice([1, 2, 3]))

            placed.append((r1, r2, c1, c2, color))
            break

    # enforce max frame with exact one left/right sprite
    if placed:
        r1, r2, c1, c2, _ = placed[0]
        mr = (r1 + r2) // 2
        if c1 - 1 >= 0:
            g[mr, c1 - 1] = 2
        if c2 + 1 < W:
            g[mr, c2 + 1] = 3

    return g

import numpy as np


def _detect_frames(g):
    h, w = g.shape
    frames = []
    for r in range(h - 2):
        for c in range(w - 2):
            b = g[r : r + 3, c : c + 3]
            if b.shape != (3, 3):
                continue
            if b[1, 1] != 0:
                continue
            col = int(b[0, 0])
            if col in (0, 5):
                continue
            if not (b[0, 0] == col and b[0, 1] == col and b[0, 2] == col and
                    b[1, 0] == col and b[1, 2] == col and
                    b[2, 0] == col and b[2, 1] == col and b[2, 2] == col):
                continue
            if b[1, 1] != 0:
                continue
            frames.append((r, c, col))
    return frames


def main(grid):
    """
    1) Detect all 3x3 hollow frames.
    2) Assign each gray marker to nearest frame (tie row then col).
    3) Sort frames by assigned marker count and paint cross by rank colors.
    4) Erase all gray markers and return recolored grid.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(out)
    if not frames:
        return out

    centers = [(r + 1, c + 1, i) for i, (r, c, _) in enumerate(frames)]
    counts = [0 for _ in centers]
    for y, x in np.argwhere(g == 5):
        y = int(y)
        x = int(x)
        best = None
        best_d = None
        best_i = None
        best_r = None
        best_c = None
        for fr, fc, i in centers:
            d = abs(y - fr) + abs(x - fc)
            if best is None or d < best_d or (d == best_d and (fr < best_r or (fr == best_r and fc < best_c))):
                best = (fr, fc)
                best_d = d
                best_i = i
                best_r = fr
                best_c = fc
        counts[best_i] += 1

    order = sorted(range(len(frames)), key=lambda i: (counts[i], i))
    rank = {idx: k + 1 for k, idx in enumerate(order)}
    palette = {1: 1, 2: 2, 3: 3, 4: 4, 5: 7}

    for i, (r, c, col) in enumerate(frames):
        rr = i + 1
        rc = palette.get(rank[i], 7)
        out[r : r + 3, c : c + 3] = 0
        out[r, c] = 0
        out[r, c + 1] = rc
        out[r, c + 2] = 0
        out[r + 1, c] = rc
        out[r + 1, c + 1] = rc
        out[r + 1, c + 2] = rc
        out[r + 2, c] = 0
        out[r + 2, c + 1] = rc
        out[r + 2, c + 2] = 0

    out[g == 5] = 0
    return out


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 47 + 3)
    H = 18
    W = 18
    g = np.zeros((H, W), dtype=int)

    n_frames = int(rng.integers(3, 6))
    colors = [1, 2, 3, 4, 6]
    rng.shuffle(colors)
    colors = colors[:n_frames]

    centers = []
    # place 3x3 frames with spacing >=2
    occupied = np.zeros((H, W), dtype=bool)
    for i in range(n_frames):
        for _ in range(400):
            r = int(rng.integers(0, H - 3))
            c = int(rng.integers(0, W - 3))
            if occupied[max(0, r - 2) : min(H, r + 5), max(0, c - 2) : min(W, c + 5)].any():
                continue
            color = colors[i]
            g[r, c : c + 3] = color
            g[r + 2, c : c + 3] = color
            g[r : r + 3, c] = color
            g[r : r + 3, c + 2] = color
            g[r, c + 1] = color
            g[r + 2, c + 1] = color
            g[r + 1, c] = color
            g[r + 1, c + 2] = color
            occupied[r : r + 3, c : c + 3] = True
            centers.append((r + 1, c + 1, i))
            break

    if len(centers) < n_frames:
        n_frames = len(centers)

    counts = list(range(1, n_frames + 1))
    rng.shuffle(counts)
    owner_cells = {}
    for y in range(H):
        for x in range(W):
            if g[y, x] != 0:
                continue
            best = None
            best_d = 10**9
            best_i = -1
            best_r = 10**9
            best_c = 10**9
            for fr, fc, i in centers:
                d = abs(y - fr) + abs(x - fc)
                if d < best_d or (d == best_d and (fr < best_r or (fr == best_r and fc < best_c))):
                    best_d = d
                    best_r = fr
                    best_c = fc
                    best_i = i
            if best_i >= 0:
                owner_cells.setdefault(best_i, []).append((y, x))

    for i in range(n_frames):
        cnt = counts[i] if i < len(counts) else 1
        cells = owner_cells.get(i, [])
        rng.shuffle(cells)
        for p in cells[:cnt]:
            g[p] = 5

    return g.astype(int)

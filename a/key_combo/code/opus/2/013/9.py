import numpy as np
from collections import deque

RNG = np.random.default_rng(909)


def _count_regions(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    counts = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            counts += 1
            q = deque([(r, c)])
            vis[r, c] = True
            while q:
                x, y = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """Task 9: mirror tiling + component histogram filling.

Input:
    Upper tile + separator + marker segment.
Output:
    Mirror-tiling summary: mirrored pattern with bars under matched component-count colors.

Reasoning:
    Extract top tile, build 2x2 mirror tiling, count 4-neighbor regions of each color,
    then draw bars of height k under columns matching marker length.

Complexity:
    O(H*W).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    sep_row = int(np.where(np.all(g == 5, axis=1))[0][0]) if np.any(np.all(g == 5, axis=1)) else None
    if sep_row is None:
        return out

    top = g[:sep_row, :]
    nz = np.argwhere(top > 0)
    if nz.size == 0:
        return out
    min_r, min_c = nz.min(axis=0)
    max_r, max_c = nz.max(axis=0)
    tile = top[min_r : max_r + 1, min_c : max_c + 1]

    h0, w0 = tile.shape
    m2 = np.fliplr(tile)
    m3 = np.flipud(tile)
    m4 = np.flipud(np.fliplr(tile))
    mirror = np.block([[tile, m2], [m3, m4]])

    # write tiled area
    out[: 2 * h0, : 2 * w0] = mirror
    out[sep_row, : 2 * w0] = 5

    k = int(np.sum(g[sep_row + 1, : (w)] == 7))

    colors = [1, 2, 3, 4]
    counts = {}
    for c in colors:
        counts[c] = _count_regions(mirror == c)

    for c in colors:
        if counts[c] != k:
            continue
        cols = [j for j in range(2 * w0) if mirror[2 * h0 - 1, j] == c]
        for cc in cols:
            for rr in range(sep_row + 2, min(h, sep_row + 2 + k)):
                out[rr, cc] = c

    return out


def generate_input() -> np.ndarray:
    while True:
        w0 = int(RNG.integers(5, 8))
        h0 = int(RNG.integers(5, 8))

        # generate top-left tile in a 2*w0 wide canvas
        W = 2 * w0
        H = 2 * h0 + 2 + 4
        g = np.zeros((H, W), dtype=int)

        # build sparse top pattern
        tile = np.zeros((h0, w0), dtype=int)
        colors = [1, 2, 3, 4]
        for i in range(h0):
            for j in range(w0):
                if RNG.random() < 0.45:
                    tile[i, j] = int(RNG.choice(colors))

        if np.all(tile == 0):
            continue

        # ensure 2-3 colors
        present = [c for c in colors if np.any(tile == c)]
        if len(present) < 2:
            tile[RNG.integers(0, h0), RNG.integers(0, w0)] = int(colors[1])

        # mirror counts
        m2 = np.fliplr(tile)
        m3 = np.flipud(tile)
        m4 = np.flipud(np.fliplr(tile))
        mirror = np.block([[tile, m2], [m3, m4]])

        counts = {}
        for c in colors:
            counts[c] = _count_regions(mirror == c)

        k = int(RNG.integers(1, 5))
        if k in counts.values() and any(v != k for v in counts.values()):
            pass
        else:
            continue

        g[:h0, :w0] = tile
        sep = h0
        g[sep, :] = 5
        g[sep + 1, 0:k] = 7

        return g

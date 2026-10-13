import numpy as np


def _flip_tile(tile: np.ndarray, r: int, c: int) -> np.ndarray:
    t = tile.copy()
    if r != 1:
        t = np.flipud(t)
    if c != 1:
        t = np.fliplr(t)
    return t


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    m_h = h // 3
    m_w = w // 3
    m = min(m_h, m_w)
    tiles = []
    for r in range(3):
        for c in range(3):
            tiles.append(g[r * m : (r + 1) * m, c * m : (c + 1) * m])
    center = tiles[4].copy()

    out = np.zeros_like(center)
    mismatch = np.zeros((m, m), dtype=np.int64)

    for r in range(3):
        for c in range(3):
            base = _flip_tile(center, r, c)
            slot = tiles[r * 3 + c]
            mismatch |= (slot != base)

    out[mismatch] = 2
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    m = int(rng.integers(3, 6))
    h = w = 3 * m
    palette = [1, 2, 3, 4, 7]

    while True:
        tile = np.zeros((m, m), dtype=np.int64)
        for r in range(m):
            for c in range(m):
                if rng.random() < 0.45:
                    tile[r, c] = int(rng.choice(palette))
        if np.count_nonzero(tile) >= 2:
            a = np.flipud(tile)
            b = np.fliplr(tile)
            if not np.array_equal(tile, a) and not np.array_equal(tile, b):
                break

    def mirrored(idx_r: int, idx_c: int):
        t = tile.copy()
        if idx_r != 1:
            t = np.flipud(t)
        if idx_c != 1:
            t = np.fliplr(t)
        return t

    grid = np.zeros((h, w), dtype=np.int64)
    used = [[False for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            t = mirrored(r, c)
            grid[r * m : (r + 1) * m, c * m : (c + 1) * m] = t
            used[r][c] = True

    corrupt_r = int(rng.integers(0, 3))
    corrupt_c = int(rng.integers(0, 3))
    while corrupt_r == 1 and corrupt_c == 1:
        corrupt_r = int(rng.integers(0, 3))
        corrupt_c = int(rng.integers(0, 3))

    rs = slice(corrupt_r * m, (corrupt_r + 1) * m)
    cs = slice(corrupt_c * m, (corrupt_c + 1) * m)
    k = int(rng.integers(1, 4))
    for _ in range(k):
        rr = int(rng.integers(rs.start, rs.stop))
        cc = int(rng.integers(cs.start, cs.stop))
        cur = grid[rr, cc]
        choices = [x for x in palette if x != cur]
        if len(choices) == 0:
            choices = palette
        grid[rr, cc] = int(rng.choice(choices))
    return grid

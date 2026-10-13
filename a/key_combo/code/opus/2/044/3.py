import numpy as np


def _compress_by_blocks(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    bh, bw = h // 3, w // 3
    t = np.zeros((3, 3), dtype=np.int64)
    for r in range(3):
        for c in range(3):
            block = grid[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            cells = np.argwhere(block != 0)
            if len(cells) == 0:
                t[r, c] = 0
            else:
                t[r, c] = int(block[cells[0][0], cells[0][1]])
    return t


def _lr_flip(tile: np.ndarray) -> np.ndarray:
    return tile[:, ::-1].copy()


def _ud_flip(tile: np.ndarray) -> np.ndarray:
    return tile[::-1, :].copy()


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    t = _compress_by_blocks(g)

    symmetric = True
    for r in range(3):
        for c in range(3):
            if t[r, c] != t[c, r]:
                symmetric = False
                break
        if not symmetric:
            break

    out = np.zeros((6, 6), dtype=np.int64)
    if symmetric:
        tr = _lr_flip(t)
    else:
        tr = _ud_flip(t)

    out[0:3, 0:3] = t
    out[0:3, 3:6] = tr
    out[3:6, 0:3] = tr
    out[3:6, 3:6] = t
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.choice([12, 15]))
    w = h
    bh, bw = h // 3, w // 3
    grid = np.zeros((h, w), dtype=np.int64)
    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for br in range(3):
        for bc in range(3):
            r0 = br * bh
            c0 = bc * bw
            r = int(rng.integers(r0, r0 + bh))
            c = int(rng.integers(c0, c0 + bw))
            grid[r, c] = int(rng.choice(palette))

    return grid

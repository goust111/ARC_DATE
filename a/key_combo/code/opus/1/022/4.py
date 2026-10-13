import random
import numpy as np


def _states(panel, color):
    seq = []
    for c in range(panel.shape[1]):
        seq.append(tuple(np.where(panel[:, c] == color)[0].tolist()))
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    h, w = grid.shape
    sep_rows = np.where(np.all(grid == 7, axis=1))[0]
    if len(sep_rows) == 0:
        return grid.copy()
    sr = int(sep_rows[0])
    top = grid[:sr, :]
    bottom = grid[sr + 1:, :]
    sep_cols = np.where(np.all(bottom == 5, axis=0))[0]
    if len(sep_cols) == 0:
        return bottom.copy()
    sc = int(sep_cols[0])
    left = bottom[:, :sc]
    right = bottom[:, sc + 1:]
    width = min(top.shape[1], left.shape[1], right.shape[1])
    ref = _states(top[:, :width], 5)

    def complete(panel, color):
        panel = panel[:, :width]
        mask = panel == color
        nonempty = [i for i in range(width) if np.any(mask[:, i])]
        prefix = max(nonempty) + 1 if nonempty else 0
        full = np.zeros_like(panel, dtype=bool)
        full[:, :prefix] = mask[:, :prefix]
        for c in range(prefix, width):
            for r in ref[c]:
                if r < full.shape[0]:
                    full[r, c] = True
        return full

    lmask = complete(left, 1)
    rmask = complete(right, 2)
    return np.where(~lmask & ~rmask, 3, 0).astype(int)


def generate_input() -> np.ndarray:
    ph = random.randint(5, 7)
    pw = random.randint(8, 11)
    seq = []
    choices = [(), (1,), (ph - 2,), (1, ph - 2), (ph // 2,)]
    offset = random.randrange(len(choices))
    for c in range(pw):
        seq.append(choices[(c + offset) % len(choices)])
    top = np.zeros((ph, pw), dtype=int)
    for c, rows in enumerate(seq):
        for r in rows:
            top[r, c] = 5
    bh = ph
    p1 = random.randint(2, pw - 3)
    p2 = random.randint(2, pw - 3)
    while p2 == p1:
        p2 = random.randint(2, pw - 3)
    left = np.zeros((bh, pw), dtype=int)
    right = np.zeros((bh, pw), dtype=int)
    for c in range(p1):
        for r in seq[c]:
            left[r, c] = 1
    for c in range(p2):
        for r in seq[c]:
            right[r, c] = 2
    grid = np.zeros((ph + 1 + bh, pw * 2 + 1), dtype=int)
    grid[:ph, :pw] = top
    grid[ph, :] = 7
    grid[ph + 1:, :pw] = left
    grid[ph + 1:, pw] = 5
    grid[ph + 1:, pw + 1:] = right
    return grid

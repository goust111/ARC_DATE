import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    rows = np.where(np.all(g == 0, axis=1))[0]
    cols = np.where(np.all(g == 0, axis=0))[0]

    keep_r = [r for r in range(g.shape[0]) if r not in set(rows)]
    keep_c = [c for c in range(g.shape[1]) if c not in set(cols)]
    c = g[np.ix_(keep_r, keep_c)]

    gray_cols = [cidx for cidx in range(c.shape[1]) if np.all(c[:, cidx] == 5)]
    if not gray_cols:
        return np.zeros((0, 0), dtype=int)
    gc = gray_cols[0]

    left = c[:, :gc]
    right = c[:, gc + 1 :]
    w = min(left.shape[1], right.shape[1])
    left = left[:, :w]
    right = right[:, :w]

    out = np.zeros((left.shape[0], w), dtype=int)
    for r in range(out.shape[0]):
        for cc in range(w):
            a = left[r, cc] != 0
            b = right[r, cc] != 0
            if a ^ b:
                out[r, cc] = 2
    return out


def generate_input() -> np.ndarray:
    h = random.choice([8, 9])
    wl = 4
    wr = 4
    base = np.zeros((h, wl + 1 + wr), dtype=int)
    base[:, wl] = 5
    for r in range(h):
        for c in range(wl):
            if random.random() < 0.45:
                base[r, c] = 1
        for c in range(wr):
            if random.random() < 0.45:
                base[r, wl + 1 + c] = 2

    H = 2 * h - 1
    W = 2 * (wl + 1 + wr) - 1
    out = np.zeros((H, W), dtype=int)

    for r in range(h):
        for c in range(wl + 1 + wr):
            out[2 * r, 2 * c] = base[r, c]

    return out

import random

import numpy as np


def _compress(grid):
    rows = [0]
    for r in range(1, grid.shape[0]):
        if not np.array_equal(grid[r], grid[r - 1]):
            rows.append(r)
    g = grid[rows, :]
    cols = [0]
    for c in range(1, g.shape[1]):
        if not np.array_equal(g[:, c], g[:, c - 1]):
            cols.append(c)
    return g[:, cols]


def main(grid: np.ndarray) -> np.ndarray:
    core = _compress(grid)
    out = core.copy()
    quads = {
        (0, 0): out[0:3, 0:3],
        (0, 1): out[0:3, 3:6],
        (1, 0): out[3:6, 0:3],
        (1, 1): out[3:6, 3:6],
    }
    missing = next(pos for pos, q in quads.items() if np.all(q == 5))
    diag = (1 - missing[0], 1 - missing[1])
    out[missing[0] * 3 : missing[0] * 3 + 3, missing[1] * 3 : missing[1] * 3 + 3] = np.flipud(np.fliplr(quads[diag]))
    return out


def _make_core():
    while True:
        a = np.random.choice([1, 2], size=(3, 3))
        b = np.random.choice([1, 2], size=(3, 3))
        full = np.block([[a, b], [np.flipud(np.fliplr(b)), np.flipud(np.fliplr(a))]])
        if all(not np.array_equal(full[i], full[i - 1]) for i in range(1, 6)) and all(
            not np.array_equal(full[:, i], full[:, i - 1]) for i in range(1, 6)
        ):
            return full


def generate_input() -> np.ndarray:
    while True:
        core = _make_core()
        qr, qc = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        core[qr * 3 : qr * 3 + 3, qc * 3 : qc * 3 + 3] = 5
        if _compress(core).shape == (6, 6):
            break
    row_reps = [random.randint(1, 3) for _ in range(6)]
    col_reps = [random.randint(1, 3) for _ in range(6)]
    return np.repeat(np.repeat(core, row_reps, axis=0), col_reps, axis=1)

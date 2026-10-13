import random
import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == 5)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == 5)]
    rows = [(sep_rows[i] + 1, sep_rows[i + 1]) for i in range(len(sep_rows) - 1)]
    cols = [(sep_cols[i] + 1, sep_cols[i + 1]) for i in range(len(sep_cols) - 1)]
    ch = rows[0][1] - rows[0][0]
    cw = cols[0][1] - cols[0][0]
    motif = None
    markers = []
    for i, (r0, r1) in enumerate(rows):
        for j, (c0, c1) in enumerate(cols):
            nz = [(r, c, int(g[r, c])) for r in range(r0, r1) for c in range(c0, c1) if g[r, c] not in (0, 5)]
            if len(nz) > 1:
                motif = (i, j, [(r - r0, c - c0, col) for r, c, col in nz])
            elif len(nz) == 1:
                markers.append((i, j, nz[0][2]))
    if motif is None:
        return np.zeros((1, 1), dtype=int)
    mi, mj, pix = motif
    out = np.zeros(((mi + 1) * ch, (mj + 1) * cw), dtype=int)
    colors = {(tr, tc): pix[0][2] for tr in range(mi + 1) for tc in range(mj + 1)}
    markers.sort(key=lambda x: (rows[x[0]][0], cols[x[1]][0]))
    for r, c, col in markers:
        colors[(r % (mi + 1), c % (mj + 1))] = col
    for tr in range(mi + 1):
        for tc in range(mj + 1):
            col = colors[(tr, tc)]
            for pr, pc, _ in pix:
                out[tr * ch + pr, tc * cw + pc] = col
    return out


def generate_input() -> np.ndarray:
    k = random.choice([3, 4, 5])
    cell = random.choice([3, 4])
    size = k * cell + k + 1
    g = np.zeros((size, size), dtype=int)
    for x in range(0, size, cell + 1):
        g[x, :] = 5
        g[:, x] = 5
    mi = random.randint(1, k - 1)
    mj = random.randint(1, k - 1)
    motif_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    shapes = [[(0, 0), (0, 1), (1, 0)], [(0, 1), (1, 0), (1, 1), (2, 1)], [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]]
    shape = random.choice([s for s in shapes if max(r for r, _ in s) < cell and max(c for _, c in s) < cell])
    r0, c0 = mi * (cell + 1) + 1, mj * (cell + 1) + 1
    for r, c in shape:
        g[r0 + r, c0 + c] = motif_color
    empty = [(i, j) for i in range(k) for j in range(k) if (i, j) != (mi, mj)]
    random.shuffle(empty)
    for idx, (i, j) in enumerate(empty[:random.randint(0, 3)]):
        col = random.choice([x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x != motif_color])
        g[i * (cell + 1) + 1 + random.randrange(cell), j * (cell + 1) + 1 + random.randrange(cell)] = col
    return g

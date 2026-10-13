from __future__ import annotations

import numpy as np

BLUE = 1
BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    markers = np.argwhere(out == BLUE)
    if markers.size == 0:
        return out

    row_pairs = {}
    for r, c in markers:
        row_pairs.setdefault(int(r), []).append(int(c))

    pairs: list[tuple[int, int, int, int]] = []
    axes = []
    for r in sorted(row_pairs.keys()):
        cols = sorted(row_pairs[r])
        if len(cols) != 2:
            continue
        c1, c2 = cols
        gap = int(c2 - c1 - 1)
        axis = (c1 + c2) / 2
        pairs.append((r, c1, c2, gap))
        axes.append(axis)
    if not pairs:
        return out

    axis = sum(axes) / len(axes)
    pairs_sorted = sorted(pairs, key=lambda x: (x[3], x[0]))
    min_gap = pairs_sorted[0][3]
    chosen = pairs_sorted[0]

    # draw breach line for min-gap pair
    r, c1, c2, _ = chosen
    for c in range(min(c1, c2) + 1, max(c1, c2)):
        out[r, c] = BLUE

    guide_c = int(out[h - 1, 0])
    if guide_c == BLUE:
        guide_c = 2
    out[h - 1, 0] = BLACK

    for r0, a, b, gap in pairs:
        if (r0, a, b) == (r, c1, c2):
            continue
        for (x, y) in ((r0, a), (r0, b)):
            for rr in range(h):
                for cc in range(w):
                    if out[rr, cc] != BLACK:
                        continue
                    if abs(rr - x) + abs(cc - y) <= min_gap:
                        out[rr, cc] = guide_c

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 14
    out = np.zeros((h, w), dtype=int)
    # choose vertical center for symmetry
    axis = (w - 1) / 2.0
    candidate_d = [1, 2, 3, 4]
    rng.shuffle(candidate_d)
    n_pairs = 5
    rows = list(range(1, h - 1))
    rng.shuffle(rows)
    rows = rows[:n_pairs]

    # ensure unique gaps
    for i, r in enumerate(rows):
        d = candidate_d[i % len(candidate_d)] + i
        c1 = int(axis - d)
        c2 = int(axis + d)
        if c1 < 1 or c2 >= w - 1:
            c1 = 1 + (i % 2)
            c2 = w - 2 - (i % 2)
        out[r, c1] = BLUE
        out[r, c2] = BLUE

    out[h - 1, 0] = 2
    return out


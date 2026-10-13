import numpy as np

RNG = np.random.default_rng(202)


def main(grid: np.ndarray) -> np.ndarray:
    """Task 2: staircase-gated column ray fill.

Input:
    Black canvas containing gray staircase and orange markers.
Output:
    Same shape grid where valid marker columns are filled downward from staircase bottom.

Reasoning:
    Count staircase height per column (gray pixels), check each marker column has gray,
    then ray-fill from staircase bottom with color mapped from height.

Complexity:
    O(H*W).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    gray_rows = {c: int(np.sum(g[:, c] == 5)) for c in range(w)}

    color_map = {1: 1, 2: 2, 3: 3, 4: 4}

    markers = list(zip(*np.where(g == 7)))
    for r, c in markers:
        hgt = gray_rows.get(c, 0)
        if hgt == 0:
            continue
        row_start_candidates = np.where(g[:, c] == 5)[0]
        if row_start_candidates.size == 0:
            continue
        r_bottom = int(row_start_candidates.max())
        if r_bottom + 1 >= h:
            continue
        color = color_map.get(hgt, 6)
        for rr in range(r_bottom + 1, h):
            out[rr, c] = color

    return out


def generate_input() -> np.ndarray:
    h = 20
    w = 20
    g = np.zeros((h, w), dtype=int)

    levels = int(RNG.integers(4, 7))
    r = int(RNG.integers(0, 2))
    c = int(RNG.integers(0, max(2, w - 2 * levels - 2)))
    for i in range(levels):
        rr = r + 2 * i
        cc = c + 2 * i
        if rr < h // 2 and cc + 1 < w:
            g[rr, cc : cc + 2] = 5
            if i < levels - 1:
                g[rr + 1, cc + 1] = 5
                g[rr + 2, cc + 1] = 5
        else:
            break

    k = int(RNG.integers(3, 6))
    used_cols = set()
    candidates = list(range(w))
    for _ in range(k):
        if not candidates:
            break
        c = int(RNG.choice(candidates))
        r0 = int(RNG.integers(1, h - 1))
        candidates.remove(c)
        g[r0, c] = 7

    return g

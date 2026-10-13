import numpy as np


def _find_separator(grid):
    h, w = grid.shape
    cols = [c for c in range(w) if np.all(grid[:, c] == 1)]
    if len(cols) == 0:
        return w // 2
    return cols[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    union = (left == 4) | (right == 4)

    # detect union cross
    plus = set()
    for r in range(h):
        for c in range(w - sep - 1):
            if r == 0 or c == 0 or r == h - 1 or c == w - sep - 2:
                continue
            if (
                union[r, c]
                and union[r - 1, c]
                and union[r + 1, c]
                and union[r, c - 1]
                and union[r, c + 1]
            ):
                plus.add((r, c))
                plus.add((r - 1, c))
                plus.add((r + 1, c))
                plus.add((r, c - 1))
                plus.add((r, c + 1))

    out = np.zeros_like(left, dtype=np.int64)
    out[union] = 8
    for r, c in plus:
        out[r, c] = 2
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 7
    w = int(rng.integers(4, 7))
    sep = w
    total_w = 2 * w + 1
    left = np.zeros((h, w), dtype=np.int64)
    right = np.zeros((h, w), dtype=np.int64)

    cr = int(rng.integers(1, h - 1))
    cc = int(rng.integers(1, w - 1))
    cross = [(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]

    for r, c in cross:
        where = int(rng.integers(0, 3))
        if where == 0:
            left[r, c] = 4
        elif where == 1:
            right[r, c] = 4
        else:
            left[r, c] = 4
            right[r, c] = 4
    # Ensure each panel misses at least one cross pixel (so no internal plus),
    # yet union becomes full plus
    # Force at least one pixel to be missing from each panel
    if np.all(left[cross,]) or np.all(right[cross,]):  # pragma: no cover
        # handled below by explicit reassignment
        pass
    # left panel
    l_missing = rng.choice(len(cross))
    rr, cc2 = cross[int(l_missing)]
    right[rr, cc2] = 4
    left[rr, cc2] = 0
    # right panel
    r_missing = (l_missing + 1) % len(cross)
    rr2, cc2 = cross[int(r_missing)]
    left[rr2, cc2] = 4
    right[rr2, cc2] = 0

    grid = np.concatenate([left, np.full((h, 1), 1, dtype=np.int64), right], axis=1)
    return grid


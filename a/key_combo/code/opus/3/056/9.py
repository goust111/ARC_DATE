import random
import numpy as np


def _intervals(lines, n):
    cuts = [-1] + sorted(lines) + [n]
    return [(cuts[i] + 1, cuts[i + 1] - 1) for i in range(len(cuts) - 1) if cuts[i] + 1 <= cuts[i + 1] - 1]


def _rot(dr, dc, q):
    q %= 4
    for _ in range(q):
        dr, dc = dc, -dr
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """Count row partitions and rotate every red segment around its gray pivot by that many clockwise quarter turns."""
    out = grid.copy()
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == 4)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == 4)]
    row_regions = _intervals(sep_rows, h)
    col_regions = _intervals(sep_cols, w)
    q = len(row_regions)
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            piv = [(int(r), int(c)) for r, c in np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == 5)]
            reds = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == 2)]
            if len(piv) != 1 or not reds:
                continue
            pr, pc = r0 + piv[0][0], c0 + piv[0][1]
            for r, c in reds:
                out[r, c] = 3
            for r, c in reds:
                dr, dc = _rot(r - pr, c - pc, q)
                out[pr + dr, pc + dc] = 2
    return out


def generate_input() -> np.ndarray:
    n = 14
    g = np.zeros((n, n), dtype=int)
    regions = random.choice([2, 3, 4])
    if regions == 2:
        sep = [6]
    elif regions == 3:
        sep = [4, 9]
    else:
        sep = [3, 6, 10]
    for s in sep:
        g[s, :] = 4
        g[:, s] = 4
    rows = _intervals(sep, n)
    cols = _intervals(sep, n)
    for r0, r1 in rows:
        for c0, c1 in cols:
            pr, pc = (r0 + r1) // 2, (c0 + c1) // 2
            g[pr, pc] = 5
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(dirs)
            length = min(2, max(1, r1 - r0, c1 - c0))
            for dr, dc in dirs:
                cells = [(pr + dr * k, pc + dc * k) for k in range(1, length + 1)]
                rots = [(_rot(r - pr, c - pc, regions)) for r, c in cells]
                newcells = [(pr + dr2, pc + dc2) for dr2, dc2 in rots]
                if all(r0 <= r <= r1 and c0 <= c <= c1 and g[r, c] == 0 for r, c in cells + newcells):
                    for r, c in cells:
                        g[r, c] = 2
                    break
    return g

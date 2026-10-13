"""Puzzle 2: Coordinate-coded cyclic shift tiling completion."""

from __future__ import annotations

from collections import Counter

import numpy as np

BG = 3
FG = 4


def _parse_structure(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    if h != w:
        raise ValueError("expected square grid")

    row_sep = [r for r in range(h) if np.all(grid[r, :] == BG)]
    col_sep = [c for c in range(w) if np.all(grid[:, c] == BG)]
    if len(row_sep) < 1 or len(col_sep) < 1:
        raise ValueError("missing separators")

    # Separator gap is k+1.
    d_row = Counter(np.diff(row_sep)).most_common(1)
    d_col = Counter(np.diff(col_sep)).most_common(1)
    if not d_row or not d_col:
        raise ValueError("invalid separator structure")

    d = min(d_row[0][0], d_col[0][0])
    k = max(1, d - 1)

    row_starts = [0]
    for r in row_sep:
        if 0 <= r + 1 < h:
            row_starts.append(r + 1)
    col_starts = [0]
    for c in col_sep:
        if 0 <= c + 1 < w:
            col_starts.append(c + 1)

    row_starts = sorted(set(row_starts))
    col_starts = sorted(set(col_starts))

    # keep only starts that allow a full kxk tile in bounds
    row_starts = [r for r in row_starts if r + k <= h]
    col_starts = [c for c in col_starts if c + k <= w]
    return k, row_starts, col_starts


def _to_tuple(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in tile.flatten())


def _from_tuple(t: tuple[int, ...], k: int) -> np.ndarray:
    return np.array(t, dtype=int).reshape((k, k))


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = np.full_like(grid, BG)

    k, row_starts, col_starts = _parse_structure(grid)
    candidates: Counter[tuple[int, ...]] = Counter()

    n_rows = len(row_starts)
    n_cols = len(col_starts)

    for ti in range(n_rows):
        rs = row_starts[ti]
        for tj in range(n_cols):
            cs = col_starts[tj]
            if rs + k > grid.shape[0] or cs + k > grid.shape[1]:
                continue
            tile = grid[rs : rs + k, cs : cs + k]
            if not np.any(tile == FG):
                continue
            shift = (ti + tj) % k
            recovered = np.roll(tile, shift=shift, axis=1)
            candidates[_to_tuple(recovered)] += 1

    if not candidates:
        return grid.copy()

    base_tuple = candidates.most_common(1)[0][0]
    base = _from_tuple(base_tuple, k)

    for ti, rs in enumerate(row_starts):
        for tj, cs in enumerate(col_starts):
            if rs + k > out.shape[0] or cs + k > out.shape[1]:
                continue
            shift = (ti + tj) % k
            filled = np.roll(base, shift=-shift, axis=1)
            out[rs : rs + k, cs : cs + k] = filled

    for i in range(1, len(row_starts)):
        r = row_starts[i] - 1
        out[r, :] = BG
    for j in range(1, len(col_starts)):
        c = col_starts[j] - 1
        out[:, c] = BG

    return out


def _generate_base(rng: np.random.Generator, k: int) -> np.ndarray:
    while True:
        base = rng.integers(0, 2, size=(k, k))
        if not np.any(base == 1):
            continue
        if not np.any(base == 0):
            continue
        # avoid fully periodic by rejecting trivial k-cycle shifts
        periodic = False
        for s in range(1, k):
            if np.array_equal(base, np.roll(base, s, axis=1)):
                periodic = True
                break
        if periodic:
            continue
        return base.astype(int) * FG


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    k = int(rng.integers(3, 6))
    n = int(rng.integers(3, 5))
    size = n * k + (n - 1)

    base = _generate_base(rng, k)
    grid = np.full((size, size), BG, dtype=int)

    for r in range(1, n):
        sep = r * (k + 1) - 1
        grid[sep, :] = BG
    for c in range(1, n):
        sep = c * (k + 1) - 1
        grid[:, sep] = BG

    total_tiles = n * n
    min_fill = max(1, int(total_tiles * 0.40))
    max_fill = max(min_fill, int(total_tiles * 0.60))
    fill_count = int(rng.integers(min_fill, max_fill + 1))

    required = {(i, int(rng.integers(0, n))) for i in range(n)}
    required.update({(int(rng.integers(0, n)), j) for j in range(n)})
    filled = set(required)
    while len(filled) < fill_count:
        filled.add((int(rng.integers(0, n)), int(rng.integers(0, n))))

    for tr, tc in filled:
        rs = tr * (k + 1)
        cs = tc * (k + 1)
        shift = (tr + tc) % k
        tile = np.roll(base, shift=-shift, axis=1)
        grid[rs : rs + k, cs : cs + k] = tile

    return grid


if __name__ == "__main__":
    g = generate_input()
    print(main(g))

"""Puzzle 4: Filter minority colors then self-masked expansion."""

from __future__ import annotations

from collections import Counter

import numpy as np

BG = 0


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected 2-D grid")
    n, m = grid.shape
    if n != m:
        raise ValueError("expected square grid")

    values = [int(v) for v in grid.flatten() if int(v) != BG]
    cnt = Counter(values)
    if len(cnt) < 2:
        return grid.copy()

    top2 = [col for col, _ in cnt.most_common(2)]
    keep = set(top2)

    filtered = np.where(np.isin(grid, list(keep)), grid, BG)

    out = np.zeros((n * n, m * m), dtype=int)
    for r in range(n):
        for c in range(m):
            col = int(filtered[r, c])
            if col == BG:
                continue
            block = filtered == col
            out[r * n : (r + 1) * n, c * m : (c + 1) * m] = np.where(block, col, 0)

    return out


def _random_counts(rng: np.random.Generator, total: int, num_colors: int) -> list[int]:
    minority_num = num_colors - 2
    for major in range(max(1, total // 3), min(total - 2, 20) + 1):
        if major <= 0:
            continue
        if total - 2 * major < minority_num:
            continue

        minors = [1] * minority_num
        remaining = total - 2 * major - minority_num
        if remaining < 0:
            continue

        for _ in range(remaining):
            idx = int(rng.integers(0, minority_num))
            if minors[idx] + 1 <= major - 1:
                minors[idx] += 1
            else:
                # find any minority bucket with capacity
                choices = [j for j, v in enumerate(minors) if v < major - 1]
                if not choices:
                    break
                idx = int(rng.choice(choices))
                minors[idx] += 1

        if all(v < major for v in minors):
            if sum(minors) == total - 2 * major:
                return [major, major] + minors
    # fallback safe counts
    return [3, 3, 1]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        n = int(rng.choice(np.array([4, 5])))
        total = n * n
        num_colors = int(rng.integers(3, 6))

        counts = _random_counts(rng, total, num_colors)
        while len(counts) < num_colors:
            counts.append(1)
        if len(counts) > num_colors:
            counts = counts[:num_colors]

        palette = [1, 2, 3, 4, 5]
        colors = palette[:num_colors]
        cells: list[int] = []
        for color, cnt in zip(colors, counts):
            cells += [color] * cnt

        if len(cells) < total:
            cells.extend([colors[1]] * (total - len(cells)))
        if len(cells) > total:
            cells = cells[:total]

        if 0 not in cells:
            rng.shuffle(cells)
            arr = np.array(cells, dtype=int).reshape((n, n))
            uniq = Counter(arr.flatten())
            top = uniq.most_common(2)
            if len(top) >= 2 and top[0][1] == top[1][1]:
                # ensure majority counts are strictly > minorities and tied
                if all(v < top[0][1] for _, v in uniq.most_common()[2:]):
                    return arr

    # fallback deterministic valid grid
    return np.array(
        [
            [1, 1, 1, 2],
            [1, 3, 2, 2],
            [4, 1, 2, 3],
            [4, 3, 3, 2],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    g = generate_input()
    print(main(g))

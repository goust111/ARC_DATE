"""Puzzle 7: gravity drop creates centers, then even Chebyshev interference."""

from __future__ import annotations

import numpy as np

BG = 0
OBSTACLE = 5
STOP = 4
PASS = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _landing_row(grid: np.ndarray, col: int, control: int) -> int:
    h, _w = grid.shape

    if control == PASS:
        return h - 1

    # STOP: first obstacle above blocks, stop on row above
    for r in range(1, h):
        if grid[r, col] == OBSTACLE:
            return max(0, r - 1)
    return h - 1


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)

    seeds: list[tuple[int, int]] = []
    for c in range(w):
        color = int(grid[0, c])
        if color in (BG, OBSTACLE):
            continue
        if not _in_bounds((h, w), 1, c):
            continue
        control = int(grid[1, c])
        if control not in (STOP, PASS):
            continue
        seeds.append((c, color))

    centers: list[tuple[int, int]] = []
    for c, color in seeds:
        r = _landing_row(grid, c, int(grid[1, c]))
        centers.append((r, c, color))

    # fill Chebyshev rings by nearest-center tie policy
    for r in range(h):
        for c in range(w):
            best = None
            best_color = BG
            for cr, cc, color in centers:
                d = max(abs(r - cr), abs(c - cc))
                if d % 2 == 1:
                    continue
                if best is None or d < best[0] or (d == best[0] and cr < best[1]):
                    best = (d, cr, color)
            if best is not None:
                _, _, color = best
                best_color = color
            out[r, c] = best_color

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        h = int(rng.integers(18, 21))
        w = int(rng.integers(18, 21))
        grid = np.zeros((h, w), dtype=int)

        seed_num = int(rng.integers(3, 5))
        cols = sorted(rng.choice(np.arange(1, w - 1), size=seed_num, replace=False))
        if np.any(np.diff(cols) < 1):
            continue

        colors = [1, 2, 3, 4, 5, 7]
        seeds: list[tuple[int, int]] = []

        for i, c in enumerate(cols):
            color = colors[i % len(colors)]
            if color in (BG, OBSTACLE, STOP, PASS):
                color = 1
            grid[0, c] = color
            control = int(rng.choice(np.array([STOP, PASS]), p=np.array([0.6, 0.4], dtype=float)))
            grid[1, c] = control
            seeds.append((int(c), control))

            if control == STOP:
                r_obs = int(rng.integers(2, h - 1))
                grid[r_obs, c] = OBSTACLE

        # optional extra obstacles, keep clear of top two rows
        for _ in range(int(rng.integers(8, 18))):
            for _ in range(50):
                r = int(rng.integers(2, h - 1))
                c = int(rng.integers(0, w))
                if grid[r, c] != BG:
                    continue
                grid[r, c] = OBSTACLE
                break

        # each stop control must have an obstacle in its column by construction
        if all(
            any(grid[r, c] == OBSTACLE for r in range(2, h)) for c, control in seeds if control == STOP
        ):
            return grid

    return np.array(
        [
            [0, 1, 0, 2, 0, 3, 0, 2, 0],
            [0, 4, 0, 7, 0, 4, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    g = generate_input()
    print(main(g))

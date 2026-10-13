"""Puzzle 1: symmetry decides growth direction with obstacle detour."""

from __future__ import annotations

import numpy as np

BG = 0
PATTERN = 2
SEEDS = {1, 3}
BLOCKER = 5


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_horizontal_symmetric(patch: np.ndarray) -> bool:
    return np.array_equal(patch[0], patch[2][::-1])


def _grow_path(grid: np.ndarray, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    path = [(r, c)]

    if direction == "up":
        step = (-1, 0)
        detour = (0, 1)
    else:
        step = (0, 1)
        detour = (1, 0)

    while True:
        nr, nc = r + step[0], c + step[1]
        if not _in_bounds((h, w), nr, nc):
            break
        if grid[nr, nc] == BLOCKER:
            dr, dc = detour
            tr, tc = r + dr, c + dc
            if not _in_bounds((h, w), tr, tc):
                break
            if grid[tr, tc] == BLOCKER:
                break
            r, c = tr, tc
            path.append((r, c))
            continue
        r, c = nr, nc
        path.append((r, c))

    return path


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    h, w = grid.shape

    for r in range(0, h - 3):
        for c in range(0, w - 3):
            patch = grid[r : r + 3, c : c + 3]
            if np.any((patch != PATTERN) & (patch != BG)):
                continue
            if not np.any(patch == PATTERN):
                continue

            seed_r, seed_c = r + 3, c + 1
            if not _in_bounds((h, w), seed_r, seed_c):
                continue
            seed_color = int(grid[seed_r, seed_c])
            if seed_color not in SEEDS:
                continue

            direction = "up" if _is_horizontal_symmetric(patch) else "right"
            for pr, pc in _grow_path(grid, (seed_r, seed_c), direction):
                out[pr, pc] = seed_color

    return out


def _random_pattern(rng: np.random.Generator, symmetric: bool) -> np.ndarray:
    while True:
        if symmetric:
            row0 = rng.integers(0, 2, size=3)
            row1 = rng.integers(0, 2, size=3)
            shape = np.array([row0, row1, row0[::-1]], dtype=int)
        else:
            shape = rng.integers(0, 2, size=(3, 3), dtype=int)
            if np.array_equal(shape[0], shape[2][::-1]):
                continue
        if np.any(shape == 1) and np.any(shape == 0):
            return np.where(shape == 1, PATTERN, BG)


def _patterns_overlap(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    rr0, rr1 = max(0, r0), min(h, r1 + 1)
    cc0, cc1 = max(0, c0), min(w, c1 + 1)
    return np.any(occupied[rr0:rr1, cc0:cc1])


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        h = int(rng.integers(15, 21))
        w = int(rng.integers(14, 21))
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        pattern_count = int(rng.integers(3, 5))
        placed = 0
        tries = 0
        while placed < pattern_count and tries < 2000:
            tries += 1
            symmetric = bool(rng.integers(0, 2))
            patch = _random_pattern(rng, symmetric)
            tr = int(rng.integers(0, h - 4))
            tc = int(rng.integers(0, w - 3))
            sr, sc = tr + 3, tc + 1

            if not _in_bounds((h, w), sr, sc):
                continue
            if _patterns_overlap(occupied, tr - 2, tc - 2, tr + 4, tc + 4):
                continue

            if grid[sr, sc] != BG:
                continue
            if np.any(grid[tr : tr + 3, tc : tc + 3] != BG):
                continue

            grid[tr : tr + 3, tc : tc + 3] = patch
            grid[sr, sc] = int(rng.choice([1, 3]))
            seed_r, seed_c = sr, sc
            occupied[tr : tr + 3, tc : tc + 3] = True
            occupied[seed_r, seed_c] = True
            # pad to keep spacing >= 4
            rr0 = max(0, tr - 2)
            rr1 = min(h, tr + 5)
            cc0 = max(0, tc - 2)
            cc1 = min(w, tc + 5)
            occupied[rr0:rr1, cc0:cc1] = True
            placed += 1

        if placed != pattern_count:
            continue

        obstacle_count = int(rng.integers(12, 26))
        for _ in range(obstacle_count):
            for _ in range(100):
                r = int(rng.integers(1, h - 1))
                c = int(rng.integers(1, w - 1))
                if grid[r, c] != BG:
                    continue
                if _in_bounds((h, w), r - 1, c) and _in_bounds((h, w), r + 1, c):
                    if r == h - 2 or grid[r + 1, c] == BLOCKER:
                        continue
                grid[r, c] = BLOCKER
                break
        return grid

    # deterministic fallback
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 3, 2, 3, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0, 0, 5, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 1, 0, 5, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 5, 0, 0, 0, 0, 5, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 2, 3, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

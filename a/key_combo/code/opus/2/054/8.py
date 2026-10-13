"""Puzzle 8: detour-frequency ranked recolor of upward-growing paths."""

from __future__ import annotations

from collections import Counter

import numpy as np

BG = 0
FG = 2
BLOCK = 5
ALT = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _grow_path(grid: np.ndarray, start_row: int, start_col: int) -> tuple[list[tuple[int, int]], int, int]:
    h, w = grid.shape
    r, c = start_row, start_col
    path = [(r, c)]
    detours = 0

    while r > 0:
        nr, nc = r - 1, c
        if not _in_bounds((h, w), nr, nc):
            break
        if grid[nr, c] == BLOCK:
            sc = c + 1
            if not _in_bounds((h, w), r, sc):
                break
            if grid[r, sc] == BLOCK:
                break
            c = sc
            detours += 1
            path.append((r, c))
            continue
        r = nr
        path.append((r, c))

    return path, detours, c


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    out = np.array(grid, dtype=int)

    seeds: list[tuple[int, int]] = []
    for c, v in enumerate(grid[h - 1]):
        if v == FG:
            seeds.append((h - 1, c))

    paths: list[list[tuple[int, int]]] = []
    detours: list[int] = []

    for sr, sc in seeds:
        path, d, _ = _grow_path(grid, sr, sc)
        if len(path) <= 1:
            continue
        paths.append(path)
        detours.append(d)
        # keep obstacle color in output initially
        for rr, cc in path:
            out[rr, cc] = FG

    if not detours:
        return out

    mode = Counter(detours).most_common(1)[0][0]
    for path, d in zip(paths, detours):
        if d == mode:
            continue
        for rr, cc in path:
            out[rr, cc] = ALT

    return out


def _simulate_with_constraints(
    h: int,
    w: int,
    cols: list[int],
    detour_counts: list[int],
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], bool]:
    # returns (path_columns_map, obstacle_cells, success)
    occupancy = set()
    obstacles = set()
    paths: dict[tuple[int, int], int] = {}

    for col, target in zip(cols, detour_counts):
        r = h - 1
        c = col
        actual = 0
        steps = 0
        while r > 0 and steps < h * 3:
            steps += 1

            if actual < target:
                tr, tc = r - 1, c
                # force obstacle above current position
                if _in_bounds((h, w), tr, c) and _in_bounds((h, w), r, c + 1):
                    if grid_ok := (tr, c) not in obstacles and (r, c + 1) not in occupancy and (r, c + 1) not in obstacles and (c + 1) < w - 1:
                        if not _in_bounds((h, w), tr, c + 1):
                            pass
                        obstacles.add((tr, c))
                        c += 1
                        actual += 1
                        occupancy.add((r, c))
                        continue

                # if cannot force detour now, fallback to natural up move

            if (r - 1, c) in obstacles:
                return {}, set(), False
            r -= 1
            occupancy.add((r, c))
        if r != 0 or actual != target:
            return {}, set(), False
        paths[(r, c)] = actual

        # ensure full vertical trace is unblocked after finishing forced detours
        # replay quickly from bottom with this column history (for collision checks only)
        # done implicitly through occupancy checks above

    return paths, obstacles, True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(300):
        h = int(rng.integers(12, 16))
        w = 15
        n = int(rng.integers(4, 7))

        # spaced starting columns
        possible = list(range(1, w - 1))
        while True:
            cols = sorted(rng.choice(possible, size=n, replace=False))
            if np.all(np.diff(cols) >= 2):
                break

        mode = int(rng.integers(0, 3))
        counts = [mode] * n
        # alter two entries to ensure unique mode
        if n > 2:
            idx = int(rng.integers(0, n))
            idx2 = int(rng.integers(0, n - 1))
            if idx2 >= idx:
                idx2 += 1
            counts[idx] = (mode + 1) % 3
            counts[idx2] = (mode + 2) % 3

        # build obstacle plan
        grid = np.zeros((h, w), dtype=int)
        for c in cols:
            grid[h - 1, c] = FG

        # place obstacles while constructing deterministic paths
        paths_ok = True
        obstacle_cells: set[tuple[int, int]] = set()
        occupied_path: set[tuple[int, int]] = {(h - 1, c) for c in cols}

        # simple path-first placement
        for col, target in zip(cols, counts):
            r = h - 1
            c = col
            det = 0
            steps = 0
            while r > 0 and steps < h * 5:
                steps += 1
                if det < target:
                    # create obstacle and detour if possible
                    if c + 1 < w - 1 and (r - 1, c) not in obstacle_cells and (r, c + 1) not in occupied_path and (r, c + 1) not in obstacle_cells:
                        obstacle_cells.add((r - 1, c))
                        c += 1
                        det += 1
                        occupied_path.add((r, c))
                        continue

                if (r - 1, c) in obstacle_cells:
                    paths_ok = False
                    break
                r -= 1
                if (r, c) in occupied_path or (r, c) in obstacle_cells:
                    if (r - 1, c) in obstacle_cells:
                        paths_ok = False
                        break
                occupied_path.add((r, c))

            if not paths_ok or r != 0 or det != target:
                paths_ok = False
                break

        # verify mode uniqueness with final counts (should already hold)
        if not paths_ok:
            continue

        freq = Counter(counts)
        if len(freq) < 2:
            continue
        top_count = freq.most_common(1)[0][1]
        second = freq.most_common(2)
        if len(second) < 2 or top_count <= second[1][1]:
            continue

        for r, c in obstacle_cells:
            if _in_bounds((h, w), r, c):
                grid[r, c] = BLOCK

        return grid[:, :w]

    # deterministic fallback
    return np.array(
        [
            [0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 5, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    g = generate_input()
    print(main(g))

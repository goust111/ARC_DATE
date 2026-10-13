import random
from typing import List, Sequence, Tuple

import numpy as np


def _ray_masks(grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    obstacles = grid == 5
    red_mask = np.zeros((h, w), dtype=bool)
    green_mask = np.zeros((h, w), dtype=bool)

    for r, c in np.argwhere(grid == 2):
        rr = r
        while rr >= 0 and not obstacles[rr, c]:
            red_mask[rr, c] = True
            rr -= 1
        rr = r + 1
        while rr < h and not obstacles[rr, c]:
            red_mask[rr, c] = True
            rr += 1

    for r, c in np.argwhere(grid == 3):
        cc = c
        while cc >= 0 and not obstacles[r, cc]:
            green_mask[r, cc] = True
            cc -= 1
        cc = c + 1
        while cc < w and not obstacles[r, cc]:
            green_mask[r, cc] = True
            cc += 1

    return red_mask, green_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Treat red pixels as vertical seeds, green pixels as horizontal seeds,
        and gray pixels as blockers on a black background.
    Analyze:
        Extend both ray families independently, then detect every non-blocked
        cell reached by both a red and a green ray.
    Transform:
        Paint red and green ray coverage, convert crossings to yellow, and
        expand each crossing into a blocker-preserving 3x3 yellow ring.
    Return:
        Produce the fully transformed 2D ndarray with untouched background and
        untouched blockers outside the ray and ring effects.
    """
    grid = np.asarray(grid, dtype=int)
    red_mask, green_mask = _ray_masks(grid)
    crossings = red_mask & green_mask & (grid != 5)

    out = grid.copy()
    out[red_mask] = 2
    out[green_mask] = 3
    out[crossings] = 4

    h, w = grid.shape
    for r, c in np.argwhere(crossings):
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] != 5:
                    out[rr, cc] = 4
    return out


def _not_adjacent(pos: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> bool:
    return all(abs(pos[0] - r) + abs(pos[1] - c) != 1 for r, c in others)


def generate_input() -> np.ndarray:
    for _ in range(5000):
        h = random.randint(12, 16)
        w = random.randint(12, 16)
        grid = np.zeros((h, w), dtype=int)

        red_count = random.randint(3, 5)
        green_count = random.randint(3, 5)
        obstacles = random.randint(4, 8)

        red_positions: List[Tuple[int, int]] = []
        green_positions: List[Tuple[int, int]] = []
        used_red_cols = set()
        used_green_rows = set()

        ok = True
        for _ in range(red_count):
            for _ in range(200):
                pos = (random.randrange(h), random.randrange(w))
                if pos[1] in used_red_cols:
                    continue
                if not _not_adjacent(pos, red_positions + green_positions):
                    continue
                red_positions.append(pos)
                used_red_cols.add(pos[1])
                break
            else:
                ok = False
                break
        if not ok:
            continue

        for _ in range(green_count):
            for _ in range(200):
                pos = (random.randrange(h), random.randrange(w))
                if pos in red_positions or pos[0] in used_green_rows:
                    continue
                if not _not_adjacent(pos, red_positions + green_positions):
                    continue
                green_positions.append(pos)
                used_green_rows.add(pos[0])
                break
            else:
                ok = False
                break
        if not ok:
            continue

        blocked = set(red_positions + green_positions)
        obstacle_positions: List[Tuple[int, int]] = []
        for _ in range(obstacles):
            for _ in range(200):
                pos = (random.randrange(h), random.randrange(w))
                if pos in blocked:
                    continue
                blocked.add(pos)
                obstacle_positions.append(pos)
                break
            else:
                ok = False
                break
        if not ok:
            continue

        for r, c in red_positions:
            grid[r, c] = 2
        for r, c in green_positions:
            grid[r, c] = 3
        for r, c in obstacle_positions:
            grid[r, c] = 5

        red_mask, green_mask = _ray_masks(grid)
        crossings = red_mask & green_mask & (grid != 5)
        if int(crossings.sum()) >= 2:
            return grid
    raise RuntimeError("failed to generate a valid task-1 input")

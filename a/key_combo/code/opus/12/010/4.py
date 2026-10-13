from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
ARM_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ALL_SUPPORTS = tuple(ARM_DIRS)

GRID_MIN = 16
GRID_MAX = 22
CROSS_COUNT_RANGE = (4, 6)
NOISE_COUNT_RANGE = (3, 8)
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _nonblack_degree(grid: np.ndarray, r: int, c: int) -> int:
    deg = 0
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
            deg += 1
    return deg


def _find_cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != YELLOW:
                continue
            if all(int(grid[r + dr, c + dc]) == ORANGE for dr, dc in DIR4):
                centers.append((r, c))
    return centers


def _filter_supported_orange(grid: np.ndarray) -> np.ndarray:
    keep = np.zeros(grid.shape, dtype=bool)
    for r, c in np.argwhere(grid == ORANGE):
        keep[int(r), int(c)] = _nonblack_degree(grid, int(r), int(c)) >= 2
    return keep


def _surviving_crosses(grid: np.ndarray) -> list[tuple[int, int]]:
    orange_keep = _filter_supported_orange(grid)
    survivors: list[tuple[int, int]] = []
    for r, c in _find_cross_centers(grid):
        if all(orange_keep[r + dr, c + dc] for dr, dc in DIR4):
            survivors.append((r, c))
    return survivors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter weakly supported orange pixels, keep only complete crosses, then expand survivors with a parity-controlled diagonal color.

    Input structure:
    - The input is a black grid containing several yellow-centered orange crosses and extra orange pixels.
    - Each intended cross has a yellow center with four orthogonal orange arm pixels.
    - Some crosses have sacrificial orange support pixels just beyond their arms, while other orange pixels are clutter.

    Rule:
    - For every orange pixel, count its non-black 4-neighbors and delete it if that support count is below 2.
    - Revisit the original yellow-centered crosses and keep only those whose four arm pixels survived the filter; remove incomplete crosses entirely.
    - Let N be the number of surviving crosses. Redraw each survivor as a 5x5 motif: extend orange orthogonally to distance 2, keep the yellow center, and color the four outer diagonal corners yellow if N is odd or green if N is even.

    Key insight:
    - Local support filtering decides which crosses remain, but the diagonal color of every remaining cross is controlled globally by the parity of the total survivor count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    survivors = _surviving_crosses(g)
    diag_color = YELLOW if len(survivors) % 2 == 1 else GREEN

    out = np.zeros_like(g)
    for r, c in survivors:
        out[r, c] = YELLOW
        for dr, dc in DIR4:
            out[r + dr, c + dc] = ORANGE
            out[r + 2 * dr, c + 2 * dc] = ORANGE
        for dr, dc in DIAG4:
            out[r + dr, c + dc] = diag_color
    return out


def _can_place_center(center: tuple[int, int], placed: Iterable[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = center
    if not (2 <= r < shape[0] - 2 and 2 <= c < shape[1] - 2):
        return False
    for pr, pc in placed:
        if max(abs(r - pr), abs(c - pc)) < 6:
            return False
    return True


def _choose_centers(shape: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    random.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for center in candidates:
        if _can_place_center(center, centers, shape):
            centers.append(center)
            if len(centers) == count:
                return centers
    return None


def _support_pos(center: tuple[int, int], name: str) -> tuple[int, int]:
    r, c = center
    dr, dc = ARM_DIRS[name]
    return r + 2 * dr, c + 2 * dc


def _paint_cross(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = YELLOW
    for dr, dc in DIR4:
        grid[r + dr, c + dc] = ORANGE


def _paint_supports(grid: np.ndarray, center: tuple[int, int], supports: Iterable[str]) -> None:
    for name in supports:
        rr, cc = _support_pos(center, name)
        grid[rr, cc] = ORANGE


def _isolated_noise_candidates(shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    cands = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            ok = True
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if abs(dr) + abs(dc) > 1:
                        continue
                    rr, cc = r + dr, c + dc
                    if (rr, cc) in forbidden:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cands.append((r, c))
    return cands


def _sample_isolated_noise(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for cell in pool:
        if cell in blocked:
            continue
        chosen.append(cell)
        r, c = cell
        for dr, dc in DIR4 + [(0, 0)]:
            blocked.add((r + dr, c + dc))
        if len(chosen) == count:
            return chosen
    return []


def _build_support_plan(count: int, survivor_count: int) -> list[set[str]]:
    plan = [set() for _ in range(count)]
    survivor_ids = set(random.sample(range(count), survivor_count))

    doomed_ids = [idx for idx in range(count) if idx not in survivor_ids]
    partial_anchor = random.choice(doomed_ids)

    for idx in range(count):
        if idx in survivor_ids:
            plan[idx] = set(ALL_SUPPORTS)
            continue
        if idx == partial_anchor:
            k = random.randint(1, 3)
        else:
            k = random.randint(0, 3)
        plan[idx] = set(random.sample(ALL_SUPPORTS, k))
    return plan


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.randint(*CROSS_COUNT_RANGE)

    parity = random.randint(0, 1)
    survivor_options = [n for n in range(1, count) if n % 2 == parity]
    if not survivor_options:
        return None
    survivor_count = random.choice(survivor_options)

    centers = _choose_centers((h, w), count)
    if centers is None:
        return None

    support_plan = _build_support_plan(count, survivor_count)
    grid = np.zeros((h, w), dtype=int)

    forbidden: set[tuple[int, int]] = set()
    for center, supports in zip(centers, support_plan):
        _paint_cross(grid, center)
        _paint_supports(grid, center, supports)

        r, c = center
        for rr in range(r - 2, r + 3):
            for cc in range(c - 2, c + 3):
                if _in_bounds(grid.shape, rr, cc):
                    forbidden.add((rr, cc))

    noise_candidates = _isolated_noise_candidates(grid.shape, forbidden)
    if len(noise_candidates) < NOISE_COUNT_RANGE[0]:
        return None
    noise_count = random.randint(NOISE_COUNT_RANGE[0], min(NOISE_COUNT_RANGE[1], len(noise_candidates)))
    noise_cells = _sample_isolated_noise(noise_candidates, noise_count)
    if len(noise_cells) != noise_count:
        return None
    for r, c in noise_cells:
        grid[r, c] = ORANGE

    centers_found = _find_cross_centers(grid)
    if len(centers_found) != count:
        return None

    survivors = _surviving_crosses(grid)
    if len(survivors) != survivor_count:
        return None

    partial_exists = False
    orange_keep = _filter_supported_orange(grid)
    for r, c in centers_found:
        kept_arms = sum(int(orange_keep[r + dr, c + dc]) for dr, dc in DIR4)
        if 0 < kept_arms < 4:
            partial_exists = True
            break
    if not partial_exists:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out == (YELLOW if len(survivors) % 2 == 1 else GREEN)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

TOWER_COLORS = (RED, BLUE, YELLOW)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_TO_MASK = {RED: 1, BLUE: 2, YELLOW: 4}
MASK_TO_COLOR = {
    0: BLACK,
    1: RED,
    2: BLUE,
    3: MAGENTA,
    4: YELLOW,
    5: ORANGE,
    6: GREEN,
    7: MAROON,
}


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    out = g.copy()
    masks = np.zeros((h, w), dtype=np.uint8)

    tower_positions = np.argwhere(np.isin(g, TOWER_COLORS))
    tower_set = {tuple(map(int, pos)) for pos in tower_positions}
    blocked_by_wall = 0
    blocked_by_tower = 0

    for r0_raw, c0_raw in tower_positions:
        r0 = int(r0_raw)
        c0 = int(c0_raw)
        ray_mask = COLOR_TO_MASK[int(g[r0, c0])]
        for dr, dc in DIRS:
            r = r0 + dr
            c = c0 + dc
            while 0 <= r < h and 0 <= c < w:
                if int(g[r, c]) == GRAY:
                    blocked_by_wall += 1
                    break
                if (r, c) in tower_set:
                    blocked_by_tower += 1
                    break
                masks[r, c] |= ray_mask
                r += dr
                c += dc

    ray_pixels = masks != 0
    out[ray_pixels] = np.vectorize(MASK_TO_COLOR.get)(masks[ray_pixels])

    stats = {
        "tower_count": int(len(tower_positions)),
        "wall_count": int(np.sum(g == GRAY)),
        "ray_pixels": int(np.sum(ray_pixels)),
        "mix2_pixels": int(np.sum(np.isin(masks, [3, 5, 6]))),
        "mix3_pixels": int(np.sum(masks == 7)),
        "blocked_by_wall": int(blocked_by_wall),
        "blocked_by_tower": int(blocked_by_tower),
        "distinct_tower_colors": int(len(set(map(int, g[np.isin(g, TOWER_COLORS)])))),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored signal towers emit orthogonal rays until blocked by gray walls or other towers.
    Crossing rays mix by visible color-combination rules.
    """
    out, _stats = _simulate(grid)
    return out


def _can_place_tower(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if grid[r, c] != BLACK:
        return False
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[rr, cc]) != BLACK:
                return False
    return True


def _can_place_wall(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = grid.shape
    rr0 = max(0, min(r0, r1) - 1)
    rr1 = min(h - 1, max(r0, r1) + 1)
    cc0 = max(0, min(c0, c1) - 1)
    cc1 = min(w - 1, max(c0, c1) + 1)
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            if int(grid[rr, cc]) in TOWER_COLORS or int(grid[rr, cc]) == GRAY:
                return False
    return True


def _draw_wall(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        c_start, c_end = sorted((c0, c1))
        grid[r0, c_start : c_end + 1] = GRAY
    else:
        r_start, r_end = sorted((r0, r1))
        grid[r_start : r_end + 1, c0] = GRAY


def _place_towers(grid: np.ndarray, rng: random.Random, desired_count: int) -> bool:
    h, w = grid.shape
    colors = list(TOWER_COLORS)
    if desired_count >= 3:
        tower_colors = colors[:]
        while len(tower_colors) < desired_count:
            tower_colors.append(rng.choice(colors))
    else:
        tower_colors = rng.sample(colors, k=2)
    rng.shuffle(tower_colors)

    placed = 0
    attempts = 0
    while placed < desired_count and attempts < 400:
        attempts += 1
        r = rng.randint(1, h - 2)
        c = rng.randint(1, w - 2)
        if not _can_place_tower(grid, r, c):
            continue
        grid[r, c] = tower_colors[placed]
        placed += 1
    return placed == desired_count


def _place_walls(grid: np.ndarray, rng: random.Random, desired_count: int) -> bool:
    h, w = grid.shape
    placed = 0
    attempts = 0
    while placed < desired_count and attempts < 500:
        attempts += 1
        horizontal = bool(rng.randint(0, 1))
        if horizontal:
            length = rng.randint(2, max(2, min(8, w - 4)))
            r = rng.randint(1, h - 2)
            c0 = rng.randint(1, w - length - 1)
            c1 = c0 + length - 1
            if not _can_place_wall(grid, r, c0, r, c1):
                continue
            _draw_wall(grid, r, c0, r, c1)
        else:
            length = rng.randint(2, max(2, min(8, h - 4)))
            c = rng.randint(1, w - 2)
            r0 = rng.randint(1, h - length - 1)
            r1 = r0 + length - 1
            if not _can_place_wall(grid, r0, c, r1, c):
                continue
            _draw_wall(grid, r0, c, r1, c)
        placed += 1
    return placed == desired_count


def _matches_mode(stats: dict, mode: str) -> bool:
    if stats["ray_pixels"] < 8:
        return False
    if mode == "simple":
        return (
            stats["mix2_pixels"] == 0
            and stats["mix3_pixels"] == 0
            and stats["blocked_by_wall"] > 0
        )
    if mode == "mix2":
        return stats["mix2_pixels"] > 0 and stats["mix3_pixels"] == 0 and stats["blocked_by_wall"] > 0
    if mode == "mix3":
        return stats["mix3_pixels"] > 0 and stats["blocked_by_wall"] > 0
    if mode == "tower_stop":
        return stats["blocked_by_tower"] > 0 and stats["mix2_pixels"] > 0
    return False


def _gen_attempt(rng: random.Random, mode: str) -> np.ndarray | None:
    h = rng.randint(11, 22)
    w = rng.randint(11, 22)
    grid = np.full((h, w), BLACK, dtype=int)

    tower_count = rng.randint(3, 5) if mode in {"mix2", "mix3", "tower_stop"} else rng.randint(2, 4)
    wall_count = rng.randint(1, 5)

    if not _place_towers(grid, rng, tower_count):
        return None
    if not _place_walls(grid, rng, wall_count):
        return None

    out, stats = _simulate(grid)
    if stats["distinct_tower_colors"] < 2:
        return None
    if np.array_equal(out, grid):
        return None
    if not _matches_mode(stats, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    modes = ("simple", "mix2", "mix3", "tower_stop")
    for _ in range(600):
        rng = random.Random()
        mode = rng.choice(modes)
        grid = _gen_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")

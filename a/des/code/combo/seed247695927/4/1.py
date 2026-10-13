from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


MAX_ATTEMPTS = 300
MIN_SIZE = 3
MAX_SIZE = 8
PALETTE = tuple(range(10))


def _side_from_top_repetition(grid: np.ndarray) -> str | None:
    width = int(grid.shape[1])
    by_color: dict[int, list[int]] = defaultdict(list)
    for col, color in enumerate(map(int, grid[0])):
        by_color[color].append(col)

    candidates: list[tuple[int, str]] = []
    for cols in by_color.values():
        if len(cols) < 2:
            continue
        for col in cols:
            left_dist = col
            right_dist = width - 1 - col
            if left_dist < right_dist:
                candidates.append((left_dist, "left"))
            elif right_dist < left_dist:
                candidates.append((right_dist, "right"))

    if not candidates:
        return None

    best_dist = min(dist for dist, _ in candidates)
    best_sides = {side for dist, side in candidates if dist == best_dist}
    assert len(best_sides) == 1, "nearest repeated top-row occurrence must select one side"
    return next(iter(best_sides))


def _side_from_rarer_corner(grid: np.ndarray) -> str:
    counts = Counter(map(int, grid.ravel()))
    left = int(grid[0, 0])
    right = int(grid[0, -1])
    assert counts[left] != counts[right], "top-corner colors must have different frequencies"
    return "left" if counts[left] < counts[right] else "right"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height >= 2, "input must be a square grid with a 2x2 corner crop"

    side = _side_from_top_repetition(g)
    if side is None:
        side = _side_from_rarer_corner(g)

    col0 = 0 if side == "left" else width - 2
    return g[:2, col0 : col0 + 2].copy()


def _random_body_positions(size: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(1, size) for c in range(size)]
    random.shuffle(positions)
    return positions


def _choose_repeat_pair(size: int, side: str) -> tuple[int, int] | None:
    valid: list[tuple[int, int]] = []
    for a in range(size):
        for b in range(a + 1, size):
            pair = [a, b]
            candidates: list[tuple[int, str]] = []
            for col in pair:
                left_dist = col
                right_dist = size - 1 - col
                if left_dist < right_dist:
                    candidates.append((left_dist, "left"))
                elif right_dist < left_dist:
                    candidates.append((right_dist, "right"))
            if not candidates:
                continue
            best_dist = min(dist for dist, _ in candidates)
            best_sides = {s for dist, s in candidates if dist == best_dist}
            if best_sides == {side}:
                valid.append((a, b))
    return random.choice(valid) if valid else None


def _gen_repetition_case(size: int, side: str) -> np.ndarray | None:
    pair = _choose_repeat_pair(size, side)
    if pair is None:
        return None

    colors = random.sample(PALETTE, size - 1)
    repeated_color = colors[0]
    other_colors = iter(colors[1:])

    top = [None] * size
    for col in pair:
        top[col] = repeated_color
    for col in range(size):
        if top[col] is None:
            top[col] = next(other_colors)

    grid = np.empty((size, size), dtype=int)
    grid[0] = np.array(top, dtype=int)

    body_palette = list(dict.fromkeys(top))
    extra_count = random.randint(0, max(0, min(3, len(PALETTE) - len(body_palette))))
    if extra_count:
        extras = random.sample([c for c in PALETTE if c not in body_palette], extra_count)
        body_palette.extend(extras)

    for r, c in _random_body_positions(size):
        grid[r, c] = random.choice(body_palette)
    return grid


def _gen_rarer_case(size: int, side: str) -> np.ndarray:
    top = random.sample(PALETTE, size)
    grid = np.empty((size, size), dtype=int)
    grid[0] = np.array(top, dtype=int)

    target_color = int(top[0] if side == "left" else top[-1])
    other_corner_color = int(top[-1] if side == "left" else top[0])
    body_positions = _random_body_positions(size)
    available = len(body_positions)

    # Keep the chosen corner rarer by adding fewer copies of it than the opposite corner.
    target_add = random.randint(0, max(0, available // 3))
    other_min = target_add + 1
    other_max = min(available, max(other_min, (2 * available) // 3))
    other_add = random.randint(other_min, other_max)

    used = target_add + other_add
    filler_palette = [c for c in PALETTE if c not in (target_color, other_corner_color)]
    forced_new_count = random.randint(0, min(3, len(filler_palette), available - used))
    forced_new = random.sample(filler_palette, forced_new_count)

    values = (
        [target_color] * target_add
        + [other_corner_color] * other_add
        + forced_new
    )
    remaining = available - len(values)
    filler_pool = list(dict.fromkeys(top + forced_new))
    filler_pool = [c for c in filler_pool if c not in (target_color, other_corner_color)] or filler_palette
    values.extend(random.choice(filler_pool) for _ in range(remaining))
    random.shuffle(values)

    for (r, c), value in zip(body_positions, values):
        grid[r, c] = value
    return grid


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(MIN_SIZE, MAX_SIZE)
    side = random.choice(("left", "right"))
    mode = random.choice(("repetition", "rarer"))

    if mode == "repetition":
        grid = _gen_repetition_case(size, side)
        if grid is None:
            return None
    else:
        grid = _gen_rarer_case(size, side)

    expected = grid[:2, :2] if side == "left" else grid[:2, -2:]
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")

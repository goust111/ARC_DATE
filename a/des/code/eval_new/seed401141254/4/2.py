import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
POSITION_RETRIES = 80


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == BG).any()):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rank_colors_by_count(grid: np.ndarray) -> list[int]:
    bg = _background_color(grid)
    cells = grid[grid != bg]
    assert cells.size > 0, "expected at least one colored pixel"

    counts = Counter(map(int, cells.tolist()))
    freqs = list(counts.values())
    assert len(freqs) == len(set(freqs)), "each color count must be unique"

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [int(color) for color, _count in ranked]


def _build_nested_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), BG, dtype=int)

    for layer, color in enumerate(colors):
        lo = int(layer)
        hi = int(side - layer - 1)
        out[lo, lo : hi + 1] = int(color)
        out[hi, lo : hi + 1] = int(color)
        out[lo : hi + 1, lo] = int(color)
        out[lo : hi + 1, hi] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert scattered same-color pixel groups into count-ordered concentric square rings.

    Input structure:
    - Black background with isolated non-black single pixels scattered across the grid.
    - Each non-black color may appear multiple times, but every color's total count is unique.
    - Pixel positions are arbitrary; only color identity and per-color multiplicity matter.

    Rule:
    - Count how many pixels belong to each non-black color.
    - Sort colors from highest count to lowest count.
    - Build a square output of side length 2K-1, where K is the number of distinct non-black colors.
    - Draw 1-pixel-thick nested square borders from outermost to innermost in that count order; if the final layer is a single cell, fill that center cell with the last color.

    Key insight: Pixel counts choose the nesting order only; ring sizes are determined solely by how many colors exist.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ranked_colors = _rank_colors_by_count(g)
    return _build_nested_rings(ranked_colors)


def _max_group_count(capacity: int) -> int:
    limit = 0
    max_colors = min(len(PALETTE), 15)
    while limit < max_colors and (limit + 1) * (limit + 2) // 2 <= capacity:
        limit += 1
    return limit


def _sample_unique_counts(rng: random.Random, color_count: int, capacity: int) -> list[int]:
    counts = list(range(1, color_count + 1))
    min_total = sum(counts)
    spare = max(0, int(capacity - min_total))
    extra_budget = rng.randint(0, min(spare, max(color_count * 5, 1)))

    while extra_budget > 0:
        valid_starts = [idx for idx in range(color_count) if color_count - idx <= extra_budget]
        if not valid_starts:
            break
        start = rng.choice(valid_starts)
        for idx in range(start, color_count):
            counts[idx] += 1
        extra_budget -= color_count - start

    counts.reverse()
    return counts


def _choose_isolated_positions(
    height: int, width: int, total_points: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(POSITION_RETRIES):
        order = cells[:]
        rng.shuffle(order)
        blocked = np.zeros((height, width), dtype=bool)
        chosen: list[tuple[int, int]] = []

        for r, c in order:
            if bool(blocked[r, c]):
                continue
            chosen.append((int(r), int(c)))
            blocked[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] = True
            if len(chosen) == total_points:
                return chosen

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(6, 30)
    width = rng.randint(6, 30)

    rough_capacity = ((height + 1) // 2) * ((width + 1) // 2)
    soft_capacity = max(1, min(rough_capacity, int(rough_capacity * 0.7)))
    max_groups = _max_group_count(soft_capacity)
    if max_groups == 0:
        return None

    color_count = rng.randint(1, max_groups)
    counts = _sample_unique_counts(rng, color_count, soft_capacity)
    total_points = int(sum(counts))

    positions = _choose_isolated_positions(height, width, total_points, rng)
    if positions is None:
        return None

    colors = rng.sample(PALETTE, color_count)
    pairs = list(zip(colors, counts))
    rng.shuffle(pairs)

    grid = np.zeros((height, width), dtype=int)
    start = 0
    for color, count in pairs:
        for r, c in positions[start : start + count]:
            grid[int(r), int(c)] = int(color)
        start += count

    non_bg = grid[grid != BG]
    if non_bg.size != total_points:
        return None

    if len(np.unique(non_bg)) != color_count:
        return None

    if max(main(grid).shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")

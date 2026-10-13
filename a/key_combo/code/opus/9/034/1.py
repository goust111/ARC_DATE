import random
from collections import Counter

import numpy as np


BG = 0
OBSTACLE = 5
BASELINE = 7
DEFAULT_SEED_COLORS = (1, 2, 3, 4)
BAR_COLOR_POOL = (1, 2, 3, 4, 6, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_baseline(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            candidates.append((r, int(row[0])))
    if not candidates:
        raise ValueError("no full-width baseline row found")
    return candidates[-1]


def _extract_bars(grid: np.ndarray, base_r: int, base_color: int, bg: int) -> list[dict]:
    bars = []
    for c in range(grid.shape[1]):
        if int(grid[base_r, c]) != base_color:
            continue
        if base_r == 0 or int(grid[base_r - 1, c]) == bg:
            continue
        color = int(grid[base_r - 1, c])
        h = 0
        r = base_r - 1
        while r >= 0 and int(grid[r, c]) == color:
            h += 1
            r -= 1
        bars.append({"col": c, "color": color, "height": h})
    bars.sort(key=lambda x: x["col"])
    return bars


def _extract_seeds_and_lengths(grid: np.ndarray, base_r: int, bg: int) -> list[dict]:
    H, W = grid.shape
    seeds = []
    for r in range(H):
        if r == base_r:
            continue
        color = int(grid[r, 0])
        if color == bg:
            continue
        obstacle_cols = [c for c in range(1, W) if int(grid[r, c]) == OBSTACLE]
        if len(obstacle_cols) != 1:
            raise ValueError("each seed row must contain exactly one obstacle")
        obstacle_c = obstacle_cols[0]
        seeds.append(
            {
                "row": r,
                "color": color,
                "obstacle_col": obstacle_c,
                "free_len": obstacle_c - 1,
            }
        )
    seeds.sort(key=lambda x: x["row"])
    return seeds


def _build_output(
    shape: tuple[int, int],
    bg: int,
    base_r: int,
    base_color: int,
    bars: list[dict],
    new_heights: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[base_r, :] = base_color
    for bar, h in zip(bars, new_heights):
        c = int(bar["col"])
        color = int(bar["color"])
        out[base_r - h : base_r, c] = color
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    base_r, base_color = _find_baseline(g, bg)
    bars = _extract_bars(g, base_r, base_color, bg)
    seeds = _extract_seeds_and_lengths(g, base_r, bg)

    if len(bars) != 4:
        raise ValueError("expected exactly four bars")
    if len(seeds) != 4:
        raise ValueError("expected exactly four seeds")

    heights = sorted(int(bar["height"]) for bar in bars)
    ray_order = sorted(seeds, key=lambda x: (int(x["free_len"]), int(x["row"])))
    if len({int(seed["free_len"]) for seed in ray_order}) != 4:
        raise ValueError("ray lengths must be unique")

    assigned = [0] * 4
    for i, _seed in enumerate(ray_order):
        assigned[i] = heights[i]

    return _build_output(g.shape, bg, base_r, base_color, bars, assigned)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use obstacle-truncated ray lengths as a hidden ranking that redistributes bottom bar heights.

    Input structure:
    - The background is black, with four seed pixels in the leftmost column on separated rows.
    - Each seed row contains exactly one grey obstacle pixel to its right, and the bottom row is a full-width baseline.
    - Four single-column colored bars stand directly above that baseline at fixed slot columns, with pairwise distinct heights.

    Rule:
    - From each seed, simulate a horizontal ray to the right and stop just before the grey obstacle on that row.
    - Record the free ray length as the number of painted cells excluding the seed and the obstacle, then sort the four rays from shortest to longest.
    - Sort the four existing bar heights from shortest to tallest, assign the i-th sorted height to the i-th left-to-right bar slot according to the ray ranking, keep each slot's column and color, and erase every seed, ray, and obstacle.

    Key insight:
    - The obstacle columns do not directly move the bars; instead they encode an implicit order through ray simulation, and that order determines how the bar heights are reassembled.
    """
    return _solve(grid)


def _sample_spaced_rows(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(lo, hi + 1))
    random.shuffle(candidates)
    chosen = []
    for value in candidates:
        if all(abs(value - prev) >= min_gap for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _sample_spaced_cols(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    return _sample_spaced_rows(lo, hi, count, min_gap)


def _structure_ok(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        base_r, base_color = _find_baseline(grid, bg)
        bars = _extract_bars(grid, base_r, base_color, bg)
        seeds = _extract_seeds_and_lengths(grid, base_r, bg)
    except Exception:
        return False

    if len(bars) != 4 or len(seeds) != 4:
        return False
    if len({int(bar["height"]) for bar in bars}) != 4:
        return False
    if len({int(seed["free_len"]) for seed in seeds}) != 4:
        return False
    if any(int(seed["obstacle_col"]) < 3 for seed in seeds):
        return False
    if any(abs(int(seeds[i]["row"]) - int(seeds[i - 1]["row"])) < 2 for i in range(1, 4)):
        return False

    max_bar_h = max(int(bar["height"]) for bar in bars)
    bar_top = base_r - max_bar_h
    if any(int(seed["row"]) >= bar_top - 1 for seed in seeds):
        return False

    non_bg_on_seed_rows = []
    for seed in seeds:
        r = int(seed["row"])
        coords = [(r, c) for c in range(grid.shape[1]) if int(grid[r, c]) != bg]
        non_bg_on_seed_rows.append(coords)
    if any(len(coords) != 2 for coords in non_bg_on_seed_rows):
        return False

    heights_by_slot = [int(bar["height"]) for bar in bars]
    if heights_by_slot == sorted(heights_by_slot):
        return False

    out = main(grid)
    out_bars = _extract_bars(out, base_r, base_color, bg)
    if [int(bar["height"]) for bar in out_bars] != sorted(heights_by_slot):
        return False
    if np.array_equal(out, grid):
        return False
    if np.any(out[:base_r, 0] != bg) or np.any(out[:-1, :] == OBSTACLE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(15, 22)
    W = random.randint(14, 22)
    base_r = H - 1

    heights = random.sample(range(2, 8), 4)
    max_bar_h = max(heights)
    seed_hi = base_r - max_bar_h - 2
    if seed_hi < 7:
        return None

    seed_rows = _sample_spaced_rows(1, seed_hi, 4, 2)
    if seed_rows is None:
        return None

    slot_cols = _sample_spaced_cols(2, W - 2, 4, 2)
    if slot_cols is None:
        return None

    obstacle_candidates = [c for c in range(3, W - 1) if c not in slot_cols]
    if len(obstacle_candidates) < 4:
        return None
    obstacle_cols = random.sample(obstacle_candidates, 4)

    bar_colors = random.sample(BAR_COLOR_POOL, 4)
    grid = np.full((H, W), BG, dtype=int)
    grid[base_r, :] = BASELINE

    for c, color, h in zip(slot_cols, bar_colors, heights):
        grid[base_r - h : base_r, c] = color

    seed_colors = list(DEFAULT_SEED_COLORS)
    random.shuffle(seed_colors)
    for r, color, obstacle_c in zip(seed_rows, seed_colors, obstacle_cols):
        grid[r, 0] = color
        grid[r, obstacle_c] = OBSTACLE

    if not _structure_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-rank bar input")

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend pattern indexed by its seed count.

    Input structure:
    - Background is black.
    - A grey partition on the right side creates closed regions.
    - Each region contains 0..4 red seed pixels.
    - Five 3x3 blue legend patterns are stacked from top to bottom on the left edge.

    Rule:
    - Count the red seeds inside each closed region.
    - Use that count as the index into the left legend.
    - Tile the selected 3x3 pattern across the region, aligned to the region's
      top-left pixel, painting only blue or black inside the region.
    - Remove the original seeds.
    - Keep the grey partition and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend_color, patterns = _extract_legend(g, bg)

    other_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, legend_color)]
    assert other_colors, "expected partition and seed colors"
    wall_color = max(other_colors, key=lambda c: int(np.count_nonzero(g == c)))

    out = g.copy()
    for comp in _region_components(g, wall_color, legend_color):
        seed_count = sum(1 for r, c in comp if int(g[r, c]) != bg)
        assert 0 <= seed_count <= 4, "seed count must index one of the five legend patterns"
        pattern = patterns[seed_count]
        anchor_r, anchor_c = min(comp)
        for r, c in comp:
            out[r, c] = legend_color if pattern[(r - anchor_r) % 3, (c - anchor_c) % 3] else bg

    return out


# --------------- generator ---------------

BG = 0
LEGEND = 1
SEED = 2
WALL = 5

GRID_H_MIN = 23
GRID_H_MAX = 30
GRID_W_MIN = 22
GRID_W_MAX = 30

LEGEND_GAPS = (1, 2)
OUTER_TOP_MIN = 1
OUTER_TOP_MAX = 3
OUTER_BOTTOM_MARGINS = (2, 3, 4)
OUTER_LEFT_MIN = 5
OUTER_LEFT_MAX = 7
OUTER_RIGHT_MARGINS = (2, 3)

MIN_REGION_COUNT = 5
MAX_REGION_COUNT = 8
MIN_INNER_SPAN = 3
MAX_GEN_ATTEMPTS = 400

PATTERN_BANK = tuple(
    np.array(rows, dtype=int)
    for rows in (
        ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
        ((1, 1, 0), (0, 1, 1), (1, 1, 0)),
        ((0, 1, 1), (1, 1, 0), (0, 1, 1)),
        ((1, 0, 0), (1, 1, 1), (0, 0, 1)),
        ((0, 0, 1), (1, 1, 1), (1, 0, 0)),
        ((0, 1, 0), (1, 1, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
        ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
        ((1, 0, 1), (1, 1, 0), (0, 1, 1)),
    )
)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")


def _background_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.reshape(-1))).most_common(1)[0][0]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)


def _contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if rows.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for value in rows[1:]:
        value = int(value)
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[int, tuple[np.ndarray, ...]]:
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = np.flatnonzero(np.any(grid[:, :3] == color, axis=1))
        runs = _contiguous_runs(rows)
        if len(runs) == 5 and all(end - start + 1 == 3 for start, end in runs):
            candidates.append((color, runs))

    assert len(candidates) == 1, "expected exactly one left-edge legend color"
    legend_color, runs = candidates[0]
    patterns = []
    for start, _ in runs:
        block = (grid[start : start + 3, :3] == legend_color).astype(int)
        assert block.shape == (3, 3)
        patterns.append(block)
    return int(legend_color), tuple(patterns)


def _region_components(grid: np.ndarray, wall_color: int, legend_color: int) -> list[list[tuple[int, int]]]:
    traversable = (grid != wall_color) & (grid != legend_color)
    traversable[:, :3] = False
    h, w = grid.shape
    return [comp for comp in _components(traversable) if not _touches_border(comp, h, w)]


def _region_area(region: tuple[int, int, int, int]) -> int:
    top, bottom, left, right = region
    return (bottom - top - 1) * (right - left - 1)


def _split_options(region: tuple[int, int, int, int]) -> list[str]:
    top, bottom, left, right = region
    inner_h = bottom - top - 1
    inner_w = right - left - 1
    options = []
    if inner_w >= 2 * MIN_INNER_SPAN + 1:
        options.append("v")
    if inner_h >= 2 * MIN_INNER_SPAN + 1:
        options.append("h")
    return options


def _choose_orientation(region: tuple[int, int, int, int], options: list[str], rng: random.Random) -> str:
    if len(options) == 1:
        return options[0]
    top, bottom, left, right = region
    inner_h = bottom - top - 1
    inner_w = right - left - 1
    if inner_w > inner_h and rng.random() < 0.65:
        return "v"
    if inner_h > inner_w and rng.random() < 0.65:
        return "h"
    return rng.choice(options)


def _split_region(
    region: tuple[int, int, int, int], orient: str, rng: random.Random
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    top, bottom, left, right = region
    if orient == "v":
        split_col = rng.choice(list(range(left + MIN_INNER_SPAN + 1, right - MIN_INNER_SPAN)))
        left_child = (top, bottom, left, split_col)
        right_child = (top, bottom, split_col, right)
        return split_col, left_child, right_child

    split_row = rng.choice(list(range(top + MIN_INNER_SPAN + 1, bottom - MIN_INNER_SPAN)))
    top_child = (top, split_row, left, right)
    bottom_child = (split_row, bottom, left, right)
    return split_row, top_child, bottom_child


def _place_legend(grid: np.ndarray, patterns: list[np.ndarray], rng: random.Random) -> bool:
    h = grid.shape[0]
    gaps = [rng.choice(LEGEND_GAPS) for _ in range(4)]
    used_rows = 15 + sum(gaps)
    slack = h - used_rows
    if slack < 2:
        return False

    top_margin = rng.randint(1, slack - 1)
    row = top_margin
    for idx, pattern in enumerate(patterns):
        grid[row : row + 3, :3][pattern == 1] = LEGEND
        row += 3
        if idx < 4:
            row += gaps[idx]
    return True


def _interior_cells(region: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    top, bottom, left, right = region
    return [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]


def _choose_seed_cells(
    cells: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]]:
    if count == 0:
        return []
    if count == 1:
        return [rng.choice(cells)]

    best = rng.sample(cells, count)
    best_score = -1
    for _ in range(20):
        candidate = rng.sample(cells, count)
        score = min(
            abs(r1 - r2) + abs(c1 - c2)
            for i, (r1, c1) in enumerate(candidate)
            for r2, c2 in candidate[i + 1 :]
        )
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    w = rng.randint(GRID_W_MIN, GRID_W_MAX)
    grid = np.full((h, w), BG, dtype=int)

    patterns = [PATTERN_BANK[i].copy() for i in rng.sample(range(len(PATTERN_BANK)), 5)]
    if not _place_legend(grid, patterns, rng):
        return None

    top = rng.randint(OUTER_TOP_MIN, OUTER_TOP_MAX)
    bottom = h - rng.choice(OUTER_BOTTOM_MARGINS)
    left = rng.randint(OUTER_LEFT_MIN, OUTER_LEFT_MAX)
    right = w - rng.choice(OUTER_RIGHT_MARGINS)

    if bottom - top - 1 < 2 * MIN_INNER_SPAN + 1 or right - left - 1 < 2 * MIN_INNER_SPAN + 1:
        return None

    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL

    regions = [(top, bottom, left, right)]
    target_regions = rng.randint(MIN_REGION_COUNT, MAX_REGION_COUNT)
    for _ in range(target_regions - 1):
        candidates = [(region, _split_options(region)) for region in regions]
        candidates = [(region, options) for region, options in candidates if options]
        if not candidates:
            break

        candidates.sort(key=lambda item: _region_area(item[0]), reverse=True)
        region, options = rng.choice(candidates[: min(3, len(candidates))])
        orient = _choose_orientation(region, options, rng)
        split_pos, child_a, child_b = _split_region(region, orient, rng)

        if orient == "v":
            grid[region[0] : region[1] + 1, split_pos] = WALL
        else:
            grid[split_pos, region[2] : region[3] + 1] = WALL

        regions.remove(region)
        regions.extend([child_a, child_b])

    if len(regions) < MIN_REGION_COUNT:
        return None

    counts = [0, 1, 2, 3, 4] + [rng.randint(0, 4) for _ in range(len(regions) - 5)]
    rng.shuffle(counts)
    shuffled_regions = regions[:]
    rng.shuffle(shuffled_regions)

    for region, count in zip(shuffled_regions, counts):
        cells = _interior_cells(region)
        if len(cells) < count:
            return None
        for r, c in _choose_seed_cells(cells, count, rng):
            grid[r, c] = SEED

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.count_nonzero(out == SEED) != 0:
        return None
    if np.array_equal(out, grid):
        return None

    detected_regions = _region_components(grid, WALL, LEGEND)
    if len(detected_regions) != len(regions):
        return None

    return grid

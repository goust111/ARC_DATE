from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

TOWER_COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE)

GEN_RETRIES = 400
GRID_H_MAX = 18
GRID_W_MIN = 10
GRID_W_MAX = 30
TOP_MARGIN_MAX = 4
BOTTOM_MARGIN_MAX = 2
INTER_BLOCK_GAP = 1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _baseline_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        color = int(row[0])
        if color != bg and bool(np.all(row == color)):
            candidates.append((int(r), color))
    assert candidates, "expected a full-width non-background baseline row"
    return max(candidates, key=lambda item: item[0])


def _detect_towers(grid: np.ndarray, baseline_r: int, baseline_color: int, bg: int) -> list[dict[str, int]]:
    assert baseline_r > 0, "baseline must have at least one row above it"
    h, w = grid.shape
    towers: list[dict[str, int]] = []
    tower_cells: set[tuple[int, int]] = set()

    for c in range(w):
        color = int(grid[baseline_r - 1, c])
        if color in {bg, baseline_color}:
            continue

        height = 0
        r = baseline_r - 1
        while r >= 0 and int(grid[r, c]) == color:
            tower_cells.add((int(r), int(c)))
            height += 1
            r -= 1

        assert height > 0, "tower height must be positive"
        if r >= 0:
            assert int(grid[r, c]) == bg, "tower must be isolated above"
        towers.append({"col": int(c), "height": int(height), "color": color})

    assert len(towers) >= 2, "expected at least two towers"

    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value == bg:
                continue
            if r == baseline_r:
                assert value == baseline_color, "baseline row must be monochrome"
                continue
            assert (int(r), int(c)) in tower_cells, "unexpected foreground outside towers"

    return towers


def _coverage_counts(
    shape: tuple[int, int],
    baseline_r: int,
    towers: list[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    counts = np.zeros(shape, dtype=int)
    single_color = np.full(shape, BLACK, dtype=int)

    for tower in towers:
        col = int(tower["col"])
        height = int(tower["height"])
        color = int(tower["color"])
        top_r = baseline_r - height

        for step in range(1, height + 1):
            r = top_r - step
            if not (0 <= r < h):
                continue
            half_width = step - 1
            left = max(0, col - half_width)
            right = min(w - 1, col + half_width)
            for c in range(left, right + 1):
                if counts[r, c] == 0:
                    single_color[r, c] = color
                counts[r, c] += 1

    return counts, single_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each signal tower into an upward-opening triangular coverage area.

    Input structure:
    - black background
    - a full-width grey baseline in the lower part of the grid
    - several vertical one-column colored towers resting on the baseline

    Rule:
    - A tower of height h projects h rows upward, starting one cell above its top.
    - The first projected row has width 1, then width grows by 2 per row upward.
    - Black cells covered by exactly one tower take that tower's color.
    - Black cells covered by multiple towers become maroon.
    - Towers and the grey baseline stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    baseline_r, baseline_color = _baseline_row(g, bg)
    towers = _detect_towers(g, baseline_r, baseline_color, bg)
    counts, single_color = _coverage_counts(g.shape, baseline_r, towers)

    out = g.copy()
    black_mask = g == bg
    unique_mask = black_mask & (counts == 1)
    overlap_mask = black_mask & (counts >= 2)
    out[unique_mask] = single_color[unique_mask]
    out[overlap_mask] = MAROON
    return out


def _composition(total: int, parts: int) -> list[int]:
    shares = [0] * parts
    for _ in range(total):
        shares[random.randrange(parts)] += 1
    return shares


def _assign_colors(n_towers: int, *, force_first_two_distinct: bool) -> list[int]:
    n_distinct = random.randint(2, min(4, n_towers))
    chosen = random.sample(list(TOWER_COLORS), n_distinct)
    colors: list[int] = []

    if force_first_two_distinct:
        pair = random.sample(chosen, 2)
        colors.extend(pair)
        required = [c for c in chosen if c not in pair]
    else:
        required = chosen[:]

    pool = required[:]
    while len(pool) < n_towers - len(colors):
        pool.append(random.choice(chosen))
    random.shuffle(pool)
    colors.extend(pool[: n_towers - len(colors)])
    assert len(colors) == n_towers
    return colors


def _height_and_baseline(heights: list[int]) -> tuple[int, int] | None:
    max_height = max(int(h) for h in heights)
    top_margin = random.randint(0, TOP_MARGIN_MAX)
    baseline_r = 2 * max_height + top_margin
    grid_h = baseline_r + 1 + random.randint(0, BOTTOM_MARGIN_MAX)
    if grid_h > GRID_H_MAX:
        return None
    return int(grid_h), int(baseline_r)


def _choose_width(base_width: int) -> int | None:
    lo = max(GRID_W_MIN, int(base_width))
    hi = min(GRID_W_MAX, int(base_width) + 8)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _layout_separated(radii: list[int]) -> tuple[int, list[int]] | None:
    block_widths = [2 * int(r) + 1 for r in radii]
    base_width = sum(block_widths) + INTER_BLOCK_GAP * max(0, len(block_widths) - 1)
    grid_w = _choose_width(base_width)
    if grid_w is None:
        return None

    slack = grid_w - base_width
    extras = _composition(slack, len(radii) + 1)
    cols: list[int] = []
    cursor = extras[0]
    for i, (radius, block_w) in enumerate(zip(radii, block_widths)):
        left = cursor
        cols.append(int(left + radius))
        cursor = left + block_w
        if i < len(radii) - 1:
            cursor += INTER_BLOCK_GAP + extras[i + 1]
    return int(grid_w), cols


def _layout_with_overlap(core_radius: int, extra_radii: list[int]) -> tuple[int, tuple[int, int], list[int]] | None:
    distance = random.randint(1, 2 * core_radius)
    pair_width = 2 * core_radius + distance + 1

    blocks: list[tuple[str, int, int | None]] = [("pair", pair_width, None)]
    blocks.extend(("single", 2 * int(radius) + 1, idx) for idx, radius in enumerate(extra_radii))
    random.shuffle(blocks)

    base_width = sum(block_width for _, block_width, _ in blocks) + INTER_BLOCK_GAP * max(0, len(blocks) - 1)
    grid_w = _choose_width(base_width)
    if grid_w is None:
        return None

    slack = grid_w - base_width
    extras = _composition(slack, len(blocks) + 1)
    pair_cols: tuple[int, int] | None = None
    extra_cols: list[int | None] = [None] * len(extra_radii)
    cursor = extras[0]

    for i, (kind, block_width, idx) in enumerate(blocks):
        left = cursor
        if kind == "pair":
            first_col = left + core_radius
            pair_cols = (int(first_col), int(first_col + distance))
        else:
            assert idx is not None
            extra_cols[idx] = int(left + extra_radii[idx])
        cursor = left + block_width
        if i < len(blocks) - 1:
            cursor += INTER_BLOCK_GAP + extras[i + 1]

    assert pair_cols is not None
    assert all(col is not None for col in extra_cols)
    return int(grid_w), pair_cols, [int(col) for col in extra_cols]


def _render_input(grid_h: int, grid_w: int, baseline_r: int, towers: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[baseline_r, :] = GREY
    for tower in towers:
        col = int(tower["col"])
        height = int(tower["height"])
        color = int(tower["color"])
        grid[baseline_r - height : baseline_r, col] = color
    return grid


def _validate_branch(inp: np.ndarray, *, expect_overlap: bool) -> bool:
    out = main(inp)
    changed = out != inp
    has_overlap = bool(np.any(out == MAROON))
    has_single_fill = bool(np.any(changed & (out != MAROON)))
    if not has_single_fill:
        return False
    if expect_overlap:
        return has_overlap
    return not has_overlap


def _gen_attempt() -> np.ndarray | None:
    expect_overlap = bool(random.getrandbits(1))
    n_towers = random.randint(2, 4)

    if expect_overlap:
        core_height = random.randint(2, 5)
        heights = [core_height, core_height]
        heights.extend(random.randint(1, 5) for _ in range(n_towers - 2))
        height_info = _height_and_baseline(heights)
        if height_info is None:
            return None
        grid_h, baseline_r = height_info

        colors = _assign_colors(n_towers, force_first_two_distinct=True)
        extras = list(zip(heights[2:], colors[2:]))
        random.shuffle(extras)
        extra_heights = [int(height) for height, _ in extras]
        extra_colors = [int(color) for _, color in extras]

        layout = _layout_with_overlap(core_height - 1, [height - 1 for height in extra_heights])
        if layout is None:
            return None
        grid_w, pair_cols, extra_cols = layout

        towers = [
            {"col": int(pair_cols[0]), "height": int(core_height), "color": int(colors[0])},
            {"col": int(pair_cols[1]), "height": int(core_height), "color": int(colors[1])},
        ]
        towers.extend(
            {"col": int(col), "height": int(height), "color": int(color)}
            for col, height, color in zip(extra_cols, extra_heights, extra_colors)
        )
    else:
        heights = [random.randint(1, 5) for _ in range(n_towers)]
        colors = _assign_colors(n_towers, force_first_two_distinct=False)
        descriptors = list(zip(heights, colors))
        random.shuffle(descriptors)
        heights = [int(height) for height, _ in descriptors]
        colors = [int(color) for _, color in descriptors]

        height_info = _height_and_baseline(heights)
        if height_info is None:
            return None
        grid_h, baseline_r = height_info

        layout = _layout_separated([height - 1 for height in heights])
        if layout is None:
            return None
        grid_w, cols = layout

        towers = [
            {"col": int(col), "height": int(height), "color": int(color)}
            for col, height, color in zip(cols, heights, colors)
        ]

    towers.sort(key=lambda tower: int(tower["col"]))
    inp = _render_input(grid_h, grid_w, baseline_r, towers)

    used_colors = {int(tower["color"]) for tower in towers}
    if not (2 <= len(used_colors) <= 4):
        return None
    if not _validate_branch(inp, expect_overlap=expect_overlap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

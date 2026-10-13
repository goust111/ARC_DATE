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
TEAL = 8

PAIR_MAP = {
    TEAL: RED,
    GREEN: BLUE,
    GREY: YELLOW,
}

INPUT_DOT_COLORS = tuple(PAIR_MAP)
REGION_SIZE_RANGE = (5, 6)
DOT_COUNT_CHOICES = (2, 3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 400

_winner_color_index = 0


def _full_constant_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]


def _full_constant_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers):
        if start < divider:
            segments.append((start, divider))
        start = divider + 1
    if start < length:
        segments.append((start, length))
    return segments


def _partition_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    row_dividers = _full_constant_rows(grid)
    col_dividers = _full_constant_cols(grid)
    assert row_dividers and col_dividers, "expected at least one full divider row and column"

    divider_colors = {int(grid[r, 0]) for r in row_dividers} | {int(grid[0, c]) for c in col_dividers}
    assert len(divider_colors) == 1, "divider rows and columns must share one color"
    divider_color = next(iter(divider_colors))

    row_segments = _segments(grid.shape[0], row_dividers)
    col_segments = _segments(grid.shape[1], col_dividers)
    assert row_segments and col_segments

    regions: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            regions.append((r0, r1, c0, c1))
    return divider_color, regions


def _region_info(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    divider_color: int,
) -> dict[str, object]:
    r0, r1, c0, c1 = region
    sub = grid[r0:r1, c0:c1]
    dot_mask = (sub != BLACK) & (sub != divider_color)
    dot_colors = sorted(set(map(int, sub[dot_mask].tolist())))
    assert len(dot_colors) <= 1, "each region must contain at most one scatter color"
    dot_color = dot_colors[0] if dot_colors else None
    return {
        "region": region,
        "mask": dot_mask,
        "count": int(dot_mask.sum()),
        "color": dot_color,
    }


def _winner_index(infos: list[dict[str, object]]) -> int:
    counts = [int(info["count"]) for info in infos]
    max_count = max(counts)
    winners = [i for i, count in enumerate(counts) if count == max_count]
    assert len(winners) == 1, "expected a unique densest region"
    return winners[0]


def _transform(grid: np.ndarray) -> np.ndarray:
    divider_color, regions = _partition_regions(grid)
    infos = [_region_info(grid, region, divider_color) for region in regions]
    winner = _winner_index(infos)
    winner_color = infos[winner]["color"]
    assert winner_color in PAIR_MAP, "winner region must use one of the paired colors"

    out = np.array(grid, dtype=int, copy=True)
    for i, info in enumerate(infos):
        r0, r1, c0, c1 = info["region"]
        sub_out = out[r0:r1, c0:c1]
        sub_out[:] = BLACK
        if i == winner:
            fill_color = PAIR_MAP[int(winner_color)]
            sub_out[:] = fill_color
            sub_out[info["mask"]] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the densest divider-bounded region, then invert its dot mask into the paired output color.

    Input structure:
    - Full orange rows and columns partition the canvas into rectangular regions.
    - Each region has a black background with scattered pixels of exactly one color from teal, green, or grey.
    - Different regions may reuse colors, but the number of scattered pixels is unique per region.

    Rule:
    - Detect the full orange divider lines and extract every rectangular region between them.
    - Count each region's non-black, non-divider scatter pixels and keep only the unique maximum region.
    - Erase all other regions to black, then invert the winner's local mask: black cells become the paired fill color and original scatter cells become black.
    - Use the fixed color pairing teal -> red, green -> blue, grey -> yellow; keep the orange divider lines unchanged.

    Key insight:
    - The winning region's scatter color serves three roles at once: it is what gets counted, it defines the mask to invert, and it selects the paired output fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _transform(g)


def _next_winner_color() -> int:
    global _winner_color_index
    color = INPUT_DOT_COLORS[_winner_color_index % len(INPUT_DOT_COLORS)]
    _winner_color_index += 1
    return int(color)


def _region_offsets(sizes: list[int]) -> list[int]:
    offsets = [0]
    cur = 0
    for size in sizes[:-1]:
        cur += size + 1
        offsets.append(cur)
    return offsets


def _inner_positions(height: int, width: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]


def _build_input() -> np.ndarray:
    heights = [random.randint(*REGION_SIZE_RANGE) for _ in range(2)]
    widths = [random.randint(*REGION_SIZE_RANGE) for _ in range(2)]
    grid = np.full((sum(heights) + 1, sum(widths) + 1), BLACK, dtype=int)

    divider_row = heights[0]
    divider_col = widths[0]
    grid[divider_row, :] = ORANGE
    grid[:, divider_col] = ORANGE

    region_origins = [
        (r0, c0)
        for r0 in _region_offsets(heights)
        for c0 in _region_offsets(widths)
    ]

    counts = random.sample(DOT_COUNT_CHOICES, 4)
    winner_region = random.randrange(4)
    max_idx = max(range(4), key=lambda idx: counts[idx])
    counts[winner_region], counts[max_idx] = counts[max_idx], counts[winner_region]

    winner_color = _next_winner_color()
    other_colors = [color for color in INPUT_DOT_COLORS if color != winner_color]
    distinct_other = random.choice(other_colors)
    palette = [winner_color, distinct_other]
    if random.random() < 0.5:
        palette.append(next(color for color in other_colors if color != distinct_other))

    region_colors = [winner_color] * 4
    forced_distinct_region = random.choice([idx for idx in range(4) if idx != winner_region])
    region_colors[forced_distinct_region] = distinct_other
    for idx in range(4):
        if idx == winner_region or idx == forced_distinct_region:
            continue
        region_colors[idx] = random.choice(palette)

    for idx, ((r0, c0), dot_count, dot_color) in enumerate(zip(region_origins, counts, region_colors)):
        height = heights[idx // 2]
        width = widths[idx % 2]
        positions = random.sample(_inner_positions(height, width), dot_count)
        for dr, dc in positions:
            grid[r0 + dr, c0 + dc] = int(dot_color)

    return grid


def _validate_input(grid: np.ndarray) -> None:
    divider_color, regions = _partition_regions(grid)
    assert divider_color == ORANGE
    assert len(regions) == 4

    colors_used: set[int] = set()
    counts: list[int] = []
    for region in regions:
        info = _region_info(grid, region, divider_color)
        r0, r1, c0, c1 = region
        sub = grid[r0:r1, c0:c1]
        assert sub.shape[0] in REGION_SIZE_RANGE and sub.shape[1] in REGION_SIZE_RANGE
        assert 2 <= int(info["count"]) <= 6
        assert int(info["color"]) in PAIR_MAP
        coords = np.argwhere(info["mask"])
        assert coords.size > 0
        assert np.all(coords[:, 0] > 0) and np.all(coords[:, 0] < sub.shape[0] - 1)
        assert np.all(coords[:, 1] > 0) and np.all(coords[:, 1] < sub.shape[1] - 1)
        counts.append(int(info["count"]))
        colors_used.add(int(info["color"]))

    assert len(set(counts)) == 4
    assert len(colors_used) >= 2

    out = _transform(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert np.all(grid[_full_constant_rows(grid), :] == ORANGE)
    assert np.all(grid[:, _full_constant_cols(grid)] == ORANGE)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        try:
            _validate_input(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for task 5")

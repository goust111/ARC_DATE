from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAGENTA = 6
INPUT_COLORS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
MIN_REGION_SIDE = 4
GRID_SIDE_MIN = 10
GRID_SIDE_MAX = 24
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    nbrs: list[tuple[int, int]] = []
    if r > 0:
        nbrs.append((r - 1, c))
    if r + 1 < h:
        nbrs.append((r + 1, c))
    if c > 0:
        nbrs.append((r, c - 1))
    if c + 1 < w:
        nbrs.append((r, c + 1))
    return nbrs


def _isolated_pixels(g: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = g.shape
    polluted: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            neighbor_colors = [int(g[nr, nc]) for nr, nc in _neighbors4(r, c, h, w)]
            if not neighbor_colors:
                continue
            same_count = sum(neighbor_color == color for neighbor_color in neighbor_colors)
            if same_count != 0:
                continue
            counts = Counter(neighbor_colors)
            top_count = max(counts.values())
            winners = [neighbor for neighbor, count in counts.items() if count == top_count]
            assert len(winners) == 1, "pollution pixel must have a unique majority neighbor color"
            polluted.append((r, c, int(winners[0])))
    return polluted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect isolated corrupted pixels inside large color blocks, restore their region color, and mark the source locations in magenta.

    Input structure:
    - The whole grid is tiled by 3-4 large rectangular color regions with no dedicated background color.
    - Inside one or two regions, a single interior pixel has been replaced by a color taken from another region.
    - Each corrupted pixel is locally inconsistent: its 4-neighbors belong to the surrounding region instead of to the pixel's own color.

    Rule:
    - For each cell, count how many 4-neighbors share the cell's color.
    - If that count is zero, treat the cell as a corruption source.
    - Recolor each corruption source to the unique majority color among its 4-neighbors, then place a magenta single-pixel marker at that same location.

    Key insight:
    - Corruption is identified by local 4-neighbor disagreement rather than by global color frequency, so an isolated legal color still stands out when all adjacent cells disagree with it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    polluted = _isolated_pixels(g)
    out = g.copy()
    for r, c, repaired_color in polluted:
        out[r, c] = int(repaired_color)
    for r, c, _repaired_color in polluted:
        out[r, c] = MAGENTA
    return out


def _sample_sizes(total: int, parts: int, min_size: int) -> list[int]:
    assert total >= parts * min_size
    sizes: list[int] = []
    remaining = total
    for index in range(parts):
        slots_left = parts - index - 1
        low = min_size
        high = remaining - slots_left * min_size
        size = random.randint(low, high)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _vertical_stripes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]]:
    widths = _sample_sizes(w, count, MIN_REGION_SIDE)
    regions: list[tuple[int, int, int, int]] = []
    c0 = 0
    for width in widths:
        regions.append((0, c0, h - 1, c0 + width - 1))
        c0 += width
    return regions


def _horizontal_stripes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]]:
    heights = _sample_sizes(h, count, MIN_REGION_SIDE)
    regions: list[tuple[int, int, int, int]] = []
    r0 = 0
    for height in heights:
        regions.append((r0, 0, r0 + height - 1, w - 1))
        r0 += height
    return regions


def _three_region_t_split(h: int, w: int) -> list[tuple[int, int, int, int]]:
    if random.choice([True, False]):
        left_w, right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
        top_h, bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
        if random.choice([True, False]):
            return [
                (0, 0, top_h - 1, left_w - 1),
                (top_h, 0, h - 1, left_w - 1),
                (0, left_w, h - 1, w - 1),
            ]
        return [
            (0, 0, h - 1, left_w - 1),
            (0, left_w, top_h - 1, w - 1),
            (top_h, left_w, h - 1, w - 1),
        ]

    top_h, bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
    left_w, right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
    if random.choice([True, False]):
        return [
            (0, 0, top_h - 1, left_w - 1),
            (0, left_w, top_h - 1, w - 1),
            (top_h, 0, h - 1, w - 1),
        ]
    return [
        (0, 0, top_h - 1, w - 1),
        (top_h, 0, h - 1, left_w - 1),
        (top_h, left_w, h - 1, w - 1),
    ]


def _four_region_grid(h: int, w: int) -> list[tuple[int, int, int, int]]:
    top_h, bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
    left_w, right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
    return [
        (0, 0, top_h - 1, left_w - 1),
        (0, left_w, top_h - 1, w - 1),
        (top_h, 0, h - 1, left_w - 1),
        (top_h, left_w, h - 1, w - 1),
    ]


def _four_region_offset_split(h: int, w: int) -> list[tuple[int, int, int, int]]:
    if random.choice([True, False]):
        left_w, right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
        left_top_h, left_bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
        right_top_h, right_bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
        return [
            (0, 0, left_top_h - 1, left_w - 1),
            (left_top_h, 0, h - 1, left_w - 1),
            (0, left_w, right_top_h - 1, w - 1),
            (right_top_h, left_w, h - 1, w - 1),
        ]

    top_h, bottom_h = _sample_sizes(h, 2, MIN_REGION_SIDE)
    top_left_w, top_right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
    bottom_left_w, bottom_right_w = _sample_sizes(w, 2, MIN_REGION_SIDE)
    return [
        (0, 0, top_h - 1, top_left_w - 1),
        (0, top_left_w, top_h - 1, w - 1),
        (top_h, 0, h - 1, bottom_left_w - 1),
        (top_h, bottom_left_w, h - 1, w - 1),
    ]


def _build_regions(region_count: int) -> list[tuple[int, int, int, int]] | None:
    if region_count == 3:
        layouts = [
            ("vertical", GRID_SIDE_MIN, 3 * MIN_REGION_SIDE),
            ("horizontal", 3 * MIN_REGION_SIDE, GRID_SIDE_MIN),
            ("t_split", 2 * MIN_REGION_SIDE, 2 * MIN_REGION_SIDE),
        ]
    else:
        layouts = [
            ("grid", 2 * MIN_REGION_SIDE, 2 * MIN_REGION_SIDE),
            ("vertical", GRID_SIDE_MIN, 4 * MIN_REGION_SIDE),
            ("horizontal", 4 * MIN_REGION_SIDE, GRID_SIDE_MIN),
            ("offset", 2 * MIN_REGION_SIDE, 2 * MIN_REGION_SIDE),
        ]

    layout_name, min_h, min_w = random.choice(layouts)
    h = random.randint(max(min_h, GRID_SIDE_MIN), GRID_SIDE_MAX)
    w = random.randint(max(min_w, GRID_SIDE_MIN), GRID_SIDE_MAX)

    if layout_name == "vertical":
        return _vertical_stripes(h, w, region_count)
    if layout_name == "horizontal":
        return _horizontal_stripes(h, w, region_count)
    if layout_name == "t_split":
        return _three_region_t_split(h, w)
    if layout_name == "grid":
        return _four_region_grid(h, w)
    if layout_name == "offset":
        return _four_region_offset_split(h, w)
    return None


def _fill_regions(regions: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    h = max(region[2] for region in regions) + 1
    w = max(region[3] for region in regions) + 1
    grid = np.empty((h, w), dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, regions):
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _inject_pollution(
    clean: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    colors: list[int],
    pollution_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = clean.copy()
    expected = clean.copy()
    host_indices = random.sample(range(len(regions)), pollution_count)
    for host_index in host_indices:
        r0, c0, r1, c1 = regions[host_index]
        row = random.randint(r0 + 1, r1 - 1)
        col = random.randint(c0 + 1, c1 - 1)
        foreign_choices = [color for idx, color in enumerate(colors) if idx != host_index]
        inp[row, col] = int(random.choice(foreign_choices))
        expected[row, col] = MAGENTA
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice([3, 4])
    pollution_count = random.choice([1, 2])
    if pollution_count > region_count:
        return None

    regions = _build_regions(region_count)
    if regions is None:
        return None

    colors = random.sample(INPUT_COLORS, region_count)
    clean = _fill_regions(regions, colors)
    inp, expected = _inject_pollution(clean, regions, colors, pollution_count)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_isolated_pixels(inp)) != pollution_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
RED = 2
GRAY = 5
DATA_COLORS = (1, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _prefix_run_length(row: np.ndarray, color: int) -> int:
    length = 0
    for value in row:
        if int(value) != color:
            break
        length += 1
    return length


def _split_regions(grid: np.ndarray) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    start = 0
    for r in range(grid.shape[0] + 1):
        if r == grid.shape[0] or np.all(grid[r] == GRAY):
            if start < r:
                regions.append((start, r))
            start = r + 1
    assert len(regions) >= 2, "expected at least one data region and one legend region"
    heights = {r1 - r0 for r0, r1 in regions}
    assert len(heights) == 1, "all non-separator regions must have the same height"
    return regions


def _bar_rows(region: np.ndarray) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for local_r in range(region.shape[0]):
        length = _prefix_run_length(region[local_r], RED)
        if 2 <= length <= 5:
            rows.append((local_r, length))
    return rows


def _parse_legend_row(row: np.ndarray) -> Tuple[int, Dict[int, int]]:
    length = _prefix_run_length(row, RED)
    assert 2 <= length <= 5, "legend rows must start with a 2..5 pixel red bar"

    payload = [int(value) for value in row[length:] if int(value) != BG]
    assert len(payload) >= 4 and len(payload) % 2 == 0, "legend rows need color pairs"

    mapping: Dict[int, int] = {}
    for i in range(0, len(payload), 2):
        src = int(payload[i])
        dst = int(payload[i + 1])
        assert src not in (BG, RED, GRAY)
        assert dst not in (BG, RED, GRAY)
        assert src not in mapping, "each source color should appear once per legend row"
        mapping[src] = dst
    return length, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each row region by looking up the color map keyed by its red ruler length.

    Input structure:
    - The background is black.
    - Full-width grey horizontal lines split the grid into equal-height row regions.
    - Each non-legend row region contains exactly one red left-edge ruler bar of length 2 to 5.
    - One top or bottom region is a legend: several rows inside it start with red bars of different lengths,
      followed by adjacent source→target color pairs.

    Rule:
    - Read the ruler length of each ordinary row region.
    - In the legend region, find the legend row with the same red-bar length and read its color pairs as a mapping.
    - Recolor every non-black, non-red, non-grey cell in that row region according to the mapping, while leaving
      unmapped colors unchanged.
    - Keep the red ruler bars, grey separator lines, and the legend region itself unchanged.

    Key insight:
    - The red bar length is a visible lookup key that selects one row-specific recoloring table from the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _split_regions(g)
    out = g.copy()

    region_bars = [_bar_rows(g[r0:r1]) for r0, r1 in regions]
    legend_candidates = [i for i, rows in enumerate(region_bars) if len(rows) >= 2]
    assert len(legend_candidates) == 1, "expected exactly one legend region"
    legend_index = int(legend_candidates[0])

    legend_region = g[regions[legend_index][0] : regions[legend_index][1]]
    legend_maps: Dict[int, Dict[int, int]] = {}
    for local_r, _ in region_bars[legend_index]:
        length, mapping = _parse_legend_row(legend_region[local_r])
        assert length not in legend_maps, "legend bar lengths must be unique"
        legend_maps[length] = mapping

    for idx, (r0, r1) in enumerate(regions):
        if idx == legend_index:
            continue
        rows = region_bars[idx]
        assert len(rows) == 1, "ordinary regions must contain exactly one ruler bar"
        bar_length = int(rows[0][1])
        assert bar_length in legend_maps, "every row ruler length must exist in the legend"

        original = g[r0:r1]
        recolored = original.copy()
        for src, dst in legend_maps[bar_length].items():
            recolored[original == src] = dst
        out[r0:r1] = recolored

    return out


def _make_mapping(pair_count: int) -> Dict[int, int]:
    srcs = random.sample(DATA_COLORS, pair_count)
    mapping: Dict[int, int] = {}
    used_dsts = set()
    for src in srcs:
        choices = [color for color in DATA_COLORS if color != src and color not in used_dsts]
        if not choices:
            choices = [color for color in DATA_COLORS if color != src]
        dst = random.choice(choices)
        mapping[int(src)] = int(dst)
        used_dsts.add(int(dst))
    return mapping


def _legend_region(height: int, width: int, mappings: Dict[int, Dict[int, int]]) -> np.ndarray:
    region = np.full((height, width), BG, dtype=int)
    rows = random.sample(range(height), len(mappings))
    entries = list(mappings.items())
    random.shuffle(entries)
    for local_r, (bar_length, mapping) in zip(rows, entries):
        region[local_r, :bar_length] = RED
        c = bar_length
        items = list(mapping.items())
        random.shuffle(items)
        for i, (src, dst) in enumerate(items):
            region[local_r, c] = int(src)
            region[local_r, c + 1] = int(dst)
            c += 2
            if i != len(items) - 1:
                c += 1
    return region


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, min_col: int) -> bool:
    hh, ww = occupied.shape
    if c < min_col or r < 0 or r + h > hh or c + w > ww:
        return False
    r0 = max(0, r - 1)
    r1 = min(hh, r + h + 1)
    c0 = max(0, c - 1)
    c1 = min(ww, c + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paint_block(region: np.ndarray, occupied: np.ndarray, color: int, min_col: int) -> bool:
    height, width = region.shape
    shapes: Sequence[Tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2))
    for _ in range(120):
        h, w = random.choice(shapes)
        if h > height:
            continue
        r = random.randint(0, height - h)
        c = random.randint(min_col, width - w)
        if not _can_place(occupied, r, c, h, w, min_col):
            continue
        region[r : r + h, c : c + w] = int(color)
        occupied[r : r + h, c : c + w] = True
        return True
    return False


def _data_region(height: int, width: int, bar_length: int, mapping: Dict[int, int]) -> np.ndarray | None:
    region = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    bar_row = random.randrange(height)
    region[bar_row, :bar_length] = RED
    occupied[bar_row, :bar_length] = True

    min_col = max(7, bar_length + 2)
    mapped_colors = list(mapping.keys())
    extra_colors = [color for color in DATA_COLORS if color not in mapping]
    block_count = random.randint(3, 4 if height == 3 else 5)

    block_colors = [random.choice(mapped_colors)]
    while len(block_colors) < block_count:
        use_extra = bool(extra_colors) and random.random() < 0.35
        if use_extra:
            block_colors.append(random.choice(extra_colors))
        else:
            block_colors.append(random.choice(mapped_colors))
    random.shuffle(block_colors)

    for color in block_colors:
        if not _paint_block(region, occupied, int(color), min_col):
            return None

    return region


def _assemble_grid(regions: Sequence[np.ndarray]) -> np.ndarray:
    width = regions[0].shape[1]
    separator = np.full((1, width), GRAY, dtype=int)
    rows: List[np.ndarray] = []
    for i, region in enumerate(regions):
        if i:
            rows.append(separator)
        rows.append(region)
    return np.vstack(rows)


def _gen_attempt() -> np.ndarray | None:
    region_height = random.randint(3, 5)
    max_total_regions = 31 // (region_height + 1)
    max_data_regions = max_total_regions - 1

    min_mapping_count = 2
    max_mapping_count = min(4, region_height, max_data_regions)
    if max_mapping_count < min_mapping_count:
        return None
    mapping_count = random.randint(min_mapping_count, max_mapping_count)

    max_data_regions = min(5, max_data_regions)
    data_region_count = random.randint(mapping_count, max_data_regions)

    bar_lengths = random.sample((2, 3, 4, 5), mapping_count)
    mappings = {int(length): _make_mapping(random.randint(2, 4)) for length in bar_lengths}

    legend_width = max(length + 3 * len(mapping) - 1 for length, mapping in mappings.items())
    width_min = max(16, legend_width + 2)
    width = random.randint(width_min, 28)

    legend = _legend_region(region_height, width, mappings)

    data_lengths = list(bar_lengths)
    while len(data_lengths) < data_region_count:
        data_lengths.append(random.choice(bar_lengths))
    random.shuffle(data_lengths)

    data_regions: List[np.ndarray] = []
    for length in data_lengths:
        region = _data_region(region_height, width, int(length), mappings[int(length)])
        if region is None:
            return None
        data_regions.append(region)

    if random.random() < 0.5:
        regions = [legend] + data_regions
    else:
        regions = data_regions + [legend]

    grid = _assemble_grid(regions)
    if max(grid.shape) > 30:
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-legend recoloring puzzle")

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIVIDER = 4
PALETTE = (1, 2, 3, 5, 6)
GENERATION_RETRIES = 500


def _full_divider_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == DIVIDER))]


def _full_divider_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == DIVIDER))]


def _spans(length: int, dividers: Iterable[int]) -> list[tuple[int, int]]:
    marks = [-1, *sorted(int(x) for x in dividers), int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(marks, marks[1:]):
        if right - left > 1:
            spans.append((int(left + 1), int(right - 1)))
    return spans


def _build_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _region_boxes(grid: np.ndarray) -> list[dict[str, int]]:
    rows = _full_divider_rows(grid)
    cols = _full_divider_cols(grid)
    assert rows and cols, "expected at least one horizontal and one vertical divider"

    row_spans = _spans(grid.shape[0], rows)
    col_spans = _spans(grid.shape[1], cols)
    regions: list[dict[str, int]] = []
    for band_r, (r0, r1) in enumerate(row_spans):
        for band_c, (c0, c1) in enumerate(col_spans):
            regions.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "band_r": int(band_r),
                    "band_c": int(band_c),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    return regions


def _extract_regions(grid: np.ndarray) -> list[dict[str, int]]:
    regions: list[dict[str, int]] = []
    for region in _region_boxes(grid):
        r0, c0, r1, c1 = (int(region["r0"]), int(region["c0"]), int(region["r1"]), int(region["c1"]))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        fg_mask = (patch != BG) & (patch != DIVIDER)
        colors = np.unique(patch[fg_mask])
        assert colors.size == 1, "each region must contain exactly one foreground color"
        count = int(fg_mask.sum())
        assert count > 0, "every region must contain at least one foreground pixel"
        regions.append(
            {
                **region,
                "color": int(colors[0]),
                "count": int(count),
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count region-local scatter pixels and convert the colors into nested square rings.

    Input structure:
    - The background is black.
    - Yellow full rows and columns act as divider lines and split the canvas into rectangular regions.
    - Each region contains scattered pixels of exactly one non-black, non-yellow foreground color.

    Rule:
    - Detect every full yellow divider row and column, then recover the rectangular regions between them.
    - In each region, count the non-black, non-yellow pixels and record that region's single foreground color.
    - Sort the region colors by descending count and draw them as concentric 1-cell-thick square rings.
    - Use side length 2N-1 where N is the number of regions; the smallest count becomes the center pixel.

    Key insight:
    - The divider lines matter only because they define the counting boundaries; ring order depends on per-region scatter counts, not on region size, position, or the original point pattern.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)

    counts = [int(region["count"]) for region in regions]
    assert len(counts) == len(set(counts)), "region counts must be unique"

    ordered = sorted(regions, key=lambda region: int(region["count"]), reverse=True)
    return _build_rings([int(region["color"]) for region in ordered])


def _interior_cells(region: dict[str, int]) -> list[tuple[int, int]]:
    rows = range(int(region["r0"]) + 1, int(region["r1"]))
    cols = range(int(region["c0"]) + 1, int(region["c1"]))
    return [(int(r), int(c)) for r in rows for c in cols]


def _sample_count_values(max_count: int, n_regions: int) -> list[int] | None:
    pool = list(range(2, int(max_count) + 1))
    if len(pool) < n_regions:
        return None

    for _ in range(80):
        vals = sorted(random.sample(pool, n_regions))
        if min(b - a for a, b in zip(vals, vals[1:])) >= 1:
            return vals
    return None


def _rank_signature(regions: list[dict[str, int]], key: str, reverse: bool) -> list[tuple[int, int]]:
    ordered = sorted(
        regions,
        key=lambda region: (int(region[key]), int(region["band_r"]), int(region["band_c"])),
        reverse=reverse,
    )
    return [(int(region["band_r"]), int(region["band_c"])) for region in ordered]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        regions = _extract_regions(grid)
        out = main(grid)
    except AssertionError:
        return False

    if len(regions) != 4:
        return False

    colors = [int(region["color"]) for region in regions]
    counts = [int(region["count"]) for region in regions]
    if len(colors) != len(set(colors)) or len(counts) != len(set(counts)):
        return False

    for region in regions:
        patch = grid[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1]
        if np.any((patch[0, :] != BG) & (patch[0, :] != DIVIDER)):
            return False
        if np.any((patch[-1, :] != BG) & (patch[-1, :] != DIVIDER)):
            return False
        if np.any((patch[:, 0] != BG) & (patch[:, 0] != DIVIDER)):
            return False
        if np.any((patch[:, -1] != BG) & (patch[:, -1] != DIVIDER)):
            return False

    count_sig = _rank_signature(regions, "count", True)
    area_sig = _rank_signature(regions, "area", True)
    raster_sig = [(int(region["band_r"]), int(region["band_c"])) for region in regions]
    if count_sig == area_sig:
        return False
    if count_sig == raster_sig:
        return False

    ordered_colors = [int(region["color"]) for region in sorted(regions, key=lambda region: int(region["count"]), reverse=True)]
    if ordered_colors == sorted(ordered_colors) or ordered_colors == sorted(ordered_colors, reverse=True):
        return False

    if np.array_equal(out, _build_rings([int(region["color"]) for region in sorted(regions, key=lambda region: int(region["area"]), reverse=True)])):
        return False

    if np.unique(out).size != len(regions):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(5, 8)
    bottom_h = random.randint(5, 8)
    left_w = random.randint(5, 8)
    right_w = random.randint(5, 8)
    height = top_h + bottom_h + 1
    width = left_w + right_w + 1

    divider_r = top_h
    divider_c = left_w
    grid = np.full((height, width), BG, dtype=int)
    grid[divider_r, :] = DIVIDER
    grid[:, divider_c] = DIVIDER

    regions = _region_boxes(grid)
    if len(regions) != 4:
        return None

    capacities = [len(_interior_cells(region)) for region in regions]
    if min(capacities) < 5:
        return None

    count_values = _sample_count_values(min(min(capacities), 12), len(regions))
    if count_values is None:
        return None
    random.shuffle(count_values)

    colors = random.sample(PALETTE, len(regions))
    random.shuffle(regions)
    for region, color, count in zip(regions, colors, count_values):
        cells = _interior_cells(region)
        if len(cells) < int(count):
            return None
        chosen = random.sample(cells, int(count))
        for r, c in chosen:
            grid[r, c] = int(color)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

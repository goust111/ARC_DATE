from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


def _sorted_regions(grid: np.ndarray) -> list[dict]:
    regions = find_components(
        grid,
        bg=Color.RED,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    return sorted(regions, key=lambda region: (int(region["bbox"][1]), int(region["bbox"][0])))


def _point_in_region(region: dict, point: tuple[int, int]) -> bool:
    r, c = map(int, point)
    coords = region["coords"]
    return bool(np.any((coords[:, 0] == r) & (coords[:, 1] == c)))


def _region_index_for_point(regions: list[dict], point: tuple[int, int]) -> int:
    hits = [idx for idx, region in enumerate(regions) if _point_in_region(region, point)]
    assert len(hits) == 1, "marker must belong to exactly one red-separated region"
    return int(hits[0])


def _parse_ruler(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    rulers = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    assert len(rulers) == 1, "expected exactly one grey ruler bar"
    ruler = rulers[0]
    r0, c0, r1, c1 = map(int, ruler["bbox"])
    assert r0 == r1 and c0 < c1, "ruler must be a horizontal bar"
    assert int(ruler["area"]) == c1 - c0 + 1, "ruler must be a solid 1-cell-thick segment"
    return int(c1 - c0 + 1), (r0, c0, r1, c1)


def _apply_period_fill(grid: np.ndarray, marker_region_index: int, period: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    regions = _sorted_regions(out)
    for idx, region in enumerate(regions):
        if idx % period != marker_region_index % period:
            continue
        coords = region["coords"]
        black_mask = out[coords[:, 0], coords[:, 1]] == Color.BLACK
        fill_coords = coords[black_mask]
        if fill_coords.size:
            out[fill_coords[:, 0], fill_coords[:, 1]] = Color.YELLOW
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the grey ruler length as a period and the yellow marker region as the phase to fill congruent divider-separated regions.

    Input structure:
    - A black canvas is partitioned into several connected non-red regions by one continuous red polyline.
    - Exactly one region contains a horizontal grey ruler bar, and a different region contains a single yellow marker pixel.
    - The regions are ordered from left to right by their bounding boxes.

    Rule:
    - Let k be the length of the grey ruler bar and let s be the index of the region containing the yellow marker.
    - Fill every black cell in each region whose index is congruent to s modulo k with yellow.
    - Keep the red divider, the grey ruler, and any already-yellow marker cell unchanged.

    Key insight:
    - The ruler supplies the repeating step size while the marker supplies the starting residue class; both clues are needed to choose the filled regions.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    period, _ = _parse_ruler(grid)
    assert period >= 2, "ruler length must define a nontrivial period"

    marker_coords = np.argwhere(grid == Color.YELLOW)
    assert marker_coords.shape[0] == 1, "expected exactly one yellow marker pixel"
    marker = tuple(map(int, marker_coords[0]))

    regions = _sorted_regions(grid)
    assert len(regions) > period, "period should not exceed the number of ordered regions"
    marker_region_index = _region_index_for_point(regions, marker)

    return _apply_period_fill(grid, marker_region_index, period)


def _random_composition(total: int, parts: int) -> list[int]:
    pieces = [0] * parts
    for _ in range(total):
        pieces[random.randrange(parts)] += 1
    return pieces


def _sample_region_widths(region_count: int) -> list[int]:
    min_width = 4
    min_total = min_width * region_count
    max_total = 30 - (region_count - 1)
    total = random.randint(min_total, max_total)
    extras = _random_composition(total - min_total, region_count)
    return [min_width + extra for extra in extras]


def _region_spans(widths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    barriers: list[int] = []
    start = 0
    for idx, width in enumerate(widths):
        end = start + width - 1
        spans.append((start, end))
        start = end + 1
        if idx < len(widths) - 1:
            barriers.append(start)
            start += 1
    return spans, barriers, start


def _build_divider_grid(height: int, widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    spans, barriers, total_width = _region_spans(widths)
    grid = np.full((height, total_width), Color.BLACK, dtype=int)

    for idx, col in enumerate(barriers):
        grid[:, col] = Color.RED
        if idx + 1 < len(barriers):
            connector_row = height - 1 if idx % 2 == 0 else 0
            next_col = barriers[idx + 1]
            lo, hi = sorted((col, next_col))
            grid[connector_row, lo : hi + 1] = Color.RED

    return grid, spans


def _class_members(region_count: int, period: int, marker_region_index: int) -> list[int]:
    return [idx for idx in range(region_count) if idx % period == marker_region_index % period]


def _expected_output(grid: np.ndarray, marker_region_index: int, period: int) -> np.ndarray:
    return _apply_period_fill(grid, marker_region_index, period)


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(5, 6)
    widths = _sample_region_widths(region_count)
    feasible_periods = [
        period
        for period in range(2, min(5, region_count - 1) + 1)
        if any(width >= period + 2 for width in widths)
    ]
    if not feasible_periods:
        return None

    marker_period_pairs = []
    for period in feasible_periods:
        for marker_region_index in range(region_count):
            members = _class_members(region_count, period, marker_region_index)
            if 2 <= len(members) <= region_count - 2:
                marker_period_pairs.append((marker_region_index, period))
    if not marker_period_pairs:
        return None

    marker_region_index, period = random.choice(marker_period_pairs)
    ruler_choices = [
        idx
        for idx, width in enumerate(widths)
        if idx != marker_region_index and width >= period + 2
    ]
    if not ruler_choices:
        return None
    ruler_region_index = random.choice(ruler_choices)

    height = random.randint(7, 13)
    grid, spans = _build_divider_grid(height, widths)

    ruler_left, ruler_right = spans[ruler_region_index]
    ruler_row = random.randint(1, height - 2)
    ruler_start = random.randint(ruler_left + 1, ruler_right - period)
    grid[ruler_row, ruler_start : ruler_start + period] = Color.GREY

    marker_left, marker_right = spans[marker_region_index]
    marker_row = random.randint(1, height - 2)
    marker_col = random.randint(marker_left, marker_right)
    grid[marker_row, marker_col] = Color.YELLOW

    if random.random() < 0.5:
        grid = np.flipud(grid).copy()

    expected = _expected_output(grid, marker_region_index, period)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    regions = _sorted_regions(grid)
    if len(regions) != region_count:
        return None
    if np.count_nonzero(actual == Color.YELLOW) <= 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

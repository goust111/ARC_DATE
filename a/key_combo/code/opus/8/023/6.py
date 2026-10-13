from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


GEN_HEIGHT_RANGE = (12, 18)
GEN_WIDTH_RANGE = (12, 18)
GEN_CONTROL_COUNTS = (2, 3, 4)
GEN_MIN_GAPS = (2, 3)
GEN_RETRIES = 400
GEN_POSITION_RETRIES = 200


def _extract_controls(grid: np.ndarray) -> tuple[list[int], list[int]]:
    top_cols = [int(c) for c in np.flatnonzero(grid[0] == Color.BLUE) if int(c) != 0]
    left_rows = [int(r) for r in np.flatnonzero(grid[:, 0] == Color.RED) if int(r) != 0]
    return top_cols, left_rows


def _wall_grid(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    top_cols, left_rows = _extract_controls(out)

    for c in top_cols:
        out[1:, c] = Color.BLUE
    for r in left_rows:
        out[r, 1:] = Color.RED
    if top_cols and left_rows:
        out[np.ix_(left_rows, top_cols)] = Color.GREEN

    out[0, 0] = int(grid[0, 0])
    for c in top_cols:
        out[0, c] = Color.BLUE
    for r in left_rows:
        out[r, 0] = Color.RED
    return out


def _sorted_black_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in components_mask(grid == Color.BLACK, connectivity=4):
        box = bbox_coords(coords)
        regions.append({"coords": coords, "bbox": box, "area": int(coords.shape[0])})
    regions.sort(key=lambda region: (region["bbox"][0], region["bbox"][1]))
    return regions


def _paint_selected_regions(base: np.ndarray, regions: list[dict], *, offset: int = 0) -> np.ndarray:
    out = np.array(base, copy=True)
    for i, region in enumerate(regions):
        if i % 2 == offset:
            coords = region["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.YELLOW
    return out


def _paint_area_sorted_regions(base: np.ndarray, regions: list[dict]) -> np.ndarray:
    reordered = sorted(regions, key=lambda region: (region["area"], *region["bbox"]))
    return _paint_selected_regions(base, reordered, offset=0)


def _paint_interior_regions_only(base: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = np.array(base, copy=True)
    for i, region in enumerate(regions):
        if i % 2 != 0:
            continue
        r0, c0, _r1, _c1 = region["bbox"]
        if r0 == 0 or c0 == 0:
            continue
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = Color.YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Build crossing ray walls from edge control pixels, then fill every other enclosed background region yellow.

    Input structure:
    - The grid is black background with blue control pixels on the top edge and red control pixels on the left edge.
    - The top-left corner stays black and all interior cells start as background.
    - Blue controls mark columns for vertical rays; red controls mark rows for horizontal rays.

    Rule:
    - Extend each top-edge blue control downward from row 1 to the bottom edge in blue.
    - Extend each left-edge red control rightward from column 1 to the right edge in red.
    - Recolor every red/blue crossing cell green, and treat all blue/red/green ray cells as walls.
    - Find all 4-connected black regions enclosed by those walls together with the grid boundary.
    - Sort the black regions by the top-left corner of their bounding boxes, top-to-bottom then left-to-right.
    - Flood-fill regions with even indices in that order with yellow; keep odd-indexed regions black.
    - Preserve the original top-edge and left-edge control pixels.

    Key insight:
    - The edge controls do not directly indicate the final yellow cells; they first generate a wall grid whose region ordering determines the periodic fill pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(g[0, 0]) == Color.BLACK
    assert np.all(np.isin(g[0, :], [Color.BLACK, Color.BLUE]))
    assert np.all(np.isin(g[:, 0], [Color.BLACK, Color.RED]))
    assert np.all(g[1:, 1:] == Color.BLACK)

    top_cols, left_rows = _extract_controls(g)
    assert 2 <= len(top_cols) <= 4
    assert 2 <= len(left_rows) <= 4

    base = _wall_grid(g)
    regions = _sorted_black_regions(base)
    assert regions
    return _paint_selected_regions(base, regions, offset=0)


def _sample_spaced_positions(size: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(1, size - 1))
    for _ in range(GEN_POSITION_RETRIES):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _region_count_formula(top_cols: list[int], left_rows: list[int]) -> int:
    return (len(top_cols) + 1) * (len(left_rows) + 1)


def _semantic_checks(grid: np.ndarray) -> bool:
    top_cols, left_rows = _extract_controls(grid)
    if not (2 <= len(top_cols) <= 4 and 2 <= len(left_rows) <= 4):
        return False
    if np.any(grid[1:, 1:] != Color.BLACK):
        return False
    if np.any(grid[0, 1:] == Color.RED) or np.any(grid[1:, 0] == Color.BLUE):
        return False

    base = _wall_grid(grid)
    if int(np.count_nonzero(base == Color.GREEN)) != len(top_cols) * len(left_rows):
        return False

    regions = _sorted_black_regions(base)
    if len(regions) != _region_count_formula(top_cols, left_rows):
        return False
    if len(regions) < 9:
        return False

    areas = {region["area"] for region in regions}
    if len(areas) < 3:
        return False

    out = main(grid)
    expected = _paint_selected_regions(base, regions, offset=0)
    if not np.array_equal(out, expected):
        return False

    odd_version = _paint_selected_regions(base, regions, offset=1)
    if np.array_equal(out, odd_version):
        return False

    area_sorted_version = _paint_area_sorted_regions(base, regions)
    if np.array_equal(out, area_sorted_version):
        return False

    interior_only_version = _paint_interior_regions_only(base, regions)
    if np.array_equal(out, interior_only_version):
        return False

    yellow_regions = len(regions[::2])
    if int(np.count_nonzero(out == Color.YELLOW)) != sum(region["area"] for region in regions[::2]):
        return False
    if yellow_regions < 4 or len(regions) - yellow_regions < 4:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    n_top = random.choice(GEN_CONTROL_COUNTS)
    n_left = random.choice(GEN_CONTROL_COUNTS)
    top_gap = random.choice(GEN_MIN_GAPS)
    left_gap = random.choice(GEN_MIN_GAPS)

    top_cols = _sample_spaced_positions(width, n_top, top_gap)
    left_rows = _sample_spaced_positions(height, n_left, left_gap)
    if top_cols is None or left_rows is None:
        return None
    if len(top_cols) == len(left_rows) and set(top_cols) == set(left_rows):
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, top_cols] = Color.BLUE
    grid[left_rows, 0] = Color.RED

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

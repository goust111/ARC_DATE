from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


BACKGROUND = Color.BLACK
WALL = Color.ORANGE
RANK_COLORS = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY]

COUNT_CHOICES = [3, 4, 4, 5, 5]
WIDTH_CHOICES = [3, 4, 5, 6]
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _exterior_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BACKGROUND and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BACKGROUND and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BACKGROUND:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    exterior = _exterior_mask(grid)
    enclosed = (grid == BACKGROUND) & (~exterior)
    regions = []
    for coords in components_mask(enclosed, connectivity=4):
        regions.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox(coords),
            }
        )
    return regions


def _ranked_regions(grid: np.ndarray) -> list[dict]:
    regions = _enclosed_regions(grid)
    return sorted(regions, key=lambda reg: (-reg["area"], reg["bbox"][0], reg["bbox"][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill enclosed regions with colors determined by their area ranking.

    Input structure:
    - A black grid contains one 4-connected orange 1-pixel-wide wall object.
    - The wall is not a closed loop by itself, but together with the canvas border it partitions
      the black background into an exterior region and 3-5 enclosed regions.
    - The enclosed black regions are 4-connected and have pairwise distinct areas.

    Rule:
    - Flood-fill black cells from the grid border using 4-connectivity to mark the exterior.
    - Split the remaining black cells into enclosed 4-connected regions and measure each region's area.
    - Sort the enclosed regions by area from largest to smallest and recolor them by rank:
      largest red, second blue, third green, fourth yellow, fifth grey.
    - Preserve the orange wall and keep the exterior black.

    Key insight:
    - The wall only identifies which black cells are enclosed; the final colors come from comparing
      enclosed-region areas, so flood-filling and counting are both necessary before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    regions = _ranked_regions(g)
    assert len(regions) <= len(RANK_COLORS)
    for rank, region in enumerate(regions):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = RANK_COLORS[rank]
    return out


def _branch_rows(start: int, heights: list[int]) -> list[int]:
    rows = [start]
    for height in heights:
        rows.append(rows[-1] + height + 1)
    return rows


def _build_wall_case(
    height: int,
    left_width: int,
    right_width: int,
    left_rows: list[int],
    right_rows: list[int],
) -> np.ndarray:
    width = left_width + right_width + 3
    spine_col = left_width + 1
    grid = np.zeros((height, width), dtype=int)
    grid[:, spine_col] = WALL
    for row in left_rows:
        grid[row, : spine_col + 1] = WALL
    for row in right_rows:
        grid[row, spine_col:] = WALL
    grid[min(left_rows) : max(left_rows) + 1, 0] = WALL
    grid[min(right_rows) : max(right_rows) + 1, width - 1] = WALL
    return grid


def _line_is_valid(grid: np.ndarray) -> bool:
    mask = grid == WALL
    comps = components_mask(mask, connectivity=4)
    if len(comps) != 1:
        return False
    if np.any(
        mask[:-1, :-1].astype(int)
        + mask[1:, :-1].astype(int)
        + mask[:-1, 1:].astype(int)
        + mask[1:, 1:].astype(int)
        == 4
    ):
        return False

    h, w = grid.shape
    degrees = []
    border_touches = 0
    for r, c in np.argwhere(mask):
        if r in (0, h - 1) or c in (0, w - 1):
            border_touches += 1
        degree = int(mask[r - 1, c]) if r > 0 else 0
        degree += int(mask[r + 1, c]) if r + 1 < h else 0
        degree += int(mask[r, c - 1]) if c > 0 else 0
        degree += int(mask[r, c + 1]) if c + 1 < w else 0
        degrees.append(degree)
    endpoint_count = sum(degree == 1 for degree in degrees)
    branch_count = sum(degree >= 3 for degree in degrees)
    return border_touches >= 4 and endpoint_count >= 2 and branch_count >= 2


def _fill_by_order(grid: np.ndarray, regions: list[dict], order_key) -> np.ndarray:
    out = grid.copy()
    ordered = sorted(regions, key=order_key)
    for idx, region in enumerate(ordered):
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = RANK_COLORS[idx]
    return out


def _validate_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _line_is_valid(grid):
        return False

    regions = _enclosed_regions(grid)
    if not 3 <= len(regions) <= 5:
        return False

    areas = [reg["area"] for reg in regions]
    if len(set(areas)) != len(areas):
        return False
    if any(area < 3 or area > 20 for area in areas):
        return False

    out = main(grid)
    exterior = _exterior_mask(grid)
    if not np.array_equal(out[grid == WALL], grid[grid == WALL]):
        return False
    if np.any(out[exterior] != BACKGROUND):
        return False

    ranked = sorted(regions, key=lambda reg: (-reg["area"], reg["bbox"][0], reg["bbox"][1]))
    for idx, region in enumerate(ranked):
        coords = region["coords"]
        vals = set(map(int, out[coords[:, 0], coords[:, 1]]))
        if vals != {RANK_COLORS[idx]}:
            return False

    by_row_major = _fill_by_order(grid, regions, lambda reg: (reg["bbox"][0], reg["bbox"][1]))
    if np.array_equal(out, by_row_major):
        return False

    by_smallest_first = _fill_by_order(grid, regions, lambda reg: (reg["area"], reg["bbox"][0], reg["bbox"][1]))
    if np.array_equal(out, by_smallest_first):
        return False

    return True


def _sample_split(count: int) -> tuple[int, int]:
    candidates = [(a, count - a) for a in range(1, count) if max(a, count - a) <= 3]
    return random.choice(candidates)


def _sample_heights(left_width: int, right_width: int, left_count: int, right_count: int) -> tuple[list[int], list[int]] | None:
    left_options = [h for h in range(1, 8) if 3 <= h * left_width <= 20]
    right_options = [h for h in range(1, 8) if 3 <= h * right_width <= 20]
    if len(left_options) < left_count or len(right_options) < right_count:
        return None

    for _ in range(120):
        left_heights = random.sample(left_options, left_count)
        right_heights = random.sample(right_options, right_count)
        pairs = [(h * left_width, "L") for h in left_heights] + [(h * right_width, "R") for h in right_heights]
        areas = [area for area, _side in pairs]
        if len(set(areas)) != len(areas):
            continue
        top_sides = {side for _area, side in sorted(pairs, reverse=True)[: min(3, len(pairs))]}
        if len(top_sides) < 2:
            continue
        random.shuffle(left_heights)
        random.shuffle(right_heights)
        return left_heights, right_heights
    return None


def _sample_row_layout(left_heights: list[int], right_heights: list[int]) -> tuple[int, list[int], list[int]] | None:
    left_span = sum(height + 1 for height in left_heights)
    right_span = sum(height + 1 for height in right_heights)
    min_height = max(12, left_span + 3, right_span + 3)
    max_height = min(20, min_height + 6)
    if min_height > max_height:
        return None

    for _ in range(120):
        height = random.randint(min_height, max_height)
        left_start_max = height - 2 - left_span
        right_start_max = height - 2 - right_span
        if left_start_max < 1 or right_start_max < 1:
            continue
        left_rows = _branch_rows(random.randint(1, left_start_max), left_heights)
        right_rows = _branch_rows(random.randint(1, right_start_max), right_heights)
        if set(left_rows) & set(right_rows):
            continue
        return height, left_rows, right_rows
    return None


def _generate_candidate() -> np.ndarray | None:
    chamber_count = random.choice(COUNT_CHOICES)
    left_count, right_count = _sample_split(chamber_count)
    widths = random.sample(WIDTH_CHOICES, 2)
    if random.random() < 0.35:
        widths[1] = random.choice(WIDTH_CHOICES)
    left_width, right_width = widths

    sampled = _sample_heights(left_width, right_width, left_count, right_count)
    if sampled is None:
        return None
    left_heights, right_heights = sampled

    layout = _sample_row_layout(left_heights, right_heights)
    if layout is None:
        return None
    height, left_rows, right_rows = layout

    grid = _build_wall_case(height, left_width, right_width, left_rows, right_rows)
    grid = np.rot90(grid, k=random.randint(0, 3))
    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region ranking puzzle")

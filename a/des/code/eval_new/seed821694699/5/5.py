"""ARC puzzle 5 solver/generator.

This module provides `main` and `generate_input` with integer colors only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


BLACK = 0
GREY = 5
NEIGHBOR_COLOR_ORDER = [2, 1, 3, 4, 6, 7, 8, 9]  # Red, Blue, Green, Yellow, Magenta, Orange, Azure, Maroon


def _color_order(color: int) -> int:
    """Return ranking key used when neighbor areas tie.

    Lower value means higher priority.
    """
    try:
        return NEIGHBOR_COLOR_ORDER.index(color)
    except ValueError:
        return len(NEIGHBOR_COLOR_ORDER)


def _extract_regions(grid: np.ndarray) -> Tuple[List[Dict[str, int]], np.ndarray]:
    """Find connected non-gray, non-black regions.

    Returns:
      - list of region metadata dicts
      - map of region ids per cell (-1 for gray/black/non-region)
    """
    h, w = grid.shape
    region_map = -np.ones((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    regions: List[Dict[str, int]] = []
    region_id = 0

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            if grid[r, c] == GREY or grid[r, c] == BLACK:
                visited[r, c] = True
                continue

            q = [(r, c)]
            visited[r, c] = True
            min_r = max_r = r
            min_c = max_c = c
            area = 0
            color = int(grid[r, c])

            while q:
                cr, cc = q.pop()
                region_map[cr, cc] = region_id
                area += 1
                if cr < min_r:
                    min_r = cr
                if cr > max_r:
                    max_r = cr
                if cc < min_c:
                    min_c = cc
                if cc > max_c:
                    max_c = cc

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc]:
                        continue
                    if grid[nr, nc] == GREY or grid[nr, nc] == BLACK:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            regions.append(
                {
                    "id": region_id,
                    "color": color,
                    "area": area,
                    "min_r": min_r,
                    "max_r": max_r,
                    "min_c": min_c,
                    "max_c": max_c,
                }
            )
            region_id += 1

    return regions, region_map


def _collect_neighbors(
    grid: np.ndarray, regions: List[Dict[str, int]], region_map: np.ndarray
) -> List[Set[int]]:
    """Collect neighboring regions across one-or-more-cell grey separators."""
    h, w = grid.shape
    adjacency: List[Set[int]] = [set() for _ in regions]
    for r, c in zip(*np.nonzero(region_map >= 0)):
        rid = int(region_map[r, c])

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue

            # Direct touch (invalid in clean data, retained for robustness).
            neighbor_rid = int(region_map[nr, nc])
            if neighbor_rid >= 0 and neighbor_rid != rid:
                adjacency[rid].add(neighbor_rid)
                continue

            # Skip only if immediate non-region is not grey.
            if grid[nr, nc] != GREY:
                continue

            rr, cc = nr, nc
            while 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == GREY:
                rr += dr
                cc += dc

            if not (0 <= rr < h and 0 <= cc < w):
                continue
            neighbor_rid = int(region_map[rr, cc])
            if neighbor_rid >= 0 and neighbor_rid != rid:
                adjacency[rid].add(neighbor_rid)

    return adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """Solve concentric-border rule for ARC 5.

    Args:
        grid: 2D int numpy array with ARC color ids.

    Returns:
        np.ndarray with the same shape/dtype as input.

    Procedure:
        1) Extract non-black/non-grey connected regions.
        2) Find adjacent regions across grey strips.
        3) Sort neighbors by area descending, with color-order tiebreak.
        4) Paint concentric 1-pixel rectangles from outside inward.

    Complexity:
        O(H*W + R*L), where H/W are grid dims and R/L regions/layers.
    """
    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("grid must use integer ARC colors")

    out = grid.astype(np.int64, copy=True)
    regions, region_map = _extract_regions(grid)
    if not regions:
        return out

    adjacency = _collect_neighbors(grid, regions, region_map)

    for region in regions:
        rid = region["id"]
        neighbors = adjacency[rid]
        if not neighbors:
            continue

        neighbor_ids = sorted(
            neighbors,
            key=lambda nid: (-regions[nid]["area"], _color_order(regions[nid]["color"])),
        )
        colors = [regions[nid]["color"] for nid in neighbor_ids]

        min_r = region["min_r"]
        max_r = region["max_r"]
        min_c = region["min_c"]
        max_c = region["max_c"]

        for layer, color in enumerate(colors):
            top = min_r + layer
            left = min_c + layer
            bottom = max_r - layer
            right = max_c - layer

            if top >= bottom or left >= right:
                break

            out[top, left : right + 1] = color
            out[bottom, left : right + 1] = color
            out[top : bottom + 1, left] = color
            out[top : bottom + 1, right] = color

    return out


def generate_input() -> np.ndarray:
    """Create a random ARC-like input under 30x30 with grey separators and rectangles."""
    rng = np.random.default_rng()
    palette = np.array([2, 1, 3, 4, 6, 7, 8, 9], dtype=np.int64)

    while True:
        region_rows = int(rng.integers(2, 6))
        region_cols = int(rng.integers(2, 6))
        row_sizes = rng.integers(2, 7, size=region_rows).astype(int)
        col_sizes = rng.integers(2, 7, size=region_cols).astype(int)

        h = int(row_sizes.sum() + max(0, region_rows - 1))
        w = int(col_sizes.sum() + max(0, region_cols - 1))
        if 1 <= h <= 30 and 1 <= w <= 30 and h >= 6 and w >= 6:
            break

    grid = np.full((h, w), BLACK, dtype=np.int64)

    y_starts = []
    cursor = 0
    for size in row_sizes:
        y_starts.append(cursor)
        cursor += size + 1
    x_starts = []
    cursor = 0
    for size in col_sizes:
        x_starts.append(cursor)
        cursor += size + 1

    # Fill rectangular color regions.
    for i, y0 in enumerate(y_starts):
        for j, x0 in enumerate(x_starts):
            y1 = y0 + row_sizes[i]
            x1 = x0 + col_sizes[j]
            color = int(rng.choice(palette))
            grid[y0:y1, x0:x1] = color

    # Draw grey separators.
    for i in range(1, region_rows):
        row = int(row_sizes[:i].sum() + (i - 1))
        grid[row, :] = GREY
    for j in range(1, region_cols):
        col = int(col_sizes[:j].sum() + (j - 1))
        grid[:, col] = GREY

    # Ensure enough color diversity.
    unique_colors = np.unique(grid[(grid != BLACK) & (grid != GREY)])
    if unique_colors.size < 2:
        grid[0, 0] = 2 if unique_colors[0] != 2 else 1

    return grid.astype(np.int64)


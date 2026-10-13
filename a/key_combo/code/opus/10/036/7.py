from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6

SOURCE_COLORS = (BLUE, RED, YELLOW)
STEP_HEIGHTS = (2, 3)
PAIR_COUNTS = (2, 3)

GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
BAR_GAP_RANGE = (4, 8)
EDGE_MARGIN = 1
ROW_MARGIN = 2
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bar_pair(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "each source color must be present"

    cols = sorted(set(map(int, coords[:, 1])))
    assert len(cols) == 2, "each source color must form exactly two bars"

    bars: list[tuple[int, int, int]] = []
    for col in cols:
        rows = sorted(map(int, coords[coords[:, 1] == col, 0]))
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be vertically contiguous"
        bars.append((rows[0], rows[-1], col))

    left, right = bars
    assert left[0] == right[0] and left[1] == right[1], "paired bars must share one row interval"
    return left[0], left[1], left[2], right[2]


def _step_offset(row: int, r0: int, r1: int, height: int) -> int:
    if row < r0:
        return ((r0 - 1 - row) // height) + 1
    if row > r1:
        return ((row - r1 - 1) // height) + 1
    return 0


def _expand_walls(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    source_colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)
    assert len(source_colors) in PAIR_COUNTS, "expected two or three colored bar pairs"
    assert GREEN not in source_colors and PINK not in source_colors, "green and pink are reserved for output"

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color in source_colors:
        r0, r1, left_col, right_col = _bar_pair(g, color)
        bar_height = r1 - r0 + 1
        assert bar_height in STEP_HEIGHTS, "bar height must be 2 or 3"
        assert 4 <= right_col - left_col <= 8, "bar gap must be 4..8"

        for row in range(height):
            offset = _step_offset(row, r0, r1, bar_height)
            left_step = left_col - offset
            right_step = right_col + offset
            if 0 <= left_step < width:
                claims[(row, left_step)].add(color)
            if 0 <= right_step < width:
                claims[(row, right_step)].add(color)

    out = np.full_like(g, BLACK)
    for (row, col), colors in claims.items():
        out[row, col] = next(iter(colors)) if len(colors) == 1 else PINK
    return out


def _black_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _has_cross_center(component: list[tuple[int, int]]) -> bool:
    cells = set(component)
    for r, c in component:
        if all((r + dr, c + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            return True
    return False


def _fill_cross_regions(expanded: np.ndarray) -> np.ndarray:
    out = np.asarray(expanded, dtype=int).copy()
    for component in _black_components(out == BLACK):
        if _has_cross_center(component):
            for r, c in component:
                out[r, c] = GREEN
    return out


def _analyze_instance(grid: np.ndarray) -> dict[str, object]:
    expanded = _expand_walls(grid)
    regions = _black_components(expanded == BLACK)
    filled_sizes: list[int] = []
    unfilled_sizes: list[int] = []
    for component in regions:
        (filled_sizes if _has_cross_center(component) else unfilled_sizes).append(len(component))
    return {
        "expanded": expanded,
        "region_count": len(regions),
        "filled_sizes": filled_sizes,
        "unfilled_sizes": unfilled_sizes,
        "pink_count": int(np.count_nonzero(expanded == PINK)),
        "wall_density": float(np.count_nonzero(expanded != BLACK)) / float(expanded.size),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored vertical-bar pair into stepped walls, then fill exactly the enclosed black regions that contain a plus center.

    Input structure:
    - The grid has a black background and contains 2 or 3 foreground colors.
    - Each foreground color forms exactly one pair of short vertical bars of equal height and distinct columns.
    - A bar pair's height is its step period, and different pairs may overlap only after expansion.

    Rule:
    - For each color, keep its two bars in their original row interval.
    - Above and below that interval, after every full bar-height block of rows, shift the left bar one column left and the right bar one column right; clip anything that leaves the grid.
    - If two different colors claim the same expanded cell, recolor that cell pink.
    - Treat the expanded colored cells and pink overlap cells as walls, find every 4-connected black region, and fill an entire region green iff it contains at least one black cell whose four orthogonal neighbors are also black within that same region.

    Key insight:
    - The stepped expansion is only an intermediate construction: it changes the black topology into separate rooms, and the final output depends on whether each room is wide enough to contain an internal 5-cell orthogonal cross center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    expanded = _expand_walls(g)
    return _fill_cross_regions(expanded)


def _choose_midpoints(width: int, pair_count: int) -> list[int]:
    attempts = 60
    for _ in range(attempts):
        mids = sorted(random.sample(range(4, width - 4), pair_count))
        if all(b - a >= 3 for a, b in zip(mids, mids[1:])):
            return mids
    return sorted(random.sample(range(4, width - 4), pair_count))


def _place_pair(
    grid: np.ndarray,
    color: int,
    midpoint: int,
    used_columns: set[int],
) -> bool:
    height, width = map(int, grid.shape)
    for _ in range(80):
        bar_height = random.choice(STEP_HEIGHTS)
        gap = random.randint(*BAR_GAP_RANGE)
        left_col = midpoint - gap // 2 + random.randint(-1, 1)
        right_col = left_col + gap
        if left_col < EDGE_MARGIN or right_col >= width - EDGE_MARGIN:
            continue
        if left_col in used_columns or right_col in used_columns:
            continue

        r0 = random.randint(ROW_MARGIN, height - bar_height - ROW_MARGIN - 1)
        if np.any(grid[r0 : r0 + bar_height, left_col] != BLACK):
            continue
        if np.any(grid[r0 : r0 + bar_height, right_col] != BLACK):
            continue

        grid[r0 : r0 + bar_height, left_col] = color
        grid[r0 : r0 + bar_height, right_col] = color
        used_columns.add(left_col)
        used_columns.add(right_col)
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.choice(PAIR_COUNTS)
    colors = random.sample(SOURCE_COLORS, pair_count)

    grid = np.full((height, width), BLACK, dtype=int)
    used_columns: set[int] = set()
    for color, midpoint in zip(colors, _choose_midpoints(width, pair_count)):
        if not _place_pair(grid, color, midpoint, used_columns):
            return None

    info = _analyze_instance(grid)
    final = _fill_cross_regions(info["expanded"])

    if info["pink_count"] < 1:
        return None
    if info["region_count"] < 3:
        return None
    if not info["filled_sizes"] or not info["unfilled_sizes"]:
        return None
    if max(info["filled_sizes"]) < 8:
        return None
    if max(info["unfilled_sizes"]) < 4:
        return None
    if info["wall_density"] > 0.42:
        return None
    if not np.array_equal(main(grid), final):
        return None
    if np.any(final[info["expanded"] != BLACK] != info["expanded"][info["expanded"] != BLACK]):
        return None
    if np.count_nonzero(final == GREEN) != sum(info["filled_sizes"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

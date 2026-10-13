from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows or cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _spans(lines: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for pos in sorted(lines):
        if start < pos:
            spans.append((start, pos))
        start = pos + 1
    if start < limit:
        spans.append((start, limit))
    return spans


def _analyze_soil(coords: np.ndarray, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if int(coords.shape[0]) < 2:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]

    if int(rows.min()) == int(rows.max()):
        row = int(rows[0])
        ordered = sorted(int(c) for c in cols)
        if any(b - a != 1 for a, b in zip(ordered, ordered[1:])):
            return None
        if ordered[0] == 0 and ordered[-1] < width - 1:
            return (row, ordered[-1]), (0, 1)
        if ordered[-1] == width - 1 and ordered[0] > 0:
            return (row, ordered[0]), (0, -1)
        return None

    if int(cols.min()) == int(cols.max()):
        col = int(cols[0])
        ordered = sorted(int(r) for r in rows)
        if any(b - a != 1 for a, b in zip(ordered, ordered[1:])):
            return None
        if ordered[0] == 0 and ordered[-1] < height - 1:
            return (ordered[-1], col), (1, 0)
        if ordered[-1] == height - 1 and ordered[0] > 0:
            return (ordered[0], col), (-1, 0)
        return None

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each region's soil segment by the number of seed pixels, repainting the growth in the seed color.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms full horizontal and vertical separator lines
      that partition the grid into rectangular regions.
    - Each region contains scattered same-color seed pixels and one straight
      horizontal or vertical soil segment of another color, with one soil end on
      the region boundary and the other end inside the region.

    Rule:
    - Process each rectangular region independently.
    - Count the number of seed pixels N in the region.
    - Erase the seed pixels to background, then continue the soil from its inner
      endpoint along the same direction for N cells, using the seed color.
    - Keep the original soil segment, separator lines, and grid size unchanged.

    Key insight:
    - The separator lines localize the task into independent mini-scenes where the
      seed count provides a length and the soil endpoint on the border provides the
      growth direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    bg = _get_bg(g)
    _sep_color, sep_rows, sep_cols = _separator_color(g, bg)
    assert sep_rows and sep_cols, "expected both horizontal and vertical separator lines"
    row_spans = _spans(sep_rows, height)
    col_spans = _spans(sep_cols, width)
    assert row_spans and col_spans, "separator lines must define at least one region"

    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            region_colors = [int(color) for color in np.unique(region) if int(color) != bg]
            assert len(region_colors) == 2, "each region must contain one seed color and one soil color"

            soil_info = None
            seed_color = None
            for color in region_colors:
                coords = np.argwhere(region == color)
                info = _analyze_soil(coords, region.shape[0], region.shape[1])
                if info is not None:
                    assert soil_info is None, "expected a unique soil segment in each region"
                    soil_info = (color, coords, info)
                else:
                    seed_color = color

            assert soil_info is not None and seed_color is not None, "failed to identify seeds and soil"
            soil_color, soil_coords, (inner_end, direction) = soil_info
            assert soil_color != seed_color

            seed_coords = np.argwhere(region == seed_color)
            seed_count = int(seed_coords.shape[0])
            assert 1 <= seed_count <= 4, "seed count must stay in the intended range"

            for local_r, local_c in seed_coords:
                out[r0 + int(local_r), c0 + int(local_c)] = bg

            dr, dc = direction
            end_r, end_c = inner_end
            for step in range(1, seed_count + 1):
                rr = end_r + dr * step
                cc = end_c + dc * step
                assert 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1], "growth must stay inside region"
                out[r0 + rr, c0 + cc] = seed_color

    return out


# --------------- generator ---------------

ARC_MAX = 30
COLOR_VALUES = tuple(range(10))
REGION_COUNT_OPTIONS = (2, 3)
REGION_SIZE_MIN = 5
REGION_SIZE_MAX = 9
SEP_WIDTH = 1
SEED_COUNT_MIN = 1
SEED_COUNT_MAX = 4
GENERATION_ATTEMPTS = 400
SEED_PLACEMENT_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _sample_sizes(count: int) -> list[int]:
    return [random.randint(REGION_SIZE_MIN, REGION_SIZE_MAX) for _ in range(count)]


def _sign_schedule(total: int) -> list[int]:
    signs = [1] * (total // 2) + [-1] * (total - total // 2)
    random.shuffle(signs)
    return signs


def _orientation_schedule(total: int) -> list[str]:
    orientations = ["h"] * (total // 2) + ["v"] * (total - total // 2)
    random.shuffle(orientations)
    return orientations


def _candidate_seed_cells(
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in forbidden
    ]
    random.shuffle(cells)
    return cells


def _pick_seeds(
    height: int,
    width: int,
    seed_count: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(SEED_PLACEMENT_ATTEMPTS):
        seeds: list[tuple[int, int]] = []
        for cell in _candidate_seed_cells(height, width, forbidden):
            if any(abs(cell[0] - sr) + abs(cell[1] - sc) == 1 for sr, sc in seeds):
                continue
            seeds.append(cell)
            if len(seeds) == seed_count:
                return seeds
    return None


def _paint_region(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    top: int,
    left: int,
    height: int,
    width: int,
    seed_color: int,
    soil_color: int,
    orientation: str,
    sign: int,
) -> bool:
    axis_len = width if orientation == "h" else height
    lateral_len = height if orientation == "h" else width
    max_seed_count = min(SEED_COUNT_MAX, axis_len - 2)
    if lateral_len < 3 or max_seed_count < SEED_COUNT_MIN:
        return False

    seed_count = random.randint(SEED_COUNT_MIN, max_seed_count)
    soil_len = random.randint(2, axis_len - seed_count)

    if orientation == "h":
        row = random.randint(1, height - 2)
        if sign > 0:
            soil = [(row, c) for c in range(soil_len)]
            extension = [(row, c) for c in range(soil_len, soil_len + seed_count)]
        else:
            soil = [(row, c) for c in range(width - soil_len, width)]
            extension = [(row, c) for c in range(width - soil_len - seed_count, width - soil_len)]
    else:
        col = random.randint(1, width - 2)
        if sign > 0:
            soil = [(r, col) for r in range(soil_len)]
            extension = [(r, col) for r in range(soil_len, soil_len + seed_count)]
        else:
            soil = [(r, col) for r in range(height - soil_len, height)]
            extension = [(r, col) for r in range(height - soil_len - seed_count, height - soil_len)]

    forbidden = set(soil) | set(extension)
    for r, c in soil:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width:
                forbidden.add((nr, nc))

    seeds = _pick_seeds(height, width, seed_count, forbidden)
    if seeds is None:
        return False

    for r, c in soil:
        grid[top + r, left + c] = soil_color
        expected[top + r, left + c] = soil_color
    for r, c in seeds:
        grid[top + r, left + c] = seed_color
        expected[top + r, left + c] = bg
    for r, c in extension:
        expected[top + r, left + c] = seed_color
    return True


def _background_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = counts[bg]
    return all(color == bg or bg_count > count for color, count in counts.items())


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    region_rows = random.choice(REGION_COUNT_OPTIONS)
    region_cols = random.choice(REGION_COUNT_OPTIONS)
    heights = _sample_sizes(region_rows)
    widths = _sample_sizes(region_cols)

    total_h = sum(heights) + SEP_WIDTH * (region_rows - 1)
    total_w = sum(widths) + SEP_WIDTH * (region_cols - 1)
    if total_h > ARC_MAX or total_w > ARC_MAX:
        return None

    bg, sep = random.sample(COLOR_VALUES, 2)
    grid = np.full((total_h, total_w), bg, dtype=int)
    expected = grid.copy()

    sep_rows: list[int] = []
    pos = 0
    for size in heights[:-1]:
        pos += size
        sep_rows.append(pos)
        pos += SEP_WIDTH
    sep_cols: list[int] = []
    pos = 0
    for size in widths[:-1]:
        pos += size
        sep_cols.append(pos)
        pos += SEP_WIDTH

    for r in sep_rows:
        grid[r, :] = sep
        expected[r, :] = sep
    for c in sep_cols:
        grid[:, c] = sep
        expected[:, c] = sep

    total_regions = region_rows * region_cols
    orientations = _orientation_schedule(total_regions)
    signs = _sign_schedule(total_regions)

    palette = [int(color) for color in COLOR_VALUES if color not in (bg, sep)]
    if len(palette) < 2:
        return None

    row_tops: list[int] = []
    pos = 0
    for size in heights:
        row_tops.append(pos)
        pos += size + SEP_WIDTH
    col_lefts: list[int] = []
    pos = 0
    for size in widths:
        col_lefts.append(pos)
        pos += size + SEP_WIDTH

    idx = 0
    for i, height in enumerate(heights):
        for j, width in enumerate(widths):
            seed_color, soil_color = random.sample(palette, 2)
            ok = _paint_region(
                grid,
                expected,
                bg,
                row_tops[i],
                col_lefts[j],
                height,
                width,
                seed_color,
                soil_color,
                orientations[idx],
                signs[idx],
            )
            if not ok:
                return None
            idx += 1

    if not _background_is_unique_dominant(grid, bg):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid soil-growth puzzle input")

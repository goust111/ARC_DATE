from __future__ import annotations

import random

import numpy as np


BG = 0
LINE = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("ul", "ur", "ll", "lr")
LAYOUTS = ((2, 2), (2, 3), (2, 4), (3, 2), (4, 2))
REGION_H_RANGE = (3, 5)
MAX_EXTRA_WIDTH = 3
MAX_GRID_H = 24
MAX_GRID_W = 30
MAX_ATTEMPTS = 400


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: set[int] = set()

    for r in range(h):
        value = int(grid[r, 0])
        if value != BG and np.all(grid[r, :] == value):
            candidates.add(value)
    for c in range(w):
        value = int(grid[0, c])
        if value != BG and np.all(grid[:, c] == value):
            candidates.add(value)

    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates.pop()


def _intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in separators:
        assert 0 <= idx < size
        assert idx >= start
        if start < idx:
            spans.append((start, idx))
        start = idx + 1
    if start < size:
        spans.append((start, size))
    assert spans, "expected at least one non-separator span"
    return spans


def _quadrant(local_r: int, local_c: int, height: int, width: int) -> str:
    row_cmp = 2 * local_r + 1 - height
    col_cmp = 2 * local_c + 1 - width
    assert row_cmp != 0 and col_cmp != 0, "seed must not lie on a center axis"

    vertical = "u" if row_cmp < 0 else "l"
    horizontal = "l" if col_cmp < 0 else "r"
    return vertical + horizontal


def _segment_length(region: np.ndarray) -> int:
    pure_bg_rows = int(np.sum(np.all(region == BG, axis=1)))
    assert pure_bg_rows > 0, "each region must contain at least one pure background row"
    return pure_bg_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an L-shaped branch inside each separator-defined region.

    Input structure:
    - Background is black.
    - Whole grey rows and columns partition the grid into rectangles.
    - Each rectangle contains exactly one colored seed; all other cells there are black.

    Rule:
    - Determine the seed quadrant relative to the region center.
    - The quadrant chooses the diagonal expansion direction:
      upper-left -> right then down,
      upper-right -> left then down,
      lower-left -> right then up,
      lower-right -> left then up.
    - Each segment length equals the number of pure-background rows in that region.
    - Draw the horizontal segment first from the seed, then the vertical segment
      from the horizontal endpoint, both in the seed color.
    - Separator lines remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    line = _separator_color(g)
    sep_rows = [r for r in range(h) if np.all(g[r, :] == line)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == line)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"

    row_spans = _intervals(h, sep_rows)
    col_spans = _intervals(w, sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            assert region.size > 0
            assert not np.any(region == line), "separator color cannot appear inside a region"

            seeds = np.argwhere(region != BG)
            assert seeds.shape == (1, 2), "each region must contain exactly one seed"
            local_r, local_c = map(int, seeds[0])
            color = int(region[local_r, local_c])

            length = _segment_length(region)
            quad = _quadrant(local_r, local_c, region.shape[0], region.shape[1])
            horiz_dir = 1 if quad in {"ul", "ll"} else -1
            vert_dir = 1 if quad in {"ul", "ur"} else -1

            end_c = local_c + horiz_dir * (length - 1)
            end_r = local_r + vert_dir * (length - 1)
            assert 0 <= end_c < region.shape[1], "horizontal segment leaves its region"
            assert 0 <= end_r < region.shape[0], "vertical segment leaves its region"

            abs_seed_r = r0 + local_r
            for step in range(length):
                out[abs_seed_r, c0 + local_c + horiz_dir * step] = color
            for step in range(length):
                out[r0 + local_r + vert_dir * step, c0 + end_c] = color

    return out


def _valid_seed_rows(height: int, quadrant: str) -> list[int]:
    length = height - 1
    rows: list[int] = []
    for r in range(height):
        is_top = 2 * r + 1 < height
        is_bottom = 2 * r + 1 > height
        if quadrant in {"ul", "ur"}:
            if is_top and r + (length - 1) < height:
                rows.append(r)
        else:
            if is_bottom and r - (length - 1) >= 0:
                rows.append(r)
    return rows


def _valid_seed_cols(width: int, height: int, quadrant: str) -> list[int]:
    length = height - 1
    cols: list[int] = []
    for c in range(width):
        is_left = 2 * c + 1 < width
        is_right = 2 * c + 1 > width
        if quadrant in {"ul", "ll"}:
            if is_left and c + (length - 1) < width:
                cols.append(c)
        else:
            if is_right and c - (length - 1) >= 0:
                cols.append(c)
    return cols


def _choose_layout() -> tuple[int, int]:
    return random.choice(LAYOUTS)


def _region_dimensions(row_regions: int, col_regions: int) -> tuple[list[int], list[int]] | None:
    heights = [random.randint(*REGION_H_RANGE) for _ in range(row_regions)]
    max_h = max(heights)
    min_w = max(4, max_h - 1)
    max_w = min(8, min_w + MAX_EXTRA_WIDTH)
    widths = [random.randint(min_w, max_w) for _ in range(col_regions)]

    total_h = sum(heights) + (row_regions - 1)
    total_w = sum(widths) + (col_regions - 1)
    if total_h > MAX_GRID_H or total_w > MAX_GRID_W:
        return None
    return heights, widths


def _spans_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        spans.append((cursor, cursor + size))
        cursor += size
        if idx + 1 < len(sizes):
            separators.append(cursor)
            cursor += 1
    return spans, separators


def _quadrants_for_regions(count: int) -> list[str]:
    quads = list(QUADRANTS)
    while len(quads) < count:
        quads.append(random.choice(QUADRANTS))
    random.shuffle(quads)
    return quads[:count]


def _place_seed(region: np.ndarray, quadrant: str, color: int) -> bool:
    h, w = region.shape
    rows = _valid_seed_rows(h, quadrant)
    cols = _valid_seed_cols(w, h, quadrant)
    if not rows or not cols:
        return False
    r = random.choice(rows)
    c = random.choice(cols)
    region[r, c] = color
    return True


def _build_input() -> np.ndarray | None:
    row_regions, col_regions = _choose_layout()
    dims = _region_dimensions(row_regions, col_regions)
    if dims is None:
        return None
    heights, widths = dims

    row_spans, sep_rows = _spans_from_sizes(heights)
    col_spans, sep_cols = _spans_from_sizes(widths)

    total_h = sum(heights) + len(sep_rows)
    total_w = sum(widths) + len(sep_cols)
    grid = np.full((total_h, total_w), BG, dtype=int)
    if sep_rows:
        grid[np.array(sep_rows), :] = LINE
    if sep_cols:
        grid[:, np.array(sep_cols)] = LINE

    region_count = row_regions * col_regions
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    quadrants = _quadrants_for_regions(region_count)

    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = grid[r0:r1, c0:c1]
            if not _place_seed(region, quadrants[idx], colors[idx]):
                return None
            idx += 1

    return grid


def _is_valid_example(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        if _is_valid_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")

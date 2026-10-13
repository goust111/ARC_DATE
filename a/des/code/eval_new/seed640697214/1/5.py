from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PATTERN_COLORS = [3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _detect_separator_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    horizontal_rows = [
        r for r in range(h) if int(g[r, 0]) != bg and bool(np.all(g[r] == g[r, 0]))
    ]
    horizontal_set = set(horizontal_rows)
    non_horizontal_rows = [r for r in range(h) if r not in horizontal_set]

    vertical_cols: list[int] = []
    for c in range(w):
        vals = g[non_horizontal_rows, c]
        if vals.size == 0:
            continue
        if int(vals[0]) != bg and bool(np.all(vals == vals[0])):
            vertical_cols.append(c)

    return horizontal_rows, vertical_cols


def _region_bounds(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + list(separators) + [size]
    regions: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        stop = bounds[i + 1] - 1
        if start <= stop:
            regions.append((start, stop))
    return regions


def _line_color_at_row(grid: np.ndarray, row: int) -> int:
    return int(grid[row, 0])


def _line_color_at_col(grid: np.ndarray, col: int, horizontal_rows: set[int]) -> int:
    h = grid.shape[0]
    for r in range(h):
        if r in horizontal_rows:
            continue
        return int(grid[r, col])
    raise ValueError("column line has no non-horizontal sample")


def _transform_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = patch
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    h, w = g.shape

    horizontal_rows, vertical_cols = _detect_separator_lines(g, bg)
    horizontal_set = set(horizontal_rows)
    horizontal_regions = _region_bounds(h, horizontal_rows)
    vertical_regions = _region_bounds(w, vertical_cols)

    row_line_colors = [_line_color_at_row(g, r) for r in horizontal_rows]
    col_line_colors = [_line_color_at_col(g, c, horizontal_set) for c in vertical_cols]

    out = g.copy()
    for r_idx, (r0, r1) in enumerate(horizontal_regions):
        if horizontal_rows:
            top_color = row_line_colors[r_idx - 1] if r_idx > 0 else None
            bottom_color = row_line_colors[r_idx] if r_idx < len(horizontal_rows) else None
            if top_color is None and bottom_color is not None:
                top_color = bottom_color
            if bottom_color is None and top_color is not None:
                bottom_color = top_color
        else:
            top_color = bottom_color = None

        for c_idx, (c0, c1) in enumerate(vertical_regions):
            if vertical_cols:
                left_color = col_line_colors[c_idx - 1] if c_idx > 0 else None
                right_color = col_line_colors[c_idx] if c_idx < len(vertical_cols) else None
                if left_color is None and right_color is not None:
                    left_color = right_color
                if right_color is None and left_color is not None:
                    right_color = left_color
            else:
                left_color = right_color = None

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            flip_ud = top_color is not None and bottom_color is not None and top_color != bottom_color
            flip_lr = left_color is not None and right_color is not None and left_color != right_color
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_patch(patch, flip_ud, flip_lr)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Transform each partitioned region according to the colors of its opposite separator lines.

    Input structure:
    - bg is black.
    - One-cell-thick full-width horizontal lines and full-height vertical lines split the grid into rectangular regions.
    - Each separator line is red or blue, and each region contains a small non-symmetric pattern in another color away from the region boundaries.

    Rule:
    - For every region, compare the top and bottom boundary colors and compare the left and right boundary colors.
    - A missing outer boundary counts as having the same color as the opposite side.
    - Same/same leaves the region interior unchanged; different/same flips it vertically; same/different flips it horizontally; different/different rotates it 180 degrees.
    - Separator lines and overall grid size stay unchanged.

    Key insight:
    - The two opposite-boundary color comparisons independently encode vertical and horizontal reflection, so their combination specifies the region's geometric transform.
    """
    return _solve(grid)


def _line_positions(sizes: list[int]) -> list[int]:
    positions: list[int] = []
    cursor = sizes[0]
    for i in range(len(sizes) - 1):
        positions.append(cursor)
        cursor += 1 + sizes[i + 1]
    return positions


def _sample_region_sizes(rng: random.Random, count: int) -> list[int]:
    min_size = 5
    max_size = 8 if count == 3 else 6
    return [rng.randint(min_size, max_size) for _ in range(count)]


def _sample_line_colors(rng: random.Random, count: int) -> list[int]:
    colors = [BLUE, RED]
    for _ in range(100):
        seq = [rng.choice(colors) for _ in range(count)]
        if any(seq[i] != seq[i + 1] for i in range(count - 1)):
            return seq
    raise RuntimeError("failed to sample non-degenerate separator colors")


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_pattern_mask(max_h: int, max_w: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        shape_h = rng.randint(2, max_h)
        shape_w = rng.randint(2, max_w)
        area = shape_h * shape_w
        count = rng.randint(3, max(3, min(area - 1, shape_h + shape_w + 1)))

        mask = np.zeros((shape_h, shape_w), dtype=bool)
        start = (rng.randrange(shape_h), rng.randrange(shape_w))
        cells = {start}
        mask[start] = True

        while len(cells) < count:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < shape_h and 0 <= nc < shape_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            cells.add((nr, nc))
            mask[nr, nc] = True

        if mask.sum() < 3:
            continue
        mask = _crop_mask(mask)
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric pattern")


def _place_patterns(
    grid: np.ndarray,
    row_regions: list[tuple[int, int]],
    col_regions: list[tuple[int, int]],
    rng: random.Random,
) -> None:
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            region_h = r1 - r0 + 1
            region_w = c1 - c0 + 1
            mask = _random_pattern_mask(min(4, region_h - 2), min(4, region_w - 2), rng)
            color = rng.choice(PATTERN_COLORS)
            top = rng.randint(r0 + 1, r1 - mask.shape[0])
            left = rng.randint(c0 + 1, c1 - mask.shape[1])
            patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
            patch[mask] = color


def _generate_candidate(rng: random.Random) -> np.ndarray:
    horizontal_line_count = rng.randint(2, 3)
    vertical_line_count = rng.randint(2, 3)

    row_band_sizes = _sample_region_sizes(rng, horizontal_line_count + 1)
    col_band_sizes = _sample_region_sizes(rng, vertical_line_count + 1)

    h = sum(row_band_sizes) + horizontal_line_count
    w = sum(col_band_sizes) + vertical_line_count
    assert h <= 30 and w <= 30

    horizontal_rows = _line_positions(row_band_sizes)
    vertical_cols = _line_positions(col_band_sizes)
    horizontal_colors = _sample_line_colors(rng, horizontal_line_count)
    vertical_colors = _sample_line_colors(rng, vertical_line_count)

    grid = np.full((h, w), BG, dtype=int)
    for c, color in zip(vertical_cols, vertical_colors):
        grid[:, c] = color
    for r, color in zip(horizontal_rows, horizontal_colors):
        grid[r, :] = color

    row_regions = _region_bounds(h, horizontal_rows)
    col_regions = _region_bounds(w, vertical_cols)
    _place_patterns(grid, row_regions, col_regions, rng)

    decoded_rows, decoded_cols = _detect_separator_lines(grid, BG)
    if decoded_rows != horizontal_rows or decoded_cols != vertical_cols:
        raise RuntimeError("separator decoding mismatch")

    line_palette = set(horizontal_colors) | set(vertical_colors)
    if line_palette != {BLUE, RED}:
        raise RuntimeError("both separator colors must appear")

    _solve(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _generate_candidate(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")

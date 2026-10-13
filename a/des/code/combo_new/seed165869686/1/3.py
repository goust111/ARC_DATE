import numpy as np


BG = 0
GREY = 5
SAMPLE_COLOR = 8
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()

PATTERN_BANK: dict[int, list[np.ndarray]] = {
    2: [
        np.array([[1, 0], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0], [0, 0]], dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 0]], dtype=bool),
        np.array([[0, 1], [0, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=bool),
    ],
}

LAYOUTS_BY_REGION_COUNT = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2), (4, 1), (1, 4)],
    5: [(5, 1), (1, 5)],
    6: [(2, 3), (3, 2)],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_reference_patterns(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    full_grey_rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r] == GREY))]
    assert len(full_grey_rows) >= 3, "expected top border, band separator, and band border"
    sep_row = int(full_grey_rows[-2])
    bottom_row = int(full_grey_rows[-1])
    sample_rows = grid[sep_row + 1 : bottom_row]
    assert sample_rows.size > 0, "reference band must contain sample rows"
    size = int(sample_rows.shape[0])

    full_grey_cols = [c for c in range(grid.shape[1]) if bool(np.all(sample_rows[:, c] == GREY))]
    assert len(full_grey_cols) >= 2, "reference band must be grey-separated"

    patterns: list[np.ndarray] = []
    for left, right in zip(full_grey_cols, full_grey_cols[1:]):
        if right - left <= 1:
            continue
        block = sample_rows[:, left + 1 : right]
        assert block.shape == (size, size), "reference block must be square"
        patterns.append(((block != BG) & (block != GREY)).copy())

    assert patterns, "reference band must contain pattern blocks"
    return patterns, sep_row


def _extract_regions(top_area: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for coords in _components(top_area != GREY):
        tower_cells = [(r, c) for r, c in coords if int(top_area[r, c]) not in (BG, GREY)]
        assert len(tower_cells) == 1, "each region must contain exactly one signal tower"
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        tr, tc = tower_cells[0]
        regions.append(
            {
                "cells": coords,
                "tower": (int(tr), int(tc)),
                "color": int(top_area[tr, tc]),
                "area": int(len(coords)),
                "bbox": (int(min(rs)), int(min(cs))),
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-partitioned region with the bottom reference pattern ranked by region area.

    Input structure:
    - Black background and grey axis-aligned line segments form closed rectangular regions.
    - Each enclosed region contains exactly one colored singleton signal tower.
    - The bottom strip is a reference band of grey-separated 2x2 or 3x3 pattern blocks.

    Rule:
    - Measure every enclosed region by its non-grey cell count, including the tower cell.
    - Sort the regions from smallest area to largest area.
    - Pair the k-th smallest region with the k-th pattern block from the left in the reference band.
    - Inside that region, tile the paired pattern periodically using the tower as the pattern origin.
    - Replace every non-black pattern cell with the tower color, keep black pattern cells black, and leave the tower and grey lines unchanged.

    Key insight:
    - The reference band does not map to regions by position; it maps by sorted region area.
    - The repeated pattern is anchored at the tower location, so the tower fixes the pattern phase inside each region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    patterns, sep_row = _extract_reference_patterns(g)
    top_area = g[: sep_row + 1]
    regions = _extract_regions(top_area)
    assert len(patterns) == len(regions), "pattern count and region count must match"

    ordered_regions = sorted(regions, key=lambda region: (region["area"], region["bbox"][0], region["bbox"][1]))
    out = g.copy()
    for region, pattern in zip(ordered_regions, patterns):
        size = int(pattern.shape[0])
        tr, tc = region["tower"]
        color = int(region["color"])
        for r, c in region["cells"]:
            if (r, c) == (tr, tc):
                continue
            if pattern[(r - tr) % size, (c - tc) % size]:
                out[r, c] = color
            else:
                out[r, c] = BG
    return out


def _random_composition(total: int, parts: int, min_part: int) -> list[int] | None:
    if total < parts * min_part:
        return None
    extra = total - parts * min_part
    values = RNG.multinomial(extra, np.full(parts, 1.0 / parts)).tolist()
    values = [int(v + min_part) for v in values]
    RNG.shuffle(values)
    return values


def _sample_layout(region_count: int, pattern_size: int) -> tuple[list[int], list[int]] | None:
    total_width = int(region_count * pattern_size + region_count + 1)
    layouts = list(LAYOUTS_BY_REGION_COUNT[region_count])
    RNG.shuffle(layouts)
    for nrows, ncols in layouts:
        inner_width = total_width - (ncols + 1)
        min_height_sum = 2 * nrows
        max_height_sum = 28 - nrows - pattern_size
        if inner_width < 2 * ncols or min_height_sum > max_height_sum:
            continue
        for _ in range(250):
            widths = _random_composition(inner_width, ncols, 2)
            if widths is None:
                continue
            total_height_sum = int(RNG.integers(min_height_sum, max_height_sum + 1))
            heights = _random_composition(total_height_sum, nrows, 2)
            if heights is None:
                continue
            areas = [h * w for h in heights for w in widths]
            if len(set(areas)) != region_count:
                continue
            return [int(v) for v in heights], [int(v) for v in widths]
    return None


def _line_positions(sizes: list[int]) -> list[int]:
    pos = 0
    lines = [0]
    for size in sizes:
        pos += int(size) + 1
        lines.append(int(pos))
    return lines


def _build_grid(
    heights: list[int],
    widths: list[int],
    pattern_size: int,
    patterns: list[np.ndarray],
    tower_colors: list[int],
) -> np.ndarray:
    nrows = len(heights)
    ncols = len(widths)
    region_count = nrows * ncols
    assert len(patterns) == region_count
    assert len(tower_colors) == region_count

    row_lines = _line_positions(heights)
    col_lines = _line_positions(widths)
    top_height = row_lines[-1] + 1
    total_width = col_lines[-1] + 1
    total_height = top_height + pattern_size + 1
    assert total_height <= 30 and total_width <= 30

    grid = np.full((total_height, total_width), BG, dtype=int)
    for r in row_lines:
        grid[r, :] = GREY
    for c in col_lines:
        grid[:top_height, c] = GREY

    cells: list[dict] = []
    color_index = 0
    for ri in range(nrows):
        for ci in range(ncols):
            r0, r1 = row_lines[ri] + 1, row_lines[ri + 1]
            c0, c1 = col_lines[ci] + 1, col_lines[ci + 1]
            tr = int(RNG.integers(r0, r1))
            tc = int(RNG.integers(c0, c1))
            color = int(tower_colors[color_index])
            color_index += 1
            grid[tr, tc] = color
            cells.append(
                {
                    "area": int((r1 - r0) * (c1 - c0)),
                    "tower": (tr, tc),
                    "color": color,
                    "bbox": (r0, c0),
                }
            )

    sample_top = top_height
    grid[sample_top : sample_top + pattern_size, :] = GREY
    grid[sample_top + pattern_size, :] = GREY

    ordered_cells = sorted(cells, key=lambda cell: (cell["area"], cell["bbox"][0], cell["bbox"][1]))
    for idx, (cell, pattern) in enumerate(zip(ordered_cells, patterns)):
        c0 = 1 + idx * (pattern_size + 1)
        grid[sample_top : sample_top + pattern_size, c0 : c0 + pattern_size] = BG
        grid[sample_top : sample_top + pattern_size, c0 : c0 + pattern_size][pattern] = SAMPLE_COLOR

    return grid


def _gen_attempt() -> np.ndarray | None:
    pattern_size = int(RNG.choice([2, 3]))
    region_count = int(RNG.choice([3, 4, 5, 6]))
    layout = _sample_layout(region_count, pattern_size)
    if layout is None:
        return None
    heights, widths = layout

    patterns = [PATTERN_BANK[pattern_size][idx].copy() for idx in RNG.choice(len(PATTERN_BANK[pattern_size]), size=region_count, replace=False)]
    tower_colors = [int(color) for color in RNG.choice(TOWER_COLORS, size=region_count, replace=False)]
    grid = _build_grid(heights, widths, pattern_size, patterns, tower_colors)

    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if len(set(heights[i] * widths[j] for i in range(len(heights)) for j in range(len(widths)))) != region_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")

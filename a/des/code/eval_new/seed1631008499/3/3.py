import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _find_divider_color(grid: np.ndarray) -> int:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        full_rows = int(np.sum(np.all(grid == color, axis=1)))
        full_cols = int(np.sum(np.all(grid == color, axis=0)))
        if full_rows >= 2 and full_cols >= 2:
            count = int(np.count_nonzero(grid == color))
            candidates.append((color == GREY, full_rows + full_cols, count, color))
    if not candidates:
        raise AssertionError("expected one color to form full-row and full-column dividers")
    return max(candidates)[-1]


def _interior_spans(line_indices: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    for start, end in zip(line_indices, line_indices[1:]):
        if int(end) - int(start) > 1:
            spans.append((int(start) + 1, int(end)))
    if not spans:
        raise AssertionError("expected at least one enclosed region")
    return spans


def _region_slices(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    divider = _find_divider_color(grid)
    row_lines = np.flatnonzero(np.all(grid == divider, axis=1))
    col_lines = np.flatnonzero(np.all(grid == divider, axis=0))
    if len(row_lines) < 2 or len(col_lines) < 2:
        raise AssertionError("divider lines must split the grid in both directions")
    return divider, _interior_spans(row_lines), _interior_spans(col_lines)


def _seed_info(patch: np.ndarray, divider: int) -> tuple[int, int] | None:
    seeds = patch[(patch != BG) & (patch != divider)]
    if seeds.size == 0:
        return None
    colors = {int(v) for v in seeds}
    if len(colors) != 1:
        raise AssertionError("all seeds in one region must share a color")
    count = int(seeds.size)
    if count not in (1, 2, 3):
        raise AssertionError("seed count must be 1, 2, or 3")
    return count, next(iter(colors))


def _fill_pattern(height: int, width: int, count: int, color: int) -> np.ndarray:
    if count == 1:
        return np.full((height, width), color, dtype=int)
    if count == 2:
        patch = np.full((height, width), BG, dtype=int)
        patch[::2, :] = color
        return patch
    rr, cc = np.indices((height, width))
    return np.where((rr + cc) % 2 == 0, color, BG).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each bordered subregion with a count-dependent pattern using its seed color.

    Input structure:
    - bg = black.
    - Grey full-width and full-height divider lines partition the grid into rectangular subregions.
    - Each subregion interior is either empty black or contains 1 to 3 scattered same-colored seed pixels.

    Rule:
    - For every non-empty subregion, count its seeds and read their shared color.
    - One seed fills the whole interior with that color; two seeds produce horizontal stripes starting with a colored top row; three seeds produce a checkerboard starting with a colored top-left cell.
    - Empty subregions and all grey divider lines stay unchanged.

    Key insight:
    - The visible seed color supplies the fill color, while the visible seed count independently selects which of the three fill patterns to draw.
    """
    g = np.asarray(grid, dtype=int)
    divider, row_spans, col_spans = _region_slices(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            info = _seed_info(g[r0:r1, c0:c1], divider)
            if info is None:
                continue
            count, color = info
            out[r0:r1, c0:c1] = _fill_pattern(r1 - r0, c1 - c0, count, color)

    return out


def _build_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    grid = np.full((height, width), BG, dtype=int)

    row_lines = [0]
    cursor = 0
    for span in row_heights:
        cursor += span + 1
        row_lines.append(cursor)

    col_lines = [0]
    cursor = 0
    for span in col_widths:
        cursor += span + 1
        col_lines.append(cursor)

    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    row_spans = [(row_lines[i] + 1, row_lines[i + 1]) for i in range(len(row_heights))]
    col_spans = [(col_lines[i] + 1, col_lines[i + 1]) for i in range(len(col_widths))]
    return grid, row_spans, col_spans


def _sample_seed_cells(rng: random.Random, height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    return rng.sample(cells, count)


def _gen_attempt(rng: random.Random) -> np.ndarray:
    row_count = rng.randint(2, 4)
    col_count = rng.randint(2, 4)
    row_heights = [rng.randint(2, 5) for _ in range(row_count)]
    col_widths = [rng.randint(2, 5) for _ in range(col_count)]

    grid, row_spans, col_spans = _build_grid(row_heights, col_widths)
    total_regions = row_count * col_count
    filled_regions = rng.randint(3, total_regions - 1)
    region_indices = rng.sample(range(total_regions), filled_regions)

    branch_counts = [1, 2, 3]
    branch_counts.extend(rng.choice((1, 2, 3)) for _ in range(filled_regions - 3))
    rng.shuffle(branch_counts)

    palette_size = rng.randint(2, min(5, len(SEED_COLORS)))
    palette = rng.sample(SEED_COLORS, palette_size)
    region_colors = [rng.choice(palette) for _ in range(filled_regions)]
    if len(set(region_colors)) == 1 and len(palette) > 1:
        region_colors[-1] = palette[1]

    for flat_index, count, color in zip(region_indices, branch_counts, region_colors):
        row_id, col_id = divmod(flat_index, col_count)
        r0, r1 = row_spans[row_id]
        c0, c1 = col_spans[col_id]
        cells = _sample_seed_cells(rng, r1 - r0, c1 - c0, count)
        for rr, cc in cells:
            grid[r0 + rr, c0 + cc] = color

    divider, parsed_rows, parsed_cols = _region_slices(grid)
    observed_counts = []
    empty_regions = 0
    for r0, r1 in parsed_rows:
        for c0, c1 in parsed_cols:
            info = _seed_info(grid[r0:r1, c0:c1], divider)
            if info is None:
                empty_regions += 1
            else:
                observed_counts.append(info[0])

    if empty_regions == 0:
        raise AssertionError("generator must keep at least one empty region")
    if {1, 2, 3} - set(observed_counts):
        raise AssertionError("generator must cover all three pattern branches")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid ARC grid")

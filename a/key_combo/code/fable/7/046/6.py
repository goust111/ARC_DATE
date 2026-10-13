from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ALL_DIRECTIONS = ("up", "down", "left", "right")
ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
MIN_REGIONS = 2
MAX_REGIONS = 3
MIN_BAND = 5
MAX_BAND = 9


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contiguous_bands(separator_flags: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(separator_flags.tolist()):
        if is_separator:
            if start is not None:
                bands.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, len(separator_flags) - 1))
    return bands


def _find_separator_layout(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    candidates = []
    for color in sorted(map(int, np.unique(grid))):
        full_rows = np.all(grid == color, axis=1)
        full_cols = np.all(grid == color, axis=0)
        if not np.any(full_rows) or not np.any(full_cols):
            continue
        mask = full_rows[:, None] | full_cols[None, :]
        if np.array_equal(grid == color, mask):
            candidates.append((color, full_rows, full_cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _region_info(region: np.ndarray, bg: int) -> tuple[int, str]:
    non_bg = region[region != bg]
    colors, counts = np.unique(non_bg, return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
    core_colors = [color for color, count in color_counts.items() if count == 1]
    outline_colors = [color for color, count in color_counts.items() if count > 1]
    assert len(core_colors) == 1, "expected exactly one core pixel in each region"
    assert len(outline_colors) == 1, "expected exactly one outline color in each region"

    core_color = core_colors[0]
    outline_color = outline_colors[0]
    outline_coords = np.argwhere(region == outline_color)
    assert outline_coords.size > 0

    r0, c0, r1, c1 = _bbox_coords(outline_coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w and side_h >= 3 and side_h % 2 == 1

    square = region[r0 : r1 + 1, c0 : c1 + 1]
    mid = side_h // 2
    gap_positions = {
        "up": (0, mid),
        "down": (side_h - 1, mid),
        "left": (mid, 0),
        "right": (mid, side_w - 1),
    }
    gap_dirs = [name for name, (rr, cc) in gap_positions.items() if int(square[rr, cc]) != outline_color]
    assert len(gap_dirs) == 1, "expected exactly one outline gap"
    gap_dir = gap_dirs[0]

    for rr in range(side_h):
        for cc in range(side_w):
            on_border = rr in (0, side_h - 1) or cc in (0, side_w - 1)
            if not on_border:
                continue
            if (rr, cc) == gap_positions[gap_dir]:
                assert int(square[rr, cc]) == bg
            else:
                assert int(square[rr, cc]) == outline_color

    core_pos = np.argwhere(region == core_color)
    assert core_pos.shape == (1, 2)
    core_r, core_c = map(int, core_pos[0])
    assert r0 < core_r < r1 and c0 < core_c < c1, "core must be inside the outline"

    return core_color, gap_dir


def _analyze_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[int]], list[list[str]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_color, full_rows, full_cols = _find_separator_layout(g)
    row_bands = _contiguous_bands(full_rows)
    col_bands = _contiguous_bands(full_cols)
    assert row_bands and col_bands

    separator_mask = full_rows[:, None] | full_cols[None, :]
    bg = int(Counter(map(int, g[~separator_mask].ravel())).most_common(1)[0][0])
    assert bg != separator_color

    core_colors: list[list[int]] = []
    gap_dirs: list[list[str]] = []
    for r0, r1 in row_bands:
        row_core_colors: list[int] = []
        row_gap_dirs: list[str] = []
        for c0, c1 in col_bands:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            core_color, gap_dir = _region_info(region, bg)
            row_core_colors.append(core_color)
            row_gap_dirs.append(gap_dir)
        core_colors.append(row_core_colors)
        gap_dirs.append(row_gap_dirs)

    return row_bands, col_bands, core_colors, gap_dirs


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize a separator partition by dereferencing each region's gap toward a neighboring core color.

    Input structure:
    - One color forms complete horizontal and vertical separator lines that split the board into rectangular regions.
    - Each region contains one hollow square outline with exactly one single-cell gap at the center of one side.
    - Each region also contains exactly one single-pixel core of a different color inside that outline.

    Rule:
    - Count the separator-defined row bands and column bands to determine the output size.
    - For each region, read which side of its square outline contains the gap.
    - Move to the orthogonally adjacent region in that direction and copy that neighbor region's core color into the corresponding output cell.

    Key insight:
    - A region contributes its position to the summary grid, but its value comes from another region: the outline gap is a pointer to the neighbor whose core color should be read.
    """
    row_bands, col_bands, core_colors, gap_dirs = _analyze_grid(grid)
    out = np.zeros((len(row_bands), len(col_bands)), dtype=int)

    for i in range(len(row_bands)):
        for j in range(len(col_bands)):
            dr, dc = DIR_TO_DELTA[gap_dirs[i][j]]
            ni = i + dr
            nj = j + dc
            assert 0 <= ni < len(row_bands) and 0 <= nj < len(col_bands), "gap must point to a neighboring region"
            out[i, j] = core_colors[ni][nj]

    return out


def _valid_dirs(i: int, j: int, rows: int, cols: int) -> list[str]:
    dirs = []
    if i > 0:
        dirs.append("up")
    if i + 1 < rows:
        dirs.append("down")
    if j > 0:
        dirs.append("left")
    if j + 1 < cols:
        dirs.append("right")
    return dirs


def _assign_gap_dirs(rows: int, cols: int) -> list[list[str]]:
    dirs = [[None for _ in range(cols)] for _ in range(rows)]
    dirs[0][0] = "right"
    dirs[0][cols - 1] = "down"
    dirs[rows - 1][0] = "up"
    dirs[rows - 1][cols - 1] = "left"

    for i in range(rows):
        for j in range(cols):
            if dirs[i][j] is None:
                dirs[i][j] = random.choice(_valid_dirs(i, j, rows, cols))

    used = {dirs[i][j] for i in range(rows) for j in range(cols)}
    assert used == set(ALL_DIRECTIONS)
    return dirs


def _assign_core_colors(rows: int, cols: int, reserved: set[int]) -> list[list[int]]:
    available = [color for color in ARC_COLORS if color not in reserved]
    palette_size = min(len(available), random.randint(4, 6))
    palette = random.sample(available, palette_size)

    for _ in range(100):
        colors = [[-1 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                forbidden = set()
                if i > 0:
                    forbidden.add(colors[i - 1][j])
                if j > 0:
                    forbidden.add(colors[i][j - 1])
                choices = [color for color in palette if color not in forbidden]
                if not choices:
                    break
                colors[i][j] = random.choice(choices)
            else:
                continue
            break
        else:
            flat = [color for row in colors for color in row]
            if len(set(flat)) >= min(4, rows * cols, len(palette)):
                return colors

    raise AssertionError("failed to assign diverse core colors")


def _make_band_sizes(count: int) -> list[int]:
    return [random.randint(MIN_BAND, MAX_BAND) for _ in range(count)]


def _band_starts(sizes: list[int]) -> list[int]:
    starts = []
    pos = 0
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx + 1 < len(sizes):
            pos += 1
    return starts


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    bg: int,
    outline_color: int,
    core_color: int,
    gap_dir: str,
) -> None:
    max_side = min(height, width) - 2
    side_choices = [side for side in range(3, max_side + 1, 2)]
    assert side_choices, "region too small for a gapped square"
    side = random.choice(side_choices)

    local_r0 = random.randint(1, height - side - 1)
    local_c0 = random.randint(1, width - side - 1)
    r0 = top + local_r0
    c0 = left + local_c0
    r1 = r0 + side - 1
    c1 = c0 + side - 1

    grid[r0, c0 : c1 + 1] = outline_color
    grid[r1, c0 : c1 + 1] = outline_color
    grid[r0 : r1 + 1, c0] = outline_color
    grid[r0 : r1 + 1, c1] = outline_color

    mid = side // 2
    gap_cell = {
        "up": (r0, c0 + mid),
        "down": (r1, c0 + mid),
        "left": (r0 + mid, c0),
        "right": (r0 + mid, c1),
    }[gap_dir]
    grid[gap_cell] = bg

    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    core_r, core_c = random.choice(interior)
    grid[core_r, core_c] = core_color


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(MIN_REGIONS, MAX_REGIONS)
    cols = random.randint(MIN_REGIONS, MAX_REGIONS)
    row_sizes = _make_band_sizes(rows)
    col_sizes = _make_band_sizes(cols)

    total_h = sum(row_sizes) + rows - 1
    total_w = sum(col_sizes) + cols - 1
    if total_h > 30 or total_w > 30:
        return None

    bg, separator_color, outline_color = random.sample(ARC_COLORS, 3)
    gap_dirs = _assign_gap_dirs(rows, cols)
    core_colors = _assign_core_colors(rows, cols, {bg, separator_color, outline_color})

    grid = np.full((total_h, total_w), bg, dtype=int)
    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)

    pos = row_sizes[0]
    for size in row_sizes[1:]:
        grid[pos, :] = separator_color
        pos += 1 + size

    pos = col_sizes[0]
    for size in col_sizes[1:]:
        grid[:, pos] = separator_color
        pos += 1 + size

    for i, (top, height) in enumerate(zip(row_starts, row_sizes)):
        for j, (left, width) in enumerate(zip(col_starts, col_sizes)):
            _draw_region(
                grid,
                top,
                left,
                height,
                width,
                bg,
                outline_color,
                core_colors[i][j],
                gap_dirs[i][j],
            )

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (rows, cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

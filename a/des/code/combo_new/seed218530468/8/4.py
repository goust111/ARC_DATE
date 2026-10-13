import random
from collections import Counter

import numpy as np


BG_COLOR = 0
GRID_COLOR = 5
SIGNAL_COLOR = 2
MAX_GENERATION_ATTEMPTS = 100


def _line_color(grid: np.ndarray) -> int:
    row_colors = Counter(
        int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])
    )
    col_colors = Counter(
        int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])
    )
    candidates = set(row_colors) & set(col_colors)
    assert candidates, "expected a full-row/full-column grid-line color"
    return max(candidates, key=lambda color: (row_colors[color] + col_colors[color], color))


def _common_pitch(boundaries: np.ndarray, *, ignore_unit: bool = False) -> int:
    diffs = [int(b - a) for a, b in zip(boundaries, boundaries[1:]) if int(b - a) > 0]
    if ignore_unit:
        diffs = [d for d in diffs if d > 1]
    assert diffs, "expected repeated grid-line spacing"
    return Counter(diffs).most_common(1)[0][0]


def _row_col_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    line = _line_color(grid)
    full_rows = np.flatnonzero(np.all(grid == line, axis=1))
    full_cols = np.flatnonzero(np.all(grid == line, axis=0))

    row_pitch = _common_pitch(full_rows)
    col_pitch = _common_pitch(full_cols, ignore_unit=True)

    assert all(int(b - a) == row_pitch for a, b in zip(full_rows, full_rows[1:]))
    col_diffs = [int(b - a) for a, b in zip(full_cols, full_cols[1:])]
    sep_indices = [i for i, diff in enumerate(col_diffs) if diff == 1]
    assert len(sep_indices) == 1, "expected one double grey separator before the legend"
    sep_idx = sep_indices[0]
    assert sep_idx + 2 < len(full_cols), "legend boundary missing"

    row_slices = [(int(full_rows[i] + 1), int(full_rows[i + 1])) for i in range(len(full_rows) - 1)]
    main_col_slices = [
        (int(full_cols[i] + 1), int(full_cols[i + 1])) for i in range(sep_idx)
    ]
    legend_slice = (int(full_cols[sep_idx + 1] + 1), int(full_cols[sep_idx + 2]))

    assert row_pitch > 1 and col_pitch > 1
    assert row_slices and main_col_slices
    return row_slices, main_col_slices, legend_slice


def _main_region_colors(
    grid: np.ndarray, row_slices: list[tuple[int, int]], col_slices: list[tuple[int, int]]
) -> tuple[int, int]:
    values: list[int] = []
    for r0, r1 in row_slices:
        for c0, c1 in col_slices:
            values.extend(map(int, grid[r0:r1, c0:c1].ravel()))
    counts = Counter(values)
    bg = counts.most_common(1)[0][0]
    others = [color for color in counts if color != bg]
    assert len(others) == 1, "main board cells should contain only background and signal color"
    return int(bg), int(others[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cover each occupied board cell with the legend sample whose red-signal count matches that cell.

    Input structure:
    - The grid is partitioned by one-pixel grey lines into equal rectangular cells on a black background.
    - Main-board cells are empty or contain 1 to 3 red signal pixels on background.
    - A legend column on the far right is separated from the main board by a double grey vertical line.
    - Three legend sample cells contain 1, 2, and 3 red pixels plus a distinct colored fill pattern.

    Rule:
    - Read the legend by counting red pixels in each non-empty legend sample cell.
    - Use that count as a key from signal-count {1,2,3} to the entire sample-cell pattern.
    - For every non-empty main-board cell, count its red pixels and replace the whole cell with the matching legend sample.
    - Leave empty cells unchanged, and erase the legend interiors to background while keeping grey grid lines.

    Key insight:
    - The red pixels are not the output pattern themselves; their count is the hidden lookup key that selects which legend sample to copy.
    """
    g = np.asarray(grid, dtype=int)
    row_slices, main_col_slices, legend_slice = _row_col_slices(g)
    bg, signal = _main_region_colors(g, row_slices, main_col_slices)

    legend_patterns: dict[int, np.ndarray] = {}
    lc0, lc1 = legend_slice
    for r0, r1 in row_slices:
        patch = g[r0:r1, lc0:lc1]
        signal_count = int(np.count_nonzero(patch == signal))
        if signal_count:
            legend_patterns[signal_count] = patch.copy()

    assert set(legend_patterns) == {1, 2, 3}, "legend must define all three signal counts"

    out = g.copy()
    for r0, r1 in row_slices:
        out[r0:r1, lc0:lc1] = bg
        for c0, c1 in main_col_slices:
            patch = g[r0:r1, c0:c1]
            signal_count = int(np.count_nonzero(patch == signal))
            assert signal_count in (0, 1, 2, 3)
            if signal_count:
                out[r0:r1, c0:c1] = legend_patterns[signal_count]
    return out


def _pattern_masks(height: int, width: int) -> list[np.ndarray]:
    rr, cc = np.indices((height, width))
    masks = [
        np.ones((height, width), dtype=bool),
        rr % 2 == 0,
        cc % 2 == 0,
        (rr + cc) % 2 == 0,
        (rr == 0) | (rr == height - 1) | (cc == 0) | (cc == width - 1),
        (rr == height // 2) | (cc == width // 2),
        (rr % 2 == 0) & (cc % 2 == 0),
        (rr - cc) % 3 == 0,
        (rr + cc) % 3 == 0,
    ]
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        if int(mask.sum()) < 4:
            continue
        key = mask.tobytes()
        if key not in seen:
            unique.append(mask)
            seen.add(key)
    assert len(unique) >= 3
    return unique


def _sample_patch(
    height: int, width: int, fill_color: int, signal_color: int, count: int, mask: np.ndarray
) -> np.ndarray:
    patch = np.full((height, width), BG_COLOR, dtype=int)
    patch[mask] = fill_color
    coords = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(50):
        sample = patch.copy()
        for r, c in random.sample(coords, count):
            sample[r, c] = signal_color
        if np.any(sample == fill_color):
            return sample
    raise RuntimeError("failed to place legend signals without erasing the fill pattern")


def _valid_layouts() -> list[tuple[int, int, int, int]]:
    layouts: list[tuple[int, int, int, int]] = []
    for rows in range(3, 7):
        for cols in range(3, 7):
            for cell_h in range(3, 6):
                for cell_w in range(3, 6):
                    height = rows * cell_h + rows + 1
                    width = (cols + 1) * cell_w + cols + 3
                    if height <= 30 and width <= 30:
                        layouts.append((rows, cols, cell_h, cell_w))
    assert layouts
    return layouts


LAYOUTS = _valid_layouts()


def _gen_attempt() -> np.ndarray:
    rows, cols, cell_h, cell_w = random.choice(LAYOUTS)
    height = rows * cell_h + rows + 1
    width = (cols + 1) * cell_w + cols + 3

    row_bounds = [i * (cell_h + 1) for i in range(rows + 1)]
    main_col_bounds = [i * (cell_w + 1) for i in range(cols + 1)]
    sep_right = main_col_bounds[-1] + 1
    legend_right = sep_right + cell_w + 1

    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[row_bounds, :] = GRID_COLOR
    grid[:, main_col_bounds] = GRID_COLOR
    grid[:, sep_right] = GRID_COLOR
    grid[:, legend_right] = GRID_COLOR

    colors = [color for color in range(1, 10) if color not in (BG_COLOR, GRID_COLOR, SIGNAL_COLOR)]
    fill_colors = random.sample(colors, 3)
    masks = random.sample(_pattern_masks(cell_h, cell_w), 3)

    legend_rows = random.sample(range(rows), 3)
    counts = [1, 2, 3]
    random.shuffle(counts)

    for legend_row, count, fill_color, mask in zip(legend_rows, counts, fill_colors, masks):
        patch = _sample_patch(cell_h, cell_w, fill_color, SIGNAL_COLOR, count, mask)
        r0, r1 = row_bounds[legend_row] + 1, row_bounds[legend_row + 1]
        c0, c1 = sep_right + 1, legend_right
        grid[r0:r1, c0:c1] = patch

    total_cells = rows * cols
    cell_counts = [0, 1, 2, 3] + [random.randint(0, 3) for _ in range(total_cells - 4)]
    random.shuffle(cell_counts)

    coords = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    idx = 0
    for gr in range(rows):
        for gc in range(cols):
            count = cell_counts[idx]
            idx += 1
            if count == 0:
                continue
            r0, r1 = row_bounds[gr] + 1, row_bounds[gr + 1]
            c0, c1 = main_col_bounds[gc] + 1, main_col_bounds[gc + 1]
            for dr, dc in random.sample(coords, count):
                grid[r0 + dr, c0 + dc] = SIGNAL_COLOR

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

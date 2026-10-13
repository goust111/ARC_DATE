from __future__ import annotations

import numpy as np


PARTITION_COLORS = (1, 2, 3, 4, 5)
BAR_COLOR = 8
BACKGROUND = 0


def _bar_info(grid: np.ndarray) -> tuple[str, int, int, int, int]:
    coords = np.argwhere(grid == BAR_COLOR)
    assert len(coords) >= 5, "expected one teal bar of length at least 5"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        orientation = "H"
        row = int(rows[0])
        start = int(cols.min())
        end = int(cols.max())
        assert len(cols) == end - start + 1, "horizontal bar must be contiguous"
        length = int(len(cols))
        return orientation, row, start, end, (length - 1) // 2

    assert len(cols) == 1, "bar must be horizontal or vertical"
    orientation = "V"
    col = int(cols[0])
    start = int(rows.min())
    end = int(rows.max())
    assert len(rows) == end - start + 1, "vertical bar must be contiguous"
    length = int(len(rows))
    return orientation, col, start, end, (length - 1) // 2


def _compatible_line(a: np.ndarray, b: np.ndarray) -> bool:
    mask = (a != BAR_COLOR) & (b != BAR_COLOR)
    return bool(np.all(a[mask] == b[mask]))


def _partition_starts_by_rows(grid: np.ndarray) -> list[int]:
    starts = [0]
    for r in range(1, grid.shape[0]):
        if not _compatible_line(grid[r - 1], grid[r]):
            starts.append(r)
    return starts


def _partition_starts_by_cols(grid: np.ndarray) -> list[int]:
    starts = [0]
    for c in range(1, grid.shape[1]):
        if not _compatible_line(grid[:, c - 1], grid[:, c]):
            starts.append(c)
    return starts


def _recover_cell_color(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    values = set(map(int, np.unique(grid[r0:r1, c0:c1])))
    values.discard(BAR_COLOR)
    assert len(values) == 1, "each partition cell must have one underlying color"
    return next(iter(values))


def _partition_color_grid(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    row_starts = _partition_starts_by_rows(grid)
    col_starts = _partition_starts_by_cols(grid)
    row_ends = row_starts[1:] + [grid.shape[0]]
    col_ends = col_starts[1:] + [grid.shape[1]]

    colors = np.zeros((len(row_starts), len(col_starts)), dtype=int)
    for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
        for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
            colors[i, j] = _recover_cell_color(grid, r0, r1, c0, c1)
    return colors, row_starts, col_starts


def _distance_to_bar(
    orientation: str, fixed: int, start: int, end: int, r: int, c: int
) -> int:
    if orientation == "H":
        return abs(r - fixed) + max(start - c, 0, c - end)
    return abs(c - fixed) + max(start - r, 0, r - end)


def _selected_sample_mask(
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    radius: int,
    row_starts: list[int],
    col_starts: list[int],
) -> np.ndarray:
    selected = np.zeros((len(row_starts), len(col_starts)), dtype=bool)
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            selected[i, j] = _distance_to_bar(orientation, fixed, start, end, r, c) <= radius
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downscale only the rectangular partitions whose sample points fall inside the teal bar's diamond envelope.

    Input structure
    - The grid is fully tiled by monochrome rectangular partitions forming a hidden row-by-column partition grid.
    - A single teal horizontal or vertical one-pixel-thick bar is drawn on top of those partitions and may cover some partition sample points.
    - Partition colors come from a small visible palette and teal is used only by the overlaid bar.

    Rule
    - Detect the teal bar and treat its pixels as transparent when reconstructing the rectangular partition grid.
    - Use each partition's upper-left pixel as its sample point, and select exactly those sample points whose Manhattan distance to the bar is at most floor((L-1)/2), where L is the bar length.
    - Reduce every selected partition to one pixel, preserve the selected partitions' relative row/column arrangement, and crop to the selected sample-point bbox.

    Key insight
    - The bar length determines a diamond-shaped selection mask, while the hidden rectangular partition grid supplies the recoverable colors to sample beneath the teal overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orientation, fixed, start, end, radius = _bar_info(g)
    colors, row_starts, col_starts = _partition_color_grid(g)
    selected = _selected_sample_mask(orientation, fixed, start, end, radius, row_starts, col_starts)

    assert selected.any(), "expected at least one selected partition"
    rows, cols = np.argwhere(selected).T
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BACKGROUND, dtype=int)
    window = selected[r0 : r1 + 1, c0 : c1 + 1]
    out[window] = colors[r0 : r1 + 1, c0 : c1 + 1][window]
    return out


GEN_ATTEMPTS = 400
ROW_COUNT_RANGE = (4, 7)
COL_COUNT_RANGE = (4, 7)
BAND_SIZE_RANGE = (2, 5)
BAR_LENGTH_RANGE = (5, 12)


def _random_band_sizes(count: int) -> list[int]:
    return [int(np.random.randint(*BAND_SIZE_RANGE)) for _ in range(count)]


def _random_color_matrix(n_rows: int, n_cols: int) -> np.ndarray:
    colors = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(n_rows):
        for j in range(n_cols):
            blocked = set()
            if i > 0:
                blocked.add(int(colors[i - 1, j]))
            if j > 0:
                blocked.add(int(colors[i, j - 1]))
            choices = [color for color in PARTITION_COLORS if color not in blocked]
            colors[i, j] = int(np.random.choice(choices))
    return colors


def _paint_partition_grid(row_sizes: list[int], col_sizes: list[int], colors: np.ndarray) -> np.ndarray:
    height = sum(row_sizes)
    width = sum(col_sizes)
    grid = np.zeros((height, width), dtype=int)

    row_starts = np.cumsum([0] + row_sizes[:-1]).tolist()
    col_starts = np.cumsum([0] + col_sizes[:-1]).tolist()
    for i, r0 in enumerate(row_starts):
        r1 = r0 + row_sizes[i]
        for j, c0 in enumerate(col_starts):
            c1 = c0 + col_sizes[j]
            grid[r0:r1, c0:c1] = int(colors[i, j])
    return grid


def _crossed_partition_count(
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    row_starts: list[int],
    row_sizes: list[int],
    col_starts: list[int],
    col_sizes: list[int],
) -> int:
    hits: set[tuple[int, int]] = set()
    if orientation == "H":
        row_idx = next(i for i, (r0, h) in enumerate(zip(row_starts, row_sizes)) if r0 <= fixed < r0 + h)
        for j, (c0, w) in enumerate(zip(col_starts, col_sizes)):
            if max(start, c0) <= min(end, c0 + w - 1):
                hits.add((row_idx, j))
    else:
        col_idx = next(j for j, (c0, w) in enumerate(zip(col_starts, col_sizes)) if c0 <= fixed < c0 + w)
        for i, (r0, h) in enumerate(zip(row_starts, row_sizes)):
            if max(start, r0) <= min(end, r0 + h - 1):
                hits.add((i, col_idx))
    return len(hits)


def _candidate_bars(
    base: np.ndarray,
    partition_colors: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    height, width = base.shape
    row_starts = np.cumsum([0] + row_sizes[:-1]).tolist()
    col_starts = np.cumsum([0] + col_sizes[:-1]).tolist()
    total_cells = partition_colors.size
    candidates: list[tuple[np.ndarray, np.ndarray, bool]] = []

    max_h_len = min(BAR_LENGTH_RANGE[1] - 1, width)
    for length in range(BAR_LENGTH_RANGE[0], max_h_len + 1):
        radius = (length - 1) // 2
        for row in range(height):
            for c0 in range(width - length + 1):
                c1 = c0 + length - 1
                if c0 == 0 and c1 == width - 1:
                    continue
                if _crossed_partition_count("H", row, c0, c1, row_starts, row_sizes, col_starts, col_sizes) < 2:
                    continue
                selected = _selected_sample_mask("H", row, c0, c1, radius, row_starts, col_starts)
                inside = int(selected.sum())
                outside = int(total_cells - inside)
                if inside < 3 or outside < 2:
                    continue
                if len(np.unique(np.argwhere(selected)[:, 0])) < 2 or len(np.unique(np.argwhere(selected)[:, 1])) < 2:
                    continue
                rows, cols = np.argwhere(selected).T
                bbox_area = (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1)
                if bbox_area > inside + 4:
                    continue
                grid = base.copy()
                grid[row, c0 : c1 + 1] = BAR_COLOR
                expected = np.full(partition_colors.shape, BACKGROUND, dtype=int)
                expected[selected] = partition_colors[selected]
                expected = expected[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
                covers_sample = any(row == r and c0 <= c <= c1 for r in row_starts for c in col_starts)
                candidates.append((grid, expected, covers_sample))

    max_v_len = min(BAR_LENGTH_RANGE[1] - 1, height)
    for length in range(BAR_LENGTH_RANGE[0], max_v_len + 1):
        radius = (length - 1) // 2
        for col in range(width):
            for r0 in range(height - length + 1):
                r1 = r0 + length - 1
                if r0 == 0 and r1 == height - 1:
                    continue
                if _crossed_partition_count("V", col, r0, r1, row_starts, row_sizes, col_starts, col_sizes) < 2:
                    continue
                selected = _selected_sample_mask("V", col, r0, r1, radius, row_starts, col_starts)
                inside = int(selected.sum())
                outside = int(total_cells - inside)
                if inside < 3 or outside < 2:
                    continue
                if len(np.unique(np.argwhere(selected)[:, 0])) < 2 or len(np.unique(np.argwhere(selected)[:, 1])) < 2:
                    continue
                rows, cols = np.argwhere(selected).T
                bbox_area = (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1)
                if bbox_area > inside + 4:
                    continue
                grid = base.copy()
                grid[r0 : r1 + 1, col] = BAR_COLOR
                expected = np.full(partition_colors.shape, BACKGROUND, dtype=int)
                expected[selected] = partition_colors[selected]
                expected = expected[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
                covers_sample = any(col == c and r0 <= r <= r1 for r in row_starts for c in col_starts)
                candidates.append((grid, expected, covers_sample))

    return candidates


def _gen_attempt() -> np.ndarray | None:
    row_count = int(np.random.randint(*ROW_COUNT_RANGE))
    col_count = int(np.random.randint(*COL_COUNT_RANGE))
    row_sizes = _random_band_sizes(row_count)
    col_sizes = _random_band_sizes(col_count)

    partition_colors = _random_color_matrix(row_count, col_count)
    base = _paint_partition_grid(row_sizes, col_sizes, partition_colors)
    candidates = _candidate_bars(base, partition_colors, row_sizes, col_sizes)
    if not candidates:
        return None

    sample_covering = [candidate for candidate in candidates if candidate[2]]
    if sample_covering and np.random.rand() < 0.6:
        candidates = sample_covering

    grid, expected, _covers_sample = candidates[int(np.random.randint(len(candidates)))]
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")

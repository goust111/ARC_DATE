from __future__ import annotations

from collections import Counter, deque
from itertools import combinations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _line_runs_1d(values: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    target = int(color)
    for idx, value in enumerate(np.asarray(values, dtype=int)):
        if int(value) == target:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _max_run_len(values: np.ndarray, color: int) -> int:
    runs = _line_runs_1d(values, color)
    return max((stop - start for start, stop in runs), default=0)


def _partition_intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        intervals.append((start, sep))
        start = sep + 1
    intervals.append((start, int(size)))
    return intervals


def _row_supported(grid: np.ndarray, color: int, row: int, sep_cols: list[int]) -> bool:
    for c0, c1 in _partition_intervals(grid.shape[1], sep_cols):
        if c0 >= c1 or not np.any(grid[int(row), c0:c1] == int(color)):
            return False
    return True


def _col_supported(grid: np.ndarray, color: int, col: int, sep_rows: list[int]) -> bool:
    for r0, r1 in _partition_intervals(grid.shape[0], sep_rows):
        if r0 >= r1 or not np.any(grid[r0:r1, int(col)] == int(color)):
            return False
    return True


def _initial_row_candidates(grid: np.ndarray, color: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        runs = _line_runs_1d(grid[r], color)
        if len(runs) >= 2 and max(stop - start for start, stop in runs) >= 2:
            rows.append(r)
    return rows


def _initial_col_candidates(grid: np.ndarray, color: int) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        runs = _line_runs_1d(grid[:, c], color)
        if len(runs) >= 2 and max(stop - start for start, stop in runs) >= 2:
            cols.append(c)
    return cols


def _recover_line_grid(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = _initial_row_candidates(grid, color)
    cols = _initial_col_candidates(grid, color)
    if not rows or not cols:
        return [], []

    while True:
        new_rows = [r for r in rows if _row_supported(grid, color, r, cols)]
        new_cols = [c for c in cols if _col_supported(grid, color, c, new_rows)]
        if new_rows == rows and new_cols == cols:
            break
        rows, cols = new_rows, new_cols
        if not rows or not cols:
            return [], []

    if not all(_row_supported(grid, color, r, cols) for r in rows):
        return [], []
    if not all(_col_supported(grid, color, c, rows) for c in cols):
        return [], []
    return rows, cols


def _extract_output(grid: np.ndarray, bg: int, sep_rows: list[int], sep_cols: list[int]) -> np.ndarray:
    row_intervals = _partition_intervals(grid.shape[0], sep_rows)
    col_intervals = _partition_intervals(grid.shape[1], sep_cols)
    out = np.full((len(row_intervals), len(col_intervals)), int(bg), dtype=int)
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            patch = grid[r0:r1, c0:c1]
            colors = sorted(set(map(int, patch.ravel())) - {int(bg)})
            assert len(colors) <= 1, "each partition must be empty or contain one monochrome object"
            if colors:
                out[i, j] = colors[0]
    return out


def _solution_candidates(grid: np.ndarray) -> list[tuple[int, list[int], list[int], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[tuple[int, list[int], list[int], np.ndarray]] = []
    for color in _colors_present(g, exclude={bg}):
        rows, cols = _recover_line_grid(g, color)
        if not rows or not cols:
            continue
        try:
            out = _extract_output(g, bg, rows, cols)
        except AssertionError:
            continue
        candidates.append((int(color), rows, cols, out))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the broken separator mesh and compress each partition into a one-cell map.

    Input structure
    - The background is the dominant color.
    - One line color appears as scattered horizontal and vertical short segments; some collinear fragments belong to hidden full-width or full-height separators, while some same-color short segments are ordinary partition contents.
    - After the hidden separators are restored, each partition interior contains at most one non-background object: either a monochrome mark of another color or a leftover short segment of the line color.

    Rule
    - Identify the unique color whose fragmented rows and columns can both be completed into spanning separators.
    - Keep only those rows and columns whose fragments are supported in every strip between the opposite separators, then treat them as the restored separator mesh.
    - Output one cell per partition, colored by the partition's non-background content; empty partitions become the background color.

    Key insight
    - A same-color short segment is separator material only when it participates in a row or column that can be justified as a full spanning divider across every reconstructed strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    candidates = _solution_candidates(g)
    assert len(candidates) == 1, "expected a unique reconstructable separator color"
    return candidates[0][3]


MAX_GENERATION_ATTEMPTS = 400
GRID_COLOR_POOL = tuple(range(10))
PARTITION_COUNT_RANGE = (2, 4)
PARTITION_SIZE_RANGE = (4, 6)
MAX_EXTRA_OCCUPIED = 2
LINE_OBJECT_COUNT_CAP = 3
LINE_SEGMENT_LEN_RANGE = (2, 3)
BLOB_SIZE_RANGE = (1, 4)


def _band_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        bands.append((pos, pos + int(size)))
        pos += int(size)
        if idx + 1 < len(sizes):
            separators.append(pos)
            pos += 1
    return bands, separators, pos


def _choose_sizes(count: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.integers(PARTITION_SIZE_RANGE[0], PARTITION_SIZE_RANGE[1] + 1)) for _ in range(count)]


def _random_connected_blob(height: int, width: int, size: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    cells = {start}
    frontier = {nbr for nbr in _neighbors4(*start) if 0 <= nbr[0] < height and 0 <= nbr[1] < width}
    while len(cells) < size:
        assert frontier
        cell = list(frontier)[int(rng.integers(len(frontier)))]
        frontier.remove(cell)
        cells.add(cell)
        mask[cell] = True
        for nbr in _neighbors4(*cell):
            rr, cc = nbr
            if 0 <= rr < height and 0 <= cc < width and nbr not in cells:
                frontier.add(nbr)
    return mask


def _count_components(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        count += 1
        dq: deque[tuple[int, int]] = deque([(int(r), int(c))])
        seen[r, c] = True
        while dq:
            rr, cc = dq.popleft()
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    dq.append((nr, nc))
    return count


def _region_semantics_ok(grid: np.ndarray, bg: int, sep_rows: list[int], sep_cols: list[int]) -> bool:
    for r0, r1 in _partition_intervals(grid.shape[0], sep_rows):
        for c0, c1 in _partition_intervals(grid.shape[1], sep_cols):
            patch = grid[r0:r1, c0:c1]
            fg = patch != int(bg)
            if not fg.any():
                continue
            if len(set(map(int, patch[fg].ravel()))) != 1:
                return False
            if _count_components(fg) > 1:
                return False
    return True


def _enumerate_valid_interpretations(
    grid: np.ndarray,
    bg: int,
    line_color: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]]:
    candidate_rows = [r for r in range(grid.shape[0]) if _max_run_len(grid[r], line_color) >= 2]
    candidate_cols = [c for c in range(grid.shape[1]) if _max_run_len(grid[:, c], line_color) >= 2]
    valid: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]] = []

    row_subsets = []
    for size in range(1, len(candidate_rows) + 1):
        row_subsets.extend(combinations(candidate_rows, size))
    col_subsets = []
    for size in range(1, len(candidate_cols) + 1):
        col_subsets.extend(combinations(candidate_cols, size))

    for rows in row_subsets:
        row_list = list(rows)
        for cols in col_subsets:
            col_list = list(cols)
            if not all(_row_supported(grid, line_color, r, col_list) for r in row_list):
                continue
            if not all(_col_supported(grid, line_color, c, row_list) for c in col_list):
                continue
            if not _region_semantics_ok(grid, bg, row_list, col_list):
                continue
            valid.append((rows, cols, _extract_output(grid, bg, row_list, col_list)))
    return valid


def _choose_output_cells(rows: int, cols: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    anchor_r = int(rng.integers(rows))
    anchor_c = int(rng.integers(cols))
    occupied = {(r, anchor_c) for r in range(rows)} | {(anchor_r, c) for c in range(cols)}
    empties = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in occupied]
    if len(empties) > 1:
        extra_cap = min(MAX_EXTRA_OCCUPIED, len(empties) - 1)
        extra_count = int(rng.integers(extra_cap + 1))
        if extra_count:
            chosen = rng.choice(len(empties), size=extra_count, replace=False)
            for idx in np.atleast_1d(chosen):
                occupied.add(empties[int(idx)])
    return occupied


def _line_row_choices(
    band: tuple[int, int],
    forbidden_rows: set[int],
    blocked_by_vertical_skeleton: set[int],
) -> list[int]:
    r0, r1 = band
    return [
        r
        for r in range(r0 + 1, r1 - 1)
        if r not in forbidden_rows and r not in blocked_by_vertical_skeleton
    ]


def _line_col_choices(
    band: tuple[int, int],
    forbidden_cols: set[int],
    blocked_by_horizontal_skeleton: set[int],
) -> list[int]:
    c0, c1 = band
    return [
        c
        for c in range(c0 + 1, c1 - 1)
        if c not in forbidden_cols and c not in blocked_by_horizontal_skeleton
    ]


def _place_line_object(
    grid: np.ndarray,
    band_r: tuple[int, int],
    band_c: tuple[int, int],
    line_color: int,
    row_choices: list[int],
    col_choices: list[int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    rng: np.random.Generator,
) -> bool:
    options: list[str] = []
    width = band_c[1] - band_c[0]
    height = band_r[1] - band_r[0]
    if row_choices and width >= 4:
        options.append("h")
    if col_choices and height >= 4:
        options.append("v")
    if not options:
        return False

    orient = options[int(rng.integers(len(options)))]
    if orient == "h":
        row = row_choices[int(rng.integers(len(row_choices)))]
        max_len = min(LINE_SEGMENT_LEN_RANGE[1], width - 2)
        seg_len = int(rng.integers(LINE_SEGMENT_LEN_RANGE[0], max_len + 1))
        start_min = band_c[0] + 1
        start_max = band_c[1] - 1 - seg_len
        start = int(rng.integers(start_min, start_max + 1))
        grid[row, start : start + seg_len] = int(line_color)
        forbidden_rows.add(row)
        forbidden_cols.update(range(start, start + seg_len))
        return True

    col = col_choices[int(rng.integers(len(col_choices)))]
    max_len = min(LINE_SEGMENT_LEN_RANGE[1], height - 2)
    seg_len = int(rng.integers(LINE_SEGMENT_LEN_RANGE[0], max_len + 1))
    start_min = band_r[0] + 1
    start_max = band_r[1] - 1 - seg_len
    start = int(rng.integers(start_min, start_max + 1))
    grid[start : start + seg_len, col] = int(line_color)
    forbidden_rows.update(range(start, start + seg_len))
    forbidden_cols.add(col)
    return True


def _place_blob_object(
    grid: np.ndarray,
    band_r: tuple[int, int],
    band_c: tuple[int, int],
    color: int,
    rng: np.random.Generator,
) -> bool:
    inner_h = band_r[1] - band_r[0] - 2
    inner_w = band_c[1] - band_c[0] - 2
    if inner_h <= 0 or inner_w <= 0:
        return False
    max_size = min(BLOB_SIZE_RANGE[1], inner_h * inner_w)
    size = int(rng.integers(BLOB_SIZE_RANGE[0], max_size + 1))
    mask = _random_connected_blob(inner_h, inner_w, size, rng)
    patch = grid[band_r[0] + 1 : band_r[1] - 1, band_c[0] + 1 : band_c[1] - 1]
    if np.any(patch[mask] != _get_bg(grid)):
        return False
    patch[mask] = int(color)
    return True


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    line_color: int,
    sep_rows: list[int],
    sep_cols: list[int],
) -> bool:
    bg = _get_bg(grid)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    candidates = _solution_candidates(grid)
    if len(candidates) != 1:
        return False
    cand_color, cand_rows, cand_cols, cand_out = candidates[0]
    if cand_color != int(line_color):
        return False
    if cand_rows != list(sep_rows) or cand_cols != list(sep_cols):
        return False
    if not np.array_equal(cand_out, expected):
        return False

    if not all(_row_supported(grid, line_color, r, sep_cols) for r in sep_rows):
        return False
    if not all(_col_supported(grid, line_color, c, sep_rows) for c in sep_cols):
        return False

    for r in range(grid.shape[0]):
        supported = _row_supported(grid, line_color, r, sep_cols)
        if supported != (r in set(sep_rows)):
            return False
    for c in range(grid.shape[1]):
        supported = _col_supported(grid, line_color, c, sep_rows)
        if supported != (c in set(sep_cols)):
            return False

    valid = _enumerate_valid_interpretations(grid, bg, line_color)
    if len(valid) != 1:
        return False
    only_rows, only_cols, only_out = valid[0]
    if only_rows != tuple(sep_rows) or only_cols != tuple(sep_cols):
        return False
    if not np.array_equal(only_out, expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    out_rows = int(rng.integers(PARTITION_COUNT_RANGE[0], PARTITION_COUNT_RANGE[1] + 1))
    out_cols = int(rng.integers(PARTITION_COUNT_RANGE[0], PARTITION_COUNT_RANGE[1] + 1))

    row_sizes = _choose_sizes(out_rows, rng)
    col_sizes = _choose_sizes(out_cols, rng)
    row_bands, sep_rows, height = _band_layout(row_sizes)
    col_bands, sep_cols, width = _band_layout(col_sizes)
    if height > 30 or width > 30:
        return None

    bg = int(rng.integers(len(GRID_COLOR_POOL)))
    usable_colors = [c for c in GRID_COLOR_POOL if c != bg]
    rng.shuffle(usable_colors)
    line_color = int(usable_colors[0])
    blob_palette = [int(c) for c in usable_colors[1:]]

    grid = np.full((height, width), bg, dtype=int)

    vertical_block_rows: list[set[int]] = []
    for r0, r1 in row_bands:
        local_start = int(rng.integers(1, (r1 - r0) - 2))
        rows = {r0 + local_start, r0 + local_start + 1}
        vertical_block_rows.append(rows)
    horizontal_block_cols: list[set[int]] = []
    for c0, c1 in col_bands:
        local_start = int(rng.integers(1, (c1 - c0) - 2))
        cols = {c0 + local_start, c0 + local_start + 1}
        horizontal_block_cols.append(cols)

    for row in sep_rows:
        for band_idx, (c0, _c1) in enumerate(col_bands):
            cols = sorted(horizontal_block_cols[band_idx])
            grid[row, cols[0] : cols[-1] + 1] = line_color
    for col in sep_cols:
        for band_idx, (r0, _r1) in enumerate(row_bands):
            rows = sorted(vertical_block_rows[band_idx])
            grid[rows[0] : rows[-1] + 1, col] = line_color

    occupied = sorted(_choose_output_cells(out_rows, out_cols, rng))
    if len(occupied) >= out_rows * out_cols:
        return None

    line_count = int(rng.integers(1, min(LINE_OBJECT_COUNT_CAP, len(occupied) - 1) + 1))
    line_indices = set(map(int, rng.choice(len(occupied), size=line_count, replace=False)))
    if len(blob_palette) < len(occupied) - line_count:
        return None

    expected = np.full((out_rows, out_cols), bg, dtype=int)
    forbidden_rows: set[int] = set()
    forbidden_cols: set[int] = set()
    blob_color_idx = 0
    placed_line = 0
    placed_blob = 0

    for idx, (out_r, out_c) in enumerate(occupied):
        band_r = row_bands[out_r]
        band_c = col_bands[out_c]
        if idx in line_indices:
            row_choices = _line_row_choices(band_r, forbidden_rows, vertical_block_rows[out_r])
            col_choices = _line_col_choices(band_c, forbidden_cols, horizontal_block_cols[out_c])
            if not _place_line_object(
                grid,
                band_r,
                band_c,
                line_color,
                row_choices,
                col_choices,
                forbidden_rows,
                forbidden_cols,
                rng,
            ):
                return None
            expected[out_r, out_c] = line_color
            placed_line += 1
            continue

        color = blob_palette[blob_color_idx]
        blob_color_idx += 1
        if not _place_blob_object(grid, band_r, band_c, color, rng):
            return None
        expected[out_r, out_c] = int(color)
        placed_blob += 1

    if placed_line < 1 or placed_blob < 1 or np.count_nonzero(expected == bg) < 1:
        return None
    if np.max(grid.shape) > 30 or np.max(expected.shape) > 30:
        return None
    if not _semantic_ok(grid, expected, line_color, sep_rows, sep_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

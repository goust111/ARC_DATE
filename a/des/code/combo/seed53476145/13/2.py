from __future__ import annotations

import random

import numpy as np


PALETTE = list(range(10))
GRID_HEIGHT_RANGE = (6, 12)
GRID_WIDTH_RANGE = (6, 12)
PALETTE_SIZE_RANGE = (4, 7)
CORE_COLOR_COUNT_RANGE = (2, 4)
FILL_MODES = ("checker", "column_stripes", "row_stripes", "split")
MAX_GENERATION_ATTEMPTS = 300


def _repair_data(
    data: np.ndarray, row_headers: np.ndarray, col_headers: np.ndarray
) -> np.ndarray:
    row_colors = np.broadcast_to(row_headers[:, None], data.shape)
    col_colors = np.broadcast_to(col_headers[None, :], data.shape)

    repaired = data.copy()
    same = row_colors == col_colors
    repaired[same] = row_colors[same]

    invalid = (~same) & (data != row_colors) & (data != col_colors)
    repaired[invalid] = row_colors[invalid]
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the first row and first column as row/column check colors.

    Interior cells are valid if they match their row header or their column
    header. Cells that match neither are repaired to the row header color.
    When the row and column headers agree, the cell must be that shared color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 2 and g.shape[1] >= 2

    out = g.copy()
    out[1:, 1:] = _repair_data(g[1:, 1:], g[1:, 0], g[0, 1:])
    return out


def _assemble_grid(
    data: np.ndarray, row_headers: np.ndarray, col_headers: np.ndarray, corner_color: int
) -> np.ndarray:
    grid = np.empty((data.shape[0] + 1, data.shape[1] + 1), dtype=int)
    grid[0, 0] = int(corner_color)
    grid[0, 1:] = col_headers
    grid[1:, 0] = row_headers
    grid[1:, 1:] = data
    return grid


def _classify_cells(
    data: np.ndarray, row_headers: np.ndarray, col_headers: np.ndarray
) -> dict[str, list[tuple[int, int]]]:
    buckets = {"same": [], "row_only": [], "col_only": [], "error": []}
    for r in range(data.shape[0]):
        row_color = int(row_headers[r])
        for c in range(data.shape[1]):
            col_color = int(col_headers[c])
            value = int(data[r, c])
            if row_color == col_color:
                buckets["same"].append((r, c))
            elif value == row_color:
                buckets["row_only"].append((r, c))
            elif value == col_color:
                buckets["col_only"].append((r, c))
            else:
                buckets["error"].append((r, c))
    return buckets


def _sample_headers(
    data_rows: int, data_cols: int, core_colors: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        row_headers = np.array([random.choice(core_colors) for _ in range(data_rows)], dtype=int)
        col_headers = np.array([random.choice(core_colors) for _ in range(data_cols)], dtype=int)
        same_mask = row_headers[:, None] == col_headers[None, :]
        mismatch_count = int((~same_mask).sum())
        if same_mask.any() and mismatch_count >= 3 and len(set(row_headers.tolist() + col_headers.tolist())) >= 2:
            return row_headers, col_headers
    return None


def _fill_clean_data(row_headers: np.ndarray, col_headers: np.ndarray) -> np.ndarray:
    data_rows = len(row_headers)
    data_cols = len(col_headers)
    data = np.empty((data_rows, data_cols), dtype=int)

    mode = random.choice(FILL_MODES)
    phase = random.randint(0, 1)
    split_row = random.randint(1, data_rows - 1)
    split_col = random.randint(1, data_cols - 1)
    stripe_cols = set(random.sample(range(data_cols), random.randint(1, data_cols - 1)))
    stripe_rows = set(random.sample(range(data_rows), random.randint(1, data_rows - 1)))

    for r in range(data_rows):
        row_color = int(row_headers[r])
        for c in range(data_cols):
            col_color = int(col_headers[c])
            if row_color == col_color:
                data[r, c] = row_color
                continue

            if mode == "checker":
                use_row = (r + c + phase) % 2 == 0
            elif mode == "column_stripes":
                use_row = c not in stripe_cols
            elif mode == "row_stripes":
                use_row = r in stripe_rows
            else:
                if random.random() < 0.5:
                    use_row = c < split_col
                else:
                    use_row = r >= split_row
            data[r, c] = row_color if use_row else col_color

    mismatch_coords = [
        (r, c)
        for r in range(data_rows)
        for c in range(data_cols)
        if int(row_headers[r]) != int(col_headers[c])
    ]
    classes = _classify_cells(data, row_headers, col_headers)
    missing_row = not classes["row_only"]
    missing_col = not classes["col_only"]
    if missing_row or missing_col:
        random.shuffle(mismatch_coords)
        if missing_row:
            r, c = mismatch_coords.pop()
            data[r, c] = int(row_headers[r])
        if missing_col:
            r, c = mismatch_coords.pop()
            data[r, c] = int(col_headers[c])

    return data


def _choose_error_color(
    palette: list[int], row_color: int, col_color: int
) -> int:
    choices = [color for color in palette if color not in {row_color, col_color}]
    assert choices
    return int(random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    data_rows = height - 1
    data_cols = width - 1

    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(PALETTE, palette_size)
    max_core = min(CORE_COLOR_COUNT_RANGE[1], palette_size - 1)
    core_count = random.randint(CORE_COLOR_COUNT_RANGE[0], max_core)
    core_colors = random.sample(palette, core_count)

    sampled = _sample_headers(data_rows, data_cols, core_colors)
    if sampled is None:
        return None
    row_headers, col_headers = sampled

    clean_data = _fill_clean_data(row_headers, col_headers)
    classes = _classify_cells(clean_data, row_headers, col_headers)
    if not classes["same"] or not classes["row_only"] or not classes["col_only"]:
        return None

    protected = {
        random.choice(classes["same"]),
        random.choice(classes["row_only"]),
        random.choice(classes["col_only"]),
    }
    all_coords = [(r, c) for r in range(data_rows) for c in range(data_cols)]
    mismatch_candidates = [
        (r, c)
        for r, c in all_coords
        if (r, c) not in protected and int(row_headers[r]) != int(col_headers[c])
    ]
    if not mismatch_candidates:
        return None
    equal_candidates = [
        (r, c)
        for r, c in all_coords
        if (r, c) not in protected and int(row_headers[r]) == int(col_headers[c])
    ]

    error_coords = {random.choice(mismatch_candidates)}
    if equal_candidates and random.random() < 0.7:
        error_coords.add(random.choice(equal_candidates))

    remaining_candidates = [coord for coord in all_coords if coord not in protected and coord not in error_coords]
    max_errors = min(len(all_coords) - len(protected), max(2, clean_data.size // 4))
    target_error_count = random.randint(len(error_coords), max_errors)
    extra_needed = target_error_count - len(error_coords)
    if extra_needed > 0:
        error_coords.update(random.sample(remaining_candidates, extra_needed))

    input_data = clean_data.copy()
    for r, c in error_coords:
        input_data[r, c] = _choose_error_color(
            palette, int(row_headers[r]), int(col_headers[c])
        )

    classes_in = _classify_cells(input_data, row_headers, col_headers)
    if not classes_in["same"] or not classes_in["row_only"] or not classes_in["col_only"] or not classes_in["error"]:
        return None

    corner_color = random.choice(palette)
    inp = _assemble_grid(input_data, row_headers, col_headers, corner_color)
    expected = _assemble_grid(clean_data, row_headers, col_headers, corner_color)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

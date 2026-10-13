from __future__ import annotations

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
SEPARATOR = 3
YELLOW = 4
BLOCK = 5
ORANGE = 7

BLOCK_SIZE = 2
GUTTER = 1

FULL_ROW_RANGE = (3, 5)
FULL_COL_RANGE = (3, 4)
MARGIN_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _true_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = int(len(flags))
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]):
            j += 1
        intervals.append((int(i), int(j)))
        i = j + 1
    return intervals


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if bbox is None:
        return out
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _find_separator_column(grid: np.ndarray, bg: int) -> int:
    height, width = map(int, grid.shape)
    assert width >= 3 and width % 2 == 1

    candidates: list[int] = []
    for col in range(1, width - 1):
        color = int(grid[0, col])
        if color == bg:
            continue
        if not np.all(grid[:, col] == color):
            continue
        if col != width - col - 1:
            continue
        remainder = np.concatenate((grid[:, :col], grid[:, col + 1 :]), axis=1)
        if np.any(remainder == color):
            continue
        candidates.append(col)

    assert len(candidates) == 1, "expected one unique separator column"
    return candidates[0]


def _band_layout(visible_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _true_intervals(visible_mask.any(axis=1))
    col_bands = _true_intervals(visible_mask.any(axis=0))

    assert len(row_bands) >= 2 and len(col_bands) >= 2
    assert all(end - start + 1 == BLOCK_SIZE for start, end in row_bands)
    assert all(end - start + 1 == BLOCK_SIZE for start, end in col_bands)

    for (prev_start, prev_end), (next_start, next_end) in zip(row_bands, row_bands[1:]):
        assert next_start - prev_end - 1 == GUTTER, (prev_start, prev_end, next_start, next_end)
    for (prev_start, prev_end), (next_start, next_end) in zip(col_bands, col_bands[1:]):
        assert next_start - prev_end - 1 == GUTTER, (prev_start, prev_end, next_start, next_end)

    return row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two block panels and color the surviving lattice gutters by the comparison-driven bbox.

    Input structure:
    - The grid has two equal-width black panels separated by one solid green divider column.
    - Each panel contains aligned grey 2x2 blocks on a regular lattice with 1-cell gutters between block bands.
    - Some block positions are missing in one panel, the other panel, or both, while the visible lattice geometry remains aligned.

    Rule:
    - Remove the divider and compare the left and right panels block-by-block.
    - Keep shared grey blocks as grey, recolor left-only blocks red, recolor right-only blocks blue, and leave doubly-missing block positions black.
    - Find the bbox of all surviving output blocks, then paint gutter pixels yellow inside that bbox and orange on the same gutter rows or columns outside the bbox.

    Key insight:
    - The panel comparison determines which block bands remain visible, and that surviving block bbox is what decides which lattice gutters count as interior versus exterior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_col = _find_separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape

    visible_mask = (left == BLOCK) | (right == BLOCK)
    row_bands, col_bands = _band_layout(visible_mask)

    out = np.full(left.shape, bg, dtype=int)
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            left_cell = left[r0 : r1 + 1, c0 : c1 + 1]
            right_cell = right[r0 : r1 + 1, c0 : c1 + 1]
            assert left_cell.shape == (BLOCK_SIZE, BLOCK_SIZE)
            assert right_cell.shape == (BLOCK_SIZE, BLOCK_SIZE)

            left_has = bool(np.all(left_cell == BLOCK))
            right_has = bool(np.all(right_cell == BLOCK))

            assert left_has or bool(np.all(left_cell == bg))
            assert right_has or bool(np.all(right_cell == bg))

            if left_has and right_has:
                out[r0 : r1 + 1, c0 : c1 + 1] = BLOCK
            elif left_has:
                out[r0 : r1 + 1, c0 : c1 + 1] = RED
            elif right_has:
                out[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    block_mask = out != bg
    block_bbox = _bbox(block_mask)
    assert block_bbox is not None
    inside_bbox = _bbox_mask(out.shape, block_bbox)

    row_gutters = np.zeros(out.shape[0], dtype=bool)
    col_gutters = np.zeros(out.shape[1], dtype=bool)
    for (_prev_start, prev_end), (next_start, _next_end) in zip(row_bands, row_bands[1:]):
        row_gutters[prev_end + 1 : next_start] = True
    for (_prev_start, prev_end), (next_start, _next_end) in zip(col_bands, col_bands[1:]):
        col_gutters[prev_end + 1 : next_start] = True

    gutter_mask = (row_gutters[:, None] | col_gutters[None, :]) & (out == bg)
    out[gutter_mask & inside_bbox] = YELLOW
    out[gutter_mask & ~inside_bbox] = ORANGE
    return out


def _panel_extent(band_count: int) -> int:
    return band_count * BLOCK_SIZE + (band_count - 1) * GUTTER


def _render_panel(occupancy: np.ndarray, top: int, left: int) -> np.ndarray:
    rows, cols = occupancy.shape
    height = top + _panel_extent(rows)
    width = left + _panel_extent(cols)
    panel = np.full((height, width), BACKGROUND, dtype=int)

    for rr in range(rows):
        r0 = top + rr * (BLOCK_SIZE + GUTTER)
        for cc in range(cols):
            if not occupancy[rr, cc]:
                continue
            c0 = left + cc * (BLOCK_SIZE + GUTTER)
            panel[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = BLOCK
    return panel


def _sample_active_slice(rng: np.random.Generator, full_count: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for start in range(full_count):
        for end in range(start + 1, full_count):
            choices.append((start, end))
    idx = int(rng.integers(0, len(choices)))
    return choices[idx]


def _fill_empty_rows_and_cols(
    rng: np.random.Generator, categories: np.ndarray, active_rows: list[int], active_cols: list[int]
) -> None:
    for rr in active_rows:
        if np.any(categories[rr, active_cols] != 0):
            continue
        cc = active_cols[int(rng.integers(0, len(active_cols)))]
        categories[rr, cc] = int(rng.choice([RED, BLUE, BLOCK]))

    for cc in active_cols:
        if np.any(categories[active_rows, cc] != 0):
            continue
        rr = active_rows[int(rng.integers(0, len(active_rows)))]
        categories[rr, cc] = int(rng.choice([RED, BLUE, BLOCK]))


def _build_category_grid(
    rng: np.random.Generator, full_rows: int, full_cols: int
) -> np.ndarray | None:
    for _ in range(80):
        row0, row1 = _sample_active_slice(rng, full_rows)
        col0, col1 = _sample_active_slice(rng, full_cols)
        if row0 == 0 and row1 == full_rows - 1 and col0 == 0 and col1 == full_cols - 1:
            continue

        categories = np.zeros((full_rows, full_cols), dtype=int)
        active_rows = list(range(row0, row1 + 1))
        active_cols = list(range(col0, col1 + 1))
        active_cells = [(r, c) for r in active_rows for c in active_cols]
        if len(active_cells) < 4:
            continue

        probs = np.array([0.20, 0.25, 0.25, 0.30], dtype=float)
        sampled = rng.choice(np.array([0, RED, BLUE, BLOCK], dtype=int), size=len(active_cells), p=probs)
        for (r, c), value in zip(active_cells, sampled):
            categories[r, c] = int(value)

        forced = rng.choice(len(active_cells), size=3, replace=False)
        categories[active_cells[int(forced[0])]] = RED
        categories[active_cells[int(forced[1])]] = BLUE
        categories[active_cells[int(forced[2])]] = BLOCK

        _fill_empty_rows_and_cols(rng, categories, active_rows, active_cols)

        if not np.all(categories[np.ix_(active_rows, active_cols)].sum(axis=1) > 0):
            continue
        if not np.all(categories[np.ix_(active_rows, active_cols)].sum(axis=0) > 0):
            continue
        if np.count_nonzero(categories == RED) < 1:
            continue
        if np.count_nonzero(categories == BLUE) < 1:
            continue
        if np.count_nonzero(categories == BLOCK) < 1:
            continue
        return categories
    return None


def _render_input_from_categories(
    categories: np.ndarray, top_margin: int, left_margin: int, bottom_margin: int, right_margin: int
) -> np.ndarray:
    left_occ = (categories == RED) | (categories == BLOCK)
    right_occ = (categories == BLUE) | (categories == BLOCK)

    left_panel = _render_panel(left_occ, top_margin, left_margin)
    right_panel = _render_panel(right_occ, top_margin, left_margin)

    panel_height = left_panel.shape[0] + bottom_margin
    panel_width = left_panel.shape[1] + right_margin

    left_canvas = np.full((panel_height, panel_width), BACKGROUND, dtype=int)
    right_canvas = np.full((panel_height, panel_width), BACKGROUND, dtype=int)
    left_canvas[: left_panel.shape[0], : left_panel.shape[1]] = left_panel
    right_canvas[: right_panel.shape[0], : right_panel.shape[1]] = right_panel

    separator_col = np.full((panel_height, 1), SEPARATOR, dtype=int)
    return np.concatenate((left_canvas, separator_col, right_canvas), axis=1)


def _choose_margins(
    rng: np.random.Generator, full_rows: int, full_cols: int
) -> tuple[int, int, int, int] | None:
    base_height = _panel_extent(full_rows)
    base_width = _panel_extent(full_cols)

    for _ in range(60):
        top = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        bottom = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
        right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))

        panel_height = top + base_height + bottom
        panel_width = left + base_width + right
        if panel_height <= 30 and 2 * panel_width + 1 <= 30:
            return top, left, bottom, right
    return None


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    full_rows = int(rng.integers(FULL_ROW_RANGE[0], FULL_ROW_RANGE[1] + 1))
    full_cols = int(rng.integers(FULL_COL_RANGE[0], FULL_COL_RANGE[1] + 1))

    categories = _build_category_grid(rng, full_rows, full_cols)
    if categories is None:
        return None

    margins = _choose_margins(rng, full_rows, full_cols)
    if margins is None:
        return None

    grid = _render_input_from_categories(categories, *margins)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if not np.any(out == YELLOW) or not np.any(out == ORANGE):
        return None
    if not np.any(out == RED) or not np.any(out == BLUE) or not np.any(out == BLOCK):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")

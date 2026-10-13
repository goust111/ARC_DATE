from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _red_left(values: tuple[int, int]) -> bool:
    return int(values[0]) == RED


def _bar_start_from_red(red_col: int, red_is_left: bool) -> int:
    return int(red_col) if red_is_left else int(red_col) - 1


def _bar_cols_from_red(red_col: int, red_is_left: bool) -> tuple[int, int]:
    start = _bar_start_from_red(red_col, red_is_left)
    return start, start + 1


def _bar_overlaps_window(red_col: int, red_is_left: bool, c0: int, c1: int) -> bool:
    left, right = _bar_cols_from_red(red_col, red_is_left)
    return not (right < c0 or left > c1)


def _place_bar(grid: np.ndarray, row: int, red_col: int, red_is_left: bool) -> None:
    c0, c1 = _bar_cols_from_red(red_col, red_is_left)
    if not (0 <= c0 < c1 < grid.shape[1]):
        raise ValueError("bar placement out of bounds")
    if red_is_left:
        grid[row, c0] = RED
        grid[row, c1] = BLUE
    else:
        grid[row, c0] = BLUE
        grid[row, c1] = RED


def _parse_reference(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(grid[0] != BG)
    assert cols.size == 2 and int(cols[1]) == int(cols[0]) + 1, "top row must contain one adjacent two-pixel reference bar"
    values = tuple(map(int, grid[0, cols]))
    assert set(values) == {RED, BLUE}, "reference bar must use red and blue"
    ref_red_col = int(cols[0] if values[0] == RED else cols[1])
    ref_left_col = int(cols[0])
    return ref_red_col, ref_left_col


def _row_bar_info(grid: np.ndarray, row: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    cols = np.flatnonzero(grid[row] != BG)
    if cols.size == 0:
        return None
    values = tuple(map(int, grid[row, cols]))
    if set(values) == {YELLOW}:
        assert cols.size == 1, "marker rows must contain a single yellow marker"
        return None
    assert cols.size == 2 and int(cols[1]) == int(cols[0]) + 1, "each non-marker foreground row must contain one adjacent two-pixel bar"
    assert set(values) == {RED, BLUE}, "bars must use red and blue only"
    return cols, values


def _align_by_red(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    ref_red_col, _ = _parse_reference(g)
    out = g.copy()

    for r in range(1, g.shape[0]):
        info = _row_bar_info(g, r)
        if info is None:
            continue
        cols, values = info
        red_col = int(cols[0] if values[0] == RED else cols[1])
        shift = ref_red_col - red_col
        dst = cols + shift
        assert int(dst[0]) >= 0 and int(dst[1]) < g.shape[1], "shifted bar must stay in bounds"
        out[r, cols] = BG
        out[r, dst] = values

    return out


def _align_by_left_edge(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, ref_left_col = _parse_reference(g)
    out = g.copy()

    for r in range(1, g.shape[0]):
        info = _row_bar_info(g, r)
        if info is None:
            continue
        cols, values = info
        shift = ref_left_col - int(cols[0])
        dst = cols + shift
        assert int(dst[0]) >= 0 and int(dst[1]) < g.shape[1]
        out[r, cols] = BG
        out[r, dst] = values

    return out


def _crop_marker_window(grid: np.ndarray) -> np.ndarray:
    markers = np.argwhere(grid == YELLOW)
    assert markers.shape == (2, 2), "expected exactly two yellow markers"
    return _crop_bbox(grid, _bbox_from_coords(markers))


def _transform_crop_then_mirror(grid: np.ndarray) -> np.ndarray:
    return _crop_marker_window(np.asarray(grid, dtype=int))[:, ::-1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Align each red-blue bar by its red column, crop the yellow marker window, then mirror the crop horizontally.

    Input structure:
    - A black background grid contains one reference red-blue bar on the top row.
    - Several lower rows each contain exactly one adjacent two-pixel red-blue bar.
    - Exactly two yellow marker pixels lie on otherwise empty rows and define a rectangular window.

    Rule:
    - Use the top-row bar as the reference and record the column occupied by its red pixel.
    - For every lower bar, shift the whole bar horizontally so its own red pixel lands in that reference red column.
    - After all bars are aligned, crop the rectangle spanned by the two yellow markers, keeping the markers.
    - Reverse the cropped rectangle left-to-right to produce the output.

    Key insight:
    - The yellow window is fixed, but its contents change because the red-blue bars must be aligned before cropping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    aligned = _align_by_red(g)
    crop = _crop_marker_window(aligned)
    return crop[:, ::-1].copy()


def _valid_red_cols(width: int, red_is_left: bool) -> list[int]:
    offset = 0 if red_is_left else 1
    return list(range(offset, width - 1 + offset))


def _pick_red_col(
    width: int,
    red_is_left: bool,
    ref_red_col: int,
    c0: int,
    c1: int,
    *,
    overlap: bool | None = None,
    side: str | None = None,
    shift: str | None = None,
) -> int | None:
    candidates: list[int] = []
    for red_col in _valid_red_cols(width, red_is_left):
        left, right = _bar_cols_from_red(red_col, red_is_left)
        has_overlap = not (right < c0 or left > c1)
        if overlap is not None and has_overlap != overlap:
            continue
        if side == "left" and right >= c0:
            continue
        if side == "right" and left <= c1:
            continue
        if shift == "zero" and red_col != ref_red_col:
            continue
        if shift == "left" and red_col <= ref_red_col:
            continue
        if shift == "right" and red_col >= ref_red_col:
            continue
        if shift == "nonzero" and red_col == ref_red_col:
            continue
        candidates.append(int(red_col))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_rows() -> tuple[list[int], int] | None:
    count = random.randint(4, 6)
    rows = [random.choice([2, 3])]
    for _ in range(count - 1):
        rows.append(rows[-1] + random.choice([2, 3]))
    height = rows[-1] + random.randint(2, 4) + 1
    if not (11 <= height <= 18):
        return None
    return rows, height


def _sample_window_indices(count: int) -> tuple[int, int]:
    max_len = min(3, count - 1)
    win_len = max_len if max_len == 2 or random.random() < 0.7 else 2
    start = random.randint(0, count - win_len)
    return start, start + win_len - 1


def _marker_rows(bar_rows: list[int], height: int, start_idx: int, end_idx: int) -> tuple[int, int] | None:
    prev_row = 0 if start_idx == 0 else bar_rows[start_idx - 1]
    next_row = height if end_idx == len(bar_rows) - 1 else bar_rows[end_idx + 1]
    top_candidates = list(range(prev_row + 1, bar_rows[start_idx]))
    bottom_candidates = list(range(bar_rows[end_idx] + 1, next_row))
    if not top_candidates or not bottom_candidates:
        return None
    return random.choice(top_candidates), random.choice(bottom_candidates)


def _build_input_grid() -> np.ndarray | None:
    rows_height = _sample_rows()
    if rows_height is None:
        return None
    bar_rows, height = rows_height
    count = len(bar_rows)

    width = random.randint(14, 18)
    ref_red_left = random.choice([True, False])
    ref_red_col = random.randint(4, width - 5)

    start_idx, end_idx = _sample_window_indices(count)
    marker_rows = _marker_rows(bar_rows, height, start_idx, end_idx)
    if marker_rows is None:
        return None
    top_marker_row, bottom_marker_row = marker_rows

    for _ in range(20):
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        if left_margin + right_margin + 1 >= 4:
            break
    c0 = ref_red_col - left_margin
    c1 = ref_red_col + right_margin
    if not (0 <= c0 < ref_red_col < c1 < width):
        return None

    inside_indices = list(range(start_idx, end_idx + 1))
    if len(inside_indices) < 2:
        return None
    outside_indices = [i for i in range(count) if i not in inside_indices]
    if not outside_indices:
        return None

    move_idx = inside_indices[0]
    enter_idx = inside_indices[1]
    reverse_idx = random.choice(inside_indices)
    stationary_idx = random.choice(outside_indices)
    enter_side = random.choice(["left", "right"])

    bars: list[tuple[bool, int] | None] = [None] * count

    for idx in range(count):
        if idx == stationary_idx:
            red_is_left = random.choice([True, False])
            red_col = ref_red_col
        elif idx == enter_idx:
            red_is_left = random.choice([True, False])
            red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=False, side=enter_side, shift="nonzero")
        elif idx == move_idx:
            red_is_left = (not ref_red_left) if idx == reverse_idx else random.choice([True, False])
            shift_dir = "left" if enter_side == "right" else "right"
            red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=True, shift=shift_dir)
            if red_col is None:
                red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=True, shift="nonzero")
        elif idx == reverse_idx:
            red_is_left = not ref_red_left
            red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=True, shift="nonzero")
            if red_col is None:
                red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=False, shift="nonzero")
        else:
            red_is_left = random.choice([True, False])
            pattern = random.choice(["inside_shift", "outside_shift", "zero"])
            if pattern == "inside_shift":
                red_col = _pick_red_col(width, red_is_left, ref_red_col, c0, c1, overlap=True, shift="nonzero")
            elif pattern == "outside_shift":
                red_col = _pick_red_col(
                    width,
                    red_is_left,
                    ref_red_col,
                    c0,
                    c1,
                    overlap=False,
                    side=random.choice(["left", "right"]),
                    shift="nonzero",
                )
            else:
                red_col = ref_red_col
        if red_col is None:
            return None
        bars[idx] = (red_is_left, int(red_col))

    grid = np.full((height, width), BG, dtype=int)
    _place_bar(grid, 0, ref_red_col, ref_red_left)

    for r, bar in zip(bar_rows, bars):
        assert bar is not None
        red_is_left, red_col = bar
        _place_bar(grid, r, red_col, red_is_left)

    if random.random() < 0.5:
        grid[top_marker_row, c0] = YELLOW
        grid[bottom_marker_row, c1] = YELLOW
    else:
        grid[top_marker_row, c1] = YELLOW
        grid[bottom_marker_row, c0] = YELLOW

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False

        markers = np.argwhere(g == YELLOW)
        if markers.shape != (2, 2):
            return False
        bbox = _bbox_from_coords(markers)
        r0, c0, r1, c1 = bbox

        ref_red_col, _ = _parse_reference(g)
        ref_values = tuple(map(int, g[0, np.flatnonzero(g[0] != BG)]))
        ref_red_left = _red_left(ref_values)

        bar_rows: list[int] = []
        initial_rows: list[tuple[int, bool, int]] = []
        for r in range(1, g.shape[0]):
            cols = np.flatnonzero(g[r] != BG)
            if cols.size == 0:
                continue
            values = tuple(map(int, g[r, cols]))
            if set(values) == {YELLOW}:
                if cols.size != 1:
                    return False
                if r in bar_rows:
                    return False
                continue
            if cols.size != 2 or int(cols[1]) != int(cols[0]) + 1 or set(values) != {RED, BLUE}:
                return False
            bar_rows.append(r)
            red_col = int(cols[0] if values[0] == RED else cols[1])
            initial_rows.append((r, _red_left(values), red_col))

        if not (4 <= len(initial_rows) <= 6):
            return False
        if any(r in {int(markers[0, 0]), int(markers[1, 0])} for r in bar_rows):
            return False

        aligned = _align_by_red(g)
        crop = _crop_marker_window(aligned)
        correct = crop[:, ::-1].copy()
        if not np.array_equal(main(g), correct):
            return False
        if np.array_equal(crop, correct):
            return False
        if np.array_equal(_transform_crop_then_mirror(g), correct):
            return False
        if np.array_equal(_crop_marker_window(_align_by_left_edge(g))[:, ::-1], correct):
            return False

        enter_count = 0
        moved_inside_count = 0
        reversed_shifted_inside = 0
        zero_shift_count = 0
        shift_left = 0
        shift_right = 0
        final_inside_rows = 0

        for row, red_is_left, red_col in initial_rows:
            before_inside = r0 <= row <= r1 and _bar_overlaps_window(red_col, red_is_left, c0, c1)
            after_inside = r0 <= row <= r1 and _bar_overlaps_window(ref_red_col, red_is_left, c0, c1)
            if after_inside:
                final_inside_rows += 1
            if red_col == ref_red_col:
                zero_shift_count += 1
            elif red_col < ref_red_col:
                shift_right += 1
            else:
                shift_left += 1
            if (not before_inside) and after_inside:
                enter_count += 1
            if before_inside and after_inside and red_col != ref_red_col:
                moved_inside_count += 1
            if after_inside and red_col != ref_red_col and red_is_left != ref_red_left:
                reversed_shifted_inside += 1

        if enter_count < 1 or moved_inside_count < 1:
            return False
        if reversed_shifted_inside < 1:
            return False
        if zero_shift_count < 1:
            return False
        if shift_left < 1 or shift_right < 1:
            return False
        if final_inside_rows < 2:
            return False
        if len(np.unique(correct)) < 3:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input_grid()
        if grid is not None and _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

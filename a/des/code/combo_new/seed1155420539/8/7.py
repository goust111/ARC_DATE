from __future__ import annotations

import random

import numpy as np


ARC_LIMIT = 30
GRAY = 5
MAX_GENERATION_ATTEMPTS = 400
ROW_COUNT_RANGE = (2, 6)
COL_COUNT_RANGE = (2, 6)
LEFT_WIDTH_RANGE = (4, 10)
RIGHT_EXTRA_WIDTH_RANGE = (2, 8)
FRAME_TOP_RANGE = (4, 12)
BOTTOM_MARGIN_RANGE = (1, 4)
HSEG_MIN_LEN = 2
VSEG_MIN_LEN = 2


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = [
        c
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected exactly one full non-background separator column"
    col = int(candidates[0])
    return col, int(grid[0, col])


def _find_target_frame(grid: np.ndarray, frame_color: int, separator_col: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == frame_color):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 <= separator_col:
            continue
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        mask = np.zeros((h, w), dtype=bool)
        rel = coords - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(mask, border):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one target frame"
    return candidates[0]


def _extract_horizontal_colors(left: np.ndarray, bg: int) -> list[int]:
    segments: list[tuple[int, int]] = []
    for r in range(left.shape[0]):
        cols = np.flatnonzero(left[r, :] != bg)
        if cols.size == 0:
            continue
        assert int(cols[-1] - cols[0] + 1) == int(cols.size), "horizontal segment must be contiguous"
        colors = np.unique(left[r, cols])
        assert len(colors) == 1, "horizontal segment must be monochrome"
        segments.append((r, int(colors[0])))
    assert segments, "expected horizontal instruction segments"
    segments.sort(key=lambda item: int(item[0]))
    return [color for _row, color in segments]


def _extract_vertical_colors(right: np.ndarray, bg: int, frame_color: int) -> list[int]:
    mask = (right != bg) & (right != frame_color)
    segments: list[tuple[int, int]] = []
    for c in range(right.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            continue
        assert int(rows[-1] - rows[0] + 1) == int(rows.size), "vertical segment must be contiguous"
        colors = np.unique(right[rows, c])
        assert len(colors) == 1, "vertical segment must be monochrome"
        segments.append((c, int(colors[0])))
    assert segments, "expected vertical instruction segments"
    segments.sort(key=lambda item: int(item[0]))
    return [color for _col, color in segments]


def _weave_patch(row_colors: list[int], col_colors: list[int], frame_color: int) -> np.ndarray:
    out = np.full((len(row_colors) + 2, len(col_colors) + 2), frame_color, dtype=int)
    for i, row_color in enumerate(row_colors):
        for j, col_color in enumerate(col_colors):
            out[i + 1, j + 1] = int(row_color) if (i + j) % 2 == 0 else int(col_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave the ordered horizontal and vertical color instructions into the grey target frame.

    Input structure:
    - A dominant background fills the grid.
    - One full-height grey column splits the grid into a left instruction area and a right instruction area.
    - The left side contains several monochrome horizontal line segments on distinct rows.
    - The right side contains several monochrome vertical line segments on distinct columns, plus one empty grey rectangular frame below them.

    Rule:
    - Read the left horizontal segments from top to bottom; their colors define the target frame's interior rows in order.
    - Read the right vertical segments from left to right; their colors define the target frame's interior columns in order.
    - Fill the target frame interior so cells with even row-plus-column parity take the row color and odd-parity cells take the column color, then crop the output to that grey frame.

    Key insight:
    - The segments are not copied as geometry; they act as ordered color labels for rows and columns, and the actual picture appears only after applying the parity-based weaving rule inside the separate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    separator_col, frame_color = _find_separator_column(g, bg)
    frame_r0, frame_c0, frame_r1, frame_c1 = _find_target_frame(g, frame_color, separator_col)

    left = g[:, :separator_col]
    right = g[:, separator_col + 1 :]
    row_colors = _extract_horizontal_colors(left, bg)
    col_colors = _extract_vertical_colors(right, bg, frame_color)

    assert len(row_colors) == frame_r1 - frame_r0 - 1, "frame interior height must match horizontal segment count"
    assert len(col_colors) == frame_c1 - frame_c0 - 1, "frame interior width must match vertical segment count"

    return _weave_patch(row_colors, col_colors, frame_color)


def _valid_bg_dominance(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    return int(values[order[0]]) == bg and (len(order) == 1 or int(counts[order[0]]) > int(counts[order[1]]))


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != GRAY])
    row_count = random.randint(*ROW_COUNT_RANGE)
    col_count = random.randint(*COL_COUNT_RANGE)
    frame_h = row_count + 2
    frame_w = col_count + 2

    left_w_max = min(LEFT_WIDTH_RANGE[1], ARC_LIMIT - 1 - (frame_w + RIGHT_EXTRA_WIDTH_RANGE[0]))
    if left_w_max < LEFT_WIDTH_RANGE[0]:
        return None
    left_w = random.randint(LEFT_WIDTH_RANGE[0], left_w_max)

    right_w_min = frame_w + 2
    right_w_max = ARC_LIMIT - left_w - 1
    if right_w_min > right_w_max:
        return None
    extra = random.randint(*RIGHT_EXTRA_WIDTH_RANGE)
    right_w = min(right_w_max, frame_w + extra)
    if right_w < right_w_min:
        right_w = right_w_min

    frame_top_max = min(FRAME_TOP_RANGE[1], ARC_LIMIT - frame_h - BOTTOM_MARGIN_RANGE[0])
    if FRAME_TOP_RANGE[0] > frame_top_max:
        return None
    frame_top = random.randint(FRAME_TOP_RANGE[0], frame_top_max)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    height = frame_top + frame_h + bottom_margin
    if height > ARC_LIMIT:
        return None

    width = left_w + 1 + right_w
    if width > ARC_LIMIT:
        return None

    frame_left_local = random.randint(1, right_w - frame_w - 1)
    frame_right_local = frame_left_local + frame_w - 1

    grid = np.full((height, width), bg, dtype=int)
    separator_col = left_w
    grid[:, separator_col] = GRAY

    frame_r0 = frame_top
    frame_r1 = frame_top + frame_h - 1
    frame_c0 = separator_col + 1 + frame_left_local
    frame_c1 = frame_c0 + frame_w - 1
    grid[frame_r0, frame_c0 : frame_c1 + 1] = GRAY
    grid[frame_r1, frame_c0 : frame_c1 + 1] = GRAY
    grid[frame_r0 : frame_r1 + 1, frame_c0] = GRAY
    grid[frame_r0 : frame_r1 + 1, frame_c1] = GRAY

    palette = [c for c in range(10) if c not in {bg, GRAY}]
    row_colors = random.sample(palette, row_count)
    col_colors = random.sample(palette, col_count)

    row_positions = sorted(random.sample(range(height), row_count))
    placed_rows: list[tuple[int, int]] = []
    for row, color in zip(row_positions, row_colors, strict=False):
        seg_len = random.randint(HSEG_MIN_LEN, left_w)
        start = random.randint(0, left_w - seg_len)
        grid[row, start : start + seg_len] = int(color)
        placed_rows.append((int(row), int(color)))

    max_v_end = frame_top - 2
    if max_v_end + 1 < VSEG_MIN_LEN:
        return None
    col_positions = sorted(random.sample(range(right_w), col_count))
    placed_cols: list[tuple[int, int]] = []
    for col_local, color in zip(col_positions, col_colors, strict=False):
        seg_len = random.randint(VSEG_MIN_LEN, max_v_end + 1)
        start = random.randint(0, max_v_end - seg_len + 1)
        global_col = separator_col + 1 + col_local
        grid[start : start + seg_len, global_col] = int(color)
        placed_cols.append((int(col_local), int(color)))

    expected = _weave_patch(
        [color for _row, color in sorted(placed_rows)],
        [color for _col, color in sorted(placed_cols)],
        GRAY,
    )

    if not _valid_bg_dominance(grid, bg):
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving puzzle input")

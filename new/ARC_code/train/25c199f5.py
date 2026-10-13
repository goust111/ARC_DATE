from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _separator_columns(g: np.ndarray, bg: int) -> list[int]:
    H, W = g.shape
    candidates: list[list[int]] = []
    for color in colors_present(g, exclude={bg}):
        mask = g == int(color)
        full_cols = [c for c in range(W) if bool(np.all(mask[:, c]))]
        if len(full_cols) != 2:
            continue
        if int(mask.sum()) != int(H * len(full_cols)):
            continue
        s0, s1 = full_cols
        if s0 > 0 and s1 > s0 + 1 and s1 < W - 1:
            panel_w = s0
            if s1 - s0 - 1 == panel_w and W - s1 - 1 == panel_w:
                candidates.append(full_cols)

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the non-background rows from three separated panels into one panel, scanning panels right-to-left.

    Input structure:
    - A rectangular grid contains three equal-size panels separated by two full-height divider columns.
    - The background is the dominant color; divider pixels use a color that appears only in the divider columns.

    Rule:
    - Read panels from right to left, and within each panel read rows from top to bottom.
    - Keep each row that contains any non-background pixel, discarding all-background rows.
    - Place the kept rows at the bottom of a single panel-shaped output, padding above with background rows.

    Key insight:
    - The task is row packing: the divider columns mark independent panels, and only the occupied rows are moved.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    s0, s1 = _separator_columns(g, bg)
    panel_w = int(s0)
    assert H == panel_w

    panels = [
        g[:, :s0],
        g[:, s0 + 1 : s1],
        g[:, s1 + 1 :],
    ]
    rows: list[np.ndarray] = []
    for panel in reversed(panels):
        assert panel.shape == (H, panel_w)
        for r in range(H):
            row = panel[r, :]
            if bool(np.any(row != bg)):
                rows.append(row.copy())

    assert 0 < len(rows) <= H
    for i in range(len(rows) - 1):
        cols_a = set(int(c) for c in range(panel_w) if int(rows[i][c]) != bg)
        cols_b = set(int(c) for c in range(panel_w) if int(rows[i + 1][c]) != bg)
        assert cols_a & cols_b, "consecutive non-bg rows must share at least one column"
    out = np.full((H, panel_w), bg, dtype=int)
    out[H - len(rows) :, :] = np.asarray(rows, dtype=int)
    return out

# --------------- generator (o5x) ---------------

# Canvas and fixed palette observed in the task.
PANEL_SIZE_RANGE = (4, 6)
BG_COLOR, SEPARATOR_COLOR = 7, 6
CONTENT_COLORS = (1, 5)
# Packed output rows and per-panel block heights.
PACKED_ROW_RANGE = (3, 5)
GROUP_ROW_RANGE = (1, 3)
# Row intervals and hollow 3-row block variants.
ROW_WIDTH_RANGE = (1, 5)
HOLLOW_HEIGHT = 3
HOLLOW_WIDTH_RANGE = (3, 5)
HOLLOW_PROBABILITY = 0.35
# Rejection-sampling retry limit.
GENERATION_RETRIES = 300


def _partition_rows(total: int) -> tuple[int, int, int]:
    min_rows, max_rows = GROUP_ROW_RANGE
    choices = [
        (right, middle, left)
        for right in range(min_rows, max_rows + 1)
        for middle in range(min_rows, max_rows + 1)
        for left in range(min_rows, max_rows + 1)
        if right + middle + left == total
    ]
    return random.choice(choices)


def _interval(width: int, must_include: int | None = None) -> tuple[int, int]:
    min_len, max_config_len = ROW_WIDTH_RANGE
    max_len = min(max_config_len, width)
    length = random.randint(min_len, max_len)
    if must_include is None:
        start = random.randint(0, width - length)
    else:
        lo = max(0, must_include - length + 1)
        hi = min(must_include, width - length)
        start = random.randint(lo, hi)
    return start, start + length - 1


def _solid_block(height: int, width: int, bg: int, color: int) -> np.ndarray:
    block = np.full((height, width), bg, dtype=int)
    left, right = _interval(width)
    block[0, left : right + 1] = color
    for r in range(1, height):
        overlap_col = random.randint(left, right)
        left, right = _interval(width, overlap_col)
        block[r, left : right + 1] = color
    return block


def _hollow_block(width: int, bg: int, color: int) -> np.ndarray:
    block = np.full((HOLLOW_HEIGHT, width), bg, dtype=int)
    min_width, max_width = HOLLOW_WIDTH_RANGE
    shape_w = random.randint(min_width, min(max_width, width))
    left = random.randint(0, width - shape_w)
    right = left + shape_w - 1
    block[0, left : right + 1] = color
    block[1, left] = color
    block[1, right] = color
    block[2, left : right + 1] = color
    return block


def _connected_block(height: int, width: int, bg: int, color: int) -> np.ndarray:
    if height == HOLLOW_HEIGHT and width >= HOLLOW_WIDTH_RANGE[0] and random.random() < HOLLOW_PROBABILITY:
        return _hollow_block(width, bg, color)
    return _solid_block(height, width, bg, color)


def _place_block(block: np.ndarray, panel_size: int, bg: int) -> np.ndarray:
    panel = np.full((panel_size, panel_size), bg, dtype=int)
    offset = random.randint(0, panel_size - block.shape[0])
    panel[offset : offset + block.shape[0], :] = block
    return panel


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*PANEL_SIZE_RANGE)
    bg = BG_COLOR
    sep = SEPARATOR_COLOR
    A, B = random.sample(CONTENT_COLORS, 2)

    K = random.randint(PACKED_ROW_RANGE[0], min(PACKED_ROW_RANGE[1], H))
    right_n, middle_n, left_n = _partition_rows(K)

    right_block = _connected_block(right_n, H, bg, A)
    middle_block = _connected_block(middle_n, H, bg, B)
    left_block = _connected_block(left_n, H, bg, A)

    stacked_rows = np.vstack([right_block, middle_block, left_block])
    out = np.full((H, H), bg, dtype=int)
    out[H - K :, :] = stacked_rows

    panels = tuple(_place_block(block, H, bg) for block in (left_block, middle_block, right_block))
    sep_col = np.full((H, 1), sep, dtype=int)
    inp = np.hstack([panels[0], sep_col, panels[1], sep_col, panels[2]])

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 25c199f5 input")

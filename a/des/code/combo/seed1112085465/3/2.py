from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _segments_between(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    runs = _group_runs(separators)
    if not runs:
        return [(0, limit - 1)]

    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if cursor <= start - 1:
            spans.append((cursor, start - 1))
        cursor = end + 1
    if cursor <= limit - 1:
        spans.append((cursor, limit - 1))
    return spans


def _find_separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    solid_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and bool(np.all(grid[r] == grid[r, 0]))
    ]
    solid_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == grid[0, c]))
    ]

    row_colors = {color for _r, color in solid_rows}
    col_colors = {color for _c, color in solid_cols}
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one separator color shared by full rows and columns"

    sep = next(iter(candidates))
    row_lines = [r for r, color in solid_rows if color == sep]
    col_lines = [c for c, color in solid_cols if color == sep]
    assert row_lines and col_lines, "expected both horizontal and vertical separator lines"
    return int(sep), row_lines, col_lines


def _extract_panel_rows(grid: np.ndarray, sep: int) -> tuple[list[list[np.ndarray]], int, int]:
    h, w = grid.shape
    row_lines = [r for r in range(h) if bool(np.all(grid[r] == sep))]
    col_lines = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]

    row_spans = _segments_between(h, row_lines)
    col_spans = _segments_between(w, col_lines)
    assert len(row_spans) >= 2, "expected at least two panel rows"
    assert len(col_spans) >= 3, "expected at least three panels per row"

    panel_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    panel_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "panels must be equal-sized"
    panel_h = next(iter(panel_heights))
    panel_w = next(iter(panel_widths))

    panel_rows: list[list[np.ndarray]] = []
    for r0, r1 in row_spans:
        row_panels: list[np.ndarray] = []
        for c0, c1 in col_spans:
            row_panels.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
        panel_rows.append(row_panels)
    return panel_rows, int(panel_h), int(panel_w)


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    ranked = counts.most_common()
    assert ranked, "majority vote requires at least one value"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "majority vote must be unique"
    return int(ranked[0][0])


def _repair_defective_panel(row_panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(row_panels, axis=0)
    n_panels, panel_h, panel_w = stack.shape
    assert n_panels >= 3, "need at least three panels for a reliable majority vote"

    consensus = np.empty((panel_h, panel_w), dtype=int)
    for r in range(panel_h):
        for c in range(panel_w):
            consensus[r, c] = _majority_color(stack[:, r, c])

    mismatch_counts = [int(np.count_nonzero(stack[i] != consensus)) for i in range(n_panels)]
    defective = [idx for idx, count in enumerate(mismatch_counts) if count > 0]
    assert len(defective) == 1, "expected exactly one defective panel in each row"

    fixed = stack[defective[0]].copy()
    mask = fixed != consensus
    fixed[mask] = consensus[mask]
    assert np.array_equal(fixed, consensus), "repair should match the row consensus"
    return fixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair one defective panel per panel-row, then output the repaired defect panels stacked vertically.

    Input structure
    - Background is black.
    - One-cell-thick grey separator rows and columns split the grid into equal rectangular panels.
    - Within each panel row, all panels share the same pattern except one panel with 1-3 wrong pixels.

    Rule
    - Majority-vote each pixel position across the panels of the same row.
    - Identify the unique panel that disagrees with that consensus.
    - Repair it and output one repaired panel per row, keeping row order and separating outputs by one black row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = 0
    sep, _row_lines, _col_lines = _find_separator_color(g, bg)
    panel_rows, panel_h, panel_w = _extract_panel_rows(g, sep)

    repaired = [_repair_defective_panel(row_panels) for row_panels in panel_rows]
    out_h = len(repaired) * panel_h + (len(repaired) - 1)
    out = np.full((out_h, panel_w), bg, dtype=int)
    for idx, panel in enumerate(repaired):
        r0 = idx * (panel_h + 1)
        out[r0 : r0 + panel_h, :] = panel
    return out


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_PANEL_ROWS = 2
MAX_PANEL_ROWS = 4
MIN_PANEL_COLS = 3
MAX_PANEL_COLS = 5
MIN_PANEL_SIZE = 3
MAX_PANEL_SIZE = 7
GRID_LIMIT = 30

PATTERN_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 400


def _paint_rectangle(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    rect_h = random.randint(1, h)
    rect_w = random.randint(1, w)
    r0 = random.randint(0, h - rect_h)
    c0 = random.randint(0, w - rect_w)
    canvas[r0 : r0 + rect_h, c0 : c0 + rect_w] = color


def _paint_segment(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    if random.random() < 0.5:
        r = random.randrange(h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        canvas[r, c0 : c1 + 1] = color
    else:
        c = random.randrange(w)
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        canvas[r0 : r1 + 1, c] = color


def _paint_scatter(canvas: np.ndarray, colors: list[int]) -> None:
    area = canvas.size
    count = random.randint(2, min(area // 2, 7))
    for idx in random.sample(range(area), count):
        canvas.flat[idx] = random.choice(colors)
    if canvas.shape[0] >= 2 and canvas.shape[1] >= 2 and random.random() < 0.5:
        r0 = random.randint(0, canvas.shape[0] - 2)
        c0 = random.randint(0, canvas.shape[1] - 2)
        canvas[r0 : r0 + 2, c0 : c0 + 2] = random.choice(colors)


def _pattern_ok(pattern: np.ndarray) -> bool:
    non_bg = int(np.count_nonzero(pattern != BG))
    if non_bg < 2:
        return False
    if non_bg > pattern.size // 2:
        return False
    return True


def _generate_pattern(panel_h: int, panel_w: int, colors: list[int]) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        pattern = np.full((panel_h, panel_w), BG, dtype=int)
        mode = random.choice(("rectangles", "segments", "scatter"))
        if mode == "rectangles":
            for _ in range(random.randint(1, 3)):
                _paint_rectangle(pattern, random.choice(colors))
        elif mode == "segments":
            for _ in range(random.randint(2, 4)):
                _paint_segment(pattern, random.choice(colors))
        else:
            _paint_scatter(pattern, colors)

        if _pattern_ok(pattern):
            return pattern
    raise RuntimeError("failed to sample a readable base pattern")


def _corrupt_panel(base: np.ndarray) -> np.ndarray:
    panel = base.copy()
    panel_h, panel_w = panel.shape
    n_defects = random.randint(1, min(3, panel.size))
    for index in random.sample(range(panel.size), n_defects):
        r, c = divmod(index, panel_w)
        original = int(panel[r, c])
        choices = [color for color in [BG] + PALETTE if color != original]
        panel[r, c] = random.choice(choices)
    assert not np.array_equal(panel, base), "corruption must change the panel"
    return panel


def _stack_expected(rows: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = rows[0].shape
    out = np.full((len(rows) * panel_h + len(rows) - 1, panel_w), BG, dtype=int)
    for idx, panel in enumerate(rows):
        r0 = idx * (panel_h + 1)
        out[r0 : r0 + panel_h, :] = panel
    return out


def _sample_layout() -> tuple[int, int, int, int] | None:
    panel_rows = random.randint(MIN_PANEL_ROWS, MAX_PANEL_ROWS)
    panel_cols = random.randint(MIN_PANEL_COLS, MAX_PANEL_COLS)
    max_h = min(MAX_PANEL_SIZE, (GRID_LIMIT - (panel_rows - 1)) // panel_rows)
    max_w = min(MAX_PANEL_SIZE, (GRID_LIMIT - (panel_cols - 1)) // panel_cols)
    if max_h < MIN_PANEL_SIZE or max_w < MIN_PANEL_SIZE:
        return None
    panel_h = random.randint(MIN_PANEL_SIZE, max_h)
    panel_w = random.randint(MIN_PANEL_SIZE, max_w)
    return panel_rows, panel_cols, panel_h, panel_w


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_layout()
    if layout is None:
        return None
    panel_rows, panel_cols, panel_h, panel_w = layout

    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for r in range(panel_h, grid_h, panel_h + 1):
        grid[r, :] = SEP
    for c in range(panel_w, grid_w, panel_w + 1):
        grid[:, c] = SEP

    repaired_rows: list[np.ndarray] = []
    for panel_row in range(panel_rows):
        colors = random.sample(PALETTE, random.randint(2, 4))
        base = _generate_pattern(panel_h, panel_w, colors)
        repaired_rows.append(base.copy())
        defect_col = random.randrange(panel_cols)

        for panel_col in range(panel_cols):
            panel = _corrupt_panel(base) if panel_col == defect_col else base
            r0 = panel_row * (panel_h + 1)
            c0 = panel_col * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    return grid, _stack_expected(repaired_rows)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_input()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random

import numpy as np


ORANGE = 7
GREY = 5
PALETTE = (0, 1, 2, 3, 4, 6, 8, 9)
MAX_COLOR_ID = 10

PANEL_ROWS_RANGE = (2, 4)
PANEL_COLS_RANGE = (4, 6)
PANEL_HEIGHT_RANGE = (4, 8)
PANEL_WIDTH_RANGE = (4, 8)
ROW_RETRIES = 200
GENERATION_RETRIES = 400


def _uniform_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx in range(grid.shape[axis]):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        if np.all(line == line.flat[0]):
            out.append((idx, int(line.flat[0])))
    return out


def _intervals_from_separators(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected separator lines on both axes"
    assert all(a + 1 < b for a, b in zip(separators, separators[1:])), "separators must be one cell thick"
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty panels"
    assert len(set(sizes)) == 1, "all panels must have equal size"
    return intervals


def _panel_intervals(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_uniform = _uniform_lines(grid, axis=0)
    col_uniform = _uniform_lines(grid, axis=1)
    common_colors = {color for _, color in row_uniform} & {color for _, color in col_uniform}
    assert len(common_colors) == 1, "expected one separator color shared by full rows and columns"
    separator = common_colors.pop()
    row_separators = [idx for idx, color in row_uniform if color == separator]
    col_separators = [idx for idx, color in col_uniform if color == separator]
    row_intervals = _intervals_from_separators(row_separators, grid.shape[0])
    col_intervals = _intervals_from_separators(col_separators, grid.shape[1])
    assert len(col_intervals) >= 3, "each panel row must contain a majority"
    return row_intervals, col_intervals, int(separator)


def _row_scores(panels: list[np.ndarray]) -> np.ndarray:
    count = len(panels)
    scores = np.zeros(count, dtype=int)
    for i in range(count):
        for j in range(i + 1, count):
            diff = int(np.count_nonzero(panels[i] != panels[j]))
            scores[i] += diff
            scores[j] += diff
    return scores


def _majority_panel(panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(panels, axis=0)
    flat = stack.reshape(stack.shape[0], -1)
    voted = np.empty(flat.shape[1], dtype=int)
    for idx in range(flat.shape[1]):
        counts = np.bincount(flat[:, idx], minlength=MAX_COLOR_ID)
        winners = np.flatnonzero(counts == counts.max())
        assert winners.size == 1, "majority vote must be unique at every pixel"
        voted[idx] = int(winners[0])
    return voted.reshape(stack.shape[1:])


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a separator-panel grid, repair the most anomalous panel in each row by voting across the others.

    Input structure
    - Orange background with grey full-width and full-height separator lines.
    - The separators split the grid into equal rectangular panels arranged in rows and columns.
    - In each panel row, exactly one panel differs most from the others.

    Rule
    - For every panel row, compute each panel's total pixel mismatch count against all other panels in that row.
    - Identify the unique panel with the largest total mismatch.
    - Replace only that panel with the pixelwise majority vote of the remaining panels.

    Key insight
    - The outlier is a corrupted version of a shared row-level pattern, so the other panels define the repair target.
    """
    g = np.asarray(grid, dtype=int)
    row_intervals, col_intervals, _separator = _panel_intervals(g)

    out = g.copy()
    for r0, r1 in row_intervals:
        panels = [g[r0:r1, c0:c1] for c0, c1 in col_intervals]
        scores = _row_scores(panels)
        top = np.flatnonzero(scores == scores.max())
        assert top.size == 1, "expected one most-different panel per row"
        anomaly_idx = int(top[0])
        repaired = _majority_panel([panel for i, panel in enumerate(panels) if i != anomaly_idx])
        c0, c1 = col_intervals[anomaly_idx]
        out[r0:r1, c0:c1] = repaired

    return out


def _different_color(base: int, palette: list[int]) -> int:
    choices = [ORANGE, *palette]
    choices = [color for color in choices if color != int(base)]
    return int(random.choice(choices))


def _paint_shape(panel: np.ndarray, color: int) -> None:
    height, width = panel.shape
    kind = random.choice(("rect", "hline", "vline", "plus"))
    if kind == "rect":
        block_h = random.randint(1, max(1, height // 2))
        block_w = random.randint(1, max(1, width // 2))
        r0 = random.randint(0, height - block_h)
        c0 = random.randint(0, width - block_w)
        panel[r0 : r0 + block_h, c0 : c0 + block_w] = color
        return
    if kind == "hline":
        row = random.randrange(height)
        c0 = random.randint(0, width - 2)
        c1 = random.randint(c0 + 2, width)
        panel[row, c0:c1] = color
        return
    if kind == "vline":
        col = random.randrange(width)
        r0 = random.randint(0, height - 2)
        r1 = random.randint(r0 + 2, height)
        panel[r0:r1, col] = color
        return

    center_r = random.randrange(height)
    center_c = random.randrange(width)
    arm = random.randint(1, max(1, min(height, width) // 3))
    panel[max(0, center_r - arm) : min(height, center_r + arm + 1), center_c] = color
    panel[center_r, max(0, center_c - arm) : min(width, center_c + arm + 1)] = color


def _make_consensus_panel(height: int, width: int, palette: list[int]) -> np.ndarray:
    area = height * width
    for _ in range(ROW_RETRIES):
        panel = np.full((height, width), ORANGE, dtype=int)
        for _ in range(random.randint(3, 6)):
            _paint_shape(panel, random.choice(palette))

        foreground = panel != ORANGE
        count = int(foreground.sum())
        if count < max(5, area // 6):
            continue
        if count > (3 * area) // 4:
            continue
        if len(set(map(int, panel[foreground].ravel()))) < 2:
            continue
        return panel
    raise RuntimeError("failed to create a consensus panel")


def _make_normal_panels(consensus: np.ndarray, count: int, palette: list[int]) -> list[np.ndarray]:
    height, width = consensus.shape
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    assert len(coords) >= count

    panels = [consensus.copy() for _ in range(count)]
    for idx in range(count):
        r, c = coords.pop()
        panels[idx][r, c] = _different_color(int(consensus[r, c]), palette)

    max_changes = (count - 1) // 2
    extra_cap = min(len(coords), max(0, (height * width) // 6))
    extra_count = random.randint(0, extra_cap)
    for _ in range(extra_count):
        r, c = coords.pop()
        changed_panels = random.sample(range(count), random.randint(1, max_changes))
        new_color = _different_color(int(consensus[r, c]), palette)
        for idx in changed_panels:
            panels[idx][r, c] = new_color

    return panels


def _corrupt_region(panel: np.ndarray, palette: list[int]) -> None:
    height, width = panel.shape
    fill_choices = [ORANGE, *palette]
    color = int(random.choice(fill_choices))
    kind = random.choice(("rect", "hline", "vline"))
    if kind == "rect":
        block_h = random.randint(1, max(1, (height + 1) // 2))
        block_w = random.randint(1, max(1, (width + 1) // 2))
        r0 = random.randint(0, height - block_h)
        c0 = random.randint(0, width - block_w)
        panel[r0 : r0 + block_h, c0 : c0 + block_w] = color
        return
    if kind == "hline":
        row = random.randrange(height)
        c0 = random.randint(0, width - 2)
        c1 = random.randint(c0 + 2, width)
        panel[row, c0:c1] = color
        return

    col = random.randrange(width)
    r0 = random.randint(0, height - 2)
    r1 = random.randint(r0 + 2, height)
    panel[r0:r1, col] = color


def _make_anomaly_panel(consensus: np.ndarray, palette: list[int]) -> np.ndarray:
    height, width = consensus.shape
    area = height * width
    min_changes = max(4, area // 4)
    max_changes = max(min_changes, min(area - 1, (2 * area) // 3))

    for _ in range(ROW_RETRIES):
        panel = consensus.copy()
        for _ in range(random.randint(2, 4)):
            _corrupt_region(panel, palette)

        changed = np.argwhere(panel != consensus)
        target = random.randint(min_changes, max_changes)
        if changed.shape[0] < target:
            remaining = [(r, c) for r in range(height) for c in range(width) if panel[r, c] == consensus[r, c]]
            random.shuffle(remaining)
            for r, c in remaining[: target - changed.shape[0]]:
                panel[r, c] = _different_color(int(consensus[r, c]), palette)

        diff = int(np.count_nonzero(panel != consensus))
        if min_changes <= diff <= max_changes:
            return panel

    raise RuntimeError("failed to create an anomalous panel")


def _build_row_panels(height: int, width: int, n_cols: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    area = height * width
    for _ in range(ROW_RETRIES):
        palette = random.sample(PALETTE, random.randint(2, 4))
        consensus = _make_consensus_panel(height, width, palette)
        normal_panels = _make_normal_panels(consensus, n_cols - 1, palette)
        anomaly = _make_anomaly_panel(consensus, palette)
        anomaly_idx = random.randrange(n_cols)

        input_panels = normal_panels.copy()
        input_panels.insert(anomaly_idx, anomaly)

        scores = _row_scores(input_panels)
        top = np.flatnonzero(scores == scores.max())
        if top.size != 1 or int(top[0]) != anomaly_idx:
            continue

        repaired = _majority_panel([panel for i, panel in enumerate(input_panels) if i != anomaly_idx])
        if not np.array_equal(repaired, consensus):
            continue

        other_scores = np.delete(scores, anomaly_idx)
        if int(scores[anomaly_idx] - other_scores.max()) < max(3, area // 12):
            continue

        output_panels = [panel.copy() for panel in input_panels]
        output_panels[anomaly_idx] = consensus.copy()
        return input_panels, output_panels

    raise RuntimeError("failed to build a valid panel row")


def _render_grid(rows: list[list[np.ndarray]], separator: int) -> np.ndarray:
    n_rows = len(rows)
    n_cols = len(rows[0])
    panel_height, panel_width = rows[0][0].shape
    grid_height = n_rows * panel_height + (n_rows - 1)
    grid_width = n_cols * panel_width + (n_cols - 1)
    grid = np.full((grid_height, grid_width), ORANGE, dtype=int)

    for r in range(1, n_rows):
        grid[r * (panel_height + 1) - 1, :] = separator
    for c in range(1, n_cols):
        grid[:, c * (panel_width + 1) - 1] = separator

    for row_idx, row_panels in enumerate(rows):
        for col_idx, panel in enumerate(row_panels):
            r0 = row_idx * (panel_height + 1)
            c0 = col_idx * (panel_width + 1)
            grid[r0 : r0 + panel_height, c0 : c0 + panel_width] = panel

    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    panel_rows = random.randint(*PANEL_ROWS_RANGE)
    panel_cols = random.randint(*PANEL_COLS_RANGE)
    panel_height = random.randint(*PANEL_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)

    input_rows: list[list[np.ndarray]] = []
    output_rows: list[list[np.ndarray]] = []
    for _ in range(panel_rows):
        row_in, row_out = _build_row_panels(panel_height, panel_width, panel_cols)
        input_rows.append(row_in)
        output_rows.append(row_out)

    inp = _render_grid(input_rows, GREY)
    out = _render_grid(output_rows, GREY)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp, expected = _generate_case()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")

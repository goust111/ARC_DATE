import random
from collections import Counter

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in separators:
        if start < idx:
            spans.append((start, idx))
        start = idx + 1
    if start < length:
        spans.append((start, length))
    return spans


def _panel_ranges(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == bg)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == bg)]
    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)
    assert len(row_spans) >= 2, "need at least two panel rows"
    assert len(col_spans) >= 2, "need label column plus at least one data column"
    panel_heights = {r1 - r0 for r0, r1 in row_spans}
    panel_widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(panel_heights) == 1 and len(panel_widths) == 1, "panels must be equal-sized"
    return row_spans, col_spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a panel table, combine each column's repeated template mask with its unique
    solid swatch color, and use the matched row label color as the border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    row_spans, col_spans = _panel_ranges(g, bg)

    panel_h = row_spans[0][1] - row_spans[0][0]
    panel_w = col_spans[0][1] - col_spans[0][0]

    labels: list[int] = []
    for r0, r1 in row_spans:
        label_panel = g[r0:r1, col_spans[0][0] : col_spans[0][1]]
        vals = np.unique(label_panel)
        assert len(vals) == 1 and int(vals[0]) != bg, "row label panel must be solid non-background"
        labels.append(int(vals[0]))

    out_panel_h = panel_h + 2
    out_panel_w = panel_w + 2
    n_cols = len(col_spans) - 1
    out = np.full((out_panel_h, n_cols * out_panel_w + (n_cols - 1)), bg, dtype=int)

    for data_ci, (c0, c1) in enumerate(col_spans[1:]):
        solid_entries: list[tuple[int, int]] = []
        masks: list[np.ndarray] = []
        for ri, (r0, r1) in enumerate(row_spans):
            panel = g[r0:r1, c0:c1]
            vals = set(map(int, np.unique(panel)))
            if bg not in vals:
                assert len(vals) == 1, "solid swatch panel must be monochrome"
                solid_entries.append((ri, next(iter(vals))))
            else:
                vals.discard(bg)
                assert len(vals) == 1, "template panel must contain bg plus one template color"
                mask = panel != bg
                assert np.any(mask) and not np.all(mask), "template mask must be non-empty and not full"
                masks.append(mask)

        assert len(solid_entries) == 1, "each data column needs exactly one solid swatch"
        assert len(masks) == len(row_spans) - 1, "remaining panels must be template masks"
        base_mask = masks[0]
        for mask in masks[1:]:
            assert np.array_equal(mask, base_mask), "template masks in the same column must match"

        solid_row, fill_color = solid_entries[0]
        border_color = labels[solid_row]
        panel_out = np.full((out_panel_h, out_panel_w), bg, dtype=int)
        panel_out[0, :] = border_color
        panel_out[-1, :] = border_color
        panel_out[:, 0] = border_color
        panel_out[:, -1] = border_color
        panel_out[1:-1, 1:-1][base_mask] = fill_color

        dst_c0 = data_ci * (out_panel_w + 1)
        out[:, dst_c0 : dst_c0 + out_panel_w] = panel_out

    return out


_ROWS_RANGE = (2, 4)
_COLS_RANGE = (2, 4)
_PANEL_H_RANGE = (3, 6)
_PANEL_W_RANGE = (3, 6)
_GEN_RETRIES = 400


def _random_mask(h: int, w: int) -> np.ndarray:
    target = random.randint(max(2, (h * w) // 4), max(2, (3 * h * w) // 5))
    seed_r = random.randrange(h)
    seed_c = random.randrange(w)
    cells = {(seed_r, seed_c)}
    frontier = [(seed_r, seed_c)]
    while len(cells) < target and frontier:
        r, c = random.choice(frontier)
        random.shuffle(_DIRS)
        expanded = False
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                cells.add((rr, cc))
                frontier.append((rr, cc))
                expanded = True
                break
        if not expanded:
            frontier.remove((r, c))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    # Occasionally carve a few holes to diversify masks while keeping them readable.
    removable = [(r, c) for r, c in cells if 0 < r < h - 1 and 0 < c < w - 1]
    random.shuffle(removable)
    for r, c in removable[: random.randint(0, min(2, len(removable)))]:
        if mask.sum() <= 2:
            break
        mask[r, c] = False

    if not np.any(mask) or np.all(mask):
        return _random_mask(h, w)
    return mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _build_expected(labels: list[int], swatch_rows: list[int], swatches: list[int], masks: list[np.ndarray], bg: int) -> np.ndarray:
    panel_h, panel_w = masks[0].shape
    out_h = panel_h + 2
    out_w = panel_w + 2
    out = np.full((out_h, len(masks) * out_w + len(masks) - 1), bg, dtype=int)
    for ci, mask in enumerate(masks):
        border = labels[swatch_rows[ci]]
        fill = swatches[ci]
        panel = np.full((out_h, out_w), bg, dtype=int)
        panel[0, :] = border
        panel[-1, :] = border
        panel[:, 0] = border
        panel[:, -1] = border
        panel[1:-1, 1:-1][mask] = fill
        c0 = ci * (out_w + 1)
        out[:, c0 : c0 + out_w] = panel
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*_ROWS_RANGE)
    n_cols = random.randint(*_COLS_RANGE)
    panel_h = random.randint(*_PANEL_H_RANGE)
    panel_w = random.randint(*_PANEL_W_RANGE)

    total_colors = 2 + n_rows + n_cols
    if total_colors > 10:
        return None

    colors = random.sample(range(10), total_colors)
    bg = colors[0]
    template_color = colors[1]
    labels = colors[2 : 2 + n_rows]
    swatches = colors[2 + n_rows :]
    swatch_rows = [random.randrange(n_rows) for _ in range(n_cols)]

    masks = [_random_mask(panel_h, panel_w) for _ in range(n_cols)]

    grid_h = n_rows * panel_h + (n_rows - 1)
    grid_w = (n_cols + 1) * panel_w + n_cols
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for ri in range(n_rows):
        r0 = ri * (panel_h + 1)
        grid[r0 : r0 + panel_h, 0:panel_w] = labels[ri]
        for ci in range(n_cols):
            c0 = (ci + 1) * (panel_w + 1)
            if ri == swatch_rows[ci]:
                grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = swatches[ci]
            else:
                panel = np.full((panel_h, panel_w), bg, dtype=int)
                panel[masks[ci]] = template_color
                grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    expected = _build_expected(labels, swatch_rows, swatches, masks, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")

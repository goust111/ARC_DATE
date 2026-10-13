from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _intervals_from_separators(separators: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(map(bool, separators)):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(separators) - 1))
    return intervals


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        full_rows = int(np.all(grid == color, axis=1).sum())
        full_cols = int(np.all(grid == color, axis=0).sum())
        if full_rows >= 2 and full_cols >= 3:
            candidates.append(color)
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    sep = candidates[0]
    assert np.all(grid[0, :] == sep) and np.all(grid[-1, :] == sep)
    assert np.all(grid[:, 0] == sep) or np.all(grid[:, -1] == sep)
    return sep


def _parse_input(grid: np.ndarray) -> tuple[int, list[int], dict[int, np.ndarray], tuple[int, int]]:
    sep = _separator_color(grid)
    row_bands = _intervals_from_separators(np.all(grid == sep, axis=1))
    col_bands = _intervals_from_separators(np.all(grid == sep, axis=0))
    assert row_bands and len(col_bands) >= 2

    row_heights = [r1 - r0 + 1 for r0, r1 in row_bands]
    assert len(set(row_heights)) == 1
    panel_h = row_heights[0]

    col_widths = [c1 - c0 + 1 for c0, c1 in col_bands]
    legend_band_ids = [i for i, width in enumerate(col_widths) if width == 1]
    assert len(legend_band_ids) == 1
    legend_id = legend_band_ids[0]
    assert legend_id in (0, len(col_bands) - 1)

    panel_band_ids = [i for i in range(len(col_bands)) if i != legend_id]
    panel_widths = [col_widths[i] for i in panel_band_ids]
    assert panel_widths and len(set(panel_widths)) == 1
    panel_w = panel_widths[0]
    assert panel_w > 1

    bg = _most_common(grid[grid != sep])
    legend_c = col_bands[legend_id][0]
    legend_order = [
        int(grid[r, legend_c])
        for r in range(grid.shape[0])
        if int(grid[r, legend_c]) not in (bg, sep)
    ]
    assert len(legend_order) == len(set(legend_order))

    color_to_panel: dict[int, np.ndarray] = {}
    for r0, r1 in row_bands:
        for band_id in panel_band_ids:
            c0, c1 = col_bands[band_id]
            panel = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(panel))) - {bg}
            assert len(colors) == 1
            color = colors.pop()
            assert color != sep
            assert color not in color_to_panel
            color_to_panel[color] = panel.copy()

    panel_colors = set(color_to_panel)
    legend_colors = set(legend_order)
    assert legend_colors < panel_colors
    assert len(panel_colors - legend_colors) == 1
    assert len(legend_order) == len(color_to_panel) - 1
    return bg, legend_order, color_to_panel, (panel_h, panel_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite the equal-size panels into one panel using the side legend's top-to-bottom paint order.

    Input structure:
    - Full-row and full-column separator lines partition the scene into equal-size panels.
    - A one-column legend strip sits on the far left or far right, outside the panel array.
    - Each panel contains exactly one distinct foreground color on a shared background, and the legend lists all but one of those foreground colors.

    Rule:
    - Read the legend colors from top to bottom.
    - Start from an empty panel-sized background canvas.
    - For each legend color in that order, locate the panel whose foreground uses that color and copy all of that panel's foreground pixels onto the canvas, letting later legend entries overwrite earlier ones.
    - Ignore the one panel whose color never appears in the legend.

    Key insight:
    - The legend is both an eligibility list and the only layer-priority cue: panel position and color id do not determine who participates or who ends up on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, legend_order, color_to_panel, (panel_h, panel_w) = _parse_input(g)
    out = np.full((panel_h, panel_w), bg, dtype=int)
    for color in legend_order:
        panel = color_to_panel[color]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


# --------------- generator ---------------

BACKGROUND = 0
LEGEND_WIDTH = 1
SEPARATOR_WIDTH = 1
LAYOUTS = ((2, 2), (1, 4), (4, 1), (2, 3), (3, 2))
LEGEND_SIDES = ("left", "right")
MIN_VISIBLE_CELLS = 3
MAX_VISIBLE_CELLS = 7
MIN_FILL_RATIO = 0.35
MAX_FILL_RATIO = 0.58
MAX_ATTEMPTS = 500


def _grow_blob(free: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    starts = np.argwhere(free)
    if starts.size == 0:
        return None

    r, c = random.choice(starts).tolist()
    cells = [(int(r), int(c))]
    used = {(int(r), int(c))}
    free[int(r), int(c)] = False

    while len(cells) < size:
        frontier = []
        for rr, cc in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in used:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        cells.append((int(nr), int(nc)))
        used.add((int(nr), int(nc)))
        free[int(nr), int(nc)] = False
    return cells


def _sample_visible_sizes(area: int, count: int) -> list[int] | None:
    min_total = max(count * MIN_VISIBLE_CELLS, int(np.ceil(area * MIN_FILL_RATIO)))
    max_total = min(count * MAX_VISIBLE_CELLS, int(np.floor(area * MAX_FILL_RATIO)), area - 5)
    if min_total > max_total:
        return None

    total = random.randint(min_total, max_total)
    sizes = [MIN_VISIBLE_CELLS] * count
    remaining = total - count * MIN_VISIBLE_CELLS
    caps = [MAX_VISIBLE_CELLS - MIN_VISIBLE_CELLS] * count
    while remaining > 0:
        eligible = [i for i, cap in enumerate(caps) if cap > 0]
        if not eligible:
            return None
        i = random.choice(eligible)
        sizes[i] += 1
        caps[i] -= 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def _overlay(order: list[int], masks: list[np.ndarray], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for idx in order:
        out[masks[idx]] = colors[idx]
    return out


def _valid_panel_size(n_rows: int, n_cols: int) -> tuple[int, int]:
    max_h = (30 - (n_rows + 1) * SEPARATOR_WIDTH) // n_rows
    max_w = (30 - LEGEND_WIDTH - (n_cols + 2) * SEPARATOR_WIDTH) // n_cols
    min_h = 5 if n_rows <= 3 else 4
    min_w = 5 if n_cols <= 2 else 4
    assert max_h >= min_h and max_w >= min_w
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _build_masks(
    panel_h: int,
    panel_w: int,
    panel_indices: list[int],
    colors: list[int],
    legend_order: list[int],
    absent_idx: int,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    n_panels = len(panel_indices)
    listed = legend_order
    listed_count = len(listed)
    area = panel_h * panel_w
    visible_sizes = _sample_visible_sizes(area, listed_count)
    if visible_sizes is None:
        return None

    owner = np.full((panel_h, panel_w), -1, dtype=int)
    free = np.ones((panel_h, panel_w), dtype=bool)
    rank_to_cells: dict[int, list[tuple[int, int]]] = {}
    for rank, size in enumerate(visible_sizes):
        cells = _grow_blob(free, size)
        if cells is None:
            return None
        rank_to_cells[rank] = cells
        rr, cc = np.array(cells).T
        owner[rr, cc] = rank

    masks = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(n_panels)]
    for rank, panel_idx in enumerate(listed):
        rr, cc = np.array(rank_to_cells[rank]).T
        masks[panel_idx][rr, cc] = True

    row_major = sorted(listed)
    row_major_pairs = [(ra, rb) for ra in range(listed_count) for rb in range(ra + 1, listed_count) if listed[ra] > listed[rb]]
    if not row_major_pairs:
        return None
    ra, rb = random.choice(row_major_pairs)
    overlap_cell = random.choice(rank_to_cells[rb])
    masks[listed[ra]][overlap_cell] = True

    color_order = sorted(listed, key=lambda idx: colors[idx])
    color_pairs = [
        (ra, rb)
        for ra in range(listed_count)
        for rb in range(ra + 1, listed_count)
        if colors[listed[ra]] > colors[listed[rb]]
    ]
    if not color_pairs:
        return None
    ra2, rb2 = random.choice(color_pairs)
    overlap_cell_2 = random.choice(rank_to_cells[rb2])
    masks[listed[ra2]][overlap_cell_2] = True

    for later_rank in range(1, listed_count):
        if random.random() < 0.65:
            earlier_rank = random.randrange(later_rank)
            extra_cell = random.choice(rank_to_cells[later_rank])
            masks[listed[earlier_rank]][extra_cell] = True

    free_bg = owner == -1
    absent_size_max = min(MAX_VISIBLE_CELLS, int(free_bg.sum()) - 2)
    if absent_size_max < MIN_VISIBLE_CELLS:
        return None
    absent_cells = _grow_blob(free_bg.copy(), random.randint(MIN_VISIBLE_CELLS, absent_size_max))
    if absent_cells is None:
        return None
    rr, cc = np.array(absent_cells).T
    masks[absent_idx][rr, cc] = True

    expected = _overlay(listed, masks, colors, (panel_h, panel_w))
    visible_colors = set(map(int, np.unique(expected))) - {BACKGROUND}
    if visible_colors != {colors[idx] for idx in listed}:
        return None

    absent_only = masks[absent_idx].copy()
    for idx in listed:
        absent_only &= ~masks[idx]
    if int(absent_only.sum()) == 0:
        return None

    if np.array_equal(expected, _overlay(row_major, masks, colors, (panel_h, panel_w))):
        return None
    if np.array_equal(expected, _overlay(color_order, masks, colors, (panel_h, panel_w))):
        return None
    if np.array_equal(expected, _overlay([absent_idx] + listed, masks, colors, (panel_h, panel_w))):
        return None
    if np.array_equal(expected, _overlay(listed + [absent_idx], masks, colors, (panel_h, panel_w))):
        return None

    return masks, expected


def _render_input(
    n_rows: int,
    n_cols: int,
    panel_h: int,
    panel_w: int,
    separator: int,
    colors: list[int],
    masks: list[np.ndarray],
    legend_order: list[int],
    legend_side: str,
) -> np.ndarray:
    height = n_rows * panel_h + (n_rows + 1) * SEPARATOR_WIDTH
    width = n_cols * panel_w + LEGEND_WIDTH + (n_cols + 2) * SEPARATOR_WIDTH
    grid = np.full((height, width), separator, dtype=int)

    row_starts = [1 + r * (panel_h + SEPARATOR_WIDTH) for r in range(n_rows)]
    if legend_side == "left":
        legend_c = 1
        col_starts = [3 + c * (panel_w + SEPARATOR_WIDTH) for c in range(n_cols)]
    else:
        col_starts = [1 + c * (panel_w + SEPARATOR_WIDTH) for c in range(n_cols)]
        legend_c = col_starts[-1] + panel_w + 1

    for panel_idx in range(n_rows * n_cols):
        pr = panel_idx // n_cols
        pc = panel_idx % n_cols
        r0 = row_starts[pr]
        c0 = col_starts[pc]
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = BACKGROUND
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w][masks[panel_idx]] = colors[panel_idx]

    legend_rows_pool = [r for r0 in row_starts for r in range(r0, r0 + panel_h)]
    legend_rows = sorted(random.sample(legend_rows_pool, len(legend_order)))
    for r in legend_rows_pool:
        grid[r, legend_c] = BACKGROUND
    for r, panel_idx in zip(legend_rows, legend_order):
        grid[r, legend_c] = colors[panel_idx]
    return grid


def _attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    n_panels = n_rows * n_cols
    panel_h, panel_w = _valid_panel_size(n_rows, n_cols)

    separator = random.randint(1, 9)
    panel_colors = random.sample([c for c in range(1, 10) if c != separator], n_panels)
    absent_idx = random.randrange(n_panels)
    legend_order = [idx for idx in range(n_panels) if idx != absent_idx]
    random.shuffle(legend_order)
    if legend_order == sorted(legend_order):
        return None
    if legend_order == sorted(legend_order, key=lambda idx: panel_colors[idx]):
        return None

    built = _build_masks(panel_h, panel_w, list(range(n_panels)), panel_colors, legend_order, absent_idx)
    if built is None:
        return None
    masks, expected = built

    grid = _render_input(
        n_rows=n_rows,
        n_cols=n_cols,
        panel_h=panel_h,
        panel_w=panel_w,
        separator=separator,
        colors=panel_colors,
        masks=masks,
        legend_order=legend_order,
        legend_side=random.choice(LEGEND_SIDES),
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(np.unique(actual).size - (BACKGROUND in np.unique(actual))) < min(3, n_panels - 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

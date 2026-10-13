from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- solver ---------------


def _separator_runs(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    col = 0
    width = int(separator_mask.size)
    while col < width:
        if not bool(separator_mask[col]):
            col += 1
            continue
        start = col
        while col < width and bool(separator_mask[col]):
            col += 1
        runs.append((start, col))
    return runs


def _panel_slices(grid: np.ndarray) -> list[slice]:
    separator_mask = np.all(grid == GREY, axis=0)
    assert bool(separator_mask.any()), "expected at least one grey separator column"
    assert not bool(separator_mask[0]) and not bool(separator_mask[-1]), "separators cannot touch the border"

    runs = _separator_runs(separator_mask)
    panels: list[slice] = []
    start = 0
    for left, right in runs:
        assert start < left, "separator columns must leave a non-empty panel before them"
        panels.append(slice(start, left))
        start = right
    assert start < grid.shape[1], "separator columns must leave a non-empty trailing panel"
    panels.append(slice(start, grid.shape[1]))

    widths = [panel.stop - panel.start for panel in panels]
    assert 3 <= len(panels) <= 5, "expected 3 to 5 panels"
    assert len(set(widths)) == 1, "panels must have equal width"
    for panel in panels:
        assert not bool(np.any(grid[:, panel] == GREY)), "grey is reserved for separator columns"
    return panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the least frequent non-black color from each panel and output them in panel order.

    Input structure:
    - Background is black.
    - One or more full-height grey separator columns split the grid into 3-5 equal-width panels.
    - Each panel contains 2-4 non-black, non-grey colors with distinct pixel counts.

    Rule:
    - Identify the ordered panels using the grey separator columns.
    - In each panel, count every non-black, non-grey color.
    - Pick the uniquely least frequent color.
    - Return a 1xN grid whose entries are those rare colors in panel order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _panel_slices(g)
    out = np.empty((1, len(panels)), dtype=int)
    for index, panel_slice in enumerate(panels):
        panel = g[:, panel_slice]
        values = panel[(panel != BLACK) & (panel != GREY)]
        assert values.size > 0, "each panel must contain foreground colors"

        counts = Counter(map(int, values.tolist()))
        assert 2 <= len(counts) <= 4, "each panel must contain 2-4 colors"
        assert len(set(counts.values())) == len(counts), "panel color counts must be distinct"

        least_count = min(counts.values())
        rare_colors = [color for color, count in counts.items() if count == least_count]
        assert len(rare_colors) == 1, "least frequent color must be unique"
        out[0, index] = rare_colors[0]

    return out


# --------------- generator ---------------

PANEL_COUNT_CHOICES = (3, 4, 5)
GRID_HEIGHT_RANGE = (6, 12)
PANEL_WIDTH_RANGE = (5, 8)
SEPARATOR_WIDTH_CHOICES = (1, 2)
PANEL_COLOR_COUNT_CHOICES = (2, 3, 4)
COUNT_SAMPLE_RETRIES = 200
MAX_GENERATION_ATTEMPTS = 300


def _sample_counts(area: int, n_colors: int) -> list[int] | None:
    min_fill = n_colors * (n_colors + 1) // 2
    max_fill = max(min_fill, int(area * 0.72))
    upper = min(12, area - (min_fill - 1))
    if upper < n_colors:
        return None

    for _ in range(COUNT_SAMPLE_RETRIES):
        counts = random.sample(range(1, upper + 1), n_colors)
        total = sum(counts)
        if min_fill <= total <= max_fill:
            return counts
    return None


def _paint_panel(
    grid: np.ndarray,
    panel_slice: slice,
    colors: list[int],
    counts: list[int],
) -> None:
    height = int(grid.shape[0])
    panel_width = int(panel_slice.stop - panel_slice.start)
    area = height * panel_width
    cells = random.sample(range(area), sum(counts))
    random.shuffle(cells)

    offset = 0
    for color, count in zip(colors, counts):
        for cell in cells[offset : offset + count]:
            row = cell // panel_width
            col = cell % panel_width
            grid[row, panel_slice.start + col] = color
        offset += count


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice(PANEL_COUNT_CHOICES)
    height = random.randint(*GRID_HEIGHT_RANGE)
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    separator_width = random.choice(SEPARATOR_WIDTH_CHOICES)

    width = panel_count * panel_width + (panel_count - 1) * separator_width
    grid = np.full((height, width), BLACK, dtype=int)

    panels: list[slice] = []
    col = 0
    for panel_index in range(panel_count):
        panels.append(slice(col, col + panel_width))
        col += panel_width
        if panel_index != panel_count - 1:
            grid[:, col : col + separator_width] = GREY
            col += separator_width

    for panel_slice in panels:
        n_colors = random.choice(PANEL_COLOR_COUNT_CHOICES)
        colors = random.sample(list(NON_SPECIAL_COLORS), n_colors)
        counts = _sample_counts(height * panel_width, n_colors)
        if counts is None:
            return None
        _paint_panel(grid, panel_slice, colors, counts)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (1, panel_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")

from __future__ import annotations

import numpy as np


PANEL_SIZE = 6
PANEL_COUNT = 4
SEPARATOR_COLOR = 5
BAR_COLOR = 7
PANEL_COLORS = (1, 2, 3, 4)
GRID_SHAPE = (PANEL_SIZE, PANEL_COUNT * PANEL_SIZE + (PANEL_COUNT - 1))
SEPARATOR_COLS = (6, 13, 20)
PANEL_STARTS = (0, 7, 14, 21)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < PANEL_SIZE:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < PANEL_SIZE:
        out.append((r, c + 1))
    return out


def _panel_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, grid.shape)
    assert height == PANEL_SIZE and width == GRID_SHAPE[1], "expected a 6x27 input grid"
    separator_cols = [c for c in range(width) if np.all(grid[:, c] == SEPARATOR_COLOR)]
    assert separator_cols == list(SEPARATOR_COLS), "expected three full-height grey separator columns"

    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separator_cols:
        spans.append((start, sep))
        start = sep + 1
    spans.append((start, width))
    assert len(spans) == PANEL_COUNT and all(stop - start == PANEL_SIZE for start, stop in spans)
    return spans


def _extract_panel(panel: np.ndarray) -> tuple[int, np.ndarray]:
    orange = np.argwhere(panel == BAR_COLOR)
    assert orange.shape[0] >= 1, "each panel must contain one orange priority bar"
    orange_cols = np.unique(orange[:, 1])
    assert orange_cols.shape[0] == 1, "priority bar must be one column wide"
    bar_col = int(orange_cols[0])
    orange_rows = np.sort(orange[:, 0])
    bar_height = int(orange_rows.shape[0])
    assert np.array_equal(orange_rows, np.arange(bar_height)), "priority bar must be top-aligned and solid"
    assert not np.any(panel[:, bar_col] == SEPARATOR_COLOR), "separator color must not appear inside panels"

    pattern = panel.copy()
    pattern[pattern == BAR_COLOR] = 0
    colors = sorted(set(map(int, pattern.flatten())) - {0})
    assert len(colors) == 1, "each panel must contain exactly one non-black pattern color"
    return bar_height, pattern


def _overlay_patterns(patterns: list[np.ndarray], order: list[int]) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    filled = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for panel_index in order:
        mask = (patterns[panel_index] != 0) & (~filled)
        out[mask] = patterns[panel_index][mask]
        filled |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay four aligned panels into one image using orange bar heights as front-to-back priority.

    Input structure:
    - The input is a 6x27 grid split by three full-height grey separator columns into four 6x6 panels.
    - Each panel contains black background, one distinct pattern color from {1,2,3,4}, and one top-aligned orange vertical bar whose height is between 1 and 6.
    - The orange bar never belongs to the final picture; it only encodes that panel's layer priority.

    Rule:
    - Measure the orange bar height in each panel and sort the panels from tallest bar to shortest bar.
    - Erase the orange bars and ignore the grey separator columns.
    - Overlay the four 6x6 panel patterns in that front-to-back order, treating black as transparent and taking the first non-black color at each cell.

    Key insight:
    - The orange bars are hidden instructions rather than drawable objects: their relative heights determine which aligned panel is visually in front after transparent overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    patterns: list[np.ndarray] = []
    heights: list[int] = []
    for start, stop in _panel_spans(g):
        panel = g[:, start:stop]
        bar_height, pattern = _extract_panel(panel)
        heights.append(bar_height)
        patterns.append(pattern)

    assert len(set(heights)) == PANEL_COUNT, "priority bar heights must be distinct"
    order = sorted(range(PANEL_COUNT), key=lambda idx: -heights[idx])
    return _overlay_patterns(patterns, order)


def _grow_region(area: int, occupied: np.ndarray) -> np.ndarray | None:
    available = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE) if not occupied[r, c]]
    if len(available) < area:
        return None

    available_set = set(available)
    region = {available[int(np.random.randint(len(available)))]}
    target_components = 1 + int(np.random.rand() < 0.35)
    started_components = 1

    while len(region) < area:
        frontier = {
            nbr
            for r, c in region
            for nbr in _neighbors4(r, c)
            if nbr in available_set and nbr not in region
        }
        if frontier and (started_components >= target_components or np.random.rand() < 0.85):
            frontier_list = list(frontier)
            region.add(frontier_list[int(np.random.randint(len(frontier_list)))])
            continue

        remaining = list(available_set - region)
        if not remaining:
            return None
        region.add(remaining[int(np.random.randint(len(remaining)))])
        if not frontier:
            started_components += 1

    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r, c in region:
        mask[r, c] = True
    return mask


def _overlay_from_masks(masks: list[np.ndarray], colors: list[int], order: list[int]) -> np.ndarray:
    patterns = [mask.astype(int) * color for mask, color in zip(masks, colors)]
    return _overlay_patterns(patterns, order)


def _render_input(masks: list[np.ndarray], colors: list[int], heights: list[int], bar_cols: list[int]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, SEPARATOR_COLS] = SEPARATOR_COLOR
    for panel_index, start in enumerate(PANEL_STARTS):
        panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
        panel[masks[panel_index]] = colors[panel_index]
        panel[: heights[panel_index], bar_cols[panel_index]] = BAR_COLOR
        grid[:, start : start + PANEL_SIZE] = panel
    return grid


def _gen_attempt() -> np.ndarray | None:
    heights = list(map(int, np.random.choice(np.arange(1, PANEL_SIZE + 1), size=PANEL_COUNT, replace=False)))
    order = sorted(range(PANEL_COUNT), key=lambda idx: -heights[idx])
    if order == [0, 1, 2, 3] or order == [3, 2, 1, 0]:
        return None

    colors = list(map(int, np.random.permutation(PANEL_COLORS)))
    visible_areas = np.random.randint(2, 6, size=PANEL_COUNT)
    if not 10 <= int(np.sum(visible_areas)) <= 18:
        return None

    visible_masks = [np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool) for _ in range(PANEL_COUNT)]
    occupied = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for panel_index in np.random.permutation(PANEL_COUNT):
        mask = _grow_region(int(visible_areas[panel_index]), occupied)
        if mask is None:
            return None
        visible_masks[panel_index] = mask
        occupied |= mask

    masks = [mask.copy() for mask in visible_masks]

    for order_index in range(PANEL_COUNT - 1):
        front = order[order_index]
        back = order[order_index + 1]
        front_cells = np.argwhere(visible_masks[front])
        np.random.shuffle(front_cells)
        overlap_count = 1 + int(np.random.rand() < 0.45)
        added = 0
        for r, c in front_cells:
            if not masks[back][r, c]:
                masks[back][r, c] = True
                added += 1
            if added == overlap_count:
                break
        if added == 0:
            return None

    for order_index in range(1, PANEL_COUNT):
        panel = order[order_index]
        earlier_union = np.any(np.stack([visible_masks[p] for p in order[:order_index]], axis=0), axis=0)
        earlier_cells = np.argwhere(earlier_union)
        np.random.shuffle(earlier_cells)
        extra_count = int(np.random.randint(0, 3))
        added = 0
        for r, c in earlier_cells:
            if not masks[panel][r, c]:
                masks[panel][r, c] = True
                added += 1
            if added == extra_count:
                break

    if any(int(mask.sum()) > 8 for mask in masks):
        return None

    bar_cols: list[int] = []
    for panel_index in range(PANEL_COUNT):
        empty_cols = [c for c in range(PANEL_SIZE) if not np.any(masks[panel_index][:, c])]
        if not empty_cols:
            return None
        bar_cols.append(int(empty_cols[int(np.random.randint(len(empty_cols)))]))

    output = _overlay_from_masks(masks, colors, order)
    if not 10 <= int(np.count_nonzero(output)) <= 18:
        return None

    for panel_index in range(PANEL_COUNT):
        if int(np.count_nonzero(output == colors[panel_index])) < 2:
            return None

    for order_index in range(PANEL_COUNT - 1):
        swapped = order.copy()
        swapped[order_index], swapped[order_index + 1] = swapped[order_index + 1], swapped[order_index]
        if np.array_equal(output, _overlay_from_masks(masks, colors, swapped)):
            return None

    grid = _render_input(masks, colors, heights, bar_cols)
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

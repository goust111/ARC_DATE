from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SHAPE = 1
SEP = 2
LEGEND_COLORS = (3, 4, 6, 7)

GEN_ATTEMPTS = 400
SLOT_SIZE = 5
PANEL_MARGIN = 1
SLOT_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_sort_key(coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return (-int(coords.shape[0]), int(r0), int(c0))


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two blue side panels, keep only their intersection regions, then recolor those regions by area rank using the legend.

    Input structure:
    - The grid has black background, two equal-width side panels, and a red vertical separator column between them.
    - The panel area contains only blue foreground shapes on black, while a horizontal legend row below the panels contains 2-4 distinct single-pixel colors chosen from green, yellow, magenta, and orange.
    - Left and right panels share some overlapping blue cells when aligned onto the same coordinate system.

    Rule:
    - Remove the separator and compare the two panels cell by cell on their shared coordinates.
    - Keep only cells that are blue in both panels, split that intersection mask into 4-connected regions, and sort the regions by descending area, then by topmost row and leftmost column.
    - Recolor rank 1, rank 2, rank 3, ... regions with the legend colors read from left to right, and return only the recolored intersection cropped to its tight bounding box on a black background.

    Key insight:
    - The logical intersection is only an intermediate object generator; the final colors come from ranking the newly created intersection regions against the visible legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep_candidates = [c for c in range(1, w - 1) if np.count_nonzero(g[:, c] == SEP) >= 2]
    assert sep_candidates, "expected a vertical separator column"
    sep_col = max(sep_candidates, key=lambda c: int(np.count_nonzero(g[:, c] == SEP)))

    panel_rows = np.flatnonzero(g[:, sep_col] == SEP)
    assert panel_rows.size > 0, "separator must span the panel rows"
    r0 = int(panel_rows.min())
    r1 = int(panel_rows.max())
    assert np.all(g[r0 : r1 + 1, sep_col] == SEP)

    left = g[r0 : r1 + 1, :sep_col]
    right = g[r0 : r1 + 1, sep_col + 1 :]
    assert left.shape == right.shape

    legend_pixels = [
        (int(r), int(c), int(g[r, c]))
        for r in range(h)
        for c in range(w)
        if int(g[r, c]) in LEGEND_COLORS
    ]
    assert len(legend_pixels) >= 2, "expected a color legend"
    legend_pixels.sort(key=lambda item: (item[0], item[1]))
    legend = [color for _r, _c, color in legend_pixels]

    inter = (left == SHAPE) & (right == SHAPE)
    components = _find_components(inter)
    assert components, "expected at least one intersection region"
    components.sort(key=_component_sort_key)
    assert len(legend) >= len(components), "legend must cover every ranked region"

    out = np.zeros(left.shape, dtype=int)
    for color, coords in zip(legend, components):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return _crop_nonzero(out)


def _random_connected_shape(area: int, h: int, w: int, *, margin: int = 1) -> set[tuple[int, int]] | None:
    rows = list(range(margin, h - margin))
    cols = list(range(margin, w - margin))
    if not rows or not cols:
        return None
    cells = [(r, c) for r in rows for c in cols]
    start = random.choice(cells)
    shape = {start}
    while len(shape) < area:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if margin <= nr < h - margin and margin <= nc < w - margin and (nr, nc) not in shape:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))
    return shape


def _grow_superset(
    base: set[tuple[int, int]],
    target_area: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    shape = set(base)
    while len(shape) < target_area:
        frontier = []
        for r, c in shape:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shape and (nr, nc) not in forbidden:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(frontier))
    return shape


def _panel_geometry() -> tuple[int, int, list[tuple[int, int]]]:
    slot_rows = 2
    slot_cols = 2
    panel_h = PANEL_MARGIN * 2 + slot_rows * SLOT_SIZE + (slot_rows - 1) * SLOT_GAP
    panel_w = PANEL_MARGIN * 2 + slot_cols * SLOT_SIZE + (slot_cols - 1) * SLOT_GAP
    slots = []
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            top = PANEL_MARGIN + sr * (SLOT_SIZE + SLOT_GAP)
            left = PANEL_MARGIN + sc * (SLOT_SIZE + SLOT_GAP)
            slots.append((top, left))
    return panel_h, panel_w, slots


def _build_slot_component(area: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(80):
        base = _random_connected_shape(area, SLOT_SIZE, SLOT_SIZE, margin=1)
        if base is None:
            continue

        min_target = max(3, area)
        max_target = min(8, area + 3)
        left_target = random.randint(min_target, max_target)
        right_target = random.randint(min_target, max_target)

        left_shape = _grow_superset(base, left_target, SLOT_SIZE, SLOT_SIZE, set())
        if left_shape is None:
            continue
        left_extra = left_shape - base

        right_shape = _grow_superset(base, right_target, SLOT_SIZE, SLOT_SIZE, left_extra)
        if right_shape is None:
            continue

        if len(left_shape) < 3 or len(right_shape) < 3:
            continue
        return base, left_shape, right_shape
    return None


def _paint_cells(mask: np.ndarray, cells: set[tuple[int, int]], offset_r: int, offset_c: int):
    for r, c in cells:
        mask[offset_r + r, offset_c + c] = True


def _build_panels(
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    panel_h, panel_w, slots = _panel_geometry()
    picked_slots = random.sample(slots, n_components)
    random.shuffle(picked_slots)

    areas = random.sample([2, 3, 4, 5], n_components)
    left_mask = np.zeros((panel_h, panel_w), dtype=bool)
    right_mask = np.zeros((panel_h, panel_w), dtype=bool)
    inter_mask = np.zeros((panel_h, panel_w), dtype=bool)

    any_difference = False
    for (slot_r, slot_c), area in zip(picked_slots, areas):
        built = _build_slot_component(area)
        if built is None:
            return None
        base, left_shape, right_shape = built
        any_difference = any_difference or (left_shape != right_shape)

        _paint_cells(inter_mask, base, slot_r, slot_c)
        _paint_cells(left_mask, left_shape, slot_r, slot_c)
        _paint_cells(right_mask, right_shape, slot_r, slot_c)

    if not any_difference:
        return None

    inter_components = _find_components(inter_mask)
    left_components = _find_components(left_mask)
    right_components = _find_components(right_mask)
    if len(inter_components) != n_components or len(left_components) != n_components or len(right_components) != n_components:
        return None

    inter_areas = [int(comp.shape[0]) for comp in inter_components]
    if len(set(inter_areas)) != n_components:
        return None
    if any(not (3 <= int(comp.shape[0]) <= 8) for comp in left_components + right_components):
        return None

    return left_mask, right_mask, inter_mask


def _render_expected(inter_mask: np.ndarray, legend: list[int]) -> np.ndarray:
    comps = _find_components(inter_mask)
    comps.sort(key=_component_sort_key)
    out = np.zeros_like(inter_mask, dtype=int)
    for color, coords in zip(legend, comps):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return _crop_nonzero(out)


def _place_legend(total_w: int, legend_len: int, sep_col: int) -> int:
    valid = [
        start
        for start in range(0, total_w - legend_len + 1)
        if not (start <= sep_col < start + legend_len)
    ]
    if not valid:
        valid = list(range(0, total_w - legend_len + 1))
    return random.choice(valid)


def _gen_attempt() -> np.ndarray | None:
    n_components = random.randint(2, 4)
    built = _build_panels(n_components)
    if built is None:
        return None
    left_mask, right_mask, inter_mask = built

    panel_h, panel_w = left_mask.shape
    sep_col = panel_w
    total_w = panel_w * 2 + 1

    gap_rows = random.randint(0, 2)
    tail_rows = random.randint(0, 1)
    legend_row = panel_h + gap_rows
    total_h = legend_row + 1 + tail_rows
    if total_h > 30 or total_w > 30 or total_w <= total_h:
        return None

    legend_len = random.randint(n_components, 4)
    legend = random.sample(list(LEGEND_COLORS), legend_len)
    legend_start = _place_legend(total_w, legend_len, sep_col)

    grid = np.zeros((total_h, total_w), dtype=int)
    grid[:panel_h, :panel_w][left_mask] = SHAPE
    grid[:panel_h, sep_col + 1 :][right_mask] = SHAPE
    grid[:panel_h, sep_col] = SEP

    for idx, color in enumerate(legend):
        grid[legend_row, legend_start + idx] = int(color)

    expected = _render_expected(inter_mask, legend)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

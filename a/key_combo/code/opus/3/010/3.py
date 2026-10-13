from __future__ import annotations

import itertools
import random

import numpy as np


PANEL_H = 5
PANEL_W = 4
PANEL_COUNT = 4
LEFT_REGION_W = 19
PANEL_STRIDE = 5
GAP_W = 2
SEPARATOR_COLOR = 6
PANEL_COLORS = (1, 2, 3, 4)
PERMUTATIONS = tuple(itertools.permutations(range(PANEL_COUNT)))
MAX_ATTEMPTS = 300


# Each family guarantees one private cell per panel and one pair-only overlap cell
# per panel pair, so all 24 overlay orders stay distinguishable.
BASE_FAMILIES = (
    (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 2), (1, 2), (2, 1), (2, 2)),
    ),
    (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 2), (1, 2), (1, 3), (2, 2)),
    ),
    (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 2), (1, 2), (2, 2), (2, 3)),
    ),
    (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
    ),
    (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 3), (1, 2), (1, 3), (2, 2)),
    ),
)


def _apply_symmetry(arr: np.ndarray, mode: int) -> np.ndarray:
    out = np.array(arr, dtype=bool, copy=True)
    if mode >= 4:
        out = np.fliplr(out)
    out = np.rot90(out, mode % 4)
    return np.array(out, dtype=bool, copy=True)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _overlay_masks(masks: list[np.ndarray], order: tuple[int, ...], colors: list[int] | None = None) -> np.ndarray:
    if colors is None:
        colors = [1, 2, 3, 4]
    out = np.zeros((PANEL_H, PANEL_W), dtype=int)
    for idx in order:
        out[masks[idx]] = int(colors[idx])
    return out


def _family_to_arrays(family: tuple[tuple[tuple[int, int], ...], ...]) -> list[np.ndarray]:
    max_r = max(r for panel in family for r, _ in panel)
    max_c = max(c for panel in family for _, c in panel)
    arrays: list[np.ndarray] = []
    for panel in family:
        arr = np.zeros((max_r + 1, max_c + 1), dtype=bool)
        for r, c in panel:
            arr[int(r), int(c)] = True
        arrays.append(arr)
    return arrays


def _materialize_panel_masks() -> list[np.ndarray]:
    family = random.choice(BASE_FAMILIES)
    base_arrays = _family_to_arrays(family)
    modes = list(range(8))
    random.shuffle(modes)
    for mode in modes:
        transformed = [_apply_symmetry(arr, mode) for arr in base_arrays]
        h, w = transformed[0].shape
        if h > PANEL_H or w > PANEL_W:
            continue
        row_offset = random.randint(0, PANEL_H - h)
        col_offset = random.randint(0, PANEL_W - w)
        masks: list[np.ndarray] = []
        for arr in transformed:
            mask = np.zeros((PANEL_H, PANEL_W), dtype=bool)
            mask[row_offset : row_offset + h, col_offset : col_offset + w] = arr
            masks.append(mask)

        occupied = np.any(np.stack(masks, axis=0), axis=0)
        panel_order = list(range(PANEL_COUNT))
        random.shuffle(panel_order)
        for idx in panel_order:
            extra_budget = random.randint(0, 2)
            for _ in range(extra_budget):
                frontier: list[tuple[int, int]] = []
                for r, c in np.argwhere(masks[idx]):
                    rr = int(r)
                    cc = int(c)
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < PANEL_H and 0 <= nc < PANEL_W and not occupied[nr, nc]:
                            frontier.append((int(nr), int(nc)))
                if not frontier:
                    break
                nr, nc = random.choice(frontier)
                masks[idx][nr, nc] = True
                occupied[nr, nc] = True

        if all(3 <= int(mask.sum()) <= 6 and _is_connected(mask) for mask in masks):
            signatures = {_overlay_masks(masks, perm).tobytes() for perm in PERMUTATIONS}
            if len(signatures) == len(PERMUTATIONS):
                return masks
    raise RuntimeError("failed to materialize a valid panel family")


def _colored_panels_from_masks(masks: list[np.ndarray], colors: list[int]) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    for mask, color in zip(masks, colors):
        panel = np.zeros((PANEL_H, PANEL_W), dtype=int)
        panel[mask] = int(color)
        panels.append(panel)
    return panels


def _panel_foreground_color(panel: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(panel) if int(color) != 0]
    assert len(colors) == 1
    return colors[0]


def _extract_panel_masks(grid: np.ndarray) -> tuple[list[np.ndarray], list[int], int, int]:
    sep_coords = np.argwhere(grid == SEPARATOR_COLOR)
    assert len(sep_coords) == 15
    rows = sorted({int(r) for r, _ in sep_coords})
    cols = sorted({int(c) for _, c in sep_coords})
    assert len(rows) == PANEL_H and rows == list(range(rows[0], rows[0] + PANEL_H))
    assert len(cols) == PANEL_COUNT - 1 and cols[1] - cols[0] == PANEL_STRIDE and cols[2] - cols[1] == PANEL_STRIDE
    top = rows[0]
    left = cols[0] - PANEL_W
    assert left >= 0
    assert left + LEFT_REGION_W <= grid.shape[1]

    panels: list[np.ndarray] = []
    colors: list[int] = []
    for idx in range(PANEL_COUNT):
        c0 = left + idx * PANEL_STRIDE
        panel = grid[top : top + PANEL_H, c0 : c0 + PANEL_W].copy()
        color = _panel_foreground_color(panel)
        mask = panel == color
        assert 3 <= int(mask.sum()) <= 6
        assert _is_connected(mask)
        panels.append(panel)
        colors.append(color)
    assert len(set(colors)) == PANEL_COUNT
    return panels, colors, top, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read marker-column order and transparently overlay the matching panel sprites.

    Input structure:
    - A 5x19 left region is split by three full-height magenta separator columns into four 5x4 panels.
    - Each panel has black background and exactly one non-black foreground color from {1,2,3,4}.
    - To the right, after two black columns, four scattered singleton markers reuse those four panel colors.

    Rule:
    - Extract the four panels from left to right and identify each panel by its unique foreground color.
    - Read the four external markers, sort them by column from left to right, and use that color order as bottom-to-top layer priority.
    - Start from a black 5x4 output and overlay the matching panels in that order, treating black as transparent.

    Key insight:
    - The panel positions do not determine overlay priority; the right-side markers encode that priority through horizontal order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    panels, panel_colors, top, left = _extract_panel_masks(g)
    left_mask = np.zeros(g.shape, dtype=bool)
    left_mask[top : top + PANEL_H, left : left + LEFT_REGION_W] = True

    marker_coords = np.argwhere((g != 0) & (g != SEPARATOR_COLOR) & (~left_mask))
    assert len(marker_coords) == PANEL_COUNT
    markers = [(int(r), int(c), int(g[int(r), int(c)])) for r, c in marker_coords]
    assert len({r for r, _, _ in markers}) == PANEL_COUNT
    assert len({c for _, c, _ in markers}) == PANEL_COUNT
    assert sorted(color for _, _, color in markers) == sorted(panel_colors)

    panels_by_color = {_panel_foreground_color(panel): panel for panel in panels}
    out = np.zeros((PANEL_H, PANEL_W), dtype=int)
    for _, _, color in sorted(markers, key=lambda item: item[1]):
        panel = panels_by_color[color]
        mask = panel != 0
        out[mask] = panel[mask]
    return out


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    masks = _materialize_panel_masks()
    panel_colors = random.sample(PANEL_COLORS, PANEL_COUNT)
    panels = _colored_panels_from_masks(masks, panel_colors)

    layer_order = random.choice([perm for perm in PERMUTATIONS if perm != tuple(range(PANEL_COUNT))])
    expected = _overlay_masks(masks, layer_order, panel_colors)
    if len({int(color) for color in expected.ravel() if int(color) != 0}) != PANEL_COUNT:
        raise RuntimeError("every layer should remain visible")

    top_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 1)
    if top_pad + bottom_pad == 0:
        top_pad = 1
    left_pad = random.randint(0, 1)
    right_width = random.randint(5, 6)
    right_pad = random.randint(0, 1)

    height = top_pad + PANEL_H + bottom_pad
    width = left_pad + LEFT_REGION_W + GAP_W + right_width + right_pad
    grid = np.zeros((height, width), dtype=int)

    top = top_pad
    left = left_pad
    for sep_col in (left + PANEL_W, left + PANEL_W + PANEL_STRIDE, left + PANEL_W + 2 * PANEL_STRIDE):
        grid[top : top + PANEL_H, sep_col] = SEPARATOR_COLOR

    for idx, panel in enumerate(panels):
        c0 = left + idx * PANEL_STRIDE
        block = grid[top : top + PANEL_H, c0 : c0 + PANEL_W]
        mask = panel != 0
        block[mask] = panel[mask]

    marker_colors = [panel_colors[idx] for idx in layer_order]
    marker_rows = random.sample(range(PANEL_H), PANEL_COUNT)
    marker_cols = sorted(random.sample(range(right_width), PANEL_COUNT))
    marker_left = left + LEFT_REGION_W + GAP_W
    for color, row, col in zip(marker_colors, marker_rows, marker_cols):
        grid[top + row, marker_left + col] = int(color)

    return grid, expected


def _validate_generated_sample(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.shape != (PANEL_H, PANEL_W):
        return False
    if max(grid.shape) > 30:
        return False

    try:
        panels, panel_colors, top, left = _extract_panel_masks(grid)
    except Exception:
        return False

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[top : top + PANEL_H, left : left + LEFT_REGION_W] = True
    marker_coords = np.argwhere((grid != 0) & (grid != SEPARATOR_COLOR) & (~left_mask))
    if len(marker_coords) != PANEL_COUNT:
        return False
    markers = [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in marker_coords]
    if len({r for r, _, _ in markers}) != PANEL_COUNT or len({c for _, c, _ in markers}) != PANEL_COUNT:
        return False

    masks = [panel != 0 for panel in panels]
    if len({_overlay_masks(masks, perm).tobytes() for perm in PERMUTATIONS}) != len(PERMUTATIONS):
        return False

    sorted_markers = sorted(markers, key=lambda item: item[1])
    layer_colors = [color for _, _, color in sorted_markers]
    if layer_colors == panel_colors:
        return False
    if len(set(int(color) for color in expected.ravel() if int(color) != 0)) != PANEL_COUNT:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected = _build_input()
        except Exception:
            continue
        if _validate_generated_sample(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")

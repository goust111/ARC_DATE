from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
FG_CHOICES = (1, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
PANEL_WIDTH_RANGE = (5, 9)
PANEL_HEIGHT_RANGE = (7, 16)
MAX_DIFF_CELLS = 6


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _panel_layout(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    bg = _dominant_color(grid)

    full_cols_by_color: dict[int, list[int]] = {}
    for c in range(w):
        col = grid[:, c]
        color = int(col[0])
        if color != bg and np.all(col == color):
            full_cols_by_color.setdefault(color, []).append(c)

    candidates: list[tuple[int, int, int, int]] = []
    for color, cols in full_cols_by_color.items():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1 = int(cols[i])
                c2 = int(cols[j])
                left_w = c1
                mid_w = c2 - c1 - 1
                right_w = w - c2 - 1
                if left_w > 0 and left_w == mid_w == right_w:
                    candidates.append((c1, c2, left_w, color))

    assert len(candidates) == 1, "expected exactly one pair of equal-width separator columns"
    sep1, sep2, panel_w, sep_color = candidates[0]
    assert h > 0 and panel_w > 0
    return sep1, sep2, panel_w, sep_color


def _single_foreground_color(panel: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(panel) if int(color) != bg]
    assert len(colors) == 1, "each filled panel must use exactly one foreground color"
    return colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the left and middle panels pixelwise, then paint directional differences into the right panel.

    Input structure:
    - The background is black.
    - Two full vertical grey separator columns split the grid into three equal-width panels.
    - The left and middle panels each contain one same-colored foreground pattern on the background.
    - The right panel is initially empty background.

    Rule:
    - Keep the separators and the two source panels in place.
    - Compare the left and middle panels at matching relative coordinates.
    - In the right panel, paint red where the left panel alone has a foreground pixel and paint green where the middle panel alone has a foreground pixel.
    - Leave shared foreground pixels and shared background pixels absent from the right panel.

    Key insight:
    - The third panel is a color-coded "spot the difference" map: red means the left panel has the extra pixel, and green means the middle panel has the extra pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    sep1, sep2, panel_w, _sep_color = _panel_layout(g)

    left = g[:, :sep1]
    middle = g[:, sep1 + 1 : sep2]
    right = g[:, sep2 + 1 :]
    assert left.shape[1] == middle.shape[1] == right.shape[1] == panel_w
    assert np.all(right == bg), "right panel must start empty"

    left_color = _single_foreground_color(left, bg)
    middle_color = _single_foreground_color(middle, bg)
    assert left_color == middle_color, "left and middle panels must share one foreground color"

    left_mask = left != bg
    middle_mask = middle != bg

    out = g.copy()
    out[:, sep2 + 1 :] = bg
    out[:, sep2 + 1 :][left_mask & ~middle_mask] = RED
    out[:, sep2 + 1 :][middle_mask & ~left_mask] = GREEN
    return out


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            pts.append((rr, cc))
    return pts


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    if not (1 <= area <= h * w):
        return None

    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set(_neighbors(start[0], start[1], h, w))

    while int(mask.sum()) < area:
        frontier = {cell for cell in frontier if not mask[cell]}
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True
        frontier.update(_neighbors(r, c, h, w))

    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) < 2 or len(cols) < 2:
        return None
    return mask


def _adjacent_empty_cells(anchor: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = anchor.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(anchor):
        for rr, cc in _neighbors(int(r), int(c), h, w):
            if not blocked[rr, cc]:
                cells.add((rr, cc))
    return list(cells)


def _grow_attached_mask(anchor: np.ndarray, blocked: np.ndarray, area: int) -> np.ndarray | None:
    h, w = anchor.shape
    seeds = _adjacent_empty_cells(anchor, blocked)
    if not seeds:
        return None

    extra = np.zeros((h, w), dtype=bool)
    seed_r, seed_c = random.choice(seeds)
    extra[seed_r, seed_c] = True
    frontier = set(_neighbors(seed_r, seed_c, h, w))

    while int(extra.sum()) < area:
        frontier = {cell for cell in frontier if not blocked[cell] and not extra[cell]}
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        extra[r, c] = True
        frontier.update(_neighbors(r, c, h, w))

    assert not np.any(extra & blocked)
    return extra


def _transform_masks(*masks: np.ndarray) -> list[np.ndarray]:
    transformed = [mask.copy() for mask in masks]
    if random.random() < 0.5:
        transformed = [np.fliplr(mask) for mask in transformed]
    if random.random() < 0.5:
        transformed = [np.flipud(mask) for mask in transformed]
    return transformed


def _embed(mask: np.ndarray, panel_h: int, panel_w: int) -> np.ndarray:
    h, w = mask.shape
    r0 = random.randint(1, panel_h - h - 1)
    c0 = random.randint(1, panel_w - w - 1)
    panel = np.zeros((panel_h, panel_w), dtype=bool)
    panel[r0 : r0 + h, c0 : c0 + w] = mask
    return panel


def _build_variant_masks(shape_h: int, shape_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    area = shape_h * shape_w
    max_common = min(area - 4, max(4, int(0.45 * area)))
    if max_common < 4:
        return None

    common_area = random.randint(4, max_common)
    common = _random_connected_mask(shape_h, shape_w, common_area)
    if common is None:
        return None

    free_cells = area - common_area
    if free_cells < 2:
        return None
    total_diff = random.randint(2, min(MAX_DIFF_CELLS, free_cells))
    left_only_area = random.randint(1, total_diff - 1)
    middle_only_area = total_diff - left_only_area

    left_only = _grow_attached_mask(common, common.copy(), left_only_area)
    if left_only is None:
        return None

    blocked = common | left_only
    middle_only = _grow_attached_mask(common, blocked, middle_only_area)
    if middle_only is None:
        return None

    common, left_only, middle_only = _transform_masks(common, left_only, middle_only)
    left_mask = common | left_only
    middle_mask = common | middle_only
    assert np.any(left_mask & ~middle_mask)
    assert np.any(middle_mask & ~left_mask)
    return left_mask, middle_mask


def _expected_output(inp: np.ndarray, panel_w: int) -> np.ndarray:
    out = inp.copy()
    sep1 = panel_w
    sep2 = 2 * panel_w + 1
    left = inp[:, :sep1] != BLACK
    middle = inp[:, sep1 + 1 : sep2] != BLACK
    out[:, sep2 + 1 :] = BLACK
    out[:, sep2 + 1 :][left & ~middle] = RED
    out[:, sep2 + 1 :][middle & ~left] = GREEN
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    grid_w = 3 * panel_w + 2
    if max(panel_h, grid_w) > 30:
        return None

    shape_h = random.randint(3, panel_h - 2)
    shape_w = random.randint(3, panel_w - 2)
    masks = _build_variant_masks(shape_h, shape_w)
    if masks is None:
        return None
    left_local, middle_local = masks

    left_panel = _embed(left_local, panel_h, panel_w)
    middle_panel = _embed(middle_local, panel_h, panel_w)
    right_panel = np.zeros((panel_h, panel_w), dtype=int)

    fg = random.choice(FG_CHOICES)
    left_grid = np.where(left_panel, fg, BLACK)
    middle_grid = np.where(middle_panel, fg, BLACK)

    sep = np.full((panel_h, 1), GREY, dtype=int)
    inp = np.hstack([left_grid, sep, middle_grid, sep.copy(), right_panel])
    expected = _expected_output(inp, panel_w)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

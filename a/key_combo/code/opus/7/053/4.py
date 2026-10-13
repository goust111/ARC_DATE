from __future__ import annotations

import random

import numpy as np


BG = 0
LEAF = 1
DIVIDER = 2
RECT = 3
ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_divider_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    candidates = [
        r
        for r in range(height)
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != BG
    ]
    valid = [r for r in candidates if r > 0 and height - r - 1 == r and width >= 3]
    assert len(valid) == 1, "expected exactly one equal-panel divider row"
    return int(valid[0])


def _rectangle_bbox(panel: np.ndarray) -> tuple[int, int, int, int]:
    mask = (panel == RECT) | (panel == ORANGE)
    box = _bbox(mask)
    assert box is not None, "expected one green/orange rectangle"
    r0, c0, r1, c1 = box
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "rectangle region must be solid"
    return r0, c0, r1, c1


def _leaf_axes(panel: np.ndarray, box: tuple[int, int, int, int]) -> tuple[set[int], set[int], list[tuple[int, int]]]:
    height, width = map(int, panel.shape)
    r0, c0, r1, c1 = box
    rows: set[int] = set()
    cols: set[int] = set()
    leaves = [tuple(map(int, rc)) for rc in np.argwhere(panel == LEAF)]
    assert leaves, "expected at least one leaf marker"

    for r, c in leaves:
        green_neighbors = [
            (nr, nc)
            for dr, dc in DIRS4
            if 0 <= (nr := r + dr) < height
            and 0 <= (nc := c + dc) < width
            and int(panel[nr, nc]) == RECT
        ]
        assert len(green_neighbors) == 1, "leaf must have exactly one green neighbor"
        nr, nc = green_neighbors[0]
        if nr == r:
            assert c in (c0 - 1, c1 + 1), "horizontal leaf must sit beside the rectangle"
            rows.add(r)
        else:
            assert r in (r0 - 1, r1 + 1), "vertical leaf must sit above or below the rectangle"
            cols.add(c)

    return rows, cols, leaves


def _transform_panel(panel: np.ndarray) -> np.ndarray:
    out = np.array(panel, dtype=int, copy=True)
    r0, c0, r1, c1 = _rectangle_bbox(out)
    marked_rows, marked_cols, leaves = _leaf_axes(out, (r0, c0, r1, c1))

    for r in sorted(marked_rows):
        out[r, c0 : c1 + 1] = out[r, c0 : c1 + 1][::-1]

    for c in sorted(marked_cols):
        out[r0 : r1 + 1, c] = out[r0 : r1 + 1, c][::-1]

    for r, c in leaves:
        out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the marked rows and columns inside each panel, then erase the divider and overlay the two panels.

    Input structure:
    - A black background contains two equal-height panels separated by one full-width red row.
    - Each panel has exactly one solid green rectangle, with some orange pixels inside it and 1-3 blue leaf cells protruding from non-corner edge positions.
    - A leaf attached from the left or right has exactly one green neighbor horizontally; a leaf attached from the top or bottom has exactly one green neighbor vertically.

    Rule:
    - Process the top and bottom panels independently.
    - In one panel, every left/right leaf marks its row; reverse that row segment inside the green rectangle.
    - Every top/bottom leaf marks its column; reverse that column segment inside the rectangle after all marked rows have already been reversed.
    - Erase the blue leaves, delete the red divider row, and overlay the two transformed panels by local coordinates while preserving non-black pixels.

    Key insight:
    - The blue leaves are directional selectors: their attachment direction tells whether to mirror a row or a column, and those local mirrors must be applied before the two panels are merged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_row(g)
    top = _transform_panel(g[:divider, :])
    bottom = _transform_panel(g[divider + 1 :, :])
    assert top.shape == bottom.shape, "panels must have equal size"

    overlap = (top != BG) & (bottom != BG)
    assert not np.any(overlap), "panels must overlay without foreground collisions"

    out = top.copy()
    bottom_mask = bottom != BG
    out[bottom_mask] = bottom[bottom_mask]
    return out


# --------------- generator ---------------

PANEL_H_RANGE = (7, 10)
PANEL_W_RANGE = (13, 19)
RECT_H_RANGE = (4, 6)
RECT_W_RANGE = (5, 8)
MAX_GEN_ATTEMPTS = 400


def _gen_choice_weighted(values: list[int], weights: list[int]) -> int:
    return int(random.choices(values, weights=weights, k=1)[0])


def _gen_nonoverlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _gen_two_boxes(panel_h: int, panel_w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(200):
        h1 = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], panel_h - 2))
        w1 = random.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], panel_w - 2))
        h2 = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], panel_h - 2))
        w2 = random.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], panel_w - 2))

        r01 = random.randint(1, panel_h - h1 - 1)
        c01 = random.randint(1, panel_w - w1 - 1)
        r02 = random.randint(1, panel_h - h2 - 1)
        c02 = random.randint(1, panel_w - w2 - 1)

        box1 = (r01, c01, r01 + h1 - 1, c01 + w1 - 1)
        box2 = (r02, c02, r02 + h2 - 1, c02 + w2 - 1)
        total_area = h1 * w1 + h2 * w2
        if total_area >= (panel_h * panel_w) // 2:
            continue
        if _gen_nonoverlap(box1, box2):
            return box1, box2
    return None


def _gen_marker_split(max_rows: int, max_cols: int) -> tuple[int, int] | None:
    max_total = min(3, max_rows + max_cols)
    if max_total < 1:
        return None

    total = _gen_choice_weighted(list(range(1, max_total + 1)), [2, 4, 3][:max_total])
    options = [
        (rows, total - rows)
        for rows in range(total + 1)
        if rows <= max_rows and total - rows <= max_cols
    ]
    if not options:
        return None

    mixed = [opt for opt in options if opt[0] > 0 and opt[1] > 0]
    if mixed and random.random() < 0.7:
        return random.choice(mixed)
    return random.choice(options)


def _gen_panel(panel_h: int, panel_w: int, box: tuple[int, int, int, int]) -> tuple[np.ndarray, set[int], set[int]] | None:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    panel[r0 : r1 + 1, c0 : c1 + 1] = RECT

    row_candidates = list(range(r0 + 1, r1))
    col_candidates = list(range(c0 + 1, c1))
    split = _gen_marker_split(len(row_candidates), len(col_candidates))
    if split is None:
        return None
    row_count, col_count = split
    row_axes = set(random.sample(row_candidates, row_count))
    col_axes = set(random.sample(col_candidates, col_count))

    reserved_green: set[tuple[int, int]] = set()
    row_sides: dict[int, str] = {}
    col_sides: dict[int, str] = {}
    for r in row_axes:
        side = random.choice(["left", "right"])
        row_sides[r] = side
        reserved_green.add((r, c0 if side == "left" else c1))
    for c in col_axes:
        side = random.choice(["top", "bottom"])
        col_sides[c] = side
        reserved_green.add((r0 if side == "top" else r1, c))

    available = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in reserved_green
    ]
    area = h * w
    max_orange = min(len(available), area - len(reserved_green) - 2)
    if max_orange < 3:
        return None
    orange_count = random.randint(3, max(3, min(max_orange, (2 * area) // 3)))
    orange_cells = set(random.sample(available, orange_count))
    for r, c in orange_cells:
        panel[r, c] = ORANGE

    for r, side in row_sides.items():
        panel[r, c0 - 1 if side == "left" else c1 + 1] = LEAF
    for c, side in col_sides.items():
        panel[r0 - 1 if side == "top" else r1 + 1, c] = LEAF

    try:
        transformed = _transform_panel(panel)
    except AssertionError:
        return None

    leafless = panel.copy()
    leafless[leafless == LEAF] = BG
    if np.array_equal(transformed, leafless):
        return None
    if int(np.count_nonzero(panel == ORANGE)) < 3 or int(np.count_nonzero(panel == RECT)) < 3:
        return None

    return panel, row_axes, col_axes


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    boxes = _gen_two_boxes(panel_h, panel_w)
    if boxes is None:
        return None

    top_info = _gen_panel(panel_h, panel_w, boxes[0])
    bottom_info = _gen_panel(panel_h, panel_w, boxes[1])
    if top_info is None or bottom_info is None:
        return None

    top_panel, top_rows, top_cols = top_info
    bottom_panel, bottom_rows, bottom_cols = bottom_info
    if not (top_rows | bottom_rows):
        return None
    if not (top_cols | bottom_cols):
        return None

    grid = np.full((2 * panel_h + 1, panel_w), BG, dtype=int)
    grid[:panel_h, :] = top_panel
    grid[panel_h, :] = DIVIDER
    grid[panel_h + 1 :, :] = bottom_panel

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (panel_h, panel_w):
        return None
    if np.array_equal(out, top_panel) or np.array_equal(out, bottom_panel):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")

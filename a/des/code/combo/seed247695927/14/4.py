import random

import numpy as np


RED = 2
GREY = 5
TEAL = 8
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 4

    bg = _get_bg(g)
    red_pos = np.argwhere(g == RED)
    teal_pos = np.argwhere(g == TEAL)
    assert len(red_pos) == 1, "expected exactly one red marker"
    assert len(teal_pos) == 1, "expected exactly one teal marker"
    red_r, red_c = map(int, red_pos[0])
    teal_r, teal_c = map(int, teal_pos[0])
    assert red_r == teal_r, "markers must share a row"
    assert red_c != teal_c, "markers must be horizontally separated"

    rect_mask = (g != bg) & (g != RED) & (g != TEAL)
    rect_bbox = _bbox(rect_mask)
    assert rect_bbox is not None, "expected one solid rectangle"
    r0, c0, r1, c1 = rect_bbox

    rect_colors = np.unique(g[rect_mask])
    assert len(rect_colors) == 1, "rectangle must be monochrome"
    assert int(rect_colors[0]) == GREY, "rectangle must be grey"
    assert np.all(rect_mask[r0 : r1 + 1, c0 : c1 + 1]), "rectangle must fill its bounding box"
    assert int(rect_mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "rectangle must be solid"

    assert red_r == r0 - 1, "markers must sit directly above the rectangle"
    assert c0 <= red_c <= c1 and c0 <= teal_c <= c1, "markers must be over the rectangle columns"

    side = 1 if teal_c > red_c else -1
    exit_r = r1
    exit_c = c1 + 1 if side > 0 else c0 - 1
    assert 0 <= exit_r < h and 0 <= exit_c < w, "exit cell must stay inside the grid"
    assert g[exit_r, exit_c] == bg, "exit cell must be background"

    out = g.copy()
    out[red_r, red_c] = bg
    out[teal_r, teal_c] = RED
    out[exit_r, exit_c] = TEAL
    return out


def _pick_grid_shape() -> tuple[int, int]:
    return random.randint(7, 18), random.randint(7, 18)


def _pick_rect_dims(h: int, w: int) -> tuple[int, int] | None:
    choices = []
    for rect_h in range(2, h - 2):
        for rect_w in range(3, w - 2):
            if rect_h * rect_w < (h * w) // 2:
                choices.append((rect_h, rect_w))
    return random.choice(choices) if choices else None


def _pick_colors() -> int:
    candidates = [c for c in range(10) if c not in {RED, GREY, TEAL}]
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    h, w = _pick_grid_shape()
    dims = _pick_rect_dims(h, w)
    if dims is None:
        return None
    rect_h, rect_w = dims
    bg = _pick_colors()
    side = random.choice([-1, 1])

    r0_min = 1
    r0_max = h - rect_h
    if r0_min > r0_max:
        return None

    if side > 0:
        c0_min = 0
        c0_max = w - rect_w - 2
    else:
        c0_min = 1
        c0_max = w - rect_w - 1
    if c0_min > c0_max:
        return None

    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    marker_r = r0 - 1
    exit_c = c1 + 1 if side > 0 else c0 - 1

    if rect_w < 2:
        return None
    sep = random.randint(1, rect_w - 1)
    left_col = random.randint(c0, c1 - sep)
    right_col = left_col + sep
    red_c, teal_c = (left_col, right_col) if side > 0 else (right_col, left_col)

    grid = np.full((h, w), bg, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[marker_r, red_c] = RED
    grid[marker_r, teal_c] = TEAL

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = grid.copy()
    expected[marker_r, red_c] = bg
    expected[marker_r, teal_c] = RED
    expected[r1, exit_c] = TEAL
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

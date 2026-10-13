from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

PANEL_SIZE = 9
GUIDE_INDEX = 4
SEPARATOR_COL = 9
GRID_SHAPE = (9, 19)
GENERATION_ATTEMPTS = 500

COMBO_CHOICES = [("h", "h"), ("v", "v"), ("h", "v"), ("v", "h")]
COMBO_WEIGHTS = [2, 2, 3, 3]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _panel_axis(panel: np.ndarray) -> str:
    has_row_axis = bool(np.all(panel[GUIDE_INDEX, :] == GREY))
    has_col_axis = bool(np.all(panel[:, GUIDE_INDEX] == GREY))
    assert has_row_axis ^ has_col_axis, "expected exactly one grey guide line per panel"
    return "h" if has_row_axis else "v"


def _complete_panel(panel: np.ndarray) -> np.ndarray:
    panel = np.asarray(panel, dtype=int)
    assert panel.shape == (PANEL_SIZE, PANEL_SIZE)
    assert set(map(int, np.unique(panel))) <= {BLACK, BLUE, GREY}

    axis = _panel_axis(panel)
    out = np.full((PANEL_SIZE, PANEL_SIZE), BLACK, dtype=int)

    if axis == "h":
        top = panel[:GUIDE_INDEX, :] == BLUE
        bottom = panel[GUIDE_INDEX + 1 :, :] == BLUE
        top_count = int(top.sum())
        bottom_count = int(bottom.sum())
        assert (top_count > 0) ^ (bottom_count > 0), "horizontal panel must occupy exactly one half"
        if top_count > 0:
            out[:GUIDE_INDEX, :] = top.astype(int)
            out[GUIDE_INDEX + 1 :, :] = np.flipud(out[:GUIDE_INDEX, :])
        else:
            out[GUIDE_INDEX + 1 :, :] = bottom.astype(int)
            out[:GUIDE_INDEX, :] = np.flipud(out[GUIDE_INDEX + 1 :, :])
        return out

    left = panel[:, :GUIDE_INDEX] == BLUE
    right = panel[:, GUIDE_INDEX + 1 :] == BLUE
    left_count = int(left.sum())
    right_count = int(right.sum())
    assert (left_count > 0) ^ (right_count > 0), "vertical panel must occupy exactly one half"
    if left_count > 0:
        out[:, :GUIDE_INDEX] = left.astype(int)
        out[:, GUIDE_INDEX + 1 :] = np.fliplr(out[:, :GUIDE_INDEX])
    else:
        out[:, GUIDE_INDEX + 1 :] = right.astype(int)
        out[:, :GUIDE_INDEX] = np.fliplr(out[:, GUIDE_INDEX + 1 :])
    return out


def _overlay(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full(left.shape, BLACK, dtype=int)
    left_fg = left == BLUE
    right_fg = right == BLUE
    out[left_fg ^ right_fg] = BLUE
    out[left_fg & right_fg] = YELLOW
    return out


def _edge_reachable_black(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if not (0 <= r < h and 0 <= c < w):
            continue
        if seen[r, c] or grid[r, c] != BLACK:
            continue
        seen[r, c] = True
        stack.extend(_neighbors4(r, c))
    return seen


def _fill_enclosed_black(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    reachable = _edge_reachable_black(out)
    out[(out == BLACK) & (~reachable)] = RED
    return out


def _count_enclosed_black(grid: np.ndarray) -> int:
    grid = np.asarray(grid, dtype=int)
    reachable = _edge_reachable_black(grid)
    return int(np.count_nonzero((grid == BLACK) & (~reachable)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete the two guide-line panels, overlay them with blue/yellow agreement colors, then fill enclosed black pockets red.

    Input structure:
    - The input is a 9x19 grid split into two 9x9 panels by a full grey separator column.
    - Inside each panel, the center row or center column is a full grey guide line.
    - Exactly one side of each guide line contains one 4-connected blue pattern; the opposite side is black.

    Rule:
    - For each panel, use the guide-line orientation to decide the mirror axis: a grey center row means mirror vertically, and a grey center column means mirror horizontally.
    - Copy the occupied half across the guide line, then clear the guide line itself to black.
    - Overlay the two completed panels cell by cell: black+black stays black, blue+black becomes blue, blue+blue becomes yellow.
    - In the overlay, recolor every black 4-connected region that cannot reach the border through black cells to red.

    Key insight:
    - The grey line does not survive; it only encodes how each half-pattern must be reflected, and the red regions appear only after comparing the two mirrored panels together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
    assert bool(np.all(g[:, SEPARATOR_COL] == GREY)), "expected a full grey separator column"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, GREY}

    left_panel = g[:, :PANEL_SIZE]
    right_panel = g[:, SEPARATOR_COL + 1 :]
    assert left_panel.shape == (PANEL_SIZE, PANEL_SIZE)
    assert right_panel.shape == (PANEL_SIZE, PANEL_SIZE)

    left_completed = _complete_panel(left_panel)
    right_completed = _complete_panel(right_panel)
    overlaid = _overlay(left_completed, right_completed)
    return _fill_enclosed_black(overlaid)


def _horizontal_open_down_shape() -> np.ndarray | None:
    patch = np.full((4, 9), BLACK, dtype=int)
    bar_r = random.choice([0, 1])
    left = random.randint(1, 2)
    right = random.randint(6, 7)

    patch[bar_r, left : right + 1] = BLUE
    for rr in range(bar_r, 4):
        patch[rr, left] = BLUE
        patch[rr, right] = BLUE

    if random.random() < 0.6:
        mid = random.randint(left + 2, right - 2)
        for rr in range(bar_r + 1, 4):
            patch[rr, mid] = BLUE

    while int(np.count_nonzero(patch == BLUE)) > 12:
        removable = [(bar_r, cc) for cc in range(left + 1, right) if patch[bar_r, cc] == BLUE]
        if not removable:
            break
        rr, cc = random.choice(removable)
        patch[rr, cc] = BLACK

    area = int(np.count_nonzero(patch == BLUE))
    return patch if 6 <= area <= 12 else None


def _horizontal_open_up_shape() -> np.ndarray | None:
    patch = np.full((4, 9), BLACK, dtype=int)
    bar_r = random.choice([2, 3])
    left = random.randint(1, 2)
    right = random.randint(6, 7)
    top_end = max(0, bar_r - 2)

    patch[bar_r, left : right + 1] = BLUE
    for rr in range(top_end, bar_r + 1):
        patch[rr, left] = BLUE
        patch[rr, right] = BLUE

    if random.random() < 0.5:
        mid = random.randint(left + 2, right - 2)
        for rr in range(top_end, bar_r):
            patch[rr, mid] = BLUE

    while int(np.count_nonzero(patch == BLUE)) > 12:
        removable = [(bar_r, cc) for cc in range(left + 1, right) if patch[bar_r, cc] == BLUE]
        if not removable:
            break
        rr, cc = random.choice(removable)
        patch[rr, cc] = BLACK

    area = int(np.count_nonzero(patch == BLUE))
    return patch if 6 <= area <= 12 else None


def _vertical_open_right_shape() -> np.ndarray | None:
    patch = _horizontal_open_down_shape()
    return None if patch is None else patch.T.copy()


def _vertical_open_left_shape() -> np.ndarray | None:
    patch = _horizontal_open_up_shape()
    return None if patch is None else patch.T.copy()


def _make_panel(axis: str, source_side: str, family: str) -> np.ndarray | None:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BLACK, dtype=int)

    if axis == "h":
        panel[GUIDE_INDEX, :] = GREY
        if family == "down":
            patch = _horizontal_open_down_shape()
        elif family == "up":
            patch = _horizontal_open_up_shape()
        else:
            raise ValueError(f"unknown horizontal family: {family}")
        if patch is None:
            return None
        if source_side == "top":
            panel[:GUIDE_INDEX, :] = patch
        elif source_side == "bottom":
            panel[GUIDE_INDEX + 1 :, :] = np.flipud(patch)
        else:
            raise ValueError(f"unknown horizontal side: {source_side}")
        return panel

    panel[:, GUIDE_INDEX] = GREY
    if family == "right":
        patch = _vertical_open_right_shape()
    elif family == "left":
        patch = _vertical_open_left_shape()
    else:
        raise ValueError(f"unknown vertical family: {family}")
    if patch is None:
        return None
    if source_side == "left":
        panel[:, :GUIDE_INDEX] = patch
    elif source_side == "right":
        panel[:, GUIDE_INDEX + 1 :] = np.fliplr(patch)
    else:
        raise ValueError(f"unknown vertical side: {source_side}")
    return panel


def _compose_grid(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BLACK, dtype=int)
    grid[:, SEPARATOR_COL] = GREY
    grid[:, :PANEL_SIZE] = left_panel
    grid[:, SEPARATOR_COL + 1 :] = right_panel
    return grid


def _sample_panels() -> tuple[np.ndarray, np.ndarray] | None:
    combo = random.choices(COMBO_CHOICES, weights=COMBO_WEIGHTS, k=1)[0]

    if combo == ("h", "h"):
        left_family, right_family = random.sample(["down", "up"], 2)
        left_panel = _make_panel("h", random.choice(["top", "bottom"]), left_family)
        right_panel = _make_panel("h", random.choice(["top", "bottom"]), right_family)
    elif combo == ("v", "v"):
        left_family, right_family = random.sample(["right", "left"], 2)
        left_panel = _make_panel("v", random.choice(["left", "right"]), left_family)
        right_panel = _make_panel("v", random.choice(["left", "right"]), right_family)
    elif combo == ("h", "v"):
        left_panel = _make_panel("h", random.choice(["top", "bottom"]), random.choice(["down", "up"]))
        right_panel = _make_panel("v", random.choice(["left", "right"]), random.choice(["right", "left"]))
    else:
        left_panel = _make_panel("v", random.choice(["left", "right"]), random.choice(["right", "left"]))
        right_panel = _make_panel("h", random.choice(["top", "bottom"]), random.choice(["down", "up"]))

    if left_panel is None or right_panel is None:
        return None
    return left_panel, right_panel


def _is_good_instance(grid: np.ndarray) -> bool:
    left_completed = _complete_panel(grid[:, :PANEL_SIZE])
    right_completed = _complete_panel(grid[:, SEPARATOR_COL + 1 :])

    if _count_enclosed_black(left_completed) != 0:
        return False
    if _count_enclosed_black(right_completed) != 0:
        return False

    overlay = _overlay(left_completed, right_completed)
    if int(np.count_nonzero(overlay == YELLOW)) == 0:
        return False
    if int(np.count_nonzero(overlay == BLUE)) == 0:
        return False

    red_count = _count_enclosed_black(overlay)
    if red_count == 0:
        return False
    if int(np.count_nonzero(overlay != BLACK)) > 48:
        return False

    expected = _fill_enclosed_black(overlay)
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        panels = _sample_panels()
        if panels is None:
            continue
        grid = _compose_grid(*panels)
        if _is_good_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid overlay-and-hole ARC input")

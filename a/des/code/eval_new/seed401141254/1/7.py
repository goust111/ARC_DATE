from __future__ import annotations

import random

import numpy as np


ARC_MAX = 30
GREY = 5
GEN_RETRIES = 800
PLACE_RETRIES = 300
PALETTE = tuple(range(10))
RECT_COUNT_RANGE = (3, 5)
HEIGHT_RANGE = (12, 20)
LEFT_WIDTH_RANGE = (12, 20)
FRONT_SIZE_RANGE = (4, 8)
CORE_SIZE_RANGE = (2, 6)
SIDE_DIRECTIONS = ("left", "right", "top", "bottom")


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _fill_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _parse_layout(grid: np.ndarray) -> tuple[np.ndarray, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    sep_col = None
    sep_color = None
    for c in range(w - 1):
        col = g[:, c]
        color = int(col[0])
        if color == bg or not np.all(col == color):
            continue
        right = g[:, c + 1 :]
        right_colors = {int(x) for x in np.unique(right)} - {bg, color}
        if right_colors:
            sep_col = c
            sep_color = color

    assert sep_col is not None and sep_color is not None, "missing separator column"
    left = g[:, :sep_col]

    legend_pixels: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(sep_col + 1, w):
            color = int(g[r, c])
            if color not in (bg, sep_color):
                legend_pixels.append((r, color))

    assert legend_pixels, "missing legend colors"
    order = [color for r, color in sorted(legend_pixels)]
    assert len(order) == len(set(order)), "legend colors must be unique"
    return left, bg, order


def _infer_bbox_from_masks(
    visible: np.ndarray, blockers: np.ndarray
) -> tuple[int, int, int, int]:
    rows, cols = np.where(visible)
    assert rows.size > 0, "color must remain visible"
    h, w = visible.shape
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())

    changed = True
    while changed:
        changed = False
        while c0 > 0 and bool(np.all(blockers[r0 : r1 + 1, c0 - 1])):
            c0 -= 1
            changed = True
        while c1 + 1 < w and bool(np.all(blockers[r0 : r1 + 1, c1 + 1])):
            c1 += 1
            changed = True
        while r0 > 0 and bool(np.all(blockers[r0 - 1, c0 : c1 + 1])):
            r0 -= 1
            changed = True
        while r1 + 1 < h and bool(np.all(blockers[r1 + 1, c0 : c1 + 1])):
            r1 += 1
            changed = True

    allowed = visible | blockers
    assert bool(np.all(allowed[r0 : r1 + 1, c0 : c1 + 1])), "invalid rectangle inference"
    return r0, r1, c0, c1


def _infer_rectangles(left: np.ndarray, order: list[int]) -> dict[int, tuple[int, int, int, int]]:
    h, w = left.shape
    blockers = np.zeros((h, w), dtype=bool)
    inferred: dict[int, tuple[int, int, int, int]] = {}

    for color in reversed(order):
        visible = left == color
        assert bool(np.any(visible)), "each legend color must appear in the left canvas"
        rect = _infer_bbox_from_masks(visible, blockers)
        inferred[color] = rect
        _fill_rect(blockers, rect, True)  # type: ignore[arg-type]

    return inferred


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct partially hidden solid rectangles from a side legend and redraw the completed stack.

    Input structure:
    - The background is the most common color.
    - The left side contains 3 to 5 solid axis-aligned rectangles in distinct colors.
      They partially overlap, so only the topmost color is visible at each covered cell.
    - A full-height grey separator column stands near the right edge.
    - To the right of the separator is a vertical color legend with one pixel per layer,
      spaced by background rows, ordered from backmost at the top to frontmost at the bottom.

    Rule:
    - Remove the separator and the right-side legend, keeping only the left canvas.
    - Use the legend order to process rectangles from front to back: a lower rectangle may
      extend through cells already occupied by known upper rectangles, but it cannot extend
      into exposed background.
    - For each color, infer the unique solid bounding rectangle obtained by expanding its
      visible part only through fully occluded rows or columns.
    - Redraw the completed rectangles on a blank copy of the left canvas from back to front.

    Key insight:
    - The visible pixels of a hidden layer are only a partial witness; once the front layers
      are known from the legend, every extra row or column of the lower rectangle must lie
      entirely inside those upper-layer occlusion masks.
    """
    left, bg, order = _parse_layout(np.asarray(grid, dtype=int))
    rects = _infer_rectangles(left, order)

    out = np.full(left.shape, bg, dtype=int)
    for color in order:
        _fill_rect(out, rects[color], color)
    return out


def _sample_front_rect(h: int, w: int) -> tuple[int, int, int, int] | None:
    max_h = min(FRONT_SIZE_RANGE[1], h - 4)
    max_w = min(FRONT_SIZE_RANGE[1], w - 4)
    if max_h < FRONT_SIZE_RANGE[0] or max_w < FRONT_SIZE_RANGE[0]:
        return None

    rect_h = random.randint(FRONT_SIZE_RANGE[0], max_h)
    rect_w = random.randint(FRONT_SIZE_RANGE[0], max_w)
    r0 = random.randint(1, h - rect_h - 2)
    c0 = random.randint(1, w - rect_w - 2)
    return r0, r0 + rect_h - 1, c0, c0 + rect_w - 1


def _rect_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        _fill_rect(mask, rect, True)  # type: ignore[arg-type]
    return mask


def _place_lower_rect(
    h: int, w: int, above_rects: list[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    blockers = _rect_mask((h, w), above_rects)
    for _ in range(PLACE_RETRIES):
        br0, br1, bc0, bc1 = random.choice(above_rects)
        side = random.choice(SIDE_DIRECTIONS)

        if side == "right":
            max_core_h = min(CORE_SIZE_RANGE[1], br1 - br0 + 1)
            max_core_w = min(CORE_SIZE_RANGE[1], bc0)
            if max_core_h < CORE_SIZE_RANGE[0] or max_core_w < CORE_SIZE_RANGE[0]:
                continue
            core_h = random.randint(CORE_SIZE_RANGE[0], max_core_h)
            core_w = random.randint(CORE_SIZE_RANGE[0], max_core_w)
            r0 = random.randint(br0, br1 - core_h + 1)
            r1 = r0 + core_h - 1
            c1 = bc0 - 1
            c0 = c1 - core_w + 1
            full = (r0, r1, c0, bc1)
            core = (r0, r1, c0, c1)
        elif side == "left":
            max_core_h = min(CORE_SIZE_RANGE[1], br1 - br0 + 1)
            max_core_w = min(CORE_SIZE_RANGE[1], w - bc1 - 1)
            if max_core_h < CORE_SIZE_RANGE[0] or max_core_w < CORE_SIZE_RANGE[0]:
                continue
            core_h = random.randint(CORE_SIZE_RANGE[0], max_core_h)
            core_w = random.randint(CORE_SIZE_RANGE[0], max_core_w)
            r0 = random.randint(br0, br1 - core_h + 1)
            r1 = r0 + core_h - 1
            c0 = bc1 + 1
            c1 = c0 + core_w - 1
            full = (r0, r1, bc0, c1)
            core = (r0, r1, c0, c1)
        elif side == "bottom":
            max_core_h = min(CORE_SIZE_RANGE[1], br0)
            max_core_w = min(CORE_SIZE_RANGE[1], br1 - br0 + 1 + bc1 - bc0 + 1)
            max_core_w = min(max_core_w, bc1 - bc0 + 1)
            if max_core_h < CORE_SIZE_RANGE[0] or max_core_w < CORE_SIZE_RANGE[0]:
                continue
            core_h = random.randint(CORE_SIZE_RANGE[0], max_core_h)
            core_w = random.randint(CORE_SIZE_RANGE[0], max_core_w)
            r1 = br0 - 1
            r0 = r1 - core_h + 1
            c0 = random.randint(bc0, bc1 - core_w + 1)
            c1 = c0 + core_w - 1
            full = (r0, br1, c0, c1)
            core = (r0, r1, c0, c1)
        else:
            max_core_h = min(CORE_SIZE_RANGE[1], h - br1 - 1)
            max_core_w = min(CORE_SIZE_RANGE[1], bc1 - bc0 + 1)
            if max_core_h < CORE_SIZE_RANGE[0] or max_core_w < CORE_SIZE_RANGE[0]:
                continue
            core_h = random.randint(CORE_SIZE_RANGE[0], max_core_h)
            core_w = random.randint(CORE_SIZE_RANGE[0], max_core_w)
            r0 = br1 + 1
            r1 = r0 + core_h - 1
            c0 = random.randint(bc0, bc1 - core_w + 1)
            c1 = c0 + core_w - 1
            full = (br0, r1, c0, c1)
            core = (r0, r1, c0, c1)

        cr0, cr1, cc0, cc1 = core
        if cr0 < 0 or cc0 < 0 or cr1 >= h or cc1 >= w:
            continue
        if bool(np.any(blockers[cr0 : cr1 + 1, cc0 : cc1 + 1])):
            continue

        visible = np.zeros((h, w), dtype=bool)
        _fill_rect(visible, core, True)  # type: ignore[arg-type]
        try:
            inferred = _infer_bbox_from_masks(visible, blockers)
        except AssertionError:
            continue
        if inferred == full:
            return full
    return None


def _render_left(
    h: int,
    w: int,
    bg: int,
    order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    left = np.full((h, w), bg, dtype=int)
    for color in order:
        _fill_rect(left, rects[color], color)
    return left


def _add_legend(left: np.ndarray, bg: int, order: list[int]) -> np.ndarray:
    h, w = left.shape
    right_w = random.randint(3, 5)
    sep_col = w
    legend_col = sep_col + random.randint(1, right_w - 1)

    grid = np.full((h, w + right_w), bg, dtype=int)
    grid[:, :w] = left
    grid[:, sep_col] = GREY

    span = 2 * len(order) - 1
    start = random.randint(0, h - span)
    for i, color in enumerate(order):
        grid[start + 2 * i, legend_col] = color
    return grid


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    by_color = {int(v): int(c) for v, c in zip(values, counts)}
    return by_color.get(bg, 0) > max(c for color, c in by_color.items() if color != bg)


def _valid_separator_uniqueness(grid: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    full_cols = []
    for c in range(w):
        col = grid[:, c]
        color = int(col[0])
        if color != bg and np.all(col == color):
            full_cols.append(c)
    return len(full_cols) == 1


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    min_h = max(HEIGHT_RANGE[0], 2 * n_rects - 1)
    h = random.randint(min_h, HEIGHT_RANGE[1])
    w = random.randint(*LEFT_WIDTH_RANGE)

    bg_choices = [c for c in PALETTE if c != GREY]
    bg = random.choice(bg_choices)
    color_choices = [c for c in PALETTE if c not in (bg, GREY)]
    colors = random.sample(color_choices, n_rects)
    order = random.sample(colors, n_rects)

    top_color = order[-1]
    top_rect = _sample_front_rect(h, w)
    if top_rect is None:
        return None

    rects: dict[int, tuple[int, int, int, int]] = {top_color: top_rect}
    above_rects = [top_rect]

    hidden_count = 0
    for color in reversed(order[:-1]):
        rect = _place_lower_rect(h, w, above_rects)
        if rect is None:
            return None
        rects[color] = rect
        above_rects.append(rect)
        hidden_count += 1

    if hidden_count == 0:
        return None

    left = _render_left(h, w, bg, order, rects)
    grid = _add_legend(left, bg, order)

    if max(grid.shape) > ARC_MAX:
        return None
    if get_bg(grid) != bg or not _background_is_dominant(grid, bg):
        return None
    if not _valid_separator_uniqueness(grid, bg):
        return None
    if any(np.count_nonzero(left == color) == 0 for color in order):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(solved, left) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

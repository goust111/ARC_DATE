from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500
ORIENTATIONS = ("tl", "tr", "bl", "br")


Rect = tuple[int, int, int, int]  # (r0, c0, h, w)


def _bbox_for_color(grid: np.ndarray, color: int) -> Rect:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        raise AssertionError(f"missing color {color}")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)


def _rect_intersects(shape: tuple[int, int], rect: Rect) -> bool:
    r0, c0, h, w = rect
    h_grid, w_grid = map(int, shape)
    return r0 < h_grid and c0 < w_grid and r0 + h > 0 and c0 + w > 0


def _paint_rect(grid: np.ndarray, color: int, rect: Rect, *, clip: bool = False) -> None:
    r0, c0, h, w = rect
    if not clip:
        grid[r0 : r0 + h, c0 : c0 + w] = color
        return

    h_grid, w_grid = grid.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h_grid, r0 + h)
    cc1 = min(w_grid, c0 + w)
    if rr0 < rr1 and cc0 < cc1:
        grid[rr0:rr1, cc0:cc1] = color


def _recover_rectangles(grid: np.ndarray) -> dict[int, Rect]:
    colors = [int(v) for v in np.unique(grid) if int(v) != BACKGROUND]
    assert len(colors) == 3, "expected exactly three non-background colors"
    return {color: _bbox_for_color(grid, color) for color in colors}


def _colors_inside_rect(grid: np.ndarray, color: int, rect: Rect) -> set[int]:
    r0, c0, h, w = rect
    inside = {int(v) for v in grid[r0 : r0 + h, c0 : c0 + w].flat}
    inside.discard(BACKGROUND)
    inside.discard(color)
    return inside


def _occlusion_order(grid: np.ndarray, rects: dict[int, Rect]) -> list[int]:
    lower_to_upper = {color: _colors_inside_rect(grid, color, rect) for color, rect in rects.items()}
    indegree = {color: 0 for color in rects}
    for uppers in lower_to_upper.values():
        for upper in uppers:
            assert upper in indegree, "unexpected color inside rectangle bbox"
            indegree[upper] += 1

    order: list[int] = []
    ready = sorted(color for color, deg in indegree.items() if deg == 0)
    while ready:
        assert len(ready) == 1, "occlusion chain must determine a unique order"
        color = ready.pop()
        order.append(color)
        for upper in sorted(lower_to_upper[color]):
            indegree[upper] -= 1
            if indegree[upper] == 0:
                ready.append(upper)

    assert len(order) == 3, "occlusion graph must cover all three rectangles"
    return order


def _stamp_copies(
    out: np.ndarray,
    *,
    color: int,
    stamp_rect: Rect,
    stride_h: int,
    stride_w: int,
) -> None:
    r0, c0, h, w = stamp_rect
    for dr, dc in ((-stride_h, -stride_w), (stride_h, stride_w)):
        step = 1
        while True:
            rect = (r0 + step * dr, c0 + step * dc, h, w)
            if not _rect_intersects(out.shape, rect):
                break
            _paint_rect(out, color, rect, clip=True)
            step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the top rectangle along the main diagonal using the bottom rectangle's recovered size as the stride.

    Input structure:
    - A black background contains exactly three different-colored solid rectangles seen under a fixed occlusion chain.
    - The top rectangle is fully visible and overlaps the middle rectangle; the middle overlaps the bottom rectangle; the top and bottom do not touch.
    - The occluded rectangles are incomplete in the input, but each one still shows pixels on all four true sides, so its full bbox is recoverable from visible pixels.

    Rule:
    - Recover each rectangle's full bbox from the visible pixels of its color.
    - If another color appears inside a rectangle's recovered bbox, that other rectangle is above it; use these relations to order the chain from bottom to top.
    - Keep the original occluded scene, then stamp the top rectangle's solid shape at offsets (-h, -w), (-2h, -2w), ... and (h, w), (2h, 2w), ... where (h, w) is the recovered height and width of the bottom rectangle, clipping copies at the grid edge.

    Key insight:
    - The same three rectangles play different roles only through depth: the top one supplies the stamp sprite, while the bottom one supplies the step size, so reconstructing the occlusion order is the whole puzzle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _recover_rectangles(g)
    bottom_color, _middle_color, top_color = _occlusion_order(g, rects)
    top_rect = rects[top_color]
    bottom_rect = rects[bottom_color]

    tr0, tc0, th, tw = top_rect
    assert np.all(g[tr0 : tr0 + th, tc0 : tc0 + tw] == top_color), "top rectangle must be fully visible"

    out = g.copy()
    _stamp_copies(
        out,
        color=top_color,
        stamp_rect=top_rect,
        stride_h=bottom_rect[2],
        stride_w=bottom_rect[3],
    )
    return out


def _edge_visible(grid: np.ndarray, color: int, rect: Rect) -> bool:
    r0, c0, h, w = rect
    return (
        np.any(grid[r0, c0 : c0 + w] == color)
        and np.any(grid[r0 + h - 1, c0 : c0 + w] == color)
        and np.any(grid[r0 : r0 + h, c0] == color)
        and np.any(grid[r0 : r0 + h, c0 + w - 1] == color)
    )


def _step_count(shape: tuple[int, int], stamp_rect: Rect, stride_h: int, stride_w: int, sign: int) -> int:
    r0, c0, h, w = stamp_rect
    count = 0
    step = 1
    while True:
        rect = (r0 + sign * step * stride_h, c0 + sign * step * stride_w, h, w)
        if not _rect_intersects(shape, rect):
            return count
        count += 1
        step += 1


def _expected_output(grid: np.ndarray, top_color: int, top_rect: Rect, bottom_rect: Rect) -> np.ndarray:
    out = np.array(grid, copy=True)
    _stamp_copies(out, color=top_color, stamp_rect=top_rect, stride_h=bottom_rect[2], stride_w=bottom_rect[3])
    return out


def _build_bottom_rect(middle_rect: Rect, bottom_h: int, bottom_w: int, overlap_h: int, overlap_w: int, corner: str) -> Rect:
    mr0, mc0, mh, mw = middle_rect
    if corner == "tl":
        return mr0 - (bottom_h - overlap_h), mc0 - (bottom_w - overlap_w), bottom_h, bottom_w
    if corner == "tr":
        return mr0 - (bottom_h - overlap_h), mc0 + mw - overlap_w, bottom_h, bottom_w
    if corner == "bl":
        return mr0 + mh - overlap_h, mc0 - (bottom_w - overlap_w), bottom_h, bottom_w
    if corner == "br":
        return mr0 + mh - overlap_h, mc0 + mw - overlap_w, bottom_h, bottom_w
    raise ValueError(f"unknown corner {corner}")


def _extra_extents(bottom_h: int, bottom_w: int, overlap_h: int, overlap_w: int, corner: str) -> tuple[int, int, int, int]:
    extra_h = bottom_h - overlap_h
    extra_w = bottom_w - overlap_w
    extra_top = extra_h if corner in ("tl", "tr") else 0
    extra_bottom = extra_h if corner in ("bl", "br") else 0
    extra_left = extra_w if corner in ("tl", "bl") else 0
    extra_right = extra_w if corner in ("tr", "br") else 0
    return extra_top, extra_right, extra_bottom, extra_left


def _make_scene() -> tuple[np.ndarray, int, Rect, Rect] | None:
    top_h = random.randint(2, 4)
    top_w = random.randint(2, 4)
    bottom_h = random.randint(3, 6)
    bottom_w = random.randint(3, 6)
    if (bottom_h, bottom_w) == (top_h, top_w):
        return None

    overlap_h = random.randint(1, bottom_h - 1)
    overlap_w = random.randint(1, bottom_w - 1)
    corner = random.choice(ORIENTATIONS)

    top_min = overlap_h + 1 if corner in ("tl", "tr") else 1
    bottom_min = overlap_h + 1 if corner in ("bl", "br") else 1
    left_min = overlap_w + 1 if corner in ("tl", "bl") else 1
    right_min = overlap_w + 1 if corner in ("tr", "br") else 1

    top_margin = top_min + random.randint(0, 3)
    bottom_margin = bottom_min + random.randint(0, 3)
    left_margin = left_min + random.randint(0, 3)
    right_margin = right_min + random.randint(0, 3)

    middle_h = top_margin + top_h + bottom_margin
    middle_w = left_margin + top_w + right_margin
    if middle_h > 12 or middle_w > 12:
        return None

    extra_top, extra_right, extra_bottom, extra_left = _extra_extents(
        bottom_h, bottom_w, overlap_h, overlap_w, corner
    )

    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)

    mr0_min = max(extra_top, bottom_h - top_margin)
    mc0_min = max(extra_left, bottom_w - left_margin)
    mr0_max = min(grid_h - middle_h - extra_bottom, grid_h - top_margin - bottom_h - top_h)
    mc0_max = min(grid_w - middle_w - extra_right, grid_w - left_margin - bottom_w - top_w)
    if mr0_min > mr0_max or mc0_min > mc0_max:
        return None

    mr0 = random.randint(mr0_min, mr0_max)
    mc0 = random.randint(mc0_min, mc0_max)
    middle_rect = (mr0, mc0, middle_h, middle_w)
    top_rect = (mr0 + top_margin, mc0 + left_margin, top_h, top_w)
    bottom_rect = _build_bottom_rect(middle_rect, bottom_h, bottom_w, overlap_h, overlap_w, corner)

    top_color, middle_color, bottom_color = random.sample(PALETTE, 3)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paint_rect(grid, bottom_color, bottom_rect)
    _paint_rect(grid, middle_color, middle_rect)
    _paint_rect(grid, top_color, top_rect)

    if np.count_nonzero(grid == BACKGROUND) <= grid.size // 2:
        return None
    if not _edge_visible(grid, middle_color, middle_rect):
        return None
    if not _edge_visible(grid, bottom_color, bottom_rect):
        return None
    if not np.all(grid[top_rect[0] : top_rect[0] + top_h, top_rect[1] : top_rect[1] + top_w] == top_color):
        return None

    neg_steps = _step_count(grid.shape, top_rect, bottom_h, bottom_w, -1)
    pos_steps = _step_count(grid.shape, top_rect, bottom_h, bottom_w, 1)
    if neg_steps < 1 or pos_steps < 1:
        return None
    if neg_steps + pos_steps > 8:
        return None

    try:
        rects = _recover_rectangles(grid)
        order = _occlusion_order(grid, rects)
    except AssertionError:
        return None
    if order != [bottom_color, middle_color, top_color]:
        return None

    expected = _expected_output(grid, top_color, top_rect, bottom_rect)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, top_color, top_rect, bottom_rect


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _make_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input for puzzle 031")

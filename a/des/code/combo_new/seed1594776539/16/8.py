from __future__ import annotations

import random

import numpy as np


BG = 0
RECT_MIN = 3
RECT_MAX = 6


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_markers(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    non_bg_cols = np.flatnonzero((grid != bg).any(axis=0))
    assert non_bg_cols.size > 0, "expected a non-empty grid"

    marker_col = int(non_bg_cols[-1])
    marker_rows = [int(r) for r in np.flatnonzero(grid[:, marker_col] != bg)]
    assert len(marker_rows) in (3, 4), "marker column must encode 3-4 layers"

    colors = [int(grid[r, marker_col]) for r in marker_rows]
    assert len(set(colors)) == len(colors), "layer colors must be distinct"
    return marker_col, marker_rows, colors


def _rect_candidates(
    scene: np.ndarray,
    color: int,
    above_mask: np.ndarray,
    above_colors: set[int],
) -> list[tuple[int, int, int, int]]:
    coords = np.argwhere(scene == color)
    assert coords.size > 0, f"color {color} must remain visible"

    h, w = scene.shape
    min_r, min_c = map(int, coords.min(axis=0))
    max_r, max_c = map(int, coords.max(axis=0))
    candidates: list[tuple[int, int, int, int]] = []

    for rect_h in range(RECT_MIN, RECT_MAX + 1):
        if rect_h < max_r - min_r + 1:
            continue
        r0_lo = max(0, max_r - rect_h + 1)
        r0_hi = min(min_r, h - rect_h)
        for rect_w in range(RECT_MIN, RECT_MAX + 1):
            if rect_w < max_c - min_c + 1:
                continue
            c0_lo = max(0, max_c - rect_w + 1)
            c0_hi = min(min_c, w - rect_w)
            for r0 in range(r0_lo, r0_hi + 1):
                r1 = r0 + rect_h - 1
                for c0 in range(c0_lo, c0_hi + 1):
                    c1 = c0 + rect_w - 1
                    valid = True
                    for rr in range(r0, r1 + 1):
                        for cc in range(c0, c1 + 1):
                            cell = int(scene[rr, cc])
                            if above_mask[rr, cc]:
                                if cell == color or cell == BG or cell not in above_colors:
                                    valid = False
                                    break
                            elif cell != color:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        candidates.append((r0, c0, r1, c1))
    return candidates


def _recover_rectangles(grid: np.ndarray) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    marker_col, marker_rows, bottom_to_top = _find_markers(grid, bg)

    scene = np.array(grid, dtype=int)
    scene[marker_rows, marker_col] = bg

    above_mask = np.zeros_like(scene, dtype=bool)
    above_colors: set[int] = set()
    rects: dict[int, tuple[int, int, int, int]] = {}

    for color in reversed(bottom_to_top):
        candidates = _rect_candidates(scene, color, above_mask, above_colors)
        assert len(candidates) == 1, f"color {color} must map to one rectangle"
        rect = candidates[0]
        rects[color] = rect
        r0, c0, r1, c1 = rect
        above_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        above_colors.add(color)

    return bg, bottom_to_top, rects


def _render_output(
    bg: int,
    order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    heights = [int(rects[color][2] - rects[color][0] + 1) for color in order]
    widths = [int(rects[color][3] - rects[color][1] + 1) for color in order]
    out_h = max(heights)
    out_w = sum(widths) + len(widths) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for color in order:
        r0, c0, r1, c1 = rects[color]
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        out[:rect_h, c : c + rect_w] = color
        c += rect_w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each occluded solid rectangle from a stacked scene and lay the recovered layers out left-to-right.

    Input structure:
    - The background is black.
    - The left part of the grid contains 3-4 differently colored solid
      rectangles that partially overlap, so only the top visible color remains
      at every covered cell.
    - The rightmost occupied column is a vertical marker list whose colors, read
      from top to bottom, give the rectangle order from bottom layer to top layer.

    Rule:
    - Read the marker column to recover the stacking order.
    - Process the rectangles from top layer down to bottom layer: for each color,
      find the unique 3x3 to 6x6 solid rectangle whose visible cells are exactly
      that color and whose hidden cells are explained by already reconstructed
      higher rectangles.
    - Output the recovered full rectangles from left to right in bottom-to-top
      order, leaving one black column between neighbors and cropping to the
      minimum canvas that fits them.

    Key insight:
    - The marker pixels are not extra objects; they encode depth order, which
      makes the hidden parts of each axis-aligned rectangle uniquely solvable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, order, rects = _recover_rectangles(g)
    return _render_output(bg, order, rects)


# --------------- generator ---------------

GEN_RETRY_LIMIT = 2000
RECT_COUNT_OPTIONS = (3, 4)
GRID_H_RANGE = (14, 19)
GRID_W_RANGE = (16, 22)
LOCAL_H_RANGE = (13, 13)
LOCAL_W_RANGE = (14, 15)
MARGIN_RANGE = (0, 2)
RIGHT_MARGIN_RANGE = (1, 4)
BOTTOM_MARGIN_RANGE = (1, 4)
MARKER_GAP_OPTIONS = (1, 2)
PALETTE = tuple(range(1, 10))


def _overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _render_scene(
    shape: tuple[int, int],
    order: list[int],
    rects: dict[int, tuple[int, int, int, int]],
    marker_rows: list[int],
    marker_col: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for row, color in zip(marker_rows, order):
        grid[row, marker_col] = color
    return grid


def _visible_stats(
    scene: np.ndarray,
    color: int,
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    sub = scene[r0 : r1 + 1, c0 : c1 + 1]
    visible = sub == color
    visible_count = int(visible.sum())
    side_count = 0
    if np.any(sub[0, :] == color):
        side_count += 1
    if np.any(sub[-1, :] == color):
        side_count += 1
    if np.any(sub[:, 0] == color):
        side_count += 1
    if np.any(sub[:, -1] == color):
        side_count += 1
    return visible_count, side_count


def _sample_local_rects(
    n_rects: int,
    local_h: int,
    local_w: int,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    sizes = [
        (random.randint(RECT_MIN, RECT_MAX), random.randint(RECT_MIN, RECT_MAX))
        for _ in range(n_rects)
    ]

    for i, (rect_h, rect_w) in enumerate(sizes):
        positions: list[tuple[int, int, int, int]] = []
        for r0 in range(local_h - rect_h + 1):
            r1 = r0 + rect_h - 1
            for c0 in range(local_w - rect_w + 1):
                c1 = c0 + rect_w - 1
                rect = (r0, c0, r1, c1)
                if i == 0:
                    positions.append(rect)
                    continue
                overlaps = [_overlap_area(rect, prev) for prev in rects]
                if max(overlaps) == 0:
                    continue
                area = rect_h * rect_w
                if sum(overlaps) >= area - 1:
                    continue
                positions.append(rect)
        if not positions:
            return None
        rects.append(random.choice(positions))
    return rects


def _shift_rect(
    rect: tuple[int, int, int, int],
    dr: int,
    dc: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = random.choice(RECT_COUNT_OPTIONS)
    colors = random.sample(PALETTE, n_rects)

    local_h = random.randint(*LOCAL_H_RANGE)
    local_w = random.randint(*LOCAL_W_RANGE)
    local_rects = _sample_local_rects(n_rects, local_h, local_w)
    if local_rects is None:
        return None

    stack_right = max(rect[3] for rect in local_rects)
    top_pad = random.randint(*MARGIN_RANGE)
    left_pad = random.randint(*MARGIN_RANGE)
    right_pad = random.randint(*RIGHT_MARGIN_RANGE)
    bottom_pad = random.randint(*BOTTOM_MARGIN_RANGE)
    marker_gap = random.choice(MARKER_GAP_OPTIONS)
    marker_start = top_pad + random.randint(0, 2)
    marker_rows = [marker_start + i * marker_gap for i in range(n_rects)]

    rects = {
        color: _shift_rect(rect, top_pad, left_pad)
        for color, rect in zip(colors, local_rects)
    }
    marker_col = left_pad + stack_right + 1

    height = max(top_pad + local_h + bottom_pad, marker_rows[-1] + 1 + bottom_pad)
    width = left_pad + local_w + 1 + right_pad
    if not (GRID_H_RANGE[0] <= height <= GRID_H_RANGE[1]):
        return None
    if not (GRID_W_RANGE[0] <= width <= GRID_W_RANGE[1]):
        return None

    scene = _render_scene((height, width), colors, rects, marker_rows, marker_col)

    # Require genuine occlusion for every non-top layer and readable visible fragments.
    for i, color in enumerate(colors):
        rect = rects[color]
        visible_count, side_count = _visible_stats(scene, color, rect)
        area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
        if visible_count < 3 or side_count < 2:
            return None
        if i < n_rects - 1 and visible_count == area:
            return None
    if int(np.count_nonzero(scene[:, marker_col])) != n_rects:
        return None

    expected = _render_output(BG, colors, rects)
    try:
        actual = main(scene)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return scene, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        attempt = _attempt_generate()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRY_LIMIT} attempts")

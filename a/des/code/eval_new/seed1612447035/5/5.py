import random

import numpy as np


BLACK = 0
GREEN = 3
AZURE = 8
MAROON = 9

LEGEND_COLORS = (AZURE, MAROON, GREEN)
OBJECT_COLORS = (1, 2, 4, 5, 6, 7)

MIN_H, MAX_H = 12, 24
MIN_W, MAX_W = 14, 26
MAX_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer hidden rectangle layers and recolor each covered cell by overlap depth.

    Input structure:
    - The background is black.
    - Several solid axis-aligned colored rectangles occupy the grid; later rectangles may partially occlude earlier ones.
    - A vertical legend sits on the right edge: one azure pixel above one maroon pixel above one green pixel.

    Rule:
    - Ignore the original rectangle colors and first recover each rectangle's full bbox from its visible pixels.
    - Count, for every non-legend cell, how many recovered rectangles cover it.
    - Recolor depth 1 cells azure, depth 2 cells maroon, depth 3 cells green, and leave uncovered cells black.
    - Keep the right-edge legend unchanged.

    Key insight:
    - Each visible color patch is only the exposed part of a full rectangle; the output depends on inferred layer depth, not on the input colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_col, legend_mask = _find_legend(g)
    object_colors = [int(color) for color in np.unique(g[(g != BLACK) & (~legend_mask)])]
    assert object_colors, "expected at least one rectangle color"

    rects = []
    for color in object_colors:
        pts = np.argwhere((g == color) & (~legend_mask))
        assert len(pts) > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects.append((int(r0), int(c0), int(r1), int(c1), color))

    depth = np.zeros_like(g, dtype=int)
    for r0, c0, r1, c1, _color in rects:
        depth[r0 : r1 + 1, c0 : c1 + 1] += 1
    assert int(depth[:, : legend_col + 1].max(initial=0)) <= 3, "depth > 3 is not allowed"

    out = np.full_like(g, BLACK)
    for cover_depth, color in enumerate(LEGEND_COLORS, start=1):
        out[depth == cover_depth] = color
    out[legend_mask] = g[legend_mask]
    return out


def _find_legend(grid: np.ndarray) -> tuple[int, np.ndarray]:
    h, w = grid.shape
    candidates = []
    for c in range(w - 1, -1, -1):
        rows = np.flatnonzero(grid[:, c] != BLACK)
        if len(rows) != 3:
            continue
        if tuple(int(grid[r, c]) for r in rows) != LEGEND_COLORS:
            continue
        candidates.append(c)
    assert candidates, "could not find right-edge depth legend"
    legend_col = int(max(candidates))

    legend_mask = np.zeros_like(grid, dtype=bool)
    rows = np.flatnonzero(grid[:, legend_col] != BLACK)
    legend_mask[rows, legend_col] = True
    return legend_col, legend_mask


def _intersect(a, b):
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _overlap_size(a, b):
    inter = _intersect(a, b)
    if inter is None:
        return (0, 0)
    return (inter[2] - inter[0] + 1, inter[3] - inter[1] + 1)


def _flip_rect(rect, h: int, w: int, flip_v: bool, flip_h: bool):
    r0, c0, r1, c1, color = rect
    if flip_v:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_h:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return (r0, c0, r1, c1, color)


def _rect_with_margin_intersects(a, b, margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _render_input(h: int, w: int, rects, legend_rows):
    grid = np.full((h, w), BLACK, dtype=int)
    for r0, c0, r1, c1, color in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for row, color in zip(legend_rows, LEGEND_COLORS):
        grid[row, w - 1] = color
    return grid


def _render_output(h: int, w: int, rects, legend_rows):
    depth = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1, _color in rects:
        depth[r0 : r1 + 1, c0 : c1 + 1] += 1
    out = np.full((h, w), BLACK, dtype=int)
    for cover_depth, color in enumerate(LEGEND_COLORS, start=1):
        out[depth == cover_depth] = color
    for row, color in zip(legend_rows, LEGEND_COLORS):
        out[row, w - 1] = color
    return out


def _has_visible_corner(input_grid: np.ndarray, rect) -> bool:
    r0, c0, r1, c1, color = rect
    top = np.all(input_grid[r0, c0 : c1 + 1] == color)
    bottom = np.all(input_grid[r1, c0 : c1 + 1] == color)
    left = np.all(input_grid[r0 : r1 + 1, c0] == color)
    right = np.all(input_grid[r0 : r1 + 1, c1] == color)
    return (top and left) or (top and right) or (bottom and left) or (bottom and right)


def _visible_bbox_matches(input_grid: np.ndarray, rect) -> bool:
    r0, c0, r1, c1, color = rect
    pts = np.argwhere(input_grid == color)
    assert len(pts) > 0
    vr0, vc0 = pts.min(axis=0)
    vr1, vc1 = pts.max(axis=0)
    return (int(vr0), int(vc0), int(vr1), int(vc1)) == (r0, c0, r1, c1)


def _sample_legend_rows(h: int):
    return sorted(random.sample(range(h), 3))


def _build_single(h: int, work_w: int, colors):
    n_rects = random.randint(2, min(4, len(colors)))
    rects = []
    for color in colors[:n_rects]:
        placed = False
        for _ in range(120):
            height = random.randint(3, min(8, h))
            width = random.randint(3, min(8, work_w))
            r0 = random.randint(0, h - height)
            c0 = random.randint(0, work_w - width)
            rect = (r0, c0, r0 + height - 1, c0 + width - 1, color)
            if any(_rect_with_margin_intersects(rect, other, margin=1) for other in rects):
                continue
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return rects


def _build_double(h: int, work_w: int, colors):
    color_a, color_b = colors[:2]
    for _ in range(160):
        max_ha = min(13, h)
        max_wa = min(13, work_w)
        if max_ha < 6 or max_wa < 6:
            return None

        ha = random.randint(6, max_ha)
        wa = random.randint(6, max_wa)
        r0a = random.randint(0, h - ha)
        c0a = random.randint(0, work_w - wa)
        a = (r0a, c0a, r0a + ha - 1, c0a + wa - 1, color_a)

        r0b = random.randint(r0a + 1, min(r0a + ha - 2, h - 3))
        c0b = random.randint(c0a + 1, min(c0a + wa - 2, work_w - 3))
        hb = random.randint(3, min(10, h - r0b))
        wb = random.randint(3, min(10, work_w - c0b))
        b = (r0b, c0b, r0b + hb - 1, c0b + wb - 1, color_b)

        oh, ow = _overlap_size(a, b)
        if oh < 2 or ow < 2:
            continue

        flip_v = random.choice([False, True])
        flip_h = random.choice([False, True])
        return [
            _flip_rect(a, h, work_w, flip_v, flip_h),
            _flip_rect(b, h, work_w, flip_v, flip_h),
        ]
    return None


def _build_triple(h: int, work_w: int, colors):
    color_a, color_b, color_c = colors[:3]
    for _ in range(200):
        max_ha = min(15, h)
        max_wa = min(15, work_w)
        if max_ha < 8 or max_wa < 8:
            return None

        ha = random.randint(8, max_ha)
        wa = random.randint(8, max_wa)
        r0a = random.randint(0, h - ha)
        c0a = random.randint(0, work_w - wa)
        a = (r0a, c0a, r0a + ha - 1, c0a + wa - 1, color_a)

        r0b = random.randint(r0a + 1, min(r0a + ha - 3, h - 5))
        c0b = random.randint(c0a + 1, min(c0a + wa - 3, work_w - 5))
        hb = random.randint(5, min(12, h - r0b))
        wb = random.randint(5, min(12, work_w - c0b))
        b = (r0b, c0b, r0b + hb - 1, c0b + wb - 1, color_b)

        inter_ab = _intersect(a, b)
        if inter_ab is None:
            continue
        if inter_ab[2] - inter_ab[0] + 1 < 3 or inter_ab[3] - inter_ab[1] + 1 < 3:
            continue

        c_r0_min = inter_ab[0]
        c_c0_min = inter_ab[1]
        c_r1_max = min(inter_ab[2], b[2] - 1)
        c_c1_max = min(inter_ab[3], b[3] - 1)
        if c_r1_max - c_r0_min + 1 < 2 or c_c1_max - c_c0_min + 1 < 2:
            continue

        ch = random.randint(2, min(5, c_r1_max - c_r0_min + 1))
        cw = random.randint(2, min(5, c_c1_max - c_c0_min + 1))
        r0c = random.randint(c_r0_min, c_r1_max - ch + 1)
        c0c = random.randint(c_c0_min, c_c1_max - cw + 1)
        c = (r0c, c0c, r0c + ch - 1, c0c + cw - 1, color_c)

        oh, ow = _overlap_size(b, c)
        if oh < 2 or ow < 2:
            continue

        flip_v = random.choice([False, True])
        flip_h = random.choice([False, True])
        return [
            _flip_rect(a, h, work_w, flip_v, flip_h),
            _flip_rect(b, h, work_w, flip_v, flip_h),
            _flip_rect(c, h, work_w, flip_v, flip_h),
        ]
    return None


def _add_isolated_rects(h: int, work_w: int, rects, colors):
    max_extra = min(2, len(colors))
    n_extra = random.randint(0, max_extra)
    extras = []
    for color in colors[:n_extra]:
        placed = False
        for _ in range(120):
            height = random.randint(3, min(7, h))
            width = random.randint(3, min(7, work_w))
            r0 = random.randint(0, h - height)
            c0 = random.randint(0, work_w - width)
            rect = (r0, c0, r0 + height - 1, c0 + width - 1, color)
            if any(_rect_with_margin_intersects(rect, other, margin=1) for other in rects + extras):
                continue
            extras.append(rect)
            placed = True
            break
        if not placed:
            return None
    return rects + extras


def _gen_attempt(return_output: bool = False):
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    work_w = w - 1
    if work_w < 8:
        return None

    colors = list(OBJECT_COLORS)
    random.shuffle(colors)
    mode = random.choice(["single", "double", "triple"])

    if mode == "single":
        rects = _build_single(h, work_w, colors)
        target_depth = 1
        used = len(rects) if rects is not None else 0
    elif mode == "double":
        rects = _build_double(h, work_w, colors)
        target_depth = 2
        used = 2
    else:
        rects = _build_triple(h, work_w, colors)
        target_depth = 3
        used = 3

    if rects is None:
        return None

    remaining = [color for color in colors if color not in {rect[4] for rect in rects}]
    if mode != "single":
        rects = _add_isolated_rects(h, work_w, rects, remaining)
        if rects is None:
            return None

    legend_rows = _sample_legend_rows(h)
    inp = _render_input(h, w, rects, legend_rows)
    out = _render_output(h, w, rects, legend_rows)

    depth = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1, _color in rects:
        depth[r0 : r1 + 1, c0 : c1 + 1] += 1
    if int(depth[:, :work_w].max(initial=0)) != target_depth:
        return None
    if int(depth.max(initial=0)) > 3:
        return None

    for rect in rects:
        if not _has_visible_corner(inp, rect):
            return None
        if not _visible_bbox_matches(inp, rect):
            return None

    if not np.array_equal(main(inp), out):
        return None

    if return_output:
        return inp, out
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt(return_output=False)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_RETRIES} attempts")

from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 500
GRID_SIDE_RANGE = (14, 22)
BASE_HEIGHT_RANGE = (5, 9)
BASE_WIDTH_RANGE = (5, 10)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _shape_border(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    border = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not region[nr, nc]:
                border[rr, cc] = True
                break
    return border


def _infer_interior(loop_mask: np.ndarray) -> np.ndarray:
    h, w = loop_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not loop_mask[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            enqueue(nr, nc)

    return (~outside) & (~loop_mask)


def _outline_black_neighbors(fill_mask: np.ndarray) -> np.ndarray:
    h, w = fill_mask.shape
    outline = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(fill_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not fill_mask[nr, nc]:
                outline[nr, nc] = True
    return outline


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        while q:
            cr, cc2 = q.popleft()
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _is_simple_cycle(loop_mask: np.ndarray) -> bool:
    points = np.argwhere(loop_mask)
    if len(points) < 8:
        return False
    if _component_count(loop_mask) != 1:
        return False
    h, w = loop_mask.shape
    for r, c in points:
        rr = int(r)
        cc = int(c)
        degree = 0
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and loop_mask[nr, nc]:
                degree += 1
        if degree != 2:
            return False
    return True


def _semantic_transform(grid: np.ndarray) -> np.ndarray:
    red_interior = _infer_interior(grid == RED)
    blue_interior = _infer_interior(grid == BLUE)
    xor_mask = red_interior ^ blue_interior
    out = np.zeros_like(grid)
    out[xor_mask] = GREEN
    out[_outline_black_neighbors(xor_mask)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    XOR the interiors of two closed outlines, erase the outlines, then draw the outside border of the XOR region.

    Input structure:
    - The background is black.
    - Exactly two 1-pixel-wide closed outlines appear fully inside the grid: one blue and one red.
    - Each outline encloses a non-empty interior region, and the two inferred interior regions overlap partially.

    Rule:
    - For each outline, treat its own pixels as a blocking closed curve and flood from the grid boundary through 4-neighbor black space.
    - The cells that are neither reachable from the boundary nor part of that outline are the outline's interior.
    - Erase both original outlines, color the pixelwise XOR of the two interiors green, and color every black cell 4-adjacent to a green cell yellow.

    Key insight:
    - The logical operands are not the visible outline pixels; they must first be inferred as enclosed interior regions, and only the XOR result receives the final border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _semantic_transform(g)


def _make_region_patch(rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((16, 16), dtype=bool)
    base_h = int(rng.integers(BASE_HEIGHT_RANGE[0], BASE_HEIGHT_RANGE[1]))
    base_w = int(rng.integers(BASE_WIDTH_RANGE[0], BASE_WIDTH_RANGE[1]))
    r0 = int(rng.integers(3, 16 - base_h - 3))
    c0 = int(rng.integers(3, 16 - base_w - 3))
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

    plain = canvas.copy()

    if rng.random() < 0.7:
        side = str(rng.choice(["top", "bottom", "left", "right"]))
        if side in ("top", "bottom") and base_w >= 5:
            span = int(rng.integers(3, base_w - 1))
            start = int(rng.integers(c0 + 1, c0 + base_w - span))
            ext = int(rng.integers(1, 3))
            rows = slice(r0 - ext, r0) if side == "top" else slice(r0 + base_h, r0 + base_h + ext)
            canvas[rows, start : start + span] = True
        if side in ("left", "right") and base_h >= 5:
            span = int(rng.integers(3, base_h - 1))
            start = int(rng.integers(r0 + 1, r0 + base_h - span))
            ext = int(rng.integers(1, 3))
            cols = slice(c0 - ext, c0) if side == "left" else slice(c0 + base_w, c0 + base_w + ext)
            canvas[start : start + span, cols] = True

    for source in (canvas, plain):
        coords = np.argwhere(source)
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0)
        patch = source[rr0 : rr1 + 1, cc0 : cc1 + 1]
        border = _shape_border(patch)
        if _is_simple_cycle(border) and (patch & (~border)).sum() >= 5:
            return patch
    raise ValueError("degenerate contour")


def _place_patch(grid_shape: tuple[int, int], patch: np.ndarray, top: int, left: int) -> np.ndarray:
    placed = np.zeros(grid_shape, dtype=bool)
    h, w = patch.shape
    placed[top : top + h, left : left + w] = patch
    return placed


def _choose_top_left(rng: np.random.Generator, grid_shape: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    ph, pw = patch_shape
    top = int(rng.integers(1, h - ph))
    left = int(rng.integers(1, w - pw))
    return top, left


def _rectangle_region(grid_shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    region = np.zeros(grid_shape, dtype=bool)
    region[top : top + height, left : left + width] = True
    return region


def _try_add_bump(
    rng: np.random.Generator,
    region: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    if rng.random() >= 0.45:
        return region

    bumped = region.copy()
    side = str(rng.choice(["top", "bottom", "left", "right"]))
    if side in ("top", "bottom") and width >= 6:
        span = int(rng.integers(3, width - 1))
        start = left + int(rng.integers(1, width - span))
        ext = int(rng.integers(1, 3))
        rows = range(top - ext, top) if side == "top" else range(top + height, top + height + ext)
        if min(rows, default=1) < 1 or max(rows, default=region.shape[0] - 2) >= region.shape[0] - 1:
            return region
        bumped[min(rows) : max(rows) + 1, start : start + span] = True
    elif side in ("left", "right") and height >= 6:
        span = int(rng.integers(3, height - 1))
        start = top + int(rng.integers(1, height - span))
        ext = int(rng.integers(1, 3))
        cols = range(left - ext, left) if side == "left" else range(left + width, left + width + ext)
        if min(cols, default=1) < 1 or max(cols, default=region.shape[1] - 2) >= region.shape[1] - 1:
            return region
        bumped[start : start + span, min(cols) : max(cols) + 1] = True
    else:
        return region

    border = _shape_border(bumped)
    if _is_simple_cycle(border) and (bumped & (~border)).sum() >= 5:
        return bumped
    return region


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 23))
    w = int(rng.integers(16, 23))

    for _ in range(80):
        ha = int(rng.integers(10, min(15, h - 2)))
        wa = int(rng.integers(10, min(15, w - 2)))
        top_a = int(rng.integers(1, h - ha))
        left_a = int(rng.integers(1, w - wa))

        max_hb = ha - 4
        max_wb = wa - 4
        if max_hb < 5 or max_wb < 5:
            continue
        hb = int(rng.integers(5, max_hb + 1))
        wb = int(rng.integers(5, max_wb + 1))
        top_b = top_a + int(rng.integers(2, ha - hb - 1))
        left_b = left_a + int(rng.integers(2, wa - wb - 1))

        region_a = _rectangle_region((h, w), top_a, left_a, ha, wa)
        region_b = _rectangle_region((h, w), top_b, left_b, hb, wb)

        border_a = _shape_border(region_a)
        border_b = _shape_border(region_b)
        if np.any(border_a & border_b):
            continue
        if not _is_simple_cycle(border_a) or not _is_simple_cycle(border_b):
            continue

        interior_a = region_a & (~border_a)
        interior_b = region_b & (~border_b)
        overlap = interior_a & interior_b
        only_a = interior_a & (~interior_b)
        only_b = interior_b & (~interior_a)
        if not overlap.any() or not only_a.any():
            continue

        min_area = min(int(interior_a.sum()), int(interior_b.sum()))
        if overlap.sum() < max(5, min_area // 2):
            continue

        inferred_a = _infer_interior(border_a)
        inferred_b = _infer_interior(border_b)
        if not np.array_equal(inferred_a, interior_a):
            continue
        if not np.array_equal(inferred_b, interior_b):
            continue

        inp = np.zeros((h, w), dtype=int)
        inp[border_a] = RED
        inp[border_b] = BLUE

        expected = np.zeros((h, w), dtype=int)
        xor_mask = interior_a ^ interior_b
        expected[xor_mask] = GREEN
        expected[_outline_black_neighbors(xor_mask)] = YELLOW
        if not np.array_equal(main(inp), expected):
            continue

        if xor_mask.sum() > (h * w) // 3:
            continue
        if _outline_black_neighbors(xor_mask).sum() == 0:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

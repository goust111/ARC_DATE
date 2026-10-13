from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

LAYER_A_COLORS = (RED, ORANGE, MAROON)
LAYER_B_COLORS = (BLUE, AZURE, GREEN)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_HEIGHT_RANGE = (10, 19)
GRID_WIDTH_RANGE = (10, 19)
LOCAL_SHAPE_MIN = 4
LOCAL_SHAPE_MAX = 9
GENERATION_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate the two semi-transparent layers into side-by-side panels.

    Layer A uses the red family (red, orange, maroon), layer B uses the blue
    family (blue, azure, green), and grey marks overlap. The output shows the
    recovered A panel, one black separator column, and the recovered B panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    mask_a = np.isin(g, LAYER_A_COLORS)
    mask_b = np.isin(g, LAYER_B_COLORS)
    mask_overlap = g == GREY

    panel_a = np.zeros_like(g)
    panel_b = np.zeros_like(g)
    panel_a[mask_a] = g[mask_a]
    panel_b[mask_b] = g[mask_b]
    panel_a[mask_overlap] = RED
    panel_b[mask_overlap] = BLUE

    separator = np.zeros((g.shape[0], 1), dtype=int)
    return np.hstack((panel_a, separator, panel_b))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _grow_blob(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    mask[start] = True
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, h, w):
                if not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
        cells.add((rr, cc))
    return mask


def _union_rectangles(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed_h = random.randint(2, max(2, h - 1))
    seed_w = random.randint(2, max(2, w - 1))
    r0 = random.randint(0, h - seed_h)
    c0 = random.randint(0, w - seed_w)
    mask[r0 : r0 + seed_h, c0 : c0 + seed_w] = True
    for _ in range(count - 1):
        pts = np.argwhere(mask)
        pr, pc = map(int, pts[random.randrange(len(pts))])
        rect_h = random.randint(2, max(2, min(h, 5)))
        rect_w = random.randint(2, max(2, min(w, 5)))
        top = max(0, min(h - rect_h, pr - random.randrange(rect_h)))
        left = max(0, min(w - rect_w, pc - random.randrange(rect_w)))
        mask[top : top + rect_h, left : left + rect_w] = True
    return mask


def _walk_shape(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = random.randrange(h)
    c = random.randrange(w)
    mask[r, c] = True
    steps = 0
    while int(mask.sum()) < target_area and steps < target_area * 8:
        rr, cc = random.choice(list(_neighbors4(r, c, h, w)))
        r, c = rr, cc
        mask[r, c] = True
        if random.random() < 0.35:
            for nr, nc in _neighbors4(r, c, h, w):
                if int(mask.sum()) >= target_area:
                    break
                if random.random() < 0.3:
                    mask[nr, nc] = True
        steps += 1
    return mask


def _random_shape(h: int, w: int) -> np.ndarray:
    bbox_area = h * w
    min_area = max(5, bbox_area // 4)
    max_area = max(min_area + 1, (bbox_area * 2) // 3)
    for _ in range(80):
        mode = random.choice(("blob", "rects", "walk"))
        if mode == "blob":
            mask = _grow_blob(h, w, random.randint(min_area, max_area))
        elif mode == "rects":
            mask = _union_rectangles(h, w, random.randint(2, 3))
        else:
            mask = _walk_shape(h, w, random.randint(min_area, max_area))

        area = int(mask.sum())
        if min_area <= area <= max_area and _mask_connected(mask):
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                mask = np.flipud(mask)
            return mask
    return _grow_blob(h, w, min_area)


def _colorize(mask: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    colors = random.sample(list(palette), random.randint(1, len(palette)))
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    if len(colors) == 1:
        out[mask] = colors[0]
        return out

    mode = random.choice(("rows", "cols", "diag", "bands"))
    row_phase = random.randrange(len(colors))
    col_phase = random.randrange(len(colors))
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if mode == "rows":
            idx = (rr + row_phase) % len(colors)
        elif mode == "cols":
            idx = (cc + col_phase) % len(colors)
        elif mode == "diag":
            idx = (rr + cc + row_phase) % len(colors)
        else:
            idx = ((rr // 2) + (cc // 2) + row_phase) % len(colors)
        out[rr, cc] = colors[idx]
    return out


def _placed_mask(local_mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    h, w = local_mask.shape
    canvas[top : top + h, left : left + w] = local_mask
    return canvas


def _find_overlay_candidates(
    a_canvas: np.ndarray,
    b_local: np.ndarray,
    heavy_overlap: bool,
) -> list[tuple[int, int]]:
    h, w = a_canvas.shape
    bh, bw = b_local.shape
    b_area = int(b_local.sum())
    a_area = int(a_canvas.sum())
    candidates = []
    for top in range(h - bh + 1):
        for left in range(w - bw + 1):
            overlap = int(np.count_nonzero(a_canvas[top : top + bh, left : left + bw] & b_local))
            if overlap == 0:
                continue
            if a_area - overlap < 2 or b_area - overlap < 2:
                continue
            ratio = overlap / min(a_area, b_area)
            if heavy_overlap:
                if 0.28 <= ratio <= 0.6:
                    candidates.append((top, left))
            else:
                if 0.06 <= ratio <= 0.3:
                    candidates.append((top, left))
    if candidates:
        return candidates

    fallback = []
    for top in range(h - bh + 1):
        for left in range(w - bw + 1):
            overlap = int(np.count_nonzero(a_canvas[top : top + bh, left : left + bw] & b_local))
            if overlap >= 1 and a_area - overlap >= 2 and b_area - overlap >= 2:
                fallback.append((top, left))
    return fallback


def _build_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    a_h = random.randint(LOCAL_SHAPE_MIN, min(LOCAL_SHAPE_MAX, height - 1))
    a_w = random.randint(LOCAL_SHAPE_MIN, min(LOCAL_SHAPE_MAX, width - 1))
    b_h = random.randint(LOCAL_SHAPE_MIN, min(LOCAL_SHAPE_MAX, height - 1))
    b_w = random.randint(LOCAL_SHAPE_MIN, min(LOCAL_SHAPE_MAX, width - 1))

    a_local = _random_shape(a_h, a_w)
    b_local = _random_shape(b_h, b_w)

    a_top = random.randint(0, height - a_h)
    a_left = random.randint(0, width - a_w)
    a_canvas = _placed_mask(a_local, a_top, a_left, (height, width))

    heavy_overlap = random.random() < 0.5
    candidates = _find_overlay_candidates(a_canvas, b_local, heavy_overlap=heavy_overlap)
    if not candidates:
        raise ValueError("failed to place second layer")
    b_top, b_left = random.choice(candidates)
    b_canvas = _placed_mask(b_local, b_top, b_left, (height, width))

    overlap = a_canvas & b_canvas
    only_a = a_canvas & ~b_canvas
    only_b = b_canvas & ~a_canvas
    if not (overlap.any() and only_a.any() and only_b.any()):
        raise ValueError("degenerate overlap split")

    grid = np.zeros((height, width), dtype=int)
    a_colors = _colorize(a_local, LAYER_A_COLORS)
    b_colors = _colorize(b_local, LAYER_B_COLORS)

    a_color_canvas = np.zeros((height, width), dtype=int)
    b_color_canvas = np.zeros((height, width), dtype=int)
    a_color_canvas[a_top : a_top + a_h, a_left : a_left + a_w] = a_colors
    b_color_canvas[b_top : b_top + b_h, b_left : b_left + b_w] = b_colors
    grid[only_a] = a_color_canvas[only_a]
    grid[only_b] = b_color_canvas[only_b]
    grid[overlap] = GREY

    return grid, a_canvas, b_canvas


def _validate_generated(grid: np.ndarray, a_canvas: np.ndarray, b_canvas: np.ndarray) -> bool:
    if not (_mask_connected(a_canvas) and _mask_connected(b_canvas)):
        return False
    overlap = a_canvas & b_canvas
    only_a = a_canvas & ~b_canvas
    only_b = b_canvas & ~a_canvas
    if not (overlap.any() and only_a.any() and only_b.any()):
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, GREY, *LAYER_A_COLORS, *LAYER_B_COLORS}):
        return False
    out = main(grid)
    expected_width = grid.shape[1] * 2 + 1
    if out.shape != (grid.shape[0], expected_width):
        return False
    left = out[:, : grid.shape[1]]
    right = out[:, grid.shape[1] + 1 :]
    if not np.all(out[:, grid.shape[1]] == BLACK):
        return False
    if not np.array_equal(left[only_a], grid[only_a]):
        return False
    if not np.array_equal(right[only_b], grid[only_b]):
        return False
    if not np.array_equal(left[overlap], np.full(int(overlap.sum()), RED, dtype=int)):
        return False
    if not np.array_equal(right[overlap], np.full(int(overlap.sum()), BLUE, dtype=int)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid, a_canvas, b_canvas = _build_case()
        except ValueError:
            continue
        if _validate_generated(grid, a_canvas, b_canvas):
            return grid
    raise RuntimeError("failed to generate valid input")

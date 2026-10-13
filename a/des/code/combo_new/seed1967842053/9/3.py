import random

import numpy as np


BG = 0


def _bbox(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Recover each solid rectangle layer from a stacked color composite.

    Input structure:
    - The background is black.
    - The top row contains one-pixel color markers; reading them from left to
      right gives the layer order from bottom to top.
    - Below the marker row is one rectangular composite region made by drawing
      solid colored rectangles on top of one another.
    - Every layer color has visible pixels on the exposed composite, so its
      visible extent reveals the original rectangle boundary.

    Rule:
    - Read the marker colors from left to right to get the bottom-to-top order.
    - Ignore the marker row and find the bounding box of the visible composite.
    - For each marked color, recover its complete rectangle as the bounding box
      of its visible pixels in the composite.
    - Output the recovered rectangles in marker order, packed left to right,
      with one black separator column between neighboring layers.  Each
      rectangle keeps its vertical offset relative to the composite region.

    Key insight:
    - The marker row gives the stacking order, and the surviving edge pixels of
      each solid rectangle make its full axis-aligned extent recoverable.
    """
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.array(arr, copy=True)

    marker_cols = np.flatnonzero(arr[0] != BG)
    colors = [int(arr[0, c]) for c in marker_cols]
    body = np.array(arr, copy=True)
    body[0, :] = BG

    area_box = _bbox(body != BG)
    if area_box is None or not colors:
        return np.array(arr, copy=True)

    area_r0, area_r1, _, _ = area_box
    out_h = area_r1 - area_r0 + 1
    rects = []
    for color in colors:
        box = _bbox(body == color)
        if box is None:
            continue
        r0, r1, c0, c1 = box
        rects.append((color, r0 - area_r0, r1 - area_r0, c1 - c0 + 1))

    if not rects:
        return np.zeros((out_h, 1), dtype=arr.dtype)

    out_w = sum(w for _, _, _, w in rects) + len(rects) - 1
    out = np.zeros((out_h, out_w), dtype=arr.dtype)
    x = 0
    for color, r0, r1, width in rects:
        out[r0 : r1 + 1, x : x + width] = color
        x += width + 1
    return out


def _render(grid_h, grid_w, marker_cols, colors, rects):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for col, color in zip(marker_cols, colors):
        grid[0, col] = color
    for color, (r0, r1, c0, c1) in zip(colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _valid_generated(grid, colors, rects):
    body = np.array(grid, copy=True)
    body[0, :] = BG
    for color, rect in zip(colors, rects):
        box = _bbox(body == color)
        if box != rect:
            return False
    area = _bbox(body != BG)
    return area == rects[0]


def _sample_widths(n):
    for _ in range(200):
        base_w = random.randint(4, 11)
        widths = [base_w] + [random.randint(2, base_w) for _ in range(n - 1)]
        if sum(widths) + n - 1 <= 30:
            return widths
    base_w = random.randint(4, 6)
    return [base_w] + [random.randint(2, base_w) for _ in range(n - 1)]


def generate_input() -> np.ndarray:
    for _ in range(1000):
        n = random.randint(2, 5)
        colors = random.sample(range(1, 10), n)
        widths = _sample_widths(n)

        area_h = random.randint(5, 14)
        heights = [area_h] + [random.randint(2, area_h) for _ in range(n - 1)]
        area_top = random.randint(2, 5)
        bottom_margin = random.randint(0, max(0, 30 - (area_top + area_h)))
        grid_h = area_top + area_h + bottom_margin

        min_w = max(widths[0] + 2, 2 * n + 1)
        grid_w = random.randint(min_w, 30)
        area_left = random.randint(1, grid_w - widths[0] - 1)
        marker_cols = sorted(random.sample(range(grid_w), n))

        rects = [(area_top, area_top + area_h - 1, area_left, area_left + widths[0] - 1)]
        for width, height in zip(widths[1:], heights[1:]):
            row_off = random.randint(0, area_h - height)
            col_off = random.randint(0, widths[0] - width)
            r0 = area_top + row_off
            c0 = area_left + col_off
            rects.append((r0, r0 + height - 1, c0, c0 + width - 1))

        grid = _render(grid_h, grid_w, marker_cols, colors, rects)
        if not _valid_generated(grid, colors, rects):
            continue

        out_w = sum(widths) + n - 1
        if max(grid.shape) <= 30 and area_h <= 30 and out_w <= 30:
            return grid

    colors = random.sample(range(1, 10), 3)
    grid_h, grid_w = 12, 16
    area_top, area_left = 3, 3
    rects = [
        (area_top, area_top + 6, area_left, area_left + 6),
        (area_top + 1, area_top + 4, area_left + 1, area_left + 4),
        (area_top + 2, area_top + 3, area_left + 2, area_left + 5),
    ]
    return _render(grid_h, grid_w, [1, 4, 7], colors, rects)

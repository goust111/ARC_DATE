import random
from collections import deque

import numpy as np


RECT_COLORS = (3, 4, 5, 7, 8)


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _find_axes(grid):
    blue_row = blue_col = red_row = red_col = None
    for r in range(grid.shape[0]):
        vals = set(grid[r, :])
        if vals <= {0, 1, 9} and 1 in vals and np.count_nonzero(grid[r, :] == 1) >= grid.shape[1] - 1:
            blue_row = r
        if vals <= {0, 2, 9} and 2 in vals and np.count_nonzero(grid[r, :] == 2) >= grid.shape[1] - 1:
            red_row = r
    for c in range(grid.shape[1]):
        vals = set(grid[:, c])
        if vals <= {0, 1, 9} and 1 in vals and np.count_nonzero(grid[:, c] == 1) >= grid.shape[0] - 1:
            blue_col = c
        if vals <= {0, 2, 9} and 2 in vals and np.count_nonzero(grid[:, c] == 2) >= grid.shape[0] - 1:
            red_col = c
    return blue_row, blue_col, red_row, red_col


def _distance(info, row_axis, col_axis):
    r0, c0, r1, c1 = info["bbox"]
    row_dist = min(abs(r - row_axis) for r in (r0, r1)) if row_axis is not None else None
    col_dist = min(abs(c - col_axis) for c in (c0, c1)) if col_axis is not None else None
    return row_dist, col_dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Sort rectangles by their distances to the blue and red coordinate axes, then emit horizontal and vertical normalized copies in two stacked bands.

    Input Model:
    A blue axis line and a red axis line meet at a brown corner, while several solid rectangles of distinct size pairs sit away from the axes inside the grid.

    Algorithm:
    Detect whether the blue axis is horizontal or vertical, compute each rectangle's shortest Manhattan distance to the blue line and then to the red line, sort by those keys, and lay out normalized horizontal and vertical versions in the top and bottom halves according to the blue-axis orientation rule.

    Output:
    Return a 10-row canvas whose two five-row bands contain the ordered normalized rectangles, with the narrower band left-aligned and all source axes removed.
    """
    grid = np.asarray(grid, dtype=int)
    blue_row, blue_col, red_row, red_col = _find_axes(grid)
    rects = []
    for cells in _components(np.isin(grid, RECT_COLORS)):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        color = int(grid[cells[0]])
        rects.append({"bbox": (min(rows), min(cols), max(rows), max(cols)), "color": color})
    blue_horizontal = blue_row is not None
    for info in rects:
        if blue_horizontal:
            info["key"] = _distance(info, blue_row, red_col)
        else:
            info["key"] = _distance(info, red_row, blue_col)
    rects.sort(key=lambda item: item["key"])
    top_parts = []
    bottom_parts = []
    for info in rects:
        r0, c0, r1, c1 = info["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        horiz = np.full((min(h, w), max(h, w)), info["color"], dtype=int)
        vert = np.full((max(h, w), min(h, w)), info["color"], dtype=int)
        if blue_horizontal:
            top_parts.append(horiz)
            bottom_parts.append(vert)
        else:
            top_parts.append(vert)
            bottom_parts.append(horiz)
    width = max(sum(p.shape[1] for p in parts) + len(parts) - 1 for parts in (top_parts, bottom_parts))
    out = np.zeros((10, width), dtype=int)
    c = 0
    for part in top_parts:
        out[: part.shape[0], c : c + part.shape[1]] = part
        c += part.shape[1] + 1
    c = 0
    for part in bottom_parts:
        out[10 - part.shape[0] :, c : c + part.shape[1]] = part
        c += part.shape[1] + 1
    return out


def _generate_case(rng: random.Random):
    h = rng.randint(12, 15)
    w = rng.randint(12, 15)
    grid = np.zeros((h, w), dtype=int)
    blue_horizontal = rng.choice((True, False))
    if blue_horizontal:
        grid[0, :] = 1
        grid[:, 0] = 2
    else:
        grid[0, :] = 2
        grid[:, 0] = 1
    grid[0, 0] = 9
    rect_count = rng.randint(3, 5)
    sizes = rng.sample([(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (4, 1)], rect_count)
    colors = rng.sample(RECT_COLORS, rect_count)
    keys = set()
    placed = []
    for (rh, rw), color in zip(sizes, colors):
        for _ in range(1000):
            r0 = rng.randint(2, h - rh)
            c0 = rng.randint(2, w - rw)
            if np.any(grid[r0 : r0 + rh, c0 : c0 + rw] != 0):
                continue
            blue_dist = r0 if blue_horizontal else c0
            red_dist = c0 if blue_horizontal else r0
            if (blue_dist, red_dist) in keys:
                continue
            keys.add((blue_dist, red_dist))
            grid[r0 : r0 + rh, c0 : c0 + rw] = color
            placed.append(((r0, c0, r0 + rh - 1, c0 + rw - 1), color))
            break
    rects = [{"bbox": bbox, "color": color, "key": ((bbox[0] if blue_horizontal else bbox[1]), (bbox[1] if blue_horizontal else bbox[0]))} for bbox, color in placed]
    rects.sort(key=lambda item: item["key"])
    top_parts = []
    bottom_parts = []
    for info in rects:
        r0, c0, r1, c1 = info["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        horiz = np.full((min(hh, ww), max(hh, ww)), info["color"], dtype=int)
        vert = np.full((max(hh, ww), min(hh, ww)), info["color"], dtype=int)
        if blue_horizontal:
            top_parts.append(horiz)
            bottom_parts.append(vert)
        else:
            top_parts.append(vert)
            bottom_parts.append(horiz)
    out_w = max(sum(p.shape[1] for p in parts) + len(parts) - 1 for parts in (top_parts, bottom_parts))
    expected = np.zeros((10, out_w), dtype=int)
    c = 0
    for part in top_parts:
        expected[: part.shape[0], c : c + part.shape[1]] = part
        c += part.shape[1] + 1
    c = 0
    for part in bottom_parts:
        expected[10 - part.shape[0] :, c : c + part.shape[1]] = part
        c += part.shape[1] + 1
    return grid, main(grid)


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]

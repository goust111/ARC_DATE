import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors(r, c, h, w):
    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid):
    grey = grid == GREY
    candidates = []
    for comp in _components(grey):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        interior = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                val = int(grid[r, c])
                if val not in (BLACK, GREY):
                    interior.append((r, c, val))
        if interior:
            candidates.append((len(interior), -(r1 - r0 + 1) * (c1 - c0 + 1), r0, c0, r1, c1, interior))

    if not candidates:
        return None
    _, _, r0, c0, r1, c1, interior = max(candidates)
    legend_colors = [v for _, _, v in sorted(interior, key=lambda item: (item[1], item[0]))]
    return (r0, c0, r1, c1), legend_colors


def _path_order(path_mask):
    h, w = path_mask.shape
    cells = [(r, c) for r in range(h) for c in range(w) if path_mask[r, c]]
    if not cells:
        return []

    degrees = {}
    for r, c in cells:
        degrees[(r, c)] = sum(1 for nr, nc in _neighbors(r, c, h, w) if path_mask[nr, nc])
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    start = min(endpoints if endpoints else cells)

    order = []
    visited = set()
    prev = None
    cur = start
    while cur is not None and cur not in visited:
        order.append(cur)
        visited.add(cur)
        r, c = cur
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors(r, c, h, w)
            if path_mask[nr, nc] and (nr, nc) != prev and (nr, nc) not in visited
        ]
        prev, cur = cur, (min(nxt) if nxt else None)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey path segment by the left-to-right legend colors, erasing path markers.

    Input structure:
    The grid has a black background, one unbranched one-pixel-wide 4-connected path made of grey
    pixels with colored landmark pixels sitting on path positions, and a separate small grey
    rectangular legend frame.  Inside that frame, the same set of colors appears once each in a
    horizontal left-to-right order.

    Rule:
    Find the two path endpoints and begin at the endpoint with the smaller row, breaking ties by
    smaller column.  Walk along the path and treat each non-grey colored path pixel as a segment
    divider.  The grey pixels before the first divider are recolored with the first legend color,
    the grey pixels between the first and second dividers with the second legend color, and so on;
    grey pixels after the last divider use the next legend color cyclically.  Colored divider pixels
    on the path become black.  The legend frame and its interior colors are left unchanged.

    Key insight:
    The colored path pixels determine only where the path is split and in what encountered order;
    the colors used for repainting come from the independent spatial order of the legend.
    """
    arr = np.asarray(grid)
    found = _find_legend(arr)
    if found is None:
        return arr.copy()

    (r0, c0, r1, c1), legend_colors = found
    out = arr.copy()
    legend_region = np.zeros(arr.shape, dtype=bool)
    legend_region[r0 : r1 + 1, c0 : c1 + 1] = True
    path_mask = (arr != BLACK) & ~legend_region
    order = _path_order(path_mask)
    if not order or not legend_colors:
        return out

    segment = 0
    for r, c in order:
        val = int(arr[r, c])
        if val == GREY:
            out[r, c] = legend_colors[segment % len(legend_colors)]
        elif val != BLACK:
            out[r, c] = BLACK
            segment += 1
    return out


def _make_path(h, w, kind):
    if kind == 0:
        r = random.randrange(1, h - 1)
        return [(r, c) for c in range(w)]
    if kind == 1:
        c = random.randrange(1, w - 1)
        return [(r, c) for r in range(h)]
    if kind == 2:
        sign = random.choice([-1, 1])
        if sign == 1:
            start_c = random.randrange(0, w - 3)
            max_shift = w - 1 - start_c
        else:
            start_c = random.randrange(3, w)
            max_shift = start_c
        shift = random.randint(2, max_shift)
        moves = ["D"] * (h - 1) + ["H"] * shift
        random.shuffle(moves)
        r, c = 0, start_c
        path = [(r, c)]
        for move in moves:
            if move == "D":
                r += 1
            else:
                c += sign
            path.append((r, c))
        return path

    sign = random.choice([-1, 1])
    if sign == 1:
        start_r = random.randrange(0, h - 3)
        max_shift = h - 1 - start_r
    else:
        start_r = random.randrange(3, h)
        max_shift = start_r
    shift = random.randint(2, max_shift)
    moves = ["R"] * (w - 1) + ["V"] * shift
    random.shuffle(moves)
    r, c = start_r, 0
    path = [(r, c)]
    for move in moves:
        if move == "R":
            c += 1
        else:
            r += sign
        path.append((r, c))
    return path


def _choose_marker_indices(length, count):
    if length < count * 2 + 3:
        return None
    choices = list(range(1, length - 1))
    for _ in range(200):
        picks = sorted(random.sample(choices, count))
        gaps = [picks[0]] + [b - a for a, b in zip(picks, picks[1:])] + [length - 1 - picks[-1]]
        if min(gaps) >= 2:
            return picks
    step = (length - 1) / (count + 1)
    picks = sorted({int(round(step * k)) for k in range(1, count + 1)})
    if len(picks) == count and picks[0] >= 1 and picks[-1] <= length - 2:
        return picks
    return None


def _place_legend(grid, legend_colors):
    h, w = grid.shape
    frame_h = 3
    frame_w = 2 * len(legend_colors) + 1
    if frame_h > h or frame_w > w:
        return False

    row_range = range(1, h - frame_h) if h - frame_h > 1 else range(0, h - frame_h + 1)
    col_range = range(1, w - frame_w) if w - frame_w > 1 else range(0, w - frame_w + 1)
    candidates = []
    for top in row_range:
        for left in col_range:
            rr0, rr1 = max(0, top - 1), min(h, top + frame_h + 1)
            cc0, cc1 = max(0, left - 1), min(w, left + frame_w + 1)
            if not np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
                candidates.append((top, left))
    if not candidates:
        return False

    top, left = random.choice(candidates)
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    for i, color in enumerate(legend_colors):
        grid[top + 1, left + 1 + 2 * i] = color
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(13, 30)
        w = random.randint(13, 30)
        color_count = random.randint(3, 5)
        colors = random.sample(COLORS, color_count)
        legend_colors = colors[:]
        marker_colors = colors[:]
        random.shuffle(legend_colors)
        random.shuffle(marker_colors)

        path = _make_path(h, w, random.randrange(4))
        marker_indices = _choose_marker_indices(len(path), color_count)
        if marker_indices is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r, c in path:
            grid[r, c] = GREY
        for idx, color in zip(marker_indices, marker_colors):
            r, c = path[idx]
            grid[r, c] = color

        if _place_legend(grid, legend_colors):
            return grid

    h, w = 18, 22
    grid = np.zeros((h, w), dtype=int)
    path = [(9, c) for c in range(w)]
    colors = random.sample(COLORS, 4)
    legend_colors = colors[:]
    marker_colors = colors[:]
    random.shuffle(legend_colors)
    random.shuffle(marker_colors)
    for r, c in path:
        grid[r, c] = GREY
    for idx, color in zip((4, 8, 13, 17), marker_colors):
        r, c = path[idx]
        grid[r, c] = color
    _place_legend(grid, legend_colors)
    return grid

import random
from collections import deque

import numpy as np


def _background_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _foreground_colors(grid, background):
    seen = set()
    colors = []
    for value in grid.ravel():
        color = int(value)
        if color != background and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _component_boxes(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    boxes = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        min_r = max_r = sr
        min_c = max_c = sc
        count = 0
        while q:
            r, c = q.popleft()
            count += 1
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            if c < min_c:
                min_c = c
            if c > max_c:
                max_c = c
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        boxes.append((min_r, max_r, min_c, max_c, count))
    return boxes


def _add_proposals(grid, background, proposals, color, r0, r1, c0, c1):
    if r0 > r1 or c0 > c1:
        return
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != background:
                continue
            key = (r, c)
            if key not in proposals:
                proposals[key] = color
            elif proposals[key] != color:
                proposals[key] = None


def main(grid: np.ndarray) -> np.ndarray:
    """Bridge each same-color rectangle pair through its shared row or column band.

    Input structure:
    - The background is the most common color in the grid.
    - Every foreground color appears as exactly two disconnected solid rectangles.
    - The two rectangles of one color share either a row interval or a column interval, but not both.
    - Rectangles of other colors may lie in the empty space between a matching pair.

    Rule:
    - For each foreground color, find the two connected components of that color and their bounding rectangles.
    - If their row ranges overlap, fill only background cells in the shared rows and in the column gap between them.
    - If their column ranges overlap, fill only background cells in the shared columns and in the row gap between them.
    - Existing non-background cells are never overwritten, so intervening rectangles remain visible and clip the bridge.

    Key insight:
    - The bridge direction is determined by the overlap axis of the paired rectangles, and other colored rectangles act as blockers because only background cells may be filled.
    """
    background = _background_color(grid)
    proposals = {}

    for color in _foreground_colors(grid, background):
        boxes = _component_boxes(grid == color)
        if len(boxes) != 2:
            continue
        a, b = boxes
        row_overlap = max(a[0], b[0]) <= min(a[1], b[1])
        col_overlap = max(a[2], b[2]) <= min(a[3], b[3])

        if row_overlap and not col_overlap:
            if a[3] < b[2]:
                left, right = a, b
            elif b[3] < a[2]:
                left, right = b, a
            else:
                continue
            r0 = max(left[0], right[0])
            r1 = min(left[1], right[1])
            c0 = left[3] + 1
            c1 = right[2] - 1
            _add_proposals(grid, background, proposals, color, r0, r1, c0, c1)
        elif col_overlap and not row_overlap:
            if a[1] < b[0]:
                top, bottom = a, b
            elif b[1] < a[0]:
                top, bottom = b, a
            else:
                continue
            r0 = top[1] + 1
            r1 = bottom[0] - 1
            c0 = max(top[2], bottom[2])
            c1 = min(top[3], bottom[3])
            _add_proposals(grid, background, proposals, color, r0, r1, c0, c1)

    out = np.array(grid, copy=True)
    for (r, c), color in proposals.items():
        if color is not None:
            out[r, c] = color
    return out


def _empty(grid, rect):
    r0, r1, c0, c1 = rect
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == 0))


def _has_blocker(grid, r0, r1, c0, c1):
    if r0 > r1 or c0 > c1:
        return False
    return bool(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != 0))


def _random_pair(grid, orientation, want_blocker):
    rows, cols = grid.shape
    tries = 700 if want_blocker else 220
    for _ in range(tries):
        if orientation == "h":
            h1 = random.randint(1, min(4, rows))
            h2 = random.randint(1, min(4, rows))
            w1 = random.randint(1, min(4, max(1, cols // 4)))
            w2 = random.randint(1, min(4, max(1, cols // 4)))
            if cols - w1 - w2 < 2:
                continue
            r1_top = random.randint(0, rows - h1)
            low = max(0, r1_top - h2 + 1)
            high = min(rows - h2, r1_top + h1 - 1)
            if low > high:
                continue
            r2_top = random.randint(low, high)
            gap = random.randint(2, cols - w1 - w2)
            c1_left = random.randint(0, cols - w1 - gap - w2)
            c2_left = c1_left + w1 + gap
            rect1 = (r1_top, r1_top + h1 - 1, c1_left, c1_left + w1 - 1)
            rect2 = (r2_top, r2_top + h2 - 1, c2_left, c2_left + w2 - 1)
            bridge = (
                max(rect1[0], rect2[0]),
                min(rect1[1], rect2[1]),
                rect1[3] + 1,
                rect2[2] - 1,
            )
        else:
            h1 = random.randint(1, min(4, max(1, rows // 4)))
            h2 = random.randint(1, min(4, max(1, rows // 4)))
            w1 = random.randint(1, min(4, cols))
            w2 = random.randint(1, min(4, cols))
            if rows - h1 - h2 < 2:
                continue
            c1_left = random.randint(0, cols - w1)
            low = max(0, c1_left - w2 + 1)
            high = min(cols - w2, c1_left + w1 - 1)
            if low > high:
                continue
            c2_left = random.randint(low, high)
            gap = random.randint(2, rows - h1 - h2)
            r1_top = random.randint(0, rows - h1 - gap - h2)
            r2_top = r1_top + h1 + gap
            rect1 = (r1_top, r1_top + h1 - 1, c1_left, c1_left + w1 - 1)
            rect2 = (r2_top, r2_top + h2 - 1, c2_left, c2_left + w2 - 1)
            bridge = (
                rect1[1] + 1,
                rect2[0] - 1,
                max(rect1[2], rect2[2]),
                min(rect1[3], rect2[3]),
            )

        if not (_empty(grid, rect1) and _empty(grid, rect2)):
            continue
        if want_blocker and not _has_blocker(grid, *bridge):
            continue
        return rect1, rect2
    return None


def _place(grid, rect, color):
    r0, r1, c0, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def generate_input() -> np.ndarray:
    for _ in range(80):
        rows = random.randint(14, 30)
        cols = random.randint(14, 30)
        pair_count = random.randint(2, 4)
        colors = random.sample(range(1, 10), pair_count)
        directions = ["h", "v"] * (pair_count // 2)
        if pair_count % 2:
            directions.append("h" if random.random() < 0.5 else "v")
        random.shuffle(directions)

        grid = np.zeros((rows, cols), dtype=np.int64)
        ok = True
        for i, (color, direction) in enumerate(zip(colors, directions)):
            want_blocker = i > 0 and random.random() < 0.45
            pair = _random_pair(grid, direction, want_blocker)
            if pair is None and want_blocker:
                pair = _random_pair(grid, direction, False)
            if pair is None:
                ok = False
                break
            rect1, rect2 = pair
            _place(grid, rect1, color)
            _place(grid, rect2, color)
        if ok:
            return grid

    grid = np.zeros((18, 18), dtype=np.int64)
    for color, rect1, rect2 in (
        (1, (2, 4, 1, 3), (3, 5, 11, 13)),
        (2, (0, 2, 7, 8), (10, 12, 7, 9)),
        (3, (8, 9, 2, 4), (8, 11, 14, 16)),
    ):
        _place(grid, rect1, color)
        _place(grid, rect2, color)
    return grid

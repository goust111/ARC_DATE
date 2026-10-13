import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
MARKER_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)


def _black_rectangles(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != BLACK:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            if bottom > top and right > left:
                rects.append((top, left, bottom, right))
    return rects


def _depth(rects, r, c):
    return sum(1 for top, left, bottom, right in rects if top < r < bottom and left < c < right)


def _line(a, b):
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    return []


def main(grid: np.ndarray) -> np.ndarray:
    """Connect same-colored marker pairs through nested rectangular fences.

    Input structure:
    The grid has a yellow background, black one-pixel rectangular frame borders,
    and colored marker pixels. Each non-yellow, non-black marker color appears
    exactly twice, and the two markers of a color share either a row or a column.

    Rule:
    For each same-colored pair, count how many black rectangular frames strictly
    contain each endpoint. If the two depths sum to an even value, color every
    yellow cell on the shared row or column between the endpoints. If the sum is
    odd, start at the upper or left endpoint and color alternating cells along
    the path. Black frame cells are never overwritten and still count in the
    alternating rhythm.

    Key insight:
    The transformation depends on pairing equal colors, measuring each marker's
    nesting depth inside the visible black frames, then using the parity of the
    depth sum to choose solid versus dashed rail drawing.
    """
    out = grid.copy()
    rects = _black_rectangles(grid)
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, YELLOW)]

    for color in colors:
        pts = np.argwhere(grid == color)
        if len(pts) != 2:
            continue
        a = tuple(int(x) for x in pts[0])
        b = tuple(int(x) for x in pts[1])
        path = _line(a, b)
        if not path:
            continue

        dashed = (_depth(rects, *a) + _depth(rects, *b)) % 2 == 1
        for i, (r, c) in enumerate(path):
            if grid[r, c] == BLACK:
                continue
            if dashed and i % 2 == 1:
                continue
            if out[r, c] == YELLOW or grid[r, c] == color:
                out[r, c] = color
    return out


def _draw_frame(grid, rect):
    top, left, bottom, right = rect
    grid[top, left : right + 1] = BLACK
    grid[bottom, left : right + 1] = BLACK
    grid[top : bottom + 1, left] = BLACK
    grid[top : bottom + 1, right] = BLACK


def _build_frames(h, w, depth_count, gaps):
    min_span = 5 + 2 * sum(gaps)
    top = random.randint(1, min(3, h - min_span - 1))
    left = random.randint(1, min(3, w - min_span - 1))
    bottom_margin = random.randint(1, min(3, h - top - min_span))
    right_margin = random.randint(1, min(3, w - left - min_span))
    bottom = h - 1 - bottom_margin
    right = w - 1 - right_margin

    frames = []
    for i in range(depth_count):
        frames.append((top, left, bottom, right))
        if i < depth_count - 1:
            gap = gaps[i]
            top += gap
            left += gap
            bottom -= gap
            right -= gap
    return frames


def _path_is_clear(grid, path, reserved):
    for r, c in path:
        if (r, c) in reserved:
            return False
        if grid[r, c] not in (YELLOW, BLACK):
            return False
    return True


def _choose_pair(grid, rects, parity, reserved):
    h, w = grid.shape
    candidates = []

    for r in range(h):
        cols = [c for c in range(w) if grid[r, c] == YELLOW and (r, c) not in reserved]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                if c2 - c1 < 4:
                    continue
                if (_depth(rects, r, c1) + _depth(rects, r, c2)) % 2 != parity:
                    continue
                path = [(r, c) for c in range(c1, c2 + 1)]
                if _path_is_clear(grid, path, reserved):
                    candidates.append(((r, c1), (r, c2), path))

    for c in range(w):
        rows = [r for r in range(h) if grid[r, c] == YELLOW and (r, c) not in reserved]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                r1, r2 = rows[i], rows[j]
                if r2 - r1 < 4:
                    continue
                if (_depth(rects, r1, c) + _depth(rects, r2, c)) % 2 != parity:
                    continue
                path = [(r, c) for r in range(r1, r2 + 1)]
                if _path_is_clear(grid, path, reserved):
                    candidates.append(((r1, c), (r2, c), path))

    if not candidates:
        return None
    return random.choice(candidates)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=np.int8)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid

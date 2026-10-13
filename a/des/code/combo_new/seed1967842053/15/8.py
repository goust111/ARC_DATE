import random
from collections import deque

import numpy as np


MAROON = 9
BACKGROUND = 0


def _background_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _maroon_components(grid):
    rows, cols = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != MAROON or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == MAROON:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _rectangle_from_component(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    top, bottom = min(rs), max(rs)
    left, right = min(cs), max(cs)
    if bottom - top < 2 or right - left < 2:
        return None
    expected = set()
    for c in range(left, right + 1):
        expected.add((top, c))
        expected.add((bottom, c))
    for r in range(top, bottom + 1):
        expected.add((r, left))
        expected.add((r, right))
    if set(comp) != expected:
        return None
    return top, left, bottom, right


def _inner_wall_path(top, left, bottom, right):
    path = []
    if bottom - top < 2 or right - left < 2:
        return path
    for c in range(left + 1, right):
        path.append((top + 1, c))
    for r in range(top + 2, bottom):
        path.append((r, right - 1))
    if bottom - top > 2:
        for c in range(right - 2, left, -1):
            path.append((bottom - 1, c))
    if right - left > 2:
        for r in range(bottom - 2, top + 1, -1):
            path.append((r, left + 1))
    return path


def _sort_inner_colors(grid, rect, bg):
    top, left, bottom, right = rect
    stats = {}
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            color = int(grid[r, c])
            if color == bg or color == MAROON:
                continue
            if color not in stats:
                stats[color] = [0, (r, c)]
            stats[color][0] += 1
    return sorted(stats.items(), key=lambda item: (-item[1][0], item[1][1][0], item[1][1][1]))


def main(grid: np.ndarray) -> np.ndarray:
    """Organize each fence's scattered interior colors onto its inner wall.

    Input structure:
    A black-background grid contains several non-overlapping one-pixel maroon
    rectangular fences. Inside each fence are scattered colored pixels that are
    neither maroon nor background, with multiple colors and varying counts.

    Rule:
    For every maroon rectangular fence, count the interior pixels by color and
    break count ties by the first row-major position where that color appears
    inside the fence. Clear the fence interior to the background color, then
    place each color's counted pixels consecutively along the inner wall,
    starting at the interior upper-left corner, moving right along the top
    inner edge, then clockwise around the remaining inner edges. The maroon
    border stays unchanged.

    Key insight:
    The random interior pixels are not moved by their original positions; they
    are grouped by color count and rewritten as a compact clockwise run along
    the inside perimeter of each rectangle.
    """
    out = np.array(grid, copy=True)
    bg = _background_color(out)
    rects = []
    for comp in _maroon_components(out):
        rect = _rectangle_from_component(comp)
        if rect is not None:
            rects.append(rect)

    for rect in rects:
        top, left, bottom, right = rect
        ordered = _sort_inner_colors(out, rect, bg)
        out[top + 1:bottom, left + 1:right] = bg
        path = _inner_wall_path(top, left, bottom, right)
        pos = 0
        for color, (count, _) in ordered:
            for _ in range(count):
                if pos >= len(path):
                    break
                r, c = path[pos]
                out[r, c] = color
                pos += 1
    return out


def _can_place(rect, rects, rows, cols):
    top, left, bottom, right = rect
    if top < 1 or left < 1 or bottom >= rows - 1 or right >= cols - 1:
        return False
    for ot, ol, ob, or_ in rects:
        if not (bottom + 1 < ot or ob + 1 < top or right + 1 < ol or or_ + 1 < left):
            return False
    return True


def _draw_rect(grid, rect):
    top, left, bottom, right = rect
    grid[top, left:right + 1] = MAROON
    grid[bottom, left:right + 1] = MAROON
    grid[top:bottom + 1, left] = MAROON
    grid[top:bottom + 1, right] = MAROON


def _random_rect(rows, cols):
    height = random.randint(5, min(11, rows - 2))
    width = random.randint(5, min(13, cols - 2))
    top = random.randint(1, rows - height - 1)
    left = random.randint(1, cols - width - 1)
    return top, left, top + height - 1, left + width - 1


def _scatter_inside(grid, rect):
    top, left, bottom, right = rect
    interior = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    path_capacity = len(_inner_wall_path(top, left, bottom, right))
    color_count = random.randint(2, 4)
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8], color_count)

    for _ in range(100):
        counts = [random.randint(1, 5) for _ in range(color_count)]
        if sum(counts) <= path_capacity and sum(counts) <= len(interior):
            break
    else:
        counts = [1] * color_count

    cells = random.sample(interior, sum(counts))
    random.shuffle(cells)
    idx = 0
    for color, count in zip(palette, counts):
        for _ in range(count):
            r, c = cells[idx]
            grid[r, c] = color
            idx += 1


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    target_rects = random.randint(1, 4)
    rects = []

    for _ in range(400):
        if len(rects) >= target_rects:
            break
        rect = _random_rect(rows, cols)
        if _can_place(rect, rects, rows, cols):
            rects.append(rect)

    if not rects:
        rects.append((1, 1, min(rows - 2, 8), min(cols - 2, 9)))

    for rect in rects:
        _draw_rect(grid, rect)
        _scatter_inside(grid, rect)

    return grid

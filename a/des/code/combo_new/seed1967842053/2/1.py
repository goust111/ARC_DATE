import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.nonzero(mask)):
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(int(sr), int(sc))])
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _legend(grid):
    h, w = grid.shape
    color_for_count = {}
    dot_cells = set()
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color in (BLACK, GRAY):
                continue
            if not np.all(grid[r : r + 2, c : c + 2] == color):
                continue
            best_len = 0
            best_row = None
            for rr in (r, r + 1):
                k = 0
                cc = c - 1
                while cc >= 0 and int(grid[rr, cc]) == GRAY:
                    k += 1
                    cc -= 1
                if k > best_len:
                    best_len = k
                    best_row = rr
            if best_len:
                color_for_count[best_len] = color
                for cc in range(c - best_len, c):
                    dot_cells.add((best_row, cc))
    return color_for_count, dot_cells


def _hole_count(obj):
    h, w = obj.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for sr in range(h):
        for sc in range(w):
            if obj[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            touches_edge = False
            while q:
                r, c = q.popleft()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not obj[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _tip_direction(obj):
    counts = {
        "up": int(obj[0, :].sum()),
        "down": int(obj[-1, :].sum()),
        "left": int(obj[:, 0].sum()),
        "right": int(obj[:, -1].sum()),
    }
    one_pixel_sides = [side for side, count in counts.items() if count == 1]
    if len(one_pixel_sides) == 1:
        return one_pixel_sides[0]
    order = {"up": 0, "down": 1, "left": 2, "right": 3}
    return min(counts, key=lambda side: (counts[side], order[side]))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each holed gray object from the dot legend after flipping it by tip orientation.

    Input structure:
    The grid has a black background.  The right side contains legend entries made from a
    2x2 non-gray color swatch with a horizontal run of gray dot pixels immediately to its
    left; the dot count is the key for that swatch color.  The left side contains separate
    4-connected gray foreground objects, each with a distinct number of enclosed black
    holes and one side of its bounding box touched by exactly one tip pixel.

    Rule:
    For every non-legend gray object, count its enclosed black holes K, look up the legend
    color whose gray dot run also has length K, and determine the object's tip direction.
    Objects with left or right tips are mirrored horizontally inside their own bounding box;
    objects with up or down tips are mirrored vertically.  The mirrored object pixels are
    drawn in the looked-up color, old gray pixels are cleared to black, and the legend stays
    unchanged.

    Key insight:
    The color is not attached to the object directly: hole count selects a legend entry,
    while the unique bounding-box tip independently selects the flip axis.
    """
    out = np.array(grid, copy=True)
    color_for_count, legend_dots = _legend(grid)

    gray_mask = grid == GRAY
    for r, c in legend_dots:
        gray_mask[r, c] = False

    for comp in _components(gray_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        obj = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            obj[r - r0, c - c0] = True

        holes = _hole_count(obj)
        if holes not in color_for_count:
            continue
        color = color_for_count[holes]
        direction = _tip_direction(obj)
        if direction in ("left", "right"):
            flipped = obj[:, ::-1]
        else:
            flipped = obj[::-1, :]

        for r, c in comp:
            out[r, c] = BLACK
        fr, fc = np.nonzero(flipped)
        out[r0 + fr, c0 + fc] = color

    return out


def _make_object(hole_count, direction):
    body_h = random.randint(5, 7)
    body_w = random.randint(5, 7)
    if direction in ("up", "down"):
        mask = np.zeros((body_h + 1, body_w), dtype=bool)
        row_offset = 1 if direction == "up" else 0
        mask[row_offset : row_offset + body_h, :] = True
        tip_col = random.randint(1, body_w - 2)
        tip_row = 0 if direction == "up" else body_h
        mask[tip_row, tip_col] = True
        body_origin = (row_offset, 0)
    else:
        mask = np.zeros((body_h, body_w + 1), dtype=bool)
        col_offset = 1 if direction == "left" else 0
        mask[:, col_offset : col_offset + body_w] = True
        tip_row = random.randint(1, body_h - 2)
        tip_col = 0 if direction == "left" else body_w
        mask[tip_row, tip_col] = True
        body_origin = (0, col_offset)

    candidates = [(r, c) for r in range(1, body_h - 1) for c in range(1, body_w - 1)]
    random.shuffle(candidates)
    holes = []
    for r, c in candidates:
        if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in holes):
            holes.append((r, c))
            if len(holes) == hole_count:
                break
    if len(holes) != hole_count:
        return _make_object(hole_count, direction)

    br, bc = body_origin
    for r, c in holes:
        mask[br + r, bc + c] = False
    return mask


def _can_place(occupied, mask, r0, c0):
    h, w = mask.shape
    r_start = max(0, r0 - 1)
    r_end = min(occupied.shape[0], r0 + h + 1)
    c_start = max(0, c0 - 1)
    c_end = min(occupied.shape[1], c0 + w + 1)
    return not occupied[r_start:r_end, c_start:c_end].any()


def generate_input() -> np.ndarray:
    n = random.randint(2, 4)
    height = random.randint(max(22, 3 * n + 7), 30)
    width = random.randint(max(25, n + 21), 30)
    grid = np.zeros((height, width), dtype=int)

    block_col = width - 2
    legend_left = block_col - n
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    counts = list(range(1, n + 1))
    random.shuffle(counts)
    start_row = random.randint(0, height - (3 * n - 1))
    color_for_count = dict(zip(counts, colors))
    for i, count in enumerate(counts):
        row = start_row + i * 3
        grid[row, block_col - count : block_col] = GRAY
        grid[row : row + 2, block_col : block_col + 2] = color_for_count[count]

    occupied = np.zeros_like(grid, dtype=bool)
    left_width = legend_left - 2
    for holes in range(1, n + 1):
        placed = False
        for _ in range(500):
            direction = random.choice(["up", "down", "left", "right"])
            obj = _make_object(holes, direction)
            oh, ow = obj.shape
            if oh > height or ow > left_width:
                continue
            r0 = random.randint(0, height - oh)
            c0 = random.randint(0, left_width - ow)
            if _can_place(occupied, obj, r0, c0):
                grid[r0 : r0 + oh, c0 : c0 + ow][obj] = GRAY
                occupied[r0 : r0 + oh, c0 : c0 + ow][obj] = True
                placed = True
                break
        if not placed:
            return generate_input()

    return grid

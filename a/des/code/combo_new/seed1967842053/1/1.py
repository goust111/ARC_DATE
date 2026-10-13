import random
from collections import deque

import numpy as np


def _component_cells(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield cells


def _count_enclosed_black(grid, cells, main_start):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = max(min(rows) - 1, main_start)
    r1 = min(max(rows) + 2, grid.shape[0])
    c0 = max(min(cols) - 1, 0)
    c1 = min(max(cols) + 2, grid.shape[1])

    black = grid[r0:r1, c0:c1] == 0
    h, w = black.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    seen = outside.copy()
    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            holes += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each closed foreground region by using its hole count as a palette index.

    Input structure:
    The background is black.  The upper palette row contains one-pixel color swatches
    separated by black cells; the first swatch means one hole, the second means two
    holes, and so on.  After one all-black separator row, the main area contains
    disconnected foreground rings or wall shapes, each enclosing one or more black
    connected holes.

    Rule:
    Keep the palette row and separator row unchanged.  For every connected non-black
    shape in the main area, count the black connected components that are enclosed by
    that shape.  If the count is N, recolor all pixels of that foreground shape with
    the Nth non-black palette swatch, while leaving enclosed black holes black.

    Key insight:
    The number of enclosed black holes is the visible attribute that selects a color
    by left-to-right position in the palette.
    """
    out = grid.copy()
    rows_with_color = np.where(np.any(grid != 0, axis=1))[0]
    if len(rows_with_color) == 0:
        return out

    palette_row = int(rows_with_color[0])
    palette = [int(v) for v in grid[palette_row] if v != 0]
    main_start = palette_row + 2
    if main_start >= grid.shape[0] or not palette:
        return out

    main_mask = grid[main_start:] != 0
    for rel_cells in _component_cells(main_mask):
        cells = [(r + main_start, c) for r, c in rel_cells]
        holes = _count_enclosed_black(grid, cells, main_start)
        if 1 <= holes <= len(palette):
            color = palette[holes - 1]
            for r, c in cells:
                out[r, c] = color
    return out


def _make_shape(hole_count, color):
    layouts = {
        1: [(1, 1)],
        2: [(1, 2), (2, 1)],
        3: [(1, 3), (3, 1)],
        4: [(1, 4), (4, 1), (2, 2)],
    }
    hole_rows, hole_cols = random.choice(layouts[hole_count])
    hole_h = random.randint(1, 2)
    hole_w = random.randint(1, 2)
    if hole_rows * hole_cols >= 4 and random.random() < 0.5:
        hole_h = hole_w = 1

    h = hole_rows * hole_h + hole_rows + 1
    w = hole_cols * hole_w + hole_cols + 1
    shape = np.full((h, w), color, dtype=int)
    for rr in range(hole_rows):
        for cc in range(hole_cols):
            r0 = 1 + rr * (hole_h + 1)
            c0 = 1 + cc * (hole_w + 1)
            shape[r0:r0 + hole_h, c0:c0 + hole_w] = 0
    return shape


def _can_place(grid, shape, r, c, main_start):
    h, w = shape.shape
    if r < main_start + 1 or c < 1 or r + h >= grid.shape[0] or c + w >= grid.shape[1]:
        return False
    r0 = max(main_start, r - 1)
    r1 = min(grid.shape[0], r + h + 1)
    c0 = max(0, c - 1)
    c1 = min(grid.shape[1], c + w + 1)
    return np.all(grid[r0:r1, c0:c1] == 0)


def generate_input() -> np.ndarray:
    for _ in range(200):
        height = random.randint(15, 30)
        width = random.randint(15, 30)
        grid = np.zeros((height, width), dtype=int)

        palette_len = random.randint(3, 4)
        colors = list(range(1, 10))
        random.shuffle(colors)
        palette = colors[:palette_len]
        foreground = colors[palette_len]

        max_start = width - (2 * palette_len - 1)
        if max_start < 0:
            continue
        palette_start = random.randint(0, max_start)
        for i, color in enumerate(palette):
            grid[0, palette_start + 2 * i] = color

        counts = list(range(1, palette_len + 1))
        extra = random.randint(0, 2)
        counts.extend(random.randint(1, palette_len) for _ in range(extra))
        random.shuffle(counts)

        ok = True
        for count in counts:
            shape = _make_shape(count, foreground)
            sh, sw = shape.shape
            placed = False
            for _try in range(100):
                if height - sh - 1 <= 3 or width - sw - 1 <= 1:
                    break
                r = random.randint(3, height - sh - 1)
                c = random.randint(1, width - sw - 1)
                if _can_place(grid, shape, r, c, 2):
                    grid[r:r + sh, c:c + sw] = shape
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok:
            return grid

    height, width = 20, 25
    grid = np.zeros((height, width), dtype=int)
    palette = [2, 3, 4]
    foreground = 5
    for i, color in enumerate(palette):
        grid[0, 2 + 2 * i] = color
    placements = [(3, 2, 1), (3, 8, 2), (10, 2, 3)]
    for r, c, count in placements:
        shape = _make_shape(count, foreground)
        grid[r:r + shape.shape[0], c:c + shape.shape[1]] = shape
    return grid

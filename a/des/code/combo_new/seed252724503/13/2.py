import random

import numpy as np

BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered pixels into a rectangle and stack the rectangles by area.

    Input structure:
    - Black background with several foreground colors scattered as disconnected fragments.
    - For each color, the total number of pixels equals the area of one rectangle.
    - For each color, the rectangle width is the span of that color's occupied columns.

    Rule:
    - For every foreground color, compute its area A and its column span width W, then derive height H = A / W.
    - Replace the scattered pixels with a solid W-by-H rectangle in the same column span.
    - Stack those rectangles from the bottom upward in ascending area order with no gaps.

    Key insight:
    - The fragments themselves are noise; the rectangle is determined only by each color's pixel count and horizontal span, and the final vertical order is determined by area.
    """
    out = np.zeros_like(grid)
    items = []
    for color in sorted(int(v) for v in np.unique(grid) if v != BLACK):
        rows, cols = np.where(grid == color)
        width = int(cols.max() - cols.min() + 1)
        area = int(len(rows))
        height = area // width
        items.append((area, int(rows.min()), int(cols.min()), int(cols.max()), height, color))
    items.sort()
    bottom = grid.shape[0]
    for _area, _rmin, c0, c1, height, color in items:
        out[bottom - height:bottom, c0:c1 + 1] = color
        bottom -= height
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(400):
        h = rng.randint(16, 24)
        w = rng.randint(16, 24)
        grid = np.zeros((h, w), dtype=int)
        n = rng.randint(2, 4)
        chosen = rng.sample(colors, n)
        specs = []
        used_areas = set()
        total_height = 0
        for color in chosen:
            for _ in range(80):
                rect_w = rng.randint(2, 5)
                rect_h = rng.randint(1, 4)
                area = rect_w * rect_h
                if area in used_areas:
                    continue
                src_h = rect_h + rng.randint(1, 3)
                if src_h * rect_w < area:
                    continue
                specs.append((color, rect_w, rect_h, src_h))
                used_areas.add(area)
                total_height += rect_h
                break
        if len(specs) != n or total_height > h - 2:
            continue
        placed = []
        ok = True
        for color, rect_w, rect_h, src_h in specs:
            for _ in range(100):
                r0 = rng.randint(1, h - src_h - 2)
                c0 = rng.randint(0, w - rect_w)
                if all(r0 + src_h < pr - 1 or pr + ph < r0 - 1 or c0 + rect_w < pc - 1 or pc + pw < c0 - 1 for pr, pc, ph, pw in placed):
                    break
            else:
                ok = False
                break
            cells = [(r, c) for r in range(r0, r0 + src_h) for c in range(c0, c0 + rect_w)]
            rng.shuffle(cells)
            chosen_cells = cells[: rect_w * rect_h]
            touched_cols = {c for _, c in chosen_cells}
            missing = [c for c in range(c0, c0 + rect_w) if c not in touched_cols]
            pool = [cell for cell in cells if cell not in chosen_cells]
            for col in missing:
                repl = next(cell for cell in pool if cell[1] == col)
                chosen_cells.pop()
                chosen_cells.append(repl)
                pool.remove(repl)
            for r, c in chosen_cells:
                grid[r, c] = color
            placed.append((r0, c0, src_h, rect_w))
        if ok:
            return grid
    raise RuntimeError("failed to generate task 13/2")

from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
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
            comps.append(cells)
    return comps


def _controls(grid):
    row_controls, col_controls = [], []
    h, w = grid.shape
    for comp in _components(grid == RED):
        if len(comp) != 3:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == r1:
            direction = "right" if c0 == 0 else "left" if c1 == w - 1 else None
            if direction:
                row_controls.append((r0, direction))
        elif c0 == c1:
            direction = "down" if r0 == 0 else "up" if r1 == h - 1 else None
            if direction:
                col_controls.append((c0, direction))
    return row_controls, col_controls


def main(grid: np.ndarray) -> np.ndarray:
    """Push colored pixels first along arrow-controlled rows and then along columns.

    Input structure: Red three-pixel edge arrows control either a row from the
    left/right edge or a column from the top/bottom edge.  Interior colored
    singleton pixels sit on a black background.
    Rule: Clear all red arrows.  Execute every row control, packing that row's
    non-black pixels left or right while preserving their order.  Then execute
    every column control on the intermediate grid, packing non-black pixels up or
    down.  Pixel colors are preserved.
    Key insight: Arrow location selects row versus column, arrow side selects the
    push direction, and the row phase happens before the column phase.
    """
    row_controls, col_controls = _controls(grid)
    out = grid.copy()
    out[out == RED] = BLACK
    for r, direction in row_controls:
        vals = [int(v) for v in out[r, :] if int(v) != BLACK]
        out[r, :] = BLACK
        if direction == "left":
            out[r, : len(vals)] = vals
        else:
            out[r, out.shape[1] - len(vals) :] = vals
    for c, direction in col_controls:
        vals = [int(v) for v in out[:, c] if int(v) != BLACK]
        out[:, c] = BLACK
        if direction == "up":
            out[: len(vals), c] = vals
        else:
            out[out.shape[0] - len(vals) :, c] = vals
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(12, 22), random.randint(12, 22)
    grid = np.zeros((h, w), dtype=int)
    row_controls = random.sample(range(2, h - 2), random.randint(1, 3))
    for r in row_controls:
        if random.random() < 0.5:
            grid[r, :3] = RED
        else:
            grid[r, w - 3 : w] = RED
    col_controls = random.sample(range(2, w - 2), random.randint(1, 3))
    for c in col_controls:
        if random.random() < 0.5:
            grid[:3, c] = RED
        else:
            grid[h - 3 : h, c] = RED
    colors = [1, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(random.randint(8, 18)):
        for _try in range(100):
            r, c = random.randint(3, h - 4), random.randint(3, w - 4)
            if grid[r, c] == BLACK:
                grid[r, c] = random.choice(colors)
                break
    return grid

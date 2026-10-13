import random

import numpy as np

BLACK = 0
GREY = 5


def _gap_direction(grid: np.ndarray):
    if np.any(grid[0, :] == BLACK):
        return "up"
    if np.any(grid[-1, :] == BLACK):
        return "down"
    if np.any(grid[:, 0] == BLACK):
        return "left"
    return "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed block toward the unique border gap, with stripe length equal to the seed area.

    Input structure:
    - A black grid is enclosed by a grey 1-pixel border with one short missing segment on exactly one side.
    - Inside the border are several small solid colored rectangles.
    - Seed blocks do not touch the border and do not touch each other.

    Rule:
    - The side containing the border gap determines both the growth axis and the growth direction.
    - For each seed rectangle, compute its area and extend a same-colored stripe from the side facing the gap.
    - The stripe keeps the seed's cross-section and grows for that many cells, clipped by the border.

    Key insight:
    - The border gap encodes global direction, while each seed's own area encodes how far its stripe should extend.
    """
    out = grid.copy()
    direction = _gap_direction(grid)
    colors = [int(v) for v in np.unique(grid) if v not in (BLACK, GREY)]
    for color in colors:
        rows, cols = np.where(grid == color)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        area = len(rows)
        if direction == "up":
            out[max(1, r0 - area):r0, c0:c1 + 1] = color
        elif direction == "down":
            out[r1 + 1:min(grid.shape[0] - 1, r1 + 1 + area), c0:c1 + 1] = color
        elif direction == "left":
            out[r0:r1 + 1, max(1, c0 - area):c0] = color
        else:
            out[r0:r1 + 1, c1 + 1:min(grid.shape[1] - 1, c1 + 1 + area)] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    sizes = [(2, 2), (2, 3), (3, 2)]
    for _ in range(400):
        h = rng.randint(14, 22)
        w = rng.randint(14, 22)
        grid = np.zeros((h, w), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = GREY
        side = rng.choice(["up", "down", "left", "right"])
        gap_len = rng.randint(1, 3)
        if side in ("up", "down"):
            start = rng.randint(1, w - gap_len - 1)
            row = 0 if side == "up" else h - 1
            grid[row, start:start + gap_len] = BLACK
        else:
            start = rng.randint(1, h - gap_len - 1)
            col = 0 if side == "left" else w - 1
            grid[start:start + gap_len, col] = BLACK
        occ = set()
        ok = True
        for color in rng.sample(colors, rng.randint(2, 4)):
            sh, sw = rng.choice(sizes)
            for _ in range(100):
                r0 = rng.randint(2, h - sh - 3)
                c0 = rng.randint(2, w - sw - 3)
                cells = {(r, c) for r in range(r0, r0 + sh) for c in range(c0, c0 + sw)}
                if any(cell in occ for cell in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = color
                occ |= cells
                break
            else:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate task 14/1")

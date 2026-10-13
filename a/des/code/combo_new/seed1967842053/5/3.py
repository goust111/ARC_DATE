import random
from collections import Counter

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Map each L shape's opening direction to the matching output quadrant.

    Input structure:
    The input is a black-background grid containing exactly four separated,
    same-size L-shaped objects.  Each object has a distinct non-background
    color and is made from one horizontal arm and one vertical arm that meet at
    a corner cell.

    Rule:
    For each colored L, locate its corner, infer the diagonal direction where
    the missing square of its bounding box lies, and fill the output quadrant
    for that direction with the L's color.  The output is a 2 by 2 arrangement
    of solid square blocks whose side length equals the L arm length.

    Key insight:
    The row-spanning arm and column-spanning arm identify the corner; the
    corner's position inside the L's square bounding box reveals the hidden
    diagonal direction.
    """
    arr = np.asarray(grid)
    flat = arr.ravel()
    background = Counter(flat.tolist()).most_common(1)[0][0]

    objects = []
    for color in np.unique(arr):
        if color == background:
            continue
        coords = np.argwhere(arr == color)
        if coords.size:
            objects.append((int(color), coords))

    if not objects:
        return np.zeros((0, 0), dtype=arr.dtype)

    arm_len = None
    placements = []
    for color, coords in objects:
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        side = max(height, width)
        arm_len = side if arm_len is None else arm_len

        cells = {(int(r), int(c)) for r, c in coords}
        full_rows = [
            r for r in range(r0, r1 + 1)
            if all((r, c) in cells for c in range(c0, c1 + 1))
        ]
        full_cols = [
            c for c in range(c0, c1 + 1)
            if all((r, c) in cells for r in range(r0, r1 + 1))
        ]

        if len(full_rows) != 1 or len(full_cols) != 1:
            raise ValueError("Each colored object must be a clean L shape.")

        corner_r = full_rows[0]
        corner_c = full_cols[0]
        out_r = 1 if corner_r == r0 else 0
        out_c = 1 if corner_c == c0 else 0
        placements.append((out_r, out_c, color))

    out = np.zeros((2 * arm_len, 2 * arm_len), dtype=arr.dtype)
    for out_r, out_c, color in placements:
        rs = slice(out_r * arm_len, (out_r + 1) * arm_len)
        cs = slice(out_c * arm_len, (out_c + 1) * arm_len)
        out[rs, cs] = color
    return out


def _l_cells(top: int, left: int, side: int, down: bool, right: bool):
    corner_r = top if down else top + side - 1
    corner_c = left if right else left + side - 1
    cells = []
    for c in range(left, left + side):
        cells.append((corner_r, c))
    for r in range(top, top + side):
        cells.append((r, corner_c))
    return cells


def generate_input() -> np.ndarray:
    side = random.randint(2, 7)
    min_dim = max(12, 2 * side + 6)
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)
    grid = np.zeros((height, width), dtype=int)

    mid_r = height // 2
    mid_c = width // 2
    regions = [
        (1, mid_r - side - 1, 1, mid_c - side - 1),
        (1, mid_r - side - 1, mid_c + 1, width - side - 1),
        (mid_r + 1, height - side - 1, 1, mid_c - side - 1),
        (mid_r + 1, height - side - 1, mid_c + 1, width - side - 1),
    ]

    directions = [(True, True), (True, False), (False, True), (False, False)]
    random.shuffle(directions)
    colors = random.sample(range(1, 10), 4)

    for (r_min, r_max, c_min, c_max), (down, right), color in zip(regions, directions, colors):
        top = random.randint(r_min, r_max)
        left = random.randint(c_min, c_max)
        for r, c in _l_cells(top, left, side, down, right):
            grid[r, c] = color

    return grid

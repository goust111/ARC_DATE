import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4


def main(grid: np.ndarray) -> np.ndarray:
    """Combine two damaged copies of one master shape and color by coverage count.

    Input structure:
    The input has a black background and one full-height red vertical separator.
    The separator divides the grid into equal-size left and right panels. Each
    panel contains a damaged copy of the same master shape, drawn in one
    non-black, non-red color per panel.

    Rule:
    Remove the red separator and compare corresponding cells of the two panels.
    If both panels have a colored cell, output yellow. If only the left panel has
    a colored cell, output the left panel color. If only the right panel has a
    colored cell, output the right panel color. If neither panel has a colored
    cell, output black.

    Key insight:
    The two panels must first be aligned as complementary incomplete versions of
    one hidden master shape; the final color is then determined independently by
    how many panels cover each master-shape position.
    """
    rows, cols = grid.shape
    mid = cols // 2

    if cols % 2 == 0 or not np.all(grid[:, mid] == RED):
        candidates = [
            c
            for c in range(1, cols - 1)
            if c == cols - c - 1 and np.all(grid[:, c] == RED)
        ]
        if not candidates:
            raise ValueError("expected one centered full-height red separator")
        mid = candidates[0]

    left = grid[:, :mid]
    right = grid[:, mid + 1 :]
    left_mask = left != BLACK
    right_mask = right != BLACK

    out = np.zeros_like(left)
    out[left_mask & right_mask] = YELLOW
    out[left_mask & ~right_mask] = left[left_mask & ~right_mask]
    out[right_mask & ~left_mask] = right[right_mask & ~left_mask]
    return out


def _draw_line(mask, r0, c0, r1, c1):
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    r, c = r0, c0
    mask[r, c] = True
    while (r, c) != (r1, c1):
        if r != r1:
            r += dr
        if c != c1:
            c += dc
        mask[r, c] = True


def _make_master_mask(height, width):
    mask = np.zeros((height, width), dtype=bool)
    kind = random.randrange(5)
    top = random.randint(1, max(1, height // 3))
    left = random.randint(1, max(1, width // 3))
    bottom = random.randint(max(top + 2, height * 2 // 3), height - 2)
    right = random.randint(max(left + 2, width * 2 // 3), width - 2)

    if kind == 0:
        mask[top : bottom + 1, left] = True
        mask[top : bottom + 1, right] = True
        mask[top, left : right + 1] = True
        mask[bottom, left : right + 1] = True
        if random.random() < 0.7 and bottom - top > 3:
            r = random.randint(top + 1, bottom - 1)
            mask[r, left : right + 1] = True
        if random.random() < 0.7 and right - left > 3:
            c = random.randint(left + 1, right - 1)
            mask[top : bottom + 1, c] = True

    elif kind == 1:
        cr = random.randint(top + 1, bottom - 1)
        cc = random.randint(left + 1, right - 1)
        mask[top : bottom + 1, cc] = True
        mask[cr, left : right + 1] = True
        arm = random.randint(1, 2)
        if arm == 2:
            c0 = max(left, cc - 1)
            c1 = min(right, cc + 1)
            r0 = max(top, cr - 1)
            r1 = min(bottom, cr + 1)
            mask[top : bottom + 1, c0 : c1 + 1] = True
            mask[r0 : r1 + 1, left : right + 1] = True
        if random.random() < 0.5:
            _draw_line(mask, top, left, bottom, right)
        else:
            _draw_line(mask, top, right, bottom, left)

    elif kind == 2:
        r, c = random.randint(top, bottom), random.randint(left, right)
        mask[r, c] = True
        steps = random.randint(12, 24)
        for _ in range(steps):
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            r = min(bottom, max(top, r + dr))
            c = min(right, max(left, c + dc))
            mask[r, c] = True
            if random.random() < 0.35:
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if top <= nr <= bottom and left <= nc <= right:
                        mask[nr, nc] = True

    elif kind == 3:
        mask[top : bottom + 1, left : right + 1] = True
        holes = random.randint(2, 5)
        for _ in range(holes):
            hr = random.randint(top + 1, bottom - 1)
            hc = random.randint(left + 1, right - 1)
            mask[hr, hc] = False
            if random.random() < 0.45 and hr + 1 < bottom:
                mask[hr + 1, hc] = False
            if random.random() < 0.45 and hc + 1 < right:
                mask[hr, hc + 1] = False

    else:
        blocks = random.randint(2, 4)
        for _ in range(blocks):
            bh = random.randint(2, min(4, bottom - top + 1))
            bw = random.randint(2, min(4, right - left + 1))
            r0 = random.randint(top, bottom - bh + 1)
            c0 = random.randint(left, right - bw + 1)
            mask[r0 : r0 + bh, c0 : c0 + bw] = True
        for _ in range(random.randint(1, 3)):
            r0 = random.randint(top, bottom)
            c0 = random.randint(left, right)
            r1 = random.randint(top, bottom)
            c1 = random.randint(left, right)
            _draw_line(mask, r0, c0, r1, c1)

    if mask.sum() < 9:
        cr = height // 2
        cc = width // 2
        mask[max(1, cr - 2) : min(height - 1, cr + 3), cc] = True
        mask[cr, max(1, cc - 2) : min(width - 1, cc + 3)] = True
    return mask


def _split_master(master):
    positions = list(zip(*np.where(master)))
    random.shuffle(positions)
    n = len(positions)
    states = [0, 1, 2] * (n // 3) + [random.randrange(3) for _ in range(n % 3)]
    random.shuffle(states)

    left_mask = np.zeros(master.shape, dtype=bool)
    right_mask = np.zeros(master.shape, dtype=bool)
    for (r, c), state in zip(positions, states):
        if state == 0:
            left_mask[r, c] = True
        elif state == 1:
            right_mask[r, c] = True
        else:
            left_mask[r, c] = True
            right_mask[r, c] = True
    return left_mask, right_mask


def generate_input() -> np.ndarray:
    height = random.randint(7, 15)
    width = random.randint(7, 14)
    colors = random.sample([1, 3, 5, 6, 7, 8, 9], 2)

    master = _make_master_mask(height, width)
    left_mask, right_mask = _split_master(master)

    grid = np.zeros((height, width * 2 + 1), dtype=int)
    grid[:, width] = RED
    grid[:, :width][left_mask] = colors[0]
    grid[:, width + 1 :][right_mask] = colors[1]
    return grid

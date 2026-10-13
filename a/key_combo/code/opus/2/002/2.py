import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    sep = max(range(w), key=lambda c: np.count_nonzero(grid[:, c] == 5))
    out = np.zeros_like(grid)
    out[:, sep] = 5

    blue = np.argwhere(grid == 1)
    red = np.argwhere(grid == 2)
    if blue.size:
        shift = (sep - 1) - int(blue[:, 1].max())
        for r, c in blue:
            out[r, c + shift] = 1
    if red.size:
        shift = (sep + 1) - int(red[:, 1].min())
        for r, c in red:
            out[r, c + shift] = 2

    left_rows = {int(r) for r in blue[:, 0]} if blue.size else set()
    right_rows = {int(r) for r in red[:, 0]} if red.size else set()
    for r in range(h):
        l = r in left_rows
        rr = r in right_rows
        if l and not rr:
            out[r, sep:] = 1
        elif rr and not l:
            out[r, : sep + 1] = 2
    return out


def _row_profile(height):
    rows = list(range(height))
    common = set(random.sample(rows, random.randint(1, height - 1)))
    left_only = set(random.sample([r for r in rows if r not in common], 1))
    remaining = [r for r in rows if r not in common and r not in left_only]
    right_only = set(random.sample(remaining or rows, 1))
    left = sorted(common | left_only)
    right = sorted(common | right_only)
    return left, right


def _make_component(row_offsets, width):
    coords = []
    for rr in row_offsets:
        c = random.randrange(width)
        coords.append((rr, c))
        if random.random() < 0.65 and c + 1 < width:
            coords.append((rr, c + 1))
    # Add a vertical spine to keep the component 4-connected.
    spine_c = width // 2
    for rr in range(min(row_offsets), max(row_offsets) + 1):
        coords.append((rr, spine_c))
    return sorted(set(coords))


def generate_input() -> np.ndarray:
    h = random.randint(11, 16)
    w = random.randint(17, 22)
    sep = random.randint(8, w - 9)
    grid = np.zeros((h, w), dtype=int)
    grid[:, sep] = 5

    span = random.randint(4, min(7, h - 2))
    top = random.randint(1, h - span - 1)
    left_rows, right_rows = _row_profile(span)
    lw = random.randint(2, 4)
    rw = random.randint(2, 4)
    left_shape = _make_component(left_rows, lw)
    right_shape = _make_component(right_rows, rw)

    l0 = random.randint(1, sep - lw - 2)
    r0 = random.randint(sep + 2, w - rw - 1)
    for rr, cc in left_shape:
        grid[top + rr, l0 + cc] = 1
    for rr, cc in right_shape:
        grid[top + rr, r0 + cc] = 2
    return grid

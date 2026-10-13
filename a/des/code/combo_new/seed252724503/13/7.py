import random

import numpy as np

BLACK = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full bottom-layer rectangle using the bottom legend's layer order.

    Input structure:
    - Black background with several overlapping solid colored rectangles.
    - A bottom legend row lists layer colors from left to right, separated by black cells.
    - The leftmost legend color is the bottom layer, whose visible pixels may be only partial edges.

    Rule:
    - Read the bottom legend and take its first listed color as the bottom-layer rectangle color.
    - Ignore the legend row itself and find all visible pixels of that color in the scene.
    - Output a black grid of the same size containing only the solid bounding rectangle spanned by those visible pixels.

    Key insight:
    - Even partial exposed edges are enough to reconstruct the hidden rectangle because the full rectangle is exactly the bounding box of its remaining visible pixels.
    """
    legend_cols = [c for c in range(grid.shape[1]) if grid[-1, c] != BLACK]
    bottom_color = int(grid[-1, legend_cols[0]])
    rows, cols = np.where(grid[:-1, :] == bottom_color)
    out = np.zeros_like(grid)
    if len(rows):
        out[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = bottom_color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 5, 6, 7, 8]
    for _ in range(400):
        h = rng.randint(14, 22)
        w = rng.randint(16, 24)
        n = rng.randint(2, 4)
        layer_colors = rng.sample(colors, n)
        grid = np.zeros((h, w), dtype=int)
        rects = []
        ok = True
        for idx, color in enumerate(layer_colors):
            for _ in range(100):
                rh = rng.randint(3, h - 4)
                rw = rng.randint(3, w - 4)
                r0 = rng.randint(0, h - 1 - rh - 1)
                c0 = rng.randint(0, w - rw)
                rect = (r0, c0, rh, rw, color)
                rects.append(rect)
                break
            else:
                ok = False
        if not ok:
            continue
        bottom = rects[0]
        visible = set((r, c) for r in range(bottom[0], bottom[0] + bottom[2]) for c in range(bottom[1], bottom[1] + bottom[3]))
        for r0, c0, rh, rw, color in rects:
            grid[r0:r0 + rh, c0:c0 + rw] = color
        for r0, c0, rh, rw, _color in rects[1:]:
            visible -= {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}
        if not visible:
            continue
        vr = [r for r, _ in visible]
        vc = [c for _, c in visible]
        if min(vr) == max(vr) or min(vc) == max(vc):
            continue
        grid[-1, :] = BLACK
        legend_positions = list(range(1, 2 * n, 2))
        if legend_positions[-1] >= w:
            continue
        for pos, color in zip(legend_positions, layer_colors):
            grid[-1, pos] = color
        return grid
    raise RuntimeError("failed to generate task 13/7")

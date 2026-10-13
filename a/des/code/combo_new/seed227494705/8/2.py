import random

import numpy as np


BG = 0


def _runs(vals):
    runs = []
    if not vals:
        return runs
    vals = sorted(vals)
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
        else:
            runs.append((start, prev))
            start = prev = v
    runs.append((start, prev))
    return runs


def _first_color_order(g):
    order = []
    seen = set()
    for v in g.ravel():
        color = int(v)
        if color != BG and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill edge-defined rectangles selected by same-color interior seed pixels.

    Input structure:
    - Black background with colored wall segments on horizontal and vertical grid edges.
    - Each active color has one interior seed, one horizontal edge segment, and one vertical edge segment.
    - The seed row aligns with the vertical segment and the seed column aligns with the horizontal segment.

    Rule:
    - For each color in first row-major appearance order, read its edge wall segments.
    - The vertical segment supplies the output row span and the horizontal segment supplies the output column span.
    - Fill that whole rectangle with the seed color, allowing later colors to overwrite earlier fills.

    Key insight:
    - The seed is a row-column locator that links two edge wall segments into one rectangle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    for color in _first_color_order(g):
        edge = np.argwhere(g == color)
        top_cols = [int(c) for r, c in edge if int(r) == 0]
        bottom_cols = [int(c) for r, c in edge if int(r) == h - 1]
        left_rows = [int(r) for r, c in edge if int(c) == 0]
        right_rows = [int(r) for r, c in edge if int(c) == w - 1]
        h_runs = _runs(top_cols) + _runs(bottom_cols)
        v_runs = _runs(left_rows) + _runs(right_rows)
        if not h_runs or not v_runs:
            continue
        c0, c1 = max(h_runs, key=lambda x: x[1] - x[0])
        r0, r1 = max(v_runs, key=lambda x: x[1] - x[0])
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    g = np.zeros((h, w), dtype=int)
    n = rng.randint(2, 5)
    for color in rng.sample(colors, n):
        h_side = rng.choice(["top", "bottom"])
        v_side = rng.choice(["left", "right"])
        if h_side == "top":
            r0 = 0
            r1 = rng.randint(3, h - 2)
        else:
            r0 = rng.randint(1, h - 4)
            r1 = h - 1
        if v_side == "left":
            c0 = 0
            c1 = rng.randint(3, w - 2)
        else:
            c0 = rng.randint(1, w - 4)
            c1 = w - 1
        if h_side == "top":
            g[0, c0 : c1 + 1] = color
        else:
            g[h - 1, c0 : c1 + 1] = color
        if v_side == "left":
            g[r0 : r1 + 1, 0] = color
        else:
            g[r0 : r1 + 1, w - 1] = color
        sr = rng.randint(max(1, r0), min(h - 2, r1))
        sc = rng.randint(max(1, c0), min(w - 2, c1))
        g[sr, sc] = color
    return g

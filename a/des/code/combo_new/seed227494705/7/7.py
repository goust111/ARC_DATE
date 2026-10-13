import random

import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize each row's left-right mismatch as a centered color pair.

    Input structure:
    - Two full-height grey vertical separators split the grid into left, middle, and right regions.
    - The left and right regions have equal width.
    - In every row, the two side regions differ at exactly one corresponding column.

    Rule:
    - Compare the left and right regions row by row.
    - For each row, take the left color and right color at the unique mismatch.
    - Write those two colors centered in the middle region; keep side regions and separators unchanged.

    Key insight:
    - The middle area records the two disagreeing colors, not the mismatch position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    if len(sep_cols) < 2:
        return out
    s1, s2 = sep_cols[0], sep_cols[1]
    left = g[:, :s1]
    right = g[:, s2 + 1 :]
    mid_w = s2 - s1 - 1
    start = s1 + 1 + (mid_w - 2) // 2
    out[:, s1 + 1 : s2] = BG
    width = min(left.shape[1], right.shape[1])
    for r in range(h):
        diffs = [c for c in range(width) if int(left[r, c]) != int(right[r, c])]
        if len(diffs) != 1:
            continue
        c = diffs[0]
        out[r, start] = int(left[r, c])
        out[r, start + 1] = int(right[r, c])
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    h = rng.randint(5, 14)
    side_w = rng.randint(3, 8)
    mid_w = rng.randint(2, 6)
    w = side_w * 2 + mid_w + 2
    g = np.zeros((h, w), dtype=int)
    s1 = side_w
    s2 = side_w + mid_w + 1
    g[:, s1] = GREY
    g[:, s2] = GREY
    for r in range(h):
        row = [rng.choice(colors) for _ in range(side_w)]
        other = list(row)
        c = rng.randrange(side_w)
        a = rng.choice(colors)
        b = rng.choice([x for x in colors if x != a])
        row[c] = a
        other[c] = b
        g[r, :side_w] = row
        g[r, s2 + 1 :] = other
    return g

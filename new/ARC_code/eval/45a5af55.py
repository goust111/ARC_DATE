from __future__ import annotations
import numpy as np

def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Input structure:
    - Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
    - Each stripe has uniform color and variable height (1-4 rows).

    Rule:
    - Read stripes bottom to top.
    - Bottom stripe becomes the center solid square with side length = stripe height.
    - Each subsequent stripe wraps the current shape with a square frame of thickness = stripe height, in that stripe's color.

    Key insight: Stripe height doubles as both frame thickness and (for the innermost) center-square side length; output dimensions are fully determined by the stripe heights.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    # Parse stripe runs from bottom to top.
    runs: list[tuple[int, int]] = []  # (color, height), bottom -> top
    r = int(H - 1)
    while r >= 0:
        col = int(g[int(r), 0])
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[int(rr)] == int(col)).all()):
            h += 1
            rr -= 1
        runs.append((int(col), int(h)))
        r = int(rr)

    assert not (not runs)

    c0, h0 = runs[0]
    cur = np.full((int(h0), int(h0)), int(c0), dtype=int)

    for col, h in runs[1:]:
        h = int(h)
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    W = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None
    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c
    stripes = [np.full((h, W), col, dtype=int) for col, h in reversed(list(zip(cols, heights)))]
    return np.vstack(stripes)

def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")



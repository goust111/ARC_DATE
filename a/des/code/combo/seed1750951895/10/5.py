from __future__ import annotations

import numpy as np


def _parse_stripes(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w > 0, "grid must be non-empty"

    for r in range(h):
        assert bool((g[r] == g[r, 0]).all()), "each row must be monochrome"

    stripes: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        height = 1
        r2 = r - 1
        while r2 >= 0 and int(g[r2, 0]) == color:
            height += 1
            r2 -= 1
        stripes.append((color, height))
        r = r2

    assert stripes, "expected at least one stripe"
    colors = [color for color, _height in stripes]
    heights = [height for _color, height in stripes]
    assert len(colors) == len(set(colors)), "stripe colors must be unique"
    assert len(heights) == len(set(heights)), "stripe heights must be unique"
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a stack of full-width horizontal stripes into nested squares.

    The bottom stripe becomes the center filled square with side length equal to
    its height. Moving upward, each stripe wraps the current shape with a square
    frame whose thickness equals that stripe's height.
    """
    stripes = _parse_stripes(grid)

    center_color, center_size = stripes[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for color, thickness in stripes[1:]:
        new_size = cur.shape[0] + 2 * thickness
        nxt = np.full((new_size, new_size), color, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    n_stripes = int(np.random.randint(2, 6))
    width = int(np.random.randint(4, 19))

    all_heights = np.arange(1, 8)
    heights = np.random.choice(all_heights, size=n_stripes, replace=False)
    np.random.shuffle(heights)
    heights = [int(v) for v in heights]

    out_size = heights[-1] + 2 * sum(heights[:-1])
    if out_size > 30:
        return None

    colors = np.random.choice(np.arange(10), size=n_stripes, replace=False)
    colors = [int(v) for v in colors]

    rows = []
    for color, height in zip(colors, heights):
        rows.append(np.full((height, width), color, dtype=int))
    grid = np.vstack(rows)

    # Re-parse to ensure the sampled layout matches the intended stripe list.
    parsed = _parse_stripes(grid)
    expected = list(reversed(list(zip(colors, heights))))
    assert parsed == [(int(color), int(height)) for color, height in expected]

    out = main(grid)
    assert out.shape == (out_size, out_size)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")

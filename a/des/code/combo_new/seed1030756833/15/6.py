from collections import Counter
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    """Return the dominant color in the grid."""
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _edge_sequence(grid: np.ndarray, bg: int) -> list[int]:
    """Read the left-anchored non-background sequence on the top row."""
    seq: list[int] = []
    for value in grid[0, :]:
        color = int(value)
        if color == int(bg):
            break
        seq.append(color)
    return seq


def _paint_ring(out: np.ndarray, layer: int, color: int) -> None:
    """Paint the cells whose distance to the grid edge is exactly layer."""
    h, w = out.shape
    r0, c0 = int(layer), int(layer)
    r1, c1 = int(h - 1 - layer), int(w - 1 - layer)
    if r0 > r1 or c0 > c1:
        return
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the top-left edge color code into nested recolored border layers.

    Input structure:
    - A rectangular grid is almost entirely one dominant background color.
    - Row 0 begins at column 0 with a short contiguous sequence of non-background
      colors, followed by the background color for the rest of that row.
    - The sequence length is small enough to fit as concentric edge-distance rings.

    Rule:
    - Read the row-0 left edge sequence as C1, C2, ..., CK.
    - Recolor all cells at distance 0 from the grid edge with C1.
    - Recolor all cells at distance 1 from the grid edge with C2, and so on.
    - After K rings are painted, every deeper cell remains the original background.

    Key insight:
    The short linear color code on the edge gives both the number of peel layers
    and the outward-to-inward color order for the two-dimensional nested borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "grid too small for border rings"

    bg = _background_color(g)
    colors = _edge_sequence(g, bg)
    assert colors, "top row must begin with a non-background color sequence"
    assert len(colors) < w, "the edge sequence must be followed by background"
    assert len(colors) <= (min(h, w) + 1) // 2, "not enough room for all encoded rings"

    out = np.full_like(g, int(bg))
    for layer, color in enumerate(colors):
        _paint_ring(out, int(layer), int(color))
    return out


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
HEIGHT_RANGE = (10, 30)
WIDTH_RANGE = (10, 30)
SEQUENCE_LENGTH_RANGE = (2, 5)
GENERATION_RETRIES = 300


def _expected_from_params(height: int, width: int, bg: int, colors: list[int]) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    for layer, color in enumerate(colors):
        _paint_ring(out, int(layer), int(color))
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(*SEQUENCE_LENGTH_RANGE)
    min_size = max(10, 2 * k + 1)
    height = random.randint(min_size, HEIGHT_RANGE[1])
    width = random.randint(max(min_size, k + 2), WIDTH_RANGE[1])

    bg = random.choice(COLOR_IDS)
    palette = [color for color in COLOR_IDS if color != bg]
    random.shuffle(palette)
    colors = palette[:k]

    grid = np.full((height, width), int(bg), dtype=int)
    grid[0, :k] = np.array(colors, dtype=int)

    expected = _expected_from_params(height, width, bg, colors)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-sequence ring input")

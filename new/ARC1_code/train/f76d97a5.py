from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the grey pixels as a mask for the only other input color and turn everything else black.

    Input structure
    - The grid contains exactly two colors.
    - Grey marks the pixels that survive into the output.
    - The other input color is the paint color and is not black.

    Rule
    - Create a black output grid with the same shape as the input.
    - Recolor every grey input pixel to the unique non-grey input color.
    - Recolor every non-grey input pixel to black.

    Key insight
    - Grey is not a background here; it is the mask showing where the colored
      shape should appear after the original colored cells are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = colors_present(g)
    assert len(colors) == 2, "expected exactly two input colors"
    assert Color.GREY in colors, "expected grey mask pixels"

    paint_colors = [color for color in colors if color != Color.GREY]
    assert len(paint_colors) == 1, "expected one paint color"
    paint_color = int(paint_colors[0])
    assert paint_color != Color.BLACK, "paint color must be distinct from output black"

    out = np.full(g.shape, Color.BLACK, dtype=int)
    out[g == Color.GREY] = paint_color
    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, grey density, and retry limits.
GRID_SIZE_RANGE = (3, 6)
PAINT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GREY_RATIO_RANGE = (0.28, 0.56)
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    size = int(np.random.randint(*GRID_SIZE_RANGE))
    total = size * size
    paint_color = int(np.random.choice(PAINT_COLORS))

    min_grey = max(1, int(np.ceil(GREY_RATIO_RANGE[0] * total)))
    max_grey = min(total - 1, int(np.floor(GREY_RATIO_RANGE[1] * total)))
    grey_count = int(np.random.randint(min_grey, max_grey + 1))

    cells = np.full(total, paint_color, dtype=int)
    cells[:grey_count] = Color.GREY
    np.random.shuffle(cells)
    inp = cells.reshape(size, size)

    expected = np.where(inp == Color.GREY, paint_color, Color.BLACK)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid f76d97a5 input")

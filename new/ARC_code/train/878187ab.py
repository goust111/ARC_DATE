from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the scattered foreground markers with a bottom-left count rectangle whose two corner diagonals encode the marker counts.

    Input structure
    - A background grid with exactly two foreground colors used as scattered marker pixels.
    - The two foreground colors have distinct pixel counts.

    Rule
    - Ignore marker positions and use only the two foreground counts.
    - Make a 16-by-16 background canvas, fill a bottom-left red rectangle whose height is the smaller count and width is the larger count.
    - Paint yellow pixels along the two rising 45-degree diagonals from the rectangle's bottom-left and bottom-right corners.

    Key insight
    - The marker colors and positions vary between examples; their counts alone determine the output rectangle dimensions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 2, "expected exactly two foreground marker colors"

    marker_counts = [int(np.count_nonzero(g == color)) for color in marker_colors]
    assert marker_counts[0] != marker_counts[1], "marker counts must determine distinct height and width"
    height, width = sorted(marker_counts)

    assert max(map(int, g.shape)) <= 16, "input must fit the task's 16-by-16 output canvas"
    side = 16
    assert height <= width <= side, "count rectangle must fit in the output canvas"
    assert bg not in (Color.RED, Color.YELLOW), "output drawing colors must differ from the background"

    out = np.full((side, side), bg, dtype=int)
    r0 = side - height
    out[r0:, :width] = Color.RED

    for rr in range(height):
        d = height - 1 - rr
        assert 0 <= d < width
        assert 0 <= width - 1 - d < width
        out[r0 + rr, d] = Color.YELLOW
        out[r0 + rr, width - 1 - d] = Color.YELLOW

    return out

# --------------- generator (att2) ---------------
# Output canvas and rectangle bounds.
BACKGROUND_COLOR = Color.ORANGE
OUTPUT_SIDE = 16
RECTANGLE_COLOR = Color.RED
DIAGONAL_COLOR = Color.YELLOW
RECT_HEIGHT_RANGE = (3, 14)
RECT_WIDTH_MIN = 5
# Input canvas bounds.
INPUT_SIDE_RANGE = (10, 16)
# Marker scatter and color assignment.
SCATTER_NEIGHBORHOOD = 1
SMALL_COLOR_FIRST_P = 0.5
# Generation retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _expected_output(bg: int, h: int, w: int) -> np.ndarray:
    out = np.full((OUTPUT_SIDE, OUTPUT_SIDE), bg, dtype=int)
    r0 = OUTPUT_SIDE - h
    out[r0:, :w] = RECTANGLE_COLOR
    rows = r0 + np.arange(h)
    offsets = h - 1 - np.arange(h)
    out[rows, offsets] = DIAGONAL_COLOR
    out[rows, w - 1 - offsets] = DIAGONAL_COLOR
    return out

def _scattered_cells(H: int, W: int, count: int) -> list[int]:
    """Prefer non-touching cells, then fall back to any unused cells."""
    cells = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    sep = SCATTER_NEIGHBORHOOD
    for r, c in cells:
        if all(abs(r - rr) > sep or abs(c - cc) > sep for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                break

    if len(chosen) < count:
        used = {r * W + c for r, c in chosen}
        remaining = [idx for idx in range(H * W) if idx not in used]
        random.shuffle(remaining)
        chosen.extend((idx // W, idx % W) for idx in remaining[: count - len(chosen)])

    random.shuffle(chosen)
    return [r * W + c for r, c in chosen]

def _gen_attempt() -> np.ndarray | None:
    bg = BACKGROUND_COLOR
    h = random.randint(*RECT_HEIGHT_RANGE)
    w = random.randint(max(h + 1, RECT_WIDTH_MIN), OUTPUT_SIDE)
    expected = _expected_output(bg, h, w)

    H = random.randint(*INPUT_SIDE_RANGE)
    W = random.randint(*INPUT_SIDE_RANGE)

    colors = [c for c in Color.ALL_COLORS if c != bg]
    color_a, color_b = random.sample(colors, 2)
    if random.random() < SMALL_COLOR_FIRST_P:
        small_color, large_color = color_a, color_b
    else:
        small_color, large_color = color_b, color_a

    inp = np.full((H, W), bg, dtype=int)
    cells = _scattered_cells(H, W, h + w)
    inp.flat[cells[:h]] = small_color
    inp.flat[cells[h:]] = large_color

    if np.array_equal(main(inp), expected):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 878187ab input")

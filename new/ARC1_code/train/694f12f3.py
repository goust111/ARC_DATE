from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the interior of the smaller yellow rectangle blue and the larger yellow rectangle red.

    Input structure:
    - The grid is a black canvas containing exactly two solid rectangles of one
      non-black foreground color.
    - Each rectangle is at least 3 cells tall and 3 cells wide, so it has a
      one-cell border and a nonempty interior.
    - The two rectangles have different areas.

    Rule:
    - Leave the black background and each foreground rectangle's one-cell border
      unchanged.
    - Fill the smaller rectangle's interior with blue.
    - Fill the larger rectangle's interior with red.

    Key insight:
    - The output colors are assigned by rectangle area rank, while the rewritten
      region is simply each rectangle's bbox interior.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert Color.BLACK in colors, "expected a black background"

    foreground_colors = colors - {Color.BLACK}
    assert len(foreground_colors) == 1, "expected one foreground rectangle color"
    foreground = next(iter(foreground_colors))

    rects = []
    for coords in components_mask(g == foreground, connectivity=4):
        assert coords.ndim == 2 and coords.shape[1] == 2, "component coords must be (N,2)"
        r0, c0, r1, c1 = bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(coords.shape[0])
        assert h >= 3 and w >= 3, "rectangles must have nonempty interiors"
        assert area == h * w, "foreground components must be solid rectangles"
        rects.append((area, r0, c0, r1, c1))

    assert len(rects) == 2, "expected exactly two foreground rectangles"
    assert len({area for area, *_ in rects}) == 2, "rectangle areas must be distinct"

    out = g.copy()
    for fill_color, (_area, r0, c0, r1, c1) in zip(
        (Color.BLUE, Color.RED),
        sorted(rects, key=lambda x: x[0]),
    ):
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_SIZE = 10
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
RECTANGLE_COLOR = Color.YELLOW
INTERIOR_COLORS_BY_AREA = (Color.BLUE, Color.RED)

# Rectangle sampling.
MIN_RECT_SIDE, MAX_RECT_SIDE, MIN_RECT_GAP = 3, 6, 1
RECT_DIMENSIONS = tuple(
    (h, w)
    for h in range(MIN_RECT_SIDE, MAX_RECT_SIDE + 1)
    for w in range(MIN_RECT_SIDE, MAX_RECT_SIDE + 1)
)
VALID_DIMENSION_PAIRS = tuple(
    ((h1, w1), (h2, w2))
    for h1, w1 in RECT_DIMENSIONS
    for h2, w2 in RECT_DIMENSIONS
    if h1 * w1 != h2 * w2
    and (h1 + h2 + MIN_RECT_GAP <= GRID_SIZE or w1 + w2 + MIN_RECT_GAP <= GRID_SIZE)
)

# Generation retry limit.
GENERATION_ATTEMPTS = 300
Placement = tuple[int, int, int, int]  # (r, c, h, w)


def _separated_offsets(size_a: int, size_b: int, limit: int) -> tuple[int, int]:
    """Return ordered starts for two 1D spans separated by at least MIN_RECT_GAP."""
    gap = random.randint(MIN_RECT_GAP, limit - size_a - size_b)
    start_a = random.randint(0, limit - size_a - gap - size_b)
    return start_a, start_a + size_a + gap


def _place_ordered_pair(
    rects: list[tuple[int, int]], order: list[int], layout: str
) -> dict[int, Placement]:
    first_i, second_i = order
    h_first, w_first = rects[first_i]
    h_second, w_second = rects[second_i]

    if layout == "horizontal":
        first_c, second_c = _separated_offsets(w_first, w_second, GRID_SIZE)
        return {
            first_i: (random.randint(0, GRID_SIZE - h_first), first_c, h_first, w_first),
            second_i: (random.randint(0, GRID_SIZE - h_second), second_c, h_second, w_second),
        }

    first_r, second_r = _separated_offsets(h_first, h_second, GRID_SIZE)
    return {
        first_i: (first_r, random.randint(0, GRID_SIZE - w_first), h_first, w_first),
        second_i: (second_r, random.randint(0, GRID_SIZE - w_second), h_second, w_second),
    }


def _gen_place_rectangles(dim_a: tuple[int, int], dim_b: tuple[int, int]) -> list[Placement]:
    h1, w1 = dim_a
    h2, w2 = dim_b
    layouts = []
    if w1 + w2 + MIN_RECT_GAP <= GRID_SIZE:
        layouts.append("horizontal")
    if h1 + h2 + MIN_RECT_GAP <= GRID_SIZE:
        layouts.append("vertical")

    rects = [dim_a, dim_b]
    layout = random.choice(layouts)
    order = [0, 1]
    random.shuffle(order)
    placed = _place_ordered_pair(rects, order, layout)

    return [placed[0], placed[1]]


def _render_input_grid(placements: list[Placement]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for r, c, h, w in placements:
        paste(grid, np.full((h, w), RECTANGLE_COLOR, dtype=int), r, c)
    return grid


def _gen_expected_output(grid: np.ndarray, placements: list[Placement]) -> np.ndarray:
    out = grid.copy()
    ranked = sorted(
        ((h * w, r, c, h, w) for r, c, h, w in placements),
        key=lambda item: item[0],
    )
    for fill, (_area, r, c, h, w) in zip(INTERIOR_COLORS_BY_AREA, ranked):
        out[r + 1 : r + h - 1, c + 1 : c + w - 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    dim_a, dim_b = random.choice(VALID_DIMENSION_PAIRS)
    placements = _gen_place_rectangles(dim_a, dim_b)
    grid = _render_input_grid(placements)
    expected = _gen_expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 694f12f3 input")

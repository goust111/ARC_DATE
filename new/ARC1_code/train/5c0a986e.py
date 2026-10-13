from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, draw_line, find_components, get_bg, paste
import random


def _assert_clear_ray(
    g: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    bg: int,
) -> None:
    height, width = map(int, g.shape)
    r, c = start
    dr, dc = direction
    while 0 <= r < height and 0 <= c < width:
        assert int(g[r, c]) in {int(bg), int(color)}, "ray path must not cross the other block"
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue square up-left and the red square down-right along their diagonals to the grid edge.

    Input structure
    - The grid has a black background.
    - There is exactly one blue 2x2 square and exactly one red 2x2 square.
    - The diagonal ray beyond each active corner is clear background until it leaves the grid.

    Rule
    - Keep the input pixels unchanged.
    - From the blue square's top-left corner, paint a blue diagonal ray in the up-left direction.
    - From the red square's bottom-right corner, paint a red diagonal ray in the down-right direction.
    - Each ray stops only when the next step would leave the grid.

    Key insight
    - The colored 2x2 squares are fixed emitters: blue uses its northwest corner and red uses its southeast corner, so the missing pixels are the two outward diagonal continuations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(comps) == 2, "expected exactly two foreground components"

    by_color: dict[int, dict] = {}
    for comp in comps:
        color = int(comp["color"])
        assert color in {Color.BLUE, Color.RED}, "expected only blue and red foreground"
        assert color not in by_color, "expected one square per foreground color"
        assert int(comp["area"]) == 4, "expected each foreground object to have four pixels"
        assert bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2, "expected solid 2x2 squares"
        by_color[color] = comp

    assert set(by_color) == {Color.BLUE, Color.RED}, "expected one blue square and one red square"

    out = g.copy()

    blue_bbox = by_color[Color.BLUE]["bbox"]
    blue_start = (int(blue_bbox[0]), int(blue_bbox[1]))
    _assert_clear_ray(g, blue_start, (-1, -1), Color.BLUE, bg)
    draw_line(out, *blue_start, direction=(-1, -1), color=Color.BLUE)

    red_bbox = by_color[Color.RED]["bbox"]
    red_start = (int(red_bbox[2]), int(red_bbox[3]))
    _assert_clear_ray(g, red_start, (1, 1), Color.RED, bg)
    draw_line(out, *red_start, direction=(1, 1), color=Color.RED)

    return out

# --------------- generator (o5x) ---------------

# Canvas and object geometry.
GEN_SHAPE = (10, 10)
GEN_SQUARE_SIZE = 2

# Colors.
GEN_BG, GEN_BLUE, GEN_RED = Color.BLACK, Color.BLUE, Color.RED

# Ray and retry constraints.
GEN_RAY_LEN_MIN, GEN_RAY_LEN_MAX = 2, 6
GEN_BLUE_DIR, GEN_RED_DIR = (-1, -1), (1, 1)
GEN_MAX_ATTEMPTS = 300


def _gen_line_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    height, width = GEN_SHAPE
    cells = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _gen_square_cells(top: int, left: int) -> set[tuple[int, int]]:
    side = GEN_SQUARE_SIZE - 1
    return {(top, left), (top + side, left), (top, left + side), (top + side, left + side)}


def _gen_square_axes(top: int, left: int) -> tuple[set[int], set[int]]:
    side = GEN_SQUARE_SIZE - 1
    return {top, top + side}, {left, left + side}


def _gen_corner_candidates(ray_len: int, *, southeast: bool = False) -> list[tuple[int, int]]:
    """Return blue top-left or red bottom-right candidates for a ray length."""
    height, width = GEN_SHAPE
    if southeast:
        rows = range(GEN_SQUARE_SIZE - 1, height)
        cols = range(GEN_SQUARE_SIZE - 1, width)
        return [(r, c) for r in rows for c in cols if min(height - 1 - r, width - 1 - c) == ray_len]
    rows = range(ray_len, height - GEN_SQUARE_SIZE + 1)
    cols = range(ray_len, width - GEN_SQUARE_SIZE + 1)
    return [(r, c) for r in rows for c in cols if min(r, c) == ray_len]


def _gen_valid_layouts(ray_len: int) -> list[tuple]:
    """Enumerate valid layouts for one ray length in sampling order."""
    valid = []
    red_bottoms = _gen_corner_candidates(ray_len, southeast=True)
    for blue_r, blue_c in _gen_corner_candidates(ray_len):
        blue_rows, blue_cols = _gen_square_axes(blue_r, blue_c)
        blue_square = _gen_square_cells(blue_r, blue_c)
        blue_ray = set(_gen_line_cells((blue_r, blue_c), GEN_BLUE_DIR))

        for red_r, red_c in red_bottoms:
            red_top = red_r - GEN_SQUARE_SIZE + 1
            red_left = red_c - GEN_SQUARE_SIZE + 1
            red_rows, red_cols = _gen_square_axes(red_top, red_left)
            if blue_rows & red_rows or blue_cols & red_cols:
                continue

            red_square = _gen_square_cells(red_top, red_left)
            red_ray = set(_gen_line_cells((red_r, red_c), GEN_RED_DIR))
            if blue_ray & red_square or red_ray & blue_square or blue_ray & red_ray:
                continue

            valid.append(((blue_r, blue_c), (red_r, red_c), blue_square, red_square, blue_ray, red_ray))

    return valid


def _gen_render_layout(layout: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Render a valid layout as (input, output)."""
    (blue_r, blue_c), (red_r, red_c), blue_square, red_square, blue_ray, red_ray = layout
    out = np.full(GEN_SHAPE, GEN_BG, dtype=int)
    square_shape = (GEN_SQUARE_SIZE, GEN_SQUARE_SIZE)
    paste(out, np.full(square_shape, GEN_BLUE, dtype=int), blue_r, blue_c)
    paste(out, np.full(square_shape, GEN_RED, dtype=int), red_r - GEN_SQUARE_SIZE + 1, red_c - GEN_SQUARE_SIZE + 1)
    draw_line(out, blue_r, blue_c, direction=GEN_BLUE_DIR, color=GEN_BLUE)
    draw_line(out, red_r, red_c, direction=GEN_RED_DIR, color=GEN_RED)

    inp = out.copy()
    for r, c in (blue_ray - blue_square) | (red_ray - red_square):
        inp[r, c] = GEN_BG
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    ray_len = random.randint(GEN_RAY_LEN_MIN, GEN_RAY_LEN_MAX)
    valid = _gen_valid_layouts(ray_len)
    if not valid:
        return None

    inp, out = _gen_render_layout(random.choice(valid))
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5c0a986e input")

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, get_bg
import random


def _paint_clipped_square(out: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = out.shape
    painted = False
    for dr in range(2):
        for dc in range(2):
            r = r0 + dr
            c = c0 + dc
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
                painted = True
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the 2x2 seed diagonally from each red corner, recoloring every stamped cell with the seed's non-red color.

    Input structure
    - The grid has a dominant background color.
    - All foreground pixels form one full 2x2 seed block.
    - The seed contains one non-red paint color and one or more red corner control pixels.

    Rule
    - Replace every red control pixel with the paint color.
    - A red pixel in the top-left, top-right, bottom-left, or bottom-right corner selects the matching diagonal direction.
    - For each selected direction, repeatedly stamp a filled 2x2 square of the paint color one row and one column at a time until the square is fully outside the grid.

    Key insight
    - The red pixels are not output colors; their corner positions encode the diagonal rays along which the 2x2 seed is copied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    seed_bbox = bbox(g != bg)
    assert seed_bbox is not None, "expected a foreground seed"
    r0, c0, r1, c1 = seed_bbox
    assert r1 - r0 == 1 and c1 - c0 == 1, "foreground seed must be exactly 2x2"

    seed = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(seed != bg), "2x2 seed must be fully occupied"
    paint_colors = {int(v) for v in np.unique(seed)} - {Color.RED}
    assert len(paint_colors) == 1, "expected exactly one non-red seed color"
    paint = paint_colors.pop()

    red_corners = np.argwhere(seed == Color.RED)
    assert len(red_corners) > 0, "expected at least one red control corner"

    out = np.full_like(g, bg)
    for local_r, local_c in red_corners:
        step_r = -1 if int(local_r) == 0 else 1
        step_c = -1 if int(local_c) == 0 else 1
        rr = r0
        cc = c0
        while _paint_clipped_square(out, rr, cc, paint):
            rr += step_r
            cc += step_c

    return out

# --------------- generator (o5x) ---------------

# Canvas and 2x2 stamp geometry.
_GEN_SHAPE = (9, 9)
_GEN_STAMP_SIZE = 2
_GEN_SEED_TOPS = range(1, 7)
_GEN_STAMP_OVERLAP_LOW = 1 - _GEN_STAMP_SIZE

# Palette and red-control selection.
_GEN_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
_GEN_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_GEN_RED_COUNT_RANGE = (1, 3)

# Ray constraints and retry policy.
_GEN_RAY_LEN_RANGE = (4, 7)
_GEN_RETRY_LIMIT = 300


def _gen_corner_step(local_r: int, local_c: int) -> tuple[int, int]:
    return -1 if local_r == 0 else 1, -1 if local_c == 0 else 1


def _gen_ray_len(r0: int, c0: int, local_r: int, local_c: int) -> int:
    step_r, step_c = _gen_corner_step(local_r, local_c)
    rr, cc = r0, c0
    length = 0
    while (
        _GEN_STAMP_OVERLAP_LOW <= rr < _GEN_SHAPE[0]
        and _GEN_STAMP_OVERLAP_LOW <= cc < _GEN_SHAPE[1]
    ):
        length += 1
        rr += step_r
        cc += step_c
    return length


def _gen_valid_positions(red_corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return top-left seed positions whose selected rays match length bounds."""
    min_len, max_len = _GEN_RAY_LEN_RANGE
    valid_positions = []
    for r0 in _GEN_SEED_TOPS:
        for c0 in _GEN_SEED_TOPS:
            lengths = [_gen_ray_len(r0, c0, lr, lc) for lr, lc in red_corners]
            if all(min_len <= length <= max_len for length in lengths):
                valid_positions.append((r0, c0))
    return valid_positions


def _gen_render_input(r0: int, c0: int, red_corners, paint: int) -> np.ndarray:
    inp = np.full(_GEN_SHAPE, Color.BLACK, dtype=int)
    inp[r0 : r0 + _GEN_STAMP_SIZE, c0 : c0 + _GEN_STAMP_SIZE] = paint
    for local_r, local_c in red_corners:
        inp[r0 + local_r, c0 + local_c] = Color.RED
    return inp


def _gen_expected_output(r0: int, c0: int, red_corners, paint: int) -> np.ndarray:
    """Render the expected diagonal stamps independently of main()."""
    out = np.full(_GEN_SHAPE, Color.BLACK, dtype=int)
    for local_r, local_c in red_corners:
        step_r, step_c = _gen_corner_step(local_r, local_c)
        rr, cc = r0, c0
        while _paint_clipped_square(out, rr, cc, paint):
            rr += step_r
            cc += step_c
    return out


def _gen_attempt() -> np.ndarray | None:
    paint = random.choice(_GEN_COLORS)
    n_red = random.randint(*_GEN_RED_COUNT_RANGE)
    red_corners = random.sample(_GEN_CORNERS, n_red)
    valid_positions = _gen_valid_positions(red_corners)
    if not valid_positions:
        return None

    r0, c0 = random.choice(valid_positions)
    inp = _gen_render_input(r0, c0, red_corners, paint)
    expected = _gen_expected_output(r0, c0, red_corners, paint)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")

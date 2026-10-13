from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

CENTER_COLOR = RED
PALETTE_POOL = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)
MAX_SIDE = 29
MAX_GENERATION_ATTEMPTS = 300

_CASE_CYCLE = (
    (1, 4),
    (2, 2),
    (3, 2),
    (1, 3),
    (2, 3),
    (1, 2),
)
_GEN_CALL_COUNT = 0


def _palette_from_right_edge(grid: np.ndarray) -> list[int]:
    col = grid[:, -1]
    palette: list[int] = []
    for value in map(int, col):
        if value == BLACK:
            break
        palette.append(value)
    assert 2 <= len(palette) <= 4, "expected a 2-4 color palette on the right edge"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color not in (BLACK, GREY) for color in palette)
    assert np.all(col[len(palette) :] == BLACK), "palette strip must be top-aligned and contiguous"
    return palette


def _run_length_from_top_left(grid: np.ndarray) -> int:
    run_length = 0
    while run_length < grid.shape[0] and int(grid[run_length, 0]) == GREY:
        run_length += 1
    assert 1 <= run_length <= 3, "expected a grey height marker of 1-3 pixels"
    assert np.all(grid[run_length:, 0] != GREY), "grey marker must be one contiguous top-left segment"
    return run_length


def _ring_layers(side: int) -> int:
    return side // 2 + 1


def _paint_rings(side: int, palette: list[int], run_length: int) -> np.ndarray:
    center = side // 2
    rows, cols = np.indices((side, side))
    distances = np.maximum(np.abs(rows - center), np.abs(cols - center))
    palette_array = np.array(palette, dtype=int)
    color_indices = (distances // int(run_length)) % len(palette)
    out = palette_array[color_indices]
    out[:run_length, 0] = BLACK
    out[: len(palette), -1] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an odd square grid into concentric Chebyshev rings using a cyclic edge palette.

    Input structure:
    - The grid is an odd-sized square with black background and a single non-black center marker.
    - A grey vertical segment in the top-left corner encodes the run length by its height.
    - A top-aligned palette strip on the right edge lists 2 to 4 distinct colors from top to bottom.

    Rule:
    - Read the grey segment height R and the right-edge palette P in top-to-bottom order.
    - For every cell, compute its Chebyshev distance d from the geometric center of the square.
    - Paint the cell with P[floor(d / R) mod len(P)], so each palette color spans R whole ring layers before cycling.
    - Erase the grey segment and the palette strip back to black; the center naturally becomes P[0].

    Key insight:
    - The grey marker does not choose one ring directly; it chooses how many consecutive concentric rings share each palette color before the palette repeats.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 1, "expected an odd side length"

    side = int(g.shape[0])
    center = side // 2
    assert int(g[center, center]) != BLACK, "expected a visible center marker"

    palette = _palette_from_right_edge(g)
    run_length = _run_length_from_top_left(g)
    return _paint_rings(side, palette, run_length)


def _min_side_for_case(run_length: int, palette_len: int) -> int:
    return 4 * int(run_length) * int(palette_len) - 1


def _choose_case() -> tuple[int, int]:
    global _GEN_CALL_COUNT
    case = _CASE_CYCLE[_GEN_CALL_COUNT % len(_CASE_CYCLE)]
    _GEN_CALL_COUNT += 1
    return case


def _sample_side(run_length: int, palette_len: int) -> int:
    min_side = _min_side_for_case(run_length, palette_len)
    assert min_side <= MAX_SIDE
    choices = list(range(min_side, MAX_SIDE + 1, 2))
    usable = choices[: min(len(choices), 5)]
    return int(random.choice(usable))


def _build_input(side: int, palette: list[int], run_length: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    center = side // 2
    grid[center, center] = CENTER_COLOR
    grid[:run_length, 0] = GREY
    grid[: len(palette), -1] = np.array(palette, dtype=int)
    return grid


def _probe_ring_colors(grid: np.ndarray) -> list[int]:
    center = grid.shape[0] // 2
    return [int(grid[center - d, center]) for d in range(_ring_layers(grid.shape[0]))]


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    side = int(grid.shape[0])
    if grid.shape[1] != side or side % 2 != 1 or side > 30:
        return False

    palette = _palette_from_right_edge(grid)
    run_length = _run_length_from_top_left(grid)
    layer_count = _ring_layers(side)
    if layer_count < 2 * len(palette) * run_length:
        return False

    if int(grid[side // 2, side // 2]) != CENTER_COLOR:
        return False

    expected = _paint_rings(side, palette, run_length)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if not np.all(actual[:run_length, 0] == BLACK):
        return False
    if not np.all(actual[: len(palette), -1] == BLACK):
        return False

    ring_colors = _probe_ring_colors(actual)
    expected_colors = [palette[(d // run_length) % len(palette)] for d in range(layer_count)]
    if ring_colors != expected_colors:
        return False
    if ring_colors[: len(palette) * run_length] != ring_colors[len(palette) * run_length : 2 * len(palette) * run_length]:
        return False

    if run_length > 1:
        alt = _paint_rings(side, palette, 1)
        if np.array_equal(actual, alt):
            return False
        if len(set(ring_colors[:run_length])) != 1:
            return False
        if ring_colors[run_length] == ring_colors[0]:
            return False
    else:
        if ring_colors[1] == ring_colors[0]:
            return False

    rotated_palette = palette[1:] + palette[:1]
    if np.array_equal(actual, _paint_rings(side, rotated_palette, run_length)):
        return False

    reversed_palette = list(reversed(palette))
    if np.array_equal(actual, _paint_rings(side, reversed_palette, run_length)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        run_length, palette_len = _choose_case()
        side = _sample_side(run_length, palette_len)
        palette = random.sample(PALETTE_POOL, palette_len)
        grid = _build_input(side, palette, run_length)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring palette instance")

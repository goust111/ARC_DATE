from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 3000


def _unique_pos(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape[0] == 1, f"expected exactly one pixel of color {color}"
    return tuple(map(int, coords[0]))


def _horizontal_then_vertical_cells(
    start: tuple[int, int], target: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, target)
    cells: list[tuple[int, int]] = []

    if c0 != c1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r0, c))

    if r0 != r1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c1))

    return [cell for cell in cells if cell != target]


def _vertical_then_horizontal_cells(
    start: tuple[int, int], target: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, target)
    cells: list[tuple[int, int]] = []

    if r0 != r1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c0))

    if c0 != c1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r1, c))

    return [cell for cell in cells if cell != target]


def _parse_instance(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int], int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30

    colors = set(map(int, np.unique(g)))
    assert colors.issubset({BLACK, RED, GREEN, YELLOW, GREY})
    red = _unique_pos(g, RED)
    green = _unique_pos(g, GREEN)
    relay = _unique_pos(g, YELLOW)

    rr, rc = red
    gr, gc = green
    yr, yc = relay
    assert len({rr, gr, yr}) == 3, "red, green, and yellow must occupy distinct rows"
    assert len({rc, gc, yc}) == 3, "red, green, and yellow must occupy distinct columns"

    k = 0
    c = yc + 1
    while c < w and int(g[yr, c]) == GREY:
        k += 1
        c += 1
    assert 1 <= k <= 4, "expected a grey counter run of length 1..4"

    expected_counter = {(yr, yc + offset) for offset in range(1, k + 1)}
    grey_cells = {tuple(map(int, pos)) for pos in np.argwhere(g == GREY)}
    assert grey_cells == expected_counter, "grey pixels must be the contiguous counter to the right of yellow"

    shifted = (yr, yc + k)
    assert 0 <= shifted[1] < w
    assert shifted not in {red, green}, "shifted relay must not overlap the endpoints"
    return g, red, green, relay, k, shifted


def _apply_paths(
    grid: np.ndarray,
    relay: tuple[int, int],
    shifted: tuple[int, int],
    path1: list[tuple[int, int]],
    path2: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BLACK
    out[relay] = BLACK
    out[shifted] = YELLOW
    for r, c in path1 + path2:
        if out[r, c] == BLACK:
            out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the grey counter, shift the yellow relay right, then connect red to green through that shifted relay with grey orthogonal routes.

    Input structure:
    - The grid has a black background with exactly one red pixel, one green pixel, and one yellow pixel.
    - A contiguous run of 1 to 4 grey pixels sits immediately to the right of the yellow pixel on the same row.
    - The red, green, and yellow markers occupy distinct rows and distinct columns.

    Rule:
    - Count the grey run length K, erase the grey counter, and move the yellow relay K cells to the right.
    - Clear the original yellow cell to black, keep the red, green, and shifted yellow markers unchanged in color.
    - Draw a grey path from red to the shifted yellow by moving horizontally first and then vertically, painting only black cells.
    - Draw a second grey path from the shifted yellow to green with the same horizontal-first routing, again painting only black cells.

    Key insight:
    - The grey pixels are not part of the final path; they act as a numeric instruction whose count changes the relay location, and that moved relay determines the geometry of both route legs.
    """
    g, red, green, relay, _k, shifted = _parse_instance(grid)
    path1 = _horizontal_then_vertical_cells(red, shifted)
    path2 = _horizontal_then_vertical_cells(shifted, green)
    return _apply_paths(g, relay, shifted, path1, path2)


def _build_input(
    h: int,
    w: int,
    red: tuple[int, int],
    green: tuple[int, int],
    relay: tuple[int, int],
    k: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[red] = RED
    grid[green] = GREEN
    grid[relay] = YELLOW
    for offset in range(1, k + 1):
        grid[relay[0], relay[1] + offset] = GREY
    return grid


def _sample_from_side(values: list[int], choose_low_side: bool) -> int:
    half = max(1, len(values) // 2)
    candidates = values[:half] if choose_low_side else values[-half:]
    return int(random.choice(candidates))


def _is_good_generator_instance(grid: np.ndarray) -> bool:
    try:
        g, red, green, relay, _k, shifted = _parse_instance(grid)
    except AssertionError:
        return False

    rr, rc = red
    gr, gc = green
    yr, yc2 = shifted
    _yr, yc0 = relay

    if yc2 in {rc, gc}:
        return False

    leg1 = _horizontal_then_vertical_cells(red, shifted)
    leg2 = _horizontal_then_vertical_cells(shifted, green)
    if not leg1 or not leg2:
        return False
    if len(leg1) < 2 or len(leg2) < 2:
        return False

    path_set = set(leg1) | set(leg2)
    if len(path_set) != len(leg1) + len(leg2):
        return False
    if len(path_set) < 6:
        return False

    direct_hv = set(_horizontal_then_vertical_cells(red, green))
    direct_vh = set(_vertical_then_horizontal_cells(red, green))
    if path_set == direct_hv or path_set == direct_vh:
        return False

    via_unshifted = set(_horizontal_then_vertical_cells(red, relay)) | set(
        _horizontal_then_vertical_cells(relay, green)
    )
    if path_set == via_unshifted:
        return False

    if abs(rr - yr) + abs(rc - yc2) < 4:
        return False
    if abs(gr - yr) + abs(gc - yc2) < 4:
        return False

    out = main(g)
    grey_cells = {tuple(map(int, pos)) for pos in np.argwhere(out == GREY)}
    if grey_cells != path_set:
        return False
    if _unique_pos(out, YELLOW) != shifted:
        return False
    if int(out[red]) != RED or int(out[green]) != GREEN:
        return False

    wrong_direct = _apply_paths(
        g,
        relay,
        shifted,
        _horizontal_then_vertical_cells(red, green),
        [],
    )
    if np.array_equal(out, wrong_direct):
        return False

    wrong_unshifted = _apply_paths(
        g,
        relay,
        relay,
        _horizontal_then_vertical_cells(red, relay),
        _horizontal_then_vertical_cells(relay, green),
    )
    if np.array_equal(out, wrong_unshifted):
        return False

    wrong_vfirst = _apply_paths(
        g,
        relay,
        shifted,
        _vertical_then_horizontal_cells(red, shifted),
        _vertical_then_horizontal_cells(shifted, green),
    )
    if np.array_equal(out, wrong_vfirst):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(1, 4)
    h = random.randint(10, 15)
    w = random.randint(max(10, k + 7), 15)

    relay_row = random.randint(2, h - 3)
    relay_col = random.randint(2, w - k - 3)
    shifted_col = relay_col + k

    upper_rows = list(range(0, relay_row))
    lower_rows = list(range(relay_row + 1, h))
    left_cols = list(range(0, relay_col))
    right_cols = list(range(shifted_col + 1, w))
    if min(len(upper_rows), len(lower_rows), len(left_cols), len(right_cols)) < 2:
        return None

    red = (
        _sample_from_side(upper_rows if random.random() < 0.5 else lower_rows, random.random() < 0.5),
        _sample_from_side(left_cols if random.random() < 0.5 else right_cols, random.random() < 0.5),
    )
    green = (
        _sample_from_side(upper_rows if random.random() < 0.5 else lower_rows, random.random() < 0.5),
        _sample_from_side(left_cols if random.random() < 0.5 else right_cols, random.random() < 0.5),
    )
    if red[0] == green[0] or red[1] == green[1]:
        return None

    relay = (relay_row, relay_col)
    grid = _build_input(h, w, red, green, relay, k)
    if not _is_good_generator_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

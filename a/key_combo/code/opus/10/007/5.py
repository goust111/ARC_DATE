from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PLUS_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _cycle_color(distance_from_arm: int) -> int:
    return YELLOW if distance_from_arm % 3 == 0 else TEAL


def _find_plus_center(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == RED)
    assert red.shape == (5, 2), "expected exactly one five-pixel red plus"
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    assert int(r1 - r0) == 2 and int(c1 - c0) == 2, "red plus must occupy a 3x3 box"
    cr = int(r0 + 1)
    cc = int(c0 + 1)
    expected = {(cr + dr, cc + dc) for dr, dc in PLUS_OFFSETS}
    assert set(map(tuple, red)) == expected, "red pixels must form a plus"
    return cr, cc


def _uniform_step(distances: list[int]) -> int:
    assert len(distances) >= 2, "each track side must contain at least two dots"
    diffs = [b - a for a, b in zip(distances, distances[1:])]
    assert len(set(diffs)) == 1, "track dots on one side must be evenly spaced"
    step = diffs[0]
    assert 2 <= step <= 4, "track spacing must be between 2 and 4"
    return step


def _parse_track_offsets(
    grid: np.ndarray,
    color: int,
    aligned_axis: int,
    offset_center: int,
    is_row_track: bool,
) -> dict[str, list[int] | int]:
    coords = np.argwhere(grid == color)
    assert coords.shape[0] >= 4, "expected at least four track dots of each axis color"

    if is_row_track:
        rows = {int(r) for r, _c in coords}
        assert rows == {aligned_axis}, "horizontal track dots must share the plus row"
        offsets = sorted(int(c) for _r, c in coords)
    else:
        cols = {int(c) for _r, c in coords}
        assert cols == {aligned_axis}, "vertical track dots must share the plus column"
        offsets = sorted(int(r) for r, _c in coords)

    lower = []
    upper = []
    for value in offsets:
        delta = value - offset_center
        assert delta not in (-1, 0, 1), "track dots must not overlap or touch the red cross arms"
        if delta < 0:
            lower.append(-delta)
        else:
            upper.append(delta)

    lower = sorted(lower)
    upper = sorted(upper)
    assert len(lower) >= 2 and len(upper) >= 2, "each side of a track must contain at least two dots"
    assert lower[0] >= 3 and upper[0] >= 3, "the first dot must leave at least one paintable ray cell"

    lower_step = _uniform_step(lower)
    upper_step = _uniform_step(upper)
    assert lower_step == upper_step, "both sides of the same track must use one common spacing"

    return {
        "negative": lower,
        "positive": upper,
        "step": lower_step,
    }


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}, "unexpected input colors"
    assert _get_bg(g) == BLACK, "background must be black"

    cr, cc = _find_plus_center(g)
    horizontal = _parse_track_offsets(g, GREEN, cr, cc, True)
    vertical = _parse_track_offsets(g, BLUE, cc, cr, False)

    return {
        "center": (cr, cc),
        "horizontal": horizontal,
        "vertical": vertical,
    }


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cr, cc = info["center"]

    for dr, dc in DIAGONAL_DIRS:
        r = cr + dr
        c = cc + dc
        while _in_bounds(g.shape, r, c):
            assert int(g[r, c]) == BLACK, "diagonal rays must traverse background only"
            out[r, c] = GREY
            r += dr
            c += dc

    for dr, dc in CARDINAL_DIRS:
        arm_r = cr + dr
        arm_c = cc + dc
        distance = 1
        r = arm_r + dr
        c = arm_c + dc
        while _in_bounds(g.shape, r, c) and int(g[r, c]) == BLACK:
            out[r, c] = _cycle_color(distance)
            distance += 1
            r += dr
            c += dc

        assert _in_bounds(g.shape, r, c), "each cardinal ray must terminate at a track dot"
        assert int(g[r, c]) in {GREEN, BLUE}, "cardinal rays must stop at track dots"
        out[r, c] = _cycle_color(distance)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the red cross into eight rays, stopping cardinal rays at track dots and recoloring those terminal dots by the stripe cycle.

    Input structure:
    - The background is black.
    - One red five-pixel plus defines a center and four cardinal arm endpoints.
    - Green single-pixel dots lie on the plus row on both sides, and blue single-pixel dots lie on the plus column above and below; each axis uses an even spacing and leaves at least one empty cell before its first dot.

    Rule:
    - Keep the red plus unchanged.
    - From each diagonal neighbor of the plus center, paint a grey diagonal ray to the grid edge.
    - From each cardinal arm endpoint, paint outward with the repeating cycle teal, teal, yellow until just before the first track dot on that ray, then recolor that stopping dot to the color the cycle would have assigned at that distance.
    - Any farther track dots remain unchanged, and if a diagonal and cardinal ray would overlap, the cardinal color takes priority.

    Key insight:
    - The nearest track dot in each cardinal direction simultaneously fixes the ray length and encodes that exact distance through its final recolored cycle value.
    """
    return _apply_rule(grid)


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cr, cc = info["center"]

    for dr, dc in DIAGONAL_DIRS:
        r = cr + dr
        c = cc + dc
        while _in_bounds(g.shape, r, c):
            out[r, c] = GREY
            r += dr
            c += dc

    terminal_dots: dict[tuple[int, int], int] = {}
    for dr, dc in CARDINAL_DIRS:
        distance = 1
        r = cr + 2 * dr
        c = cc + 2 * dc
        while int(g[r, c]) == BLACK:
            out[r, c] = _cycle_color(distance)
            distance += 1
            r += dr
            c += dc
        terminal_dots[(r, c)] = _cycle_color(distance)

    for (r, c), color in terminal_dots.items():
        out[r, c] = color
    return out


def _build_input(
    height: int,
    width: int,
    center: tuple[int, int],
    step_h: int,
    step_v: int,
    left: tuple[int, int],
    right: tuple[int, int],
    up: tuple[int, int],
    down: tuple[int, int],
) -> np.ndarray:
    cr, cc = center
    grid = np.full((height, width), BLACK, dtype=int)
    for dr, dc in PLUS_OFFSETS:
        grid[cr + dr, cc + dc] = RED

    left_first, left_count = left
    right_first, right_count = right
    up_first, up_count = up
    down_first, down_count = down

    for i in range(left_count):
        grid[cr, cc - (left_first + i * step_h)] = GREEN
    for i in range(right_count):
        grid[cr, cc + (right_first + i * step_h)] = GREEN
    for i in range(up_count):
        grid[cr - (up_first + i * step_v), cc] = BLUE
    for i in range(down_count):
        grid[cr + (down_first + i * step_v), cc] = BLUE
    return grid


def _side_options(available: int, step: int) -> list[tuple[int, int]]:
    options = []
    for first in range(3, min(7, available + 1)):
        for count in range(2, 5):
            if first + (count - 1) * step <= available:
                options.append((first, count))
    return options


def _terminal_cycle_signature(first_offsets: list[int]) -> set[int]:
    return {_cycle_color(offset - 1) for offset in first_offsets}


def _sample_layout() -> tuple[int, int, tuple[int, int], int, int, tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    height = random.randint(13, 22)
    width = random.randint(13, 22)
    if height < 9 or width < 9:
        return None

    cr = random.randint(4, height - 5)
    cc = random.randint(4, width - 5)
    step_h = random.randint(2, 4)
    step_v = random.randint(2, 4)

    left_options = _side_options(cc, step_h)
    right_options = _side_options(width - 1 - cc, step_h)
    up_options = _side_options(cr, step_v)
    down_options = _side_options(height - 1 - cr, step_v)
    if not (left_options and right_options and up_options and down_options):
        return None

    left = random.choice(left_options)
    right = random.choice(right_options)
    up = random.choice(up_options)
    down = random.choice(down_options)

    first_offsets = [left[0], right[0], up[0], down[0]]
    if _terminal_cycle_signature(first_offsets) != {TEAL, YELLOW}:
        return None
    if len(set(first_offsets)) < 2:
        return None
    if left[0] == right[0] and up[0] == down[0]:
        return None
    if cr * 2 == height - 1 and cc * 2 == width - 1 and step_h == step_v:
        return None

    return height, width, (cr, cc), step_h, step_v, left, right, up, down


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    info = _analyze_input(inp)
    cr, cc = info["center"]
    expected = _oracle_output(inp)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(inp, out):
        return False

    first_offsets = [
        info["horizontal"]["negative"][0],
        info["horizontal"]["positive"][0],
        info["vertical"]["negative"][0],
        info["vertical"]["positive"][0],
    ]
    if _terminal_cycle_signature(first_offsets) != {TEAL, YELLOW}:
        return False

    for dr, dc in DIAGONAL_DIRS:
        r = cr + dr
        c = cc + dc
        while _in_bounds(inp.shape, r, c):
            if out[r, c] != GREY:
                return False
            r += dr
            c += dc

    terminals = [
        (cr, cc - info["horizontal"]["negative"][0], GREEN),
        (cr, cc + info["horizontal"]["positive"][0], GREEN),
        (cr - info["vertical"]["negative"][0], cc, BLUE),
        (cr + info["vertical"]["positive"][0], cc, BLUE),
    ]
    recolored = set()
    unchanged_nonterminal_green = 0
    unchanged_nonterminal_blue = 0

    for r, c, original_color in terminals:
        if out[r, c] == original_color:
            return False
        recolored.add(int(out[r, c]))

    for c in np.argwhere(inp == GREEN):
        rr, cc0 = map(int, c)
        if rr == cr and abs(cc0 - cc) > info["horizontal"]["negative"][0] and cc0 < cc:
            unchanged_nonterminal_green += int(out[rr, cc0] == GREEN)
        if rr == cr and abs(cc0 - cc) > info["horizontal"]["positive"][0] and cc0 > cc:
            unchanged_nonterminal_green += int(out[rr, cc0] == GREEN)
    for c in np.argwhere(inp == BLUE):
        rr, cc0 = map(int, c)
        if cc0 == cc and abs(rr - cr) > info["vertical"]["negative"][0] and rr < cr:
            unchanged_nonterminal_blue += int(out[rr, cc0] == BLUE)
        if cc0 == cc and abs(rr - cr) > info["vertical"]["positive"][0] and rr > cr:
            unchanged_nonterminal_blue += int(out[rr, cc0] == BLUE)

    if recolored != {TEAL, YELLOW}:
        return False
    if unchanged_nonterminal_green < 2 or unchanged_nonterminal_blue < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    height, width, center, step_h, step_v, left, right, up, down = layout
    inp = _build_input(height, width, center, step_h, step_v, left, right, up, down)
    out = main(inp)
    if not _semantic_check(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-track sample")

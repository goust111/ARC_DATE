from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6

DIRECTION_BY_COLOR: dict[int, tuple[int, int]] = {
    BLUE: (0, 1),    # right
    RED: (1, 0),     # down
    GREEN: (0, -1),  # left
    YELLOW: (-1, 0), # up
}
DIRECTION_COLORS = tuple(DIRECTION_BY_COLOR.keys())

MIN_INSTRUCTIONS = 6
MAX_INSTRUCTIONS = 9
MAX_GENERATION_ATTEMPTS = 500
PROGRAM_ATTEMPTS = 300


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(w):
        col = grid[:, c]
        if bool(np.all(col == int(col[0]))) and int(col[0]) != int(bg):
            candidates.append(int(c))
    assert len(candidates) == 1, "expected one full-height separator column"
    return candidates[0]


def _parse_program(left: np.ndarray, bg: int) -> list[tuple[int, int]]:
    instructions: list[tuple[int, int]] = []
    for r in range(left.shape[0]):
        cols = np.flatnonzero(left[r] != int(bg))
        if len(cols) == 0:
            continue
        assert len(cols) == 2, "each instruction row must have exactly two markers"
        c0, c1 = map(int, cols)
        color0 = int(left[r, c0])
        color1 = int(left[r, c1])
        assert color0 == color1, "marker pair must be monochromatic"
        assert color0 in DIRECTION_BY_COLOR, "marker color must be a direction color"
        length = int(c1 - c0 - 1)
        assert length > 0, "marker gap must encode a positive length"
        instructions.append((color0, length))
    assert instructions, "expected at least one instruction row"
    return instructions


def _execute_program(
    shape: tuple[int, int],
    start: tuple[int, int],
    program: list[tuple[int, int]],
    bg: int = BLACK,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r, c = map(int, start)
    h, w = shape
    assert 0 <= r < h and 0 <= c < w, "start marker must lie in the right panel"
    out[r, c] = GRAY
    start_cell = (int(r), int(c))

    for color, length in program:
        dr, dc = DIRECTION_BY_COLOR[int(color)]
        for _ in range(int(length)):
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "drawn segment exits the right panel"
            if (int(r), int(c)) != start_cell:
                out[r, c] = int(color)
    out[start_cell] = GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode marker-pair drawing instructions and execute them from the gray start marker.

    Input structure:
    - A full-height separator column splits the grid into a left program panel and a right drawing panel.
    - Each non-empty row of the left panel has exactly two same-colored marker pixels; their color is one
      of four direction colors, and their horizontal gap is positive.
    - The right panel is otherwise background and contains exactly one gray start pixel.

    Rule:
    - Read the left-panel marker rows from top to bottom.
    - For each row, use the number of cells between the two markers as the segment length, and decode
      the marker color as direction: blue=right, red=down, green=left, yellow=up.
    - Starting at the gray pixel in the right panel, draw each segment in order using that row's marker
      color, then continue the next segment from the last endpoint; the original gray start remains gray.
    - Return only the completed right panel, omitting the left program panel and separator column.

    Key insight:
    - The marker pair is not a gap to fill: it is a ruler that writes a numeric instruction, while the
      right panel is the separate execution canvas for the decoded polyline program.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background(g)
    sep_c = _find_separator_col(g, bg)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert left.size and right.size, "separator must leave non-empty panels"

    starts = np.argwhere(right == GRAY)
    assert starts.shape[0] == 1, "right panel must contain one gray start marker"
    start = tuple(map(int, starts[0]))

    program = _parse_program(left, bg)
    return _execute_program(right.shape, start, program, bg=bg)


def _program_stats_ok(program: list[tuple[int, int]]) -> bool:
    colors = [color for color, _length in program]
    lengths = [length for _color, length in program]
    if set(colors) != set(DIRECTION_COLORS):
        return False

    same_color_diff_len = False
    for color in DIRECTION_COLORS:
        vals = {length for c, length in program if c == color}
        same_color_diff_len |= len(vals) >= 2

    same_len_diff_color = False
    for length in set(lengths):
        vals = {color for color, l in program if l == length}
        same_len_diff_color |= len(vals) >= 2

    turns = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i - 1])
    return same_color_diff_len and same_len_diff_color and turns >= 3


def _trace_relative(program: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    r = c = 0
    visited = {(0, 0)}
    coords = [(0, 0)]
    for color, length in program:
        dr, dc = DIRECTION_BY_COLOR[int(color)]
        for _ in range(int(length)):
            r += dr
            c += dc
            if (r, c) in visited:
                return None
            visited.add((r, c))
            coords.append((r, c))

    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return coords, (min(rs), min(cs), max(rs), max(cs))


def _sample_program(max_len: int) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    for _ in range(PROGRAM_ATTEMPTS):
        n = random.randint(MIN_INSTRUCTIONS, MAX_INSTRUCTIONS)
        duplicate = random.choice(DIRECTION_COLORS)
        colors = list(DIRECTION_COLORS) + [duplicate]
        while len(colors) < n:
            colors.append(random.choice(DIRECTION_COLORS))
        random.shuffle(colors)

        lengths = [random.randint(1, max_len) for _ in colors]

        dup_idxs = [i for i, color in enumerate(colors) if color == duplicate]
        if len(dup_idxs) >= 2 and max_len >= 2:
            a, b = random.sample(range(1, max_len + 1), 2)
            lengths[dup_idxs[0]] = a
            lengths[dup_idxs[1]] = b

        diff_color_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if colors[i] != colors[j]]
        if diff_color_pairs:
            i, j = random.choice(diff_color_pairs)
            common = random.randint(1, max_len)
            lengths[i] = common
            lengths[j] = common

        program = list(zip(colors, lengths))
        if not _program_stats_ok(program):
            continue
        traced = _trace_relative(program)
        if traced is None:
            continue
        _coords, bbox = traced
        return program, bbox
    return None


def _render_expected(shape: tuple[int, int], start: tuple[int, int], program: list[tuple[int, int]]) -> np.ndarray:
    return _execute_program(shape, start, program, bg=BLACK)


def _render_input(
    h: int,
    left_w: int,
    right_w: int,
    start: tuple[int, int],
    rows: list[int],
    program: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, left_w + 1 + right_w), BLACK, dtype=int)
    grid[:, left_w] = PINK
    sr, sc = map(int, start)
    grid[sr, left_w + 1 + sc] = GRAY

    for row, (color, length) in zip(rows, program):
        c0 = random.randint(0, left_w - int(length) - 2)
        c1 = c0 + int(length) + 1
        grid[int(row), c0] = int(color)
        grid[int(row), c1] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    left_w = random.randint(7, 12)
    max_len = min(5, left_w - 2)
    sampled = _sample_program(max_len)
    if sampled is None:
        return None
    program, (min_r, min_c, max_r, max_c) = sampled

    path_h = int(max_r - min_r + 1)
    path_w = int(max_c - min_c + 1)
    right_w_min = max(8, path_w + 2)
    right_w_max = min(16, 30 - left_w - 1)
    h_min = max(10, len(program) + 1, path_h + 2)
    h_max = 24
    if right_w_min > right_w_max or h_min > h_max:
        return None

    h = random.randint(h_min, h_max)
    right_w = random.randint(right_w_min, right_w_max)

    sr_low = -int(min_r)
    sr_high = h - 1 - int(max_r)
    sc_low = -int(min_c)
    sc_high = right_w - 1 - int(max_c)
    if sr_low < sr_high:
        sr_low += 1
        sr_high -= 1
    if sc_low < sc_high:
        sc_low += 1
        sc_high -= 1
    if sr_low > sr_high or sc_low > sc_high:
        return None

    start = (random.randint(sr_low, sr_high), random.randint(sc_low, sc_high))
    rows = sorted(random.sample(range(h), len(program)))
    grid = _render_input(h, left_w, right_w, start, rows, program)
    expected = _render_expected((h, right_w), start, program)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    parsed = _parse_program(grid[:, :left_w], BLACK)
    if parsed != program or not _program_stats_ok(parsed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")

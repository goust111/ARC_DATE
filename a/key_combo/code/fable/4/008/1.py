from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GEN_ATTEMPTS = 600


def _read_tape(grid: np.ndarray) -> list[tuple[int, int]]:
    assert grid.ndim == 2 and min(grid.shape) > 0, "expected a non-empty 2D grid"
    assert grid.shape[1] >= 2, "seed area must exist to the right of the tape"

    tape = [int(v) for v in grid[:, 0]]
    assert all(v != BLACK for v in tape), "left edge tape must be visible"

    runs: list[tuple[int, int]] = []
    color = tape[0]
    length = 1
    for value in tape[1:]:
        if value == color:
            length += 1
        else:
            runs.append((int(color), int(length)))
            color = value
            length = 1
    runs.append((int(color), int(length)))
    return runs


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    body = grid[:, 1:]
    coords = np.argwhere(body == GREY)
    assert coords.shape[0] == 1, "expected exactly one grey seed outside the tape"
    r, c = coords[0]
    return int(r), int(c + 1)


def _assert_clean_input(grid: np.ndarray, seed: tuple[int, int]) -> None:
    body = grid[:, 1:].copy()
    sr, sc = seed
    body[sr, sc - 1] = BLACK
    assert np.all(body == BLACK), "outside the tape only the grey seed may be present"


def _draw_from_instructions(
    grid: np.ndarray,
    instructions: list[tuple[int, int]],
    seed: tuple[int, int],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape
    r, c = seed
    seen = {(int(r), int(c))}

    for i, (color, length) in enumerate(instructions):
        dr, dc = DIRECTIONS[i % len(DIRECTIONS)]
        for _ in range(int(length)):
            r += dr
            c += dc
            assert 0 <= r < height and 0 <= c < width, "polyline must stay in bounds"
            assert c != 0, "polyline must not intersect the tape"
            assert (int(r), int(c)) not in seen, "polyline must visit new cells only"
            assert int(out[r, c]) == BLACK, "polyline may only draw on blank cells"
            out[r, c] = int(color)
            seen.add((int(r), int(c)))

    sr, sc = seed
    out[sr, sc] = GREY
    assert np.array_equal(out[:, 0], grid[:, 0]), "tape must remain unchanged"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left-edge stripe tape as drawing instructions and draw a clockwise polyline from the grey seed.

    Input structure:
    - The background is black, and the full leftmost column is a visible vertical stripe tape made of consecutive same-color runs.
    - Outside that tape there is exactly one grey seed pixel and no other non-black content.
    - The encoded path is guaranteed to stay in bounds, avoid the tape, and visit new cells only.

    Rule:
    - Compress the left column from top to bottom into ordered runs; each run supplies a drawing color and a step length equal to its thickness.
    - Starting at the grey seed, draw the first run upward, then turn clockwise by ninety degrees for each next run: right, down, left, and repeat.
    - For each run, advance one cell at a time and color exactly those newly visited cells with the run color, while leaving the seed grey and the tape unchanged.

    Key insight:
    - The left-edge stripe is a run-length encoded instruction tape, not a target pattern; its colors are inks and its run thicknesses are movement lengths.
    """
    g = np.asarray(grid, dtype=int)
    instructions = _read_tape(g)
    seed = _seed_coord(g)
    _assert_clean_input(g, seed)
    return _draw_from_instructions(g, instructions, seed)


def _trace_relative(lengths: list[int]) -> list[list[tuple[int, int]]] | None:
    r = 0
    c = 0
    seen = {(0, 0)}
    segments: list[list[tuple[int, int]]] = []

    for i, length in enumerate(lengths):
        dr, dc = DIRECTIONS[i % len(DIRECTIONS)]
        segment: list[tuple[int, int]] = []
        for _ in range(int(length)):
            r += dr
            c += dc
            cell = (int(r), int(c))
            if cell in seen:
                return None
            seen.add(cell)
            segment.append(cell)
        segments.append(segment)

    return segments


def _random_lengths(n_segments: int) -> list[int] | None:
    for _ in range(250):
        lengths = [random.randint(1, 6) for _ in range(n_segments)]
        total = sum(lengths)
        if not 9 <= total <= 29:
            continue
        if max(lengths) < 2 or len(set(lengths)) < 2:
            continue
        if _trace_relative(lengths) is None:
            continue
        return lengths

    first = random.randint(1, 2)
    second = random.randint(1, 2)
    lengths = [first, second]
    for i in range(2, n_segments):
        lengths.append(lengths[i - 2] + 1)
    if sum(lengths) <= 29 and _trace_relative(lengths) is not None:
        return lengths
    return None


def _random_colors(n_segments: int) -> list[int]:
    for _ in range(100):
        colors: list[int] = []
        for _i in range(n_segments):
            options = [c for c in DRAW_COLORS if not colors or c != colors[-1]]
            colors.append(int(random.choice(options)))
        if len(set(colors)) >= min(3, n_segments):
            return colors
    return list(DRAW_COLORS[:n_segments])


def _bounds_for_segments(
    segments: list[list[tuple[int, int]]],
) -> tuple[int, int, int, int]:
    cells = [(0, 0)] + [cell for segment in segments for cell in segment]
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _place_seed(
    height: int,
    width: int,
    segments: list[list[tuple[int, int]]],
) -> tuple[int, int] | None:
    min_r, max_r, min_c, max_c = _bounds_for_segments(segments)
    r_low = -min_r
    r_high = height - 1 - max_r
    c_low = 2 - min_c
    c_high = width - 2 - max_c
    if r_low > r_high or c_low > c_high:
        return None
    return int(random.randint(r_low, r_high)), int(random.randint(c_low, c_high))


def _render_input(
    height: int,
    width: int,
    colors: list[int],
    lengths: list[int],
    seed: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    r0 = 0
    for color, length in zip(colors, lengths):
        grid[r0 : r0 + int(length), 0] = int(color)
        r0 += int(length)
    sr, sc = seed
    grid[sr, sc] = GREY
    return grid


def _absolute_segments(
    seed: tuple[int, int],
    relative_segments: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    sr, sc = seed
    return [
        [(int(sr + r), int(sc + c)) for r, c in segment]
        for segment in relative_segments
    ]


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if max(inp.shape) > 30:
        return False
    if not np.array_equal(inp[:, 0], out[:, 0]):
        return False

    try:
        instructions = _read_tape(inp)
        seed = _seed_coord(inp)
        _assert_clean_input(inp, seed)
    except AssertionError:
        return False

    colors = [color for color, _length in instructions]
    lengths = [length for _color, length in instructions]
    if len(instructions) < 4:
        return False
    if len(set(colors)) < 3 or len(set(lengths)) < 2 or max(lengths) < 2:
        return False

    relative = _trace_relative(lengths)
    if relative is None:
        return False
    abs_segments = _absolute_segments(seed, relative)
    path_cells = {cell for segment in abs_segments for cell in segment}
    if len(path_cells) != sum(lengths):
        return False

    sr, sc = seed
    if int(inp[sr, sc]) != GREY or int(out[sr, sc]) != GREY:
        return False
    for r, c in path_cells:
        if c == 0 or int(inp[r, c]) != BLACK:
            return False

    diff_cells = {(int(r), int(c)) for r, c in np.argwhere(inp != out)}
    if diff_cells != path_cells:
        return False

    for (color, length), segment in zip(instructions, abs_segments):
        if len(segment) != int(length):
            return False
        if any(int(out[r, c]) != int(color) for r, c in segment):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(4, 8)
    lengths = _random_lengths(n_segments)
    if lengths is None:
        return None

    relative = _trace_relative(lengths)
    if relative is None:
        return None

    height = int(sum(lengths))
    _min_r, _max_r, min_c, max_c = _bounds_for_segments(relative)
    width_needed = int(max_c - min_c + 4)
    if height > 30 or width_needed > 30:
        return None

    width = random.randint(max(8, width_needed), 30)
    seed = _place_seed(height, width, relative)
    if seed is None:
        return None

    colors = _random_colors(n_segments)
    inp = _render_input(height, width, colors, lengths, seed)
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction tape polyline input")

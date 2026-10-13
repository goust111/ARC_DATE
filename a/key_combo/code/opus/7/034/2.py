from __future__ import annotations

import random

import numpy as np


OUTPUT_CYCLE = (3, 5, 9)
INPUT_COLORS = (1, 2, 4, 6, 7, 8)
PAIR_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (12, 16)
GRID_WIDTH_RANGE = (12, 16)
STEP_LENGTH_MIN = 2
STEP_LENGTH_MAX = 7
GENERATION_ATTEMPTS = 500


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "endpoints must define a strict 45-degree diagonal"

    step_r = _sign(dr)
    step_c = _sign(dc)
    return [(r0 + k * step_r, c0 + k * step_c) for k in range(abs(dr) + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored diagonal endpoint pair, then recolor the completed segments by length rank.

    Input structure:
    - The grid uses black (0) as background.
    - There are 3-5 non-black colors, and each color appears exactly twice as isolated single pixels.
    - Each same-colored pair forms the endpoints of a strict 45-degree diagonal, and different pairs do not share cells.

    Rule:
    - For every non-black color, connect its two endpoints by filling the full diagonal segment, including endpoints.
    - Measure each segment by its pixel length.
    - Sort the segments from shortest to longest.
    - Recolor the shortest, second-shortest, third-shortest, ... segments with the repeating palette green (3), grey (5), maroon (9).
    - Leave all background cells black.

    Key insight:
    - The input colors matter only for pairing endpoints; the output colors ignore the original labels and depend only on the rank order of segment lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert 3 <= len(colors) <= 5, "expected 3-5 endpoint pairs"

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        cells = _segment_cells(a, b)
        segments.append((len(cells), cells))

    lengths = [length for length, _cells in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be unique"

    out = np.zeros_like(g)
    for index, (_length, cells) in enumerate(sorted(segments, key=lambda item: item[0])):
        new_color = OUTPUT_CYCLE[index % len(OUTPUT_CYCLE)]
        for r, c in cells:
            assert out[r, c] == 0, "segments must not overlap"
            out[r, c] = new_color

    return out


def _segment_candidates(shape: tuple[int, int], step_length: int, step_c: int) -> list[list[tuple[int, int]]]:
    height, width = shape
    candidates = []
    for r0 in range(height - step_length):
        for c0 in range(width):
            c1 = c0 + step_c * step_length
            if not (0 <= c1 < width):
                continue
            cells = [(r0 + k, c0 + k * step_c) for k in range(step_length + 1)]
            candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _can_place(cells: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _mark_blocked(cells: list[tuple[int, int]], blocked: np.ndarray) -> None:
    height, width = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < height and 0 <= nc < width:
                blocked[nr, nc] = True


def _render_output(shape: tuple[int, int], placed: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked = sorted(((len(cells), cells) for _color, cells in placed), key=lambda item: item[0])
    for index, (_length, cells) in enumerate(ranked):
        color = OUTPUT_CYCLE[index % len(OUTPUT_CYCLE)]
        rows, cols = np.array(cells, dtype=int).T
        out[rows, cols] = color
    return out


def _build_input(shape: tuple[int, int], placed: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for color, cells in placed:
        endpoints = [cells[0], cells[-1]]
        rows, cols = np.array(endpoints, dtype=int).T
        inp[rows, cols] = int(color)
    return inp


def _valid_input(grid: np.ndarray) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    if not (3 <= len(colors) <= 5):
        return False

    height, width = grid.shape
    for color in colors:
        coords = np.argwhere(grid == color)
        if len(coords) != 2:
            return False
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) != 0:
                    return False
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if dr == 0 or abs(dr) != abs(dc):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    max_step = min(STEP_LENGTH_MAX, height - 3, width - 3)
    if max_step - STEP_LENGTH_MIN + 1 < pair_count:
        return None

    step_lengths = random.sample(range(STEP_LENGTH_MIN, max_step + 1), pair_count)
    colors = random.sample(INPUT_COLORS, pair_count)
    specs = list(zip(colors, step_lengths, [random.choice((-1, 1)) for _ in range(pair_count)]))
    specs.sort(key=lambda item: item[1], reverse=True)

    blocked = np.zeros((height, width), dtype=bool)
    placed: list[tuple[int, list[tuple[int, int]]]] = []
    for color, step_length, step_c in specs:
        cells = None
        for candidate in _segment_candidates((height, width), step_length, step_c):
            if _can_place(candidate, blocked):
                cells = candidate
                break
        if cells is None:
            return None
        _mark_blocked(cells, blocked)
        placed.append((int(color), cells))

    inp = _build_input((height, width), placed)
    out = _render_output((height, width), placed)
    if not _valid_input(inp):
        return None
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

from __future__ import annotations

import random

import numpy as np


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
MAIN_DIAG = "main_diag"
ANTI_DIAG = "anti_diag"

RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4

DIRECTION_TO_COLOR = {
    HORIZONTAL: RED,
    VERTICAL: BLUE,
    MAIN_DIAG: GREEN,
    ANTI_DIAG: YELLOW,
}

NEUTRAL_COLORS = (0, 5, 6, 7, 8, 9)
DIRECTIONS = (HORIZONTAL, VERTICAL, MAIN_DIAG, ANTI_DIAG)

HEIGHT_BOUNDS = (12, 20)
WIDTH_BOUNDS = (12, 20)
PAIR_COUNT_BOUNDS = (4, 5)
MIN_SEGMENT_LENGTH = 2
MAX_SEGMENT_LENGTH = 8
BUFFER_RADIUS = 1
PAIR_PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _direction_and_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[str, list[tuple[int, int]]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 or dc != 0, "pair endpoints must be distinct"

    if dr == 0:
        direction = HORIZONTAL
    elif dc == 0:
        direction = VERTICAL
    elif abs(dr) == abs(dc) and r0 - c0 == r1 - c1:
        direction = MAIN_DIAG
    elif abs(dr) == abs(dc) and r0 + c0 == r1 + c1:
        direction = ANTI_DIAG
    else:
        raise AssertionError("each pair must be horizontal, vertical, or diagonal")

    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    length = max(abs(dr), abs(dc)) + 1
    cells = [(r0 + i * step_r, c0 + i * step_c) for i in range(length)]
    return direction, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each same-color pixel pair by line direction and connect it using a fixed direction color.

    Input structure:
    - The background is the dominant color.
    - Every non-background color appears exactly twice, forming one pixel pair.
    - Each pair lies on one of four directions: horizontal, vertical, main diagonal, or anti-diagonal.

    Rule:
    - Replace each original pair with the full inclusive segment between its two pixels.
    - Paint horizontal pairs red, vertical pairs blue, main-diagonal pairs green, and anti-diagonal pairs yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = np.full(g.shape, bg, dtype=int)
    painted: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        direction, cells = _direction_and_cells(a, b)
        paint = DIRECTION_TO_COLOR[direction]

        for cell in cells:
            prev = painted.get(cell)
            assert prev in (None, paint), "output lines must not overlap ambiguously"
            painted[cell] = paint
            out[cell] = paint

    return out


def _sample_segment(height: int, width: int, direction: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == HORIZONTAL:
        max_len = min(MAX_SEGMENT_LENGTH, width)
        if max_len < MIN_SEGMENT_LENGTH:
            return None
        length = random.randint(MIN_SEGMENT_LENGTH, max_len)
        r = random.randrange(height)
        c0 = random.randint(0, width - length)
        return (r, c0), (r, c0 + length - 1)

    if direction == VERTICAL:
        max_len = min(MAX_SEGMENT_LENGTH, height)
        if max_len < MIN_SEGMENT_LENGTH:
            return None
        length = random.randint(MIN_SEGMENT_LENGTH, max_len)
        c = random.randrange(width)
        r0 = random.randint(0, height - length)
        return (r0, c), (r0 + length - 1, c)

    max_len = min(MAX_SEGMENT_LENGTH, height, width)
    if max_len < MIN_SEGMENT_LENGTH:
        return None
    length = random.randint(MIN_SEGMENT_LENGTH, max_len)

    if direction == MAIN_DIAG:
        r0 = random.randint(0, height - length)
        c0 = random.randint(0, width - length)
        return (r0, c0), (r0 + length - 1, c0 + length - 1)

    if direction == ANTI_DIAG:
        r0 = random.randint(0, height - length)
        c0 = random.randint(length - 1, width - 1)
        return (r0, c0), (r0 + length - 1, c0 - length + 1)

    raise AssertionError("unknown direction")


def _halo(cells: list[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in cells:
        for dr in range(-BUFFER_RADIUS, BUFFER_RADIUS + 1):
            for dc in range(-BUFFER_RADIUS, BUFFER_RADIUS + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked.add((rr, cc))
    return blocked


def _expected_output(
    height: int,
    width: int,
    bg: int,
    placements: list[tuple[str, tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for direction, a, b, _pair_color in placements:
        _, cells = _direction_and_cells(a, b)
        out[tuple(zip(*cells))] = DIRECTION_TO_COLOR[direction]
    return out


def _build_input(
    height: int,
    width: int,
    bg: int,
    placements: list[tuple[str, tuple[int, int], tuple[int, int], int]],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for _direction, a, b, pair_color in placements:
        grid[a] = pair_color
        grid[b] = pair_color
    return grid


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*HEIGHT_BOUNDS)
    width = random.randint(*WIDTH_BOUNDS)
    bg = random.choice(NEUTRAL_COLORS)

    pair_count = random.randint(*PAIR_COUNT_BOUNDS)
    pair_colors = random.sample([c for c in NEUTRAL_COLORS if c != bg], pair_count)

    directions = list(DIRECTIONS)
    while len(directions) < pair_count:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)

    placements: list[tuple[str, tuple[int, int], tuple[int, int], int]] = []
    blocked: set[tuple[int, int]] = set()

    for direction, pair_color in zip(directions, pair_colors):
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            segment = _sample_segment(height, width, direction)
            if segment is None:
                return None
            a, b = segment
            _, cells = _direction_and_cells(a, b)
            if any(cell in blocked for cell in cells):
                continue
            placements.append((direction, a, b, pair_color))
            blocked.update(_halo(cells, height, width))
            placed = True
            break
        if not placed:
            return None

    inp = _build_input(height, width, bg, placements)
    expected = _expected_output(height, width, bg, placements)

    if np.count_nonzero(inp != bg) != 2 * pair_count:
        return None
    if np.count_nonzero(expected != bg) <= np.count_nonzero(inp != bg):
        return None
    if np.count_nonzero(inp == bg) <= np.count_nonzero(inp != bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")

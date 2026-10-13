from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0

# Fixed operation codebook shared by every generated example.
ROTATE_CW = 1
MIRROR_LEFT_RIGHT = 2
MIRROR_UP_DOWN = 3
OP_CODES = (ROTATE_CW, MIRROR_LEFT_RIGHT, MIRROR_UP_DOWN)

PATTERN_COLORS = (4, 5, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 700
PATTERN_ATTEMPTS = 160
PROGRAM_CYCLE = (
    (ROTATE_CW, MIRROR_LEFT_RIGHT),
    (MIRROR_UP_DOWN, ROTATE_CW),
    (ROTATE_CW, MIRROR_LEFT_RIGHT, MIRROR_UP_DOWN, MIRROR_LEFT_RIGHT),
    (ROTATE_CW, MIRROR_UP_DOWN, MIRROR_LEFT_RIGHT, MIRROR_UP_DOWN),
    (MIRROR_LEFT_RIGHT, ROTATE_CW, MIRROR_LEFT_RIGHT, MIRROR_UP_DOWN),
    (MIRROR_UP_DOWN, ROTATE_CW, MIRROR_UP_DOWN, MIRROR_LEFT_RIGHT),
)

_PROGRAM_QUEUE: list[tuple[int, ...]] = []
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _instruction_strip(row: np.ndarray, bg: int) -> tuple[int, ...]:
    cols = np.flatnonzero(row != int(bg))
    assert 2 <= len(cols) <= 4, "instruction strip must contain two to four cells"
    assert np.all(np.diff(cols) == 1), "instruction strip must be one contiguous run"

    program = tuple(int(row[c]) for c in cols)
    assert all(code in OP_CODES for code in program), "unknown instruction color"
    return program


def _extract_pattern_and_program(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4, "grid is too small for a pattern and instruction strip"

    separator = g[-2]
    sep_colors = np.unique(separator)
    assert len(sep_colors) == 1, "the row above the strip must be a solid background separator"
    bg = int(sep_colors[0])

    program = _instruction_strip(g[-1], bg)
    assert np.all((g[-1] == bg) | np.isin(g[-1], OP_CODES)), "strip row may contain only background and instructions"

    upper = g[:-2]
    box = _bbox(upper != bg)
    assert box is not None, "expected a visible main pattern above the strip"
    r0, c0, r1, c1 = box
    pattern = upper[r0 : r1 + 1, c0 : c1 + 1].copy()
    ph, pw = map(int, pattern.shape)
    assert ph == pw and ph >= 2, "main pattern bbox must be square"

    visible_colors = set(map(int, np.unique(pattern))) - {bg}
    assert len(visible_colors) >= 2, "main pattern must be multicolor"
    return pattern, program


def _apply_op(pattern: np.ndarray, code: int) -> np.ndarray:
    if int(code) == ROTATE_CW:
        return np.rot90(pattern, -1).copy()
    if int(code) == MIRROR_LEFT_RIGHT:
        return np.fliplr(pattern).copy()
    if int(code) == MIRROR_UP_DOWN:
        return np.flipud(pattern).copy()
    raise AssertionError("unknown operation code")


def _apply_program(pattern: np.ndarray, program: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(pattern, dtype=int).copy()
    for code in program:
        out = _apply_op(out, int(code))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the bottom instruction strip as an ordered sequence of whole-pattern geometric operations.

    Input structure:
    - A multicolor square main pattern appears above a one-row solid background separator.
    - The bottom row contains one contiguous instruction strip of two to four colored cells,
      with any remaining bottom cells left as background.
    - The fixed codebook is blue = rotate 90 degrees clockwise, red = mirror left-right,
      and green = mirror up-down.

    Rule:
    - Crop the square main pattern from the area above the separator.
    - Read the instruction strip from left to right.
    - Apply each decoded whole-grid operation in that order to the cropped pattern.
    - Return only the transformed square pattern, without the separator or instruction strip.

    Key insight:
    - The colored strip is an ordered program, not a palette or a count marker; changing the
      reading order changes the composed geometric transform on an asymmetric pattern.
    """
    pattern, program = _extract_pattern_and_program(grid)
    return _apply_program(pattern, program)


def _next_program() -> tuple[int, ...]:
    global _PROGRAM_QUEUE
    if not _PROGRAM_QUEUE:
        _PROGRAM_QUEUE = list(PROGRAM_CYCLE)
    return _PROGRAM_QUEUE.pop(0)


def _forced_cells(n: int) -> list[tuple[int, int]]:
    return [
        (0, 1),
        (1, n - 1),
        (n - 1, max(0, n - 2)),
        (max(0, n - 2), 0),
        (n // 2, max(0, n // 2 - 1)),
    ]


def _grow_positions(n: int, target: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    positions = set(_forced_cells(n))
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))

    attempts = 0
    while len(positions) < int(target) and attempts < 600:
        attempts += 1
        if positions and rng.random() < 0.68:
            r, c = random.choice(tuple(positions))
            dr, dc = moves[int(rng.integers(0, len(moves)))]
            rr, cc = r + dr, c + dc
        else:
            rr = int(rng.integers(0, n))
            cc = int(rng.integers(0, n))

        if 0 <= rr < n and 0 <= cc < n:
            positions.add((int(rr), int(cc)))

    return positions


def _random_pattern(n: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        grid = np.full((n, n), BACKGROUND, dtype=int)
        color_count = int(rng.integers(3, min(5, len(PATTERN_COLORS)) + 1))
        palette = random.sample(PATTERN_COLORS, color_count)

        min_cells = max(7, n + 2)
        max_cells = max(min_cells, int(round(n * n * rng.uniform(0.34, 0.52))))
        max_cells = min(max_cells, n * n - 2)
        if min_cells > max_cells:
            continue

        target = int(rng.integers(min_cells, max_cells + 1))
        positions = _grow_positions(n, target, rng)
        if len(positions) < min_cells:
            continue

        forced = _forced_cells(n)
        for i, cell in enumerate(forced):
            r, c = cell
            grid[r, c] = int(palette[i % len(palette)])

        remaining = [cell for cell in positions if cell not in forced]
        random.shuffle(remaining)
        for i, (r, c) in enumerate(remaining):
            if i < len(palette):
                color = palette[i]
            else:
                color = int(rng.choice(palette))
            grid[r, c] = int(color)

        # The visible bbox must be the entire square and the pattern must remain sparse enough
        # that the surrounding background is a clear separate object in the full input.
        if _bbox(grid != BACKGROUND) != (0, 0, n - 1, n - 1):
            continue
        if len(set(map(int, np.unique(grid))) - {BACKGROUND}) < 3:
            continue
        return grid
    return None


def _semantically_strong(pattern: np.ndarray, program: tuple[int, ...]) -> bool:
    full = _apply_program(pattern, program)

    if np.array_equal(full, pattern):
        return False

    # Every primitive operation must visibly affect this pattern.
    for code in OP_CODES:
        if np.array_equal(_apply_op(pattern, code), pattern):
            return False

    # The program must be an ordered composition, not a removable list of decorations.
    if program != tuple(reversed(program)) and np.array_equal(full, _apply_program(pattern, tuple(reversed(program)))):
        return False

    for i in range(len(program)):
        shortened = program[:i] + program[i + 1 :]
        if np.array_equal(full, _apply_program(pattern, shortened)):
            return False

    sensitive_pair = False
    for i in range(len(program) - 1):
        if program[i] == program[i + 1]:
            return False
        swapped = list(program)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        swapped_out = _apply_program(pattern, tuple(swapped))
        if np.array_equal(full, swapped_out):
            if {program[i], program[i + 1]} != {MIRROR_LEFT_RIGHT, MIRROR_UP_DOWN}:
                return False
        else:
            sensitive_pair = True

    return sensitive_pair


def _compose_input(pattern: np.ndarray, program: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    n = int(pattern.shape[0])
    top_margin = int(rng.integers(0, 4))
    min_width = max(n, len(program) + 2)
    width = int(rng.integers(min_width, min(30, min_width + 10) + 1))
    height = top_margin + n + 2

    grid = np.full((height, width), BACKGROUND, dtype=int)
    left = int(rng.integers(0, width - n + 1))
    grid[top_margin : top_margin + n, left : left + n] = pattern

    strip_left = int(rng.integers(0, width - len(program) + 1))
    for i, code in enumerate(program):
        grid[-1, strip_left + i] = int(code)
    return grid


def _gen_attempt(program: tuple[int, ...], rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(4, 10))
    pattern = _random_pattern(n, rng)
    if pattern is None:
        return None
    if not _semantically_strong(pattern, program):
        return None

    grid = _compose_input(pattern, program, rng)
    if max(grid.shape) > 30:
        return None
    if _most_common(grid) != BACKGROUND:
        return None

    expected = _apply_program(pattern, program)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    out_sig = (tuple(map(int, expected.shape)), expected.tobytes())
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    program = _next_program()
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(program, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-strip program input")

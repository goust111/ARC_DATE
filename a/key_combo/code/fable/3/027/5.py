from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
COLOR_TO_DIR = {
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}
LETTER_TO_DIR = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_TO_LETTER = {v: k for k, v in LETTER_TO_DIR.items()}
DIR_TO_COLOR = {v: k for k, v in COLOR_TO_DIR.items()}

BASE_PERIODS = (
    "UURULLDL",
    "LLULLDRDD",
    "LDLDDRDLLU",
)
DIHEDRAL_MATRICES = (
    (1, 0, 0, 1),
    (-1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, -1, -1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
)
MARKER_COLORS = (5, 6, 7, 8, 9)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _minimal_period(values: list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
    seq = [int(v) for v in values]
    assert seq, "period source must be nonempty"
    for p in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % p] for i in range(len(seq))):
            return tuple(seq[:p])
    raise AssertionError("unreachable minimal period state")


def _visible_tape_prefix(row: np.ndarray, bg: int) -> tuple[int, ...]:
    vals = [int(v) for v in row]
    assert vals and vals[0] != bg, "instruction tape must start at the left edge"
    first_bg = next((i for i, v in enumerate(vals) if v == bg), len(vals))
    assert 0 < first_bg < len(vals), "instruction tape must have a hidden background suffix"
    assert all(v == bg for v in vals[first_bg:]), "visible tape prefix must be contiguous"
    return tuple(vals[:first_bg])


def _complete_tape(prefix: tuple[int, ...], width: int) -> np.ndarray:
    period = _minimal_period(prefix)
    assert all(color in COLOR_TO_DIR for color in period), "unknown instruction color"
    return np.array([period[i % len(period)] for i in range(width)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the periodic instruction tape, then draw the tape-driven path from the start marker.

    Input structure:
    - The background is the dominant color. The bottom row is an instruction tape whose left
      contiguous prefix is colored and whose remaining right suffix is background.
    - The colored prefix is long enough for its own minimal repeating period to be the true
      program period. Instruction colors are the fixed direction codes shared by all examples.
    - Above the tape there is exactly one non-background start marker, and its color is not an
      instruction color used by the tape.

    Rule:
    - Recover the minimal period from the visible tape prefix and repeat it across the full width.
    - Starting at the marker, read the completed bottom row from left to right. For each instruction
      color, move one cell in its fixed cardinal direction and paint the reached cell with that
      instruction color.
    - The start marker keeps its original color. The completed instruction tape remains visible in
      the bottom row, and the path is required to stay above it, in bounds, and non-self-intersecting.

    Key insight:
    - The bottom stripe is a program, not just a decorative periodic pattern: period completion is
      the setup step, and every completed tape cell contributes one move to the drawn path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 4 and width >= 6, "grid too small for tape and path"

    bg = _get_bg(g)
    prefix = _visible_tape_prefix(g[-1, :], bg)
    full_tape = _complete_tape(prefix, width)
    period = _minimal_period(prefix)
    assert all(int(c) != bg for c in period), "instruction colors must not be background"

    marker_coords = np.argwhere(g[:-1, :] != bg)
    assert marker_coords.shape[0] == 1, "expected exactly one start marker above the tape"
    r, c = map(int, marker_coords[0])
    marker_color = int(g[r, c])
    assert marker_color not in set(period), "start marker color must be separate from tape colors"

    out = g.copy()
    out[-1, :] = full_tape
    seen = {(r, c)}
    for color in map(int, full_tape):
        dr, dc = COLOR_TO_DIR[color]
        r += dr
        c += dc
        assert 0 <= r < height - 1 and 0 <= c < width, "instruction path must stay above the tape"
        assert (r, c) not in seen, "instruction path must not self-intersect"
        assert int(out[r, c]) == bg, "path may only enter background cells"
        out[r, c] = color
        seen.add((r, c))

    return out


def _transform_letters(seq: str) -> str:
    a, b, c, d = random.choice(DIHEDRAL_MATRICES)
    transformed = []
    for letter in seq:
        dr, dc = LETTER_TO_DIR[letter]
        nr = a * dr + b * dc
        nc = c * dr + d * dc
        transformed.append(DIR_TO_LETTER[(nr, nc)])
    shift = random.randrange(len(transformed))
    transformed = transformed[shift:] + transformed[:shift]
    return "".join(transformed)


def _period_from_letters(letters: str) -> tuple[int, ...]:
    return tuple(DIR_TO_COLOR[LETTER_TO_DIR[ch]] for ch in letters)


def _trace_path(tape: list[int] | np.ndarray) -> list[tuple[int, int]] | None:
    r = c = 0
    points = [(0, 0)]
    seen = {(0, 0)}
    for color in map(int, tape):
        dr, dc = COLOR_TO_DIR[color]
        r += dr
        c += dc
        if (r, c) in seen:
            return None
        points.append((r, c))
        seen.add((r, c))
    return points


def _bbox_size(points: list[tuple[int, int]]) -> tuple[int, int, int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    return min_r, max_r, min_c, max_c, max_r - min_r + 1, max_c - min_c + 1


def _render_expected(inp: np.ndarray, full_tape: np.ndarray) -> np.ndarray:
    out = inp.copy()
    out[-1, :] = full_tape
    marker = np.argwhere(inp[:-1, :] != BACKGROUND)
    assert marker.shape[0] == 1
    r, c = map(int, marker[0])
    for color in map(int, full_tape):
        dr, dc = COLOR_TO_DIR[color]
        r += dr
        c += dc
        assert int(out[r, c]) == BACKGROUND
        out[r, c] = color
    return out


def _choose_period() -> tuple[int, ...]:
    for _ in range(100):
        letters = _transform_letters(random.choice(BASE_PERIODS))
        period = _period_from_letters(letters)
        if len(set(period)) == 4 and _minimal_period(period) == period:
            return period
    raise RuntimeError("failed to choose primitive instruction period")


def _gen_attempt() -> np.ndarray | None:
    period = _choose_period()
    p = len(period)
    width = random.randint(max(12, p + 5), 28)
    full_tape = np.array([period[i % p] for i in range(width)], dtype=int)

    points = _trace_path(full_tape)
    if points is None:
        return None
    min_r, max_r, min_c, max_c, path_h, path_w = _bbox_size(points)
    if path_h > 26 or path_w > width:
        return None

    extra_rows = random.randint(1, min(6, 29 - path_h))
    top_h = path_h + extra_rows
    height = top_h + 1

    start_r_lo = -min_r
    start_r_hi = top_h - 1 - max_r
    start_c_lo = -min_c
    start_c_hi = width - 1 - max_c
    if start_r_lo > start_r_hi or start_c_lo > start_c_hi:
        return None
    start_r = random.randint(start_r_lo, start_r_hi)
    start_c = random.randint(start_c_lo, start_c_hi)

    visible_choices = [
        n
        for n in range(p, width - 2)
        if _minimal_period(full_tape[:n]) == period and len(set(map(int, full_tape[n:]))) >= 2
    ]
    if not visible_choices:
        return None
    visible_len = random.choice(visible_choices)

    inp = np.full((height, width), BACKGROUND, dtype=int)
    inp[start_r, start_c] = random.choice(MARKER_COLORS)
    inp[-1, :visible_len] = full_tape[:visible_len]

    expected = _render_expected(inp, full_tape)

    # Semantic guardrails: both stages are necessary, all four codes are visible,
    # and hidden instructions add path cells that are not present in the input.
    assert _minimal_period(tuple(map(int, inp[-1, :visible_len]))) == period
    assert set(map(int, full_tape)) == set(COLOR_TO_DIR)
    assert visible_len < width
    assert np.any(expected[-1, visible_len:] != inp[-1, visible_len:])
    assert np.count_nonzero(expected[:-1, :] != inp[:-1, :]) == width

    hidden_points = points[visible_len + 1 :]
    assert hidden_points, "hidden tape suffix must drive extra path steps"
    assert len({tuple(pt) for pt in hidden_points}) == len(hidden_points)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

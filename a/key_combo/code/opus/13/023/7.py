from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GEN_RETRIES = 400


def _complete_arithmetic(indices: np.ndarray) -> np.ndarray:
    values = np.array(sorted({int(v) for v in indices}), dtype=int)
    assert values.ndim == 1 and len(values) >= 3
    diffs = np.diff(values)
    assert np.all(diffs > 0)
    step = int(diffs[0])
    for diff in diffs[1:]:
        step = math.gcd(step, int(diff))
    assert step >= 2
    start = int(values[0])
    end = int(values[-1])
    assert (end - start) % step == 0
    full = np.arange(start, end + 1, step, dtype=int)
    assert set(values).issubset(set(map(int, full)))
    return full


def _render_output(shape: tuple[int, int], full_rows: np.ndarray, full_cols: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[full_rows, :] = BLUE
    out[:, full_cols] = RED
    out[np.ix_(full_rows, full_cols)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the missing arithmetic bars and mark every completed crossing yellow.

    Input structure:
    - The grid has black background plus one-pixel-wide full-width blue horizontal bars and full-height red vertical bars.
    - Horizontal-bar row positions follow one arithmetic progression and vertical-bar column positions follow another.
    - In each progression, 1-2 internal members are missing while the first and last members remain visible.
    - Existing crossings may currently show blue or red depending on drawing order, but every visible bar still spans the full grid extent.

    Rule:
    - Find the rows that are entirely non-black and treat them as the observed horizontal bars.
    - Find the columns that are entirely non-black and treat them as the observed vertical bars.
    - For rows and columns separately, take the gcd of adjacent observed gaps as the progression step and fill every missing member between the observed minimum and maximum.
    - Draw the completed blue rows and red columns, then recolor every row-column intersection yellow.

    Key insight:
    - The visible crossings are not the target by themselves; they are evidence for two incomplete arithmetic bar families whose completion creates additional crossings that must all be marked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED})

    observed_rows = np.flatnonzero(np.all(g != BLACK, axis=1))
    observed_cols = np.flatnonzero(np.all(g != BLACK, axis=0))
    assert len(observed_rows) >= 3
    assert len(observed_cols) >= 3

    full_rows = _complete_arithmetic(observed_rows)
    full_cols = _complete_arithmetic(observed_cols)
    return _render_output(g.shape, full_rows, full_cols)


def _sequence_options(length: int) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    for count in range(4, 7):
        for step in range(2, 6):
            max_start = length - 2 - (count - 1) * step
            if max_start < 1:
                continue
            for start in range(1, max_start + 1):
                seq = np.arange(start, start + count * step, step, dtype=int)
                options.append(seq)
    return options


def _remove_internal_terms(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    internal = list(range(1, len(seq) - 1))
    assert internal
    max_missing = min(2, len(seq) - 3)
    missing_count = rng.randint(1, max_missing)
    missing = set(rng.sample(internal, missing_count))
    return np.array([int(v) for i, v in enumerate(seq) if i not in missing], dtype=int)


def _paint_input(
    shape: tuple[int, int],
    observed_rows: np.ndarray,
    observed_cols: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    mode = rng.choice(("mixed", "horizontal_top", "vertical_top"))

    actions = [("h", int(r)) for r in observed_rows] + [("v", int(c)) for c in observed_cols]
    if mode == "mixed":
        rng.shuffle(actions)
    elif mode == "horizontal_top":
        rng.shuffle(actions)
        actions.sort(key=lambda item: item[0] == "h")
    else:
        rng.shuffle(actions)
        actions.sort(key=lambda item: item[0] == "v")

    for kind, idx in actions:
        if kind == "h":
            grid[idx, :] = BLUE
        else:
            grid[:, idx] = RED
    return grid


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(13, 22)
    width = rng.randint(13, 22)

    row_options = _sequence_options(height)
    col_options = _sequence_options(width)
    if not row_options or not col_options:
        return None

    full_rows = rng.choice(row_options)
    full_cols = rng.choice(col_options)
    observed_rows = _remove_internal_terms(full_rows, rng)
    observed_cols = _remove_internal_terms(full_cols, rng)

    grid = _paint_input((height, width), observed_rows, observed_cols, rng)
    expected = _render_output((height, width), full_rows, full_cols)

    if np.array_equal(grid, expected):
        return None
    if len(np.flatnonzero(np.all(grid != BLACK, axis=1))) != len(observed_rows):
        return None
    if len(np.flatnonzero(np.all(grid != BLACK, axis=0))) != len(observed_cols):
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

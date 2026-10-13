from __future__ import annotations

import random

import numpy as np


_BG = 0
_FG = 4

_GEN_HEIGHT = 5
_GEN_WIDTH = 10
_GEN_PERIODS = (2, 3)
_GEN_SHIFTS = (1, 2)
_GEN_MAX_ATTEMPTS = 500


def _roll_row(row: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(np.asarray(row, dtype=int), int(shift))


def _is_empty_row(row: np.ndarray, bg: int) -> bool:
    return not bool(np.any(np.asarray(row, dtype=int) != int(bg)))


def _known_height(grid: np.ndarray, bg: int) -> int:
    nonempty_rows = np.flatnonzero(np.any(grid != int(bg), axis=1))
    assert nonempty_rows.size > 0, "expected at least one visible non-background row"
    last = int(nonempty_rows[-1]) + 1
    assert all(_is_empty_row(grid[r], bg) for r in range(last, grid.shape[0])), (
        "expected missing rows to be a pure-background suffix"
    )
    return last


def _candidate_period_shifts(rows: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, rows.shape)
    candidates: list[tuple[int, int]] = []
    for period in range(1, h):
        for shift in range(w):
            if all(
                np.array_equal(_roll_row(rows[r - period], shift), rows[r])
                for r in range(period, h)
            ):
                candidates.append((int(period), int(shift)))
    return candidates


def _infer_period_shift(rows: np.ndarray) -> tuple[int, int]:
    candidates = _candidate_period_shifts(rows)
    assert candidates, "expected a periodic translated row pattern"
    min_period = min(period for period, _shift in candidates)
    best = [pair for pair in candidates if pair[0] == min_period]
    assert len(best) == 1, "expected a unique shift for the minimal vertical period"
    return best[0]


def _complete_rows(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    known = _known_height(out, bg)
    assert known < out.shape[0], "expected at least one missing row"
    period, shift = _infer_period_shift(out[:known])
    for r in range(known, out.shape[0]):
        out[r] = _roll_row(out[r - period], shift)
    return out


def _mirror_expand(quadrant: np.ndarray) -> np.ndarray:
    q = np.asarray(quadrant, dtype=int)
    return np.block(
        [
            [np.flipud(np.fliplr(q)), np.flipud(q)],
            [np.fliplr(q), q],
        ]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing periodic rows, then expand the finished block by four-way mirroring.

    Input structure:
    - The input is a small black-background grid whose bottom 1-2 rows are fully black.
    - Visible rows use yellow pixels to form a row pattern on the upper part of the grid.
    - After a fixed vertical period, each row reappears with a fixed horizontal wraparound shift.

    Rule:
    - Find the minimal vertical period and its unique horizontal wraparound shift from the visible rows.
    - Fill each missing bottom row by shifting the row one period above it by that amount, wrapping past the right edge back to the left.
    - Treat the completed rectangle as the lower-right quadrant of the answer.
    - Fill the other three quadrants with left-right, top-bottom, and double reflections of that completed rectangle.

    Key insight:
    - The bottom rows are not arbitrary blanks: they continue a translated periodic row sequence, and that recovered rectangle is then mirrored into all four quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    completed = _complete_rows(g, _BG)
    return _mirror_expand(completed)


def _row_has_nontrivial_shift_symmetry(row: np.ndarray) -> bool:
    return any(np.array_equal(row, _roll_row(row, shift)) for shift in range(1, row.size))


def _make_row() -> np.ndarray:
    row = np.full(_GEN_WIDTH, _BG, dtype=int)
    style = random.choice(("run", "split", "scatter"))

    if style == "run":
        length = random.randint(2, 4)
        start = random.randint(0, _GEN_WIDTH - length)
        row[start : start + length] = _FG
        return row

    if style == "split":
        left_len = random.randint(1, 2)
        right_len = random.randint(1, 2)
        total = left_len + right_len
        start = random.randint(0, _GEN_WIDTH - total - 1)
        gap = random.randint(1, _GEN_WIDTH - start - total)
        row[start : start + left_len] = _FG
        row[start + left_len + gap : start + left_len + gap + right_len] = _FG
        return row

    cols = sorted(random.sample(range(_GEN_WIDTH), random.randint(2, 4)))
    row[cols] = _FG
    return row


def _random_base_rows(period: int) -> np.ndarray:
    rows = []
    while len(rows) < period:
        row = _make_row()
        if _is_empty_row(row, _BG) or _row_has_nontrivial_shift_symmetry(row):
            continue
        rows.append(row)
    return np.array(rows, dtype=int)


def _build_full_pattern(period: int, shift: int, base_rows: np.ndarray) -> np.ndarray:
    full = np.full((_GEN_HEIGHT, _GEN_WIDTH), _BG, dtype=int)
    full[:period] = base_rows
    for r in range(period, _GEN_HEIGHT):
        full[r] = _roll_row(full[r - period], shift)
    return full


def _is_valid_full_pattern(full: np.ndarray, period: int, shift: int, visible: int) -> bool:
    if any(_is_empty_row(full[r], _BG) for r in range(_GEN_HEIGHT)):
        return False
    if np.array_equal(full, np.fliplr(full)) or np.array_equal(full, np.flipud(full)):
        return False

    visible_rows = full[:visible]
    full_pair = _infer_period_shift(full)
    visible_pair = _infer_period_shift(visible_rows)
    if full_pair != (period, shift) or visible_pair != (period, shift):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    period = random.choice(_GEN_PERIODS)
    shift = random.choice(_GEN_SHIFTS)
    missing_choices = [m for m in (1, 2) if _GEN_HEIGHT - m >= period + 1]
    missing = random.choice(missing_choices)
    visible = _GEN_HEIGHT - missing

    base_rows = _random_base_rows(period)
    full = _build_full_pattern(period, shift, base_rows)
    if not _is_valid_full_pattern(full, period, shift, visible):
        return None

    inp = full.copy()
    inp[visible:, :] = _BG

    if not np.array_equal(main(inp), _mirror_expand(full)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

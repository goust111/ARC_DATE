from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GREY = 5
CONTROL_COLORS = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _minimal_period(core: np.ndarray) -> int:
    """Return the smallest column period consistent with all visible non-black cells."""
    coords = np.argwhere(core != BG)
    assert coords.size > 0, "expected a visible motif"
    assert np.any(coords[:, 1] == 0), "motif must touch the left edge"

    width = core.shape[1]
    for period in range(1, width):
        seen: dict[tuple[int, int], int] = {}
        ok = True
        for r, c in coords:
            key = (int(r), int(c) % period)
            color = int(core[r, c])
            if key in seen and seen[key] != color:
                ok = False
                break
            seen[key] = color
        if ok:
            return period

    raise AssertionError("failed to infer a consistent period")


def _repeat_core(core: np.ndarray, period: int) -> np.ndarray:
    """Expand the visible periodic prefix to the full non-control width."""
    height, width = core.shape
    tile = np.full((height, period), BG, dtype=int)
    for r, c in np.argwhere(core != BG):
        tile[int(r), int(c) % period] = int(core[r, c])

    out = np.empty_like(core)
    for c in range(width):
        out[:, c] = tile[:, c % period]
    return out


def _recolor_rows(periodic_core: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Replace grey placeholders with the row control color."""
    out = periodic_core.copy()
    for r in range(out.shape[0]):
        grey_mask = out[r] == GREY
        row_control = int(control[r])
        if np.any(grey_mask):
            assert row_control not in (BG, GREY), "grey rows need one visible control color"
            out[r, grey_mask] = row_control
        else:
            assert row_control == BG, "control pixels should appear only on grey rows"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partial horizontal motif and recolor its repeated grey placeholders row by row.

    Input structure:
    - The grid is mostly black, and the last column is a control column.
    - In the non-control area, only the left prefix is visible: a blue-and-grey motif already showing at least two repeats
      of its true column period, while the right side is black because later repeats are hidden.
    - Each row that contains grey placeholder cells has exactly one non-black control pixel in the last column; rows
      without grey placeholders have black in the control column.

    Rule:
    - Ignore the last column and find the smallest column period consistent with all visible blue and grey cells.
    - Repeat that period across the full non-control width, so blue and grey cells both propagate into the hidden right side.
    - For each row, replace every grey cell in the completed motif with that row's control color.
    - Keep blue cells unchanged and preserve the control column.

    Key insight:
    - Grey placeholders are part of the repeated motif itself, but their final color is decided only after repetition by the
      per-row control marker in the last column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 3, "expected a non-control area plus one control column"

    core = g[:, :-1]
    control = g[:, -1]
    period = _minimal_period(core)
    periodic_core = _repeat_core(core, period)
    recolored = _recolor_rows(periodic_core, control)

    out = g.copy()
    out[:, :-1] = recolored
    out[:, -1] = control
    return out


def _interval_rows(period: int, active_h: int) -> list[set[int]]:
    """Build a compact staircase-like local motif using contiguous intervals."""
    rows: list[set[int]] = []
    bottom_len = random.randint(max(2, period - 1), period)
    bottom_start = random.randint(0, period - bottom_len)
    prev = set(range(bottom_start, bottom_start + bottom_len))
    rows.append(prev)

    for _ in range(active_h - 1):
        prev_min = min(prev)
        prev_max = max(prev)
        max_len = len(prev)
        min_len = 1 if period == 2 else max(1, max_len - 1)
        length = random.randint(min_len, max_len)
        left = max(0, prev_min - 1)
        right = min(period - length, prev_max)
        if left > right:
            left = right = max(0, min(period - length, prev_min))
        start = random.randint(left, right)
        prev = set(range(start, start + length))
        rows.append(prev)

    rows.reverse()
    return rows


def _make_tile(height: int, period: int, active_h: int, top: int) -> tuple[np.ndarray, dict[int, int]] | None:
    tile = np.full((height, period), BG, dtype=int)
    row_sets = _interval_rows(period, active_h)
    active_rows = list(range(top, top + active_h))

    if len({frozenset(cols) for cols in row_sets}) < 2:
        return None
    if not any(0 in cols for cols in row_sets):
        return None
    if not any(len(cols) < period for cols in row_sets):
        return None

    for r, cols in zip(active_rows, row_sets):
        for c in cols:
            tile[r, c] = BLUE

    grey_row_count = random.randint(2, min(3, active_h - 1))
    candidate_rows = [r for r, cols in zip(active_rows, row_sets) if len(cols) >= 2]
    if len(candidate_rows) < grey_row_count:
        return None
    grey_rows = sorted(random.sample(candidate_rows, grey_row_count))

    controls = random.sample(CONTROL_COLORS, random.randint(2, min(4, grey_row_count)))
    control_by_row: dict[int, int] = {}
    for idx, r in enumerate(grey_rows):
        choices = [c for c in range(period) if tile[r, c] == BLUE]
        grey_col = random.choice(choices)
        tile[r, grey_col] = GREY
        control_by_row[r] = controls[idx % len(controls)]

    if len(set(control_by_row.values())) < 2:
        return None

    grey_counts = np.sum(tile == GREY, axis=1)
    if np.any(grey_counts > 1):
        return None
    if np.count_nonzero(tile == GREY) not in (2, 3):
        return None
    if not np.any(tile == BLUE):
        return None
    if not np.any(np.sum(tile == BLUE, axis=1) >= 2):
        return None
    return tile, control_by_row


def _build_input(height: int, repeats: int, tile: np.ndarray, control_by_row: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    period = tile.shape[1]
    non_control_w = period * repeats
    visible_cols = 2 * period

    repeated = np.tile(tile, (1, repeats))
    inp_core = repeated.copy()
    inp_core[:, visible_cols:] = BG

    inp = np.zeros((height, non_control_w + 1), dtype=int)
    inp[:, :-1] = inp_core
    for r, color in control_by_row.items():
        inp[r, -1] = color

    expected = np.zeros_like(inp)
    expected[:, :-1] = repeated
    expected[:, -1] = inp[:, -1]
    expected[:, :-1] = _recolor_rows(expected[:, :-1], expected[:, -1])
    return inp, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, true_period: int) -> bool:
    core = inp[:, :-1]
    control = inp[:, -1]
    if _minimal_period(core) != true_period:
        return False

    visible_cols = 2 * true_period
    hidden = expected[:, visible_cols:-1]
    if hidden.size == 0 or not np.any(hidden != BG):
        return False
    if not np.any(hidden == BLUE):
        return False
    if not np.any(expected[:, visible_cols:-1] == control[:, None]):
        return False

    unrecolored = np.zeros_like(inp)
    unrecolored[:, :-1] = _repeat_core(core, true_period)
    unrecolored[:, -1] = control
    if np.array_equal(unrecolored, expected):
        return False

    recolor_only = inp.copy()
    recolor_only[:, :-1] = _recolor_rows(recolor_only[:, :-1], control)
    if np.array_equal(recolor_only, expected):
        return False

    if not np.array_equal(main(inp), expected):
        return False

    grey_rows = np.flatnonzero(np.any(inp[:, :-1] == GREY, axis=1))
    if grey_rows.size < 2:
        return False
    if len(set(map(int, inp[grey_rows, -1]))) < 2:
        return False
    if np.any(np.sum(inp[:, :-1] == GREY, axis=1) > 2):
        return False
    total_visible_grey = int(np.sum(inp[:, :-1] == GREY))
    if not (3 <= total_visible_grey <= 6):
        return False
    if np.any((inp[:, -1] != BG) & ~np.any(inp[:, :-1] == GREY, axis=1)):
        return False
    if np.any(np.any(inp[:, :-1] == GREY, axis=1) & (inp[:, -1] == BG)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.randint(2, 4)
    repeats = random.randint(4, min(7, 29 // period))
    non_control_w = period * repeats
    if non_control_w + 1 > 30:
        return None

    height = random.randint(10, 15)
    active_h = random.randint(4, min(7, height - 2))
    top = random.randint(1, height - active_h - 1)

    made = _make_tile(height, period, active_h, top)
    if made is None:
        return None
    tile, control_by_row = made

    inp, expected = _build_input(height, repeats, tile, control_by_row)
    if not _semantic_checks(inp, expected, period):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")

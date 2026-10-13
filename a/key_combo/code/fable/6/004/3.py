from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
WIDTH_RANGE = (8, 12)
TOTAL_ROWS_RANGE = (7, 11)
BLANK_ROWS_RANGE = (1, 4)
PALETTE_SIZE_RANGE = (3, 5)
RUN_COUNT_RANGE = (3, 6)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _is_primitive_row(row: np.ndarray) -> bool:
    width = int(row.size)
    return all(not np.array_equal(row, np.roll(row, shift)) for shift in range(1, width))


def _roll_row(row: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(row, int(shift)).copy()


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(row, dtype=int) for row in rows], axis=0)


def _candidate_predictions(
    filled_rows: list[np.ndarray], total_rows: int
) -> tuple[int, list[np.ndarray], list[tuple[int, int]]]:
    n_rows = len(filled_rows)
    width = int(filled_rows[0].size)
    best_support = -1
    best_outputs: dict[bytes, list[np.ndarray]] = {}
    witnesses: dict[bytes, list[tuple[int, int]]] = {}

    for step in range(width):
        for anchor in range(n_rows):
            pred = [
                _roll_row(filled_rows[anchor], step * (row_idx - anchor))
                for row_idx in range(total_rows)
            ]
            support = sum(
                np.array_equal(pred[row_idx], filled_rows[row_idx])
                for row_idx in range(n_rows)
            )
            if support < best_support:
                continue
            sig = _stack_rows(pred).tobytes()
            if support > best_support:
                best_support = support
                best_outputs = {sig: pred}
                witnesses = {sig: [(step, anchor)]}
            else:
                best_outputs.setdefault(sig, pred)
                witnesses.setdefault(sig, []).append((step, anchor))

    assert best_support >= n_rows - 1, "failed to recover a global arithmetic-progression phase"
    assert len(best_outputs) == 1, "input does not determine a unique repaired-and-completed output"
    pred = next(iter(best_outputs.values()))
    bad_rows = [
        row_idx
        for row_idx in range(n_rows)
        if not np.array_equal(pred[row_idx], filled_rows[row_idx])
    ]
    assert len(bad_rows) == 1, "expected exactly one anomalous filled row"
    return bad_rows[0], pred, next(iter(witnesses.values()))


def _recover_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blank_color = int(g[-1, 0])
    blank_mask = np.all(g == blank_color, axis=1)
    assert bool(blank_mask[-1]), "expected at least one blank row at the bottom"
    first_blank = int(np.flatnonzero(blank_mask)[0])
    assert first_blank > 0, "expected at least one nonblank row above the blank tail"
    assert np.all(blank_mask[first_blank:]), "blank rows must form one contiguous bottom block"
    assert not np.any(blank_mask[:first_blank]), "filled rows must form one contiguous top block"

    filled_rows = [g[row_idx].copy() for row_idx in range(first_blank)]
    for row in filled_rows:
        assert not np.all(row == blank_color), "filled motif rows cannot be blank"

    _bad_idx, pred_rows, _witnesses = _candidate_predictions(filled_rows, g.shape[0])
    return _stack_rows(pred_rows)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single off-phase motif row, then continue the cyclic shift progression through the blank tail.

    Input structure:
    - The grid has a contiguous top block of nonblank rows and a contiguous bottom block of all-background rows.
    - Every nonblank row is a cyclic shift of the same primitive motif row, so the row-wise color multiset is identical.
    - All but one nonblank row follow one row-index arithmetic progression of shift amounts modulo the row width.

    Rule:
    - Use the globally consistent progression, not local adjacent differences, to identify the unique row whose phase breaks the arithmetic shift pattern.
    - Replace that row by the cyclic shift that the progression requires at its own row index.
    - Fill each blank bottom row with the next cyclic shifts from the same progression while preserving grid shape.

    Key insight:
    - The pixel content of the anomalous row is still a valid shifted copy of the motif; only its phase is wrong, so solving the task means recovering the hidden modulo-arithmetic phase sequence and extrapolating it.
    """
    return _recover_output(grid)


def _sample_run_lengths(width: int, run_count: int) -> list[int]:
    cuts = sorted(random.sample(range(1, width), run_count - 1))
    bounds = [0] + cuts + [width]
    return [bounds[i + 1] - bounds[i] for i in range(run_count)]


def _sample_run_colors(palette: list[int], run_count: int) -> list[int]:
    colors = [random.choice(palette)]
    for idx in range(1, run_count):
        choices = [color for color in palette if color != colors[-1]]
        if idx == run_count - 1 and colors[0] in choices:
            choices = [color for color in choices if color != colors[0]] or choices
        colors.append(random.choice(choices))
    return colors


def _make_motif_row(width: int, bg: int) -> np.ndarray | None:
    for _ in range(200):
        palette_size = random.randint(PALETTE_SIZE_RANGE[0], min(PALETTE_SIZE_RANGE[1], width))
        palette = [bg] + random.sample([color for color in range(10) if color != bg], palette_size - 1)
        run_count = random.randint(RUN_COUNT_RANGE[0], min(RUN_COUNT_RANGE[1], width))
        lengths = _sample_run_lengths(width, run_count)
        colors = _sample_run_colors(palette, run_count)
        row = np.concatenate(
            [np.full(length, color, dtype=int) for length, color in zip(lengths, colors)]
        )
        if len(set(map(int, row))) < 3:
            continue
        if np.all(row == bg):
            continue
        if not _is_primitive_row(row):
            continue
        return row
    return None


def _choose_dimensions() -> tuple[int, int, int] | None:
    width = random.randint(*WIDTH_RANGE)
    max_total = min(TOTAL_ROWS_RANGE[1], width - 1)
    if max_total < TOTAL_ROWS_RANGE[0]:
        return None
    total_rows = random.randint(TOTAL_ROWS_RANGE[0], max_total)
    max_blank = min(BLANK_ROWS_RANGE[1], total_rows - 6)
    if max_blank < BLANK_ROWS_RANGE[0]:
        return None
    blank_rows = random.randint(BLANK_ROWS_RANGE[0], max_blank)
    filled_rows = total_rows - blank_rows
    if filled_rows < 6:
        return None
    return width, filled_rows, blank_rows


def _progression_choices(width: int, total_rows: int) -> list[tuple[int, int, list[int]]]:
    choices = []
    for step in range(1, width):
        if width // math.gcd(width, step) < total_rows:
            continue
        for start in range(1, width):
            shifts = [(start + step * row_idx) % width for row_idx in range(total_rows)]
            if len(set(shifts)) != total_rows:
                continue
            if not any(shifts[row_idx] < shifts[row_idx - 1] for row_idx in range(1, total_rows)):
                continue
            choices.append((start, step, shifts))
    return choices


def _choose_bad_index(filled_rows: int) -> int:
    candidates = sorted({0, 1, filled_rows // 2, filled_rows - 2, filled_rows - 1})
    weighted = []
    for idx in candidates:
        weight = 3 if idx in {0, 1, filled_rows - 2, filled_rows - 1} else 2
        weighted.extend([idx] * weight)
    return random.choice(weighted)


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    motif: np.ndarray,
    shifts: list[int],
    bad_idx: int,
    wrong_shift: int,
    bg: int,
) -> bool:
    total_rows, width = inp.shape
    filled_rows = int(np.flatnonzero(np.all(inp == bg, axis=1))[0])
    blank_rows = total_rows - filled_rows

    if blank_rows < 1 or filled_rows < 6:
        return False
    if shifts[0] == 0:
        return False
    if wrong_shift == shifts[bad_idx]:
        return False
    if not _is_primitive_row(motif):
        return False
    if len(set(shifts)) != len(shifts):
        return False
    if not any(shifts[row_idx] < shifts[row_idx - 1] for row_idx in range(1, total_rows)):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    repaired = _recover_output(inp)
    if not np.array_equal(repaired, expected):
        return False
    if np.count_nonzero(np.any(repaired[:filled_rows] != inp[:filled_rows], axis=1)) != 1:
        return False
    if np.array_equal(repaired[filled_rows], inp[filled_rows - 1]):
        return False
    if blank_rows > 1 and np.array_equal(repaired[-1], repaired[-2]):
        return False

    same_multiset = [
        np.array_equal(np.bincount(inp[row_idx], minlength=10), np.bincount(motif, minlength=10))
        for row_idx in range(filled_rows)
    ]
    if not all(same_multiset):
        return False

    bad_row = inp[bad_idx]
    if np.array_equal(bad_row, expected[bad_idx]):
        return False
    if bad_idx > 0 and np.array_equal(bad_row, expected[bad_idx - 1]):
        return False
    if bad_idx + 1 < filled_rows and np.array_equal(bad_row, expected[bad_idx + 1]):
        return False

    candidate_bad, pred_rows, witnesses = _candidate_predictions(
        [inp[row_idx].copy() for row_idx in range(filled_rows)],
        total_rows,
    )
    if candidate_bad != bad_idx:
        return False
    if len(witnesses) < 1:
        return False
    if not np.array_equal(_stack_rows(pred_rows), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    dims = _choose_dimensions()
    if dims is None:
        return None
    width, filled_rows, blank_rows = dims
    total_rows = filled_rows + blank_rows

    bg = random.randint(0, 9)
    motif = _make_motif_row(width, bg)
    if motif is None:
        return None

    progression = _progression_choices(width, total_rows)
    if not progression:
        return None
    _start, _step, shifts = random.choice(progression)

    expected_rows = [_roll_row(motif, shift) for shift in shifts]
    expected = _stack_rows(expected_rows)

    bad_idx = _choose_bad_index(filled_rows)
    wrong_choices = [
        shift
        for shift in range(width)
        if shift not in shifts and shift != shifts[bad_idx]
    ]
    if not wrong_choices:
        return None
    wrong_shift = random.choice(wrong_choices)

    inp = np.full((total_rows, width), bg, dtype=int)
    for row_idx in range(filled_rows):
        shift = wrong_shift if row_idx == bad_idx else shifts[row_idx]
        inp[row_idx] = _roll_row(motif, shift)

    if not _validate_instance(inp, expected, motif, shifts, bad_idx, wrong_shift, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid progression-repair ARC input")

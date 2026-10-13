from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 500


def _spiral_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    top, left = 0, 0
    bottom, right = h - 1, w - 1
    coords: list[tuple[int, int]] = []

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        for r in range(top + 1, bottom + 1):
            coords.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                coords.append((r, left))

        top += 1
        bottom -= 1
        left += 1
        right -= 1

    assert len(coords) == h * w, "spiral path must visit every cell exactly once"
    return coords


def _candidate_for_period(seq: list[int], period: int) -> tuple[list[int], int, int] | None:
    pattern: list[int] = []
    mismatches: list[tuple[int, int]] = []

    for offset in range(period):
        idxs = list(range(offset, len(seq), period))
        if len(idxs) < 2:
            return None

        counts = Counter(seq[i] for i in idxs)
        ranked = counts.most_common()
        if len(ranked) == 1:
            pattern.append(ranked[0][0])
            continue

        if len(ranked) != 2:
            return None
        (major_color, major_count), (_minor_color, minor_count) = ranked
        if minor_count != 1 or major_count <= minor_count:
            return None

        pattern.append(int(major_color))
        for i in idxs:
            if seq[i] != major_color:
                mismatches.append((i, int(major_color)))

    if len(mismatches) != 1:
        return None

    bad_index, expected_color = mismatches[0]
    return pattern, bad_index, expected_color


def _solve_sequence(seq: list[int]) -> tuple[int, list[int], int, int]:
    for period in range(1, len(seq) // 2 + 1):
        candidate = _candidate_for_period(seq, period)
        if candidate is not None:
            pattern, bad_index, expected_color = candidate
            return period, pattern, bad_index, expected_color
    raise AssertionError("expected a unique single-cell break in a repeated spiral sequence")


def _sequence_from_coords(g: np.ndarray, coords: list[tuple[int, int]]) -> list[int]:
    return [int(g[r, c]) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one cell that breaks a short color period in clockwise spiral order.

    Input structure:
    - A non-empty rectangular grid is completely filled with ARC colors.
    - Row, column, and diagonal readings are distractors; the intended one-dimensional reading is the spiral from the top-left cell, first moving right, then turning clockwise inward.
    - Along that spiral reading, all cells except one follow one short repeated color cycle.

    Rule:
    - Read every cell in the top-left clockwise inward spiral order.
    - Find the shortest period whose residue classes agree everywhere except for one uniquely identified cell.
    - Return a same-sized grid with only that deviating cell recolored to the period's required color.

    Key insight:
    - The damaged pixel and the color cycle become simultaneously visible only after flattening the filled grid along the hidden spiral path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    coords = _spiral_coords(tuple(map(int, g.shape)))
    seq = _sequence_from_coords(g, coords)
    _period, _pattern, bad_index, expected_color = _solve_sequence(seq)

    out = g.copy()
    r, c = coords[bad_index]
    assert int(out[r, c]) != expected_color, "the repaired cell must visibly change"
    out[r, c] = expected_color
    return out


# --------------- generator ---------------


def _exact_min_period(seq: list[int]) -> int:
    for period in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return period
    raise AssertionError("finite sequence always has a period")


def _is_primitive_pattern(pattern: list[int]) -> bool:
    return _exact_min_period(pattern) == len(pattern)


def _reading_orders(shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = shape
    row_major = [(r, c) for r in range(h) for c in range(w)]
    col_major = [(r, c) for c in range(w) for r in range(h)]

    diag_down: list[tuple[int, int]] = []
    for s in range(h + w - 1):
        cells = [(r, s - r) for r in range(h) if 0 <= s - r < w]
        diag_down.extend(cells)

    diag_up: list[tuple[int, int]] = []
    for d in range(-(w - 1), h):
        cells = [(r, r - d) for r in range(h) if 0 <= r - d < w]
        diag_up.extend(cells)

    return [row_major, col_major, diag_down, diag_up]


def _has_short_single_break(seq: list[int], max_period: int) -> bool:
    limit = min(max_period, len(seq) // 2)
    return any(_candidate_for_period(seq, p) is not None for p in range(1, limit + 1))


def _fill_by_spiral(shape: tuple[int, int], seq: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (r, c), color in zip(_spiral_coords(shape), seq):
        out[r, c] = int(color)
    return out


def _period_residue_sizes(length: int, period: int) -> list[int]:
    return [len(range(offset, length, period)) for offset in range(period)]


def _make_pattern(period: int) -> list[int]:
    colors = random.sample(PALETTE, random.randint(3, min(5, period)))
    for _ in range(100):
        pattern = [random.choice(colors) for _ in range(period)]
        if len(set(pattern)) >= 3 and _is_primitive_pattern(pattern):
            return pattern

    pattern = colors[:]
    while len(pattern) < period:
        pattern.append(random.choice(colors))
    random.shuffle(pattern)
    if not _is_primitive_pattern(pattern):
        return _make_pattern(period)
    return pattern


def _gen_shape(period: int) -> tuple[int, int]:
    for _ in range(100):
        h = random.randint(7, 16)
        w = random.randint(7, 16)
        if random.random() < 0.65:
            while w == h:
                w = random.randint(7, 16)
        if h * w >= 4 * period and max(h, w) <= 30:
            return h, w
    return 12, 13


def _candidate_periods(seq: list[int]) -> list[tuple[int, int, int]]:
    out = []
    for p in range(1, len(seq) // 2 + 1):
        candidate = _candidate_for_period(seq, p)
        if candidate is not None:
            _pattern, bad_idx, expected = candidate
            out.append((p, bad_idx, expected))
    return out


def _gen_attempt() -> np.ndarray | None:
    period = random.choice((3, 4, 5, 6, 7))
    shape = _gen_shape(period)
    length = shape[0] * shape[1]
    if min(_period_residue_sizes(length, period)) < 4:
        return None

    pattern = _make_pattern(period)
    correct_seq = [pattern[i % period] for i in range(length)]
    if _exact_min_period(correct_seq) != period:
        return None

    residue_sizes = _period_residue_sizes(length, period)
    eligible_residues = [i for i, count in enumerate(residue_sizes) if count >= 4]
    if not eligible_residues:
        return None
    bad_residue = random.choice(eligible_residues)
    bad_slots = list(range(bad_residue, length, period))
    bad_index = random.choice(bad_slots)
    expected_color = correct_seq[bad_index]
    replacement_choices = sorted(set(pattern) - {expected_color})
    if not replacement_choices:
        return None

    broken_seq = correct_seq.copy()
    broken_seq[bad_index] = random.choice(replacement_choices)

    candidates = _candidate_periods(broken_seq)
    shortest = [candidate for candidate in candidates if candidate[0] == period]
    if len(shortest) != 1 or shortest[0][1:] != (bad_index, expected_color):
        return None
    if any(p < period for p, _idx, _expected in candidates):
        return None

    inp = _fill_by_spiral(shape, broken_seq)
    expected = _fill_by_spiral(shape, correct_seq)

    max_distractor_period = min(10, period + 3)
    for order in _reading_orders(shape):
        if order == _spiral_coords(shape):
            continue
        distractor_seq = _sequence_from_coords(inp, order)
        if _has_short_single_break(distractor_seq, max_distractor_period):
            return None

    values = Counter(int(x) for x in inp.ravel())
    if len(values) < 3 or min(values.values()) < 4:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    diff = np.argwhere(inp != actual)
    if len(diff) != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-period repair input")

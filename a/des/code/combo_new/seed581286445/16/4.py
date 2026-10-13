from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


def _segment_bounds(row: np.ndarray, bg: int = BG) -> tuple[int, int] | None:
    cols = np.flatnonzero(row != int(bg))
    if cols.size == 0:
        return None
    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert cols.size == c1 - c0 + 1, "each row must contain one contiguous colored segment"
    return c0, c1


def _unique_mismatch_pair(segment: np.ndarray) -> tuple[int, int]:
    n = int(segment.shape[0])
    mismatches = [(i, n - 1 - i) for i in range(n // 2) if int(segment[i]) != int(segment[n - 1 - i])]
    assert len(mismatches) == 1, "each segment must have exactly one mirrored mismatch pair"
    return mismatches[0]


def _paired_support(segment: np.ndarray, color: int) -> int:
    n = int(segment.shape[0])
    support = 0
    for i in range(n // 2):
        if int(segment[i]) == int(color) and int(segment[n - 1 - i]) == int(color):
            support += 1
    if n % 2 == 1 and int(segment[n // 2]) == int(color):
        support += 1
    return support


def _defect_index(segment: np.ndarray, left_idx: int, right_idx: int) -> int:
    counts = Counter(map(int, segment))
    left_color = int(segment[left_idx])
    right_color = int(segment[right_idx])

    left_score = (int(counts[left_color]), _paired_support(segment, left_color))
    right_score = (int(counts[right_color]), _paired_support(segment, right_color))
    assert left_score != right_score, "defect side must be identifiable from visible row structure"
    return int(left_idx if left_score < right_score else right_idx)


def _shift_row(row: np.ndarray, delta: int, bg: int = BG) -> np.ndarray:
    out = np.full_like(row, int(bg))
    if delta < 0:
        out[:delta] = row[-delta:]
    elif delta > 0:
        out[delta:] = row[:-delta]
    else:
        out[:] = row
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each nearly palindromic row by the side of its lone defect, then repair the defect into perfect mirror symmetry.

    Input structure:
    - Black background surrounds one contiguous non-black pixel sequence in each non-empty row.
    - Each row's colored sequence is almost left-right symmetric about its own center.
    - Exactly one mirrored pair disagrees in every sequence, and visible color support makes one side the unique defect pixel.

    Rule:
    - For each row, isolate the contiguous colored segment and find its unique mirrored mismatch pair.
    - Identify which member of that pair is the defect pixel, replace it with the mirror-side color, and thus restore a perfect palindrome.
    - If the defect lies in the left half, shift the whole repaired row left by 1; if it lies in the right half, shift the whole repaired row right by 1.
    - Pixels shifted beyond the canvas are dropped and newly exposed cells become black.

    Key insight:
    - The row must be read in three steps: detect the almost-palindrome, infer which mismatching side is the true defect, then use that side both as the shift direction code and as the pixel to repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    for r in range(h):
        bounds = _segment_bounds(g[r], bg=BG)
        if bounds is None:
            continue

        c0, c1 = bounds
        segment = g[r, c0 : c1 + 1].copy()
        left_idx, right_idx = _unique_mismatch_pair(segment)
        defect_idx = _defect_index(segment, left_idx, right_idx)
        segment[defect_idx] = segment[int(segment.shape[0] - 1 - defect_idx)]

        repaired_row = np.full(w, BG, dtype=int)
        repaired_row[c0 : c1 + 1] = segment

        shift = -1 if defect_idx < segment.shape[0] // 2 else 1
        out[r] = _shift_row(repaired_row, int(shift), bg=BG)

    return out


def _sample_left_half(half_len: int, palette: list[int]) -> tuple[list[int], int]:
    assert half_len >= 2
    repeated_color = int(random.choice(palette))
    max_repeat = half_len if len(palette) == 1 else max(2, half_len - 1)
    repeat_count = random.randint(2, max_repeat)

    left_half = [repeated_color] * repeat_count
    other_colors = [c for c in palette if c != repeated_color]
    while len(left_half) < half_len:
        if other_colors:
            left_half.append(int(random.choice(other_colors)))
        else:
            left_half.append(repeated_color)
    random.shuffle(left_half)
    return left_half, repeated_color


def _make_palindrome(length: int, palette: list[int]) -> tuple[list[int], int]:
    half_len = length // 2
    left_half, repeated_color = _sample_left_half(half_len, palette)
    center = [int(random.choice(palette))] if length % 2 == 1 else []
    full = left_half + center + list(reversed(left_half))
    return full, repeated_color


def _make_row_case(width: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_len = min(width - 2, 13)
    if max_len < 4:
        return None

    valid_lengths = [length for length in range(4, max_len + 1) if length // 2 >= 2]
    length = int(random.choice(valid_lengths))

    palette_size = random.randint(1, min(4, len(COLORS) - 1))
    row_palette = random.sample(COLORS, palette_size)
    base_segment, repeated_color = _make_palindrome(length, row_palette)

    half_len = length // 2
    candidate_pairs = [i for i in range(half_len) if int(base_segment[i]) == int(repeated_color)]
    assert candidate_pairs, "repeated color must appear in the left half"
    pair_idx = int(random.choice(candidate_pairs))

    branch_left = bool(random.getrandbits(1))
    defect_idx = int(pair_idx if branch_left else length - 1 - pair_idx)

    row_colors = set(map(int, base_segment))
    bad_choices = [c for c in COLORS if c not in row_colors]
    if not bad_choices:
        return None
    bad_color = int(random.choice(bad_choices))

    broken_segment = list(base_segment)
    broken_segment[defect_idx] = bad_color

    start_choices = list(range(1, width - length))
    if not start_choices:
        return None
    start = int(random.choice(start_choices))

    inp = np.full(width, BG, dtype=int)
    inp[start : start + length] = np.array(broken_segment, dtype=int)

    expected = np.full(width, BG, dtype=int)
    delta = -1 if branch_left else 1
    shifted_start = start + delta
    expected[shifted_start : shifted_start + length] = np.array(base_segment, dtype=int)
    return inp, expected


def _sample_grid_shape() -> tuple[int, int]:
    height = random.randint(4, 10)
    width = random.randint(max(8, height + 2), 26)
    return height, width


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_grid_shape()
    inp = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    saw_left = False
    saw_right = False
    saw_even = False
    saw_odd = False

    for r in range(height):
        row_case = _make_row_case(width)
        if row_case is None:
            return None
        row_inp, row_expected = row_case
        inp[r] = row_inp
        expected[r] = row_expected

        c0, c1 = _segment_bounds(row_inp, bg=BG)
        assert c0 is not None and c1 is not None
        segment = row_inp[c0 : c1 + 1]
        left_idx, right_idx = _unique_mismatch_pair(segment)
        defect_idx = _defect_index(segment, left_idx, right_idx)
        saw_left |= defect_idx < segment.shape[0] // 2
        saw_right |= defect_idx >= segment.shape[0] // 2
        saw_even |= segment.shape[0] % 2 == 0
        saw_odd |= segment.shape[0] % 2 == 1

    if not saw_left or not saw_right:
        return None
    if not saw_even or not saw_odd:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-row shift task")

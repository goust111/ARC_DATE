from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

LEFT_HEIGHT = 12
LEFT_WIDTH = 9
RIGHT_HEIGHT = 12
RIGHT_WIDTH = 9
INPUT_SHAPE = (12, 19)
OUTPUT_SHAPE = (12, 9)
LOOKUP_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL)
LEFT_NOISE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)
RIGHT_SPRITE_COLORS = (MAROON,)
GEN_ATTEMPTS = 400


def _dominant_non_bg(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row if int(v) not in {BLACK, GREY})
    assert counts, "each lookup row must contain at least one non-background color"
    best_count = max(counts.values())
    best_colors = [color for color, count in counts.items() if count == best_count]
    assert len(best_colors) == 1, "each lookup row must have a unique dominant color"
    return int(best_colors[0])


def _bands_from_separator_lines(is_separator: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(map(bool, is_separator)):
        if flag:
            if start is not None:
                bands.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, len(is_separator) - 1))
    return bands


def _row_col_bands(right: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = _bands_from_separator_lines(np.all(right == GREY, axis=1))
    col_bands = _bands_from_separator_lines(np.all(right == GREY, axis=0))
    assert len(row_bands) == 4, "expected four cell rows in the right half"
    assert len(col_bands) == 3, "expected three cell columns in the right half"
    row_heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert row_heights == {2}, "expected 2-pixel-tall cell interiors"
    assert col_widths == {2}, "expected 2-pixel-wide cell interiors"
    return row_bands, col_bands


def _expected_output_from_lookup(
    right: np.ndarray, lookup_colors: list[int]
) -> np.ndarray:
    row_bands, col_bands = _row_col_bands(right)
    assert len(lookup_colors) == len(row_bands) * len(col_bands)
    out = right.copy()
    idx = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            maroon_mask = patch == MAROON
            assert 1 <= int(np.count_nonzero(maroon_mask)) <= 3, "each cell must contain a 1-3 pixel maroon sprite"
            patch[maroon_mask] = int(lookup_colors[idx])
            idx += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise noisy lookup stripes and use the decoded color sequence to recolor a divided sprite grid.

    Input structure:
    - The input is a 12x19 grid split by one full-height grey column into a 12x9 left half and a 12x9 right half.
    - The left half contains 12 horizontal noisy rows whose dominant non-black, non-grey color is the row's true stripe color.
    - The right half contains a grey divider lattice forming 3 columns by 4 rows of 2x2 cells, each with a small maroon sprite on black background.

    Rule:
    - Find the central full-height grey separator column and split the grid into left and right halves.
    - For each row of the left half, take the unique most frequent color among non-black, non-grey pixels to obtain a 12-color lookup sequence from top to bottom.
    - In the right half, identify the 12 cells by reading order, left-to-right then top-to-bottom.
    - Recolor every maroon sprite pixel in cell i with lookup color i, keeping black background and grey divider lines unchanged.
    - Return only the right half after recoloring.

    Key insight:
    - The left half is an implicit lookup table hidden by sparse noise: only row-wise dominant-color denoising reveals the color sequence that indexes the right-hand cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert tuple(map(int, g.shape)) == INPUT_SHAPE, f"expected input shape {INPUT_SHAPE}"

    sep_candidates = [
        c
        for c in range(g.shape[1])
        if np.all(g[:, c] == GREY) and c > 0 and c < g.shape[1] - 1
    ]
    sep_candidates = [
        c for c in sep_candidates
        if g[:, :c].shape == (LEFT_HEIGHT, LEFT_WIDTH) and g[:, c + 1 :].shape == (RIGHT_HEIGHT, RIGHT_WIDTH)
    ]
    assert len(sep_candidates) == 1, "expected a unique full-height separator between equal halves"
    sep_col = int(sep_candidates[0])

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    lookup = [_dominant_non_bg(left[r, :]) for r in range(left.shape[0])]
    return _expected_output_from_lookup(right, lookup)


def _random_sprite_mask(size: int | None = None) -> np.ndarray:
    cells = [(r, c) for r in range(2) for c in range(2)]
    if size is None:
        size = random.randint(1, 3)
    chosen = random.sample(cells, size)
    mask = np.zeros((2, 2), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _sample_lookup_sequence() -> list[int]:
    while True:
        seq = [random.choice(LOOKUP_COLORS) for _ in range(12)]
        row_major = seq
        column_major = [seq[idx] for idx in (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)]
        reversed_seq = list(reversed(seq))
        if row_major == column_major or row_major == reversed_seq:
            continue
        row_triples = [tuple(seq[3 * r : 3 * r + 3]) for r in range(4)]
        if not any(len(set(triple)) >= 2 for triple in row_triples):
            continue
        if len(set(seq)) < 3:
            continue
        return seq


def _make_noisy_row(color: int) -> tuple[np.ndarray, dict[str, bool]]:
    row = np.full(LEFT_WIDTH, color, dtype=int)
    noise_count = random.choice((1, 2))
    noise_positions = random.sample(range(LEFT_WIDTH), noise_count)
    same_noise_color = noise_count == 2 and bool(random.getrandbits(1))
    if same_noise_color:
        noise_colors = [random.choice([c for c in LEFT_NOISE_COLORS if c != color])] * noise_count
    else:
        noise_colors = [
            random.choice([c for c in LEFT_NOISE_COLORS if c != color])
            for _ in range(noise_count)
        ]
    for pos, noise_color in zip(noise_positions, noise_colors):
        row[pos] = noise_color
    return row, {
        "first_mismatch": int(row[0]) != color,
        "last_mismatch": int(row[-1]) != color,
        "noise_count_two": noise_count == 2,
    }


def _right_separator_positions(length: int, bands: int) -> tuple[list[int], list[tuple[int, int]]]:
    assert length == bands * 2 + bands
    boundary_at_start = bool(random.getrandbits(1))
    if boundary_at_start:
        separators = [3 * i for i in range(bands)]
        interiors = [(3 * i + 1, 3 * i + 2) for i in range(bands)]
    else:
        separators = [3 * i + 2 for i in range(bands)]
        interiors = [(3 * i, 3 * i + 1) for i in range(bands)]
    assert len(interiors) == bands and len(separators) == bands
    assert separators[-1] == length - 1 or interiors[-1][1] == length - 1
    return separators, interiors


def _build_right_half(lookup: list[int]) -> tuple[np.ndarray, np.ndarray]:
    right = np.full(OUTPUT_SHAPE, BLACK, dtype=int)
    expected = np.full(OUTPUT_SHAPE, BLACK, dtype=int)

    sep_rows, row_bands = _right_separator_positions(RIGHT_HEIGHT, 4)
    sep_cols, col_bands = _right_separator_positions(RIGHT_WIDTH, 3)
    right[sep_rows, :] = GREY
    right[:, sep_cols] = GREY
    expected[sep_rows, :] = GREY
    expected[:, sep_cols] = GREY

    forced_sizes = [1, 3] + [random.randint(1, 3) for _ in range(10)]
    random.shuffle(forced_sizes)
    idx = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            mask = _random_sprite_mask(forced_sizes[idx])
            right[r0 : r1 + 1, c0 : c1 + 1][mask] = MAROON
            expected[r0 : r1 + 1, c0 : c1 + 1][mask] = lookup[idx]
            idx += 1
    return right, expected


def _build_input_and_expected() -> tuple[np.ndarray, np.ndarray, list[int], dict[str, int]]:
    lookup = _sample_lookup_sequence()
    left = np.zeros((LEFT_HEIGHT, LEFT_WIDTH), dtype=int)
    stats = {"first_mismatch_rows": 0, "last_mismatch_rows": 0, "two_noise_rows": 0}
    for r, color in enumerate(lookup):
        row, row_stats = _make_noisy_row(color)
        left[r, :] = row
        stats["first_mismatch_rows"] += int(row_stats["first_mismatch"])
        stats["last_mismatch_rows"] += int(row_stats["last_mismatch"])
        stats["two_noise_rows"] += int(row_stats["noise_count_two"])

    right, expected_right = _build_right_half(lookup)
    inp = np.concatenate([left, np.full((LEFT_HEIGHT, 1), GREY, dtype=int), right], axis=1)
    return inp, expected_right, lookup, stats


def _alt_output(right: np.ndarray, seq: list[int]) -> np.ndarray:
    return _expected_output_from_lookup(right, seq)


def _semantic_checks(inp: np.ndarray, expected_right: np.ndarray, lookup: list[int], stats: dict[str, int]) -> bool:
    if stats["first_mismatch_rows"] == 0 or stats["last_mismatch_rows"] == 0 or stats["two_noise_rows"] == 0:
        return False

    left = inp[:, :LEFT_WIDTH]
    right = inp[:, LEFT_WIDTH + 1 :]

    solved = main(inp)
    if not np.array_equal(solved, expected_right):
        return False

    if np.array_equal(_alt_output(right, [int(left[r, 0]) for r in range(LEFT_HEIGHT)]), expected_right):
        return False
    if np.array_equal(_alt_output(right, [int(left[r, -1]) for r in range(LEFT_HEIGHT)]), expected_right):
        return False
    if np.array_equal(_alt_output(right, list(reversed(lookup))), expected_right):
        return False
    col_major = [lookup[idx] for idx in (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)]
    if np.array_equal(_alt_output(right, col_major), expected_right):
        return False

    row_modes = [_dominant_non_bg(left[r, :]) for r in range(LEFT_HEIGHT)]
    if row_modes != lookup:
        return False
    if np.array_equal(expected_right, right):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, expected_right, lookup, stats = _build_input_and_expected()
        try:
            if _semantic_checks(inp, expected_right, lookup, stats):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid noisy-lookup grid")

from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _row_dominant_color(row: np.ndarray) -> int:
    counts = Counter(map(int, row.tolist()))
    assert counts, "expected a non-empty row"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "each row must have a unique dominant color"
    dominant = int(ranked[0][0])
    assert dominant != 0, "stripe main colors must be non-black"
    return dominant


def _row_noise_count(row: np.ndarray) -> tuple[int, int]:
    dominant = _row_dominant_color(row)
    noise = int(np.count_nonzero(row != dominant))
    return dominant, noise


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean noisy horizontal stripe rows and reorder them by increasing noise count.

    Input structure:
    - The grid is a rectangle of horizontal stripe rows on a black-noise-capable canvas.
    - Each row has one unique non-black dominant color and 0-3 noise pixels of other colors.
    - Every row's noise count is distinct, so the hidden ordering key is unambiguous.

    Rule:
    - For each row, identify its dominant color and count how many pixels differ from it.
    - Replace the entire row by its dominant color to remove the noise.
    - Sort the cleaned rows from top to bottom by increasing per-row noise count.

    Key insight:
    - The noise pixels are both contamination and the only ordering code, so their counts
      must be extracted before denoising destroys that information.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    clean_rows: list[np.ndarray] = []
    keyed_rows: list[tuple[int, np.ndarray]] = []
    seen_noise_counts: set[int] = set()

    for r in range(g.shape[0]):
        dominant, noise = _row_noise_count(g[r, :])
        assert 0 <= noise <= 3, "each row must contain 0-3 noise pixels"
        assert noise not in seen_noise_counts, "row noise counts must be unique"
        seen_noise_counts.add(noise)

        clean = np.full(g.shape[1], dominant, dtype=int)
        clean_rows.append(clean)
        keyed_rows.append((noise, clean))

    keyed_rows.sort(key=lambda item: item[0])
    return np.stack([row for _noise, row in keyed_rows], axis=0)


_WIDTH_RANGE = (9, 16)
_ROW_COUNTS = (4,)
_NOISE_COUNTS_BY_ROWS = {
    4: (0, 1, 2, 3),
}
_GEN_ATTEMPTS = 400
_COLORS = tuple(range(1, 10))
_NOISE_COLORS = tuple(range(10))


def _perm_is_identity(perm: tuple[int, ...]) -> bool:
    return all(i == val for i, val in enumerate(perm))


def _make_noisy_row(width: int, main_color: int, noise_count: int) -> tuple[np.ndarray, int]:
    row = np.full(width, main_color, dtype=int)
    if noise_count == 0:
        return row, 0

    positions = random.sample(range(width), noise_count)
    black_count = 0
    for pos in positions:
        color = random.choice([c for c in _NOISE_COLORS if c != main_color])
        if color == 0:
            black_count += 1
        row[pos] = color
    return row, black_count


def _stable_argsort(values: list[int]) -> list[int]:
    return [idx for idx, _val in sorted(enumerate(values), key=lambda item: item[1])]


def _validate_instance(inp: np.ndarray, expected: np.ndarray, weights: list[int], black_counts: list[int]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    if np.array_equal(inp, expected):
        return False

    if np.array_equal(np.stack([np.full(inp.shape[1], _row_dominant_color(inp[r, :]), dtype=int) for r in range(inp.shape[0])]), expected):
        return False

    row_majorities = [_row_dominant_color(inp[r, :]) for r in range(inp.shape[0])]
    if len(set(row_majorities)) != len(row_majorities):
        return False

    if weights == sorted(weights):
        return False

    if _stable_argsort(weights) == _stable_argsort(black_counts):
        return False

    if not any(count == 0 for count in black_counts):
        return False
    if not any(0 < count < weight for count, weight in zip(black_counts, weights)):
        return False
    if not any(count > 0 for count in black_counts):
        return False

    color_order = [row_majorities[idx] for idx in _stable_argsort(weights)]
    if color_order == sorted(color_order) or color_order == sorted(color_order, reverse=True):
        return False

    sorted_raw_rows = inp[_stable_argsort(weights), :]
    if np.array_equal(sorted_raw_rows, expected):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.choice(_ROW_COUNTS)
    width = random.randint(*_WIDTH_RANGE)
    clean_colors = random.sample(_COLORS, height)

    target_weights = list(_NOISE_COUNTS_BY_ROWS[height])
    input_order = list(range(height))
    random.shuffle(input_order)
    if _perm_is_identity(tuple(input_order)):
        return None

    shuffled_weights = [target_weights[idx] for idx in input_order]
    shuffled_colors = [clean_colors[idx] for idx in input_order]

    rows = []
    black_counts = []
    for color, weight in zip(shuffled_colors, shuffled_weights):
        row, black_count = _make_noisy_row(width, color, weight)
        rows.append(row)
        black_counts.append(black_count)

    inp = np.stack(rows, axis=0)
    sorted_pairs = sorted(zip(target_weights, clean_colors), key=lambda item: item[0])
    expected = np.stack([np.full(width, color, dtype=int) for _weight, color in sorted_pairs], axis=0)

    if not _validate_instance(inp, expected, shuffled_weights, black_counts):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noisy-row sorting input")

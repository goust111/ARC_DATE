from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
MAX_ATTEMPTS = 300
ODD_WIDTHS = tuple(range(11, 30, 2))
HEIGHT_RANGE = (9, 25)
MAX_KEY_ROWS = 6
MAX_NOISE_PIXELS_PER_SIDE = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_on_side(row: np.ndarray, bg: int, start: int, stop: int) -> set[int]:
    return {int(v) for v in row[start:stop] if int(v) not in (bg, BLACK)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each center-column key color to count matching side pixels and redraw the row as one centered band.

    Input structure:
    - The grid has odd width and a non-black dominant background color.
    - Some rows contain a non-background key pixel on the center column; some key rows instead have a black hole at the center.
    - Colored scatter may appear on either side of the center column, while black is only used for missing center keys.

    Rule:
    - Start from an all-background output grid.
    - If a row has a visible center key, that color is the row key; if the center cell is black, infer the missing key as the unique non-background color appearing on both the left and right sides.
    - Count how many pixels on the left and right sides match the row key, then paint a solid horizontal band of that color from center-L through center+R, inclusive.
    - Rows with neither a visible key nor a black center hole are cleared to background.

    Key insight:
    - The center column is a key belt: each key both selects which side pixels to count and supplies the color used to repaint the row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert w % 2 == 1, "expected an odd grid width"

    bg = _get_bg(g)
    center = w // 2
    out = np.full_like(g, bg)

    for r in range(h):
        row = g[r]
        center_value = int(row[center])
        key_color: int | None = None

        if center_value not in (bg, BLACK):
            key_color = center_value
        elif center_value == BLACK:
            common = _colors_on_side(row, bg, 0, center) & _colors_on_side(row, bg, center + 1, w)
            assert len(common) == 1, "black center row must imply exactly one common side color"
            key_color = next(iter(common))

        if key_color is None:
            continue

        left_count = int(np.count_nonzero(row[:center] == key_color))
        right_count = int(np.count_nonzero(row[center + 1 :] == key_color))
        out[r, center - left_count : center + right_count + 1] = key_color

    return out


def _sample_key_counts(rng: random.Random, left_room: int, right_room: int, missing: bool) -> tuple[int, int]:
    if missing:
        return rng.randint(1, min(6, left_room)), rng.randint(1, min(6, right_room))

    left_count = rng.randint(0, min(6, left_room))
    right_count = rng.randint(0, min(6, right_room))
    if left_count == 0 and right_count == 0:
        if rng.random() < 0.5:
            left_count = 1
        else:
            right_count = 1
    return left_count, right_count


def _paint_side(
    row: np.ndarray,
    positions: list[int],
    key_color: int,
    key_count: int,
    side_only_colors: list[int],
    rng: random.Random,
) -> None:
    rng.shuffle(positions)
    for col in positions[:key_count]:
        row[col] = key_color

    free_positions = positions[key_count:]
    rng.shuffle(free_positions)
    noise_total = min(len(free_positions), rng.randint(0, MAX_NOISE_PIXELS_PER_SIDE))
    for col in free_positions[:noise_total]:
        row[col] = rng.choice(side_only_colors)


def _add_noise_row(
    grid: np.ndarray,
    row_index: int,
    bg: int,
    center: int,
    palette: list[int],
    rng: random.Random,
) -> None:
    row = grid[row_index]
    row[center] = bg
    pixel_total = rng.randint(1, min(8, row.size - 1))
    positions = list(range(center)) + list(range(center + 1, row.size))
    rng.shuffle(positions)
    for col in positions[:pixel_total]:
        row[col] = rng.choice(palette)


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()

    height = rng.randint(*HEIGHT_RANGE)
    width = rng.choice(ODD_WIDTHS)
    center = width // 2
    bg = rng.choice(list(range(1, 10)))
    palette = [color for color in range(1, 10) if color != bg]

    key_limit = min(MAX_KEY_ROWS, height - 1)
    key_rows_total = rng.randint(2, key_limit)
    missing_rows_total = rng.randint(1, min(2, key_rows_total - 1)) if rng.random() < 0.75 else 0
    present_rows_total = key_rows_total - missing_rows_total
    if present_rows_total <= 0:
        return None

    available_rows = list(range(height))
    rng.shuffle(available_rows)
    key_rows = sorted(available_rows[:key_rows_total])
    remaining_rows = [r for r in range(height) if r not in key_rows]
    noise_rows_total = 0 if not remaining_rows else rng.randint(1, min(3, len(remaining_rows)))
    noise_rows = set(rng.sample(remaining_rows, noise_rows_total)) if noise_rows_total else set()

    missing_rows = set(rng.sample(key_rows, missing_rows_total)) if missing_rows_total else set()

    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)

    left_positions = list(range(center))
    right_positions = list(range(center + 1, width))
    left_room = len(left_positions)
    right_room = len(right_positions)

    for r in key_rows:
        row = inp[r]
        missing = r in missing_rows
        key_color = rng.choice(palette)
        left_count, right_count = _sample_key_counts(rng, left_room, right_room, missing)

        if missing:
            row[center] = BLACK
            extras = [color for color in palette if color != key_color]
            rng.shuffle(extras)
            split = len(extras) // 2
            left_only = extras[:split] or extras[-1:]
            right_only = extras[split:] or extras[:1]
        else:
            row[center] = key_color
            extras = [color for color in palette if color != key_color]
            rng.shuffle(extras)
            split = rng.randint(1, max(1, len(extras) - 1))
            left_only = extras[:split]
            right_only = extras[split:] or extras[:1]

        _paint_side(row, left_positions.copy(), key_color, left_count, left_only, rng)
        _paint_side(row, right_positions.copy(), key_color, right_count, right_only, rng)
        expected[r, center - left_count : center + right_count + 1] = key_color

    for r in noise_rows:
        _add_noise_row(inp, r, bg, center, palette, rng)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")

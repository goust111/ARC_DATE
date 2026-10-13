from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import Color
import math
import random


def _row_color(row: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in row if int(v) != bg)
    assert counts, "each row must contain a non-bg color"

    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    if len(candidates) > 1:
        first_col = {color: int(np.flatnonzero(row == color)[0]) for color in candidates}
        leftmost = min(first_col.values())
        candidates = [color for color in candidates if first_col[color] == leftmost]

    assert len(candidates) == 1, "row color must be structurally unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each row's native foreground color against the black background.

    Input structure
    - A black-background grid whose rows each have one native foreground color plus optional distractor colors.
    - The native row color is the most frequent non-bg color in that row; if tied, its run starts leftmost among tied colors.

    Rule
    - For each row independently, turn pixels of the native row color into bg.
    - Turn every other pixel in that row, including bg and distractor colors, into the native row color.

    Key insight
    - Distractor colors are not preserved objects; they mark cells that participate in the binary complement of the row's native color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = Color.BLACK
    assert bool(np.any(g == bg)), "black background must be present"

    out = np.empty_like(g)
    for r in range(g.shape[0]):
        color = _row_color(g[r], bg)
        out[r] = np.where(g[r] == color, bg, color)

    return out

# --------------- generator (o5x) ---------------

# Canvas and retry bounds.
GRID_HEIGHT, WIDTH_MIN, WIDTH_MAX = 3, 8, 14
GENERATION_RETRIES = 300

# Row native-color density.
NATIVE_FRACTION_MIN, NATIVE_FRACTION_MAX = 0.30, 0.70

# Distractor palette and per-row count limits.
DISTRACTOR_PALETTE_MIN, DISTRACTOR_PALETTE_MAX = 0, 2
ROW_DISTRACTOR_PIXEL_MAX, ROW_DISTRACTOR_COLOR_MAX = 3, 2


def _infer_native_color(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row if int(v) != Color.BLACK)
    if not counts:
        raise ValueError("row has no foreground color")

    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    if len(candidates) > 1:
        first_col = {color: int(np.flatnonzero(row == color)[0]) for color in candidates}
        leftmost = min(first_col.values())
        candidates = [color for color in candidates if first_col[color] == leftmost]

    if len(candidates) != 1:
        raise ValueError("row native color is ambiguous")
    return candidates[0]


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = np.empty_like(grid)
    for r, row in enumerate(grid):
        native = _infer_native_color(row)
        out[r] = np.where(row == native, Color.BLACK, native)
    return out


def _bounded_positive_split(total: int, parts: int, max_each: int) -> list[int]:
    """Return shuffled positive part sizes, capped by max_each."""
    counts = [1] * parts
    remaining = total - parts
    while remaining:
        choices = [i for i, count in enumerate(counts) if count < max_each]
        i = random.choice(choices)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _sample_palette() -> tuple[list[int], list[int]]:
    """Return one native color per row plus the shared distractor palette."""
    native_colors = random.sample(Color.NOT_BLACK, GRID_HEIGHT)
    distractor_pool = [color for color in Color.NOT_BLACK if color not in native_colors]
    distractor_color_count = random.randint(DISTRACTOR_PALETTE_MIN, DISTRACTOR_PALETTE_MAX)
    return native_colors, random.sample(distractor_pool, distractor_color_count)


def _fill_row(row: np.ndarray, native: int, distractor_colors: list[int]) -> None:
    """Place one row's native pixels first, then optional distractor pixels."""
    width = row.size
    min_native = math.ceil(NATIVE_FRACTION_MIN * width)
    max_native = math.floor(NATIVE_FRACTION_MAX * width)
    native_count = random.randint(min_native, max_native)

    max_distractor_pixels = min(ROW_DISTRACTOR_PIXEL_MAX, width - native_count - 1)
    if distractor_colors:
        max_distractor_pixels = min(max_distractor_pixels, len(distractor_colors) * (native_count - 1))
    else:
        max_distractor_pixels = 0
    distractor_count = random.randint(0, max_distractor_pixels)

    native_cols = random.sample(range(width), native_count)
    row[native_cols] = native
    if not distractor_count:
        return

    min_distractor_colors = math.ceil(distractor_count / (native_count - 1))
    max_distractor_colors = min(len(distractor_colors), ROW_DISTRACTOR_COLOR_MAX, distractor_count)
    row_distractor_count = random.randint(min_distractor_colors, max_distractor_colors)
    row_distractors = random.sample(distractor_colors, row_distractor_count)
    per_color_counts = _bounded_positive_split(
        distractor_count,
        row_distractor_count,
        native_count - 1,
    )

    free_cols = [c for c in range(width) if c not in native_cols]
    distractor_cols = random.sample(free_cols, distractor_count)
    start = 0
    for color, count in zip(row_distractors, per_color_counts):
        cols = distractor_cols[start : start + count]
        row[cols] = color
        start += count


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(WIDTH_MIN, WIDTH_MAX)
    grid = np.full((GRID_HEIGHT, width), Color.BLACK, dtype=int)
    native_colors, distractor_colors = _sample_palette()
    for row, native in zip(grid, native_colors):
        _fill_row(row, native, distractor_colors)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid 2de01db2 input")

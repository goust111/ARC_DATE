from __future__ import annotations

import random

import numpy as np


HEIGHT_RANGE = (4, 10)
WIDTH_RANGE = (5, 12)
BAR_COUNT_RANGE = (3, 6)
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bar_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert rows.size > 0, "expected at least one foreground bar"
    return [int(r) for r in rows]


def _extract_bar(grid: np.ndarray, row: int, bg: int) -> tuple[int, int, int]:
    cols = np.flatnonzero(grid[row] != bg)
    assert cols.size == 2, "each foreground row must contain exactly one two-pixel bar"
    c0, c1 = map(int, cols)
    assert c1 == c0 + 1, "each foreground row must contain one adjacent two-pixel bar"
    left_color = int(grid[row, c0])
    right_color = int(grid[row, c1])
    assert left_color != right_color, "bar pixels must use two distinct colors"
    return c0, left_color, right_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift two-color row bars so the left pixel lands in that color's reference column.

    Input structure:
    - The dominant background surrounds horizontal adjacent two-pixel bars.
    - Every foreground row contains exactly one bar using the same two colors.
    - The topmost bar is the reference bar.

    Rule:
    - The reference bar maps its left and right colors to their reference columns.
    - For every lower bar, keep it if its left color already sits in that color's
      reference column.
    - Otherwise shift the whole bar horizontally until that left pixel reaches its
      reference column, restoring vacated cells to background.

    Key insight:
    - Rows are translated without recoloring; the left pixel's color determines the
      column anchor to match against the topmost bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _dominant_color(g)
    rows = _bar_rows(g, bg)

    ref_row = rows[0]
    ref_c0, ref_left, ref_right = _extract_bar(g, ref_row, bg)
    ref_colors = {ref_left, ref_right}
    reference_col = {ref_left: ref_c0, ref_right: ref_c0 + 1}

    out = g.copy()
    for row in rows[1:]:
        c0, left_color, right_color = _extract_bar(g, row, bg)
        assert {left_color, right_color} == ref_colors, "all bars must use the reference colors"

        target_c0 = reference_col[left_color]
        if c0 == target_c0:
            continue

        target_c1 = target_c0 + 1
        assert 0 <= target_c0 and target_c1 < g.shape[1], "shifted bar must stay inside the grid"

        out[row, :] = bg
        out[row, target_c0] = left_color
        out[row, target_c1] = right_color

    return out


def _sample_size() -> tuple[int, int]:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    return height, width


def _sample_bar_rows(height: int) -> list[int]:
    max_bars = min(BAR_COUNT_RANGE[1], height)
    count = random.randint(BAR_COUNT_RANGE[0], max_bars)
    return sorted(random.sample(range(height), count))


def _row_kinds(nonref_count: int) -> list[str]:
    assert nonref_count >= 2
    kinds = ["same", "reversed"]
    for _ in range(nonref_count - 2):
        kinds.append(random.choice(("same", "reversed")))
    random.shuffle(kinds)
    return kinds


def _misaligned_start(width: int, target_start: int) -> int:
    starts = [c for c in range(width - 1) if c != target_start]
    left_choices = [c for c in starts if c < target_start]
    right_choices = [c for c in starts if c > target_start]
    if left_choices and right_choices:
        starts = left_choices if random.random() < 0.5 else right_choices
    return random.choice(starts)


def _build_expected(
    shape: tuple[int, int],
    rows: list[int],
    ref_start: int,
    left_color: int,
    right_color: int,
    kinds: list[str],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    ref_row = rows[0]
    out[ref_row, ref_start] = left_color
    out[ref_row, ref_start + 1] = right_color

    for row, kind in zip(rows[1:], kinds):
        if kind == "same":
            start = ref_start
            colors = (left_color, right_color)
        else:
            start = ref_start + 1
            colors = (right_color, left_color)
        out[row, start] = colors[0]
        out[row, start + 1] = colors[1]
    return out


def _build_input(expected: np.ndarray, rows: list[int], ref_start: int, right_color: int, left_color: int, kinds: list[str], bg: int) -> np.ndarray:
    inp = expected.copy()
    target_start = ref_start + 1
    for row, kind in zip(rows[1:], kinds):
        if kind != "reversed":
            continue
        inp[row, :] = bg
        start = _misaligned_start(inp.shape[1], target_start)
        inp[row, start] = right_color
        inp[row, start + 1] = left_color
    return inp


def _generate_attempt() -> np.ndarray | None:
    height, width = _sample_size()
    rows = _sample_bar_rows(height)
    kinds = _row_kinds(len(rows) - 1)

    bg, left_color, right_color = random.sample(PALETTE, 3)
    ref_start = random.randint(0, width - 3)

    expected = _build_expected(
        (height, width),
        rows,
        ref_start,
        left_color,
        right_color,
        kinds,
        bg,
    )
    inp = _build_input(expected, rows, ref_start, right_color, left_color, kinds, bg)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")

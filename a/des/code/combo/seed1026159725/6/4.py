from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
ROW_COUNT_RANGE = (3, 6)
TARGET_WIDTH_RANGE = (8, 18)
ROW_MODES = ("contiguous", "split", "scatter", "alternating")


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    assert counts, "expected at least one value"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _row_dominant_color(row: np.ndarray, bg: int) -> int:
    non_bg = [int(v) for v in np.asarray(row).reshape(-1) if int(v) != bg]
    assert non_bg, "each target row must contain foreground pixels"
    counts = Counter(non_bg)
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "row dominant color must be unique"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder target rows according to the color sequence in the left control column.

    Input structure:
    - The most common color is the background.
    - The leftmost column is a control strip whose colors are all distinct.
    - The remaining columns form a target area with one row per control pixel.
    - In each target row, one non-background color is uniquely dominant.

    Rule:
    - Read the control colors from top to bottom.
    - Match each control color to the target row whose dominant non-background
      color is the same.
    - Reorder the target rows in control order and erase the control column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] >= 2, "expected a 2D grid with a control column"

    bg = _unique_mode(g)
    control = [int(v) for v in g[:, 0]]
    assert all(color != bg for color in control), "control colors must be non-background"
    assert len(set(control)) == len(control), "control colors must be distinct"

    target = g[:, 1:]
    row_by_color: dict[int, int] = {}
    for r in range(target.shape[0]):
        dominant = _row_dominant_color(target[r], bg)
        assert dominant not in row_by_color, "each dominant color must map to exactly one row"
        row_by_color[dominant] = r

    assert set(control) == set(row_by_color), "control colors must match target-row dominant colors"
    out = target[[row_by_color[color] for color in control], :].copy()
    return out


def _sample_main_positions(width: int, count: int, mode: str) -> set[int]:
    if mode == "contiguous":
        start = random.randint(0, width - count)
        return set(range(start, start + count))

    if mode == "split" and count >= 4 and width >= count + 1:
        gap = random.randint(1, min(3, width - count))
        left = random.randint(1, count - 1)
        right = count - left
        start = random.randint(0, width - (count + gap))
        return set(range(start, start + left)) | set(range(start + left + gap, start + left + gap + right))

    if mode == "alternating" and width >= count:
        min_span = count
        max_span = min(width, count * 2 + 1)
        span = random.randint(min_span, max_span)
        start = random.randint(0, width - span)
        candidates = [idx for idx in range(start, start + span) if (idx - start) % 2 == 0]
        if len(candidates) >= count:
            return set(random.sample(candidates, count))

    return set(random.sample(range(width), count))


def _partition_positive(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    result: list[int] = []
    for cut in cuts + [total]:
        result.append(cut - prev)
        prev = cut
    return result


def _paint_accents(row: np.ndarray, available: list[int], accent_palette: list[int], accent_total: int) -> None:
    if accent_total <= 0:
        return

    n_colors = 1 if accent_total == 1 else random.randint(1, min(2, accent_total, len(accent_palette)))
    accent_colors = random.sample(accent_palette, n_colors)
    counts = _partition_positive(accent_total, n_colors)

    random.shuffle(available)
    offset = 0
    for color, count in zip(accent_colors, counts):
        for idx in available[offset : offset + count]:
            row[idx] = color
        offset += count


def _make_target_row(width: int, bg: int, main_color: int, accent_palette: list[int]) -> np.ndarray:
    row = np.full(width, bg, dtype=int)

    min_main = max(3, width // 4)
    max_main = max(min_main, width // 2)
    main_count = random.randint(min_main, max_main)
    mode = random.choice(ROW_MODES)
    main_positions = _sample_main_positions(width, main_count, mode)
    for idx in main_positions:
        row[idx] = main_color

    remaining = [idx for idx in range(width) if idx not in main_positions]
    max_accent = min(3, len(remaining) - 2, main_count - 1)
    accent_total = random.randint(0, max(0, max_accent))
    _paint_accents(row, remaining, accent_palette, accent_total)

    assert _row_dominant_color(row, bg) == main_color
    return row


def _non_identity_shuffle(colors: list[int]) -> list[int]:
    shuffled = colors[:]
    random.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == colors:
        i, j = random.sample(range(len(shuffled)), 2)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*ROW_COUNT_RANGE)
    target_width = random.randint(*TARGET_WIDTH_RANGE)

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    row_colors = palette[:n_rows]
    accent_palette = palette[n_rows:]
    if len(accent_palette) < 2:
        return None

    control_order = _non_identity_shuffle(row_colors)
    source_order = _non_identity_shuffle(row_colors)
    if source_order == control_order:
        source_order = source_order[1:] + source_order[:1]

    grid = np.full((n_rows, target_width + 1), bg, dtype=int)
    expected_rows: dict[int, np.ndarray] = {}
    for r, row_color in enumerate(source_order):
        grid[r, 0] = control_order[r]
        target_row = _make_target_row(target_width, bg, row_color, accent_palette)
        grid[r, 1:] = target_row
        expected_rows[row_color] = target_row.copy()

    try:
        out = main(grid)
    except Exception:
        return None

    expected = np.stack([expected_rows[color] for color in control_order], axis=0)
    if not np.array_equal(out, expected):
        return None
    if _unique_mode(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

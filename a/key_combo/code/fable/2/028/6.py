from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Match the shared arc_utils convention: background is the most common color."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    """Match the shared arc_utils convention for listing visible colors."""
    cols = set(map(int, grid.flatten()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def _both_blank_scores(top: np.ndarray, bottom: np.ndarray, bg: int) -> np.ndarray:
    return np.sum((top == bg) & (bottom == bg), axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder whole columns by how often aligned cells are background in both stacked panels.

    Input structure:
    - The grid is split into an upper panel and a lower panel of equal height.
    - Both panels share one background color.
    - The upper panel uses exactly one foreground color, and the lower panel uses a different one.

    Rule:
    - Align the two panels by row and column.
    - For each column, count how many aligned row positions are background in both panels.
    - Reorder the entire input columns from left to right by that count in descending order.
    - Move each full column rigidly, preserving all cell colors and the vertical arrangement inside the column.

    Key insight:
    - The visible output is not a mask or overlap picture; the hidden double-background count only acts as a score that chooses the column order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0, "expected two equally tall stacked panels"

    bg = get_bg(g)
    panel_h = g.shape[0] // 2
    top = g[:panel_h]
    bottom = g[panel_h:]

    top_colors = colors_present(top, exclude={bg})
    bottom_colors = colors_present(bottom, exclude={bg})
    assert len(top_colors) == 1, "expected exactly one upper-panel foreground color"
    assert len(bottom_colors) == 1, "expected exactly one lower-panel foreground color"
    assert top_colors[0] != bottom_colors[0], "panel foreground colors must differ"

    scores = _both_blank_scores(top, bottom, bg)
    assert len(np.unique(scores)) == g.shape[1], "column scores must be unique"
    order = np.argsort(-scores, kind="stable")
    return g[:, order]


COLOR_POOL = tuple(range(10))
HEIGHT_RANGE = (8, 12)
WIDTH_MIN = 5
WIDTH_MAX = 7
GENERATION_RETRIES = 400


def _sample_union_rows(height: int, count: int) -> np.ndarray:
    mask = np.zeros(height, dtype=bool)
    if count == 0:
        return mask
    if count == height:
        mask[:] = True
        return mask
    if count == 1 or random.random() < 0.65:
        start = random.randint(0, height - count)
        mask[start : start + count] = True
        return mask

    first_len = random.randint(1, count - 1)
    second_len = count - first_len
    gap = random.randint(1, height - count)
    start1 = random.randint(0, height - count - gap)
    start2 = start1 + first_len + gap
    mask[start1 : start1 + first_len] = True
    mask[start2 : start2 + second_len] = True
    return mask


def _choose_rows(rows: np.ndarray, k: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    if k == 0:
        return np.empty(0, dtype=int)
    if k == len(rows):
        return rows.copy()
    return np.sort(np.random.choice(rows, size=k, replace=False))


def _build_column_masks(height: int, union_size: int, top_count: int, bottom_count: int) -> tuple[np.ndarray, np.ndarray]:
    overlap = top_count + bottom_count - union_size
    assert 0 <= overlap <= min(top_count, bottom_count)

    union_mask = _sample_union_rows(height, union_size)
    union_rows = np.flatnonzero(union_mask)
    overlap_rows = _choose_rows(union_rows, overlap)
    overlap_set = set(map(int, overlap_rows))
    remaining_rows = np.array([r for r in union_rows if int(r) not in overlap_set], dtype=int)

    top_only_rows = _choose_rows(remaining_rows, top_count - overlap)
    top_only_set = set(map(int, top_only_rows))
    bottom_only_rows = np.array([r for r in remaining_rows if int(r) not in top_only_set], dtype=int)

    top_mask = np.zeros(height, dtype=bool)
    bottom_mask = np.zeros(height, dtype=bool)
    top_mask[overlap_rows] = True
    bottom_mask[overlap_rows] = True
    top_mask[top_only_rows] = True
    bottom_mask[bottom_only_rows] = True

    assert int(top_mask.sum()) == top_count
    assert int(bottom_mask.sum()) == bottom_count
    assert int(np.logical_or(top_mask, bottom_mask).sum()) == union_size
    return top_mask, bottom_mask


def _strict_conflict(scores: np.ndarray, proxy_values: np.ndarray, *, proxy_prefers_higher: bool) -> bool:
    width = len(scores)
    for i in range(width):
        for j in range(width):
            if scores[i] <= scores[j]:
                continue
            if proxy_prefers_higher and proxy_values[i] < proxy_values[j]:
                return True
            if not proxy_prefers_higher and proxy_values[i] > proxy_values[j]:
                return True
    return False


def _has_same_visible_counts_but_different_score(
    scores: np.ndarray, top_counts: np.ndarray, bottom_counts: np.ndarray
) -> bool:
    width = len(scores)
    for i in range(width):
        for j in range(i + 1, width):
            if top_counts[i] != top_counts[j]:
                continue
            if bottom_counts[i] != bottom_counts[j]:
                continue
            if scores[i] != scores[j]:
                return True
    return False


def _validate_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    panel_h, width = g.shape[0] // 2, g.shape[1]

    bg = get_bg(g)
    top = g[:panel_h]
    bottom = g[panel_h:]
    top_colors = colors_present(top, exclude={bg})
    bottom_colors = colors_present(bottom, exclude={bg})
    assert len(top_colors) == 1 and len(bottom_colors) == 1
    assert top_colors[0] != bottom_colors[0]

    top_counts = np.sum(top != bg, axis=0)
    bottom_counts = np.sum(bottom != bg, axis=0)
    scores = _both_blank_scores(top, bottom, bg)
    assert len(np.unique(scores)) == width

    # These checks force the sample to rule out simpler but wrong shortcuts.
    assert _strict_conflict(scores, top_counts, proxy_prefers_higher=False)
    assert _strict_conflict(scores, bottom_counts, proxy_prefers_higher=False)
    assert _strict_conflict(scores, top_counts + bottom_counts, proxy_prefers_higher=False)
    assert _has_same_visible_counts_but_different_score(scores, top_counts, bottom_counts)

    bg_count = int(np.count_nonzero(g == bg))
    assert bg_count > int(np.count_nonzero(g == top_colors[0]))
    assert bg_count > int(np.count_nonzero(g == bottom_colors[0]))

    expected = g[:, np.argsort(-scores, kind="stable")]
    assert np.array_equal(main(g), expected)
    assert not np.array_equal(g, expected)


def _extra_column_counts(union_size: int) -> tuple[int, int]:
    kind = random.choice(("disjoint", "top_heavy", "bottom_heavy", "overlap_rich"))
    if kind == "disjoint":
        top_count = random.randint(1, union_size - 1)
        bottom_count = union_size - top_count
        return top_count, bottom_count
    if kind == "top_heavy":
        return union_size, random.randint(1, min(2, union_size))
    if kind == "bottom_heavy":
        return random.randint(1, min(2, union_size)), union_size

    overlap = random.randint(1, union_size - 2)
    top_only = random.randint(1, union_size - overlap - 1)
    bottom_only = union_size - overlap - top_only
    return overlap + top_only, overlap + bottom_only


def _gen_attempt() -> np.ndarray:
    panel_h = random.randint(*HEIGHT_RANGE)
    width = random.randint(WIDTH_MIN, min(WIDTH_MAX, panel_h - 2))
    base_union = random.randint(3, panel_h - width + 1)
    union_sizes = [base_union + i for i in range(width)]

    # The first four sorted-score columns are a built-in anti-shortcut pack:
    # 1) perfect overlap, 2) top-light/bottom-heavy, 3) same visible counts but different overlap,
    # 4) top-heavy/bottom-light.
    count_specs: list[tuple[int, int]] = [
        (base_union, base_union),
        (1, union_sizes[1]),
        (base_union, base_union),
        (union_sizes[3], 1),
    ]
    for union_size in union_sizes[4:]:
        count_specs.append(_extra_column_counts(union_size))

    top_columns: list[np.ndarray] = []
    bottom_columns: list[np.ndarray] = []
    for union_size, (top_count, bottom_count) in zip(union_sizes, count_specs):
        top_mask, bottom_mask = _build_column_masks(panel_h, union_size, top_count, bottom_count)
        top_columns.append(top_mask)
        bottom_columns.append(bottom_mask)

    permutation = np.random.permutation(width)
    if np.array_equal(permutation, np.arange(width)):
        permutation = np.roll(permutation, 1)

    bg, top_color, bottom_color = random.sample(COLOR_POOL, 3)
    grid = np.full((panel_h * 2, width), bg, dtype=int)
    for out_col, src_col in enumerate(permutation):
        grid[:panel_h, out_col][top_columns[int(src_col)]] = top_color
        grid[panel_h:, out_col][bottom_columns[int(src_col)]] = bottom_color

    _validate_grid(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-count column-reordering sample")

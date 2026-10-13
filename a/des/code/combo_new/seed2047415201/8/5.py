from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
INPUT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _dominant_color(values: np.ndarray) -> int | None:
    colors = [int(v) for v in values.tolist() if int(v) != BLACK]
    if not colors:
        return None

    counts = Counter(colors)
    best_count = max(counts.values())
    best = [color for color, count in counts.items() if count == best_count]
    assert len(best) == 1, "foreground colors must have a unique dominant color"
    return int(best[0])


def _axis_dominants(g: np.ndarray) -> tuple[list[int | None], list[int | None]]:
    row_colors = [_dominant_color(g[r, :]) for r in range(g.shape[0])]
    col_colors = [_dominant_color(g[:, c]) for c in range(g.shape[1])]
    return row_colors, col_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill black intersections from the dominant foreground colors of their row and column.

    Input structure:
    - Black-background grid with sparse foreground pixels.
    - Each row and each column has either no foreground pixels or a unique most frequent non-black color.
    - Some row/column intersections remain black.

    Rule:
    - Determine the unique dominant non-black color for every row and every column.
    - For each black cell whose row and column both have dominant colors, paint the row color when the two colors match and grey when they differ.
    - Leave cells black when either axis has no dominant color, and preserve all original foreground pixels.

    Key insight:
    - Rows and columns independently vote for a color, and the intersection records whether the two votes agree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_colors, col_colors = _axis_dominants(g)
    out = g.copy()
    height, width = g.shape
    for r in range(height):
        row_color = row_colors[r]
        if row_color is None:
            continue
        for c in range(width):
            if int(g[r, c]) != BLACK:
                continue
            col_color = col_colors[c]
            if col_color is None:
                continue
            out[r, c] = int(row_color if row_color == col_color else GREY)
    return out


def _balanced_labels(count: int, colors: list[int], duplicate_color: int) -> list[int]:
    labels = list(colors)
    labels.append(int(duplicate_color))
    while len(labels) < count:
        labels.append(random.choice(colors))
    random.shuffle(labels)
    return labels[:count]


def _expected_from_dominants(
    g: np.ndarray,
    row_colors: list[int | None],
    col_colors: list[int | None],
) -> np.ndarray:
    out = g.copy()
    height, width = g.shape
    for r in range(height):
        for c in range(width):
            if int(g[r, c]) != BLACK:
                continue
            row_color = row_colors[r]
            col_color = col_colors[c]
            if row_color is None or col_color is None:
                continue
            out[r, c] = int(row_color if row_color == col_color else GREY)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(6, 14)
    width = random.randint(6, 14)
    max_color_count = min(4, height - 2, width - 2)
    if max_color_count < 2:
        return None

    if max_color_count == 2:
        color_count = 2
    else:
        color_count = random.choice([2, 2, 2, *range(3, max_color_count + 1)])

    colors = random.sample(INPUT_COLORS, color_count)
    duplicate_color = random.choice(colors)

    active_row_count = random.randint(color_count + 1, height - 1)
    active_col_count = random.randint(color_count + 1, width - 1)
    active_rows = sorted(random.sample(range(height), active_row_count))
    active_cols = sorted(random.sample(range(width), active_col_count))

    row_labels = _balanced_labels(active_row_count, colors, duplicate_color)
    col_labels = _balanced_labels(active_col_count, colors, duplicate_color)
    row_color = {r: row_labels[i] for i, r in enumerate(active_rows)}
    col_color = {c: col_labels[i] for i, c in enumerate(active_cols)}

    rows_by_color = {color: [r for r in active_rows if row_color[r] == color] for color in colors}
    cols_by_color = {color: [c for c in active_cols if col_color[c] == color] for color in colors}

    reservable_same = [
        (r, c)
        for r in rows_by_color[duplicate_color]
        for c in cols_by_color[duplicate_color]
        if len(rows_by_color[duplicate_color]) >= 2 and len(cols_by_color[duplicate_color]) >= 2
    ]
    if not reservable_same:
        return None
    reserve_same = random.choice(reservable_same)

    mismatch_cells = [
        (r, c)
        for r in active_rows
        for c in active_cols
        if row_color[r] != col_color[c]
    ]
    if not mismatch_cells:
        return None
    reserve_mismatch = random.choice(mismatch_cells)
    reserved = {reserve_same, reserve_mismatch}

    cells: dict[tuple[int, int], int] = {}

    def add_cell(r: int, c: int, color: int) -> bool:
        if (r, c) in reserved:
            return False
        if (r, c) in cells:
            return int(cells[(r, c)]) == int(color)
        cells[(r, c)] = int(color)
        return True

    for r in active_rows:
        candidates = [c for c in cols_by_color[row_color[r]] if (r, c) not in reserved]
        if not candidates:
            return None
        add_cell(r, random.choice(candidates), row_color[r])

    for c in active_cols:
        if any(cc == c for _rr, cc in cells):
            continue
        candidates = [r for r in rows_by_color[col_color[c]] if (r, c) not in reserved]
        if not candidates:
            return None
        add_cell(random.choice(candidates), c, col_color[c])

    same_candidates = [
        (r, c)
        for r in active_rows
        for c in active_cols
        if row_color[r] == col_color[c] and (r, c) not in cells and (r, c) not in reserved
    ]
    random.shuffle(same_candidates)
    extra_same = random.randint(0, min(len(same_candidates), max(2, active_row_count * active_col_count // 6)))
    for r, c in same_candidates[:extra_same]:
        add_cell(r, c, row_color[r])

    col_anchor_counts = {
        c: sum(1 for r in active_rows if cells.get((r, c)) == col_color[c])
        for c in active_cols
    }
    distractor_candidates = [
        (r, c)
        for r in active_rows
        for c in active_cols
        if (
            row_color[r] != col_color[c]
            and (r, c) not in cells
            and (r, c) not in reserved
            and col_anchor_counts[c] >= 2
        )
    ]
    random.shuffle(distractor_candidates)
    used_distractor_cols: set[int] = set()
    max_distractors = min(len(distractor_candidates), max(1, min(active_row_count, active_col_count) // 3))
    distractor_count = random.randint(0, max_distractors)
    for r, c in distractor_candidates:
        if distractor_count <= 0:
            break
        if c in used_distractor_cols:
            continue
        add_cell(r, c, row_color[r])
        used_distractor_cols.add(c)
        distractor_count -= 1

    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in cells.items():
        grid[r, c] = int(color)

    if int(np.count_nonzero(grid)) * 2 >= height * width:
        return None

    found_row_colors, found_col_colors = _axis_dominants(grid)
    for r in range(height):
        expected = row_color.get(r)
        if found_row_colors[r] != expected:
            return None
    for c in range(width):
        expected = col_color.get(c)
        if found_col_colors[c] != expected:
            return None

    has_same_fill = False
    has_mismatch_fill = False
    has_no_axis_fill = False
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) != BLACK:
                continue
            row_dom = found_row_colors[r]
            col_dom = found_col_colors[c]
            if row_dom is None or col_dom is None:
                has_no_axis_fill = True
            elif row_dom == col_dom:
                has_same_fill = True
            else:
                has_mismatch_fill = True

    if not (has_same_fill and has_mismatch_fill and has_no_axis_fill):
        return None

    expected_output = _expected_from_dominants(grid, found_row_colors, found_col_colors)
    try:
        actual_output = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual_output, expected_output):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

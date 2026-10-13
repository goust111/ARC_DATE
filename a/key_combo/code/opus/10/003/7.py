from __future__ import annotations

import random

import numpy as np


BG = 0
CHEVRON = 7
SCATTER_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _chevron_coords(top: int, left: int, span: int, orientation: str) -> list[tuple[int, int]]:
    half = span // 2
    coords: set[tuple[int, int]] = set()
    for i in range(half):
        coords.add((top + i, left + i))
        coords.add((top + i + 1, left + i))
        col = left + span - 1 - i
        coords.add((top + i, col))
        coords.add((top + i + 1, col))
    coords.add((top + half, left + half))

    if orientation == "up":
        height = half + 1
        coords = {(top + height - 1 - (r - top), c) for r, c in coords}

    return sorted(coords)


def _covered_columns_and_bottoms(grid: np.ndarray) -> tuple[set[int], dict[int, int]]:
    coords = np.argwhere(grid == CHEVRON)
    covered = {int(c) for _, c in coords}
    bottoms: dict[int, int] = {}
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        bottoms[c] = max(bottoms.get(c, -1), r)
    return covered, bottoms


def _scatter_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    pixels = []
    for r_raw, c_raw in np.argwhere((grid != BG) & (grid != CHEVRON)):
        r = int(r_raw)
        c = int(c_raw)
        pixels.append((r, c, int(grid[r, c])))
    return pixels


def _row_color_counts(scatter: list[tuple[int, int, int]]) -> dict[int, dict[int, int]]:
    counts: dict[int, dict[int, int]] = {}
    for r, _c, color in scatter:
        counts.setdefault(r, {})
        counts[r][color] = counts[r].get(color, 0) + 1
    return counts


def _qualifying_pixels(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[int, dict[int, int]], set[int], dict[int, int]]:
    scatter = _scatter_pixels(grid)
    row_counts = _row_color_counts(scatter)
    covered, bottoms = _covered_columns_and_bottoms(grid)
    qualifying = []
    for r, c, color in scatter:
        in_covered_col = c in covered
        has_same_row_pair = row_counts[r].get(color, 0) >= 2
        if in_covered_col and has_same_row_pair:
            qualifying.append((r, c, color))
    return qualifying, row_counts, covered, bottoms


def _oracle(grid: np.ndarray, require_pair: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    scatter = _scatter_pixels(g)
    row_counts = _row_color_counts(scatter)
    covered, bottoms = _covered_columns_and_bottoms(g)

    for r, c, _color in scatter:
        out[r, c] = BG

    best_by_col: dict[int, tuple[int, int]] = {}
    for r, c, color in scatter:
        if c not in covered:
            continue
        if require_pair and row_counts[r].get(color, 0) < 2:
            continue
        best = best_by_col.get(c)
        if best is None or r < best[0]:
            best_by_col[c] = (r, color)

    for c, (_r, color) in best_by_col.items():
        out[bottoms[c] + 1 :, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill covered columns downward only for paired scatter pixels under the orange chevron.

    Input structure:
    - The grid has a black background with one orange chevron-like object near the top.
    - Other foreground pixels are isolated scatter markers in colors 1-4, with at most two such pixels per row.
    - A chevron-covered column is any column containing at least one orange pixel.

    Rule:
    - Erase every non-orange scatter pixel from the output.
    - A scatter pixel activates its own column only if its column is chevron-covered and its row contains another scatter pixel of the same color.
    - For each activated column, fill from the row just below that column's lowest orange pixel to the bottom using the color of the highest activating scatter pixel in that column.

    Key insight:
    - Row pairing and chevron coverage are independent tests, and a column fills only where both tests hold for the same scatter pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    qualifying, _row_counts, _covered, bottoms = _qualifying_pixels(g)
    out = g.copy()
    out[(out != BG) & (out != CHEVRON)] = BG

    best_by_col: dict[int, tuple[int, int]] = {}
    for r, c, color in qualifying:
        best = best_by_col.get(c)
        if best is None or r < best[0]:
            best_by_col[c] = (r, color)

    for c, (_r, color) in best_by_col.items():
        out[bottoms[c] + 1 :, c] = color

    return out


def _choose_color(exclude: tuple[int, ...] = ()) -> int:
    options = [color for color in SCATTER_COLORS if color not in exclude]
    return random.choice(options)


def _paint_row(grid: np.ndarray, row: int, placements: list[tuple[int, int]]) -> None:
    cols = [c for c, _color in placements]
    assert len(cols) == len(set(cols)), "a row cannot reuse the same column"
    for c, color in placements:
        assert grid[row, c] == BG, "scatter pixels cannot overlap"
        grid[row, c] = color


def _build_case() -> np.ndarray | None:
    height = random.randint(13, 16)
    width = random.randint(14, 18)
    span_options = [span for span in range(7, min(11, width - 4) + 1, 2)]
    if not span_options:
        return None

    span = random.choice(span_options)
    top = random.randint(0, 2)
    left = random.randint(2, width - span - 2)
    orientation = random.choice(("down", "up"))
    chevron = _chevron_coords(top, left, span, orientation)

    grid = np.zeros((height, width), dtype=int)
    for r, c in chevron:
        grid[r, c] = CHEVRON

    covered_cols = sorted({c for _, c in chevron})
    outside_cols = [c for c in range(width) if c not in covered_cols]
    if len(covered_cols) < 5 or len(outside_cols) < 3:
        return None

    lowest_orange = max(r for r, _c in chevron)
    available_rows = list(range(lowest_orange + 2, height))
    extra_rows = random.randint(0, 2)
    if len(available_rows) < 5 + extra_rows:
        return None

    rows = sorted(random.sample(available_rows, 5 + extra_rows))
    r_top, r_lower, r_pair_only, r_mask_only, r_mixed = rows[:5]

    shared_col = random.choice(covered_cols[1:-1] or covered_cols)
    mask_mode = random.choice(("single", "mismatch"))
    needed_covered = 4 + (1 if mask_mode == "mismatch" else 0)
    other_covered = [c for c in covered_cols if c != shared_col]
    if len(other_covered) < needed_covered:
        return None
    chosen_covered = random.sample(other_covered, needed_covered)
    c_top_other, c_lower_other, c_mixed, c_mask_a = chosen_covered[:4]
    c_mask_b = chosen_covered[4] if mask_mode == "mismatch" else None

    outside_pair_a, outside_pair_b, outside_mixed = random.sample(outside_cols, 3)

    color_top = _choose_color()
    color_lower = _choose_color((color_top,))
    color_pair_only = _choose_color()
    color_mixed = _choose_color()
    color_mask_a = _choose_color()
    color_mask_b = _choose_color((color_mask_a,)) if mask_mode == "mismatch" else None

    _paint_row(grid, r_top, [(shared_col, color_top), (c_top_other, color_top)])
    _paint_row(grid, r_lower, [(shared_col, color_lower), (c_lower_other, color_lower)])
    _paint_row(grid, r_pair_only, [(outside_pair_a, color_pair_only), (outside_pair_b, color_pair_only)])

    if mask_mode == "single":
        _paint_row(grid, r_mask_only, [(c_mask_a, color_mask_a)])
    else:
        _paint_row(grid, r_mask_only, [(c_mask_a, color_mask_a), (c_mask_b, color_mask_b)])

    _paint_row(grid, r_mixed, [(c_mixed, color_mixed), (outside_mixed, color_mixed)])

    for row in rows[5:]:
        extra_kind = random.choice(
            ("single_outside", "single_covered", "pair_outside", "pair_covered", "mismatch_mixed")
        )
        if extra_kind == "single_outside":
            _paint_row(grid, row, [(random.choice(outside_cols), _choose_color())])
        elif extra_kind == "single_covered":
            _paint_row(grid, row, [(random.choice(other_covered), _choose_color())])
        elif extra_kind == "pair_outside":
            c0, c1 = random.sample(outside_cols, 2)
            color = _choose_color()
            _paint_row(grid, row, [(c0, color), (c1, color)])
        elif extra_kind == "pair_covered":
            c0, c1 = random.sample(other_covered, 2)
            color = _choose_color()
            _paint_row(grid, row, [(c0, color), (c1, color)])
        else:
            c0 = random.choice(other_covered)
            c1 = random.choice(outside_cols)
            color0 = _choose_color()
            color1 = _choose_color((color0,))
            _paint_row(grid, row, [(c0, color0), (c1, color1)])

    return grid


def _validate_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    chevron_coords = np.argwhere(g == CHEVRON)
    if len(chevron_coords) == 0:
        return False

    scatter = _scatter_pixels(g)
    row_usage: dict[int, int] = {}
    for r, _c, _color in scatter:
        row_usage[r] = row_usage.get(r, 0) + 1
    if any(count > 2 for count in row_usage.values()):
        return False

    qualifying, row_counts, covered, bottoms = _qualifying_pixels(g)
    qualified_set = {(r, c, color) for r, c, color in qualifying}

    a_only = 0
    b_only = 0
    for r, c, color in scatter:
        in_covered_col = c in covered
        has_same_row_pair = row_counts[r].get(color, 0) >= 2
        if in_covered_col and has_same_row_pair:
            continue
        if in_covered_col:
            a_only += 1
        if has_same_row_pair:
            b_only += 1

    if len(qualifying) < 2 or a_only < 1 or b_only < 1:
        return False

    by_col: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in qualifying:
        by_col.setdefault(c, []).append((r, color))
    conflict_cols = [
        c
        for c, items in by_col.items()
        if len(items) >= 2 and len({color for _r, color in items}) >= 2
    ]
    if not conflict_cols:
        return False

    expected = _oracle(g, require_pair=True)
    if not np.array_equal(main(g), expected):
        return False

    mask_only = _oracle(g, require_pair=False)
    if np.array_equal(expected, mask_only):
        return False

    for c in conflict_cols:
        items = sorted(by_col[c])
        top_color = items[0][1]
        start = bottoms[c] + 1
        if not np.all(expected[start:, c] == top_color):
            return False

    if np.array_equal(expected, g):
        return False

    if len({tuple(row) for row in expected.tolist()}) <= 1:
        return False

    return bool(qualified_set)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None and _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

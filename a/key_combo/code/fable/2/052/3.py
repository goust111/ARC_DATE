from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
STRONG_SHAPES = ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (4, 5), (5, 4))
BOUNDARY_SHAPES = ((2, 4), (4, 2), (2, 5), (5, 2))
GEN_ATTEMPTS = 500


def _extract_palette(grid: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    fg = [int(v) for v in values if int(v) != bg]
    assert len(fg) == 1, "expected exactly one foreground color"
    return bg, fg[0]


def _point_set(grid: np.ndarray, bg: int) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(grid != bg)}


def _indices_from_mask(mask: int, size: int) -> tuple[int, ...]:
    return tuple(i for i in range(size) if (mask >> i) & 1)


def _closed_bicliques_from_row_masks(row_masks: list[int], n_cols: int) -> list[dict]:
    full_mask = (1 << n_cols) - 1
    seen = {full_mask}
    stack = [full_mask]
    while stack:
        col_mask = stack.pop()
        for row_mask in row_masks:
            nxt = col_mask & row_mask
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    out = []
    for col_mask in seen:
        col_count = int(col_mask.bit_count())
        if col_count < 2:
            continue
        row_bits = 0
        for idx, row_mask in enumerate(row_masks):
            if row_mask & col_mask == col_mask:
                row_bits |= 1 << idx
        row_count = int(row_bits.bit_count())
        if row_count < 2:
            continue
        row_idx = _indices_from_mask(row_bits, len(row_masks))
        col_idx = _indices_from_mask(col_mask, n_cols)
        out.append({"rows_idx": row_idx, "cols_idx": col_idx, "area": row_count * col_count})
    return out


def _best_complete_product(points: set[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, ...], list[dict]]:
    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    row_to_idx = {r: i for i, r in enumerate(rows)}
    col_to_idx = {c: i for i, c in enumerate(cols)}

    row_masks = [0] * len(rows)
    for r, c in points:
        row_masks[row_to_idx[r]] |= 1 << col_to_idx[c]

    if len(cols) <= len(rows):
        candidates = _closed_bicliques_from_row_masks(row_masks, len(cols))
        actual = [
            {
                "rows": tuple(rows[i] for i in cand["rows_idx"]),
                "cols": tuple(cols[j] for j in cand["cols_idx"]),
                "area": cand["area"],
            }
            for cand in candidates
        ]
    else:
        col_masks = [0] * len(cols)
        for r, c in points:
            col_masks[col_to_idx[c]] |= 1 << row_to_idx[r]
        dual = _closed_bicliques_from_row_masks(col_masks, len(rows))
        actual = [
            {
                "rows": tuple(rows[j] for j in cand["cols_idx"]),
                "cols": tuple(cols[i] for i in cand["rows_idx"]),
                "area": cand["area"],
            }
            for cand in dual
        ]

    dedup = {}
    for cand in actual:
        key = (cand["rows"], cand["cols"])
        dedup[key] = cand
    candidates = list(dedup.values())
    assert candidates, "expected at least one 2x2 complete product"

    max_area = max(cand["area"] for cand in candidates)
    best = [cand for cand in candidates if cand["area"] == max_area]
    assert len(best) == 1, "largest complete product must be unique"
    return best[0]["rows"], best[0]["cols"], candidates


def _render_solution(shape: tuple[int, int], rows: tuple[int, ...], cols: tuple[int, ...], fg: int, bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    for r in rows:
        out[r, c0 : c1 + 1] = fg
    for c in cols:
        out[r0 : r1 + 1, c] = fg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unique largest hidden row-column intersection table and erase off-line noise.

    Input structure:
    - A black background contains exactly one non-black color, appearing only as single-pixel dots.
    - Some dots are all intersections of a hidden row set R and column set C, with |R|, |C| >= 2.
    - Extra same-color dots act as noise; some align with true rows or columns, but the maximal complete Cartesian product is unique.

    Rule:
    - Interpret the dots as a bipartite row-column incidence graph and find the unique largest subset equal to R x C exactly.
    - For every row in R, draw a horizontal segment from the leftmost true intersection column to the rightmost true intersection column.
    - For every column in C, draw a vertical segment from the topmost true intersection row to the bottommost true intersection row, and erase every other dot not covered by those segments.

    Key insight:
    - A real hidden line is certified only by a fully completed intersection table with the other hidden lines; mere alignment of a few noisy dots is insufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, fg = _extract_palette(g)
    points = _point_set(g, bg)
    assert points, "expected at least one foreground dot"

    rows, cols, _ = _best_complete_product(points)
    return _render_solution(g.shape, rows, cols, fg, bg)


def _sample_axis(count: int) -> tuple[int, tuple[int, ...]] | None:
    for _ in range(120):
        gaps = [random.randint(2, 5) for _ in range(count - 1)]
        span = 1 + sum(gaps)
        size_lo = max(12, span + 3)
        size_hi = min(30, span + 10)
        if size_lo > size_hi:
            continue
        size = random.randint(size_lo, size_hi)
        start = random.randint(1, size - span - 1)
        positions = [start]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        return size, tuple(positions)
    return None


def _max_extra_row_cols(n_rows: int, n_cols: int) -> int:
    return min(n_cols - 1, (n_rows * n_cols - 1) // (n_rows + 1))


def _max_extra_col_rows(n_rows: int, n_cols: int) -> int:
    return min(n_rows - 1, (n_rows * n_cols - 1) // (n_cols + 1))


def _choose_shape() -> tuple[int, int]:
    if random.random() < 0.25:
        return random.choice(BOUNDARY_SHAPES)
    return random.choice(STRONG_SHAPES)


def _line_mask(shape: tuple[int, int], rows: tuple[int, ...], cols: tuple[int, ...]) -> np.ndarray:
    return _render_solution(shape, rows, cols, 1, 0).astype(bool)


def _build_attempt() -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], int, dict] | None:
    n_rows, n_cols = _choose_shape()
    row_axis = _sample_axis(n_rows)
    col_axis = _sample_axis(n_cols)
    if row_axis is None or col_axis is None:
        return None
    height, rows = row_axis
    width, cols = col_axis
    fg = random.choice(COLOR_POOL)

    row_set = set(rows)
    col_set = set(cols)
    points = {(r, c) for r in rows for c in cols}
    meta = {"survivor_noise": set(), "tail_noise": set(), "erased_noise": set()}

    inside_rows = [r for r in range(rows[0], rows[-1] + 1) if r not in row_set]
    inside_cols = [c for c in range(cols[0], cols[-1] + 1) if c not in col_set]
    outside_rows = [r for r in range(height) if r not in row_set and not (rows[0] <= r <= rows[-1])]
    outside_cols = [c for c in range(width) if c not in col_set and not (cols[0] <= c <= cols[-1])]
    if not inside_rows or not inside_cols or not outside_rows or not outside_cols:
        return None

    extra_row = random.choice(inside_rows)
    extra_col = random.choice(inside_cols)
    max_row_cols = _max_extra_row_cols(n_rows, n_cols)
    max_col_rows = _max_extra_col_rows(n_rows, n_cols)
    row_subset_size = random.randint(2, max_row_cols) if max_row_cols >= 2 else 1
    col_subset_size = random.randint(2, max_col_rows) if max_col_rows >= 2 else 1
    for c in random.sample(list(cols), row_subset_size):
        points.add((extra_row, c))
        meta["survivor_noise"].add((extra_row, c))
    for r in random.sample(list(rows), col_subset_size):
        points.add((r, extra_col))
        meta["survivor_noise"].add((r, extra_col))

    tail_row = random.choice(rows)
    tail_col = random.choice(outside_cols)
    points.add((tail_row, tail_col))
    meta["tail_noise"].add((tail_row, tail_col))
    meta["erased_noise"].add((tail_row, tail_col))

    tail_row2 = random.choice(outside_rows)
    tail_col2 = random.choice(cols)
    points.add((tail_row2, tail_col2))
    meta["tail_noise"].add((tail_row2, tail_col2))
    meta["erased_noise"].add((tail_row2, tail_col2))

    reserved_rows = row_set | {extra_row, tail_row2}
    reserved_cols = col_set | {extra_col, tail_col}
    isolated = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r not in reserved_rows and c not in reserved_cols
    ]
    if not isolated:
        return None
    random.shuffle(isolated)
    for cell in isolated[: random.randint(1, min(5, len(isolated)))]:
        points.add(cell)
        meta["erased_noise"].add(cell)

    grid = np.zeros((height, width), dtype=int)
    for r, c in points:
        grid[r, c] = fg
    return grid, rows, cols, fg, meta


def _validate_instance(grid: np.ndarray, rows: tuple[int, ...], cols: tuple[int, ...], fg: int, meta: dict) -> bool:
    bg, got_fg = _extract_palette(grid)
    assert bg == BG and got_fg == fg

    points = _point_set(grid, bg)
    got_rows, got_cols, candidates = _best_complete_product(points)
    assert got_rows == rows and got_cols == cols

    areas = sorted((cand["area"] for cand in candidates), reverse=True)
    if len(areas) > 1:
        assert areas[0] > areas[1], "expected a strict maximum"

    expected = _render_solution(grid.shape, rows, cols, fg, bg)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    product = {(r, c) for r in rows for c in cols}
    on_lines = _line_mask(grid.shape, rows, cols)
    survivor_noise = {p for p in points if p not in product and on_lines[p]}
    erased_noise = {p for p in points if p not in product and not on_lines[p]}

    assert survivor_noise, "need at least one noisy point preserved by a drawn segment"
    assert erased_noise, "need at least one noisy point erased in the output"
    assert meta["survivor_noise"] <= survivor_noise
    assert meta["tail_noise"] <= erased_noise
    assert meta["erased_noise"] <= erased_noise | survivor_noise
    assert any(p[0] in rows or p[1] in cols for p in erased_noise), "need endpoint-tail noise that aligned but gets erased"

    for r in rows:
        assert np.all(expected[r, cols[0] : cols[-1] + 1] == fg)
    for c in cols:
        assert np.all(expected[rows[0] : rows[-1] + 1, c] == fg)
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, rows, cols, fg, meta = built
        if _validate_instance(grid, rows, cols, fg, meta):
            return grid
    raise RuntimeError("failed to generate a valid hidden-grid reconstruction input")

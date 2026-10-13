from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3


GRID_H_RANGE = (14, 19)
GRID_W_RANGE = (16, 22)
BRANCH_COUNT_RANGE = (2, 4)
BAR_COUNT_RANGE = (3, 5)
SPINE_TOTAL_LEN_RANGE = (6, 11)
BRANCH_OFFSET_RANGE = (2, 5)
GEN_RETRY_LIMIT = 400


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _find_spine(red_coords: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in red_coords:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    row_spines = [(r, sorted(cs)) for r, cs in rows.items() if len(cs) == 2]
    col_spines = [(c, sorted(rs)) for c, rs in cols.items() if len(rs) == 2]
    assert len(row_spines) + len(col_spines) == 1, "expected exactly one aligned red pair"

    if row_spines:
        row, cols_on_row = row_spines[0]
        return "row", int(row), int(cols_on_row[0]), int(cols_on_row[1])

    col, rows_on_col = col_spines[0]
    return "col", int(col), int(rows_on_col[0]), int(rows_on_col[1])


def _green_network(
    shape: tuple[int, int],
    axis: str,
    fixed: int,
    lo: int,
    hi: int,
    branches: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[int]]:
    green: set[tuple[int, int]] = set()
    active: set[int] = set()

    if axis == "row":
        for c in range(lo + 1, hi):
            green.add((fixed, c))
        for r, c in branches:
            assert lo < c < hi
            active.add(c)
            r0, r1 = sorted((r, fixed))
            for rr in range(r0 + 1, r1):
                green.add((rr, c))
        return green, active

    for r in range(lo + 1, hi):
        green.add((r, fixed))
    for r, c in branches:
        assert lo < r < hi
        active.add(r)
        c0, c1 = sorted((c, fixed))
        for cc in range(c0 + 1, c1):
            green.add((r, cc))
    return green, active


def _paint_green(out: np.ndarray, green: set[tuple[int, int]]) -> None:
    for r, c in green:
        if int(out[r, c]) == BLACK:
            out[r, c] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the unique green spine-and-branch network, then use its perpendicular channels to select which dashed blue gaps turn red.

    Input structure:
    - A black background contains red control pixels and several horizontal blue dashed bars.
    - Exactly two red pixels share one row or one column; they define the spine endpoints.
    - Every other red pixel lies inside the spine span so it can connect orthogonally without ambiguity.

    Rule:
    - Fill the black cells strictly between the aligned red endpoints with green, keeping the red endpoints unchanged.
    - Connect each non-spine red pixel to the spine with a green orthogonal segment over black cells only.
    - Each resulting perpendicular connector activates one column (for a horizontal spine) or one row (for a vertical spine); recolor a dashed bar's one-cell black gap to red exactly when it lies on an activated column or row.

    Key insight:
    - The green network is a spatial gate: first identify the unique spine, then use the attached branch connectors as the mask that decides which dashed gaps qualify for red completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected input colors"

    red_coords = [(int(r), int(c)) for r, c in np.argwhere(g == RED)]
    assert len(red_coords) >= 4, "expected the red spine controls plus branches"

    axis, fixed, lo, hi = _find_spine(red_coords)
    branches = [(r, c) for r, c in red_coords if (r != fixed if axis == "row" else c != fixed)]
    green, active = _green_network(g.shape, axis, fixed, lo, hi, branches)

    out = g.copy()
    _paint_green(out, green)

    gap_mask = np.zeros_like(g, dtype=bool)
    gap_mask[:, 1:-1] = (
        (g[:, 1:-1] == BLACK)
        & (g[:, :-2] == BLUE)
        & (g[:, 2:] == BLUE)
    )

    if axis == "row":
        active_mask = np.zeros(g.shape[1], dtype=bool)
        for c in active:
            active_mask[int(c)] = True
        fill_mask = gap_mask & active_mask[None, :]
    else:
        active_mask = np.zeros(g.shape[0], dtype=bool)
        for r in active:
            active_mask[int(r)] = True
        fill_mask = gap_mask & active_mask[:, None]

    out[fill_mask] = RED
    return out


def _choose_spine_length(parallel_dim: int, branch_count: int) -> int | None:
    min_len = max(SPINE_TOTAL_LEN_RANGE[0], branch_count + 2)
    max_len = min(SPINE_TOTAL_LEN_RANGE[1], parallel_dim - 2)
    if min_len > max_len:
        return None
    return _randint(min_len, max_len)


def _offset_options(fixed: int, size: int) -> tuple[list[int], list[int]]:
    neg = [d for d in range(BRANCH_OFFSET_RANGE[0], BRANCH_OFFSET_RANGE[1] + 1) if fixed - d >= 0]
    pos = [d for d in range(BRANCH_OFFSET_RANGE[0], BRANCH_OFFSET_RANGE[1] + 1) if fixed + d < size]
    return neg, pos


def _pick_offsets(fixed: int, size: int, branch_count: int) -> list[int] | None:
    neg, pos = _offset_options(fixed, size)
    lo = max(1, branch_count - len(pos))
    hi = min(len(neg), branch_count - 1)
    if lo > hi:
        return None

    n_neg = _randint(lo, hi)
    n_pos = branch_count - n_neg
    neg_pick = list(np.random.choice(np.asarray(neg, dtype=int), size=n_neg, replace=False))
    pos_pick = list(np.random.choice(np.asarray(pos, dtype=int), size=n_pos, replace=False))
    offsets = [-int(d) for d in neg_pick] + [int(d) for d in pos_pick]
    np.random.shuffle(offsets)
    return offsets


def _sample_layout(axis: str, h: int, w: int, branch_count: int):
    parallel_dim = w if axis == "row" else h
    perp_dim = h if axis == "row" else w
    spine_len = _choose_spine_length(parallel_dim, branch_count)
    if spine_len is None:
        return None

    fixed = _randint(2, perp_dim - 3)
    offsets = _pick_offsets(fixed, perp_dim, branch_count)
    if offsets is None:
        return None

    lo = _randint(1, parallel_dim - spine_len - 1)
    hi = lo + spine_len - 1
    attach_positions = list(
        np.random.choice(np.arange(lo + 1, hi, dtype=int), size=branch_count, replace=False)
    )
    return fixed, lo, hi, attach_positions, offsets


def _branch_coords(axis: str, fixed: int, attach_positions: list[int], offsets: list[int]) -> list[tuple[int, int]]:
    coords = []
    for attach, offset in zip(attach_positions, offsets):
        if axis == "row":
            coords.append((fixed + int(offset), int(attach)))
        else:
            coords.append((int(attach), fixed + int(offset)))
    return coords


def _build_red_grid(
    h: int,
    w: int,
    axis: str,
    fixed: int,
    lo: int,
    hi: int,
    branches: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    if axis == "row":
        grid[fixed, lo] = RED
        grid[fixed, hi] = RED
    else:
        grid[lo, fixed] = RED
        grid[hi, fixed] = RED
    for r, c in branches:
        grid[r, c] = RED
    return grid


def _max_spaced_count(rows: list[int]) -> int:
    count = 0
    last = -10
    for row in sorted(rows):
        if row - last >= 2:
            count += 1
            last = row
    return count


def _pick_spaced_rows(candidates: list[int], count: int) -> list[int] | None:
    if len(candidates) < count:
        return None
    arr = np.asarray(sorted(candidates), dtype=int)
    for _ in range(200):
        pick = sorted(int(v) for v in np.random.choice(arr, size=count, replace=False))
        if all(b - a >= 2 for a, b in zip(pick, pick[1:])):
            return pick
    return None


def _row_candidates(
    row: int,
    width: int,
    red_cells: set[tuple[int, int]],
    green_cells: set[tuple[int, int]],
) -> list[tuple[int, list[int], list[int]]]:
    red_cols = {c for rr, c in red_cells if rr == row}
    green_cols = {c for rr, c in green_cells if rr == row}
    candidates: list[tuple[int, list[int], list[int]]] = []
    max_blue = min(6, (width + 1) // 2)

    for blue_count in range(3, max_blue + 1):
        span = 2 * blue_count - 1
        for start in range(0, width - span + 1):
            span_cols = range(start, start + span)
            blue_cols = list(range(start, start + span, 2))
            gap_cols = list(range(start + 1, start + span - 1, 2))
            if any(c in red_cols for c in span_cols):
                continue
            if any(c in green_cols for c in blue_cols):
                continue
            candidates.append((start, blue_cols, gap_cols))
    return candidates


def _place_horizontal_bars(
    h: int,
    w: int,
    red_cells: set[tuple[int, int]],
    green_cells: set[tuple[int, int]],
    active_cols: set[int],
    forbidden_rows: set[int],
) -> list[tuple[int, int, list[int], list[int]]] | None:
    available_rows = [r for r in range(h) if r not in forbidden_rows]
    max_count = min(BAR_COUNT_RANGE[1], _max_spaced_count(available_rows))
    if max_count < BAR_COUNT_RANGE[0]:
        return None

    bar_count = _randint(BAR_COUNT_RANGE[0], max_count)
    bar_rows = _pick_spaced_rows(available_rows, bar_count)
    if bar_rows is None:
        return None

    row_to_candidates = {row: _row_candidates(row, w, red_cells, green_cells) for row in bar_rows}
    if any(not cands for cands in row_to_candidates.values()):
        return None

    for _ in range(250):
        bars = []
        rows_with_fill = 0
        rows_without_fill = 0
        total_filled = 0
        total_unfilled = 0

        for row in bar_rows:
            start, blue_cols, gap_cols = row_to_candidates[row][_randint(0, len(row_to_candidates[row]) - 1)]
            filled_here = [c for c in gap_cols if c in active_cols]
            unfilled_here = [c for c in gap_cols if c not in active_cols]
            rows_with_fill += int(bool(filled_here))
            rows_without_fill += int(bool(unfilled_here))
            total_filled += len(filled_here)
            total_unfilled += len(unfilled_here)
            bars.append((row, start, blue_cols, gap_cols))

        if rows_with_fill >= 1 and rows_without_fill >= 1 and total_filled >= 2 and total_unfilled >= 2:
            return bars
    return None


def _place_vertical_bars(
    h: int,
    w: int,
    red_cells: set[tuple[int, int]],
    green_cells: set[tuple[int, int]],
    active_rows: set[int],
    trunk_span_rows: set[int],
    forbidden_rows: set[int],
) -> list[tuple[int, int, list[int], list[int]]] | None:
    inactive_rows = [r for r in range(h) if r not in trunk_span_rows and r not in forbidden_rows]
    if not active_rows or not inactive_rows:
        return None

    union_rows = sorted(active_rows | set(inactive_rows))
    max_count = min(BAR_COUNT_RANGE[1], _max_spaced_count(union_rows))
    if max_count < BAR_COUNT_RANGE[0]:
        return None

    for _ in range(250):
        bar_count = _randint(BAR_COUNT_RANGE[0], max_count)
        rows = _pick_spaced_rows(union_rows, bar_count)
        if rows is None:
            continue
        if not any(r in active_rows for r in rows):
            continue
        if not any(r in inactive_rows for r in rows):
            continue

        bars = []
        ok = True
        for row in rows:
            candidates = _row_candidates(row, w, red_cells, green_cells)
            if not candidates:
                ok = False
                break
            bars.append((row, *candidates[_randint(0, len(candidates) - 1)]))
        if ok:
            return bars
    return None


def _paint_bars(grid: np.ndarray, bars: list[tuple[int, int, list[int], list[int]]]) -> None:
    for row, _start, blue_cols, _gap_cols in bars:
        for col in blue_cols:
            grid[row, col] = BLUE


def _attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)
    axis = "row" if _randint(0, 1) == 0 else "col"
    branch_count = _randint(*BRANCH_COUNT_RANGE)

    layout = _sample_layout(axis, h, w, branch_count)
    if layout is None:
        return None
    fixed, lo, hi, attach_positions, offsets = layout
    branches = _branch_coords(axis, fixed, attach_positions, offsets)
    grid = _build_red_grid(h, w, axis, fixed, lo, hi, branches)

    green_cells, active = _green_network((h, w), axis, fixed, lo, hi, branches)
    red_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == RED)}

    if axis == "row":
        forbidden_rows = {fixed} | {r for r, _c in branches}
        bars = _place_horizontal_bars(h, w, red_cells, green_cells, active, forbidden_rows)
    else:
        trunk_span_rows = set(range(lo + 1, hi))
        forbidden_rows = {lo, hi}
        bars = _place_vertical_bars(h, w, red_cells, green_cells, active, trunk_span_rows, forbidden_rows)
    if bars is None:
        return None

    _paint_bars(grid, bars)
    out = main(grid)

    new_red = np.argwhere((grid != RED) & (out == RED))
    new_green = np.argwhere((grid == BLACK) & (out == GREEN))
    if len(new_red) == 0 or len(new_green) == 0:
        return None
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated dashed-bar input")

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PATH = 3
SEP = 5
ODD_COLOR = 2
EVEN_COLOR = 1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 400


def _neighbors4(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _path_degrees(points: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {p: sum((q in points) for q in _neighbors4(p)) for p in points}


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert pts, "expected a non-empty path"

    deg = _path_degrees(pts)
    ends = sorted([p for p, d in deg.items() if d == 1])
    assert len(ends) == 2, "expected exactly two endpoints"

    order: list[tuple[int, int]] = []
    prev = None
    cur = ends[0]
    while True:
        order.append(cur)
        nxts = [q for q in _neighbors4(cur) if q in pts and q != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]

    assert len(order) == len(pts), "path must be a single simple chain"
    return order


def _detect_separators(grid: np.ndarray) -> tuple[int, tuple[int, ...]]:
    row_mask = np.all(np.isin(grid, [PATH, SEP]), axis=1) & np.any(grid == SEP, axis=1)
    col_mask = np.all(np.isin(grid, [PATH, SEP]), axis=0) & np.any(grid == SEP, axis=0)
    row_lines = tuple(int(i) for i in np.flatnonzero(row_mask))
    col_lines = tuple(int(i) for i in np.flatnonzero(col_mask))

    if row_lines and not col_lines:
        return 0, row_lines
    if col_lines and not row_lines:
        return 1, col_lines
    raise AssertionError("expected 1-3 non-crossing separator lines with a single orientation")


def _coord_on_line(cell: tuple[int, int], axis: int, line_positions: set[int]) -> bool:
    return cell[axis] in line_positions


def _side_index(cell: tuple[int, int], axis: int, line_positions: tuple[int, ...]) -> int:
    coord = cell[axis]
    return sum(pos < coord for pos in line_positions)


def _starts_new_segment(
    prev: tuple[int, int],
    cur: tuple[int, int],
    axis: int,
    line_positions: tuple[int, ...],
) -> bool:
    line_set = set(line_positions)
    prev_on = _coord_on_line(prev, axis, line_set)
    cur_on = _coord_on_line(cur, axis, line_set)
    if cur_on and (not prev_on or prev[axis] != cur[axis]):
        return True
    if not prev_on and not cur_on:
        return _side_index(prev, axis, line_positions) != _side_index(cur, axis, line_positions)
    return False


def _recolor_straightened(order: list[tuple[int, int]], axis: int, line_positions: tuple[int, ...]) -> np.ndarray:
    colors: list[int] = []
    segment_id = 1
    for i, cell in enumerate(order):
        if i > 0 and _starts_new_segment(order[i - 1], cell, axis, line_positions):
            segment_id += 1
        colors.append(ODD_COLOR if segment_id % 2 == 1 else EVEN_COLOR)
    return np.array(colors, dtype=int).reshape((-1, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a separator-cut snake path into a single column with alternating segment colors.

    Input structure:
    - The background is black.
    - Grey pixels form 1-3 complete horizontal or vertical separator lines of width 1.
    - A single green 4-connected snake path crosses those separators, and crossing pixels stay green.

    Rule:
    - Detect the snake path and choose the endpoint with smaller row, then smaller column, as the start.
    - Traverse the path in order; whenever the traversal enters a separator crossing pixel, begin a new segment.
    - Recolor odd-numbered segments red and even-numbered segments blue, then straighten the path into an N×1 column in traversal order.

    Key insight:
    - The separator lines are not output directly; they act only as ordered cut points that partition one traced path into alternating recolor segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    axis, line_positions = _detect_separators(g)
    order = _trace_path(g == PATH)
    return _recolor_straightened(order, axis, line_positions)


def _random_composition(total: int, parts: int, minimum: int) -> list[int] | None:
    base = parts * minimum
    if base > total:
        return None
    vals = [minimum] * parts
    extra = total - base
    for _ in range(extra):
        vals[random.randrange(parts)] += 1
    return vals


def _append_step(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    nxt: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    H, W = shape
    r, c = nxt
    if not (0 <= r < H and 0 <= c < W):
        return False
    if nxt in occupied:
        return False
    prev = path[-1]
    for nb in _neighbors4(nxt):
        if nb in occupied and nb != prev:
            return False
    path.append(nxt)
    occupied.add(nxt)
    return True


def _append_line(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    while path[-1] != target:
        r, c = path[-1]
        tr, tc = target
        if r != tr:
            step = (r + (1 if tr > r else -1), c)
        else:
            step = (r, c + (1 if tc > c else -1))
        if not _append_step(path, occupied, step, shape):
            return False
    return True


def _separator_rows(H: int, count: int) -> list[int] | None:
    interior = H - 2 - count
    band_sizes = _random_composition(interior, count + 1, minimum=2)
    if band_sizes is None:
        return None
    rows: list[int] = []
    r = 1
    for band in band_sizes[:-1]:
        r += band
        rows.append(r)
        r += 1
    return rows


def _band_ranges(H: int, sep_rows: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 1
    for row in sep_rows:
        bands.append((start, row - 1))
        start = row + 1
    bands.append((start, H - 2))
    return bands


def _pick_band_rows(bands: list[tuple[int, int]]) -> list[int]:
    rows = []
    for lo, hi in bands:
        rows.append(random.randint(lo, hi))
    return rows


def _pick_crossing_columns(W: int, count: int) -> list[int] | None:
    cols = [random.randint(2, W - 3)]
    direction = random.choice([-1, 1])
    for _ in range(1, count):
        candidates = []
        for delta in range(2, 5):
            nc = cols[-1] + direction * delta
            if 2 <= nc <= W - 3:
                candidates.append(nc)
        if not candidates:
            return None
        cols.append(random.choice(candidates))
        if random.random() < 0.8:
            direction *= -1
    return cols


def _build_multi_separator_path(H: int, W: int, sep_rows: list[int]) -> list[tuple[int, int]] | None:
    bands = _band_ranges(H, sep_rows)
    band_rows = _pick_band_rows(bands)
    crossing_cols = _pick_crossing_columns(W, len(sep_rows))
    if crossing_cols is None:
        return None

    start_col = max(1, crossing_cols[0] - random.randint(0, 2))
    end_col = min(W - 2, crossing_cols[-1] + random.randint(1, 3))

    path = [(band_rows[0], start_col)]
    occupied = {path[0]}

    if not _append_line(path, occupied, (band_rows[0], crossing_cols[0]), (H, W)):
        return None

    for i, sep_row in enumerate(sep_rows):
        if not _append_line(path, occupied, (sep_row - 1, crossing_cols[i]), (H, W)):
            return None
        if not _append_line(path, occupied, (sep_row, crossing_cols[i]), (H, W)):
            return None
        if not _append_line(path, occupied, (sep_row + 1, crossing_cols[i]), (H, W)):
            return None
        if not _append_line(path, occupied, (band_rows[i + 1], crossing_cols[i]), (H, W)):
            return None
        if i + 1 < len(sep_rows):
            if not _append_line(path, occupied, (band_rows[i + 1], crossing_cols[i + 1]), (H, W)):
                return None

    if not _append_line(path, occupied, (band_rows[-1], end_col), (H, W)):
        return None
    return path


def _build_single_separator_path(H: int, W: int, sep_row: int) -> list[tuple[int, int]] | None:
    top_band = (1, sep_row - 1)
    bottom_band = (sep_row + 1, H - 2)
    if top_band[0] > top_band[1] or bottom_band[0] > bottom_band[1]:
        return None

    top_row = random.randint(*top_band)
    bottom_row = random.randint(*bottom_band)
    c1 = random.randint(2, W - 6)
    c2 = random.randint(c1 + 3, W - 3)
    start_col = max(1, c1 - random.randint(0, 2))
    end_col = min(W - 2, c2 + random.randint(1, 3))

    path = [(top_row, start_col)]
    occupied = {path[0]}

    checkpoints = [
        (top_row, c1),
        (sep_row - 1, c1),
        (sep_row, c1),
        (sep_row + 1, c1),
        (bottom_row, c1),
        (bottom_row, c2),
        (sep_row + 1, c2),
        (sep_row, c2),
        (sep_row - 1, c2),
        (top_row, c2),
        (top_row, end_col),
    ]
    for target in checkpoints:
        if not _append_line(path, occupied, target, (H, W)):
            return None
    return path


def _place_grid(
    H: int,
    W: int,
    sep_rows: list[int],
    path: list[tuple[int, int]],
    transpose: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, ...], int]:
    if transpose:
        grid = np.full((W, H), BG, dtype=int)
        for row in sep_rows:
            grid[:, row] = SEP
        path_final = [(c, r) for r, c in path]
        axis = 1
    else:
        grid = np.full((H, W), BG, dtype=int)
        for row in sep_rows:
            grid[row, :] = SEP
        path_final = list(path)
        axis = 0

    for r, c in path_final:
        grid[r, c] = PATH

    line_positions = tuple(int(x) for x in sep_rows)
    if path_final[0] > path_final[-1]:
        path_final = list(reversed(path_final))
    return grid, path_final, line_positions, axis


def _crossing_count(order: list[tuple[int, int]], axis: int, line_positions: tuple[int, ...]) -> int:
    return sum(
        _starts_new_segment(order[i - 1], order[i], axis, line_positions)
        for i in range(1, len(order))
    )


def _validate_grid_structure(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BG, PATH, SEP}

    axis, line_positions = _detect_separators(grid)
    assert 1 <= len(line_positions) <= 3

    order = _trace_path(grid == PATH)
    assert 8 <= len(order) <= 30

    touched = {cell[axis] for cell in order if cell[axis] in set(line_positions)}
    assert touched == set(line_positions), "every separator must be crossed at least once"
    assert _crossing_count(order, axis, line_positions) >= 2

    pts = set(order)
    for cell in order:
        if cell[axis] not in line_positions:
            continue
        nbs = [q for q in _neighbors4(cell) if q in pts]
        assert len(nbs) == 2, "separator crossing cell must be internal"
        if axis == 0:
            assert {q[0] for q in nbs} == {cell[0] - 1, cell[0] + 1}
            assert all(q[1] == cell[1] for q in nbs)
        else:
            assert {q[1] for q in nbs} == {cell[1] - 1, cell[1] + 1}
            assert all(q[0] == cell[0] for q in nbs)


def _gen_attempt() -> np.ndarray | None:
    separator_count = random.choice([1, 2, 3])
    H = random.randint(max(12, 3 * separator_count + 4), 16)
    W = random.randint(10, 16)

    sep_rows = _separator_rows(H, separator_count)
    if sep_rows is None:
        return None

    if separator_count == 1:
        path = _build_single_separator_path(H, W, sep_rows[0])
    else:
        path = _build_multi_separator_path(H, W, sep_rows)
    if path is None:
        return None

    transpose = bool(random.getrandbits(1))
    grid, expected_order, line_positions, axis = _place_grid(H, W, sep_rows, path, transpose)
    expected = _recolor_straightened(expected_order, axis, line_positions)

    try:
        _validate_grid_structure(grid)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")

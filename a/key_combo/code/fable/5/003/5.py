from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
GEN_ATTEMPTS = 1000


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def _intervals_between_line_runs(runs: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    assert len(runs) >= 2, "expected at least outer separator lines"
    assert runs[0][0] == 0 and runs[-1][1] == int(length) - 1, "table must have an outer separator frame"
    intervals: list[tuple[int, int]] = []
    for (_a0, a1), (b0, _b1) in zip(runs, runs[1:]):
        start = int(a1) + 1
        end = int(b0) - 1
        assert start <= end, "separator lines must enclose non-empty cells"
        intervals.append((start, end))
    return intervals


def _find_table(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    H, W = map(int, grid.shape)
    for color in np.unique(grid):
        color = int(color)
        full_rows = [r for r in range(H) if bool(np.all(grid[r, :] == color))]
        full_cols = [c for c in range(W) if bool(np.all(grid[:, c] == color))]
        if len(full_rows) < 2 or len(full_cols) < 2:
            continue

        row_runs = _runs(full_rows)
        col_runs = _runs(full_cols)
        try:
            row_intervals = _intervals_between_line_runs(row_runs, H)
            col_intervals = _intervals_between_line_runs(col_runs, W)
        except AssertionError:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_intervals}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_intervals}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if min(row_sizes) < 2 or min(col_sizes) < 2:
            continue

        line_cells = sum(r1 - r0 + 1 for r0, r1 in row_runs) * W
        line_cells += sum(c1 - c0 + 1 for c0, c1 in col_runs) * H
        candidates.append((line_cells, len(row_intervals), len(col_intervals), row_intervals, col_intervals))

    assert candidates, "expected a framed separator table"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    ties = [cand for cand in candidates if cand[0] == best[0]]
    assert len(ties) == 1, "separator color must be unique"

    _score, _rows, _cols, row_intervals, col_intervals = best
    sep_color = None
    for color in np.unique(grid):
        color = int(color)
        full_rows = [r for r in range(H) if bool(np.all(grid[r, :] == color))]
        full_cols = [c for c in range(W) if bool(np.all(grid[:, c] == color))]
        if _runs(full_rows) and _runs(full_cols):
            try:
                if (
                    _intervals_between_line_runs(_runs(full_rows), H) == row_intervals
                    and _intervals_between_line_runs(_runs(full_cols), W) == col_intervals
                ):
                    sep_color = color
                    break
            except AssertionError:
                continue
    assert sep_color is not None
    return int(sep_color), row_intervals, col_intervals


def _dominant_in_cells(
    grid: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
    sep: int,
) -> int:
    vals: list[int] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            vals.extend(int(v) for v in grid[r0 : r1 + 1, c0 : c1 + 1].ravel() if int(v) != int(sep))
    counts = Counter(vals)
    assert counts, "expected non-separator cell interiors"
    bg, _count = counts.most_common(1)[0]
    return int(bg)


def _parse_cells(
    grid: np.ndarray,
    sep: int,
    bg: int,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rows = len(row_intervals)
    cols = len(col_intervals)
    body_colors = np.zeros((rows, cols), dtype=int)
    directions = np.zeros((rows, cols), dtype=int)
    start_cells: list[tuple[int, int]] = []
    global_counts = Counter(int(v) for v in grid.ravel())
    singletons = {
        color
        for color, count in global_counts.items()
        if int(count) == 1 and color not in {int(sep), int(bg), GREY}
    }

    for tr, (r0, r1) in enumerate(row_intervals):
        for tc, (c0, c1) in enumerate(col_intervals):
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            rel = np.argwhere(window != int(bg))
            assert rel.shape[0] == 2, "each cell must contain exactly a body pixel and a pointer pixel"
            vals = [int(window[tuple(rc)]) for rc in rel]
            assert int(sep) not in vals, "separator color cannot appear inside a cell"

            grey_indices = [i for i, v in enumerate(vals) if v == GREY]
            if len(grey_indices) == 1:
                pointer_idx = grey_indices[0]
                body_idx = 1 - pointer_idx
                assert vals[body_idx] not in {GREY, int(bg), int(sep)}
            else:
                assert len(grey_indices) == 0, "only non-start pointers may be grey"
                marker_indices = [i for i, v in enumerate(vals) if v in singletons]
                assert len(marker_indices) == 1, "start marker must be the unique singleton color in its cell"
                pointer_idx = marker_indices[0]
                body_idx = 1 - pointer_idx
                assert vals[body_idx] not in singletons, "start body color must not be confused with the marker"
                start_cells.append((tr, tc))

            br, bc = map(int, rel[body_idx])
            pr, pc = map(int, rel[pointer_idx])
            delta = (int(pr - br), int(pc - bc))
            assert delta in DIR_TO_INDEX, "pointer pixel must touch the body by a cardinal edge"
            body_colors[tr, tc] = int(vals[body_idx])
            directions[tr, tc] = int(DIR_TO_INDEX[delta])

    assert len(start_cells) == 1, "expected exactly one marked start cell"
    return body_colors, directions, start_cells[0]


def _simulate(directions: np.ndarray, start: tuple[int, int], max_steps: int = 10000) -> list[tuple[int, int]]:
    rows, cols = map(int, directions.shape)
    visits = np.zeros((rows, cols), dtype=int)
    first_order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    r, c = map(int, start)

    for _ in range(max_steps):
        if not (0 <= r < rows and 0 <= c < cols):
            return first_order
        pos = (int(r), int(c))
        if pos not in seen:
            seen.add(pos)
            first_order.append(pos)
        direction = int(directions[r, c] + visits[r, c]) % 4
        visits[r, c] += 1
        dr, dc = DIRS[direction]
        r += dr
        c += dc

    raise AssertionError("pointer walk did not leave the table")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a pointer table with hidden clockwise rewrites and return the first-visit color band.

    Input structure:
    - A single-color separator frame and separator lines divide the grid into
      equal-sized table cells.
    - Each cell interior has exactly one colored body pixel and one cardinally
      adjacent pointer pixel on the cell background.
    - Ordinary pointers are grey; the unique start cell uses a one-off marker
      color for its pointer instead of grey.

    Rule:
    - The pointer's offset from the body gives that cell's initial exit direction.
    - Starting at the marked cell, move from cell to adjacent cell.  On the first
      visit use the drawn direction; on each later visit to the same cell, rotate
      the previously used exit direction one quarter-turn clockwise.
    - Stop when the next move leaves the table.  Output one row containing the
      body colors in first-visit order, with revisited cells recorded only once.

    Key insight:
    - The grey/marked pointer is only the cell's initial state.  A loop cannot be
      solved by static arrow tracing; each cell needs its own visit count because
      revisits silently rewrite that cell's actual outgoing direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep, row_intervals, col_intervals = _find_table(g)
    bg = _dominant_in_cells(g, row_intervals, col_intervals, sep)
    assert GREY not in {int(sep), int(bg)}, "grey is reserved for ordinary pointers"

    body_colors, directions, start = _parse_cells(g, sep, bg, row_intervals, col_intervals)
    first_order = _simulate(directions, start)
    assert first_order, "walk must visit at least the start cell"
    return np.array([[int(body_colors[r, c]) for r, c in first_order]], dtype=int)


def _simulate_for_generation(
    directions: np.ndarray,
    start: tuple[int, int],
    max_steps: int = 1000,
) -> tuple[bool, list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    rows, cols = map(int, directions.shape)
    visits = np.zeros((rows, cols), dtype=int)
    first_order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    full_trace: list[tuple[int, int]] = []
    r, c = map(int, start)

    for _ in range(max_steps):
        if not (0 <= r < rows and 0 <= c < cols):
            return True, first_order, full_trace, visits
        pos = (int(r), int(c))
        full_trace.append(pos)
        if pos not in seen:
            seen.add(pos)
            first_order.append(pos)
        direction = int(directions[r, c] + visits[r, c]) % 4
        visits[r, c] += 1
        dr, dc = DIRS[direction]
        r += dr
        c += dc
    return False, first_order, full_trace, visits


def _static_prefix(directions: np.ndarray, start: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    rows, cols = map(int, directions.shape)
    prefix: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    r, c = map(int, start)
    for _ in range(rows * cols + 2):
        if not (0 <= r < rows and 0 <= c < cols):
            return prefix, False
        pos = (int(r), int(c))
        if pos in seen:
            return prefix, True
        seen.add(pos)
        prefix.append(pos)
        dr, dc = DIRS[int(directions[r, c])]
        r += dr
        c += dc
    return prefix, True


def _has_new_cell_after_revisit(full_trace: list[tuple[int, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    had_revisit = False
    for pos in full_trace:
        if pos in seen:
            had_revisit = True
        else:
            if had_revisit:
                return True
            seen.add(pos)
    return False


def _choose_path(rows: int, cols: int) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(800):
        directions = np.random.default_rng().integers(0, 4, size=(rows, cols), dtype=int)
        start = (random.randrange(rows), random.randrange(cols))
        exited, first_order, full_trace, visits = _simulate_for_generation(directions, start)
        static, static_loops = _static_prefix(directions, start)
        if not exited or not static_loops:
            continue
        if len(first_order) < 4 or len(first_order) <= len(static):
            continue
        if not _has_new_cell_after_revisit(full_trace):
            continue
        if int(visits.max()) < 2:
            continue
        return directions, start, first_order, static
    return None


def _assign_body_colors(n_cells: int, available: list[int]) -> list[int]:
    max_k = min(len(available), max(2, n_cells // 2), 6)
    assert max_k >= 2
    k = random.randint(2, max_k)
    chosen = random.sample(available, k)
    values = chosen[:] + chosen[:]
    while len(values) < n_cells:
        values.append(random.choice(chosen))
    random.shuffle(values)
    return values


def _build_grid(
    rows: int,
    cols: int,
    cell_size: int,
    sep_width: int,
    sep: int,
    bg: int,
    marker: int,
    body_colors: np.ndarray,
    directions: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    H = rows * cell_size + (rows + 1) * sep_width
    W = cols * cell_size + (cols + 1) * sep_width
    grid = np.full((H, W), int(sep), dtype=int)

    row_starts = [sep_width + r * (cell_size + sep_width) for r in range(rows)]
    col_starts = [sep_width + c * (cell_size + sep_width) for c in range(cols)]
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = int(bg)

    for tr, r0 in enumerate(row_starts):
        for tc, c0 in enumerate(col_starts):
            dr, dc = DIRS[int(directions[tr, tc])]
            valid_body_rows = [rr for rr in range(cell_size) if 0 <= rr + dr < cell_size]
            valid_body_cols = [cc for cc in range(cell_size) if 0 <= cc + dc < cell_size]
            br = random.choice(valid_body_rows)
            bc = random.choice(valid_body_cols)
            pr = br + dr
            pc = bc + dc
            grid[r0 + br, c0 + bc] = int(body_colors[tr, tc])
            grid[r0 + pr, c0 + pc] = int(marker if (tr, tc) == start else GREY)

    return grid


def _semantic_accepts(
    grid: np.ndarray,
    body_colors: np.ndarray,
    first_order: list[tuple[int, int]],
    static_prefix: list[tuple[int, int]],
    marker: int,
) -> bool:
    expected = np.array([[int(body_colors[r, c]) for r, c in first_order]], dtype=int)
    naive = np.array([[int(body_colors[r, c]) for r, c in static_prefix]], dtype=int)
    if expected.shape == naive.shape and bool(np.array_equal(expected, naive)):
        return False
    if len(first_order) <= len(static_prefix):
        return False
    if len(set(int(v) for v in expected.ravel())) < 2:
        return False

    counts = Counter(int(v) for v in grid.ravel())
    if counts[int(marker)] != 1:
        return False
    body_counts = Counter(int(v) for v in body_colors.ravel())
    if any(count == 1 for count in body_counts.values()):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice([(2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)])
    path = _choose_path(rows, cols)
    if path is None:
        return None
    directions, start, first_order, static = path

    cell_size = random.randint(3, 5)
    sep_width = random.choice([1, 1, 2])
    if rows * cell_size + (rows + 1) * sep_width > 30:
        return None
    if cols * cell_size + (cols + 1) * sep_width > 30:
        return None

    palette = list(range(10))
    sep = random.choice([c for c in palette if c != GREY])
    bg = random.choice([c for c in palette if c not in {GREY, sep}])
    marker = random.choice([c for c in palette if c not in {GREY, sep, bg}])
    body_pool = [c for c in palette if c not in {GREY, sep, bg, marker}]
    flat_bodies = _assign_body_colors(rows * cols, body_pool)
    body_colors = np.array(flat_bodies, dtype=int).reshape(rows, cols)

    grid = _build_grid(
        rows,
        cols,
        cell_size,
        sep_width,
        sep,
        bg,
        marker,
        body_colors,
        directions,
        start,
    )
    if _semantic_accepts(grid, body_colors, first_order, static, marker):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid rewrite-pointer input")

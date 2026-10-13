from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

CELL_SIZE = 3
SEP_SIZE = 1
ALL_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

Span = tuple[int, int]
Coord = tuple[int, int]


def _groups(indices: list[int]) -> list[Span]:
    assert indices, "expected separator lines"
    groups: list[Span] = []
    start = prev = int(indices[0])
    for idx0 in indices[1:]:
        idx = int(idx0)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_between(line_groups: list[Span], limit: int) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))

    assert spans, "expected macro cells between separator lines"
    assert all(end - start + 1 == CELL_SIZE for start, end in spans), "macro cells must be 3x3"
    return spans


def _separator_color(g: np.ndarray) -> int:
    row_colors = {
        int(g[r, 0])
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0])
    }
    col_colors = {
        int(g[0, c])
        for c in range(g.shape[1])
        if np.all(g[:, c] == g[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one full-row/full-column separator color"
    separator = candidates.pop()
    assert separator == GREY, "separator lines must be grey"
    return separator


def _compress_macro_grid(g: np.ndarray) -> tuple[list[Span], list[Span], dict[Coord, Coord]]:
    separator = _separator_color(g)
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    row_spans = _spans_between(_groups(row_lines), g.shape[0])
    col_spans = _spans_between(_groups(col_lines), g.shape[1])

    pointer_dirs: dict[Coord, Coord] = {}
    for mr, (r0, r1) in enumerate(row_spans):
        for mc, (c0, c1) in enumerate(col_spans):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert block.shape == (CELL_SIZE, CELL_SIZE)
            assert np.all((block == BLACK) | (block == BLUE)), "macro blocks must be black with optional blue pointer"
            blue_cells = np.argwhere(block == BLUE)
            assert len(blue_cells) in (0, 1), "each macro block has at most one pointer pixel"
            if len(blue_cells) == 0:
                continue
            br, bc = map(int, blue_cells[0])
            dr, dc = br - 1, bc - 1
            assert (dr, dc) in ALL_DIRECTIONS, "blue pixel must be off-center"
            pointer_dirs[(int(mr), int(mc))] = (int(dr), int(dc))

    assert pointer_dirs, "expected at least one pointer macro cell"
    return row_spans, col_spans, pointer_dirs


def _trace_chain(
    pointer_dirs: dict[Coord, Coord],
    macro_shape: tuple[int, int],
    start: Coord,
) -> tuple[list[Coord], list[Coord]] | None:
    rows, cols = macro_shape
    current = start
    visited_pointers: list[Coord] = []
    visited_pointer_set: set[Coord] = set()
    traversed: list[Coord] = []
    traversed_set: set[Coord] = set()

    while True:
        if current in visited_pointer_set:
            return None
        visited_pointer_set.add(current)
        visited_pointers.append(current)
        if current not in traversed_set:
            traversed_set.add(current)
            traversed.append(current)

        dr, dc = pointer_dirs[current]
        r, c = current
        while True:
            r += dr
            c += dc
            if not (0 <= r < rows and 0 <= c < cols):
                return visited_pointers, traversed

            cell = (int(r), int(c))
            if cell not in traversed_set:
                traversed_set.add(cell)
                traversed.append(cell)
            if cell in pointer_dirs:
                current = cell
                break


def _full_coverage_solution(
    pointer_dirs: dict[Coord, Coord],
    macro_shape: tuple[int, int],
) -> tuple[Coord, list[Coord], list[Coord]]:
    all_pointers = set(pointer_dirs)
    candidates: list[tuple[Coord, list[Coord], list[Coord]]] = []
    for start in sorted(pointer_dirs):
        traced = _trace_chain(pointer_dirs, macro_shape, start)
        if traced is None:
            continue
        pointer_order, traversed = traced
        if len(pointer_order) == len(all_pointers) and set(pointer_order) == all_pointers:
            candidates.append((start, pointer_order, traversed))
    assert len(candidates) == 1, "expected a unique full-coverage start"
    return candidates[0]


def _paint_macro_cells(
    grid: np.ndarray,
    row_spans: list[Span],
    col_spans: list[Span],
    traversed: list[Coord],
    start: Coord,
) -> np.ndarray:
    out = grid.copy()
    for mr, mc in traversed:
        r0, r1 = row_spans[mr]
        c0, c1 = col_spans[mc]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREEN

    sr, sc = start
    r0, r1 = row_spans[sr]
    c0, c1 = col_spans[sc]
    out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unique macro-grid pointer-chain start and fill the whole traversed chain.

    Input structure:
    - Full grey separator rows and columns divide the image into a rectangular
      macro grid of 3x3 black blocks.
    - Some macro blocks contain exactly one blue pixel. Its offset from the
      block center is one of the eight neighboring offsets and encodes that
      block's pointer direction.
    - No macro block is explicitly marked as the start.

    Rule:
    - For each possible pointer block start, follow its direction through macro
      cells, passing through empty macro cells until the next pointer block is
      reached, then switch to that pointer's direction.
    - Exactly one start reaches every pointer block before leaving the macro
      grid. Fill every macro cell traversed by that chain green, then recolor
      the start macro block red. Keep separator lines and all other cells
      unchanged.

    Key insight:
    - The start is not locally marked; it is the only pointer block whose
      directed ray chain globally covers all pointer blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_spans, col_spans, pointer_dirs = _compress_macro_grid(g)
    macro_shape = (len(row_spans), len(col_spans))
    start, _pointer_order, traversed = _full_coverage_solution(pointer_dirs, macro_shape)
    return _paint_macro_cells(g, row_spans, col_spans, traversed, start)


# --------------- generator ---------------

GEN_MACRO_ROWS = (4, 7)
GEN_MACRO_COLS = (4, 7)
GEN_POINTERS = (4, 7)
GEN_ATTEMPTS = 600
GEN_CHAIN_ATTEMPTS = 250


def _macro_grid_size(macro_rows: int, macro_cols: int) -> tuple[int, int]:
    return (
        macro_rows * CELL_SIZE + (macro_rows - 1) * SEP_SIZE,
        macro_cols * CELL_SIZE + (macro_cols - 1) * SEP_SIZE,
    )


def _ray_cells(start: Coord, direction: Coord, macro_shape: tuple[int, int]) -> list[Coord]:
    rows, cols = macro_shape
    dr, dc = direction
    r, c = start
    cells: list[Coord] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < rows and 0 <= c < cols):
            return cells
        cells.append((int(r), int(c)))


def _segment_candidates(
    current: Coord,
    macro_shape: tuple[int, int],
    used: set[Coord],
    forbidden_pointer_cells: set[Coord],
    used_dirs: set[Coord],
) -> list[tuple[Coord, int, Coord, list[Coord], int]]:
    candidates: list[tuple[Coord, int, Coord, list[Coord], int]] = []
    for direction in ALL_DIRECTIONS:
        ray = _ray_cells(current, direction, macro_shape)
        for idx, nxt in enumerate(ray):
            if nxt in used or nxt in forbidden_pointer_cells:
                continue
            before = ray[:idx]
            if any(cell in used for cell in before):
                continue
            length = idx + 1
            new_dir_bonus = 2 if direction not in used_dirs else 0
            empty_bonus = 3 if length > 1 else 0
            diagonal_bonus = 1 if direction[0] != 0 and direction[1] != 0 else 0
            weight = 1 + new_dir_bonus + empty_bonus + diagonal_bonus
            candidates.append((direction, length, nxt, before, weight))
    return candidates


def _sample_chain(
    macro_rows: int,
    macro_cols: int,
    pointer_count: int,
) -> tuple[list[Coord], list[Coord]] | None:
    macro_shape = (macro_rows, macro_cols)
    for _ in range(GEN_CHAIN_ATTEMPTS):
        start = (random.randrange(macro_rows), random.randrange(macro_cols))
        positions: list[Coord] = [start]
        directions: list[Coord] = []
        used: set[Coord] = {start}
        forbidden_pointer_cells: set[Coord] = set()
        used_dirs: set[Coord] = set()
        failed = False

        for _step in range(pointer_count - 1):
            current = positions[-1]
            candidates = _segment_candidates(current, macro_shape, used, forbidden_pointer_cells, used_dirs)
            if not candidates:
                failed = True
                break
            weights = [item[4] for item in candidates]
            direction, _length, nxt, before, _weight = random.choices(candidates, weights=weights, k=1)[0]
            positions.append(nxt)
            directions.append(direction)
            used.add(nxt)
            used_dirs.add(direction)
            forbidden_pointer_cells.update(before)

        if failed:
            continue

        last = positions[-1]
        exit_dirs = [
            direction
            for direction in ALL_DIRECTIONS
            if not any(cell in used for cell in _ray_cells(last, direction, macro_shape))
        ]
        if not exit_dirs:
            continue
        random.shuffle(exit_dirs)
        directions.append(exit_dirs[0])

        has_cardinal = any((dr == 0) ^ (dc == 0) for dr, dc in directions)
        has_diagonal = any(dr != 0 and dc != 0 for dr, dc in directions)
        if not (has_cardinal and has_diagonal):
            continue

        pointer_dirs = {pos: direction for pos, direction in zip(positions, directions)}
        solved = _full_coverage_solution(pointer_dirs, macro_shape)
        solved_start, pointer_order, traversed = solved
        if solved_start != positions[0] or pointer_order != positions:
            continue
        if len(traversed) < pointer_count + 2:
            continue
        if len(traversed) > (macro_rows * macro_cols * 3) // 4:
            continue
        if all(positions[i + 1] in _ray_cells(positions[i], directions[i], macro_shape)[:1] for i in range(pointer_count - 1)):
            continue
        return positions, directions
    return None


def _expand_macro_grid(
    macro_rows: int,
    macro_cols: int,
    pointer_dirs: dict[Coord, Coord],
) -> np.ndarray:
    height, width = _macro_grid_size(macro_rows, macro_cols)
    grid = np.full((height, width), GREY, dtype=int)

    for mr in range(macro_rows):
        r0 = mr * (CELL_SIZE + SEP_SIZE)
        for mc in range(macro_cols):
            c0 = mc * (CELL_SIZE + SEP_SIZE)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = BLACK

    for (mr, mc), (dr, dc) in pointer_dirs.items():
        r0 = mr * (CELL_SIZE + SEP_SIZE)
        c0 = mc * (CELL_SIZE + SEP_SIZE)
        grid[r0 + 1 + dr, c0 + 1 + dc] = BLUE
    return grid


def _expected_from_macro(
    grid: np.ndarray,
    macro_rows: int,
    macro_cols: int,
    pointer_dirs: dict[Coord, Coord],
) -> np.ndarray:
    row_spans = [(r * (CELL_SIZE + SEP_SIZE), r * (CELL_SIZE + SEP_SIZE) + CELL_SIZE - 1) for r in range(macro_rows)]
    col_spans = [(c * (CELL_SIZE + SEP_SIZE), c * (CELL_SIZE + SEP_SIZE) + CELL_SIZE - 1) for c in range(macro_cols)]
    start, _pointer_order, traversed = _full_coverage_solution(pointer_dirs, (macro_rows, macro_cols))
    return _paint_macro_cells(grid, row_spans, col_spans, traversed, start)


def _gen_attempt() -> np.ndarray | None:
    macro_rows = random.randint(*GEN_MACRO_ROWS)
    macro_cols = random.randint(*GEN_MACRO_COLS)
    max_pointers = min(GEN_POINTERS[1], max(GEN_POINTERS[0], (macro_rows * macro_cols) // 3 + 1))
    pointer_count = random.randint(GEN_POINTERS[0], max_pointers)

    sampled = _sample_chain(macro_rows, macro_cols, pointer_count)
    if sampled is None:
        return None
    positions, directions = sampled
    pointer_dirs = {pos: direction for pos, direction in zip(positions, directions)}

    grid = _expand_macro_grid(macro_rows, macro_cols, pointer_dirs)
    if max(grid.shape) > 30:
        return None

    expected = _expected_from_macro(grid, macro_rows, macro_cols, pointer_dirs)
    if np.array_equal(expected, grid):
        return None
    if int(np.count_nonzero(expected == RED)) != CELL_SIZE * CELL_SIZE:
        return None
    if int(np.count_nonzero(expected == GREEN)) < CELL_SIZE * CELL_SIZE * 2:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid macro pointer-chain input")

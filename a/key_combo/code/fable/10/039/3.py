from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
TEAL = 8

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MIN_SCENE_ROW = 3
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_top_runs(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    width = len(row)
    while c < width:
        if int(row[c]) != color:
            c += 1
            continue
        start = c
        while c < width and int(row[c]) == color:
            c += 1
        runs.append((start, c - 1))
    return runs


def _parse_program(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    top = np.asarray(grid[0], dtype=int)
    assert set(map(int, np.unique(top))).issubset({bg, TEAL}), "top row must contain only background and the teal instruction tape"
    runs = _find_top_runs(top, TEAL)
    assert len(runs) >= 2, "instruction tape must contain multiple runs"

    run_lengths = [end - start + 1 for start, end in runs]
    gap_widths: list[int] = []
    for i in range(len(runs) - 1):
        gap = runs[i + 1][0] - runs[i][1] - 1
        assert gap in (1, 2), "instruction gaps must have width 1 or 2"
        assert np.all(top[runs[i][1] + 1 : runs[i + 1][0]] == bg)
        gap_widths.append(gap)
    return run_lengths, gap_widths


def _find_start_and_heading(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    red_cells = np.argwhere(grid == RED)
    assert len(red_cells) == 1, "expected exactly one red start pixel"
    sr, sc = map(int, red_cells[0])

    heading_neighbors: list[tuple[int, int]] = []
    for dr, dc in DIRS_URDL:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == YELLOW:
            heading_neighbors.append((nr, nc))
    assert len(heading_neighbors) == 1, "red start must have exactly one yellow 4-neighbor"
    hr, hc = heading_neighbors[0]
    return (sr, sc), (hr - sr, hc - sc)


def _turn(direction: tuple[int, int], gap_width: int) -> tuple[int, int]:
    idx = DIRS_URDL.index(direction)
    if gap_width == 1:
        return DIRS_URDL[(idx + 1) % 4]
    if gap_width == 2:
        return DIRS_URDL[(idx - 1) % 4]
    raise AssertionError("unsupported gap width")


def _trace_path(
    start: tuple[int, int],
    direction: tuple[int, int],
    run_lengths: list[int],
    gap_widths: list[int],
    shape: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    assert len(gap_widths) == len(run_lengths) - 1
    cells: list[tuple[int, int]] = []
    cur_r, cur_c = start
    cur_dir = direction

    for i, seg_len in enumerate(run_lengths):
        for _ in range(seg_len):
            cur_r += cur_dir[0]
            cur_c += cur_dir[1]
            if shape is not None:
                assert 0 <= cur_r < shape[0] and 0 <= cur_c < shape[1], "path left the grid"
            cells.append((cur_r, cur_c))
        if i < len(gap_widths):
            cur_dir = _turn(cur_dir, gap_widths[i])
    return cells


def _render_output(grid: np.ndarray, path_cells: list[tuple[int, int]], start: tuple[int, int], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in path_cells:
        out[r, c] = TEAL
    out[start] = RED
    out[0, :] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top dashed tape as a one-dimensional program and draw the teal polyline it encodes.

    Input structure:
    - The background is a single dominant color.
    - The top row contains a teal dashed instruction tape: teal runs separated by background gaps of width 1 or 2.
    - The rest of the scene starts at least two rows lower and contains exactly one red start pixel with exactly one yellow 4-neighbor that indicates the initial heading.

    Rule:
    - Read the top tape from left to right. The length of run i is the length of segment i.
    - Start at the red pixel and move in the direction from red to yellow; paint exactly L_1 teal cells, so the yellow heading pixel becomes the first teal path cell.
    - After each segment, turn according to the following gap on the tape: width 1 means turn right 90 degrees, width 2 means turn left 90 degrees.
    - After the last segment, stop. Then erase the entire top instruction row and keep the red start pixel.

    Key insight:
    - The teal dashes on the top row are not a scene object to complete; their run lengths and their gaps together form a turtle-like program, and the tape disappears after execution.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 4, "grid too small for the described layout"

    bg = _most_common_color(g)
    assert np.all(g[1:MIN_SCENE_ROW, :] == bg), "the scene must be separated from the top tape by two blank rows"

    run_lengths, gap_widths = _parse_program(g, bg)
    start, direction = _find_start_and_heading(g)
    path_cells = _trace_path(start, direction, run_lengths, gap_widths, g.shape)
    return _render_output(g, path_cells, start, bg)


def _relative_path(run_lengths: list[int], gap_widths: list[int], direction_idx: int) -> list[tuple[int, int]]:
    return _trace_path((0, 0), DIRS_URDL[direction_idx], run_lengths, gap_widths, None)


def _is_simple_non_touching_path(cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    if len(set(cells)) != len(cells):
        return False

    index = {cell: i for i, cell in enumerate(cells)}
    for i, (r, c) in enumerate(cells):
        for dr, dc in DIRS_URDL:
            neighbor = (r + dr, c + dc)
            if neighbor in index and abs(index[neighbor] - i) != 1:
                return False
    origin_neighbors = {(dr, dc) for dr, dc in DIRS_URDL}
    if cells[0] not in origin_neighbors:
        return False
    for cell in cells[1:]:
        if cell in origin_neighbors:
            return False
    return True


def _path_signature(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(cells)


def _sample_gap_widths(n_runs: int) -> list[int]:
    assert n_runs >= 3
    while True:
        gaps = [random.choice((1, 2)) for _ in range(n_runs - 1)]
        if 1 in gaps and 2 in gaps:
            return gaps


def _sample_program() -> tuple[list[int], list[int], int]:
    for _ in range(200):
        n_runs = random.randint(3, 6)
        run_lengths = [random.randint(1, 5) for _ in range(n_runs)]
        gap_widths = _sample_gap_widths(n_runs)
        direction_idx = random.randrange(4)

        cells = _relative_path(run_lengths, gap_widths, direction_idx)
        if not _is_simple_non_touching_path(cells):
            continue
        swapped = [3 - gap for gap in gap_widths]
        if _path_signature(cells) == _path_signature(_relative_path(run_lengths, swapped, direction_idx)):
            continue
        if _path_signature(cells) == _path_signature(_relative_path(run_lengths, [1] * len(gap_widths), direction_idx)):
            continue
        if _path_signature(cells) == _path_signature(_relative_path(run_lengths, [2] * len(gap_widths), direction_idx)):
            continue
        opposite = _relative_path(run_lengths, gap_widths, (direction_idx + 2) % 4)
        if _path_signature(cells) == _path_signature(opposite):
            continue
        return run_lengths, gap_widths, direction_idx
    raise RuntimeError("failed to sample a non-degenerate path program")


def _program_width(run_lengths: list[int], gap_widths: list[int]) -> int:
    return sum(run_lengths) + sum(gap_widths)


def _place_decoys(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    bg: int,
) -> bool:
    del bg
    height, width = grid.shape
    decoy_count = random.randint(1, 2)

    def far_enough(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
        r, c = cell
        for orow, ocol in others:
            if abs(r - orow) + abs(c - ocol) <= 1:
                return False
        return True

    for _ in range(decoy_count):
        placed = False
        for _ in range(80):
            n_runs = random.randint(2, 4)
            run_lengths = [random.randint(1, 3) for _ in range(n_runs)]
            gap_widths = [random.choice((1, 2)) for _ in range(n_runs - 1)]
            row = random.randint(MIN_SCENE_ROW, height - 1)
            span = _program_width(run_lengths, gap_widths)
            if span > width:
                continue
            start_col = random.randint(0, width - span)

            cells: list[tuple[int, int]] = []
            col = start_col
            for i, seg_len in enumerate(run_lengths):
                for offset in range(seg_len):
                    cells.append((row, col + offset))
                col += seg_len
                if i < len(gap_widths):
                    col += gap_widths[i]

            cell_set = set(cells)
            if cell_set & occupied:
                continue
            if not all(far_enough(cell, occupied | path_cells) for cell in cell_set):
                continue

            for rr, cc in cells:
                grid[rr, cc] = TEAL
            occupied |= cell_set
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(
    grid: np.ndarray,
    run_lengths: list[int],
    gap_widths: list[int],
    start: tuple[int, int],
    direction_idx: int,
) -> np.ndarray:
    path_cells = _trace_path(start, DIRS_URDL[direction_idx], run_lengths, gap_widths, grid.shape)
    return _render_output(grid, path_cells, start, BG)


def _safe_render_variant(
    grid: np.ndarray,
    run_lengths: list[int],
    gap_widths: list[int],
    start: tuple[int, int],
    direction_idx: int,
) -> np.ndarray | None:
    try:
        path_cells = _trace_path(start, DIRS_URDL[direction_idx], run_lengths, gap_widths, grid.shape)
    except AssertionError:
        return None
    return _render_output(grid, path_cells, start, BG)


def _gen_attempt() -> np.ndarray | None:
    run_lengths, gap_widths, direction_idx = _sample_program()
    rel_cells = _relative_path(run_lengths, gap_widths, direction_idx)

    rows = [0] + [r for r, _ in rel_cells]
    cols = [0] + [c for _, c in rel_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    min_height = max(10, max_r - min_r + 1 + MIN_SCENE_ROW)
    min_width = max(10, max(max_c - min_c + 1, _program_width(run_lengths, gap_widths)))
    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, min(30, min_height + 6))
    width = random.randint(min_width, min(30, min_width + 8))

    row_lo = max(MIN_SCENE_ROW, 1 - min_r)
    row_hi = height - 1 - max_r
    col_lo = -min_c
    col_hi = width - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    start = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    path_cells = _trace_path(start, DIRS_URDL[direction_idx], run_lengths, gap_widths, (height, width))
    path_set = set(path_cells)
    if any(r == 0 for r, _ in path_cells):
        return None

    grid = np.full((height, width), BG, dtype=int)

    col = 0
    for i, seg_len in enumerate(run_lengths):
        grid[0, col : col + seg_len] = TEAL
        col += seg_len
        if i < len(gap_widths):
            col += gap_widths[i]

    sr, sc = start
    dr, dc = DIRS_URDL[direction_idx]
    heading = (sr + dr, sc + dc)
    grid[start] = RED
    grid[heading] = YELLOW

    occupied = {start, heading}
    if not _place_decoys(grid, occupied, path_set, BG):
        return None

    expected = _build_expected_output(grid, run_lengths, gap_widths, start, direction_idx)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    alt_swapped = _safe_render_variant(grid, run_lengths, [3 - gap for gap in gap_widths], start, direction_idx)
    if alt_swapped is not None and np.array_equal(expected, alt_swapped):
        return None

    if np.count_nonzero(expected == TEAL) <= len(path_cells):
        return None

    teal_neighbors_of_start = 0
    for ddr, ddc in DIRS_URDL:
        nr, nc = sr + ddr, sc + ddc
        if 0 <= nr < height and 0 <= nc < width and int(expected[nr, nc]) == TEAL:
            teal_neighbors_of_start += 1
    if teal_neighbors_of_start != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

import random

import numpy as np


BG = 0
RUN = 1
CAP = 4
BARRIER = 7

GEN_ATTEMPTS = 500
MIN_ROWS = 15
MAX_ROWS = 24
MIN_COLS = 18
MAX_COLS = 26
RUN_COUNT_RANGE = (3, 5)
RUN_WIDTHS = tuple(range(3, 10))
MIN_ROW_GAP = 4


def _cap_height(width: int) -> int:
    return (int(width) + 1) // 2


def _run_cells(row: int, start: int, width: int) -> set[tuple[int, int]]:
    return {(int(row), c) for c in range(int(start), int(start) + int(width))}


def _cap_cells(shape: tuple[int, int], row: int, start: int, width: int) -> set[tuple[int, int]]:
    height, _ = shape
    cells: set[tuple[int, int]] = set()
    for trim in range(int(width)):
        cap_width = int(width) - 2 * int(trim)
        if cap_width <= 0:
            break
        left = int(start) + int(trim)
        right = int(start) + int(width) - 1 - int(trim)
        up = int(row) - int(trim) - 1
        down = int(row) + int(trim) + 1
        if 0 <= up < height:
            for col in range(left, right + 1):
                cells.add((up, col))
        if 0 <= down < height:
            for col in range(left, right + 1):
                cells.add((down, col))
    return cells


def _paint_caps(grid: np.ndarray, row: int, start: int, width: int) -> None:
    height, _ = grid.shape
    for trim in range(int(width)):
        cap_width = int(width) - 2 * int(trim)
        if cap_width <= 0:
            break
        left = int(start) + int(trim)
        right = int(start) + int(width) - 1 - int(trim)
        up = int(row) - int(trim) - 1
        down = int(row) + int(trim) + 1

        if 0 <= up < height:
            for col in range(left, right + 1):
                if grid[up, col] == BG:
                    grid[up, col] = CAP
        if 0 <= down < height:
            for col in range(left, right + 1):
                if grid[down, col] == BG:
                    grid[down, col] = CAP


def _extract_right_edge_runs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    runs: list[tuple[int, int, int]] = []
    for row in range(height):
        cols = np.flatnonzero(grid[row] == RUN)
        if len(cols) == 0:
            continue
        start = int(cols[0])
        end = int(cols[-1])
        run_width = int(len(cols))
        assert end == width - 1, "moving run must touch the right edge"
        assert run_width == end - start + 1, "moving pixels must form one contiguous run"
        runs.append((row, start, run_width))
    assert runs, "expected at least one moving run"
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge blue run left to its row blocker, then grow yellow shrinking caps above and below it.

    Input structure:
    - The grid is a black background with scattered orange single-pixel barriers.
    - Several rows contain one contiguous blue horizontal run, and each such run ends at the right edge.
    - Different runs lie on different rows and are processed independently within their own rows.

    Rule:
    - For each blue run, erase its original right-edge position and redraw the same-width run starting immediately to the right of the row's rightmost orange barrier, or at column 0 if none exists.
    - From the slid run, paint yellow cap rows one step above and below: first the full run width, then shrink by two cells per farther row while staying horizontally centered.
    - Yellow cap pixels are only added onto black background cells; orange barriers and blue run pixels stay unchanged.

    Key insight:
    - The row barrier fixes the run's final horizontal position, and that stopped run width directly determines both the width and height of the symmetric shrinking caps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    shifted_runs: list[tuple[int, int, int]] = []
    width = g.shape[1]

    for row, start, run_width in _extract_right_edge_runs(g):
        barrier_cols = np.flatnonzero(g[row, :start] == BARRIER)
        target_start = int(barrier_cols[-1]) + 1 if len(barrier_cols) else 0
        target_end = target_start + run_width
        assert target_end <= width, "shifted run must stay within bounds"
        assert np.all(g[row, target_start:target_end] != BARRIER), "shifted run cannot cover barriers"

        out[row, start : start + run_width] = BG
        out[row, target_start:target_end] = RUN
        shifted_runs.append((row, target_start, run_width))

    for row, start, run_width in shifted_runs:
        _paint_caps(out, row, start, run_width)

    return out


def _rows_have_min_gap(rows: list[int]) -> bool:
    return all(b - a >= MIN_ROW_GAP for a, b in zip(rows, rows[1:]))


def _caps_conflict(specs: list[tuple[int, int, int]], shape: tuple[int, int]) -> bool:
    used_caps: set[tuple[int, int]] = set()
    used_runs: set[tuple[int, int]] = set()
    for row, start, width in specs:
        run_cells = _run_cells(row, start, width)
        cap_cells = _cap_cells(shape, row, start, width)
        if used_caps & cap_cells:
            return True
        if used_runs & cap_cells:
            return True
        if used_caps & run_cells:
            return True
        used_runs |= run_cells
        used_caps |= cap_cells
    return False


def _has_boundary_clipping(specs: list[tuple[int, int, int]], height: int) -> bool:
    for row, _start, width in specs:
        cap_h = _cap_height(width)
        if row < cap_h or height - 1 - row < cap_h:
            return True
    return False


def _choose_run_specs(height: int, width: int, want_boundary_clip: bool) -> list[tuple[int, int, int]] | None:
    max_run_width = min(width - 1, max(RUN_WIDTHS))
    width_pool = [run_width for run_width in RUN_WIDTHS if run_width <= max_run_width]
    if len(width_pool) < RUN_COUNT_RANGE[0]:
        return None

    for _ in range(200):
        run_count = random.randint(*RUN_COUNT_RANGE)
        widths = random.sample(width_pool, run_count)
        random.shuffle(widths)
        rows = sorted(random.sample(range(height), run_count))
        if not _rows_have_min_gap(rows):
            continue

        specs: list[tuple[int, int, int]] = []
        any_slide = False
        valid = True
        for row, run_width in zip(rows, widths):
            max_slide = width - run_width - 1
            if max_slide < 0:
                valid = False
                break

            min_slide = 1 if not any_slide and len(specs) == run_count - 1 and max_slide > 0 else 0
            if min_slide > max_slide:
                valid = False
                break
            slide = random.randint(min_slide, max_slide)
            target_start = width - run_width - slide
            specs.append((row, target_start, run_width))
            any_slide = any_slide or slide > 0

        if not valid or not any_slide:
            continue
        if _caps_conflict(specs, (height, width)):
            continue
        if want_boundary_clip and not _has_boundary_clipping(specs, height):
            continue
        return specs
    return None


def _place_barriers(
    output: np.ndarray,
    specs: list[tuple[int, int, int]],
    want_cap_hole: bool,
) -> bool:
    height, width = output.shape
    row_counts = [0] * height
    cap_union: set[tuple[int, int]] = set()

    for row, start, run_width in specs:
        output[row, start : start + run_width] = RUN
        barrier_col = start - 1
        if barrier_col < 0:
            return False
        output[row, barrier_col] = BARRIER
        row_counts[row] += 1
        cap_union |= _cap_cells(output.shape, row, start, run_width)

        extra_choices = [col for col in range(start - 1) if output[row, col] == BG]
        if extra_choices and row_counts[row] < 2 and random.random() < 0.45:
            extra_col = random.choice(extra_choices)
            output[row, extra_col] = BARRIER
            row_counts[row] += 1

    hole_rows: set[int] = set()
    hole_candidates = [
        (r, c)
        for r, c in cap_union
        if output[r, c] == BG and row_counts[r] < 2
    ]
    random.shuffle(hole_candidates)
    if want_cap_hole:
        for r, c in hole_candidates:
            if r in hole_rows:
                continue
            output[r, c] = BARRIER
            row_counts[r] += 1
            hole_rows.add(r)
            break
        if not hole_rows:
            return False

    for row in range(height):
        if row_counts[row] >= 2 or any(spec_row == row for spec_row, _start, _width in specs):
            continue
        extra_count = random.choices([0, 1, 2], weights=[5, 3, 1], k=1)[0]
        extra_count = min(extra_count, 2 - row_counts[row])
        if extra_count == 0:
            continue

        cols = [col for col in range(width) if output[row, col] == BG]
        if len(cols) < extra_count:
            continue
        for col in random.sample(cols, extra_count):
            output[row, col] = BARRIER
            row_counts[row] += 1

    return True


def _build_expected_output(base_output: np.ndarray, specs: list[tuple[int, int, int]]) -> np.ndarray:
    expected = base_output.copy()
    for row, start, run_width in specs:
        _paint_caps(expected, row, start, run_width)
    return expected


def _derive_input_from_output(base_output: np.ndarray, specs: list[tuple[int, int, int]]) -> np.ndarray:
    inp = base_output.copy()
    width = inp.shape[1]
    for row, start, run_width in specs:
        inp[row, start : start + run_width] = BG
        inp[row, width - run_width : width] = RUN
    return inp


def _valid_input_structure(inp: np.ndarray, specs: list[tuple[int, int, int]]) -> bool:
    if np.any(inp == CAP):
        return False

    runs = _extract_right_edge_runs(inp)
    if len(runs) != len(specs):
        return False

    rows = [row for row, _start, _width in runs]
    if not _rows_have_min_gap(rows):
        return False

    widths = [run_width for _row, _start, run_width in runs]
    if len(set(widths)) != len(widths):
        return False

    for row, start, _run_width in runs:
        if not np.any(inp[row, :start] == BARRIER):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_ROWS, MAX_ROWS)
    width = random.randint(MIN_COLS, MAX_COLS)

    want_boundary_clip = random.random() < 0.55
    want_cap_hole = random.random() < 0.5
    specs = _choose_run_specs(height, width, want_boundary_clip)
    if specs is None:
        return None

    output = np.full((height, width), BG, dtype=int)
    if not _place_barriers(output, specs, want_cap_hole):
        return None

    expected = _build_expected_output(output, specs)
    inp = _derive_input_from_output(output, specs)
    if not _valid_input_structure(inp, specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if np.count_nonzero(expected == CAP) < 8:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

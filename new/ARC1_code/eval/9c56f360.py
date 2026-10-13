from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import math
import random


def _right_edge_run_color(g: np.ndarray, bg: int) -> int:
    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 2, "expected one moving color and one barrier color"

    width = g.shape[1]
    candidates: list[int] = []
    for color in non_bg:
        has_run = False
        valid = True
        for row in g:
            cols = np.flatnonzero(row == color)
            if len(cols) == 0:
                continue

            has_run = True
            start = int(cols[0])
            end = int(cols[-1])
            if end != width - 1 or len(cols) != end - start + 1:
                valid = False
                break

        if has_run and valid:
            candidates.append(int(color))

    assert len(candidates) == 1, "expected a unique color forming right-edge horizontal runs"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge colored run left within its row until it is packed against the nearest fixed barrier or the left edge.

    Input structure
    - The background is the dominant color.
    - There are exactly two foreground colors: fixed barrier pixels and moving pixels.
    - In every row where the moving color appears, it forms one contiguous horizontal run ending at the right edge.

    Rule
    - Process each occupied row independently.
    - Clear the moving run from the right edge.
    - Repaint that run starting immediately to the right of the row's rightmost barrier pixel, or at column 0 when no barrier is present.
    - Keep the barrier pixels and all rows without moving pixels unchanged.

    Key insight
    - The fixed pixels are row-wise stoppers; each moving run simply slides left as far as possible without crossing the nearest stopper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    moving = _right_edge_run_color(g, bg)
    blockers = colors_present(g, exclude={bg, moving})
    assert len(blockers) == 1, "expected a unique barrier color"
    barrier = int(blockers[0])

    height, width = g.shape
    out = g.copy()
    moved_rows = 0

    for r in range(height):
        cols = np.flatnonzero(g[r] == moving)
        if len(cols) == 0:
            continue

        moved_rows += 1
        start = int(cols[0])
        end = int(cols[-1])
        run_len = int(len(cols))
        assert end == width - 1 and run_len == end - start + 1, "moving pixels must form a right-edge run"

        barrier_cols = np.flatnonzero(g[r, :start] == barrier)
        target_start = int(barrier_cols[-1]) + 1 if len(barrier_cols) else 0
        target_end = target_start + run_len
        assert target_end <= width, "slid run must fit in its row"
        assert np.all(g[r, target_start:target_end] != barrier), "slid run cannot cover a barrier"

        out[r, start : end + 1] = bg
        out[r, target_start:target_end] = moving

    assert moved_rows > 0, "expected at least one moving run"
    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
BG_COLOR, BARRIER_COLOR, MOVING_COLOR = 0, 8, 3
GRID_HEIGHT_BOUNDS = GRID_WIDTH_BOUNDS = (6, 9)

# Moving-run group structure.
GROUP_COUNT_OPTIONS = (1, 2)
GROUP_HEIGHT_RANGE, RUN_LEN_RANGE = range(2, 5), range(1, 3)
MOVING_ROWS_MIN, MOVING_ROWS_MAX, NONMOVING_ROWS_MIN, NONMOVING_ROWS_MAX = 2, 6, 3, 5
MOVING_CELLS_MIN, MOVING_CELLS_MAX, MIN_GROUP_GAP = 4, 9, 1

# Slide and barrier placement limits.
MAX_SLIDE = 6
BARRIER_DENSITY_MIN, BARRIER_DENSITY_MAX = 0.29, 0.41
MOVING_ROW_BARRIER_MAX, NONMOVING_ROW_BARRIER_MAX = 5, 7
# Generation retry budget.
GENERATION_ATTEMPTS = 300


def _gen_valid_group_specs(height: int) -> list[list[tuple[int, int]]]:
    specs: list[list[tuple[int, int]]] = []
    for group_count in GROUP_COUNT_OPTIONS:
        if group_count == 1:
            candidates = [((h, run_len),) for h in GROUP_HEIGHT_RANGE for run_len in RUN_LEN_RANGE]
        else:
            candidates = [
                ((h0, run0), (h1, run1))
                for h0 in GROUP_HEIGHT_RANGE
                for h1 in GROUP_HEIGHT_RANGE
                for run0 in RUN_LEN_RANGE
                for run1 in RUN_LEN_RANGE
            ]
        for spec_tuple in candidates:
            moving_rows = sum(h for h, _ in spec_tuple)
            moving_cells = sum(h * run_len for h, run_len in spec_tuple)
            nonmoving_rows = height - moving_rows
            if (
                MOVING_ROWS_MIN <= moving_rows <= MOVING_ROWS_MAX
                and NONMOVING_ROWS_MIN <= nonmoving_rows <= NONMOVING_ROWS_MAX
                and MOVING_CELLS_MIN <= moving_cells <= MOVING_CELLS_MAX
            ):
                specs.append(list(spec_tuple))
    return specs


def _choose_row_run_lengths(height, group_specs):
    """Pick a valid group placement and expand it to per-row run lengths."""
    if len(group_specs) == 1:
        group_height = group_specs[0][0]
        placements = [[(start, start + group_height)] for start in range(height - group_height + 1)]
    else:
        first_height, second_height = group_specs[0][0], group_specs[1][0]
        placements = []
        for first_start in range(height - first_height + 1):
            first_end = first_start + first_height
            for second_start in range(first_end + MIN_GROUP_GAP, height - second_height + 1):
                placements.append([(first_start, first_end), (second_start, second_start + second_height)])
    if not placements:
        return None

    placement = random.choice(placements)
    row_run_len: dict[int, int] = {}
    for (start, end), (_, run_len) in zip(placement, group_specs):
        row_run_len.update({row: run_len for row in range(start, end)})
    return row_run_len


def _choose_target_starts(width, row_run_len):
    """Choose slide targets, forcing at least one row to move left."""
    moving_rows = sorted(row_run_len)
    forced_slide_row = random.choice(moving_rows)
    row_target_start: dict[int, int] = {}

    for row in moving_rows:
        run_len = row_run_len[row]
        max_slide = min(MAX_SLIDE, width - run_len)
        if row == forced_slide_row:
            slide = random.randint(1, min(max_slide, width - run_len - 1))
        else:
            slide = random.randint(0, max_slide)
        row_target_start[row] = width - run_len - slide
    return moving_rows, row_target_start


def _gen_distribute_counts(total: int, mins: list[int], maxs: list[int]) -> list[int] | None:
    if total < sum(mins) or total > sum(maxs):
        return None
    counts = mins[:]
    remaining = total - sum(counts)
    while remaining:
        open_rows = [i for i, count in enumerate(counts) if count < maxs[i]]
        if not open_rows:
            return None
        row_i = random.choice(open_rows)
        add = random.randint(1, min(remaining, maxs[row_i] - counts[row_i]))
        counts[row_i] += add
        remaining -= add
    return counts


def _choose_barrier_counts(height, width, row_run_len, row_target_start):
    """Choose per-row barrier counts satisfying density and dominance bounds."""
    moving_cells = sum(row_run_len.values())
    total_cells = height * width
    density_low = math.ceil(total_cells * BARRIER_DENSITY_MIN)
    density_high = math.floor(total_cells * BARRIER_DENSITY_MAX)
    bg_dominant_high = (total_cells - moving_cells - 1) // 2

    row_mins: list[int] = []
    row_maxs: list[int] = []
    for row in range(height):
        target_start = row_target_start.get(row)
        row_mins.append(1 if target_start is None or target_start > 0 else 0)
        row_maxs.append(
            min(NONMOVING_ROW_BARRIER_MAX, width)
            if target_start is None
            else min(MOVING_ROW_BARRIER_MAX, target_start)
        )

    min_barriers = sum(row_mins)
    max_barriers = min(sum(row_maxs), density_high, bg_dominant_high)
    min_barriers = max(min_barriers, density_low)
    if min_barriers > max_barriers:
        return None

    barrier_count = random.randint(min_barriers, max_barriers)
    return _gen_distribute_counts(barrier_count, row_mins, row_maxs)


def _render_pair(height, width, moving_rows, row_run_len, row_target_start, row_barrier_counts):
    """Render output, then move runs back to the right edge for input."""
    output = np.full((height, width), BG_COLOR, dtype=int)
    for row in moving_rows:
        run_len = row_run_len[row]
        target_start = row_target_start[row]
        output[row, target_start : target_start + run_len] = MOVING_COLOR

    for row, count in enumerate(row_barrier_counts):
        if row in row_run_len:
            target_start = row_target_start[row]
            if count:
                required = target_start - 1
                cols = [required] + random.sample(list(range(required)), count - 1)
                output[row, cols] = BARRIER_COLOR
        else:
            cols = random.sample(range(width), count)
            output[row, cols] = BARRIER_COLOR
    inp = output.copy()
    for row in moving_rows:
        run_len = row_run_len[row]
        target_start = row_target_start[row]
        inp[row, target_start : target_start + run_len] = BG_COLOR
        inp[row, width - run_len : width] = MOVING_COLOR
    return inp, output


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)

    group_specs = random.choice(_gen_valid_group_specs(height))
    random.shuffle(group_specs)
    row_run_len = _choose_row_run_lengths(height, group_specs)
    if row_run_len is None:
        return None

    moving_rows, row_target_start = _choose_target_starts(width, row_run_len)
    row_barrier_counts = _choose_barrier_counts(height, width, row_run_len, row_target_start)
    if row_barrier_counts is None:
        return None

    inp, output = _render_pair(height, width, moving_rows, row_run_len, row_target_start, row_barrier_counts)
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9c56f360 input")

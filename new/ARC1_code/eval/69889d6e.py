from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _hits_obstacle(obstacles: np.ndarray, row: int, left: int) -> bool:
    _, width = obstacles.shape
    cols = range(left, min(left + 2, width))
    return any(bool(obstacles[row, c]) for c in cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the bottom source as a two-cell diagonal band upward-right, sliding it right around foreground blockers.

    Input structure
    - The grid has a dominant background color.
    - Exactly one foreground pixel lies on the bottom row; this is the colored source.
    - Any other foreground pixels are blockers that remain fixed.

    Rule
    - Copy the input, then emit the source color upward one row at a time.
    - The first row above the source starts a two-cell horizontal band at the source column; each next row shifts that band one column right.
    - If the band would touch a blocker in its row, shift it further right until both band cells avoid blockers.
    - When such an extra shift leaves a gap from the previous row's band, fill the gap on the previous row with the source color.
    - Stop when the band has moved completely past the right edge.

    Key insight
    - Blockers do not stop the diagonal ray; they push the two-cell-wide band right, and the previous row is widened only enough to keep the trail connected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = get_bg(g)
    bottom_cols = np.flatnonzero(g[-1] != bg)
    assert bottom_cols.size == 1, "expected exactly one bottom-row source pixel"
    source_row = height - 1
    source_col = int(bottom_cols[0])
    source_color = int(g[source_row, source_col])
    assert np.count_nonzero(g == source_color) == 1, "source color should appear once in the input"

    obstacles = g != bg
    obstacles[source_row, source_col] = False

    out = g.copy()
    prev_row = source_row
    prev_left = source_col
    prev_right = source_col
    first_band_row = True

    for row in range(source_row - 1, -1, -1):
        left = prev_left if first_band_row else prev_left + 1
        first_band_row = False
        if left >= width:
            break

        while left < width and _hits_obstacle(obstacles, row, left):
            left += 1
        if left >= width:
            break

        if left > prev_right:
            for col in range(prev_right + 1, min(left, width - 1) + 1):
                assert g[prev_row, col] == bg, "connector gap should be background"
                out[prev_row, col] = source_color

        for col in range(left, min(left + 2, width)):
            assert g[row, col] == bg, "shifted band should land on background cells"
            out[row, col] = source_color

        prev_row = row
        prev_left = left
        prev_right = left + 1

    return out

# --------------- generator (o5x) ---------------
# Canvas, colors, and retry budget.
GRID_SIZE = 10
BACKGROUND_COLOR, BLOCKER_COLOR, SOURCE_COLOR = 0, 1, 2
GEN_RETRY_LIMIT = 300

# Path geometry.
BAND_WIDTH = 2
SOURCE_COL_MAX_AFTER_SHIFT = 4
PATH_ROW_MIN, PATH_ROW_MAX = 6, 9
EVENT_POS_START, EVENT_POS_END_PAD = 1, 1

# Blocker event sampling.
BLOCKER_CELL_MIN, BLOCKER_CELL_MAX = 0, 3
ONE_BLOCKER_SHIFT_CHOICES = (1, 2)
TWO_BLOCKER_DOMINO_PROB = 0.5
TWO_BLOCKER_SPLIT_SHIFTS = ((1, 1), (1, 2), (2, 1))
THREE_BLOCKER_EVENTS = ((2, 2), (1, 1))

def _gen_plan_for_blocker_cells(blocker_cells: int) -> list[tuple[int, int]]:
    if blocker_cells == 0:
        return []
    if blocker_cells == 1:
        return [(1, random.choice(ONE_BLOCKER_SHIFT_CHOICES))]
    if blocker_cells == 2:
        if random.random() < TWO_BLOCKER_DOMINO_PROB:
            return [(2, 2)]
        shifts = random.choice(TWO_BLOCKER_SPLIT_SHIFTS)
        return [(1, shifts[0]), (1, shifts[1])]
    events = list(THREE_BLOCKER_EVENTS)
    random.shuffle(events)
    return events


def _sample_layout() -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    """Sample blocker count, source column, and per-path-position blocker events."""
    blocker_cells = random.randint(BLOCKER_CELL_MIN, BLOCKER_CELL_MAX)
    events = _gen_plan_for_blocker_cells(blocker_cells)
    total_shift = sum(shift for _, shift in events)
    max_source_col = SOURCE_COL_MAX_AFTER_SHIFT - total_shift
    if max_source_col < 0:
        return None

    source_col = random.randint(0, max_source_col)
    final_offset = source_col + total_shift
    path_rows = min(GRID_SIZE - 1, GRID_SIZE - final_offset)
    if not PATH_ROW_MIN <= path_rows <= PATH_ROW_MAX:
        return None

    event_count = len(events)
    eligible_positions = list(range(EVENT_POS_START, path_rows - EVENT_POS_END_PAD))
    if event_count > len(eligible_positions):
        return None
    event_positions = sorted(random.sample(eligible_positions, event_count))
    return blocker_cells, source_col, dict(zip(event_positions, events))


def _render_path(source_col: int, events_by_pos: dict[int, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Paint planned blockers and output trail, returning blocker mask, output, and encountered cells."""
    blockers = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    out = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    out[GRID_SIZE - 1, source_col] = SOURCE_COLOR

    prev_row, prev_left, prev_right = GRID_SIZE - 1, source_col, source_col
    first_band_row, encountered_cells = True, 0

    for pos, row in enumerate(range(GRID_SIZE - 2, -1, -1)):
        left = prev_left if first_band_row else prev_left + 1
        first_band_row = False
        if left >= GRID_SIZE:
            break

        event = events_by_pos.get(pos)
        if event is not None:
            length, shift = event
            if shift == 1:
                cols = [left]
            elif length == 1:
                cols = [left + 1]
            else:
                cols = [left, left + 1]
            if any(c < 0 or c >= GRID_SIZE for c in cols):
                return None
            blockers[row, cols] = True
            out[row, cols] = BLOCKER_COLOR

        while left < GRID_SIZE and np.any(blockers[row, left : min(left + BAND_WIDTH, GRID_SIZE)]):
            left += 1
        if left >= GRID_SIZE:
            break

        if left > prev_right:
            connector = slice(prev_right + 1, min(left, GRID_SIZE - 1) + 1)
            if np.any(blockers[prev_row, connector]):
                return None
            out[prev_row, connector] = SOURCE_COLOR

        band = slice(left, min(left + BAND_WIDTH, GRID_SIZE))
        if np.any(blockers[row, band]):
            return None
        out[row, band] = SOURCE_COLOR

        if event is not None:
            encountered_cells += len(cols)

        prev_row = row
        prev_left = left
        prev_right = left + BAND_WIDTH - 1

    return blockers, out, encountered_cells


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    blocker_cells, source_col, events_by_pos = layout

    rendered = _render_path(source_col, events_by_pos)
    if rendered is None:
        return None
    blockers, out, encountered_cells = rendered

    inp = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    inp[GRID_SIZE - 1, source_col] = SOURCE_COLOR
    inp[blockers] = BLOCKER_COLOR

    bad_counts = encountered_cells != blocker_cells or np.count_nonzero(inp == SOURCE_COLOR) != 1
    bad_oracle = not np.any(out[:, -1] == SOURCE_COLOR) or not np.array_equal(main(inp), out)
    if bad_counts or bad_oracle:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

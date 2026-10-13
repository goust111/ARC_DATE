from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _straight_guide(g: np.ndarray, bg: int) -> tuple[int, set[int], set[int]]:
    candidates: list[tuple[int, set[int], set[int]]] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        rows = set(map(int, coords[:, 0]))
        cols = set(map(int, coords[:, 1]))
        if coords.shape[0] > 1 and ((len(rows) == 1) ^ (len(cols) == 1)):
            candidates.append((int(color), rows, cols))

    assert len(candidates) == 1, "expected exactly one straight guide bar"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the scattered non-guide pixels orthogonally onto the cells immediately beside the straight guide bar.

    Input structure
    - The background is the dominant color.
    - One non-background color forms a one-cell-thick horizontal or vertical guide bar, possibly with gaps along the bar axis.
    - The only other non-background color appears as isolated source pixels scattered on either side of the guide.

    Rule
    - Start from an otherwise empty background canvas and keep the guide bar fixed.
    - If the guide is horizontal, each source pixel whose column contains a guide pixel moves to the row directly above or below that guide pixel, preserving its side of the guide.
    - If the guide is vertical, each source pixel whose row contains a guide pixel moves to the column directly left or right of that guide pixel, preserving its side of the guide.
    - Source pixels whose perpendicular projection would hit a gap in the guide are erased.

    Key insight
    - The guide bar defines the valid landing slots; the scattered pixels are not copied in place, but compressed along perpendicular sightlines to the nearest cells beside that guide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    guide_color, guide_rows, guide_cols = _straight_guide(g, bg)

    moving_colors = set(colors_present(g, exclude={bg, guide_color}))
    assert len(moving_colors) == 1, "expected exactly one moving color"
    moving_color = next(iter(moving_colors))

    out = np.full(g.shape, bg, dtype=int)
    out[g == guide_color] = guide_color

    if len(guide_rows) == 1:
        guide_row = next(iter(guide_rows))
        assert 0 < guide_row < g.shape[0] - 1, "horizontal guide must have space on both sides"
        for r_raw, c_raw in np.argwhere(g == moving_color):
            r = int(r_raw)
            c = int(c_raw)
            assert r != guide_row, "moving pixels cannot sit on the guide row"
            if c in guide_cols:
                out[guide_row - 1 if r < guide_row else guide_row + 1, c] = moving_color
    else:
        guide_col = next(iter(guide_cols))
        assert 0 < guide_col < g.shape[1] - 1, "vertical guide must have space on both sides"
        for r_raw, c_raw in np.argwhere(g == moving_color):
            r = int(r_raw)
            c = int(c_raw)
            assert c != guide_col, "moving pixels cannot sit on the guide column"
            if r in guide_rows:
                out[r, guide_col - 1 if c < guide_col else guide_col + 1] = moving_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and color choices.
GRID_HEIGHT_RANGE = (10, 17)
GRID_WIDTH_RANGE = (10, 15)
BACKGROUND_COLOR, GUIDE_COLOR = 0, 8
SOURCE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Guide axis construction.
GUIDE_COVERAGE_NUMERATOR, GUIDE_COVERAGE_DENOMINATOR = 7, 10
GUIDE_AXIS_RETRIES = 80
GUIDE_GAP_COUNT_RANGE = (0, 2)
GUIDE_GAP_SIZE_RANGE = (1, 3)
GUIDE_MIN_SPAN_FOR_GAPS = 4

# Guide placement and source counts.
MIN_CROSS_SIZE, LARGE_CROSS_SIZE_MIN = 5, 11
LARGE_CROSS_MARGIN, SMALL_CROSS_MARGIN = 5, 2
OUTPUT_SLOT_RANGE = (6, 12)
MIN_SLOTS_PER_SIDE = 2
SOURCE_COUNT_RANGE = (11, 15)
ERASED_SOURCE_RANGE = (3, 7)
SIDE_A, SIDE_B = -1, 1

# Retry budget.
GENERATION_RETRIES = 300


def _gen_pick_guide_axis(axis_len: int) -> tuple[list[int], list[int]]:
    min_guides = (GUIDE_COVERAGE_NUMERATOR * axis_len + GUIDE_COVERAGE_DENOMINATOR - 1) // GUIDE_COVERAGE_DENOMINATOR
    for _ in range(GUIDE_AXIS_RETRIES):
        span_len = random.randint(min_guides, axis_len)
        max_gap_cells = span_len - min_guides
        n_gaps = random.randint(*GUIDE_GAP_COUNT_RANGE)
        gap_cells: set[int] = set()

        if n_gaps and max_gap_cells > 0 and span_len >= GUIDE_MIN_SPAN_FOR_GAPS:
            for _ in range(n_gaps):
                remaining = max_gap_cells - len(gap_cells)
                if remaining <= 0:
                    break
                max_size = min(GUIDE_GAP_SIZE_RANGE[1], remaining, span_len - 2)
                size = random.randint(GUIDE_GAP_SIZE_RANGE[0], max_size)
                starts = list(range(1, span_len - size))
                random.shuffle(starts)
                placed = False
                for start in starts:
                    cells = set(range(start, start + size))
                    if cells & gap_cells:
                        continue
                    if any((cell - 1 in gap_cells) or (cell + 1 in gap_cells) for cell in cells):
                        continue
                    gap_cells.update(cells)
                    placed = True
                    break
                if not placed:
                    break

        guide_axis = [i for i in range(span_len) if i not in gap_cells]
        invalid_axis = [i for i in range(axis_len) if i not in guide_axis]
        if len(guide_axis) >= min_guides and invalid_axis:
            return guide_axis, invalid_axis

    return list(range(min_guides)), list(range(min_guides, axis_len))


def _gen_line_pos(size: int) -> int:
    if size >= LARGE_CROSS_SIZE_MIN:
        return random.randint(LARGE_CROSS_MARGIN, size - LARGE_CROSS_MARGIN - 1)
    return random.randint(SMALL_CROSS_MARGIN, size - SMALL_CROSS_MARGIN - 1)


def _gen_can_place_source(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BACKGROUND_COLOR:
        return False
    r0 = max(0, r - 1)
    r1 = min(grid.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(grid.shape[1], c + 2)
    return not np.any(grid[r0:r1, c0:c1] != BACKGROUND_COLOR)


def _gen_place_source_from_candidates(grid, candidates, color) -> bool:
    random.shuffle(candidates)
    for r, c in candidates:
        if _gen_can_place_source(grid, r, c):
            grid[r, c] = color
            return True
    return False


def _gen_side_coords(shape, horizontal: bool, line_pos: int, axis: int, side: int):
    h, w = shape
    if horizontal:
        rows = range(0, line_pos - 1) if side < 0 else range(line_pos + 2, h)
        return [(r, axis) for r in rows]
    cols = range(0, line_pos - 1) if side < 0 else range(line_pos + 2, w)
    return [(axis, c) for c in cols]


def _gen_pick_output_slots(
    out: np.ndarray, horizontal: bool, line_pos: int, guide_axis: list[int], source_color: int
) -> list[tuple[int, int]] | None:
    """Draw output-side slots; return (side, guide-axis) pairs."""
    max_slots = min(OUTPUT_SLOT_RANGE[1], 2 * len(guide_axis))
    n_slots = random.randint(OUTPUT_SLOT_RANGE[0], max_slots)
    side_a_min = max(MIN_SLOTS_PER_SIDE, n_slots - len(guide_axis))
    side_a_max = min(len(guide_axis), n_slots - MIN_SLOTS_PER_SIDE)
    if side_a_min > side_a_max:
        return None
    n_side_a = random.randint(side_a_min, side_a_max)
    n_side_b = n_slots - n_side_a

    slots: list[tuple[int, int]] = []
    for side, n_side in ((SIDE_A, n_side_a), (SIDE_B, n_side_b)):
        for axis in random.sample(guide_axis, n_side):
            slots.append((side, axis))
            if horizontal:
                out[line_pos - 1 if side < 0 else line_pos + 1, axis] = source_color
            else:
                out[axis, line_pos - 1 if side < 0 else line_pos + 1] = source_color
    return slots


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    if h < LARGE_CROSS_SIZE_MIN and w >= LARGE_CROSS_SIZE_MIN:
        horizontal = False
    elif w < LARGE_CROSS_SIZE_MIN and h >= LARGE_CROSS_SIZE_MIN:
        horizontal = True
    else:
        horizontal = bool(random.getrandbits(1))

    line_size = w if horizontal else h
    cross_size = h if horizontal else w
    if cross_size < MIN_CROSS_SIZE:
        return None

    guide_axis, invalid_axis = _gen_pick_guide_axis(line_size)
    if not invalid_axis:
        return None

    line_pos = _gen_line_pos(cross_size)
    source_color = random.choice(SOURCE_COLORS)

    out = np.zeros((h, w), dtype=int)
    if horizontal:
        out[line_pos, guide_axis] = GUIDE_COLOR
    else:
        out[guide_axis, line_pos] = GUIDE_COLOR

    slots = _gen_pick_output_slots(out, horizontal, line_pos, guide_axis, source_color)
    if slots is None:
        return None

    n_slots = len(slots)
    total_sources_min = max(SOURCE_COUNT_RANGE[0], n_slots + ERASED_SOURCE_RANGE[0])
    if total_sources_min > SOURCE_COUNT_RANGE[1]:
        return None
    total_sources = random.randint(total_sources_min, SOURCE_COUNT_RANGE[1])
    erased_count = random.randint(
        ERASED_SOURCE_RANGE[0], min(ERASED_SOURCE_RANGE[1], total_sources - n_slots)
    )
    duplicate_count = total_sources - erased_count - n_slots
    duplicate_slots = set(random.sample(range(len(slots)), duplicate_count))

    inp = np.zeros((h, w), dtype=int)
    if horizontal:
        inp[line_pos, guide_axis] = GUIDE_COLOR
    else:
        inp[guide_axis, line_pos] = GUIDE_COLOR

    random.shuffle(slots)
    for i, (side, axis) in enumerate(slots):
        count = 2 if i in duplicate_slots else 1
        for _ in range(count):
            candidates = _gen_side_coords((h, w), horizontal, line_pos, axis, side)
            if not _gen_place_source_from_candidates(inp, candidates, source_color):
                return None

    erased_candidates: list[tuple[int, int]] = []
    for axis in invalid_axis:
        erased_candidates.extend(_gen_side_coords((h, w), horizontal, line_pos, axis, SIDE_A))
        erased_candidates.extend(_gen_side_coords((h, w), horizontal, line_pos, axis, SIDE_B))
    for _ in range(erased_count):
        if not _gen_place_source_from_candidates(inp, erased_candidates.copy(), source_color):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f83cb3f6 input")

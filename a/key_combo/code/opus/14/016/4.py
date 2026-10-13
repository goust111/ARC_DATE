import random
from typing import Iterable

import numpy as np


BACKGROUND = 5
MISSING = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _infer_periodic_legend(damaged_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row = np.asarray(damaged_row, dtype=int)
    width = int(row.shape[0])
    candidates: list[tuple[int, tuple[int, ...]]] = []

    for period in range(1, width + 1):
        phase_colors = [-1] * period
        seen = [False] * period
        valid = True
        for c, value in enumerate(row):
            value = int(value)
            if value == MISSING:
                continue
            phase = c % period
            if not seen[phase]:
                phase_colors[phase] = value
                seen[phase] = True
            elif phase_colors[phase] != value:
                valid = False
                break
        if not valid or not all(seen):
            continue
        if len(set(phase_colors)) != period:
            continue
        candidates.append((period, tuple(phase_colors)))

    assert candidates, "no consistent legend period found"
    min_period = min(period for period, _ in candidates)
    minimal = [colors for period, colors in candidates if period == min_period]
    assert len(minimal) == 1, "minimal legend period must be unique"

    period_colors = np.array(minimal[0], dtype=int)
    full_row = np.array([period_colors[c % min_period] for c in range(width)], dtype=int)
    assert np.all(full_row[row != MISSING] == row[row != MISSING])
    return period_colors, full_row


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = []
    height, width = map(int, grid.shape)
    for r in range(1, height):
        for c in range(width):
            value = int(grid[r, c])
            if value in (BACKGROUND, MISSING):
                continue
            markers.append((r, c))
    return sorted(markers)


def _paint_row(out: np.ndarray, row_idx: int, full_row: np.ndarray, start_c: int) -> None:
    width = int(out.shape[1])
    for offset, value in enumerate(full_row):
        c = start_c + offset
        if 0 <= c < width:
            out[row_idx, c] = int(value)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer and repair the damaged top periodic legend, then expand it onto marker rows using color-index alignment.

    Input structure:
    - The background is grey (5).
    - Row 0 is a horizontal legend string made by repeating a short color period, with 1-3 cells replaced by black (0) holes.
    - All other non-background, non-black cells are isolated one-pixel markers whose colors appear in exactly one phase of the true period.

    Rule:
    - Treat black cells on the top legend row as missing data and infer the smallest period consistent with every visible legend cell.
    - Fill the holes with their period phase colors to recover the complete top-row legend.
    - For each interior marker, find the unique period index of its color, copy the repaired full top row onto the marker row, and shift that copy so this index lands on the marker column.
    - Clip any copied cells that fall outside the grid.

    Key insight:
    - A marker color identifies not an absolute visible column but a hidden period phase, so the legend must be repaired before the expansion offset is knowable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2

    period_colors, full_row = _infer_periodic_legend(g[0])
    out = g.copy()
    out[0] = full_row

    markers = _marker_positions(g)
    color_to_phase = {int(color): phase for phase, color in enumerate(period_colors)}

    for r, c in markers:
        phase = color_to_phase[int(g[r, c])]
        _paint_row(out, r, full_row, int(c) - int(phase))

    return out


def _phase_visible(width: int, period: int, holes: Iterable[int]) -> bool:
    blocked = set(int(x) for x in holes)
    for phase in range(period):
        if not any((c % period) == phase and c not in blocked for c in range(width)):
            return False
    return True


def _expected_output(grid: np.ndarray, period_colors: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    width = int(grid.shape[1])
    full_row = np.array([period_colors[c % len(period_colors)] for c in range(width)], dtype=int)
    out[0] = full_row
    for r, c in _marker_positions(grid):
        phase = int(np.where(period_colors == grid[r, c])[0][0])
        _paint_row(out, r, full_row, int(c) - phase)
    return out


def _choose_holes(width: int, period: int, required_hole: int, hole_count: int) -> set[int]:
    holes = {int(required_hole)}
    positions = list(range(width))
    random.shuffle(positions)
    for pos in positions:
        if len(holes) >= hole_count:
            break
        candidate = holes | {pos}
        if _phase_visible(width, period, candidate):
            holes = candidate
    return holes


def _sample_rows(height: int, count: int) -> list[int]:
    rows = list(range(2, height, 2))
    random.shuffle(rows)
    chosen = sorted(rows[:count])
    assert len(chosen) == count
    return chosen


def _marker_col(phase: int, width: int, clip_type: str) -> int:
    if clip_type == "left":
        assert phase > 0
        return random.randint(0, phase - 1)
    if clip_type == "none":
        return phase
    assert clip_type == "right"
    return random.randint(phase + 1, width - 1)


def _validate_input(grid: np.ndarray, period_colors: np.ndarray, hidden_phase: int, holes: set[int]) -> bool:
    period = len(period_colors)
    width = int(grid.shape[1])
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert 1 <= len(holes) <= 3
    assert hidden_phase in range(period)
    assert int(grid[0, hidden_phase]) == MISSING
    assert _phase_visible(width, period, holes)

    inferred_period_colors, repaired = _infer_periodic_legend(grid[0])
    assert np.array_equal(inferred_period_colors, period_colors)
    expected = _expected_output(grid, period_colors)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    marker_positions = _marker_positions(grid)
    assert marker_positions
    for r, c in marker_positions:
        assert int(grid[r, c]) in period_colors
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    assert int(grid[rr, cc]) in (BACKGROUND, MISSING)

    hidden_color = int(period_colors[hidden_phase])
    hidden_markers = [(r, c) for r, c in marker_positions if int(grid[r, c]) == hidden_color]
    assert hidden_markers, "at least one marker must depend on a hidden first-cycle anchor"
    visible_occurrences = [c for c in range(width) if int(grid[0, c]) == hidden_color]
    assert visible_occurrences and visible_occurrences[0] > hidden_phase

    sample_r, sample_c = hidden_markers[0]
    correct_row = actual[sample_r].copy()
    naive = grid.copy()
    naive[0] = repaired
    _paint_row(naive, sample_r, repaired, int(sample_c) - visible_occurrences[0])
    assert not np.array_equal(correct_row, naive[sample_r]), "repair step must change the anchor choice"
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    period = random.randint(2, 4)

    colors = np.array(random.sample(PALETTE, period), dtype=int)
    hidden_phase = random.randint(1, period - 1)
    hole_count = random.randint(1, min(3, width - 1))
    holes = _choose_holes(width, period, hidden_phase, hole_count)
    if len(holes) != hole_count:
        return None

    marker_slots = list(range(2, height, 2))
    if len(marker_slots) < 3:
        return None
    marker_count = random.randint(3, min(5, len(marker_slots)))
    rows = _sample_rows(height, marker_count)

    clip_plan = ["left", "none", "right"]
    while len(clip_plan) < marker_count:
        clip_plan.append(random.choice(("left", "none", "right")))
    random.shuffle(clip_plan)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    full_row = np.array([int(colors[c % period]) for c in range(width)], dtype=int)
    grid[0] = full_row
    for hole in holes:
        grid[0, hole] = MISSING

    available_left_phases = [phase for phase in range(period) if phase > 0]
    if not available_left_phases:
        return None

    hidden_assigned = False
    for row, clip_type in zip(rows, clip_plan):
        if clip_type == "left":
            phase = hidden_phase if not hidden_assigned else random.choice(available_left_phases)
        else:
            phase = random.randint(0, period - 1)
        col = _marker_col(phase, width, clip_type)
        grid[row, col] = int(colors[phase])
        hidden_assigned = hidden_assigned or (phase == hidden_phase)

    if not hidden_assigned:
        row = rows[0]
        col = _marker_col(hidden_phase, width, "left")
        grid[row] = BACKGROUND
        grid[row, col] = int(colors[hidden_phase])

    try:
        if _validate_input(grid, colors, hidden_phase, holes):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

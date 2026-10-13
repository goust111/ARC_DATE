"""ARC task 7: cascade water through successively lower terrain basins."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
WATER = 1
MARKER = 2
TERRAIN = 5

_MAX_GENERATION_ATTEMPTS = 400


def _terrain_heights(grid: np.ndarray) -> np.ndarray:
    """Return bottom-up gray-column heights, rejecting non-column terrain."""
    h, w = grid.shape
    heights = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == TERRAIN)
        assert rows.size > 0, "every column must contain terrain"
        height = int(rows.size)
        assert np.array_equal(rows, np.arange(h - height, h)), (
            "gray terrain in each column must be one bottom-anchored run"
        )
        heights[c] = height
    return heights


def _strict_peaks(heights: np.ndarray) -> list[int]:
    """Find the single-column summits which delimit the visible basins."""
    return [
        c
        for c in range(1, len(heights) - 1)
        if int(heights[c]) > int(heights[c - 1])
        and int(heights[c]) > int(heights[c + 1])
    ]


def _entry_basin(peaks: list[int], marker_col: int) -> int:
    candidates = [
        i
        for i in range(len(peaks) - 1)
        if peaks[i] < marker_col < peaks[i + 1]
    ]
    assert len(candidates) == 1, "the marker must lie over exactly one bounded basin"
    return candidates[0]


def _paint_basin(
    out: np.ndarray,
    heights: np.ndarray,
    left_peak: int,
    right_peak: int,
) -> None:
    """Fill one open peak-to-peak interval up to its lower summit."""
    canvas_h = out.shape[0]
    level = min(int(heights[left_peak]), int(heights[right_peak]))
    for c in range(left_peak + 1, right_peak):
        terrain_top = canvas_h - int(heights[c])
        water_top = canvas_h - level
        if water_top < terrain_top:
            cells = out[water_top:terrain_top, c]
            cells[cells == BACKGROUND] = WATER


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the marked basin and relay overflow through successively lower basins to an edge.

    Input structure:
    - A black background lies above bottom-anchored, contiguous gray terrain columns.
    - Strict single-column terrain peaks delimit basins, and one airborne red singleton
      lies over exactly one internal basin.
    - Every pair of rims encountered by the water has unequal heights; after the first
      spill, each next far rim is lower than the peak just crossed.

    Rule:
    - Fill all cells above a basin floor and no higher than its lower rim with blue.
    - Compare that basin's two peak heights, cross the lower peak, and repeat in the
      adjacent basin in that direction. Stop after crossing the outermost peak to the
      canvas edge; the unbounded edge region receives no blue.
    - Peak columns, gray terrain, and the red marker remain unchanged.

    Key insight:
    - The marker selects only the first basin. Unequal successive rim heights determine
      both each local waterline and a one-way causal chain of downstream basins.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional grid"
    assert g.size > 0
    assert set(map(int, np.unique(g))) <= {BACKGROUND, MARKER, TERRAIN}, (
        "input may contain only black background, red marker, and gray terrain"
    )

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one red singleton marker"
    marker_r, marker_c = map(int, marker_coords[0])

    heights = _terrain_heights(g)
    assert marker_r < g.shape[0] - int(heights[marker_c]), (
        "the marker must be in the air above its terrain column"
    )
    peaks = _strict_peaks(heights)
    assert len(peaks) >= 2, "at least two terrain peaks are required"
    basin = _entry_basin(peaks, marker_c)

    out = g.copy()
    visited: set[int] = set()
    direction = 0

    while True:
        assert basin not in visited, "overflow chain must never reverse or cycle"
        visited.add(basin)
        left_peak, right_peak = peaks[basin], peaks[basin + 1]
        left_h = int(heights[left_peak])
        right_h = int(heights[right_peak])
        assert left_h != right_h, "each traversed basin must have unequal rim heights"

        step = -1 if left_h < right_h else 1
        if direction:
            assert step == direction, "the next far rim must keep overflow moving forward"
        else:
            direction = step

        _paint_basin(out, heights, left_peak, right_peak)

        next_basin = basin + step
        if next_basin < 0 or next_basin >= len(peaks) - 1:
            break
        basin = next_basin

    assert np.array_equal(out[g == TERRAIN], g[g == TERRAIN])
    assert int(out[marker_r, marker_c]) == MARKER
    return out


@dataclass(frozen=True)
class _Plan:
    peaks: tuple[int, ...]
    peak_heights: tuple[int, ...]
    entry: int
    direction: int
    flooded: tuple[int, ...]


def _sample_peak_heights(n_peaks: int, entry: int, direction: int) -> list[int]:
    """Build a strictly descending outward spill chain plus unconstrained distractors."""
    values: list[int | None] = [None] * n_peaks
    outer = random.randint(5, 7)

    if direction == 1:
        values[-1] = outer
        for j in range(n_peaks - 2, entry, -1):
            values[j] = int(values[j + 1]) + random.randint(1, 2)
        values[entry] = int(values[entry + 1]) + random.randint(2, 4)
        for j in range(entry - 1, -1, -1):
            forbidden = int(values[j + 1])
            choices = [v for v in range(5, 14) if v != forbidden]
            values[j] = random.choice(choices)
    else:
        values[0] = outer
        for j in range(1, entry + 1):
            values[j] = int(values[j - 1]) + random.randint(1, 2)
        values[entry + 1] = int(values[entry]) + random.randint(2, 4)
        for j in range(entry + 2, n_peaks):
            forbidden = int(values[j - 1])
            choices = [v for v in range(5, 14) if v != forbidden]
            values[j] = random.choice(choices)

    return [int(v) for v in values]


def _sample_peak_positions(n_peaks: int) -> tuple[list[int], int]:
    for _ in range(100):
        left_margin = random.randint(2, 4)
        right_margin = random.randint(2, 4)
        gaps = [random.randint(3, 5) for _ in range(n_peaks - 1)]
        peaks = [left_margin]
        for gap in gaps:
            peaks.append(peaks[-1] + gap)
        width = peaks[-1] + right_margin + 1
        if width <= 30:
            return peaks, width
    raise RuntimeError("could not fit terrain peaks within ARC width")


def _valley_profile(left_h: int, right_h: int, width: int) -> list[int]:
    """Return a U-shaped interior contour with no accidental local summit."""
    assert width >= 2
    rim = min(left_h, right_h)
    bottom = max(1, rim - random.randint(2, min(5, rim - 1)))
    center = random.randrange(width)
    values = [bottom] * width

    current = bottom
    for j in range(center - 1, -1, -1):
        current = min(rim - 1, current + random.randint(0, 1))
        values[j] = current
    current = bottom
    for j in range(center + 1, width):
        current = min(rim - 1, current + random.randint(0, 1))
        values[j] = current
    return values


def _edge_profile(peak_h: int, width: int, *, reverse: bool) -> list[int]:
    """Make a monotone slope between an outer peak and the canvas edge."""
    shoulder = peak_h - random.randint(1, min(3, peak_h - 1))
    near_to_far = [shoulder]
    for _ in range(width - 1):
        near_to_far.append(max(1, near_to_far[-1] - random.randint(0, 2)))
    return near_to_far if reverse else list(reversed(near_to_far))


def _planned_output(
    inp: np.ndarray,
    heights: np.ndarray,
    plan: _Plan,
) -> np.ndarray:
    """Generator oracle based on stored construction metadata, not peak detection."""
    out = inp.copy()
    canvas_h = inp.shape[0]
    for basin in plan.flooded:
        left, right = plan.peaks[basin], plan.peaks[basin + 1]
        level = min(plan.peak_heights[basin], plan.peak_heights[basin + 1])
        for c in range(left + 1, right):
            for r in range(canvas_h - level, canvas_h - int(heights[c])):
                if int(out[r, c]) == BACKGROUND:
                    out[r, c] = WATER
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    heights: np.ndarray,
    plan: _Plan,
) -> bool:
    """Check uniqueness, anti-shortcuts, and that every cascade step matters."""
    try:
        detected_heights = _terrain_heights(inp)
        detected_peaks = _strict_peaks(detected_heights)
        marker_col = int(np.argwhere(inp == MARKER)[0, 1])
        if not np.array_equal(detected_heights, heights):
            return False
        if detected_peaks != list(plan.peaks):
            return False
        if _entry_basin(detected_peaks, marker_col) != plan.entry:
            return False

        # At least two basins prove relay; at least one untouched basin disproves
        # "fill every basin" and makes the singleton marker operationally necessary.
        all_basins = set(range(len(plan.peaks) - 1))
        flooded = set(plan.flooded)
        if len(flooded) < 2 or not (all_basins - flooded):
            return False

        levels = []
        for basin in plan.flooded:
            left, right = plan.peaks[basin], plan.peaks[basin + 1]
            level = min(plan.peak_heights[basin], plan.peak_heights[basin + 1])
            levels.append(level)
            if not np.any(expected[:, left + 1 : right] == WATER):
                return False
            # Every planned waterline must actually be visible at its exact row.
            if not np.any(expected[inp.shape[0] - level, left + 1 : right] == WATER):
                return False
        if len(levels) != len(set(levels)):
            return False

        for basin in all_basins - flooded:
            left, right = plan.peaks[basin], plan.peaks[basin + 1]
            if np.any(expected[:, left + 1 : right] == WATER):
                return False

        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(inp, expected):
            return False
        if not np.array_equal(actual[inp != BACKGROUND], inp[inp != BACKGROUND]):
            return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _generate_attempt() -> np.ndarray | None:
    n_peaks = random.randint(4, 6)
    entry = random.randint(1, n_peaks - 3)
    direction = random.choice((-1, 1))

    peak_heights = _sample_peak_heights(n_peaks, entry, direction)
    peaks, width = _sample_peak_positions(n_peaks)
    min_height = max(14, max(peak_heights) + 4)
    if min_height > 30:
        return None
    canvas_h = random.randint(min_height, min(30, min_height + 7))

    heights = np.zeros(width, dtype=int)
    for c, value in zip(peaks, peak_heights):
        heights[c] = value

    for i in range(n_peaks - 1):
        left, right = peaks[i], peaks[i + 1]
        profile = _valley_profile(
            peak_heights[i], peak_heights[i + 1], right - left - 1
        )
        heights[left + 1 : right] = profile

    heights[: peaks[0]] = _edge_profile(
        peak_heights[0], peaks[0], reverse=False
    )
    edge_width = width - peaks[-1] - 1
    heights[peaks[-1] + 1 :] = _edge_profile(
        peak_heights[-1], edge_width, reverse=True
    )

    inp = np.full((canvas_h, width), BACKGROUND, dtype=np.int8)
    for c, terrain_h in enumerate(heights):
        inp[canvas_h - int(terrain_h) :, c] = TERRAIN

    left, right = peaks[entry], peaks[entry + 1]
    marker_col = random.randint(left + 1, right - 1)
    highest_rim = max(peak_heights[entry], peak_heights[entry + 1])
    marker_max_row = canvas_h - highest_rim - 2
    if marker_max_row < 0:
        return None
    marker_row = random.randint(0, marker_max_row)
    inp[marker_row, marker_col] = MARKER

    flooded = (
        tuple(range(entry, n_peaks - 1))
        if direction == 1
        else tuple(range(entry, -1, -1))
    )
    plan = _Plan(
        tuple(peaks), tuple(peak_heights), entry, direction, flooded
    )
    expected = _planned_output(inp, heights, plan)
    if not _semantic_checks(inp, expected, heights, plan):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate left- and right-going overflow chains."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-overflow input")

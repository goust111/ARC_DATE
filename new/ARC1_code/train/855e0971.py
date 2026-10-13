from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present


def _unique_rarest_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    assert len(counts) >= 2, "expected marker pixels and at least one region color"
    min_count = min(counts.values())
    rare = [color for color, count in counts.items() if count == min_count]
    assert len(rare) == 1, "expected a unique rare marker color"
    return int(rare[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each black marker into a full divider line through its colored rectangle.

    Input structure
    - The grid is partitioned into non-square monochrome rectangular regions.
    - A unique rare marker color appears as isolated pixels inside those regions.
    - Each region is otherwise filled with exactly one non-marker color.

    Rule
    - For each wide region, replace the full region-height column through every
      marker with the marker color.
    - For each tall region, replace the full region-width row through every
      marker with the marker color.
    - Preserve colored regions that contain no marker.

    Key insight
    - The marker does not spread globally; the aspect ratio of its own rectangle
      chooses the perpendicular divider direction and the rectangle bounds stop it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker = _unique_rarest_color(g)
    out = g.copy()
    assigned_markers = np.zeros(g.shape, dtype=bool)

    for color in colors_present(g, exclude={marker}):
        region_mask = g == color
        region_bbox = bbox(region_mask)
        assert region_bbox is not None, "expected each region color to be present"
        r0, c0, r1, c1 = region_bbox

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((region == color) | (region == marker)), (
            "expected a rectangular region with only its own color and markers"
        )

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height != width, "expected each colored region to be non-square"

        markers = np.argwhere(region == marker)
        for rr, cc in markers:
            r = int(r0 + rr)
            c = int(c0 + cc)
            assigned_markers[r, c] = True
            if width > height:
                out[r0 : r1 + 1, c] = marker
            else:
                out[r, c0 : c1 + 1] = marker

    assert np.array_equal(assigned_markers, g == marker), (
        "expected every marker to lie in exactly one colored region"
    )
    return out

# --------------- generator (o5x) ---------------

# Canvas and region dimensions.
MIN_GRID_SIZE, MAX_GRID_SIZE = 13, 19
MIN_REGION_COUNT, MAX_REGION_COUNT = 2, 4
MIN_PART_SIZE, MAX_PART_SIZE = 2, 8

# Marker placement constraints.
MARKER_COLOR = 0
MIN_MARKER_TOTAL, MAX_MARKER_TOTAL = 2, 5
MIN_MARKED_REGIONS, MAX_UNMARKED_REGIONS = 2, 1
MIN_MARKERS_PER_REGION, MAX_MARKERS_PER_REGION = 0, 2
MARKER_EDGE_PADDING = 1

# Color selection and generation retry limits.
REGION_COLOR_POOL = np.arange(1, 10)
GENERATION_RETRIES = 300


def _bounded_compositions(total, count, min_value, max_value):
    values: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 0:
            if remaining == 0:
                values.append(tuple(prefix))
            return
        tail_slots = slots - 1
        low = max(min_value, remaining - max_value * tail_slots)
        high = min(max_value, remaining - min_value * tail_slots)
        for value in range(low, high + 1):
            prefix.append(value)
            rec(prefix, remaining - value, tail_slots)
            prefix.pop()

    rec([], total, count)
    return values


def _gen_marker_distributions(region_count: int, total: int) -> list[tuple[int, ...]]:
    min_marked_regions = max(MIN_MARKED_REGIONS, region_count - MAX_UNMARKED_REGIONS)
    all_counts = _bounded_compositions(
        total, region_count, MIN_MARKERS_PER_REGION, MAX_MARKERS_PER_REGION
    )
    return [
        counts
        for counts in all_counts
        if sum(value > 0 for value in counts) >= min_marked_regions
    ]


def _gen_choice(items):
    return items[int(np.random.randint(len(items)))]


def _design_layout() -> tuple[bool, int, int, tuple[int, ...]]:
    wide_regions = bool(np.random.randint(2))
    region_count = int(np.random.randint(MIN_REGION_COUNT, MAX_REGION_COUNT + 1))

    valid_stack_sizes = [
        size
        for size in range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        if _bounded_compositions(size, region_count, MIN_PART_SIZE, MAX_PART_SIZE)
    ]
    stack_size = int(_gen_choice(valid_stack_sizes))
    long_size = int(np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    parts = _gen_choice(
        _bounded_compositions(stack_size, region_count, MIN_PART_SIZE, MAX_PART_SIZE)
    )

    if wide_regions:
        rows, cols = stack_size, long_size
    else:
        rows, cols = long_size, stack_size
    return wide_regions, rows, cols, parts


def _gen_marker_counts(region_count: int) -> tuple[int, ...]:
    marker_total_options = [
        total
        for total in range(MIN_MARKER_TOTAL, MAX_MARKER_TOTAL + 1)
        if _gen_marker_distributions(region_count, total)
    ]
    return _gen_choice(
        _gen_marker_distributions(region_count, int(_gen_choice(marker_total_options)))
    )


def _gen_marker_positions(row_options, col_options, marker_count):
    marker_rows = np.random.choice(row_options, size=marker_count, replace=False)
    marker_cols = np.random.choice(col_options, size=marker_count, replace=False)
    return zip(marker_rows, marker_cols)


def _render_regions(wide_regions, rows, cols, parts, colors, marker_counts):
    inp = np.empty((rows, cols), dtype=int)
    expected = np.empty((rows, cols), dtype=int)

    offset = 0
    for idx, part in enumerate(parts):
        color = int(colors[idx])
        marker_count = marker_counts[idx]
        if wide_regions:
            r0, r1 = offset, offset + part
            inp[r0:r1, :] = color
            expected[r0:r1, :] = color
            if marker_count:
                row_options = np.arange(r0, r1)
                col_options = np.arange(MARKER_EDGE_PADDING, cols - MARKER_EDGE_PADDING)
                for r, c in _gen_marker_positions(row_options, col_options, marker_count):
                    inp[int(r), int(c)] = MARKER_COLOR
                    expected[r0:r1, int(c)] = MARKER_COLOR
            offset = r1
        else:
            c0, c1 = offset, offset + part
            inp[:, c0:c1] = color
            expected[:, c0:c1] = color
            if marker_count:
                row_options = np.arange(MARKER_EDGE_PADDING, rows - MARKER_EDGE_PADDING)
                col_options = np.arange(c0, c1)
                for r, c in _gen_marker_positions(row_options, col_options, marker_count):
                    inp[int(r), int(c)] = MARKER_COLOR
                    expected[int(r), c0:c1] = MARKER_COLOR
            offset = c1

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    wide_regions, rows, cols, parts = _design_layout()
    colors = np.random.choice(REGION_COLOR_POOL, size=len(parts), replace=False)
    marker_counts = _gen_marker_counts(len(parts))
    inp, expected = _render_regions(
        wide_regions, rows, cols, parts, colors, marker_counts
    )
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 855e0971")

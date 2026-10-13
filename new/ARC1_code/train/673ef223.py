from __future__ import annotations

import numpy as np

from arc_utils import Color, components_mask, get_bg


def _vertical_edge_bars(g: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = map(int, g.shape)
    bars: list[tuple[int, int, int]] = []

    for coords in components_mask(g == Color.RED, connectivity=4):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert c0 == c1, "red objects must be vertical bars"
        assert c0 in (0, width - 1), "red bars must lie on the left or right edge"
        assert coords.shape[0] == r1 - r0 + 1, "red bars must be solid"
        assert 0 <= r0 <= r1 < height
        bars.append((r0, r1, c0))

    assert len(bars) == 2, "expected exactly two red edge bars"
    assert {bar[2] for bar in bars} == {0, width - 1}, "expected one bar on each side"
    assert bars[0][1] - bars[0][0] == bars[1][1] - bars[1][0], "bars must have equal height"
    assert bars[0][1] < bars[1][0] or bars[1][1] < bars[0][0], "bar row ranges must not overlap"
    return bars


def _paint_between_marker_and_bar(out: np.ndarray, row: int, marker_col: int, bar_col: int) -> None:
    if bar_col == 0:
        out[row, bar_col + 1 : marker_col] = Color.TEAL
    else:
        out[row, marker_col + 1 : bar_col] = Color.TEAL
    out[row, marker_col] = Color.YELLOW


def _paint_full_ray_from_bar(out: np.ndarray, row: int, bar_col: int) -> None:
    if bar_col == 0:
        out[row, bar_col + 1 :] = Color.TEAL
    else:
        out[row, :bar_col] = Color.TEAL
    out[row, bar_col] = Color.RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw horizontal teal rays from paired red edge bars on rows selected by teal markers, turning each marker yellow.

    Input structure
    - The black background contains exactly two equal-height solid vertical red bars, one on each side edge.
    - Teal marker pixels are isolated single cells whose rows overlap exactly one of the two red bars.
    - All teal markers are aligned with the same red bar.

    Rule
    - For each teal marker, paint teal background cells horizontally between that marker and its red bar, then recolor the marker yellow.
    - Use the marker's vertical offset within the source red bar to select the same offset row of the opposite red bar.
    - On that paired row, paint a full teal ray from the opposite red bar to the far edge, preserving the red bar cell.

    Key insight
    - The teal markers do not move; they choose relative rows in a pair of equal-height edge bars, and the opposite bar emits the same row selections as complete rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.RED, Color.TEAL}, "unexpected input colors"

    height, width = map(int, g.shape)
    bars = _vertical_edge_bars(g)

    marker_components = components_mask(g == Color.TEAL, connectivity=4)
    assert marker_components, "expected at least one teal marker"
    assert all(comp.shape[0] == 1 for comp in marker_components), "teal markers must be isolated pixels"
    markers = [tuple(map(int, comp[0])) for comp in marker_components]

    source_indices: list[int] = []
    offsets: list[int] = []
    for row, col in markers:
        containing = [idx for idx, (r0, r1, _bar_col) in enumerate(bars) if r0 <= row <= r1]
        assert len(containing) == 1, "each marker row must overlap exactly one bar"
        source_idx = containing[0]
        bar_col = bars[source_idx][2]
        assert col != bar_col, "marker cannot sit on its source red bar"
        assert 0 <= col < width
        source_indices.append(source_idx)
        offsets.append(row - bars[source_idx][0])

    assert len(set(source_indices)) == 1, "all markers must refer to the same source bar"
    assert len(set(offsets)) == len(offsets), "each selected bar offset must be unique"

    source_idx = source_indices[0]
    target_idx = 1 - source_idx
    source_r0, _source_r1, source_col = bars[source_idx]
    target_r0, _target_r1, target_col = bars[target_idx]

    out = g.copy()
    for marker_row, marker_col in markers:
        offset = marker_row - source_r0
        target_row = target_r0 + offset
        assert 0 <= target_row < height
        assert int(g[target_row, target_col]) == Color.RED
        source_path = slice(min(marker_col, source_col) + 1, max(marker_col, source_col))
        target_path = slice(1, width) if target_col == 0 else slice(0, width - 1)
        assert np.all(g[marker_row, source_path] == bg), "source ray path must be empty"
        assert np.all(g[target_row, target_path] == bg), "paired full ray path must be empty"

        _paint_between_marker_and_bar(out, marker_row, marker_col, source_col)
        _paint_full_ray_from_bar(out, target_row, target_col)

    return out

# --------------- generator (o5x) ---------------

# Canvas and red bar geometry.
GRID_HEIGHT_RANGE = (19, 22)  # np.random.randint high bound is exclusive.
GRID_WIDTH_RANGE = (8, 13)
BAR_HEIGHT_RANGE = (4, 7)
EDGE_MARGIN = 1
MIN_BAR_GAP = 4

# Teal marker placement.
MARKER_COUNT_RANGE = (1, 3)
MIN_MARKER_DISTANCE_FROM_SOURCE = 3

# Generation retry budget.
GENERATION_RETRIES = 300


def _sample_bar_layout(height: int, width: int, bar_height: int) -> tuple[int, int, int, int] | None:
    """Return source/target row starts and columns for the paired edge bars."""
    valid_starts = range(EDGE_MARGIN, height - bar_height - EDGE_MARGIN + 1)
    valid_pairs = [
        (source_r0, target_r0)
        for source_r0 in valid_starts
        for target_r0 in valid_starts
        if target_r0 >= source_r0 + bar_height + MIN_BAR_GAP
    ]
    if not valid_pairs:
        return None
    source_r0, target_r0 = valid_pairs[int(np.random.randint(len(valid_pairs)))]
    source_col = int(np.random.choice([0, width - 1]))
    target_col = width - 1 - source_col
    return source_r0, target_r0, source_col, target_col


def _sample_marker_specs(bar_height: int, width: int, source_col: int) -> list[tuple[int, int]] | None:
    """Return (bar offset, marker column) pairs while preserving marker isolation."""
    n_markers = int(np.random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], bar_height) + 1))
    offsets = np.random.choice(np.arange(bar_height), size=n_markers, replace=False)
    offsets = [int(offset) for offset in offsets]
    if source_col == 0:
        marker_col_options = list(range(MIN_MARKER_DISTANCE_FROM_SOURCE, width - EDGE_MARGIN))
    else:
        marker_col_options = list(range(EDGE_MARGIN, width - MIN_MARKER_DISTANCE_FROM_SOURCE))
    if not marker_col_options:
        return None

    marker_specs: list[tuple[int, int]] = []
    for offset in offsets:
        banned_cols = {
            col
            for prev_offset, col in marker_specs
            if abs(prev_offset - offset) == 1
        }
        choices = [col for col in marker_col_options if col not in banned_cols]
        if not choices:
            return None
        marker_specs.append((offset, int(np.random.choice(choices))))
    return marker_specs


def _render_grids(
    height: int, width: int, bar_height: int,
    source_r0: int, target_r0: int, source_col: int, target_col: int,
    marker_specs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampled input and the output expected from main()."""
    inp = np.zeros((height, width), dtype=int)
    inp[source_r0 : source_r0 + bar_height, source_col] = Color.RED
    inp[target_r0 : target_r0 + bar_height, target_col] = Color.RED
    for offset, marker_col in marker_specs:
        inp[source_r0 + offset, marker_col] = Color.TEAL
    expected = inp.copy()
    for offset, marker_col in marker_specs:
        marker_row = source_r0 + offset
        target_row = target_r0 + offset
        _paint_between_marker_and_bar(expected, marker_row, marker_col, source_col)
        _paint_full_ray_from_bar(expected, target_row, target_col)
    return inp, expected


def _passes_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*GRID_HEIGHT_RANGE))
    width = int(np.random.randint(*GRID_WIDTH_RANGE))
    bar_height = int(np.random.randint(*BAR_HEIGHT_RANGE))

    layout = _sample_bar_layout(height, width, bar_height)
    if layout is None:
        return None
    source_r0, target_r0, source_col, target_col = layout

    marker_specs = _sample_marker_specs(bar_height, width, source_col)
    if marker_specs is None:
        return None

    inp, expected = _render_grids(height, width, bar_height, source_r0, target_r0, source_col, target_col, marker_specs)
    if not _passes_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 673ef223 input")

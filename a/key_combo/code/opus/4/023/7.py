from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, find_components
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    find_components = _ARC_UTILS.find_components


BG = int(Color.BLACK)
BAR = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)

HEIGHT_OPTIONS = (25, 27, 29)
WIDTH_RANGE = (12, 20)
BAR_COUNT_RANGE = (9, 12)
GENERATION_RETRIES = 400


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, BAR}, "input must contain only black and blue"

    objects = find_components(g, bg=BG, connectivity=4, monochromatic=True)
    assert objects, "expected at least one foreground bar"

    bars: list[dict] = []
    for obj in objects:
        assert int(obj["color"]) == BAR, "all foreground objects must be blue"
        r0, c0, r1, c1 = map(int, obj["bbox"])
        coords = obj["coords"]
        assert r0 == r1, "each bar must have height 1"
        assert c1 - c0 + 1 >= 3, "each bar must be at least three cells wide"
        assert int(obj["area"]) == c1 - c0 + 1, "each object must be a solid horizontal rectangle"
        assert np.all(g[r0, c0 : c1 + 1] == BAR), "bar interior must be solid blue"
        assert (r0, c0) == tuple(coords.min(axis=0)), "bbox must match the object coords"
        assert (r1, c1) == tuple(coords.max(axis=0)), "bbox must match the object coords"
        bars.append({"row": r0, "c0": c0, "c1": c1, "coords": coords})

    bars.sort(key=lambda bar: (bar["row"], bar["c0"]))
    return bars


def _symmetry_flags(bars: list[dict], height: int) -> list[bool]:
    span_by_row = {int(bar["row"]): (int(bar["c0"]), int(bar["c1"])) for bar in bars}
    flags = []
    for bar in bars:
        mirror_row = int(height - 1 - bar["row"])
        flags.append(
            mirror_row != int(bar["row"])
            and span_by_row.get(mirror_row) == (int(bar["c0"]), int(bar["c1"]))
        )
    return flags


def _bar_color(is_selected: bool, is_symmetric: bool) -> int:
    if is_selected and is_symmetric:
        return RED
    if is_selected:
        return YELLOW
    if is_symmetric:
        return GREEN
    return BAR


def _transform_from_bars(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    symmetric = _symmetry_flags(bars, out.shape[0])
    for i, bar in enumerate(bars):
        color = _bar_color(i % 3 == 0, symmetric[i])
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _selection_only_output(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for i, bar in enumerate(bars):
        if i % 3 == 0:
            coords = bar["coords"]
            out[coords[:, 0], coords[:, 1]] = RED
    return out


def _symmetry_only_output(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    symmetric = _symmetry_flags(bars, out.shape[0])
    for bar, is_symmetric in zip(bars, symmetric):
        if is_symmetric:
            coords = bar["coords"]
            out[coords[:, 0], coords[:, 1]] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal blue bars by combining every-third selection with horizontal reflection pairing.

    Input structure:
    - Black cells are the background.
    - The foreground consists of several solid blue horizontal bars of height 1.
    - Some bars have a reflected partner on the mirror row across the grid's horizontal midline with exactly the same column span.

    Rule:
    - Order all blue bars from top to bottom, breaking ties by left column.
    - Mark bars with ranks 1, 4, 7, 10, ... as selected.
    - Independently mark any bar as symmetric if another bar lies on row `H-1-r` and has the same left and right columns.
    - Recolor selected+symmetric bars red, selected-only bars yellow, unselected+symmetric bars green, and leave unselected non-symmetric bars blue.

    Key insight:
    - The final color is the conjunction of two independent properties: ordinal position in the vertical sequence and geometric reflection symmetry.
    """
    bars = _extract_bars(grid)
    return _transform_from_bars(np.asarray(grid, dtype=int), bars)


def _row_groups(height: int) -> list[tuple[int, int]]:
    mid = height // 2
    return [(r, height - 1 - r) for r in range(1, mid, 2)]


def _feasible_heights(n_bars: int) -> list[int]:
    out = []
    for height in HEIGHT_OPTIONS:
        max_bars = 2 * len(_row_groups(height)) - 2
        if max_bars >= n_bars:
            out.append(height)
    return out


def _feasible_unmatched_counts(n_bars: int, group_count: int) -> list[int]:
    choices = []
    for unmatched in range(2, n_bars - 3):
        if (n_bars - unmatched) % 2 != 0:
            continue
        pair_count = (n_bars - unmatched) // 2
        if pair_count < 2:
            continue
        if pair_count + unmatched > group_count:
            continue
        choices.append(unmatched)
    return choices


def _sample_rows(
    rng: np.random.Generator, n_bars: int
) -> tuple[int, list[int], list[tuple[int, int]], set[int]]:
    heights = _feasible_heights(n_bars)
    if not heights:
        raise RuntimeError("no feasible grid height")

    for _ in range(GENERATION_RETRIES):
        height = int(rng.choice(heights))
        groups = _row_groups(height)
        unmatched_choices = _feasible_unmatched_counts(n_bars, len(groups))
        unmatched = int(rng.choice(unmatched_choices))
        pair_count = (n_bars - unmatched) // 2

        order = rng.permutation(len(groups))
        paired_ids = order[:pair_count]
        single_ids = order[pair_count : pair_count + unmatched]

        paired_groups = [groups[int(idx)] for idx in paired_ids]
        paired_rows = {row for pair in paired_groups for row in pair}

        rows = []
        for top_row, bottom_row in paired_groups:
            rows.extend([top_row, bottom_row])
        for idx in single_ids:
            top_row, bottom_row = groups[int(idx)]
            rows.append(top_row if int(rng.integers(0, 2)) == 0 else bottom_row)
        rows.sort()

        selected_rows = {rows[i] for i in range(0, len(rows), 3)}
        categories = {
            (rows[i] in selected_rows, rows[i] in paired_rows) for i in range(len(rows))
        }
        mixed_pair = any(
            (top_row in selected_rows) != (bottom_row in selected_rows)
            for top_row, bottom_row in paired_groups
        )
        if categories == {(True, True), (True, False), (False, True), (False, False)} and mixed_pair:
            return height, rows, paired_groups, paired_rows

    raise RuntimeError("failed to sample a non-degenerate row layout")


def _random_span(rng: np.random.Generator, width: int) -> tuple[int, int]:
    max_bar_width = min(9, width - 2)
    bar_width = int(rng.integers(3, max_bar_width + 1))
    c0 = int(rng.integers(1, width - bar_width))
    return c0, c0 + bar_width - 1


def _sample_spans(
    rng: np.random.Generator,
    width: int,
    rows: list[int],
    paired_groups: list[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    paired_lookup = {}
    for top_row, bottom_row in paired_groups:
        paired_lookup[top_row] = bottom_row
        paired_lookup[bottom_row] = top_row

    for _ in range(GENERATION_RETRIES):
        row_to_span: dict[int, tuple[int, int]] = {}
        used_spans: set[tuple[int, int]] = set()

        for top_row, bottom_row in paired_groups:
            span = _random_span(rng, width)
            row_to_span[top_row] = span
            row_to_span[bottom_row] = span
            used_spans.add(span)

        for row in rows:
            if row in row_to_span:
                continue
            span = _random_span(rng, width)
            row_to_span[row] = span
            used_spans.add(span)

        widths = {c1 - c0 + 1 for c0, c1 in row_to_span.values()}
        lefts = {c0 for c0, _ in row_to_span.values()}
        if len(used_spans) >= 3 and len(widths) >= 2 and len(lefts) >= 2:
            return row_to_span

    raise RuntimeError("failed to sample diverse bar spans")


def _build_input(height: int, width: int, row_to_span: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for row, (c0, c1) in row_to_span.items():
        grid[int(row), int(c0) : int(c1) + 1] = BAR
    return grid


def _expected_from_layout(grid: np.ndarray, rows: list[int], row_to_span: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    paired_rows = {
        row
        for row in rows
        if row_to_span.get(grid.shape[0] - 1 - row) == row_to_span[row]
        and grid.shape[0] - 1 - row != row
    }
    for i, row in enumerate(rows):
        c0, c1 = row_to_span[row]
        out[row, c0 : c1 + 1] = _bar_color(i % 3 == 0, row in paired_rows)
    return out


def _validate_generated_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bars = _extract_bars(inp)
    except Exception:
        return False

    if not (BAR_COUNT_RANGE[0] <= len(bars) <= BAR_COUNT_RANGE[1]):
        return False

    symmetric = _symmetry_flags(bars, inp.shape[0])
    pair_count = sum(symmetric) // 2
    unmatched_count = len(bars) - sum(symmetric)
    if pair_count < 2 or unmatched_count < 2:
        return False

    selected = [i % 3 == 0 for i in range(len(bars))]
    if sum(selected) < 3:
        return False

    categories = {(is_selected, is_symmetric) for is_selected, is_symmetric in zip(selected, symmetric)}
    if categories != {(True, True), (True, False), (False, True), (False, False)}:
        return False

    if np.array_equal(expected, _selection_only_output(inp, bars)):
        return False
    if np.array_equal(expected, _symmetry_only_output(inp, bars)):
        return False

    mixed_pair = False
    span_by_row = {int(bar["row"]): (int(bar["c0"]), int(bar["c1"])) for bar in bars}
    selected_by_row = {int(bar["row"]): selected[i] for i, bar in enumerate(bars)}
    for row, is_symmetric in zip(span_by_row, symmetric):
        mirror_row = inp.shape[0] - 1 - row
        if not is_symmetric or row > mirror_row:
            continue
        if selected_by_row[row] != selected_by_row[mirror_row]:
            mixed_pair = True
            break
    if not mixed_pair:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    return set(map(int, np.unique(expected))) == {BG, BAR, RED, GREEN, YELLOW}


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GENERATION_RETRIES):
        n_bars = int(rng.integers(BAR_COUNT_RANGE[0], BAR_COUNT_RANGE[1] + 1))
        width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))

        try:
            height, rows, paired_groups, _paired_rows = _sample_rows(rng, n_bars)
            row_to_span = _sample_spans(rng, width, rows, paired_groups)
        except RuntimeError:
            continue

        inp = _build_input(height, width, row_to_span)
        expected = _expected_from_layout(inp, rows, row_to_span)
        if _validate_generated_instance(inp, expected):
            return inp

    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = spec_from_file_location("_arc_utils_for_032_3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color


SOURCE_TO_FILL = {
    Color.BLUE: Color.PINK,
    Color.RED: Color.ORANGE,
    Color.GREEN: Color.TEAL,
    Color.YELLOW: Color.MAROON,
}
SOURCE_COLORS = tuple(SOURCE_TO_FILL)

WINDOW_COUNT_RANGE = (3, 6)
BORDER_CHOICES = (False, True)
BRANCHES = ("rows", "cols", "both")
BRANCH_WEIGHTS = (2, 2, 4)
GEN_RETRIES = 300


def _build_axis_layout(window_count: int, use_border: bool) -> dict[str, object]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    internal_separators: list[int] = []
    pos = 0

    if use_border:
        separators.append(pos)
        pos += 1

    for idx in range(window_count):
        bands.append((pos, pos + 2))
        pos += 3
        if idx < window_count - 1:
            internal_separators.append(pos)
            separators.append(pos)
            pos += 1
        elif use_border:
            separators.append(pos)
            pos += 1

    non_separator = [coord for start, end in bands for coord in range(start, end + 1)]
    return {
        "size": pos,
        "bands": bands,
        "separator_indices": separators,
        "internal_separators": internal_separators,
        "non_separator_indices": non_separator,
    }


def _complement_bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separator_set = set(int(v) for v in separators)
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in separator_set:
            if start is not None:
                bands.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        bands.append((start, length - 1))
    return bands


def _parse_structure(grid: np.ndarray) -> dict[str, object]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.isin(grid[r], [Color.GREY, Color.TEAL]).all()]
    sep_cols = [c for c in range(w) if np.isin(grid[:, c], [Color.GREY, Color.TEAL]).all()]

    row_bands = _complement_bands(h, sep_rows)
    col_bands = _complement_bands(w, sep_cols)
    assert row_bands and col_bands
    assert all(r1 - r0 + 1 == 3 for r0, r1 in row_bands)
    assert all(c1 - c0 + 1 == 3 for c0, c1 in col_bands)

    marked_rows: list[int] = []
    marked_cols: list[int] = []
    for r in sep_rows:
        teal_count = int(np.count_nonzero(grid[r] == Color.TEAL))
        assert teal_count in (0, 1)
        if teal_count == 1:
            marked_rows.append(r)
    for c in sep_cols:
        teal_count = int(np.count_nonzero(grid[:, c] == Color.TEAL))
        assert teal_count in (0, 1)
        if teal_count == 1:
            marked_cols.append(c)

    return {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "marked_rows": marked_rows,
        "marked_cols": marked_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each 3x3 window from its center marker, then delete the separator lines tagged by teal pixels.

    Input structure:
    - Grey separator rows and columns partition the grid into 3x3 windows; grey outer border lines may or may not be present.
    - Each 3x3 window has exactly one center marker in blue, red, green, or yellow, with black elsewhere in the window.
    - Some internal separator rows or columns contain exactly one teal pixel replacing grey; that teal pixel marks the whole line for removal.

    Rule:
    - Map blue, red, green, yellow center markers to magenta, orange, teal, and maroon respectively.
    - Replace every 3x3 window by a solid block of its mapped color while keeping every unmarked separator line grey.
    - Delete every marked separator row and marked separator column, compressing the filled blocks together.

    Key insight:
    - The teal pixel is a tag on a separator line, so window colors must be read before compression; after the tagged lines are removed, same-colored neighboring windows merge seamlessly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    structure = _parse_structure(g)
    row_bands = structure["row_bands"]
    col_bands = structure["col_bands"]
    marked_rows = set(structure["marked_rows"])
    marked_cols = set(structure["marked_cols"])

    filled = np.full(g.shape, Color.GREY, dtype=int)
    for r0, r1 in row_bands:
        center_r = r0 + 1
        for c0, c1 in col_bands:
            center_c = c0 + 1
            marker = int(g[center_r, center_c])
            assert marker in SOURCE_TO_FILL
            filled[r0 : r1 + 1, c0 : c1 + 1] = SOURCE_TO_FILL[marker]

    keep_rows = [r for r in range(g.shape[0]) if r not in marked_rows]
    keep_cols = [c for c in range(g.shape[1]) if c not in marked_cols]
    return filled[np.ix_(keep_rows, keep_cols)]


def _max_mark_count(window_count: int) -> int:
    return max(1, min(2, window_count - 2))


def _choose_marked_gaps(row_count: int, col_count: int) -> tuple[list[int], list[int]]:
    branch = random.choices(BRANCHES, weights=BRANCH_WEIGHTS, k=1)[0]
    row_gap_ids = list(range(row_count - 1))
    col_gap_ids = list(range(col_count - 1))

    marked_row_gaps: list[int] = []
    marked_col_gaps: list[int] = []

    if branch in {"rows", "both"}:
        take = random.randint(1, _max_mark_count(row_count))
        marked_row_gaps = sorted(random.sample(row_gap_ids, take))
    if branch in {"cols", "both"}:
        take = random.randint(1, _max_mark_count(col_count))
        marked_col_gaps = sorted(random.sample(col_gap_ids, take))
    return marked_row_gaps, marked_col_gaps


def _groups_from_marked(window_count: int, marked_gaps: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    start = 0
    marked = set(marked_gaps)
    for gap in range(window_count - 1):
        if gap not in marked:
            groups.append(list(range(start, gap + 1)))
            start = gap + 1
    groups.append(list(range(start, window_count)))
    return groups


def _assign_window_markers(
    row_count: int,
    col_count: int,
    row_groups: list[list[int]],
    col_groups: list[list[int]],
) -> np.ndarray:
    markers = np.zeros((row_count, col_count), dtype=int)
    component_positions = [(ri, ci) for ri in range(len(row_groups)) for ci in range(len(col_groups))]
    random.shuffle(component_positions)

    component_colors: dict[tuple[int, int], int] = {}
    for idx, pos in enumerate(component_positions):
        if idx < len(SOURCE_COLORS):
            component_colors[pos] = SOURCE_COLORS[idx]
        else:
            component_colors[pos] = random.choice(SOURCE_COLORS)

    for ri, rows in enumerate(row_groups):
        for ci, cols in enumerate(col_groups):
            color = component_colors[(ri, ci)]
            for r in rows:
                for c in cols:
                    markers[r, c] = color

    assert set(np.unique(markers)) == set(SOURCE_COLORS)
    return markers


def _render_input(
    markers: np.ndarray,
    row_layout: dict[str, object],
    col_layout: dict[str, object],
    marked_row_gaps: list[int],
    marked_col_gaps: list[int],
) -> np.ndarray:
    h = int(row_layout["size"])
    w = int(col_layout["size"])
    grid = np.full((h, w), Color.BLACK, dtype=int)

    row_bands = row_layout["bands"]
    col_bands = col_layout["bands"]
    sep_rows = row_layout["separator_indices"]
    sep_cols = col_layout["separator_indices"]
    grid[sep_rows, :] = Color.GREY
    grid[:, sep_cols] = Color.GREY

    for r_idx, (r0, _r1) in enumerate(row_bands):
        for c_idx, (c0, _c1) in enumerate(col_bands):
            grid[r0 + 1, c0 + 1] = int(markers[r_idx, c_idx])

    available_marker_cols = list(col_layout["non_separator_indices"])
    available_marker_rows = list(row_layout["non_separator_indices"])
    internal_row_seps = row_layout["internal_separators"]
    internal_col_seps = col_layout["internal_separators"]

    for gap in marked_row_gaps:
        marker_col = random.choice(available_marker_cols)
        grid[int(internal_row_seps[gap]), marker_col] = Color.TEAL
    for gap in marked_col_gaps:
        marker_row = random.choice(available_marker_rows)
        grid[marker_row, int(internal_col_seps[gap])] = Color.TEAL

    return grid


def _render_expected_output(
    markers: np.ndarray,
    row_layout: dict[str, object],
    col_layout: dict[str, object],
    marked_row_gaps: list[int],
    marked_col_gaps: list[int],
) -> np.ndarray:
    h = int(row_layout["size"])
    w = int(col_layout["size"])
    filled = np.full((h, w), Color.GREY, dtype=int)

    for r_idx, (r0, r1) in enumerate(row_layout["bands"]):
        for c_idx, (c0, c1) in enumerate(col_layout["bands"]):
            filled[r0 : r1 + 1, c0 : c1 + 1] = SOURCE_TO_FILL[int(markers[r_idx, c_idx])]

    drop_rows = {int(row_layout["internal_separators"][gap]) for gap in marked_row_gaps}
    drop_cols = {int(col_layout["internal_separators"][gap]) for gap in marked_col_gaps}
    keep_rows = [r for r in range(h) if r not in drop_rows]
    keep_cols = [c for c in range(w) if c not in drop_cols]
    return filled[np.ix_(keep_rows, keep_cols)]


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*WINDOW_COUNT_RANGE)
    col_count = random.randint(*WINDOW_COUNT_RANGE)
    row_border = random.choice(BORDER_CHOICES)
    col_border = random.choice(BORDER_CHOICES)

    row_layout = _build_axis_layout(row_count, row_border)
    col_layout = _build_axis_layout(col_count, col_border)
    if max(int(row_layout["size"]), int(col_layout["size"])) > 30:
        return None

    marked_row_gaps, marked_col_gaps = _choose_marked_gaps(row_count, col_count)
    row_groups = _groups_from_marked(row_count, marked_row_gaps)
    col_groups = _groups_from_marked(col_count, marked_col_gaps)
    if len(row_groups) * len(col_groups) < len(SOURCE_COLORS):
        return None

    markers = _assign_window_markers(row_count, col_count, row_groups, col_groups)
    grid = _render_input(markers, row_layout, col_layout, marked_row_gaps, marked_col_gaps)
    expected = _render_expected_output(markers, row_layout, col_layout, marked_row_gaps, marked_col_gaps)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if actual.shape == grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid separator-compression input")

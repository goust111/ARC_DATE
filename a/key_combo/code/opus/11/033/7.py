from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MARKER_COLORS = (BLUE, RED)
GRID_PANEL_SIZES = (2, 3)
SEPARATOR_COUNT_RANGE = (4, 6)
MARKER_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 800


def _separator_axes(grid: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = list(map(int, np.flatnonzero(np.all(grid != BLACK, axis=1))))
    sep_cols = list(map(int, np.flatnonzero(np.all(grid != BLACK, axis=0))))
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "expected a separator lattice"
    return sep_rows, sep_cols


def _validate_lattice(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> None:
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    for r in range(grid.shape[0]):
        if r in sep_row_set:
            assert np.all(grid[r, :] != BLACK), "separator rows must span the full width"
        else:
            nonzero_cols = {int(c) for c in np.flatnonzero(grid[r, :] != BLACK)}
            assert nonzero_cols.issubset(sep_col_set), "non-separator rows may only contain separator columns"

    for c in range(grid.shape[1]):
        if c in sep_col_set:
            assert np.all(grid[:, c] != BLACK), "separator columns must span the full height"
        else:
            nonzero_rows = {int(r) for r in np.flatnonzero(grid[:, c] != BLACK)}
            assert nonzero_rows.issubset(sep_row_set), "non-separator columns may only contain separator rows"


def _extract_markers(
    grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]
) -> list[dict[str, int]]:
    row_to_idx = {r: i for i, r in enumerate(sep_rows)}
    col_to_idx = {c: i for i, c in enumerate(sep_cols)}
    markers: list[dict[str, int]] = []
    for r in sep_rows:
        for c in sep_cols:
            color = int(grid[r, c])
            assert color in (GREEN, *MARKER_COLORS), "intersections may only contain line or marker colors"
            if color != GREEN:
                markers.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "color": color,
                        "ri": row_to_idx[int(r)],
                        "ci": col_to_idx[int(c)],
                    }
                )
    assert 2 <= len(markers) <= 3, "expected two or three markers"
    assert len({marker["ri"] for marker in markers}) == len(markers), "markers must occupy distinct separator rows"
    assert len({marker["ci"] for marker in markers}) == len(markers), "markers must occupy distinct separator columns"
    return markers


def _row_interval_cells(row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    lo, hi = sorted((int(c0), int(c1)))
    return [(int(row), c) for c in range(lo + 1, hi)]


def _col_interval_cells(col: int, r0: int, r1: int) -> list[tuple[int, int]]:
    lo, hi = sorted((int(r0), int(r1)))
    return [(r, int(col)) for r in range(lo + 1, hi)]


def _collect_propagation(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int], list[dict[str, int]], dict[tuple[int, int], set[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    sep_rows, sep_cols = _separator_axes(g)
    _validate_lattice(g, sep_rows, sep_cols)
    markers = _extract_markers(g, sep_rows, sep_cols)

    marker_cells = {(marker["r"], marker["c"]) for marker in markers}
    paint_sources: dict[tuple[int, int], set[int]] = {}

    def add_cells(marker_id: int, cells: list[tuple[int, int]]) -> None:
        for cell in cells:
            if cell in marker_cells:
                continue
            paint_sources.setdefault(cell, set()).add(marker_id)

    for marker_id, marker in enumerate(markers):
        r, c = marker["r"], marker["c"]
        ri, ci = marker["ri"], marker["ci"]

        for dist in range(1, ci + 1):
            if dist % 2 != 0:
                continue
            near_idx = ci - dist + 1
            far_idx = ci - dist
            add_cells(marker_id, _row_interval_cells(r, sep_cols[near_idx], sep_cols[far_idx]))
            add_cells(marker_id, [(r, sep_cols[far_idx])])

        for dist in range(1, len(sep_cols) - ci):
            if dist % 2 != 0:
                continue
            near_idx = ci + dist - 1
            far_idx = ci + dist
            add_cells(marker_id, _row_interval_cells(r, sep_cols[near_idx], sep_cols[far_idx]))
            add_cells(marker_id, [(r, sep_cols[far_idx])])

        for dist in range(1, ri + 1):
            if dist % 2 != 0:
                continue
            near_idx = ri - dist + 1
            far_idx = ri - dist
            add_cells(marker_id, _col_interval_cells(c, sep_rows[near_idx], sep_rows[far_idx]))
            add_cells(marker_id, [(sep_rows[far_idx], c)])

        for dist in range(1, len(sep_rows) - ri):
            if dist % 2 != 0:
                continue
            near_idx = ri + dist - 1
            far_idx = ri + dist
            add_cells(marker_id, _col_interval_cells(c, sep_rows[near_idx], sep_rows[far_idx]))
            add_cells(marker_id, [(sep_rows[far_idx], c)])

    return g, sep_rows, sep_cols, markers, paint_sources


def _render_output(grid: np.ndarray, *, collisions: bool = True, segments: bool = True) -> np.ndarray:
    g, sep_rows, sep_cols, markers, paint_sources = _collect_propagation(grid)
    marker_cells = {(marker["r"], marker["c"]) for marker in markers}
    out = g.copy()

    if not segments:
        filtered: dict[tuple[int, int], set[int]] = {}
        sep_row_set = set(sep_rows)
        sep_col_set = set(sep_cols)
        for cell, ids in paint_sources.items():
            if cell[0] in sep_row_set and cell[1] in sep_col_set:
                filtered[cell] = set(ids)
        paint_sources = filtered

    for cell, ids in paint_sources.items():
        if cell in marker_cells:
            continue
        if collisions and len(ids) >= 2:
            out[cell] = YELLOW
        else:
            first_id = min(ids)
            out[cell] = int(markers[first_id]["color"])

    return out


def _pair_collision_count(markers: list[dict[str, int]]) -> int:
    count = 0
    for a, b in combinations(markers, 2):
        if abs(a["ri"] - b["ri"]) % 2 == 0 and abs(a["ci"] - b["ci"]) % 2 == 0:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each lattice marker periodically along its separator row and column, recoloring overlap cells yellow.

    Input structure:
    - The grid is a black background overlaid with single-pixel green separator rows and columns that form a regular lattice.
    - Two or three separator intersections are replaced by red or blue marker pixels, with no two markers sharing a separator row or separator column.
    - Yellow does not appear in the input, and non-separator panel interiors remain black.

    Rule:
    - Keep the grid size, the black panel interiors, and every original marker pixel unchanged.
    - For each marker, propagate its color independently along its own separator row and separator column in both directions.
    - Count distance in separator intersections from the marker: skip odd-distance intersections, but paint every even-distance intersection.
    - For each painted intersection, also paint the separator segment immediately before it along that direction; skipped intersections and skipped segments stay green.
    - Whenever two different markers would paint the same non-marker separator cell, recolor that cell yellow instead of either marker color.

    Key insight:
    - The lattice itself defines the coordinate system: propagation is periodic in intersection steps rather than pixel distance, so overlap depends on parity across the separator graph, not on simple nearest-neighbor spreading.
    """
    return _render_output(grid, collisions=True, segments=True)


def _sample_lattice_geometry() -> tuple[int, int, list[int], list[int]]:
    panel_size = random.choice(GRID_PANEL_SIZES)
    step = panel_size + 1
    row_count = random.randint(*SEPARATOR_COUNT_RANGE)
    col_count = random.randint(*SEPARATOR_COUNT_RANGE)
    tail_rows = random.randint(0, panel_size)
    tail_cols = random.randint(0, panel_size)

    height = row_count * step + tail_rows
    width = col_count * step + tail_cols
    sep_rows = [panel_size + i * step for i in range(row_count)]
    sep_cols = [panel_size + i * step for i in range(col_count)]
    return height, width, sep_rows, sep_cols


def _build_lattice(height: int, width: int, sep_rows: list[int], sep_cols: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[sep_rows, :] = GREEN
    grid[:, sep_cols] = GREEN
    return grid


def _assign_marker_colors(marker_count: int) -> list[int]:
    if marker_count == 2:
        colors = [BLUE, RED]
        random.shuffle(colors)
        return colors
    patterns = ([BLUE, BLUE, RED], [BLUE, RED, RED], [BLUE, RED, BLUE], [RED, BLUE, RED])
    return list(random.choice(patterns))


def _profile_matches(pair_count: int, marker_count: int, profile: str) -> bool:
    if marker_count == 2:
        return (profile == "collision" and pair_count == 1) or (profile == "no_collision" and pair_count == 0)
    if profile == "mixed":
        return 1 <= pair_count <= 2
    if profile == "none":
        return pair_count == 0
    if profile == "all":
        return pair_count == 3
    return False


def _sample_marker_layout(row_count: int, col_count: int, marker_count: int, profile: str) -> list[tuple[int, int]] | None:
    for _ in range(300):
        row_indices = sorted(random.sample(range(row_count), marker_count))
        col_indices = sorted(random.sample(range(col_count), marker_count))
        order = list(range(marker_count))
        random.shuffle(order)
        layout = [(row_indices[i], col_indices[order[i]]) for i in range(marker_count)]
        fake_markers = [
            {"ri": ri, "ci": ci, "r": ri, "c": ci, "color": BLUE}
            for ri, ci in layout
        ]
        if _profile_matches(_pair_collision_count(fake_markers), marker_count, profile):
            return layout
    return None


def _expected_collision_cells(markers: list[dict[str, int]], sep_rows: list[int], sep_cols: list[int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in combinations(markers, 2):
        if abs(a["ri"] - b["ri"]) % 2 == 0 and abs(a["ci"] - b["ci"]) % 2 == 0:
            cells.add((a["r"], b["c"]))
            cells.add((b["r"], a["c"]))
    return cells


def _full_propagation_output(grid: np.ndarray) -> np.ndarray:
    g, _sep_rows, _sep_cols, markers, _paint_sources = _collect_propagation(grid)
    out = g.copy()
    marker_cells = {(marker["r"], marker["c"]) for marker in markers}
    covered_by: dict[tuple[int, int], set[int]] = {}
    for marker_id, marker in enumerate(markers):
        for c in range(g.shape[1]):
            cell = (marker["r"], c)
            if cell not in marker_cells and g[cell] != BLACK:
                covered_by.setdefault(cell, set()).add(marker_id)
        for r in range(g.shape[0]):
            cell = (r, marker["c"])
            if cell not in marker_cells and g[cell] != BLACK:
                covered_by.setdefault(cell, set()).add(marker_id)
    for cell, ids in covered_by.items():
        out[cell] = YELLOW if len(ids) >= 2 else int(markers[min(ids)]["color"])
    return out


def _validate_generated_case(grid: np.ndarray) -> None:
    g, sep_rows, sep_cols, markers, paint_sources = _collect_propagation(grid)
    out = main(g)

    assert g.shape == out.shape, "transform must preserve the grid shape"
    assert max(g.shape) <= 30 and max(out.shape) <= 30, "ARC grids must stay within 30x30"
    assert np.any(out != g), "output must not be identical to the input"
    assert np.all(out[(g == BLACK)] == BLACK), "panel interiors must stay black"

    marker_cells = {(marker["r"], marker["c"]) for marker in markers}
    for marker in markers:
        assert out[marker["r"], marker["c"]] == marker["color"], "markers must preserve their own colors"

    expected_collision_cells = _expected_collision_cells(markers, sep_rows, sep_cols)
    actual_collision_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == YELLOW)}
    assert actual_collision_cells == expected_collision_cells, "yellow cells must match even-parity pair collisions exactly"

    assert any(cell[0] not in sep_rows or cell[1] not in sep_cols for cell in paint_sources), "sample must include painted line segments"
    assert not np.array_equal(out, _render_output(g, collisions=True, segments=False)), "segment painting must matter"
    assert not np.array_equal(out, _full_propagation_output(g)), "periodicity must matter"

    if expected_collision_cells:
        assert not np.array_equal(out, _render_output(g, collisions=False, segments=True)), "collision recoloring must matter when collisions exist"

    assert np.any(out == GREEN), "some separator cells should remain green to show the skip phase"
    assert len({marker["color"] for marker in markers}) == 2, "both marker colors should appear in every generated input"

    swap = g.copy()
    for marker in markers:
        swap[marker["r"], marker["c"]] = BLUE if marker["color"] == RED else RED
    assert not np.array_equal(main(swap), out), "marker colors must affect the output"

    pair_count = _pair_collision_count(markers)
    if len(markers) == 2:
        assert pair_count in (0, 1)
    else:
        assert pair_count in (0, 1, 2, 3)


def _gen_attempt() -> np.ndarray | None:
    height, width, sep_rows, sep_cols = _sample_lattice_geometry()
    marker_count = random.choice(MARKER_COUNT_CHOICES)

    if marker_count == 2:
        profile = random.choice(("collision", "no_collision"))
    else:
        profile = random.choices(("mixed", "none", "all"), weights=(5, 3, 2), k=1)[0]

    layout = _sample_marker_layout(len(sep_rows), len(sep_cols), marker_count, profile)
    if layout is None:
        return None

    grid = _build_lattice(height, width, sep_rows, sep_cols)
    colors = _assign_marker_colors(marker_count)

    if random.random() < 0.35:
        order = list(range(marker_count))
        random.shuffle(order)
        layout = [layout[i] for i in order]

    for color, (ri, ci) in zip(colors, layout):
        grid[sep_rows[ri], sep_cols[ci]] = int(color)

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-lattice collision input")

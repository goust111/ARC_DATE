from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def _paint_between_positions(out: np.ndarray, g: np.ndarray, fixed: int, positions: list[int], axis: int) -> int:
    assert axis in (0, 1), "axis must select row or column painting"
    painted = 0
    for a, b in zip(positions, positions[1:]):
        assert a < b, "marker positions must be strictly ordered"
        if b == a + 1:
            continue
        if axis == 0:
            segment = g[fixed, a + 1 : b]
            assert np.all(segment == Color.BLACK), "row marker interval must contain only background"
            out[fixed, a + 1 : b] = Color.TEAL
        else:
            segment = g[a + 1 : b, fixed]
            assert np.all(segment == Color.BLACK), "column marker interval must contain only background"
            out[a + 1 : b, fixed] = Color.TEAL
        painted += b - a - 1
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every horizontally or vertically aligned pair of blue marker pixels with teal segments.

    Input structure
    - A 2D black-background grid.
    - The only foreground pixels are blue markers.
    - Some rows and columns contain two or more blue markers.

    Rule
    - Preserve every input pixel.
    - In each row, sort the blue marker columns and paint teal on the background cells strictly between adjacent markers.
    - In each column, sort the blue marker rows and paint teal on the background cells strictly between adjacent markers.

    Key insight
    - The markers are endpoints for straight horizontal and vertical connections; unrelated markers are ignored unless they share the same row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert colors_present(g, exclude={bg}) == [Color.BLUE], "expected only blue foreground markers"

    out = g.copy()
    marker_coords = np.argwhere(g == Color.BLUE)
    assert marker_coords.shape[0] >= 2, "expected at least two blue markers"

    painted = 0
    height, width = map(int, g.shape)
    for r in range(height):
        cols = sorted(int(c) for c in np.flatnonzero(g[r, :] == Color.BLUE))
        painted += _paint_between_positions(out, g, r, cols, axis=0)
    for c in range(width):
        rows = sorted(int(r) for r in np.flatnonzero(g[:, c] == Color.BLUE))
        painted += _paint_between_positions(out, g, c, rows, axis=1)

    assert painted > 0, "expected at least one aligned marker interval"
    assert np.array_equal(out[g == Color.BLUE], g[g == Color.BLUE]), "blue markers must be preserved"
    return out

# --------------- generator (o5x) ---------------

# Canvas and marker counts.
GRID_HEIGHT_RANGE = (8, 19)
GRID_WIDTH_RANGE = (9, 21)
MARKER_COUNT_MIN = 7
MARKER_COUNT_MAX = 19
MARKER_DENSITY_LIMIT = 0.13

# Connection structure.
LINE_COUNT_MIN = 1
LINE_COUNT_MAX = 5
TOTAL_SEGMENTS_MIN = 3
TOTAL_SEGMENTS_MAX = 12
SEGMENTS_PER_LINE_MIN = 1
SEGMENTS_PER_LINE_MAX = 3
SEGMENT_GAP_MIN = 1
SEGMENT_GAP_MAX = 13
MARKERS_PER_LINE_MAX = 4
TEAL_CROSSES_MAX = 1

# Layout shape.
ISOLATED_MARKER_MIN = 1
ISOLATED_MARKER_MAX = 4
LAYOUT_WINDOW_MAX = 15

# Retry budgets.
ACTIVE_EDGE_RETRIES = 80
PLAN_RETRIES = 250
GRID_BUILD_RETRIES = 80
GENERATE_RETRIES = 300

AXIS_ROWS = 0
AXIS_COLS = 1


def _gen_marker_lines(grid: np.ndarray, axis: int):
    """Yield aligned marker positions for rows or columns with at least two markers."""
    limit = grid.shape[0] if axis == AXIS_ROWS else grid.shape[1]
    for fixed in range(limit):
        if axis == AXIS_ROWS:
            positions = [int(c) for c in np.flatnonzero(grid[fixed] == Color.BLUE)]
        else:
            positions = [int(r) for r in np.flatnonzero(grid[:, fixed] == Color.BLUE)]
        if len(positions) >= 2:
            yield fixed, positions


def _gen_line_summary(grid: np.ndarray) -> dict | None:
    """Summarize generated connections, rejecting segment gaps outside the spec."""
    summary = {"rows": 0, "cols": 0, "segments": 0, "max_markers_per_line": 0}
    row_cells = set()
    col_cells = set()

    for axis, line_key, cells in ((AXIS_ROWS, "rows", row_cells), (AXIS_COLS, "cols", col_cells)):
        for fixed, positions in _gen_marker_lines(grid, axis):
            summary[line_key] += 1
            summary["max_markers_per_line"] = max(summary["max_markers_per_line"], len(positions))
            for a, b in zip(positions, positions[1:]):
                gap = b - a - 1
                if gap < SEGMENT_GAP_MIN or gap > SEGMENT_GAP_MAX:
                    return None
                summary["segments"] += 1
                if axis == AXIS_ROWS:
                    cells.update((fixed, c) for c in range(a + 1, b))
                else:
                    cells.update((r, fixed) for r in range(a + 1, b))

    summary["teal_crosses"] = len(row_cells & col_cells)
    return summary


def _gen_render_expected(grid: np.ndarray) -> np.ndarray:
    """Render the expected output for solver-equivalence validation."""
    out = grid.copy()
    for axis in (AXIS_ROWS, AXIS_COLS):
        for fixed, positions in _gen_marker_lines(grid, axis):
            for a, b in zip(positions, positions[1:]):
                if axis == AXIS_ROWS:
                    out[fixed, a + 1 : b] = Color.TEAL
                else:
                    out[a + 1 : b, fixed] = Color.TEAL
    return out


def _gen_random_segment_counts(n_lines: int, total_segments: int) -> list[int] | None:
    """Split total segments across active lines within per-line limits."""
    if not SEGMENTS_PER_LINE_MIN * n_lines <= total_segments <= SEGMENTS_PER_LINE_MAX * n_lines:
        return None
    counts = [SEGMENTS_PER_LINE_MIN] * n_lines
    remaining = total_segments - sum(counts)
    order = list(range(n_lines))
    random.shuffle(order)
    while remaining:
        candidates = [i for i in order if counts[i] < SEGMENTS_PER_LINE_MAX]
        if not candidates:
            return None
        i = random.choice(candidates)
        add = random.randint(1, min(SEGMENTS_PER_LINE_MAX - counts[i], remaining))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _gen_active_edges(row_degrees: list[int], col_degrees: list[int], n_edges: int) -> list[tuple[int, int]] | None:
    """Choose shared row/column marker intersections for the requested degrees."""
    n_rows, n_cols = len(row_degrees), len(col_degrees)
    if n_edges == 0:
        return []
    if n_edges > n_rows * n_cols:
        return None

    for _ in range(ACTIVE_EDGE_RETRIES):
        row_rem = row_degrees[:]
        col_rem = col_degrees[:]
        used = set()
        edges = []
        for _ in range(n_edges):
            candidates = [
                (r, c)
                for r in range(n_rows)
                for c in range(n_cols)
                if row_rem[r] > 0 and col_rem[c] > 0 and (r, c) not in used
            ]
            if not candidates:
                break
            weights = [row_rem[r] * col_rem[c] for r, c in candidates]
            edge = random.choices(candidates, weights=weights, k=1)[0]
            r, c = edge
            used.add(edge)
            edges.append(edge)
            row_rem[r] -= 1
            col_rem[c] -= 1
        if len(edges) == n_edges:
            return edges
    return None


def _gen_active_counts(n_rows: int, n_cols: int, active_edges: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    active_in_row = [0] * n_rows
    active_in_col = [0] * n_cols
    for r, c in active_edges:
        active_in_row[r] += 1
        active_in_col[c] += 1
    return active_in_row, active_in_col


def _gen_choose_plan(h: int, w: int, n_markers: int) -> tuple[list[int], list[int], int, list[tuple[int, int]]] | None:
    """Choose line degrees, isolated markers, and shared intersections."""
    for _ in range(PLAN_RETRIES):
        n_rows = random.randint(LINE_COUNT_MIN, min(LINE_COUNT_MAX, h, n_markers // 2))
        n_cols = random.randint(LINE_COUNT_MIN, min(LINE_COUNT_MAX, w, n_markers // 2))
        n_lines = n_rows + n_cols
        min_segments = max(TOTAL_SEGMENTS_MIN, SEGMENTS_PER_LINE_MIN * n_lines)
        max_segments = min(TOTAL_SEGMENTS_MAX, SEGMENTS_PER_LINE_MAX * n_lines)
        if min_segments > max_segments:
            continue

        total_segments = random.randint(min_segments, max_segments)
        segment_counts = _gen_random_segment_counts(n_lines, total_segments)
        if segment_counts is None:
            continue
        row_degrees = [v + 1 for v in segment_counts[:n_rows]]
        col_degrees = [v + 1 for v in segment_counts[n_rows:]]
        degree_sum = sum(row_degrees) + sum(col_degrees)

        iso_options = list(range(ISOLATED_MARKER_MIN, min(ISOLATED_MARKER_MAX, n_markers) + 1))
        random.shuffle(iso_options)
        iso_options.append(0)
        for n_isolated in iso_options:
            n_active_edges = degree_sum + n_isolated - n_markers
            if n_active_edges < 0:
                continue
            if n_active_edges > min(sum(row_degrees), sum(col_degrees), n_rows * n_cols):
                continue

            active_edges = _gen_active_edges(row_degrees, col_degrees, n_active_edges)
            if active_edges is None:
                continue
            active_in_row, active_in_col = _gen_active_counts(n_rows, n_cols, active_edges)
            inactive_cols_needed = sum(d - a for d, a in zip(row_degrees, active_in_row))
            inactive_rows_needed = sum(d - a for d, a in zip(col_degrees, active_in_col))

            if inactive_cols_needed + n_isolated > w - n_cols:
                continue
            if inactive_rows_needed + n_isolated > h - n_rows:
                continue
            if n_cols + inactive_cols_needed > min(w, LAYOUT_WINDOW_MAX):
                continue
            if n_rows + inactive_rows_needed > min(h, LAYOUT_WINDOW_MAX):
                continue
            return row_degrees, col_degrees, n_isolated, active_edges
    return None


def _gen_build_grid(
    h: int,
    w: int,
    row_degrees: list[int],
    col_degrees: list[int],
    n_isolated: int,
    active_edges: list[tuple[int, int]],
) -> np.ndarray | None:
    """Place planned markers into a bounded row/column window."""
    n_rows, n_cols = len(row_degrees), len(col_degrees)
    active_in_row, active_in_col = _gen_active_counts(n_rows, n_cols, active_edges)
    extra_cols_by_row = [row_degrees[r] - active_in_row[r] for r in range(n_rows)]
    extra_rows_by_col = [col_degrees[c] - active_in_col[c] for c in range(n_cols)]
    row_node_count = n_rows + sum(extra_rows_by_col)
    col_node_count = n_cols + sum(extra_cols_by_row)
    row_window = min(h, LAYOUT_WINDOW_MAX)
    col_window = min(w, LAYOUT_WINDOW_MAX)
    if row_node_count > row_window or col_node_count > col_window:
        return None

    for _ in range(GRID_BUILD_RETRIES):
        r0 = random.randint(0, h - row_window)
        c0 = random.randint(0, w - col_window)
        row_positions = random.sample(range(r0, r0 + row_window), row_node_count)
        col_positions = random.sample(range(c0, c0 + col_window), col_node_count)
        random.shuffle(row_positions)
        random.shuffle(col_positions)

        markers = set()
        for r, c in active_edges:
            markers.add((row_positions[r], col_positions[c]))

        next_extra_col = n_cols
        for r, need in enumerate(extra_cols_by_row):
            for _ in range(need):
                markers.add((row_positions[r], col_positions[next_extra_col]))
                next_extra_col += 1

        next_extra_row = n_rows
        for c, need in enumerate(extra_rows_by_col):
            for _ in range(need):
                markers.add((row_positions[next_extra_row], col_positions[c]))
                next_extra_row += 1

        used_rows = {r for r, _ in markers}
        used_cols = {c for _, c in markers}
        free_rows = [r for r in range(h) if r not in used_rows]
        free_cols = [c for c in range(w) if c not in used_cols]
        if len(free_rows) < n_isolated or len(free_cols) < n_isolated:
            continue
        iso_rows = random.sample(free_rows, n_isolated)
        iso_cols = random.sample(free_cols, n_isolated)
        for r, c in zip(iso_rows, iso_cols):
            markers.add((r, c))

        if len(markers) != sum(row_degrees) + sum(col_degrees) - len(active_edges) + n_isolated:
            continue
        grid = np.zeros((h, w), dtype=int)
        for r, c in markers:
            grid[r, c] = Color.BLUE
        return grid
    return None


def _gen_choose_canvas() -> tuple[int, int, int] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    max_markers = min(MARKER_COUNT_MAX, int(h * w * MARKER_DENSITY_LIMIT))
    if max_markers < MARKER_COUNT_MIN:
        return None
    return h, w, random.randint(MARKER_COUNT_MIN, max_markers)


def _gen_summary_is_valid(summary: dict) -> bool:
    if not (LINE_COUNT_MIN <= summary["rows"] <= LINE_COUNT_MAX and LINE_COUNT_MIN <= summary["cols"] <= LINE_COUNT_MAX):
        return False
    if not (TOTAL_SEGMENTS_MIN <= summary["segments"] <= TOTAL_SEGMENTS_MAX):
        return False
    if summary["max_markers_per_line"] > MARKERS_PER_LINE_MAX:
        return False
    if summary["teal_crosses"] > TEAL_CROSSES_MAX:
        return False
    return True


def _gen_matches_solver(grid: np.ndarray) -> bool:
    expected = _gen_render_expected(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_grid_is_valid(grid: np.ndarray, n_markers: int) -> bool:
    """Apply final count, summary, and solver-equivalence checks."""
    if int(np.count_nonzero(grid == Color.BLUE)) != n_markers:
        return False
    summary = _gen_line_summary(grid)
    if summary is None or not _gen_summary_is_valid(summary):
        return False
    return _gen_matches_solver(grid)


def _gen_attempt() -> np.ndarray | None:
    canvas = _gen_choose_canvas()
    if canvas is None:
        return None
    h, w, n_markers = canvas

    plan = _gen_choose_plan(h, w, n_markers)
    if plan is None:
        return None
    row_degrees, col_degrees, n_isolated, active_edges = plan
    grid = _gen_build_grid(h, w, row_degrees, col_degrees, n_isolated, active_edges)
    if grid is None:
        return None
    if not _gen_grid_is_valid(grid, n_markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid dbc1a6ce input")

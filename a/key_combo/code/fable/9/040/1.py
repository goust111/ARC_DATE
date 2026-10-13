from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Run = dict[str, object]
SegmentSpec = dict[str, object]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, *, exclude: Iterable[int] = ()) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid, dtype=int).ravel())) - set(map(int, exclude)))
    return colors


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if len(cols) == 0:
        return runs
    start = prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
        else:
            runs.append((start, prev))
            start = prev = col
    runs.append((start, prev))
    return runs


def _nearest_side(col: int, left: int, right: int, width: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    assert candidates
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _classify_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    top_colors = _colors_present(g[0], exclude={bg})
    assert len(top_colors) == 1, "row 0 must contain exactly one source color"
    source = top_colors[0]

    others = _colors_present(g, exclude={bg, source})
    assert len(others) == 2, "expected obstacle color and fracture-marker color"
    counts = {color: int(np.count_nonzero(g == color)) for color in others}
    marker = min(counts, key=counts.get)
    obstacle = max(counts, key=counts.get)
    assert counts[marker] < counts[obstacle], "fracture markers must be rarer than obstacle cells"
    return bg, source, obstacle, marker


def _split_layout(
    grid: np.ndarray, bg: int, obstacle: int, marker: int
) -> tuple[np.ndarray, dict[int, list[Run]], dict[int, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    runs_by_row: dict[int, list[Run]] = {}
    gap_by_row: dict[int, tuple[int, int, int]] = {}

    for r in range(1, h):
        row_mask = (g[r] == obstacle) | (g[r] == marker)
        for left, right in _contiguous_runs(np.flatnonzero(row_mask)):
            markers = np.flatnonzero(g[r, left : right + 1] == marker) + left
            assert len(markers) <= 1, "each obstacle segment may contain at most one fracture marker"
            if len(markers) == 0:
                out[r, left : right + 1] = obstacle
                runs_by_row.setdefault(r, []).append(
                    {
                        "left": left,
                        "right": right,
                        "parent_marked": False,
                        "gap": None,
                    }
                )
                continue

            m = int(markers[0])
            assert left < m < right, "fracture marker must lie inside its obstacle segment"
            gap = (m - 1, m, m + 1)
            gap_by_row[r] = gap

            left_piece = (left - 1, m - 2)
            right_piece = (m + 2, right + 1)
            assert left_piece[0] >= 0 and right_piece[1] < w, "split obstacle must stay in bounds"

            if left_piece[0] <= left_piece[1]:
                out[r, left_piece[0] : left_piece[1] + 1] = obstacle
                runs_by_row.setdefault(r, []).append(
                    {
                        "left": left_piece[0],
                        "right": left_piece[1],
                        "parent_marked": True,
                        "gap": gap,
                    }
                )
            if right_piece[0] <= right_piece[1]:
                out[r, right_piece[0] : right_piece[1] + 1] = obstacle
                runs_by_row.setdefault(r, []).append(
                    {
                        "left": right_piece[0],
                        "right": right_piece[1],
                        "parent_marked": True,
                        "gap": gap,
                    }
                )

    for runs in runs_by_row.values():
        runs.sort(key=lambda run: int(run["left"]))
    return out, runs_by_row, gap_by_row


def _unsplit_layout(
    grid: np.ndarray, bg: int, obstacle: int, marker: int
) -> tuple[np.ndarray, dict[int, list[Run]], dict[int, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    runs_by_row: dict[int, list[Run]] = {}
    gap_by_row: dict[int, tuple[int, int, int]] = {}

    for r in range(1, h):
        row_mask = (g[r] == obstacle) | (g[r] == marker)
        for left, right in _contiguous_runs(np.flatnonzero(row_mask)):
            markers = np.flatnonzero(g[r, left : right + 1] == marker) + left
            assert len(markers) <= 1
            out[r, left : right + 1] = obstacle
            gap = None
            if len(markers) == 1:
                m = int(markers[0])
                gap = (m - 1, m, m + 1)
                gap_by_row[r] = gap
            runs_by_row.setdefault(r, []).append(
                {
                    "left": left,
                    "right": right,
                    "parent_marked": False,
                    "gap": gap,
                }
            )
    for runs in runs_by_row.values():
        runs.sort(key=lambda run: int(run["left"]))
    return out, runs_by_row, gap_by_row


def _run_containing(runs: list[Run], col: int) -> Run | None:
    for run in runs:
        if int(run["left"]) <= col <= int(run["right"]):
            return run
    return None


def _trace_source(
    height: int,
    width: int,
    runs_by_row: dict[int, list[Run]],
    gap_by_row: dict[int, tuple[int, int, int]],
    start_col: int,
) -> dict[str, object]:
    r = 0
    c = int(start_col)
    path: set[tuple[int, int]] = {(0, c)}
    detour_rows_marked: set[int] = set()
    detour_rows_unmarked: set[int] = set()
    direct_gap_rows: set[int] = set()
    inner_gap_arrivals: set[int] = set()

    while r < height - 1:
        blocker_row = r + 1
        run = _run_containing(runs_by_row.get(blocker_row, []), c)
        if run is None:
            r = blocker_row
            path.add((r, c))
            if blocker_row in gap_by_row and c in gap_by_row[blocker_row]:
                direct_gap_rows.add(blocker_row)
            continue

        left = int(run["left"])
        right = int(run["right"])
        side_col = _nearest_side(c, left, right, width)
        for cc in range(min(c, side_col), max(c, side_col) + 1):
            path.add((r, cc))

        if bool(run["parent_marked"]):
            detour_rows_marked.add(blocker_row)
            gap = run["gap"]
            if gap is not None and side_col in gap:
                inner_gap_arrivals.add(blocker_row)
        else:
            detour_rows_unmarked.add(blocker_row)

        c = side_col

    return {
        "start_col": int(start_col),
        "final_col": c,
        "path": path,
        "detour_rows_marked": detour_rows_marked,
        "detour_rows_unmarked": detour_rows_unmarked,
        "direct_gap_rows": direct_gap_rows,
        "inner_gap_arrivals": inner_gap_arrivals,
        "detour_count": len(detour_rows_marked) + len(detour_rows_unmarked),
    }


def _paint_paths(
    base: np.ndarray, source_color: int, obstacle_color: int, traces: list[dict[str, object]]
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    for trace in traces:
        for r, c in trace["path"]:
            out[r, c] = source_color
    out[base == obstacle_color] = obstacle_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Split each marked obstacle segment open, then drop the top source columns through the new layout.

    Input structure:
    - The background is the dominant color.
    - Row 0 contains several single-cell starts of one source color.
    - Lower rows contain horizontal obstacle segments in another color; some segments contain one fracture-marker pixel of a third color.

    Rule:
    - First erase every fracture marker and split that obstacle segment: the cells left of the marker shift one column left, the cells right of it shift one column right, and a three-cell gap opens at the former center. Unmarked obstacle segments stay fixed.
    - Then extend each source column downward to the bottom edge. Whenever the falling column would enter an obstacle segment, paint a horizontal detour on the row above that segment to the nearer outside side and continue falling from there.
    - If a falling column is aligned with a split gap, it passes straight through. Exact left/right distance ties detour to the right.

    Key insight:
    - The fracture markers are not decorations: they must be converted into moved obstacle geometry before any detour decisions are made, because the opened gap and the shifted segment ends change the available routes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, source, obstacle, marker = _classify_colors(g)
    split_grid, runs_by_row, gap_by_row = _split_layout(g, bg, obstacle, marker)
    source_cols = [int(c) for c in np.flatnonzero(g[0] == source)]
    assert source_cols, "expected at least one source column"

    traces = [_trace_source(g.shape[0], g.shape[1], runs_by_row, gap_by_row, col) for col in source_cols]
    return _paint_paths(split_grid, source, obstacle, traces)


# --------------- generator ---------------

_GEN_RETRIES = 400
_GEN_HEIGHT_RANGE = (11, 16)
_GEN_WIDTH_RANGE = (14, 20)
_GEN_SOURCE_COUNT_RANGE = (4, 6)
_GEN_COLOR_POOL = tuple(range(1, 10))


def _rand_marked_segment(width: int) -> SegmentSpec:
    length = random.randint(5, min(8, width - 4))
    left = random.randint(2, width - length - 3)
    right = left + length - 1
    marker_col = random.randint(left + 2, right - 2)
    return {
        "marked": True,
        "left": left,
        "right": right,
        "marker_col": marker_col,
    }


def _possible_lower_segments(width: int, target_col: int, forbidden_gap_col: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for length in range(2, min(6, width - 2) + 1):
        min_left = max(1, target_col - length + 1)
        max_left = min(target_col, width - length - 1)
        for left in range(min_left, max_left + 1):
            right = left + length - 1
            if not (left <= target_col <= right):
                continue
            if left <= forbidden_gap_col <= right:
                continue
            choices.append((left, right))
    return choices


def _rand_extra_segment(width: int) -> SegmentSpec:
    marked = random.random() < 0.45
    if marked:
        return _rand_marked_segment(width)

    length = random.randint(3, min(6, width - 2))
    left = random.randint(1, width - length - 2)
    return {
        "marked": False,
        "left": left,
        "right": left + length - 1,
        "marker_col": None,
    }


def _row_choices(height: int, used_rows: set[int]) -> list[int]:
    choices = []
    for row in range(3, height - 1):
        if all(abs(row - old) >= 2 for old in used_rows):
            choices.append(row)
    return choices


def _build_input_from_specs(
    height: int,
    width: int,
    source_color: int,
    obstacle_color: int,
    marker_color: int,
    specs: list[SegmentSpec],
    source_cols: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for col in source_cols:
        grid[0, col] = source_color
    for spec in specs:
        row = int(spec["row"])
        left = int(spec["left"])
        right = int(spec["right"])
        grid[row, left : right + 1] = obstacle_color
        if bool(spec["marked"]):
            grid[row, int(spec["marker_col"])] = marker_color
    return grid


def _trace_all_sources(
    grid: np.ndarray, source_color: int, obstacle_color: int, marker_color: int
) -> tuple[np.ndarray, dict[int, list[Run]], dict[int, tuple[int, int, int]], list[dict[str, object]]]:
    split_grid, runs_by_row, gap_by_row = _split_layout(grid, 0, obstacle_color, marker_color)
    traces = [
        _trace_source(grid.shape[0], grid.shape[1], runs_by_row, gap_by_row, col)
        for col in range(grid.shape[1])
    ]
    return split_grid, runs_by_row, gap_by_row, traces


def _choose_sources(
    candidates: list[dict[str, object]],
    target_count: int,
    marked_row: int,
    lower_row: int,
) -> list[dict[str, object]] | None:
    enriched: list[dict[str, object]] = []
    for cand in candidates:
        enriched.append(
            {
                **cand,
                "need_gap": marked_row in cand["direct_gap_rows"],
                "need_split_detour": marked_row in cand["detour_rows_marked"],
                "need_lower_detour": lower_row in cand["detour_rows_unmarked"],
                "need_straight": cand["detour_count"] == 0 and not cand["direct_gap_rows"],
            }
        )

    if not any(c["need_gap"] for c in enriched):
        return None
    if not any(c["need_split_detour"] for c in enriched):
        return None
    if not any(c["need_lower_detour"] for c in enriched):
        return None
    if not any(c["need_straight"] for c in enriched):
        return None

    random.shuffle(enriched)
    enriched.sort(
        key=lambda c: (
            -int(c["need_gap"]),
            -int(c["need_split_detour"]),
            -int(c["need_lower_detour"]),
            -int(c["need_straight"]),
            int(c["start_col"]),
        )
    )

    def rec(
        idx: int,
        chosen: list[dict[str, object]],
        used_cells: set[tuple[int, int]],
        used_final_cols: set[int],
        used_start_cols: set[int],
        has_gap: bool,
        has_split_detour: bool,
        has_lower_detour: bool,
        has_straight: bool,
    ) -> list[dict[str, object]] | None:
        if len(chosen) == target_count:
            if has_gap and has_split_detour and has_lower_detour and has_straight:
                return chosen
            return None
        if idx == len(enriched):
            return None
        if len(chosen) + (len(enriched) - idx) < target_count:
            return None

        cand = enriched[idx]
        start_col = int(cand["start_col"])
        final_col = int(cand["final_col"])
        path = set(cand["path"])
        spacing_ok = all(abs(start_col - other) >= 2 for other in used_start_cols)
        compatible = spacing_ok and final_col not in used_final_cols and path.isdisjoint(used_cells)
        if compatible:
            picked = rec(
                idx + 1,
                chosen + [cand],
                used_cells | path,
                used_final_cols | {final_col},
                used_start_cols | {start_col},
                has_gap or bool(cand["need_gap"]),
                has_split_detour or bool(cand["need_split_detour"]),
                has_lower_detour or bool(cand["need_lower_detour"]),
                has_straight or bool(cand["need_straight"]),
            )
            if picked is not None:
                return picked

        return rec(
            idx + 1,
            chosen,
            used_cells,
            used_final_cols,
            used_start_cols,
            has_gap,
            has_split_detour,
            has_lower_detour,
            has_straight,
        )

    return rec(0, [], set(), set(), set(), False, False, False, False)


def _check_semantics(
    grid: np.ndarray,
    traces: list[dict[str, object]],
    marked_row: int,
    lower_row: int,
    obstacle_color: int,
    marker_color: int,
    source_color: int,
) -> bool:
    split_grid, split_runs, split_gaps = _split_layout(grid, 0, obstacle_color, marker_color)
    out = _paint_paths(split_grid, source_color, obstacle_color, traces)
    if np.array_equal(out, grid):
        return False

    has_gap = any(marked_row in trace["direct_gap_rows"] for trace in traces)
    has_split_detour = any(marked_row in trace["detour_rows_marked"] for trace in traces)
    has_lower_detour = any(lower_row in trace["detour_rows_unmarked"] for trace in traces)
    has_straight = any(trace["detour_count"] == 0 and not trace["direct_gap_rows"] for trace in traces)
    if not (has_gap and has_split_detour and has_lower_detour and has_straight):
        return False

    unsplit_grid, unsplit_runs, unsplit_gaps = _unsplit_layout(grid, 0, obstacle_color, marker_color)
    unsplit_traces = [
        _trace_source(grid.shape[0], grid.shape[1], unsplit_runs, unsplit_gaps, int(col))
        for col in np.flatnonzero(grid[0] == source_color)
    ]
    wrong = _paint_paths(unsplit_grid, source_color, obstacle_color, unsplit_traces)
    if np.array_equal(out, wrong):
        return False

    for trace in traces:
        for r, c in trace["path"]:
            if split_grid[r, c] == obstacle_color:
                return False

    # At least one source must use a gap made by the split before being redirected later.
    if not any(marked_row in trace["direct_gap_rows"] and lower_row in trace["detour_rows_unmarked"] for trace in traces):
        return False

    # At least one source must detour on a split child segment itself.
    if not any(marked_row in trace["detour_rows_marked"] for trace in traces):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GEN_HEIGHT_RANGE)
    width = random.randint(*_GEN_WIDTH_RANGE)
    source_color, obstacle_color, marker_color = random.sample(_GEN_COLOR_POOL, 3)

    marked_row = random.randint(3, max(3, height // 2))
    lower_choices = list(range(marked_row + 3, height - 1))
    if not lower_choices:
        return None
    lower_row = random.choice(lower_choices)

    marked = _rand_marked_segment(width)
    marked["row"] = marked_row
    marker_col = int(marked["marker_col"])
    gap_cols = (marker_col - 1, marker_col, marker_col + 1)
    target_gap_col = random.choice((gap_cols[0], gap_cols[2]))
    lower_options = _possible_lower_segments(width, target_gap_col, gap_cols[2] if target_gap_col == gap_cols[0] else gap_cols[0])
    if not lower_options:
        return None
    lower_left, lower_right = random.choice(lower_options)
    lower = {
        "row": lower_row,
        "marked": False,
        "left": lower_left,
        "right": lower_right,
        "marker_col": None,
    }

    specs: list[SegmentSpec] = [marked, lower]
    used_rows = {marked_row, lower_row}
    if random.random() < 0.4:
        extra_rows = _row_choices(height, used_rows)
        if extra_rows:
            extra = _rand_extra_segment(width)
            extra["row"] = random.choice(extra_rows)
            specs.append(extra)

    # Build a temporary grid with all columns considered as candidate sources.
    candidate_grid = _build_input_from_specs(
        height,
        width,
        source_color,
        obstacle_color,
        marker_color,
        specs,
        list(range(width)),
    )
    split_grid, _runs_by_row, _gap_by_row, traces = _trace_all_sources(
        candidate_grid, source_color, obstacle_color, marker_color
    )
    candidates = []
    for trace in traces:
        # Keep only visually distinct columns whose route is fully determined by the obstacle layout.
        if int(trace["start_col"]) in {0, width - 1} and trace["detour_count"] == 0:
            continue
        candidates.append(trace)

    target_counts = list(range(_GEN_SOURCE_COUNT_RANGE[1], _GEN_SOURCE_COUNT_RANGE[0] - 1, -1))
    random.shuffle(target_counts)
    target_counts.sort(reverse=True)
    chosen: list[dict[str, object]] | None = None
    for target_count in target_counts:
        picked = _choose_sources(candidates, target_count, marked_row, lower_row)
        if picked is not None:
            chosen = picked
            break
    if chosen is None:
        return None

    source_cols = sorted(int(trace["start_col"]) for trace in chosen)
    grid = _build_input_from_specs(
        height, width, source_color, obstacle_color, marker_color, specs, source_cols
    )
    out = main(grid)
    expected = _paint_paths(split_grid, source_color, obstacle_color, chosen)
    if not np.array_equal(out, expected):
        return None
    if not _check_semantics(grid, chosen, marked_row, lower_row, obstacle_color, marker_color, source_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split-before-fall ARC input")

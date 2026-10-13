from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
HV_OPTIONS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _count_runs(values: list[bool]) -> int:
    runs = 0
    prev = False
    for cur in values:
        if cur and not prev:
            runs += 1
        prev = cur
    return runs


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    non_bg_cols = np.flatnonzero(np.any(grid != bg, axis=0))
    assert non_bg_cols.size >= 1, "expected a non-empty legend column"
    legend_c = int(non_bg_cols[-1])
    if non_bg_cols.size >= 2:
        assert legend_c - int(non_bg_cols[-2]) >= 2, "legend must be separated from the paths"

    legend_rows = [int(r) for r in np.flatnonzero(grid[:, legend_c] != bg)]
    assert legend_rows, "legend column must contain at least one color"

    entries: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    for r in legend_rows:
        color = int(grid[r, legend_c])
        assert color != bg
        assert color not in seen_colors, "each legend color must appear exactly once"
        seen_colors.add(color)
        entries.append((r, color))
    entries.sort()
    return legend_c, entries


def _recover_paths(
    grid: np.ndarray,
    bg: int,
    legend_c: int,
    legend: list[tuple[int, int]],
) -> dict[int, dict[str, int | str]]:
    main_area = grid[:, :legend_c]
    paths: dict[int, dict[str, int | str]] = {}
    for legend_row, color in legend:
        coords = np.argwhere(main_area == color)
        assert coords.size > 0, f"color {color} must appear in the main area"

        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})
        if len(rows) == 1:
            row = rows[0]
            c0 = min(cols)
            c1 = max(cols)
            assert legend_row == row, "horizontal path legend row must match the path row"
            paths[color] = {
                "color": color,
                "orientation": "h",
                "row": row,
                "start": c0,
                "end": c1,
                "anchor_row": row,
                "anchor_col": c1,
            }
        elif len(cols) == 1:
            col = cols[0]
            r0 = min(rows)
            r1 = max(rows)
            assert legend_row in (r0, r1), "vertical path legend row must match one endpoint"
            paths[color] = {
                "color": color,
                "orientation": "v",
                "col": col,
                "start": r0,
                "end": r1,
                "anchor_row": legend_row,
                "anchor_col": col,
            }
        else:
            raise AssertionError(f"color {color} is not a single straight path")
    return paths


def _path_cells(path: dict[str, int | str]) -> list[tuple[int, int]]:
    if path["orientation"] == "h":
        row = int(path["row"])
        return [(row, c) for c in range(int(path["start"]), int(path["end"]) + 1)]
    col = int(path["col"])
    return [(r, col) for r in range(int(path["start"]), int(path["end"]) + 1)]


def _recolored_paths(
    shape: tuple[int, int],
    paths: dict[int, dict[str, int | str]],
    priority: dict[int, int],
    bg: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    out = np.full(shape, bg, dtype=int)
    coverage: dict[tuple[int, int], list[int]] = {}
    for color, path in paths.items():
        for cell in _path_cells(path):
            coverage.setdefault(cell, []).append(int(color))

    winners: dict[tuple[int, int], int] = {}
    for cell, colors in coverage.items():
        winner = min(colors, key=lambda color: priority[color])
        winners[cell] = int(winner)
        out[cell] = int(winner)
    return out, winners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor crossed straight paths by legend priority, then encode each path's surviving segment count as a rightward bar.

    Input structure:
    - bg is black and the main field contains several one-cell-wide straight colored paths, each using a unique color and running either horizontally or vertically.
    - Different-colored paths can cross on shared cells; in the input, each crossing cell already shows one of the two path colors.
    - A one-column legend sits on the far right with at least one blank column of separation, and each legend pixel uses one path color on the same row as that path's legend-side endpoint.

    Rule:
    - Read the legend from top to bottom to get the path priority order: higher legend pixels have higher priority.
    - Reconstruct each full straight path from its visible same-color cells, then repaint every crossed path cell with the highest-priority color among the paths covering that cell.
    - For each path, count how many 4-connected same-color segments remain along that reconstructed line after the priority repaint.
    - Keep the recolored path network and legend, then extend a horizontal bar of that path color to the right from the endpoint whose row matches the legend pixel; the bar length equals the segment count.

    Key insight:
    - The right-side legend is not just a color list: its vertical order defines which path wins every crossing, and those crossing winners in turn determine the segment count later encoded by the small output bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_c, legend = _find_legend(g, bg)
    priority = {color: idx for idx, (_row, color) in enumerate(legend)}
    paths = _recover_paths(g, bg, legend_c, legend)

    recolored_main, winners = _recolored_paths((g.shape[0], legend_c), paths, priority, bg)

    out = np.full_like(g, bg)
    out[:, :legend_c] = recolored_main
    out[:, legend_c] = g[:, legend_c]

    for color, path in paths.items():
        visible = [winners[cell] == color for cell in _path_cells(path)]
        segment_count = _count_runs(visible)
        anchor_row = int(path["anchor_row"])
        anchor_col = int(path["anchor_col"])
        assert anchor_col + segment_count < legend_c, "insufficient legend gap for the count bar"
        for step in range(1, segment_count + 1):
            out[anchor_row, anchor_col + step] = color

    return out


def _sample_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(120):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_interval(count: int) -> tuple[int, int]:
    a = random.randrange(count)
    b = random.randrange(count)
    return (a, b) if a <= b else (b, a)


def _assign_vertical_legend_rows(
    endpoint_options: list[list[int]],
    forbidden_rows: set[int],
) -> list[int] | None:
    order = list(range(len(endpoint_options)))
    random.shuffle(order)
    assignment = [None] * len(endpoint_options)
    used = set(forbidden_rows)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        options = endpoint_options[idx][:]
        random.shuffle(options)
        for row in options:
            if row in used:
                continue
            used.add(row)
            assignment[idx] = row
            if backtrack(pos + 1):
                return True
            used.remove(row)
            assignment[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(v) for v in assignment]


def _build_spec() -> dict[str, object] | None:
    n_h, n_v = random.choice(HV_OPTIONS)
    total_paths = n_h + n_v
    gap = max(n_h, n_v) + 2

    height = random.randint(max(10, total_paths + 5), 18)
    main_w = random.randint(max(8, total_paths + 5), 18)
    width = main_w + gap + 1

    h_rows = _sample_spaced_values(2, height - 3, n_h, 2)
    v_cols = _sample_spaced_values(1, main_w - 2, n_v, 2)
    if h_rows is None or v_cols is None:
        return None

    for _ in range(150):
        h_ranges = [_sample_interval(n_v) for _ in range(n_h)]
        v_ranges = [_sample_interval(n_h) for _ in range(n_v)]

        cross = np.zeros((n_h, n_v), dtype=bool)
        for i, (h_lo, h_hi) in enumerate(h_ranges):
            for j, (v_lo, v_hi) in enumerate(v_ranges):
                cross[i, j] = h_lo <= j <= h_hi and v_lo <= i <= v_hi

        if not np.all(cross.any(axis=1)) or not np.all(cross.any(axis=0)):
            continue
        cross_count = int(cross.sum())
        if cross_count < total_paths - 1:
            continue
        if cross_count == n_h * n_v and n_h * n_v > 2 and random.random() < 0.65:
            continue

        horizontals = []
        for i, row in enumerate(h_rows):
            lo_idx, hi_idx = h_ranges[i]
            base_left = v_cols[lo_idx]
            base_right = v_cols[hi_idx]
            left_margin = random.randint(1, min(3, base_left))
            right_margin = random.randint(1, min(3, main_w - 1 - base_right))
            left = base_left - left_margin
            right = base_right + right_margin
            horizontals.append({"row": row, "col0": left, "col1": right})

        verticals = []
        legend_choices: list[list[int]] = []
        for j, col in enumerate(v_cols):
            lo_idx, hi_idx = v_ranges[j]
            top = h_rows[lo_idx] - 1
            bottom = h_rows[hi_idx] + 1
            if top < 0 or bottom >= height or bottom - top < 2:
                break
            verticals.append({"col": col, "row0": top, "row1": bottom})
            legend_choices.append([top, bottom])
        else:
            vertical_legend_rows = _assign_vertical_legend_rows(legend_choices, set(h_rows))
            if vertical_legend_rows is None:
                continue

            colors = random.sample(COLOR_POOL, total_paths)
            random.shuffle(colors)

            paths = []
            for i, spec in enumerate(horizontals):
                paths.append(
                    {
                        "color": colors[i],
                        "orientation": "h",
                        "row": int(spec["row"]),
                        "start": int(spec["col0"]),
                        "end": int(spec["col1"]),
                        "legend_row": int(spec["row"]),
                    }
                )
            for j, spec in enumerate(verticals):
                paths.append(
                    {
                        "color": colors[n_h + j],
                        "orientation": "v",
                        "col": int(spec["col"]),
                        "start": int(spec["row0"]),
                        "end": int(spec["row1"]),
                        "legend_row": int(vertical_legend_rows[j]),
                    }
                )

            legend = sorted((int(path["legend_row"]), int(path["color"])) for path in paths)
            priority = {color: idx for idx, (_row, color) in enumerate(legend)}

            max_segments = 0
            for path in paths:
                path_color = int(path["color"])
                if path["orientation"] == "h":
                    row = int(path["row"])
                    cols_here = [
                        int(other["col"])
                        for other in paths
                        if other["orientation"] == "v"
                        and int(other["start"]) <= row <= int(other["end"])
                        and int(path["start"]) <= int(other["col"]) <= int(path["end"])
                        and priority[int(other["color"])] < priority[path_color]
                    ]
                    segments = 1 + len(cols_here)
                else:
                    col = int(path["col"])
                    rows_here = [
                        int(other["row"])
                        for other in paths
                        if other["orientation"] == "h"
                        and int(other["start"]) <= col <= int(other["end"])
                        and int(path["start"]) <= int(other["row"]) <= int(path["end"])
                        and priority[int(other["color"])] < priority[path_color]
                    ]
                    segments = 1 + len(rows_here)
                max_segments = max(max_segments, segments)

            if max_segments > gap:
                continue

            return {
                "height": height,
                "width": width,
                "main_w": main_w,
                "legend": legend,
                "paths": paths,
                "priority": priority,
                "cross": cross,
            }
    return None


def _render_input(spec: dict[str, object]) -> np.ndarray:
    height = int(spec["height"])
    width = int(spec["width"])
    main_w = int(spec["main_w"])
    legend = list(spec["legend"])
    priority = dict(spec["priority"])
    paths = list(spec["paths"])

    grid = np.full((height, width), BACKGROUND, dtype=int)
    legend_c = width - 1

    coverage: dict[tuple[int, int], list[int]] = {}
    for path in paths:
        color = int(path["color"])
        if path["orientation"] == "h":
            row = int(path["row"])
            for col in range(int(path["start"]), int(path["end"]) + 1):
                coverage.setdefault((row, col), []).append(color)
        else:
            col = int(path["col"])
            for row in range(int(path["start"]), int(path["end"]) + 1):
                coverage.setdefault((row, col), []).append(color)

    crossings = [cell for cell, colors in coverage.items() if len(colors) == 2]
    forced_wrong = random.choice(crossings) if crossings else None

    for cell, colors in coverage.items():
        if len(colors) == 1:
            grid[cell] = colors[0]
            continue
        higher = min(colors, key=lambda color: priority[color])
        lower = max(colors, key=lambda color: priority[color])
        if cell == forced_wrong:
            grid[cell] = lower
        else:
            grid[cell] = random.choice(colors)

    for row, color in legend:
        grid[int(row), legend_c] = int(color)

    assert not np.any(grid[:, main_w:legend_c] != BACKGROUND), "gap must stay empty in the input"
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        if _is_valid_input(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")

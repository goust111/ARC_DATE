from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _ray_path_color(
    grid: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    play_width: int,
) -> int | None:
    """Walk away from a grey crossing until a colored path segment or background appears."""
    h = grid.shape[0]
    rr = row + dr
    cc = col + dc
    while 0 <= rr < h and 0 <= cc < play_width:
        value = int(grid[rr, cc])
        if value == BG:
            return None
        if value != MARKER:
            return value
        rr += dr
        cc += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint grey crossing markers with the highest-priority path color and remove the edge legend column.

    Input structure:
    - The background is black.
    - Several differently colored single-pixel orthogonal paths run through the grid; each path is 4-connected and its endpoints lie on the outer border.
    - Whenever two or more paths share one crossing pixel, that shared pixel is replaced by a special grey marker.
    - The rightmost column contains a vertical color legend whose colored cells, read top to bottom, encode path priority from highest to lowest.

    Rule:
    - Read the non-background cells of the rightmost column from top to bottom to obtain the priority order.
    - Remove the legend column in the output.
    - For each grey crossing marker, trace outward in the four cardinal directions to find which path colors meet at that pixel.
    - Recolor the grey crossing with the highest-priority color among those incident paths, while leaving every non-crossing path pixel and every background pixel unchanged.

    Key insight:
    - The grey cells mark exactly where arbitration is needed, and the winner is not determined geometrically; it is determined by the separate priority legend on the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must have room for content and a legend column"

    legend = [int(v) for v in g[:, -1] if int(v) not in {BG, MARKER}]
    assert legend, "expected at least one legend color in the rightmost column"
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    priority = {color: rank for rank, color in enumerate(legend)}

    out = g[:, :-1].copy()
    play_width = w - 1

    for row, col in np.argwhere(g[:, :-1] == MARKER):
        row = int(row)
        col = int(col)
        incident: set[int] = set()
        for dr, dc in DIRS4:
            color = _ray_path_color(g, row, col, dr, dc, play_width)
            if color is None:
                continue
            assert color in priority, f"path color {color} is missing from the legend"
            incident.add(int(color))

        assert incident, f"grey marker at {(row, col)} has no incident path colors"
        out[row, col] = min(incident, key=priority.get)

    return out


def _sample_even_positions(limit: int, low: int, high: int) -> list[int]:
    return [v for v in range(low, high + 1) if 0 <= v < limit and v % 2 == 0]


def _add_cells(cell_to_colors: dict[tuple[int, int], set[int]], cells: list[tuple[int, int]], color: int) -> None:
    for cell in cells:
        cell_to_colors.setdefault(cell, set()).add(int(color))


def _vertical_path_cells(height: int, col: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(height)]


def _left_bent_path_cells(height: int, row: int, turn_col: int, target: str) -> list[tuple[int, int]]:
    assert target in {"top", "bottom"}
    cells = {(row, col) for col in range(turn_col + 1)}
    if target == "top":
        cells.update((rr, turn_col) for rr in range(row + 1))
    else:
        cells.update((rr, turn_col) for rr in range(row, height))
    return sorted(cells)


def _legend_rows(height: int, count: int) -> list[int]:
    rows = sorted(random.sample(range(height), count))
    return [int(r) for r in rows]


def _expected_output(
    grid: np.ndarray,
    intersections: dict[tuple[int, int], set[int]],
    priority_order: list[int],
) -> np.ndarray:
    priority = {color: rank for rank, color in enumerate(priority_order)}
    out = grid[:, :-1].copy()
    for (row, col), colors in intersections.items():
        out[row, col] = min(colors, key=priority.get)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(list(range(12, 25, 2)))
    core_width = random.randint(11, 22)
    total_width = core_width + 2  # blank separator + legend

    vertical_candidates = [col for col in range(3, core_width - 2, 2)]
    turn_candidates = [col for col in range(4, core_width - 1, 2)]
    if not vertical_candidates or not turn_candidates:
        return None

    all_path_rows = _sample_even_positions(height, 2, height - 3)
    top_row_candidates = [row for row in all_path_rows if row < height // 2]
    bottom_row_candidates = [row for row in all_path_rows if row > height // 2]
    if not top_row_candidates and not bottom_row_candidates:
        return None

    n_vertical = random.randint(1, min(4, len(vertical_candidates)))
    vertical_cols = sorted(random.sample(vertical_candidates, n_vertical))
    eligible_turns = [col for col in turn_candidates if col > vertical_cols[0]]
    if not eligible_turns:
        return None

    max_left = min(3, len(eligible_turns), len(all_path_rows))
    if max_left == 0:
        return None
    n_left = random.randint(1, max_left)

    feasible_top_counts = [
        n_top
        for n_top in range(n_left + 1)
        if n_top <= len(top_row_candidates) and (n_left - n_top) <= len(bottom_row_candidates)
    ]
    if not feasible_top_counts:
        return None
    n_top = random.choice(feasible_top_counts)
    n_bottom = n_left - n_top

    top_rows = sorted(random.sample(top_row_candidates, n_top)) if n_top else []
    bottom_rows = sorted(random.sample(bottom_row_candidates, n_bottom)) if n_bottom else []

    chosen_turns = random.sample(eligible_turns, n_left)
    top_turns = sorted(chosen_turns[:n_top]) if n_top else []
    bottom_turns = sorted(chosen_turns[n_top:], reverse=True) if n_bottom else []

    total_paths = n_vertical + n_left
    colors = random.sample(PATH_COLORS, total_paths)
    vertical_colors = colors[:n_vertical]
    left_colors = colors[n_vertical:]
    priority_order = random.sample(colors, len(colors))

    grid = np.full((height, total_width), BG, dtype=int)
    cell_to_colors: dict[tuple[int, int], set[int]] = {}

    vertical_paths = list(zip(vertical_cols, vertical_colors, strict=True))
    for col, color in vertical_paths:
        _add_cells(cell_to_colors, _vertical_path_cells(height, int(col)), int(color))

    left_specs: list[tuple[int, int, str, int]] = []
    color_index = 0
    for row, turn in zip(top_rows, top_turns, strict=True):
        color = int(left_colors[color_index])
        color_index += 1
        left_specs.append((int(row), int(turn), "top", color))
    for row, turn in zip(bottom_rows, bottom_turns, strict=True):
        color = int(left_colors[color_index])
        color_index += 1
        left_specs.append((int(row), int(turn), "bottom", color))

    for row, turn_col, target, color in left_specs:
        _add_cells(cell_to_colors, _left_bent_path_cells(height, row, turn_col, target), color)

    intersections = {
        cell: colors_here
        for cell, colors_here in cell_to_colors.items()
        if len(colors_here) >= 2
    }
    if not intersections:
        return None
    if any(len(colors_here) != 2 for colors_here in intersections.values()):
        return None

    for row, turn_col, _target, color in left_specs:
        crossed = [
            col
            for col, vertical_color in vertical_paths
            if col <= turn_col and intersections.get((row, col)) == {int(color), int(vertical_color)}
        ]
        if not crossed:
            return None

    for (row, col), colors_here in cell_to_colors.items():
        if len(colors_here) == 1:
            grid[row, col] = next(iter(colors_here))
        else:
            grid[row, col] = MARKER

    legend_col = total_width - 1
    for row, color in zip(_legend_rows(height, total_paths), priority_order, strict=True):
        grid[row, legend_col] = int(color)

    expected = _expected_output(grid, intersections, priority_order)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")

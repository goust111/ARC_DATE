from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (2, 3, 4)
GRID_BOUNDS = (20, 28)
MAX_ACTIVE_COLORS = 2
MAX_ARM_LENGTH = 6
MAX_CENTERS_PER_COLOR = 4
CLIP_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 400


def _collect_markers(
    grid: np.ndarray, frame_color: int, bg_color: int
) -> tuple[dict[int, dict[str, list[int] | int]], list[int]]:
    height, width = grid.shape
    info: dict[int, dict[str, list[int] | int]] = {}
    draw_order: list[int] = []

    def add_marker(color: int, axis: str, position: int) -> None:
        if color not in info:
            info[color] = {"rows": [], "cols": [], "count": 0}
            draw_order.append(color)
        assert color != bg_color, "border markers must differ from the interior background"
        info[color][axis].append(int(position))
        info[color]["count"] = int(info[color]["count"]) + 1

    for col in range(1, width - 1):
        color = int(grid[0, col])
        if color != frame_color:
            add_marker(color, "cols", col)
    for col in range(1, width - 1):
        color = int(grid[height - 1, col])
        if color != frame_color:
            add_marker(color, "cols", col)
    for row in range(1, height - 1):
        color = int(grid[row, 0])
        if color != frame_color:
            add_marker(color, "rows", row)
    for row in range(1, height - 1):
        color = int(grid[row, width - 1])
        if color != frame_color:
            add_marker(color, "rows", row)

    return info, draw_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw interior crosses from colored border ticks.

    Valid generated instances keep different colors' cross footprints disjoint,
    so no hidden color-order tiebreak is required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4

    frame_color = int(g[0, 0])
    assert frame_color == int(g[0, -1]) == int(g[-1, 0]) == int(g[-1, -1])
    bg_color = int(g[1, 1])
    assert frame_color != bg_color
    assert np.all(g[1:-1, 1:-1] == bg_color), "input interior must start empty"

    marker_info, draw_order = _collect_markers(g, frame_color, bg_color)
    out = g.copy()

    for color in draw_order:
        cols = sorted(set(int(v) for v in marker_info[color]["cols"]))
        rows = sorted(set(int(v) for v in marker_info[color]["rows"]))
        arm = int(marker_info[color]["count"])
        if not cols or not rows:
            continue

        for row in rows:
            r0 = max(1, row - arm)
            r1 = min(height - 2, row + arm)
            for col in cols:
                c0 = max(1, col - arm)
                c1 = min(width - 2, col + arm)
                out[r0 : r1 + 1, col] = color
                out[row, c0 : c1 + 1] = color

    return out


def _sample_spaced_positions(
    candidates: list[int], count: int, required: set[int] | None = None
) -> list[int] | None:
    if count == 0:
        return []
    candidates = sorted(set(int(v) for v in candidates))
    if len(candidates) < count:
        return None

    options = []
    for combo in itertools.combinations(candidates, count):
        if any(b - a < 2 for a, b in zip(combo, combo[1:])):
            continue
        if required is not None and not any(value in required for value in combo):
            continue
        options.append(combo)
    if not options:
        return None
    return list(random.choice(options))


def _sample_axis_positions(length: int, count: int, arm: int, mode: str) -> list[int] | None:
    inner_max = length - 2
    if mode == "safe":
        return _sample_spaced_positions(list(range(1 + arm, length - 1 - arm)), count)

    clip_candidates = set(range(1, min(arm, inner_max) + 1))
    clip_candidates.update(range(max(1, length - 1 - arm), inner_max + 1))
    return _sample_spaced_positions(list(range(1, inner_max + 1)), count, clip_candidates)


def _is_clipped(height: int, width: int, rows: list[int], cols: list[int], arm: int) -> bool:
    return any(row <= arm or row >= height - 1 - arm for row in rows) or any(
        col <= arm or col >= width - 1 - arm for col in cols
    )


def _cross_footprint(
    height: int, width: int, rows: list[int], cols: list[int], arm: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for row in rows:
        for col in cols:
            for rr in range(max(1, row - arm), min(height - 2, row + arm) + 1):
                cells.add((rr, col))
            for cc in range(max(1, col - arm), min(width - 2, col + arm) + 1):
                cells.add((row, cc))
    return cells


def _sample_color_layout(
    height: int, width: int, prefer_clip: bool
) -> dict[str, list[int] | int] | None:
    for _ in range(200):
        unique_cols = random.choice((1, 1, 2, 2, 3))
        unique_rows = random.choice((1, 1, 2, 2, 3))
        if unique_cols * unique_rows > MAX_CENTERS_PER_COLOR:
            continue

        col_states = [random.choice(("top", "bottom", "both")) for _ in range(unique_cols)]
        row_states = [random.choice(("left", "right", "both")) for _ in range(unique_rows)]

        top_count = sum(state in ("top", "both") for state in col_states)
        bottom_count = sum(state in ("bottom", "both") for state in col_states)
        left_count = sum(state in ("left", "both") for state in row_states)
        right_count = sum(state in ("right", "both") for state in row_states)
        arm = top_count + bottom_count + left_count + right_count
        if not 2 <= arm <= MAX_ARM_LENGTH:
            continue

        if prefer_clip:
            clip_axis = random.choice(("row", "col"))
            row_mode = "clip" if clip_axis == "row" else "safe"
            col_mode = "clip" if clip_axis == "col" else "safe"
        else:
            row_mode = "safe"
            col_mode = "safe"

        rows = _sample_axis_positions(height, unique_rows, arm, row_mode)
        cols = _sample_axis_positions(width, unique_cols, arm, col_mode)
        if rows is None or cols is None:
            continue

        layout = {
            "rows": rows,
            "cols": cols,
            "top": [cols[i] for i, state in enumerate(col_states) if state in ("top", "both")],
            "bottom": [
                cols[i] for i, state in enumerate(col_states) if state in ("bottom", "both")
            ],
            "left": [rows[i] for i, state in enumerate(row_states) if state in ("left", "both")],
            "right": [
                rows[i] for i, state in enumerate(row_states) if state in ("right", "both")
            ],
            "arm": arm,
        }
        if _is_clipped(height, width, rows, cols, arm) == prefer_clip:
            return layout

    return None


def _border_cells(
    height: int, width: int, layout: dict[str, list[int] | int]
) -> set[tuple[int, int]]:
    return {
        *((0, col) for col in layout["top"]),
        *((height - 1, col) for col in layout["bottom"]),
        *((row, 0) for row in layout["left"]),
        *((row, width - 1) for row in layout["right"]),
    }


def _paint_input(
    height: int, width: int, specs: list[tuple[int, dict[str, list[int] | int]]]
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    for color, layout in specs:
        for col in layout["top"]:
            grid[0, col] = color
        for col in layout["bottom"]:
            grid[height - 1, col] = color
        for row in layout["left"]:
            grid[row, 0] = color
        for row in layout["right"]:
            grid[row, width - 1] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_BOUNDS)
    width = random.randint(*GRID_BOUNDS)
    prefer_clip = random.random() < CLIP_PROB
    color_count = random.choice((1, 1, 2))
    active_colors = random.sample(COLOR_POOL, color_count)
    clip_index = random.randrange(color_count) if prefer_clip else -1

    specs: list[tuple[int, dict[str, list[int] | int]]] = []
    used_border: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    saw_clip = False

    for index, color in enumerate(active_colors):
        layout = _sample_color_layout(height, width, prefer_clip and index == clip_index)
        if layout is None:
            return None

        border = _border_cells(height, width, layout)
        if used_border & border:
            return None

        rows = list(layout["rows"])
        cols = list(layout["cols"])
        arm = int(layout["arm"])
        footprint = _cross_footprint(height, width, rows, cols, arm)
        if occupied & footprint:
            return None

        used_border |= border
        occupied |= footprint
        saw_clip = saw_clip or _is_clipped(height, width, rows, cols, arm)
        specs.append((color, layout))

    if saw_clip != prefer_clip:
        return None

    grid = _paint_input(height, width, specs)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")

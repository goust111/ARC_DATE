import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7
PATH_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (18, 28)
MARKED_PATH_COUNT_RANGE = (2, 3)
CROSS_PATH_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each activated path to orange, letting only visibly top-layer crossings remain traversable.

    Input structure:
    - The background is black.
    - Several distinct non-orange colors form one-pixel-wide orthogonal paths made
      of horizontal and vertical segments; some paths cross by sharing one cell.
    - Orange marker pixels lie on a single outer edge and align with the activated
      path endpoints just inside the grid.

    Rule:
    - For each orange edge marker, follow the inward normal direction to the first
      non-background, non-orange cell and treat that visible cell as the signal's
      entry point.
    - Propagate through the 4-connected visible cells of that same color and
      recolor every visited path pixel to orange.
    - A crossing is traversable exactly when that path is visible on top there:
      then the crossing cell already has the path's color and stays connected.
      If another color is visible at the crossing, the lower path is split there
      and the signal cannot continue through it.
    - Keep orange markers and background unchanged.

    Key insight:
    - The stacking rule is encoded directly in visibility: top-layer crossings
      preserve a same-colored bridge, while lower-layer crossings appear as a
      one-cell gap that breaks visible connectivity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    marker_coords = np.argwhere(g == ORANGE)
    for marker_r, marker_c in marker_coords:
        start = _scan_inward_to_path(g, int(marker_r), int(marker_c))
        if start is None:
            continue
        sr, sc = start
        color = int(g[sr, sc])
        if color in (BG, ORANGE):
            continue
        for r, c in _visible_component(g, (sr, sc), color):
            out[r, c] = ORANGE

    return out


def _scan_inward_to_path(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    h, w = grid.shape
    directions: list[tuple[int, int]] = []
    if r == 0:
        directions.append((1, 0))
    if r == h - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == w - 1:
        directions.append((0, -1))

    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            value = int(grid[rr, cc])
            if value not in (BG, ORANGE):
                return rr, cc
            rr += dr
            cc += dc
    return None


def _visible_component(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) in seen or int(grid[rr, cc]) != color:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return seen


def _sample_spaced_positions(
    count: int,
    low: int,
    high: int,
    min_gap: int,
    rng: random.Random,
    attempts: int = 200,
) -> list[int] | None:
    values = list(range(low, high + 1))
    for _ in range(attempts):
        rng.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        else:
            raise ValueError("segments must be axis-aligned")
        if cells:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _render_scene(
    paths: list[dict],
    markers: list[tuple[int, int]],
    shape: tuple[int, int],
    top_color_at: dict[tuple[int, int], int],
) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    coverage: dict[tuple[int, int], list[int]] = {}

    for idx, path in enumerate(paths):
        for cell in path["cells"]:
            coverage.setdefault(cell, []).append(idx)

    for cell, ids in coverage.items():
        if len(ids) > 2:
            return None
        if len(ids) == 2 and cell not in top_color_at:
            return None

    for path in paths:
        for r, c in path["cells"]:
            if int(grid[r, c]) == BG:
                grid[r, c] = path["color"]

    for (r, c), color in top_color_at.items():
        grid[r, c] = color

    for r, c in markers:
        if int(grid[r, c]) != BG:
            return None
        grid[r, c] = ORANGE

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marked_count = rng.randint(*MARKED_PATH_COUNT_RANGE)
    cross_count = rng.randint(*CROSS_PATH_COUNT_RANGE)

    if h < 4 * marked_count + 4:
        return None

    band_starts = _sample_spaced_positions(marked_count, 2, h - 4, 4, rng)
    cross_cols = _sample_spaced_positions(cross_count, 4, w - 4, 3, rng)
    if band_starts is None or cross_cols is None:
        return None

    turn_cols = [
        c
        for c in range(3, w - 2)
        if all(abs(c - cross_col) >= 2 for cross_col in cross_cols)
    ]
    rng.shuffle(turn_cols)

    path_colors = rng.sample(PATH_COLORS, marked_count + cross_count)
    paths: list[dict] = []
    markers: list[tuple[int, int]] = []
    marked_specs: list[dict] = []
    for i, band_start in enumerate(band_starts):
        band_rows = [band_start, band_start + 1, band_start + 2]
        start_row = rng.choice(band_rows)
        use_turn = bool(turn_cols) and rng.random() < 0.6

        if use_turn:
            end_row_choices = [row for row in band_rows if row != start_row]
            end_row = rng.choice(end_row_choices)
            turn_col = turn_cols.pop()
            points = [(start_row, 1), (start_row, turn_col), (end_row, turn_col), (end_row, w - 2)]
        else:
            end_row = start_row
            turn_col = None
            points = [(start_row, 1), (start_row, w - 2)]

        cells = _polyline_cells(points)
        path_idx = len(paths)
        paths.append({"color": path_colors[i], "cells": cells, "marked": True})
        markers.append((start_row, 0))
        marked_specs.append(
            {
                "path_idx": path_idx,
                "start_row": start_row,
                "end_row": end_row,
                "turn_col": turn_col,
            }
        )

    top_rows = [spec["start_row"] for spec in marked_specs] + [spec["end_row"] for spec in marked_specs]
    min_row = min(top_rows)
    max_row = max(top_rows)
    top_limit = max(1, min_row - rng.randint(1, 2))
    bottom_limit = min(h - 2, max_row + rng.randint(1, 2))
    if top_limit >= bottom_limit:
        return None

    for i, col in enumerate(cross_cols):
        cells = _polyline_cells([(top_limit, col), (bottom_limit, col)])
        paths.append({"color": path_colors[marked_count + i], "cells": cells, "marked": False})

    path_at_cross: dict[tuple[int, int], tuple[int, int]] = {}
    top_color_at: dict[tuple[int, int], int] = {}

    sorted_cols = sorted(cross_cols)
    if len(marked_specs) == 2:
        path_modes = ["pass", "block"]
        rng.shuffle(path_modes)
    else:
        path_modes = ["pass", "early_block", "late_block"]
        rng.shuffle(path_modes)
        path_modes = path_modes[:marked_count]

    for spec_idx, spec in enumerate(marked_specs):
        mode = path_modes[spec_idx % len(path_modes)]
        if mode == "pass":
            block_col = None
        elif mode == "early_block":
            block_col = sorted_cols[rng.randrange(max(1, len(sorted_cols) // 2))]
        elif mode == "late_block":
            lo = max(1, len(sorted_cols) // 2)
            block_col = sorted_cols[rng.randrange(lo, len(sorted_cols))]
        else:
            block_col = sorted_cols[rng.randrange(len(sorted_cols))]

        for col in sorted_cols:
            if spec["turn_col"] is None:
                cross_row = spec["start_row"]
            else:
                cross_row = spec["start_row"] if col < spec["turn_col"] else spec["end_row"]

            cell = (cross_row, col)
            cross_path_idx = marked_count + cross_cols.index(col)
            path_at_cross[cell] = (spec["path_idx"], cross_path_idx)

            if block_col is None or col < block_col:
                top_color_at[cell] = paths[spec["path_idx"]]["color"]
            elif col == block_col:
                top_color_at[cell] = paths[cross_path_idx]["color"]
            else:
                if rng.random() < 0.5:
                    top_color_at[cell] = paths[spec["path_idx"]]["color"]
                else:
                    top_color_at[cell] = paths[cross_path_idx]["color"]

    if not any(top_color_at[cell] == paths[path_idx]["color"] for cell, (path_idx, _) in path_at_cross.items()):
        return None
    if not any(top_color_at[cell] != paths[path_idx]["color"] for cell, (path_idx, _) in path_at_cross.items()):
        return None

    grid = _render_scene(paths, markers, (h, w), top_color_at)
    if grid is None:
        return None

    k = rng.randrange(4)
    grid = np.rot90(grid, k)

    out = main(grid)
    path_mask = (grid != BG) & (grid != ORANGE)
    recolored_mask = (out == ORANGE) & path_mask

    if recolored_mask.sum() < 12:
        return None
    if not np.any(path_mask & (out == grid)):
        return None

    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    marker_count = int(((grid == ORANGE) & border_mask).sum())
    if marker_count != marked_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

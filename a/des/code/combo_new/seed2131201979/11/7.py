from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge_cell(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _step(cell: tuple[int, int], direction: int, amount: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return cell[0] + dr * amount, cell[1] + dc * amount


def _dir_from_edge(edge: str) -> int:
    return {"top": 2, "right": 3, "bottom": 0, "left": 1}[edge]


def _source_cell(edge: str, coord: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if edge == "top":
        return 0, coord
    if edge == "bottom":
        return h - 1, coord
    if edge == "left":
        return coord, 0
    return coord, w - 1


def _source_sort_key(edge: str, cell: tuple[int, int]) -> int:
    return cell[1] if edge in ("top", "bottom") else cell[0]


def _is_parallel(direction: int, orientation: str) -> bool:
    return (direction % 2 == 0 and orientation == "v") or (
        direction % 2 == 1 and orientation == "h"
    )


def _steps_until_edge(shape: tuple[int, int], cell: tuple[int, int], direction: int) -> int:
    r, c = cell
    dr, dc = DIRS[direction]
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return steps
        steps += 1


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _extract_bars(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    bars: list[dict] = []
    cell_to_bar: dict[tuple[int, int], dict] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in (BLACK, GREY)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "bar colors must appear in the grid"
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == 1 or w == 1, "each bar must be one pixel thick"
        assert h + w >= 3, "bars must have length at least 2"
        assert coords.shape[0] == h * w, "each bar must be a solid rectangle"
        orientation = "h" if h == 1 else "v"
        cells = {(int(r), int(c)) for r, c in coords}
        bar = {"color": color, "cells": cells, "orientation": orientation}
        bars.append(bar)
        for cell in cells:
            assert cell not in cell_to_bar, "bars must not overlap"
            cell_to_bar[cell] = bar
    return bars, cell_to_bar


def _detect_sources(grid: np.ndarray) -> tuple[str, list[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert coords, "expected at least one grey source pixel"
    h, w = grid.shape
    candidates = []
    if all(r == 0 for r, _c in coords):
        candidates.append("top")
    if all(r == h - 1 for r, _c in coords):
        candidates.append("bottom")
    if all(c == 0 for _r, c in coords):
        candidates.append("left")
    if all(c == w - 1 for _r, c in coords):
        candidates.append("right")
    assert len(candidates) == 1, "all grey sources must lie on exactly one grid edge"
    edge = candidates[0]
    coords.sort(key=lambda cell: _source_sort_key(edge, cell))
    return edge, coords


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    source: tuple[int, int],
    direction: int,
    cell_to_bar: dict[tuple[int, int], dict],
) -> None:
    shape = tuple(map(int, grid.shape))
    r, c = source
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, direction)
        assert state not in seen, "ray path should not loop"
        seen.add(state)

        nr, nc = _step((r, c), direction)
        if not _in_bounds(shape, nr, nc):
            return

        value = int(grid[nr, nc])
        if value in (BLACK, GREY):
            out[nr, nc] = GREY
            r, c = nr, nc
            continue

        bar = cell_to_bar.get((nr, nc))
        assert bar is not None, "all foreground pixels must belong to bars"
        if _is_parallel(direction, bar["orientation"]):
            dr, dc = DIRS[direction]
            tr, tc = nr, nc
            while _in_bounds(shape, tr, tc) and cell_to_bar.get((tr, tc)) is bar:
                tr += dr
                tc += dc
            if not _in_bounds(shape, tr, tc):
                return
            assert int(grid[tr, tc]) in (BLACK, GREY), "parallel traversal must exit into background"
            out[tr, tc] = GREY
            r, c = tr, tc
            continue

        exit_r, exit_c = nr + DIRS[direction][0], nc + DIRS[direction][1]
        if not _in_bounds(shape, exit_r, exit_c):
            return
        assert int(grid[exit_r, exit_c]) in (BLACK, GREY), "perpendicular traversal must exit into background"
        out[exit_r, exit_c] = GREY
        r, c = exit_r, exit_c
        direction = (direction + 1) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace grey edge rays through one-pixel-wide colored bars, turning only after perpendicular crossings.

    Input structure:
    - The background is black.
    - Several separated solid bars of distinct colors appear in the grid; each bar is
      exactly one pixel thick and is either horizontal or vertical.
    - One grid edge contains one or more grey source pixels, and each source emits a
      ray straight inward, perpendicular to that edge.

    Rule:
    - Copy the input grid, then simulate one ray from each grey edge source.
    - Whenever a ray moves through background, paint those traversed background pixels
      grey.
    - If the ray enters a bar whose orientation is parallel to the current ray
      direction, it passes through the full bar and keeps the same direction.
    - If the ray enters a bar whose orientation is perpendicular to the current ray
      direction, it crosses the bar's one-pixel thickness, then turns 90 degrees
      clockwise from the first background cell beyond the bar.
    - Colored bar pixels never change, and each ray stops when it leaves the grid.

    Key insight:
    - A bar is not an endpoint or a mirror: its effect depends on its orientation
      relative to the ray's current direction, so earlier turns can change how later
      bars behave.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    _bars, cell_to_bar = _extract_bars(g)
    edge, sources = _detect_sources(g)
    out = g.copy()
    direction = _dir_from_edge(edge)
    for source in sources:
        _trace_ray(g, out, source, direction, cell_to_bar)
    return out


GRID_SIDE_RANGE = (16, 28)
NUM_SOURCES_RANGE = (2, 4)
MIN_SOURCE_GAP = 3
SOURCE_MARGIN = 2
BAR_LENGTH_RANGE = (2, 5)
FEATURE_PATTERNS = [
    ["parallel", "turn"],
    ["turn", "parallel"],
    ["parallel", "turn", "parallel"],
    ["turn", "parallel", "turn"],
]
AUX_PATTERNS = [
    [],
    ["parallel"],
    ["turn"],
    ["parallel", "turn"],
    ["turn", "parallel"],
]
RAY_ATTEMPTS = 120
SCENE_ATTEMPTS = 400


def _sample_edge_positions(length: int, count: int) -> list[int] | None:
    low = SOURCE_MARGIN
    high = length - SOURCE_MARGIN - 1
    if high < low:
        return None
    pool = list(range(low, high + 1))
    for _ in range(200):
        random.shuffle(pool)
        chosen: list[int] = []
        for pos in pool:
            if all(abs(pos - prev) >= MIN_SOURCE_GAP for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return None


def _sample_patterns(num_sources: int) -> list[list[str]] | None:
    for _ in range(200):
        patterns = [random.choice(FEATURE_PATTERNS)]
        while len(patterns) < num_sources:
            patterns.append(random.choice(AUX_PATTERNS))
        random.shuffle(patterns)
        total_bars = sum(len(pattern) for pattern in patterns)
        if 2 <= total_bars <= len(BAR_COLORS):
            return patterns
    return None


def _cells_clear(
    cells: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    reserved_sources: set[tuple[int, int]],
    local_used: set[tuple[int, int]],
) -> bool:
    return all(cell not in blocked and cell not in reserved_sources and cell not in local_used for cell in cells)


def _parallel_candidates(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: int,
    blocked: set[tuple[int, int]],
    reserved_sources: set[tuple[int, int]],
    local_used: set[tuple[int, int]],
    remaining_events: int,
) -> list[dict]:
    min_future = 4 if remaining_events else 1
    candidates = []
    for gap in range(1, 7):
        for length in range(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1] + 1):
            path_cells = [_step(current, direction, offset) for offset in range(1, gap + 1)]
            bar_cells = [_step(current, direction, offset) for offset in range(gap + 1, gap + length + 1)]
            exit_cell = _step(current, direction, gap + length + 1)
            all_cells = path_cells + bar_cells + [exit_cell]
            if not all(_in_bounds(shape, r, c) for r, c in all_cells):
                continue
            if any(_is_edge_cell(shape, r, c) for r, c in bar_cells):
                continue
            next_cell = _step(exit_cell, direction)
            if not _in_bounds(shape, *next_cell):
                continue
            if _steps_until_edge(shape, exit_cell, direction) < min_future:
                continue
            if not _cells_clear(path_cells + [exit_cell], blocked, reserved_sources, local_used):
                continue
            if not _cells_clear(bar_cells, blocked, reserved_sources, local_used):
                continue
            if next_cell in blocked or next_cell in reserved_sources or next_cell in local_used:
                continue
            candidates.append(
                {
                    "path_cells": path_cells + [exit_cell],
                    "bar_cells": bar_cells,
                    "orientation": "v" if direction % 2 == 0 else "h",
                    "new_pos": exit_cell,
                    "new_dir": direction,
                }
            )
    random.shuffle(candidates)
    return candidates


def _perpendicular_candidates(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: int,
    blocked: set[tuple[int, int]],
    reserved_sources: set[tuple[int, int]],
    local_used: set[tuple[int, int]],
    remaining_events: int,
) -> list[dict]:
    new_direction = (direction + 1) % 4
    min_future = 4 if remaining_events else 1
    candidates = []
    for gap in range(1, 7):
        path_cells = [_step(current, direction, offset) for offset in range(1, gap + 1)]
        cross_cell = _step(current, direction, gap + 1)
        exit_cell = _step(current, direction, gap + 2)
        base_cells = path_cells + [cross_cell, exit_cell]
        if not all(_in_bounds(shape, r, c) for r, c in base_cells):
            continue
        if _is_edge_cell(shape, *cross_cell):
            continue
        for total_len in range(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1] + 1):
            span = total_len - 1
            for before in range(span + 1):
                after = span - before
                if direction % 2 == 0:
                    bar_cells = [(cross_cell[0], cross_cell[1] + delta) for delta in range(-before, after + 1)]
                    orientation = "h"
                else:
                    bar_cells = [(cross_cell[0] + delta, cross_cell[1]) for delta in range(-before, after + 1)]
                    orientation = "v"
                if not all(_in_bounds(shape, r, c) for r, c in bar_cells):
                    continue
                if any(_is_edge_cell(shape, r, c) for r, c in bar_cells):
                    continue
                turn_step = _step(exit_cell, new_direction)
                if not _in_bounds(shape, *turn_step):
                    continue
                if _steps_until_edge(shape, exit_cell, new_direction) < min_future:
                    continue
                if not _cells_clear(path_cells + [exit_cell], blocked, reserved_sources, local_used):
                    continue
                if not _cells_clear(bar_cells, blocked, reserved_sources, local_used):
                    continue
                if turn_step in blocked or turn_step in reserved_sources or turn_step in local_used:
                    continue
                candidates.append(
                    {
                        "path_cells": path_cells + [exit_cell],
                        "bar_cells": bar_cells,
                        "orientation": orientation,
                        "new_pos": exit_cell,
                        "new_dir": new_direction,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _build_tail(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: int,
    blocked: set[tuple[int, int]],
    reserved_sources: set[tuple[int, int]],
    local_used: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = []
    r, c = current
    dr, dc = DIRS[direction]
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        cell = (r, c)
        if cell in blocked or cell in reserved_sources or cell in local_used:
            return None
        cells.append(cell)
    return cells if cells else None


def _build_ray(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: int,
    pattern: list[str],
    blocked: set[tuple[int, int]],
    reserved_sources: set[tuple[int, int]],
) -> dict | None:
    for _ in range(RAY_ATTEMPTS):
        local_used = {source}
        bar_specs: list[dict] = []
        path_cells = {source}
        current = source
        current_dir = direction
        ok = True

        for event_i, event in enumerate(pattern):
            remaining = len(pattern) - event_i - 1
            if event == "parallel":
                candidates = _parallel_candidates(
                    shape, current, current_dir, blocked, reserved_sources, local_used, remaining
                )
            else:
                candidates = _perpendicular_candidates(
                    shape, current, current_dir, blocked, reserved_sources, local_used, remaining
                )
            if not candidates:
                ok = False
                break
            choice = random.choice(candidates)
            for cell in choice["path_cells"]:
                path_cells.add(cell)
                local_used.add(cell)
            for cell in choice["bar_cells"]:
                local_used.add(cell)
            bar_specs.append({"cells": choice["bar_cells"], "orientation": choice["orientation"]})
            current = choice["new_pos"]
            current_dir = choice["new_dir"]

        if not ok:
            continue

        tail = _build_tail(shape, current, current_dir, blocked, reserved_sources, local_used)
        if tail is None:
            continue
        for cell in tail:
            path_cells.add(cell)
        if len(path_cells) < 6:
            continue
        return {"source": source, "path": path_cells, "bars": bar_specs}
    return None


def _render_scene(
    shape: tuple[int, int],
    edge: str,
    sources: list[tuple[int, int]],
    rays: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for source in sources:
        grid[source] = GREY

    all_bars = [bar for ray in rays for bar in ray["bars"]]
    colors = random.sample(BAR_COLORS, len(all_bars))
    for bar, color in zip(all_bars, colors):
        for r, c in bar["cells"]:
            grid[r, c] = color

    detected_edge, detected_sources = _detect_sources(grid)
    assert detected_edge == edge
    assert detected_sources == sorted(sources, key=lambda cell: _source_sort_key(edge, cell))
    return grid


def _attempt_scene() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    shape = (h, w)
    edge = random.choice(["top", "right", "bottom", "left"])
    num_sources = random.randint(*NUM_SOURCES_RANGE)
    edge_length = w if edge in ("top", "bottom") else h
    positions = _sample_edge_positions(edge_length, num_sources)
    if positions is None:
        return None

    sources = [_source_cell(edge, pos, shape) for pos in positions]
    patterns = _sample_patterns(num_sources)
    if patterns is None:
        return None

    ray_specs = list(zip(sources, patterns))
    ray_specs.sort(key=lambda item: (-len(item[1]), _source_sort_key(edge, item[0])))

    direction = _dir_from_edge(edge)
    source_set = set(sources)
    blocked: set[tuple[int, int]] = set()
    rays = []
    branch_counts = {"parallel": 0, "turn": 0}

    for source, pattern in ray_specs:
        ray = _build_ray(shape, source, direction, pattern, blocked, source_set - {source})
        if ray is None:
            return None
        rays.append(ray)
        occupied = set(ray["path"])
        for bar in ray["bars"]:
            occupied.update(bar["cells"])
        blocked |= _dilate(occupied, shape)
        for event in pattern:
            branch_counts[event] += 1

    if branch_counts["parallel"] == 0 or branch_counts["turn"] == 0:
        return None

    grid = _render_scene(shape, edge, sources, rays)
    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out != grid):
        return None
    if np.count_nonzero(out == GREY) <= len(sources):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        if h < 14 or w < 12:
            continue

        grid = np.full((h, w), BLACK, dtype=int)
        source_col = random.randint(5, w - 5)
        grid[0, source_col] = GREY

        v_len = random.randint(2, 4)
        v_start = random.randint(3, max(3, h // 3))
        v_end = v_start + v_len - 1
        h_row = random.randint(v_end + 3, h - 4)
        left_span = random.randint(1, min(4, source_col - 2))
        right_span = random.randint(1, min(4, w - source_col - 3))

        colors = random.sample(BAR_COLORS, 2)
        grid[v_start : v_end + 1, source_col] = colors[0]
        grid[h_row, source_col - left_span : source_col + right_span + 1] = colors[1]

        try:
            out = main(grid)
        except Exception:
            continue
        if np.any(out != grid) and np.count_nonzero(out == GREY) > 4:
            return grid
    raise RuntimeError("failed to generate a valid ray-lens ARC sample")

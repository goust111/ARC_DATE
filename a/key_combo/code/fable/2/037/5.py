from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]

BG = 0
DIVIDER = 5
WALL = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _bfs_distance(passable: np.ndarray, start: Coord, goal: Coord) -> int:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[Coord] = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return int(dist[r, c])
        nd = int(dist[r, c]) + 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    raise AssertionError("endpoints must be connected")


def _detect_divider(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    row_candidates = [
        ("row", r, int(g[r, 0]))
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r, :] == g[r, 0])
    ]
    col_candidates = [
        ("col", c, int(g[0, c]))
        for c in range(g.shape[1])
        if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])
    ]
    candidates = row_candidates + col_candidates
    assert len(candidates) == 1, "expected exactly one non-background full divider line"
    kind, index, color = candidates[0]
    if kind == "row":
        assert 0 < index < g.shape[0] - 1
        return {
            "orientation": "horizontal",
            "index": int(index),
            "color": int(color),
            "slices": [
                (slice(0, index), slice(None)),
                (slice(index + 1, None), slice(None)),
            ],
        }
    assert 0 < index < g.shape[1] - 1
    return {
        "orientation": "vertical",
        "index": int(index),
        "color": int(color),
        "slices": [
            (slice(None), slice(0, index)),
            (slice(None), slice(index + 1, None)),
        ],
    }


def _extract_strip_info(panel: np.ndarray, bg: int) -> dict:
    coords = np.argwhere(panel != bg)
    assert coords.size > 0, "strip panel must contain foreground"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = int(coords.shape[0])
    if int(r0) == int(r1):
        segment = panel[int(r0), int(c0) : int(c1) + 1]
        assert area == int(segment.size), "strip must be one cell thick"
        assert np.all(segment != bg), "strip prefix must be contiguous"
        if int(c0) == 0 and int(c1) < panel.shape[1] - 1:
            direction = 1
            prefix = segment.copy()
            start = (int(r0), 0)
        elif int(c1) == panel.shape[1] - 1 and int(c0) > 0:
            direction = -1
            prefix = segment[::-1].copy()
            start = (int(r0), panel.shape[1] - 1)
        else:
            raise AssertionError("horizontal strip must start from exactly one border")
        return {
            "orientation": "horizontal",
            "start": start,
            "step": (0, int(direction)),
            "prefix": [int(v) for v in prefix.tolist()],
            "axis_len": int(panel.shape[1]),
        }
    if int(c0) == int(c1):
        segment = panel[int(r0) : int(r1) + 1, int(c0)]
        assert area == int(segment.size), "strip must be one cell thick"
        assert np.all(segment != bg), "strip prefix must be contiguous"
        if int(r0) == 0 and int(r1) < panel.shape[0] - 1:
            direction = 1
            prefix = segment.copy()
            start = (0, int(c0))
        elif int(r1) == panel.shape[0] - 1 and int(r0) > 0:
            direction = -1
            prefix = segment[::-1].copy()
            start = (panel.shape[0] - 1, int(c0))
        else:
            raise AssertionError("vertical strip must start from exactly one border")
        return {
            "orientation": "vertical",
            "start": start,
            "step": (int(direction), 0),
            "prefix": [int(v) for v in prefix.tolist()],
            "axis_len": int(panel.shape[0]),
        }
    raise AssertionError("strip foreground must lie on a single row or column")


def _extract_maze_info(panel: np.ndarray, bg: int) -> dict:
    colors = sorted(int(v) for v in np.unique(panel) if int(v) != bg)
    endpoint_color = None
    endpoints: list[Coord] = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(panel == color)]
        if len(coords) != 2:
            continue
        if all(sum(int(panel[nr, nc] == color) for nr, nc in neighbors4(r, c) if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1]) == 0 for r, c in coords):
            endpoint_color = int(color)
            endpoints = coords
            break
    assert endpoint_color is not None and len(endpoints) == 2, "maze panel must contain exactly two isolated same-color endpoints"
    wall_colors = [color for color in colors if color != endpoint_color]
    assert len(wall_colors) == 1, "maze panel must contain exactly one wall color"
    wall = int(wall_colors[0])
    passable = panel != wall
    dist = _bfs_distance(passable, endpoints[0], endpoints[1])
    length = int(dist + 1)
    assert length > _manhattan(endpoints[0], endpoints[1]) + 1, "maze path must require a detour"
    return {
        "wall": wall,
        "endpoint_color": endpoint_color,
        "endpoints": endpoints,
        "path_length": length,
    }


def _minimal_period(seq: list[int]) -> list[int]:
    n = len(seq)
    for p in range(1, n):
        if n < 2 * p:
            continue
        if all(int(seq[i]) == int(seq[i % p]) for i in range(n)):
            return [int(v) for v in seq[:p]]
    raise AssertionError("strip prefix must contain at least two full repeats of a primitive period")


def _repeat_to_length(pattern: list[int], length: int) -> list[int]:
    return [int(pattern[i % len(pattern)]) for i in range(length)]


def _completed_strip_panel(panel: np.ndarray, strip: dict, total_len: int) -> np.ndarray:
    out = np.array(panel, dtype=int, copy=True)
    sr, sc = strip["start"]
    dr, dc = strip["step"]
    pattern = _minimal_period(strip["prefix"])
    colors = _repeat_to_length(pattern, total_len)
    for k, color in enumerate(colors):
        r = int(sr + dr * k)
        c = int(sc + dc * k)
        assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "extended strip must stay in bounds"
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the geodesic distance between two maze endpoints as the exact span for completing a periodic strip in the other panel.

    Input structure:
    - A single full divider line splits the grid into two rectangular panels.
    - One panel is a maze with background cells, one wall color, and exactly two isolated same-color endpoint pixels.
    - The other panel contains one straight one-cell-thick periodic strip, drawn only for its initial repeated prefix.

    Rule:
    - Keep the divider and the maze panel unchanged.
    - Measure the 4-connected shortest path between the maze endpoints through all non-wall cells, counting both endpoints.
    - Detect the strip's primitive period and continue tiling along the same direction until its total occupied length equals that maze path length, truncating inside the last period if needed.

    Key insight:
    - The strip's final length is not determined inside its own panel; it is transmitted from the other panel by a shortest-path count through the maze.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    divider = _detect_divider(g)
    panels = [g[rs, cs].copy() for rs, cs in divider["slices"]]

    strip_index = None
    strip_info = None
    for idx, panel in enumerate(panels):
        try:
            candidate = _extract_strip_info(panel, bg)
            strip_index = idx
            strip_info = candidate
            break
        except AssertionError:
            continue
    assert strip_index is not None and strip_info is not None, "one panel must be the periodic strip"
    maze_index = 1 - strip_index
    maze_info = _extract_maze_info(panels[maze_index], bg)
    total_len = int(maze_info["path_length"])
    assert len(strip_info["prefix"]) < total_len < strip_info["axis_len"], "target strip length must extend the prefix without filling the whole axis"

    out = np.array(g, dtype=int, copy=True)
    rs, cs = divider["slices"][strip_index]
    out[rs, cs] = _completed_strip_panel(panels[strip_index], strip_info, total_len)
    return out


def _vertical_maze_candidates(height: int, width: int, target_cap: int) -> list[tuple[int, int, int, bool]]:
    max_walls = max(2, (width - 1) // 2)
    candidates: list[tuple[int, int, int, bool]] = []
    for top_gap in range(1, height - 2):
        for bottom_gap in range(top_gap + 2, height - 1):
            d = bottom_gap - top_gap
            for n_walls in range(2, max_walls + 1):
                target = width + (n_walls - 1) * d
                if 10 <= target <= target_cap:
                    for start_top in (False, True):
                        candidates.append((top_gap, bottom_gap, n_walls, start_top))
    return candidates


def _horizontal_maze_candidates(height: int, width: int, target_cap: int) -> list[tuple[int, int, int, bool]]:
    max_walls = max(2, (height - 1) // 2)
    candidates: list[tuple[int, int, int, bool]] = []
    for left_gap in range(1, width - 2):
        for right_gap in range(left_gap + 2, width - 1):
            d = right_gap - left_gap
            for n_walls in range(2, max_walls + 1):
                target = height + (n_walls - 1) * d
                if 10 <= target <= target_cap:
                    for start_left in (False, True):
                        candidates.append((left_gap, right_gap, n_walls, start_left))
    return candidates


def _spaced_positions(low: int, high: int, count: int) -> list[int] | None:
    if count <= 0:
        return []
    for _ in range(100):
        picks = sorted(random.sample(range(low, high + 1), count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:], strict=False)):
            return picks
    return None


def _build_vertical_maze(height: int, width: int, endpoint_color: int, target_cap: int) -> tuple[np.ndarray, int] | None:
    candidates = _vertical_maze_candidates(height, width, target_cap)
    random.shuffle(candidates)
    for top_gap, bottom_gap, n_walls, start_top in candidates:
        cols = _spaced_positions(1, width - 2, n_walls)
        if cols is None:
            continue
        panel = np.full((height, width), BG, dtype=int)
        gap_rows = [top_gap if (i % 2 == 0) == start_top else bottom_gap for i in range(n_walls)]
        for col, gap_row in zip(cols, gap_rows, strict=True):
            panel[:, col] = WALL
            panel[gap_row, col] = BG
        start = (gap_rows[0], 0)
        end = (gap_rows[-1], width - 1)
        panel[start] = endpoint_color
        panel[end] = endpoint_color
        dist = _bfs_distance(panel != WALL, start, end)
        length = int(dist + 1)
        if length >= target_cap or length <= width:
            continue
        if length <= _manhattan(start, end) + 1:
            continue
        return panel, length
    return None


def _build_horizontal_maze(height: int, width: int, endpoint_color: int, target_cap: int) -> tuple[np.ndarray, int] | None:
    candidates = _horizontal_maze_candidates(height, width, target_cap)
    random.shuffle(candidates)
    for left_gap, right_gap, n_walls, start_left in candidates:
        rows = _spaced_positions(1, height - 2, n_walls)
        if rows is None:
            continue
        panel = np.full((height, width), BG, dtype=int)
        gap_cols = [left_gap if (i % 2 == 0) == start_left else right_gap for i in range(n_walls)]
        for row, gap_col in zip(rows, gap_cols, strict=True):
            panel[row, :] = WALL
            panel[row, gap_col] = BG
        start = (0, gap_cols[0])
        end = (height - 1, gap_cols[-1])
        panel[start] = endpoint_color
        panel[end] = endpoint_color
        dist = _bfs_distance(panel != WALL, start, end)
        length = int(dist + 1)
        if length >= target_cap or length <= height:
            continue
        if length <= _manhattan(start, end) + 1:
            continue
        return panel, length
    return None


def _choose_pattern(target_len: int, available_colors: list[int], want_exact: bool) -> tuple[list[int], int]:
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    max_period = min(4, len(available_colors), target_len - 1)
    for period_len in range(2, max_period + 1):
        if target_len <= 2 * period_len:
            continue
        max_cycles = min(3, (target_len - 1) // period_len)
        for cycles in (2, 3):
            if cycles > max_cycles:
                continue
            rec = (period_len, cycles)
            if (target_len % period_len == 0) == want_exact:
                preferred.append(rec)
            fallback.append(rec)
    options = preferred or fallback
    assert options, "target length must allow at least two full visible periods"
    period_len, cycles = random.choice(options)
    pattern = random.sample(available_colors, period_len)
    return pattern, cycles * period_len


def _build_strip_panel(
    height: int,
    width: int,
    orientation: str,
    direction: int,
    pattern: list[int],
    prefix_len: int,
) -> np.ndarray:
    panel = np.full((height, width), BG, dtype=int)
    prefix = _repeat_to_length(pattern, prefix_len)
    if orientation == "vertical":
        col = random.randint(1, width - 2)
        if direction > 0:
            for k, color in enumerate(prefix):
                panel[k, col] = int(color)
        else:
            for k, color in enumerate(prefix):
                panel[height - 1 - k, col] = int(color)
        return panel
    row = random.randint(1, height - 2)
    if direction > 0:
        for k, color in enumerate(prefix):
            panel[row, k] = int(color)
    else:
        for k, color in enumerate(prefix):
            panel[row, width - 1 - k] = int(color)
    return panel


def _assemble_grid(
    divider_orientation: str,
    maze_first: bool,
    maze_panel: np.ndarray,
    strip_panel: np.ndarray,
) -> tuple[np.ndarray, tuple[slice, slice]]:
    if divider_orientation == "vertical":
        h = maze_panel.shape[0]
        w0, w1 = maze_panel.shape[1], strip_panel.shape[1]
        grid = np.full((h, w0 + w1 + 1), BG, dtype=int)
        if maze_first:
            grid[:, :w0] = maze_panel
            grid[:, w0] = DIVIDER
            grid[:, w0 + 1 :] = strip_panel
            return grid, (slice(None), slice(w0 + 1, None))
        grid[:, :w1] = strip_panel
        grid[:, w1] = DIVIDER
        grid[:, w1 + 1 :] = maze_panel
        return grid, (slice(None), slice(0, w1))

    w = maze_panel.shape[1]
    h0, h1 = maze_panel.shape[0], strip_panel.shape[0]
    grid = np.full((h0 + h1 + 1, w), BG, dtype=int)
    if maze_first:
        grid[:h0, :] = maze_panel
        grid[h0, :] = DIVIDER
        grid[h0 + 1 :, :] = strip_panel
        return grid, (slice(h0 + 1, None), slice(None))
    grid[:h1, :] = strip_panel
    grid[h1, :] = DIVIDER
    grid[h1 + 1 :, :] = maze_panel
    return grid, (slice(0, h1), slice(None))


def _build_case(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    want_exact = (mode % 3) == 0
    endpoint_color = random.choice(PALETTE)
    strip_direction = 1 if mode % 2 == 0 else -1

    if mode % 4 in (0, 2):
        height = random.randint(15, 22)
        maze_width = random.randint(8, 11)
        strip_width = random.randint(8, 11)
        maze_first = mode % 4 == 0
        maze_built = _build_vertical_maze(height, maze_width, endpoint_color, target_cap=height - 1)
        if maze_built is None:
            return None
        maze_panel, target_len = maze_built
        available_colors = [c for c in PALETTE if c != endpoint_color]
        pattern, prefix_len = _choose_pattern(target_len, available_colors, want_exact)
        strip_panel = _build_strip_panel(height, strip_width, "vertical", strip_direction, pattern, prefix_len)
        grid, strip_slice = _assemble_grid("vertical", maze_first, maze_panel, strip_panel)
    else:
        width = random.randint(15, 22)
        maze_height = random.randint(8, 11)
        strip_height = random.randint(8, 11)
        maze_first = mode % 4 == 1
        maze_built = _build_horizontal_maze(maze_height, width, endpoint_color, target_cap=width - 1)
        if maze_built is None:
            return None
        maze_panel, target_len = maze_built
        available_colors = [c for c in PALETTE if c != endpoint_color]
        pattern, prefix_len = _choose_pattern(target_len, available_colors, want_exact)
        strip_panel = _build_strip_panel(strip_height, width, "horizontal", strip_direction, pattern, prefix_len)
        grid, strip_slice = _assemble_grid("horizontal", maze_first, maze_panel, strip_panel)

    expected = np.array(grid, dtype=int, copy=True)
    expected[strip_slice] = _completed_strip_panel(expected[strip_slice], _extract_strip_info(expected[strip_slice], BG), target_len)
    if get_bg(grid) != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    divider = _detect_divider(grid)
    panels = [grid[rs, cs].copy() for rs, cs in divider["slices"]]
    strip_info = None
    maze_info = None
    for panel in panels:
        try:
            strip_info = _extract_strip_info(panel, BG)
        except AssertionError:
            maze_info = _extract_maze_info(panel, BG)
    assert strip_info is not None and maze_info is not None
    period = _minimal_period(strip_info["prefix"])
    start, end = maze_info["endpoints"]
    assert len(strip_info["prefix"]) == prefix_len
    assert prefix_len >= 2 * len(period)
    assert prefix_len < target_len < strip_info["axis_len"]
    assert target_len == maze_info["path_length"]
    assert target_len > _manhattan(start, end) + 1
    divisors = [p for p in range(2, min(4, target_len - 1) + 1) if target_len % p == 0]
    if want_exact and divisors:
        assert target_len % len(period) == 0
    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = _GEN_COUNTER % 4
        _GEN_COUNTER += 1
        built = _build_case(mode)
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid maze-span tiling input")

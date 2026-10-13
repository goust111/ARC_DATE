from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == RED)
    assert starts.shape == (1, 2), "expected one red border marker"
    marker_r, marker_c = map(int, starts[0])
    assert marker_c == 0, "marker must be on the left border"

    bars: list[dict[str, int]] = []
    for color in (BLUE, GREEN):
        for comp in _components(g == color):
            rows = np.unique(comp[:, 0])
            assert len(rows) == 1, "each obstacle must be horizontal"
            cols = sorted(map(int, comp[:, 1]))
            start, end = cols[0], cols[-1]
            assert cols == list(range(start, end + 1)), "obstacle must be a solid bar"
            assert 2 <= len(cols) <= 4, "obstacle length must be 2..4"
            bars.append(
                {
                    "color": int(color),
                    "row": int(rows[0]),
                    "start": start,
                    "end": end,
                }
            )

    bars.sort(key=lambda bar: (bar["start"], bar["row"], bar["color"]))
    for i, bar in enumerate(bars):
        assert 0 < bar["row"] < g.shape[0] - 1, "bars must not touch top/bottom border"
        if i:
            assert bar["start"] > bars[i - 1]["end"] + 1, "bars must be separated"
    return marker_r, bars


def _paint_hline(out: np.ndarray, row: int, c0: int, c1: int) -> None:
    step = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step, step):
        if out[row, c] in (BLACK, RED):
            out[row, c] = RED


def _trace_path(grid: np.ndarray, mode: str = "normal") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker_r, bars = _extract_scene(g)
    out = g.copy()
    cur_c = 0

    for bar in bars:
        color = bar["color"]
        bar_r = bar["row"]
        start = bar["start"]
        end = bar["end"]
        assert start >= cur_c, "bars must be ordered left to right"

        _paint_hline(out, marker_r, cur_c, start - 1)

        if mode == "all_up":
            detour_r = marker_r - 1
        elif mode == "all_down":
            detour_r = marker_r + 1
        else:
            detour_r = marker_r - 1 if color == BLUE else marker_r + 1

        assert 0 <= detour_r < h
        _paint_hline(out, detour_r, start - 1, end + 1)
        if out[marker_r, end + 1] in (BLACK, RED):
            out[marker_r, end + 1] = RED
        cur_c = end + 1

    _paint_hline(out, marker_r, cur_c, w - 1)
    return out


def _path_rows_by_col(path_grid: np.ndarray) -> dict[int, tuple[int, ...]]:
    h, w = path_grid.shape
    profile: dict[int, tuple[int, ...]] = {}
    for c in range(w):
        rows = []
        for r in range(h):
            if path_grid[r, c] != RED:
                continue
            has_horizontal_neighbor = (
                (c > 0 and path_grid[r, c - 1] == RED)
                or (c + 1 < w and path_grid[r, c + 1] == RED)
            )
            if has_horizontal_neighbor:
                rows.append(r)
        assert rows, "path must have a horizontal trace in every column"
        profile[c] = tuple(sorted(rows))
    return profile


def _enclosed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    regions: list[np.ndarray] = []
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                regions.append(np.array(coords, dtype=int))
    return regions


def _fill_regions(path_grid: np.ndarray) -> np.ndarray:
    out = np.asarray(path_grid, dtype=int).copy()
    profile = _path_rows_by_col(out)
    for region in _enclosed_black_regions(out):
        above = all(int(r) < min(profile[int(c)]) for r, c in region)
        below = all(int(r) > max(profile[int(c)]) for r, c in region)
        assert above ^ below, "enclosed region must lie on exactly one side of the path"
        fill_color = YELLOW if above else ORANGE
        out[region[:, 0], region[:, 1]] = fill_color
    return out


def _solve(grid: np.ndarray, *, mode: str = "normal") -> np.ndarray:
    path_grid = _trace_path(grid, mode=mode)
    return _fill_regions(path_grid)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a red path from the left marker, detour around colored bars, then fill closed regions by whether they lie above or below the path.

    Input structure:
    - The grid has a black background, one red marker on the left border, and several blue/green horizontal bars.
    - Bars are one cell thick, two to four cells long, and stay away from the top and bottom borders.
    - The bars sit near the marker row so the detoured path, the bars, and the outer boundary can create closed black pockets.

    Rule:
    - Start from the red marker and draw a red path to the right border along the marker row.
    - Before each bar span, leave the marker row and pass around it: blue bars force an upward bulge, green bars force a downward bulge; then return to the marker row and continue right.
    - Bars keep their original colors. After the full path is drawn, find every 4-connected black region enclosed by the red path, the obstacle bars, and the grid border.
    - Fill each enclosed region yellow if all of its cells are above the path row in their columns, or orange if all of its cells are below the path row. Black regions still connected to the outer border stay black.

    Key insight:
    - The obstacle colors matter only because they choose the path's detour side; that detoured shape then determines which black pockets become enclosed and whether each pocket lies above or below the red trace.
    """
    return _solve(np.asarray(grid, dtype=int), mode="normal")


def _sample_spans(width: int, count: int) -> list[tuple[int, int]] | None:
    lengths = [random.randint(2, 4) for _ in range(count)]
    usable = width - 4
    min_total = sum(lengths) + 2 * (count - 1)
    if min_total > usable:
        return None

    extras = [0] * (count + 1)
    for _ in range(usable - min_total):
        extras[random.randrange(len(extras))] += 1

    spans = []
    start = 2 + extras[0]
    for i, length in enumerate(lengths):
        end = start + length - 1
        spans.append((start, end))
        if i < count - 1:
            start = end + 1 + 2 + extras[i + 1]
    if spans[-1][1] > width - 3:
        return None
    return spans


def _sample_colors(count: int) -> list[int]:
    blue_count = random.randint(1, count - 1)
    colors = [BLUE] * blue_count + [GREEN] * (count - blue_count)
    random.shuffle(colors)
    return colors


def _build_input(height: int, width: int, marker_r: int, spans: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[marker_r, 0] = RED
    for (start, end), color in zip(spans, colors):
        row = marker_r + 1 if color == BLUE else marker_r - 1
        grid[row, start : end + 1] = color
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    marker_r, bars = _extract_scene(inp)
    if not bars:
        return False
    for bar in bars:
        expected_row = marker_r + 1 if bar["color"] == BLUE else marker_r - 1
        if bar["row"] != expected_row:
            return False
    if not any(bar["color"] == BLUE for bar in bars):
        return False
    if not any(bar["color"] == GREEN for bar in bars):
        return False

    path_grid = _trace_path(inp)
    profile = _path_rows_by_col(path_grid)
    enclosed = _enclosed_black_regions(path_grid)
    if len(enclosed) < 2:
        return False

    side_colors = set()
    for region in enclosed:
        above = all(int(r) < min(profile[int(c)]) for r, c in region)
        below = all(int(r) > max(profile[int(c)]) for r, c in region)
        if above == below:
            return False
        side_colors.add(YELLOW if above else ORANGE)
    if side_colors != {YELLOW, ORANGE}:
        return False

    out = _fill_regions(path_grid)
    if np.array_equal(out, path_grid):
        return False

    swapped = inp.copy()
    swapped[inp == BLUE] = GREEN
    swapped[inp == GREEN] = BLUE
    if np.array_equal(path_grid, _trace_path(swapped)):
        return False

    all_up = _trace_path(inp, mode="all_up")
    all_down = _trace_path(inp, mode="all_down")
    if np.array_equal(path_grid, all_up) or np.array_equal(path_grid, all_down):
        return False

    orange_count = int(np.sum(out == ORANGE))
    yellow_count = int(np.sum(out == YELLOW))
    return orange_count > 0 and yellow_count > 0


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.randint(11, 18)
        width = random.randint(15, 24)
        marker_r = random.randint(3, height - 4)
        bar_count = random.randint(3, 5)
        spans = _sample_spans(width, bar_count)
        if spans is None:
            continue
        colors = _sample_colors(bar_count)
        inp = _build_input(height, width, marker_r, spans, colors)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid input")

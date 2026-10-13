from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import colors_present, get_bg


PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400
AXIS_RANGE = (12, 19)
CROSS_RANGE = (10, 18)
EXTRA_PATTERN_RANGE = (0, 2)


def _full_row_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    bars: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        values = set(map(int, grid[r, :]))
        if len(values) == 1:
            color = values.pop()
            if color != bg:
                bars.append((r, color))
    return bars


def _full_col_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    bars: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        values = set(map(int, grid[:, c]))
        if len(values) == 1:
            color = values.pop()
            if color != bg:
                bars.append((c, color))
    return bars


def _detect_layout(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    height, width = grid.shape
    row_bars = sorted(_full_row_bars(grid, bg))
    col_bars = sorted(_full_col_bars(grid, bg))

    if len(row_bars) == 2 and not col_bars:
        assert [r for r, _ in row_bars] == [0, height - 1]
        assert row_bars[0][1] != row_bars[1][1]
        return {
            "bg": bg,
            "axis": 0,
            "edges": (0, height - 1),
            "bar_colors": (int(row_bars[0][1]), int(row_bars[1][1])),
        }

    assert len(col_bars) == 2 and not row_bars
    assert [c for c, _ in col_bars] == [0, width - 1]
    assert col_bars[0][1] != col_bars[1][1]
    return {
        "bg": bg,
        "axis": 1,
        "edges": (0, width - 1),
        "bar_colors": (int(col_bars[0][1]), int(col_bars[1][1])),
    }


def _marker_color(grid: np.ndarray, bg: int, bar_colors: tuple[int, int]) -> int:
    marker_colors = colors_present(grid, exclude={bg, *bar_colors})
    assert len(marker_colors) == 1
    return int(marker_colors[0])


def _blocked(grid: np.ndarray, marker: int, axis: int, edge: int, r: int, c: int) -> bool:
    if axis == 0:
        lo = min(edge, r) + 1
        hi = max(edge, r)
        if hi <= lo:
            return False
        return bool(np.any(grid[lo:hi, c] == marker))

    lo = min(edge, c) + 1
    hi = max(edge, c)
    if hi <= lo:
        return False
    return bool(np.any(grid[r, lo:hi] == marker))


def _path_cells(axis: int, edge: int, r: int, c: int) -> list[tuple[int, int]]:
    if axis == 0:
        if edge < r:
            return [(rr, c) for rr in range(edge + 1, r + 1)]
        return [(rr, c) for rr in range(r, edge)]

    if edge < c:
        return [(r, cc) for cc in range(edge + 1, c + 1)]
    return [(r, cc) for cc in range(c, edge)]


def _route_markers(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    layout = _detect_layout(g)
    bg = int(layout["bg"])
    axis = int(layout["axis"])
    edges = tuple(map(int, layout["edges"]))
    bar_colors = tuple(map(int, layout["bar_colors"]))
    marker = _marker_color(g, bg, bar_colors)

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == marker)]
    planned: dict[tuple[int, int], int] = {}
    routes: list[dict] = []

    for r, c in markers:
        coord = r if axis == 0 else c
        dist_a = coord - edges[0]
        dist_b = edges[1] - coord
        assert dist_a != dist_b
        clear_a = not _blocked(g, marker, axis, edges[0], r, c)
        clear_b = not _blocked(g, marker, axis, edges[1], r, c)

        preferred = 0 if dist_a < dist_b else 1
        if preferred == 0:
            chosen = 0 if clear_a else (1 if clear_b else None)
        else:
            chosen = 1 if clear_b else (0 if clear_a else None)

        route = {
            "marker": (r, c),
            "preferred": preferred,
            "clear": (clear_a, clear_b),
            "chosen": chosen,
        }
        routes.append(route)

        if chosen is None:
            continue

        for cell in _path_cells(axis, edges[chosen], r, c):
            previous = planned.get(cell)
            color = bar_colors[chosen]
            assert previous is None or previous == color
            planned[cell] = color

    out = g.copy()
    out[g == marker] = bg
    for (r, c), color in planned.items():
        out[r, c] = int(color)
    return out, routes


def _branch_summary(routes: list[dict]) -> dict[str, int]:
    summary = {
        "near_a": 0,
        "near_b": 0,
        "fallback_a_to_b": 0,
        "fallback_b_to_a": 0,
        "erase_a": 0,
        "erase_b": 0,
    }
    for route in routes:
        preferred = int(route["preferred"])
        chosen = route["chosen"]
        if chosen is None:
            key = "erase_a" if preferred == 0 else "erase_b"
        elif chosen == preferred:
            key = "near_a" if chosen == 0 else "near_b"
        elif preferred == 0:
            key = "fallback_a_to_b"
        else:
            key = "fallback_b_to_a"
        summary[key] += 1
    return summary


def main(grid: np.ndarray) -> np.ndarray:
    """Route each control pixel to the nearer edge bar unless a same-color blocker forces fallback or erasure.

    Input structure:
    - The grid has a dominant background plus two full opposite edge bars, either on the top/bottom rows or the left/right columns.
    - The two bars have different colors, and every remaining foreground cell is a control pixel of one shared marker color.
    - Control pixels lie strictly between the bars, and each one is at unequal axial distances from the two bars.

    Rule:
    - For each control pixel, inspect only the input marker distribution along the bar-to-bar axis on its own row or column.
    - Prefer the nearer bar: if that straight path contains no other control pixel, paint the entire path from the control pixel up to that bar with the nearer bar's color.
    - If the nearer side is blocked, instead paint the path to the farther bar with the farther bar's color; if both sides are blocked, erase that control pixel to background. The edge bars themselves stay unchanged.

    Key insight:
    - Distance only picks the first attempt; the real decision is a static blocked-ray test on the input markers, and the final path color always comes from the bar that is actually reached.
    """
    out, _routes = _route_markers(grid)
    return out


def _available_side_positions(axis_len: int, side: int) -> list[int]:
    positions: list[int] = []
    for pos in range(1, axis_len - 1):
        dist_a = pos
        dist_b = axis_len - 1 - pos
        if dist_a == dist_b:
            continue
        if side == 0 and dist_a < dist_b:
            positions.append(pos)
        if side == 1 and dist_b < dist_a:
            positions.append(pos)
    return positions


def _pattern_positions(axis_len: int, pattern: str) -> list[int] | None:
    side_a = _available_side_positions(axis_len, 0)
    side_b = _available_side_positions(axis_len, 1)

    if pattern == "near_a":
        return [random.choice(side_a)]
    if pattern == "near_b":
        return [random.choice(side_b)]

    if pattern == "fallback_a_to_b":
        targets = [pos for pos in side_a if pos >= 3]
        random.shuffle(targets)
        for target in targets:
            blockers = [pos for pos in side_a if pos <= target - 2]
            if blockers:
                return sorted([random.choice(blockers), target])
        return None

    if pattern == "fallback_b_to_a":
        targets = [pos for pos in side_b if pos <= axis_len - 4]
        random.shuffle(targets)
        for target in targets:
            blockers = [pos for pos in side_b if pos >= target + 2]
            if blockers:
                return sorted([target, random.choice(blockers)])
        return None

    if pattern == "erase_a":
        targets = [pos for pos in side_a if 3 <= pos <= axis_len - 4]
        random.shuffle(targets)
        for target in targets:
            above = [pos for pos in range(1, target - 1) if pos in side_a]
            below = [pos for pos in range(target + 2, axis_len - 1) if pos in side_b or pos in side_a]
            if above and below:
                return sorted([random.choice(above), target, random.choice(below)])
        return None

    if pattern == "erase_b":
        targets = [pos for pos in side_b if 3 <= pos <= axis_len - 4]
        random.shuffle(targets)
        for target in targets:
            below = [pos for pos in range(target + 2, axis_len - 1) if pos in side_b]
            above = [pos for pos in range(1, target - 1) if pos in side_a or pos in side_b]
            if above and below:
                return sorted([random.choice(above), target, random.choice(below)])
        return None

    raise ValueError(f"unknown pattern: {pattern}")


def _pick_lines(cross_len: int, count: int) -> list[int] | None:
    candidates = list(range(1, cross_len - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for line in candidates:
        if all(abs(line - other) >= 2 for other in chosen):
            chosen.append(line)
            if len(chosen) == count:
                return chosen
    return None


def _paint_markers(grid: np.ndarray, axis: int, line: int, positions: list[int], marker: int) -> None:
    for pos in positions:
        if axis == 0:
            grid[pos, line] = marker
        else:
            grid[line, pos] = marker


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice((0, 1))
    axis_len = random.randint(*AXIS_RANGE)
    cross_len = random.randint(*CROSS_RANGE)
    height, width = (axis_len, cross_len) if axis == 0 else (cross_len, axis_len)

    bg, bar_a, bar_b, marker = random.sample(PALETTE, 4)
    grid = np.full((height, width), bg, dtype=int)
    if axis == 0:
        grid[0, :] = bar_a
        grid[height - 1, :] = bar_b
    else:
        grid[:, 0] = bar_a
        grid[:, width - 1] = bar_b

    required = [
        "near_a",
        "near_b",
        random.choice(("fallback_a_to_b", "fallback_b_to_a")),
        random.choice(("erase_a", "erase_b")),
    ]
    optional_pool = [
        "near_a",
        "near_b",
        "fallback_a_to_b",
        "fallback_b_to_a",
        "erase_a",
        "erase_b",
    ]
    extra = random.randint(*EXTRA_PATTERN_RANGE)
    patterns = required + random.sample(optional_pool, k=extra)
    line_count = len(patterns)

    lines = _pick_lines(cross_len, line_count)
    if lines is None:
        return None

    random.shuffle(patterns)
    for line, pattern in zip(lines, patterns):
        positions = _pattern_positions(axis_len, pattern)
        if positions is None:
            return None
        _paint_markers(grid, axis, line, positions, marker)

    try:
        output, routes = _route_markers(grid)
    except AssertionError:
        return None

    summary = _branch_summary(routes)
    if summary["near_a"] == 0 or summary["near_b"] == 0:
        return None
    if summary["fallback_a_to_b"] + summary["fallback_b_to_a"] == 0:
        return None
    if summary["erase_a"] + summary["erase_b"] == 0:
        return None
    if np.array_equal(grid, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for double-bar fallback routing")

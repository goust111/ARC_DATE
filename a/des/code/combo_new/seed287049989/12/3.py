from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
DIVIDER = 5
BORDER = 0
MARKER = 9
BAR_COLORS = (1, 2, 3, 4, 6, 7)


def _find_divider_col(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(candidates) == 1, "expected exactly one full-height gray divider"
    return int(candidates[0])


def _legend_color_on_start_row(grid: np.ndarray, row: int, divider_col: int) -> int:
    values = {
        int(v)
        for v in grid[row, 1 : divider_col - 1]
        if int(v) not in {BG, DIVIDER, MARKER}
    }
    assert len(values) == 1, "each start row must expose exactly one visible bar color"
    return int(next(iter(values)))


def _reconstruct_bars(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    h, _w = grid.shape
    divider_col = _find_divider_col(grid)
    left_right_marker_col = divider_col - 1

    start_rows = {int(r) for r in np.flatnonzero(grid[:, 0] == MARKER)}
    end_rows = {int(r) for r in np.flatnonzero(grid[:, left_right_marker_col] == MARKER)}
    assert start_rows and len(start_rows) == len(end_rows)

    bars: list[dict[str, int]] = []
    stack: list[tuple[int, int]] = []
    for row in range(h):
        if row in start_rows:
            color = _legend_color_on_start_row(grid, row, divider_col)
            stack.append((row, color))
        if row in end_rows:
            assert stack, "maroon end marker must close the latest open bar"
            start, color = stack.pop()
            bars.append(
                {
                    "start": int(start),
                    "end": int(row),
                    "height": int(row - start + 1),
                    "color": int(color),
                }
            )

    assert not stack, "all started bars must end"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be unique"
    return divider_col, bars


def _components_of_color(grid: np.ndarray, color: int, min_col: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(max(0, min_col), w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and min_col <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _detect_rectangles(grid: np.ndarray, divider_col: int) -> list[dict[str, int]]:
    rectangles: list[dict[str, int]] = []
    for comp in _components_of_color(grid, BORDER, divider_col + 1):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        height = bottom - top + 1
        width = right - left + 1
        assert height >= 3 and width >= 3, "each canvas object must be a closed rectangle"

        border = set()
        for c in range(left, right + 1):
            border.add((top, c))
            border.add((bottom, c))
        for r in range(top, bottom + 1):
            border.add((r, left))
            border.add((r, right))

        comp_set = set(comp)
        assert comp_set == border, "black components on the canvas side must be rectangular frames"
        rectangles.append(
            {
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "height": int(height),
                "width": int(width),
            }
        )

    return rectangles


def _paint_canvas(grid: np.ndarray, bars: list[dict[str, int]], rectangles: list[dict[str, int]]) -> np.ndarray:
    rect_by_height = {rect["height"]: rect for rect in rectangles}
    assert len(rect_by_height) == len(rectangles) == len(bars), "rectangle heights must match bars one-to-one"

    out = grid.copy()
    for bar in sorted(bars, key=lambda item: item["start"]):
        rect = rect_by_height[bar["height"]]
        out[rect["top"] + 1 : rect["bottom"], rect["left"] + 1 : rect["right"]] = bar["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill canvas-side rectangle interiors by reconstructing the hidden layer intervals from the legend-side bars.

    Input structure:
    - Background is Azure, and a solid 1-pixel Gray divider column splits the grid into legend and canvas halves.
    - On the left, each colored horizontal bar spans the legend width except for Maroon endpoint markers: a left-edge marker marks the bar's start row and a right-edge marker marks its end row.
    - Bars may overlap vertically, and when two bars overlap the one whose start marker sits lower is the visible top layer.
    - On the right, Black 1-pixel rectangular frames sit on the Azure background; each frame's outer height matches exactly one legend bar height.

    Rule:
    - Read the divider, then reconstruct every legend bar from its Maroon start and end rows.
    - Use the bar's visible start row to recover its color, and interpret lower start rows as higher layers.
    - Detect the Black rectangular frames on the canvas side and match each one to the unique bar with the same height.
    - Copy the input grid and fill each rectangle interior with its matched bar color, processing bars from lowest layer to highest layer.

    Key insight:
    - The visible legend colors alone are incomplete because overlaps hide parts of lower bars; the Maroon endpoint rows are the real interval encoding, and height is the bridge from legend bars to canvas rectangles.
    """
    g = np.asarray(grid, dtype=int)
    divider_col, bars = _reconstruct_bars(g)
    rectangles = _detect_rectangles(g, divider_col)
    return _paint_canvas(g, bars, rectangles)


def _sample_chain_heights(depth: int, used: set[int], rng: random.Random) -> list[int]:
    available = [height for height in range(3, 11) if height not in used]
    for _ in range(200):
        heights = sorted(rng.sample(available, depth), reverse=True)
        if all(a - b >= 2 for a, b in zip(heights, heights[1:])):
            used.update(heights)
            return heights
    raise RuntimeError("failed to sample unique nested heights")


def _chain_intervals(heights: list[int], top: int, rng: random.Random) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    prev_start = top
    prev_height = heights[0]
    intervals.append((prev_start, prev_start + prev_height - 1))
    for height in heights[1:]:
        max_offset = prev_height - height - 1
        assert max_offset >= 1
        offset = rng.randint(1, max_offset)
        start = prev_start + offset
        intervals.append((start, start + height - 1))
        prev_start = start
        prev_height = height
    return intervals


def _mode_depths(rng: random.Random) -> list[int]:
    mode = rng.choice(("nested", "disjoint", "mixed"))
    if mode == "nested":
        return [rng.randint(2, 4)]
    if mode == "disjoint":
        return [1] * rng.randint(3, 4)
    layouts = (
        [2, 1],
        [1, 2],
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2],
        [3, 1],
        [1, 3],
    )
    return list(rng.choice(layouts))


def _block_gaps(block_count: int, has_only_disjoint: bool, rng: random.Random) -> list[int]:
    if block_count <= 1:
        return []
    if has_only_disjoint and block_count >= 3:
        gaps = [0, rng.randint(1, 2)]
        while len(gaps) < block_count - 1:
            gaps.append(rng.randint(0, 2))
        rng.shuffle(gaps)
        return gaps[: block_count - 1]
    return [rng.randint(0, 2) for _ in range(block_count - 1)]


def _build_bar_specs(rng: random.Random) -> tuple[int, list[dict[str, int]]]:
    for _ in range(400):
        depths = _mode_depths(rng)
        used_heights: set[int] = set()
        chains: list[list[int]] = []
        try:
            for depth in depths:
                chains.append(_sample_chain_heights(depth, used_heights, rng))
        except RuntimeError:
            continue

        gaps = _block_gaps(len(depths), all(depth == 1 for depth in depths), rng)
        top_margin = rng.randint(0, 2)
        bottom_margin = rng.randint(0, 2)

        tops: list[int] = []
        current_top = top_margin
        for index, heights in enumerate(chains):
            tops.append(current_top)
            current_top += heights[0]
            if index < len(gaps):
                current_top += gaps[index]
        total_height = current_top + bottom_margin
        if total_height > 30:
            continue

        intervals: list[tuple[int, int]] = []
        for top, heights in zip(tops, chains):
            intervals.extend(_chain_intervals(heights, top, rng))

        if len(intervals) < 2 or max(end for _start, end in intervals) >= total_height:
            continue

        colors = rng.sample(BAR_COLORS, len(intervals))
        bars = [
            {
                "start": int(start),
                "end": int(end),
                "height": int(end - start + 1),
                "color": int(color),
            }
            for (start, end), color in zip(intervals, colors)
        ]
        if len({bar["start"] for bar in bars}) != len(bars):
            continue
        if len({bar["end"] for bar in bars}) != len(bars):
            continue
        return total_height, bars
    raise RuntimeError("failed to build bar intervals")


def _place_rectangles(
    grid_height: int, bars: list[dict[str, int]], left_width: int, rng: random.Random
) -> tuple[int, list[dict[str, int]]]:
    max_right_width = 30 - left_width - 1
    bar_order = bars[:]
    rng.shuffle(bar_order)

    for _ in range(400):
        widths = [rng.randint(3, 6) for _ in bar_order]
        gaps = [rng.randint(1, 2) for _ in range(max(0, len(bar_order) - 1))]
        left_margin = rng.randint(0, 1)
        right_margin = rng.randint(0, 1)
        used_width = left_margin + right_margin + sum(widths) + sum(gaps)
        if used_width > max_right_width:
            continue

        rectangles: list[dict[str, int]] = []
        col = left_width + 1 + left_margin
        for index, (bar, width) in enumerate(zip(bar_order, widths)):
            rectangles.append(
                {
                    "top": int(bar["start"]),
                    "bottom": int(bar["end"]),
                    "left": int(col),
                    "right": int(col + width - 1),
                    "height": int(bar["height"]),
                    "width": int(width),
                }
            )
            col += width
            if index < len(gaps):
                col += gaps[index]

        total_width = col + right_margin
        if total_width > 30:
            continue
        return total_width, rectangles
    raise RuntimeError("failed to place rectangles")


def _render_input(height: int, width: int, left_width: int, bars: list[dict[str, int]], rectangles: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_width] = DIVIDER

    for bar in sorted(bars, key=lambda item: item["start"]):
        grid[bar["start"] : bar["end"] + 1, 1 : left_width - 1] = bar["color"]

    for bar in bars:
        grid[bar["start"], 0] = MARKER
        grid[bar["end"], left_width - 1] = MARKER

    for rect in rectangles:
        top, bottom = rect["top"], rect["bottom"]
        left, right = rect["left"], rect["right"]
        grid[top, left : right + 1] = BORDER
        grid[bottom, left : right + 1] = BORDER
        grid[top : bottom + 1, left] = BORDER
        grid[top : bottom + 1, right] = BORDER

    return grid


def _expected_output(
    grid: np.ndarray, bars: list[dict[str, int]], rectangles: list[dict[str, int]]
) -> np.ndarray:
    return _paint_canvas(grid, bars, rectangles)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        height, bars = _build_bar_specs(rng)
        left_width = rng.randint(5, 8)
        try:
            width, rectangles = _place_rectangles(height, bars, left_width, rng)
        except RuntimeError:
            continue

        grid = _render_input(height, width, left_width, bars, rectangles)
        expected = _expected_output(grid, bars, rectangles)

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            continue
        if len({bar["height"] for bar in bars}) != len(rectangles):
            continue
        if np.array_equal(grid, expected):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")

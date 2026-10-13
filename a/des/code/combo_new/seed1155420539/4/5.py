from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
COUNT_OPTIONS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3))
PRIORITY_MODES = ("horizontal_top", "vertical_top", "mixed")
LINE_GAP = 4
GEN_RETRIES = 400


def _legend_order(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    assert h >= 2, "expected bottom legend rows"

    bar_cols = [c for c in range(w) if int(grid[h - 2, c]) == int(grid[h - 1, c]) != BLACK]
    assert bar_cols, "expected at least one legend bar"
    if len(bar_cols) > 1:
        assert all(b - a == 2 for a, b in zip(bar_cols, bar_cols[1:])), "legend bars must be separated by one cell"

    legend = [int(grid[h - 1, c]) for c in bar_cols]
    assert len(set(legend)) == len(legend), "legend colors must be unique"

    allowed = set(bar_cols)
    for c in range(w):
        if c in allowed:
            continue
        assert int(grid[h - 2, c]) == BLACK and int(grid[h - 1, c]) == BLACK, "non-legend bottom cells must stay black"

    return legend


def _reconstruct_segments(body: np.ndarray, legend: list[int]) -> dict[int, dict[str, int | str | set[int]]]:
    segments: dict[int, dict[str, int | str | set[int]]] = {}
    for color in legend:
        coords = np.argwhere(body == color)
        assert coords.size > 0, f"legend color {color} must appear in the line area"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            segments[color] = {
                "orient": "h",
                "fixed": int(rows[0]),
                "start": int(cols.min()),
                "end": int(cols.max()),
                "crosses": set(),
            }
        elif len(cols) == 1:
            segments[color] = {
                "orient": "v",
                "fixed": int(cols[0]),
                "start": int(rows.min()),
                "end": int(rows.max()),
                "crosses": set(),
            }
        else:
            raise AssertionError(f"color {color} must belong to one straight horizontal or vertical segment")

    return segments


def _intersections(
    body: np.ndarray,
    segments: dict[int, dict[str, int | str | set[int]]],
) -> list[tuple[int, int, int, int]]:
    horiz = [(color, seg) for color, seg in segments.items() if seg["orient"] == "h"]
    vert = [(color, seg) for color, seg in segments.items() if seg["orient"] == "v"]

    intersections: list[tuple[int, int, int, int]] = []
    for h_color, h_seg in horiz:
        row = int(h_seg["fixed"])
        for v_color, v_seg in vert:
            col = int(v_seg["fixed"])
            if int(h_seg["start"]) <= col <= int(h_seg["end"]) and int(v_seg["start"]) <= row <= int(v_seg["end"]):
                shown = int(body[row, col])
                assert shown in (h_color, v_color), "intersection must display one of the two segment colors"
                intersections.append((row, col, h_color, v_color))
                cast_h = h_seg["crosses"]
                cast_v = v_seg["crosses"]
                assert isinstance(cast_h, set) and isinstance(cast_v, set)
                cast_h.add(col)
                cast_v.add(row)

    assert intersections, "expected at least one horizontal/vertical crossing"
    return intersections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor line crossings using the bottom legend's layer priority and extend the winning color one step onto lower layers.

    Input structure:
    - The background is black.
    - Above the bottom legend are several one-pixel-wide monochrome horizontal or vertical line segments, and each segment uses a unique color.
    - A horizontal and a vertical segment may share one grid cell; that crossing cell currently shows one of the two segment colors.
    - The bottom two rows contain a left-to-right legend of 2x1 vertical color bars, separated by one black cell, that orders the colors from highest to lowest priority.

    Rule:
    - Read the legend from left to right to obtain the color priority order.
    - Reconstruct each visible segment by taking all body pixels of one legend color; those pixels must lie on one row or one column, while hidden crossing cells are inferred from overlaps with perpendicular segments.
    - For every reconstructed horizontal/vertical crossing, recolor the shared cell with the higher-priority segment color.
    - If the other segment has lower priority, recolor the immediately adjacent cell on each side of the crossing along that losing segment, provided such a cell exists before the next crossing or the segment endpoint. All non-crossing cells and the legend stay unchanged.

    Key insight:
    - The line colors themselves do not encode which segment passes over another; only the bottom legend reveals the stacking order, and that order must then be applied both at crossings and on the first hidden-looking pixels beside each losing segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid must have room for lines plus a two-row legend"

    legend = _legend_order(g)
    priority = {color: idx for idx, color in enumerate(legend)}
    body = g[:-2]
    segments = _reconstruct_segments(body, legend)
    intersections = _intersections(body, segments)

    proposals: dict[tuple[int, int], list[int]] = {}

    def propose(r: int, c: int, color: int) -> None:
        proposals.setdefault((r, c), []).append(color)

    for r, c, h_color, v_color in intersections:
        h_seg = segments[h_color]
        v_seg = segments[v_color]
        winner = h_color if priority[h_color] < priority[v_color] else v_color
        propose(r, c, winner)

        if winner != h_color:
            h_crosses = h_seg["crosses"]
            assert isinstance(h_crosses, set)
            for dc in (-1, 1):
                cc = c + dc
                if int(h_seg["start"]) <= cc <= int(h_seg["end"]) and cc not in h_crosses:
                    propose(r, cc, winner)

        if winner != v_color:
            v_crosses = v_seg["crosses"]
            assert isinstance(v_crosses, set)
            for dr in (-1, 1):
                rr = r + dr
                if int(v_seg["start"]) <= rr <= int(v_seg["end"]) and rr not in v_crosses:
                    propose(rr, c, winner)

    out = g.copy()
    for (r, c), colors in proposals.items():
        out[r, c] = min(colors, key=lambda color: priority[color])

    return out


def _sample_spaced_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    candidates = list(range(low, high + 1))
    if len(candidates) < count:
        return None

    for _ in range(100):
        random.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(value - prev) >= gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _mixed_legend(horizontal_colors: list[int], vertical_colors: list[int]) -> list[int] | None:
    pool = horizontal_colors + vertical_colors
    for _ in range(80):
        legend = random.sample(pool, len(pool))
        rank = {color: idx for idx, color in enumerate(legend)}
        saw_h = any(rank[h] < rank[v] for h in horizontal_colors for v in vertical_colors)
        saw_v = any(rank[v] < rank[h] for h in horizontal_colors for v in vertical_colors)
        if saw_h and saw_v:
            return legend
    return None


def _priority_legend(horizontal_colors: list[int], vertical_colors: list[int], mode: str) -> list[int] | None:
    if mode == "horizontal_top":
        return random.sample(horizontal_colors, len(horizontal_colors)) + random.sample(vertical_colors, len(vertical_colors))
    if mode == "vertical_top":
        return random.sample(vertical_colors, len(vertical_colors)) + random.sample(horizontal_colors, len(horizontal_colors))
    assert mode == "mixed"
    return _mixed_legend(horizontal_colors, vertical_colors)


def _draw_scene(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    horizontal_colors: list[int],
    vertical_colors: list[int],
    legend: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for row, color in zip(rows, horizontal_colors):
        start = random.randint(1, min(cols) - 1)
        end = random.randint(max(cols) + 1, width - 2)
        grid[row, start : end + 1] = color

    for col, color in zip(cols, vertical_colors):
        start = random.randint(1, min(rows) - 1)
        end = random.randint(max(rows) + 1, height - 4)
        grid[start : end + 1, col] = color

    for row, h_color in zip(rows, horizontal_colors):
        for col, v_color in zip(cols, vertical_colors):
            grid[row, col] = random.choice((h_color, v_color))

    legend_len = 2 * len(legend) - 1
    start_col = random.randint(0, width - legend_len)
    for idx, color in enumerate(legend):
        col = start_col + 2 * idx
        grid[height - 2 : height, col] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    h_count, v_count = random.choice(COUNT_OPTIONS)
    total_lines = h_count + v_count

    allowed_modes = ["horizontal_top", "vertical_top"]
    if total_lines >= 3:
        allowed_modes.append("mixed")
    mode = random.choice(allowed_modes)

    min_height = max(8, 7 + LINE_GAP * (h_count - 1))
    min_width = max(2 * total_lines - 1, 5 + LINE_GAP * (v_count - 1))
    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, 30)
    width = random.randint(min_width, 30)
    rows = _sample_spaced_positions(2, height - 5, h_count, LINE_GAP)
    cols = _sample_spaced_positions(2, width - 3, v_count, LINE_GAP)
    if rows is None or cols is None:
        return None

    colors = random.sample(PALETTE, total_lines)
    horizontal_colors = colors[:h_count]
    vertical_colors = colors[h_count:]
    legend = _priority_legend(horizontal_colors, vertical_colors, mode)
    if legend is None:
        return None

    grid = _draw_scene(height, width, rows, cols, horizontal_colors, vertical_colors, legend)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

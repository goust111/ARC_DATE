from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
LEGEND_ORIENTATIONS = ("top", "left")
LINE_COUNT_CHOICES = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 20
LINE_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _legend_info(grid: np.ndarray, bg: int) -> tuple[str, tuple[int, int], list[int], np.ndarray]:
    h, w = grid.shape
    color_counts = Counter(map(int, grid.ravel().tolist()))
    candidates: list[tuple[tuple[int, int, int], str, tuple[int, int], list[int]]] = []

    if h >= 2:
        for c0, c1 in _runs(grid[0] != bg):
            colors = [int(v) for v in grid[0, c0 : c1 + 1].tolist()]
            if len(colors) < 2 or len(colors) != len(set(colors)):
                continue
            if np.any(grid[1, c0 : c1 + 1] != bg):
                continue
            if any(color_counts[color] < 2 for color in colors):
                continue
            support = sum(color_counts[color] for color in colors)
            candidates.append(((len(colors), support, 1), "top", (c0, c1), colors))

    if w >= 2:
        for r0, r1 in _runs(grid[:, 0] != bg):
            colors = [int(v) for v in grid[r0 : r1 + 1, 0].tolist()]
            if len(colors) < 2 or len(colors) != len(set(colors)):
                continue
            if np.any(grid[r0 : r1 + 1, 1] != bg):
                continue
            if any(color_counts[color] < 2 for color in colors):
                continue
            support = sum(color_counts[color] for color in colors)
            candidates.append(((len(colors), support, 0), "left", (r0, r1), colors))

    assert candidates, "expected a top-edge or left-edge priority legend"
    _score, side, span, colors = max(candidates, key=lambda item: item[0])

    legend_mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        c0, c1 = span
        legend_mask[0, c0 : c1 + 1] = True
    else:
        r0, r1 = span
        legend_mask[r0 : r1 + 1, 0] = True
    return side, span, colors, legend_mask


def _horizontal_segments(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    segments: list[dict[str, int]] = []
    for r in range(h):
        c = 0
        while c < w:
            if legend_mask[r, c] or int(grid[r, c]) == bg:
                c += 1
                continue
            c0 = c
            while c < w and not legend_mask[r, c] and int(grid[r, c]) != bg:
                c += 1
            c1 = c - 1
            color = int(grid[r, c0])
            if c1 - c0 + 1 >= 2 and np.all(grid[r, c0 : c1 + 1] == color):
                segments.append({"row": r, "c0": c0, "c1": c1, "color": color})
    return segments


def _row_segment_covering(
    row_segments: dict[int, list[dict[str, int]]], row: int, col: int
) -> dict[str, int] | None:
    matches = [segment for segment in row_segments.get(row, []) if segment["c0"] <= col <= segment["c1"]]
    assert len(matches) <= 1, "row segments must not overlap"
    return matches[0] if matches else None


def _vertical_segments(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    row_segments: dict[int, list[dict[str, int]]],
) -> list[dict[str, int]]:
    h, w = grid.shape
    segments: list[dict[str, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            if legend_mask[r, c] or int(grid[r, c]) == bg:
                r += 1
                continue
            r0 = r
            while r < h and not legend_mask[r, c] and int(grid[r, c]) != bg:
                r += 1
            r1 = r - 1
            if r1 - r0 + 1 < 2:
                continue

            counts = Counter(int(v) for v in grid[r0 : r1 + 1, c].tolist())
            color, count = counts.most_common(1)[0]
            if count < 2:
                continue
            if sum(1 for value in counts.values() if value == count) != 1:
                continue

            valid = True
            for rr in range(r0, r1 + 1):
                shown = int(grid[rr, c])
                hseg = _row_segment_covering(row_segments, rr, c)
                if hseg is None:
                    if shown != color:
                        valid = False
                        break
                else:
                    if shown != hseg["color"]:
                        valid = False
                        break
            if not valid:
                continue
            segments.append({"col": c, "r0": r0, "r1": r1, "color": color})
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave colored horizontal and vertical line segments using an edge priority legend.

    Input structure:
    - Black background with several one-pixel-wide horizontal and vertical colored segments.
    - Every crossing currently shows the horizontal segment color.
    - A one-cell legend on the top edge or left edge lists the colors from highest to lowest priority.

    Rule:
    - Read the color priority from the edge legend.
    - Re-evaluate every horizontal/vertical crossing.
    - Display the higher-priority color at that crossing and keep every non-crossing pixel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    assert bg == BACKGROUND, "expected black background"

    _side, _span, priority_colors, legend_mask = _legend_info(g, bg)
    priority = {color: rank for rank, color in enumerate(priority_colors)}

    horizontals = _horizontal_segments(g, bg, legend_mask)
    row_segments: dict[int, list[dict[str, int]]] = {}
    for segment in horizontals:
        row_segments.setdefault(segment["row"], []).append(segment)

    verticals = _vertical_segments(g, bg, legend_mask, row_segments)

    horizontal_colors = [segment["color"] for segment in horizontals]
    vertical_colors = [segment["color"] for segment in verticals]
    all_line_colors = horizontal_colors + vertical_colors
    assert all_line_colors, "expected at least one line segment"
    assert len(all_line_colors) == len(set(all_line_colors)), "each line segment must use a distinct color"
    assert set(all_line_colors) == set(priority_colors), "legend colors must match the line colors"

    out = g.copy()
    for hseg in horizontals:
        row = hseg["row"]
        h_color = hseg["color"]
        for vseg in verticals:
            col = vseg["col"]
            if not (hseg["c0"] <= col <= hseg["c1"] and vseg["r0"] <= row <= vseg["r1"]):
                continue
            v_color = vseg["color"]
            assert h_color in priority and v_color in priority
            out[row, col] = h_color if priority[h_color] < priority[v_color] else v_color
    return out


def _sample_positions(lo: int, hi: int, count: int, min_step: int) -> list[int] | None:
    span = hi - lo + 1
    compressed = span - (count - 1) * (min_step - 1)
    if compressed < count:
        return None
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[idx] + idx * (min_step - 1) for idx in range(count)]


def _mixed_priority_order(horizontal_colors: list[int], vertical_colors: list[int]) -> list[int]:
    selected = horizontal_colors + vertical_colors
    for _ in range(200):
        order = selected[:]
        random.shuffle(order)
        rank = {color: idx for idx, color in enumerate(order)}
        horizontal_wins = any(rank[h] < rank[v] for h in horizontal_colors for v in vertical_colors)
        vertical_wins = any(rank[v] < rank[h] for h in horizontal_colors for v in vertical_colors)
        if horizontal_wins and vertical_wins:
            return order
    raise RuntimeError("failed to interleave priorities")


def _expected_output(
    grid: np.ndarray,
    horizontal_specs: list[tuple[int, int, int, int]],
    vertical_specs: list[tuple[int, int, int, int]],
    priority_order: list[int],
) -> np.ndarray:
    rank = {color: idx for idx, color in enumerate(priority_order)}
    out = grid.copy()
    for row, c0, c1, h_color in horizontal_specs:
        for col, r0, r1, v_color in vertical_specs:
            if c0 <= col <= c1 and r0 <= row <= r1:
                out[row, col] = h_color if rank[h_color] < rank[v_color] else v_color
    return out


def _legend_cells(
    grid: np.ndarray,
    orientation: str,
    priority_order: list[int],
) -> None:
    if orientation == "top":
        start = random.randint(0, grid.shape[1] - len(priority_order))
        grid[0, start : start + len(priority_order)] = np.asarray(priority_order, dtype=int)
    else:
        start = random.randint(0, grid.shape[0] - len(priority_order))
        grid[start : start + len(priority_order), 0] = np.asarray(priority_order, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(LEGEND_ORIENTATIONS)
    n_h, n_v = random.choice(LINE_COUNT_CHOICES)
    total_colors = n_h + n_v

    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if orientation == "top" and width < total_colors:
        return None
    if orientation == "left" and height < total_colors:
        return None

    row_base = 2 if orientation == "top" else 1
    col_base = 1 if orientation == "top" else 2
    line_row_lo = row_base + 1
    line_row_hi = height - 3
    line_col_lo = col_base + 1
    line_col_hi = width - 3
    if line_row_lo > line_row_hi or line_col_lo > line_col_hi:
        return None

    rows = _sample_positions(line_row_lo, line_row_hi, n_h, LINE_GAP)
    cols = _sample_positions(line_col_lo, line_col_hi, n_v, LINE_GAP)
    if rows is None or cols is None:
        return None

    palette = random.sample(COLORS, total_colors)
    horizontal_colors = palette[:n_h]
    vertical_colors = palette[n_h:]
    priority_order = _mixed_priority_order(horizontal_colors, vertical_colors)

    left_choices = list(range(col_base, cols[0]))
    right_choices = list(range(cols[-1] + 1, width - 1))
    top_choices = list(range(row_base, rows[0]))
    bottom_choices = list(range(rows[-1] + 1, height - 1))
    if not left_choices or not right_choices or not top_choices or not bottom_choices:
        return None

    grid = np.zeros((height, width), dtype=int)
    _legend_cells(grid, orientation, priority_order)

    horizontal_specs: list[tuple[int, int, int, int]] = []
    vertical_specs: list[tuple[int, int, int, int]] = []

    for row, color in zip(rows, horizontal_colors, strict=False):
        c0 = random.choice(left_choices)
        c1 = random.choice(right_choices)
        grid[row, c0 : c1 + 1] = int(color)
        horizontal_specs.append((int(row), int(c0), int(c1), int(color)))

    for col, color in zip(cols, vertical_colors, strict=False):
        r0 = random.choice(top_choices)
        r1 = random.choice(bottom_choices)
        grid[r0 : r1 + 1, col] = int(color)
        vertical_specs.append((int(col), int(r0), int(r1), int(color)))

    for row, c0, c1, h_color in horizontal_specs:
        grid[row, c0 : c1 + 1] = int(h_color)

    expected = _expected_output(grid, horizontal_specs, vertical_specs, priority_order)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-grid input")

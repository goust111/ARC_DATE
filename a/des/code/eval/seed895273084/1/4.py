import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_ATTEMPTS = 400


def _priority_rows(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    rows = [int(r) for r in np.flatnonzero(grid[:, 0] != BG)]
    assert rows, "expected a priority sample in the left column"
    for i in range(1, len(rows)):
        assert rows[i] - rows[i - 1] == 2, "priority sample rows must be separated by one background cell"
    colors = [int(grid[r, 0]) for r in rows]
    assert len(colors) == len(set(colors)), "priority colors must be distinct"
    return rows, colors


def _extract_lines(grid: np.ndarray, priority_colors: List[int]) -> List[Dict[str, int]]:
    line_colors = [color for color in priority_colors if np.any(grid[:, 1:] == color)]
    assert line_colors, "expected at least one colored line"

    lines: List[Dict[str, int]] = []
    for color in line_colors:
        coords = np.argwhere(grid[:, 1:] == color)
        assert coords.size > 0, f"line color {color} must appear outside the priority column"
        coords[:, 1] += 1

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        assert len(rows) == 1 or len(cols) == 1, f"color {color} must form one straight line"
        assert not (len(rows) == 1 and len(cols) == 1), f"line color {color} must have visible length"

        if len(rows) == 1:
            row = int(rows[0])
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            lines.append(
                {
                    "color": color,
                    "orientation": 0,  # horizontal
                    "row": row,
                    "c0": c0,
                    "c1": c1,
                }
            )
        else:
            col = int(cols[0])
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            lines.append(
                {
                    "color": color,
                    "orientation": 1,  # vertical
                    "col": col,
                    "r0": r0,
                    "r1": r1,
                }
            )

    assert any(line["orientation"] == 0 for line in lines), "expected at least one horizontal line"
    assert any(line["orientation"] == 1 for line in lines), "expected at least one vertical line"
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fix wrong line crossings using the left-column weaving priority sample.

    Input structure
    - Black background.
    - The leftmost column contains a spaced vertical priority sample of colors.
    - Elsewhere, each sample color is used by exactly one horizontal or vertical
      single-cell-wide line segment.
    - At a crossing, only one of the two line colors is visible.

    Rule
    - Higher sample entries have higher global priority at crossings.
    - Every crossing in the output must show the higher-priority line color.
    - If an input crossing showed the wrong line color, recolor the crossing and
      paint the two immediate neighbors on that wrong line maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert MAROON not in set(map(int, np.unique(g))), "maroon is reserved for output markings"

    _rows, priority_colors = _priority_rows(g)
    priority = {color: rank for rank, color in enumerate(priority_colors)}
    lines = _extract_lines(g, priority_colors)

    horizontals = [line for line in lines if line["orientation"] == 0]
    verticals = [line for line in lines if line["orientation"] == 1]
    out = g.copy()

    for h in horizontals:
        for v in verticals:
            if not (h["c0"] <= v["col"] <= h["c1"]):
                continue
            if not (v["r0"] <= h["row"] <= v["r1"]):
                continue

            r = int(h["row"])
            c = int(v["col"])
            h_color = int(h["color"])
            v_color = int(v["color"])
            assert h_color in priority and v_color in priority

            if priority[h_color] < priority[v_color]:
                top_color = h_color
                bottom_is_horizontal = False
            else:
                top_color = v_color
                bottom_is_horizontal = True

            visible = int(g[r, c])
            assert visible in {h_color, v_color}, "crossing must show one of its two line colors"
            out[r, c] = top_color

            if visible == top_color:
                continue

            if bottom_is_horizontal:
                for dc in (-1, 1):
                    nc = c + dc
                    if h["c0"] <= nc <= h["c1"] and 0 <= nc < g.shape[1]:
                        out[r, nc] = MAROON
            else:
                for dr in (-1, 1):
                    nr = r + dr
                    if v["r0"] <= nr <= v["r1"] and 0 <= nr < g.shape[0]:
                        out[nr, c] = MAROON

    return out


# --------------- generator ---------------

HORIZONTAL_COUNT_RANGE = (2, 3)
VERTICAL_COUNT_RANGE = (2, 3)
GRID_SIZE_RANGE = (14, 24)
POSITION_GAP = 3
ROW_MARGIN = 1
COL_MARGIN = 3


def _sample_spaced_positions(
    count: int, lo: int, hi: int, gap: int, rng: random.Random
) -> List[int] | None:
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(picks[i] - picks[i - 1] >= gap for i in range(1, len(picks))):
            return picks
    return None


def _sample_orientation_order(total: int, h_count: int, rng: random.Random) -> List[str] | None:
    base = ["H"] * h_count + ["V"] * (total - h_count)
    for _ in range(200):
        order = base[:]
        rng.shuffle(order)
        has_h_above_v = any(order[i] == "H" and order[j] == "V" for i in range(total) for j in range(i + 1, total))
        has_v_above_h = any(order[i] == "V" and order[j] == "H" for i in range(total) for j in range(i + 1, total))
        if has_h_above_v and has_v_above_h:
            return order
    return None


def _build_input(rng: random.Random) -> np.ndarray | None:
    h_count = rng.randint(*HORIZONTAL_COUNT_RANGE)
    v_count = rng.randint(*VERTICAL_COUNT_RANGE)
    total_colors = h_count + v_count

    min_h = max(GRID_SIZE_RANGE[0], 2 * total_colors + 2, ROW_MARGIN * 2 + POSITION_GAP * (h_count - 1) + 5)
    min_w = max(GRID_SIZE_RANGE[0], COL_MARGIN + 1 + POSITION_GAP * (v_count - 1) + 5)
    if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
        return None

    height = rng.randint(min_h, GRID_SIZE_RANGE[1])
    width = rng.randint(min_w, GRID_SIZE_RANGE[1])
    sample_start_max = height - 1 - 2 * (total_colors - 1)
    if sample_start_max < 0:
        return None

    row_positions = _sample_spaced_positions(h_count, 1, height - 2, POSITION_GAP, rng)
    col_positions = _sample_spaced_positions(v_count, 3, width - 2, POSITION_GAP, rng)
    if row_positions is None or col_positions is None:
        return None

    orientation_order = _sample_orientation_order(total_colors, h_count, rng)
    if orientation_order is None:
        return None

    priority_colors = rng.sample(COLOR_POOL, total_colors)
    sample_start = rng.randint(0, sample_start_max)
    sample_rows = [sample_start + 2 * i for i in range(total_colors)]

    horizontal_colors = [color for color, orientation in zip(priority_colors, orientation_order) if orientation == "H"]
    vertical_colors = [color for color, orientation in zip(priority_colors, orientation_order) if orientation == "V"]
    if len(horizontal_colors) != h_count or len(vertical_colors) != v_count:
        return None

    min_v = min(col_positions)
    max_v = max(col_positions)
    min_h_row = min(row_positions)
    max_h_row = max(row_positions)
    if min_v < 3 or min_h_row < 1:
        return None

    lines: List[Dict[str, int]] = []
    for row, color in zip(row_positions, horizontal_colors):
        c0 = rng.randint(2, min_v - 1)
        c1 = rng.randint(max_v + 1, width - 1)
        lines.append({"color": color, "orientation": 0, "row": row, "c0": c0, "c1": c1})
    for col, color in zip(col_positions, vertical_colors):
        r0 = rng.randint(0, min_h_row - 1)
        r1 = rng.randint(max_h_row + 1, height - 1)
        lines.append({"color": color, "orientation": 1, "col": col, "r0": r0, "r1": r1})

    rank = {color: i for i, color in enumerate(priority_colors)}
    pairs_higher_h: List[Tuple[int, int]] = []
    pairs_higher_v: List[Tuple[int, int]] = []
    for hi, h_color in enumerate(horizontal_colors):
        for vi, v_color in enumerate(vertical_colors):
            if rank[h_color] < rank[v_color]:
                pairs_higher_h.append((hi, vi))
            else:
                pairs_higher_v.append((hi, vi))
    if not pairs_higher_h or not pairs_higher_v:
        return None

    crossing_state: Dict[Tuple[int, int], int] = {}

    wrong_vertical = rng.choice(pairs_higher_h)
    crossing_state[wrong_vertical] = vertical_colors[wrong_vertical[1]]

    wrong_horizontal = rng.choice(pairs_higher_v)
    crossing_state[wrong_horizontal] = horizontal_colors[wrong_horizontal[0]]

    remaining_pairs = [(hi, vi) for hi in range(h_count) for vi in range(v_count) if (hi, vi) not in crossing_state]
    if not remaining_pairs:
        return None
    forced_correct = rng.choice(remaining_pairs)
    crossing_state[forced_correct] = (
        horizontal_colors[forced_correct[0]]
        if rank[horizontal_colors[forced_correct[0]]] < rank[vertical_colors[forced_correct[1]]]
        else vertical_colors[forced_correct[1]]
    )

    for hi in range(h_count):
        for vi in range(v_count):
            key = (hi, vi)
            if key in crossing_state:
                continue
            h_color = horizontal_colors[hi]
            v_color = vertical_colors[vi]
            higher = h_color if rank[h_color] < rank[v_color] else v_color
            lower = v_color if higher == h_color else h_color
            crossing_state[key] = rng.choice((higher, lower))

    grid = np.zeros((height, width), dtype=int)
    for row, color in zip(sample_rows, priority_colors):
        grid[row, 0] = color

    for line in lines:
        color = int(line["color"])
        if line["orientation"] == 0:
            grid[line["row"], line["c0"] : line["c1"] + 1] = color
        else:
            grid[line["r0"] : line["r1"] + 1, line["col"]] = color

    for hi, h_line in enumerate([line for line in lines if line["orientation"] == 0]):
        for vi, v_line in enumerate([line for line in lines if line["orientation"] == 1]):
            grid[h_line["row"], v_line["col"]] = crossing_state[(hi, vi)]

    if np.any(grid[:, 0] == MAROON):
        return None
    if MAROON in crossing_state.values():
        return None

    out = main(grid)
    if not np.any(out == MAROON):
        return None

    wrong_horizontal_marked = False
    wrong_vertical_marked = False
    for hi, h_color in enumerate(horizontal_colors):
        for vi, v_color in enumerate(vertical_colors):
            r = row_positions[hi]
            c = col_positions[vi]
            visible = int(grid[r, c])
            higher = h_color if rank[h_color] < rank[v_color] else v_color
            if visible == higher:
                continue
            if visible == h_color:
                wrong_horizontal_marked |= out[r, c - 1] == MAROON and out[r, c + 1] == MAROON
            else:
                wrong_vertical_marked |= out[r - 1, c] == MAROON and out[r + 1, c] == MAROON

    if not (wrong_horizontal_marked and wrong_vertical_marked):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_input(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-priority puzzle")

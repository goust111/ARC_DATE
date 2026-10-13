import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREY = 5


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor exactly the vertical bars reached by a one-direction domino chain started by the red trigger pixel.

    Input structure:
    - The bottom row is a full grey ground line.
    - Above the ground stand several separated one-cell-wide vertical color bars, each ending on the row just above the ground.
    - Exactly one red trigger pixel sits immediately to the left or right of one bar's top cell.

    Rule:
    - The triggered bar falls away from the trigger pixel: a trigger on the left makes the bar fall right, and a trigger on the right makes it fall left.
    - If a falling bar of height h has the next still-standing bar in that same direction at column distance at most h, that next bar also falls in the same direction and continues the chain using its own height.
    - The chain stops at the first gap that is too wide to reach.
    - Output erases the trigger pixel and recolors every reached bar entirely red; all other cells stay unchanged.

    Key insight:
    - Propagation is not based on absolute height or color class; it is a step-by-step reachability test against the next bar in the trigger-determined direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert 3 <= h <= 30 and 3 <= w <= 30

    bg = get_bg(g)
    ground_color = int(g[-1, 0])
    assert np.all(g[-1, :] == ground_color), "bottom row must be a full ground line"

    trigger_coords = np.argwhere(g[:-1, :] == Color.RED)
    assert len(trigger_coords) == 1, "expected exactly one red trigger pixel above the ground"
    trigger_r, trigger_c = map(int, trigger_coords[0])

    bars = []
    for c in range(w):
        column = g[:-1, c]
        mask = (column != bg) & (column != Color.RED)
        if not bool(mask.any()):
            continue
        rows = np.flatnonzero(mask)
        top = int(rows[0])
        assert np.array_equal(rows, np.arange(top, h - 1)), "each bar must be a contiguous vertical run resting on the ground"
        color = int(column[top])
        assert np.all(column[top:] == color), "each bar must be monochromatic"
        bars.append({"col": c, "top": top, "height": h - 1 - top, "color": color})

    assert len(bars) >= 2, "expected at least two bars"
    bars.sort(key=lambda bar: int(bar["col"]))
    bar_cols = [int(bar["col"]) for bar in bars]

    start_idx = None
    direction = None
    for idx, bar in enumerate(bars):
        if trigger_r != int(bar["top"]):
            continue
        if trigger_c == int(bar["col"]) - 1:
            start_idx = idx
            direction = 1
            break
        if trigger_c == int(bar["col"]) + 1:
            start_idx = idx
            direction = -1
            break

    assert start_idx is not None and direction in (-1, 1), "trigger pixel must touch exactly one bar top from the side"

    toppled = [start_idx]
    current = start_idx
    while True:
        nxt = current + direction
        if not (0 <= nxt < len(bars)):
            break
        gap = abs(int(bars[nxt]["col"]) - int(bars[current]["col"]))
        if int(bars[current]["height"]) < gap:
            break
        toppled.append(nxt)
        current = nxt

    out = g.copy()
    out[trigger_r, trigger_c] = bg
    for idx in toppled:
        col = int(bars[idx]["col"])
        top = int(bars[idx]["top"])
        out[top : h - 1, col] = Color.RED
    return out


def _choose_chain_length(side_count: int) -> int:
    if side_count == 1:
        return 1
    if side_count == 2:
        return 1 if random.random() < 0.45 else 2
    roll = random.random()
    if roll < 0.2:
        return 1
    if roll < 0.75:
        return random.randint(2, side_count - 1)
    return side_count


def _parse_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    bars = []
    for c in range(w):
        col = g[:-1, c]
        mask = (col != bg) & (col != Color.RED)
        if not bool(mask.any()):
            continue
        rows = np.flatnonzero(mask)
        top = int(rows[0])
        if np.array_equal(rows, np.arange(top, h - 1)):
            bars.append({"col": c, "top": top, "height": h - 1 - top, "color": int(col[top])})
    return sorted(bars, key=lambda bar: int(bar["col"]))


def _oracle_from_layout(columns, heights, start_idx, direction):
    toppled = [start_idx]
    cur = start_idx
    while True:
        nxt = cur + direction
        if not (0 <= nxt < len(columns)):
            break
        if int(heights[cur]) < abs(int(columns[nxt]) - int(columns[cur])):
            break
        toppled.append(nxt)
        cur = nxt
    return toppled


def _build_output_from_layout(grid, columns, heights, start_idx, direction, top_color=Color.RED):
    out = np.asarray(grid, dtype=int).copy()
    h, _w = out.shape
    trigger_col = int(columns[start_idx]) - direction
    trigger_row = h - 1 - int(heights[start_idx])
    out[trigger_row, trigger_col] = get_bg(out)
    for idx in _oracle_from_layout(columns, heights, start_idx, direction):
        col = int(columns[idx])
        top = h - 1 - int(heights[idx])
        out[top : h - 1, col] = int(top_color)
    return out


def _validate_instance(grid: np.ndarray, columns, heights, start_idx, direction, intended_toppled) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.all(g[-1, :] == Color.GREY), "ground must be a full grey row"
    assert len(np.argwhere(g[:-1, :] == Color.RED)) == 1, "trigger must be unique"
    assert len(columns) >= 4, "need enough bars to create meaningful context"
    assert 0 < start_idx < len(columns) - 1, "keep bars on both sides of the triggered bar"
    assert direction in (-1, 1)
    assert min(np.diff(columns)) >= 2, "bars must stay separated"
    assert all(1 <= int(x) <= h - 2 for x in heights)

    parsed = _parse_bars(g)
    assert [int(bar["col"]) for bar in parsed] == list(map(int, columns))
    assert [int(bar["height"]) for bar in parsed] == list(map(int, heights))

    actual_toppled = _oracle_from_layout(columns, heights, start_idx, direction)
    assert actual_toppled == list(intended_toppled), "layout must realize the intended chain exactly"
    assert len(actual_toppled) >= 1

    far_side_count = (len(columns) - start_idx) if direction == 1 else (start_idx + 1)
    assert len(actual_toppled) <= far_side_count
    assert len(actual_toppled) < len(columns), "at least one bar must remain untouched"

    path = actual_toppled
    for left, right in zip(path, path[1:]):
        gap = abs(int(columns[right]) - int(columns[left]))
        assert int(heights[left]) >= gap, "each successful propagation step must be reachable"

    stop_idx = path[-1]
    next_idx = stop_idx + direction
    if 0 <= next_idx < len(columns):
        stop_gap = abs(int(columns[next_idx]) - int(columns[stop_idx]))
        assert int(heights[stop_idx]) < stop_gap, "chain must stop only when the next gap is too wide"

    opposite_indices = range(0, start_idx) if direction == 1 else range(start_idx + 1, len(columns))
    opposite_heights = [int(heights[i]) for i in opposite_indices]
    toppled_heights = [int(heights[i]) for i in actual_toppled]
    assert opposite_heights, "need opposite-side context bars"
    assert max(opposite_heights) >= min(toppled_heights), "include a counterexample to any height-only shortcut"

    oracle = _build_output_from_layout(g, columns, heights, start_idx, direction)
    solved = main(g)
    assert np.array_equal(solved, oracle), "solver must match the independently checked oracle"


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(4, 6)
    direction = random.choice((-1, 1))
    start_idx = random.randint(1, bar_count - 2)
    side_count = (bar_count - start_idx) if direction == 1 else (start_idx + 1)
    topple_count = _choose_chain_length(side_count)
    intended_toppled = [start_idx + k * direction for k in range(topple_count)]

    gaps = [random.randint(2, 5) for _ in range(bar_count - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    width = left_margin + 1 + sum(gaps) + right_margin
    if width > 30:
        return None

    columns = [left_margin]
    for gap in gaps:
        columns.append(columns[-1] + gap)

    max_height = random.randint(5, min(9, 28))
    height = max_height + 2
    if height > 30:
        return None

    heights = [random.randint(1, max_height) for _ in range(bar_count)]
    for idx in intended_toppled[:-1]:
        next_idx = idx + direction
        gap = abs(columns[next_idx] - columns[idx])
        heights[idx] = random.randint(gap, max_height)

    last_idx = intended_toppled[-1]
    blocked_idx = last_idx + direction
    if 0 <= blocked_idx < bar_count:
        blocked_gap = abs(columns[blocked_idx] - columns[last_idx])
        heights[last_idx] = random.randint(1, blocked_gap - 1)
    else:
        heights[last_idx] = random.randint(1, max_height)

    opposite_indices = list(range(0, start_idx) if direction == 1 else range(start_idx + 1, bar_count))
    if not opposite_indices:
        return None
    tallest_required = min(max_height, max(heights[i] for i in intended_toppled))
    heights[random.choice(opposite_indices)] = random.randint(tallest_required, max_height)

    palette = [1, 3, 4, 6, 7, 8, 9]
    colors = random.sample(palette, bar_count)
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[-1, :] = Color.GREY
    for idx, col in enumerate(columns):
        top = height - 1 - heights[idx]
        grid[top : height - 1, col] = colors[idx]

    trigger_col = columns[start_idx] - direction
    trigger_row = height - 1 - heights[start_idx]
    if not (0 <= trigger_col < width):
        return None
    if grid[trigger_row, trigger_col] != Color.BLACK:
        return None
    grid[trigger_row, trigger_col] = Color.RED

    try:
        _validate_instance(grid, columns, heights, start_idx, direction, intended_toppled)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")

from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 1
PUSHER = 2
GROUND = 5

MIN_H = 9
MAX_H = 15
MIN_BARS = 4
MAX_BARS = 7
MIN_GAP = 2
MAX_GAP = 4
MAX_ATTEMPTS = 400


def _uniform_ground_row(grid: np.ndarray) -> tuple[int, int]:
    row = grid.shape[0] - 1
    color = int(grid[row, 0])
    assert np.all(grid[row] == color), "bottom row must be a solid ground row"
    return row, color


def _bar_height(grid: np.ndarray, ground_r: int, col: int, color: int) -> int:
    rows = np.flatnonzero(grid[:ground_r, col] == color)
    assert rows.size > 0, "bar must have at least one cell"
    r0 = int(rows[0])
    r1 = int(rows[-1])
    assert r1 == ground_r - 1, "bar must stand on the ground row"
    assert rows.size == r1 - r0 + 1, "bar must be vertically solid"
    return int(rows.size)


def _extract_bars(grid: np.ndarray) -> tuple[int, int, list[dict[str, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    ground_r, ground_color = _uniform_ground_row(g)
    bar_mask = np.isin(g[:ground_r], [BAR, PUSHER])
    cols = [int(c) for c in np.flatnonzero(bar_mask.any(axis=0))]
    assert len(cols) >= 2, "expected at least two standing bars"

    red_cols = [c for c in cols if np.any(g[:ground_r, c] == PUSHER)]
    assert len(red_cols) == 1, "expected exactly one red pusher"
    pusher_col = red_cols[0]
    assert pusher_col in {min(cols), max(cols)}, "pusher must stand at one end"
    direction = 1 if pusher_col == min(cols) else -1

    bars = []
    for c in cols:
        col_vals = set(map(int, g[:ground_r, c])) - {BG}
        assert len(col_vals) == 1, "each bar column must use exactly one color"
        color = next(iter(col_vals))
        assert color in {BAR, PUSHER}, "unexpected bar color"
        height = _bar_height(g, ground_r, c, color)

        other = set(map(int, g[: ground_r - height, c])) - {BG}
        assert not other, "bar column must be empty above the segment"
        bars.append({"col": c, "height": height, "color": color})

    ordered = sorted(bars, key=lambda bar: direction * bar["col"])

    occupied = set()
    for bar in ordered:
        c = int(bar["col"])
        assert c not in occupied
        occupied.add(c)

    for c in range(g.shape[1]):
        if c in occupied:
            continue
        assert not np.any(g[:ground_r, c] != BG), "only bar columns may contain foreground above ground"

    return ground_r, ground_color, ordered, direction


def _chain_length(ordered: list[dict[str, int]]) -> int:
    toppled = 1
    while toppled < len(ordered):
        prev = ordered[toppled - 1]
        curr = ordered[toppled]
        gap = abs(int(curr["col"]) - int(prev["col"]))
        if gap <= int(prev["height"]):
            toppled += 1
        else:
            break
    return toppled


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    ground_r, ground_color, ordered, direction = _extract_bars(g)
    toppled_count = _chain_length(ordered)

    out = g.copy()
    line_r = ground_r - 1

    for idx in range(toppled_count):
        bar = ordered[idx]
        col = int(bar["col"])
        height = int(bar["height"])
        color = int(bar["color"])

        out[:ground_r, col] = BG
        for step in range(1, height + 1):
            cc = col + direction * step
            assert 0 <= cc < out.shape[1], "fallen bar must stay inside the grid"
            out[line_r, cc] = color

    out[ground_r, :] = ground_color
    return out, ordered, direction, toppled_count


def main(grid: np.ndarray) -> np.ndarray:
    """Topple the end pusher and keep the domino chain going until the first unreachable gap.

    Input structure:
    - The bottom row is a solid ground row.
    - Above it stand one-cell-wide vertical bars with varying heights and gaps.
    - Exactly one end bar is red; it is the pusher, while the remaining bars are blue.

    Rule:
    - The red end bar falls toward the other bars.
    - A bar of height h disappears from its column and becomes a horizontal segment on
      the row just above the ground, starting from the next column in the fall direction
      and extending for h cells.
    - If the next standing bar's base column is within distance h, that bar also topples
      in the same direction. The chain stops at the first gap that is too large, and all
      farther bars remain standing.
    - When fallen segments overlap, later topples overwrite earlier ones.

    Key insight:
    - Each bar's height is its reach, so the output must be built by simulating the chain
      one bar at a time in order rather than by moving all bars together.
    """
    out, _ordered, _direction, _toppled = _simulate(grid)
    return out


def _make_grid(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1, :] = GROUND
    return grid


def _draw_bar(grid: np.ndarray, ground_r: int, col: int, height: int, color: int) -> None:
    grid[ground_r - height : ground_r, col] = color


def _branch_choice(total_bars: int) -> tuple[str, int]:
    branch = random.choice(["stop_exact", "all_exact", "overshoot"])
    if branch == "stop_exact":
        toppled = random.randint(2, total_bars - 1)
    elif branch == "all_exact":
        toppled = total_bars
    else:
        toppled = random.randint(2, min(4, total_bars))
    return branch, toppled


def _heights_for_case(max_height: int, gaps: list[int], total_bars: int, toppled: int, branch: str) -> list[int] | None:
    heights = [0] * total_bars

    overshoot_idx = toppled - 2 if branch == "overshoot" else -1

    if branch == "overshoot":
        next_hi_max = min(3, max_height)
        if toppled < total_bars:
            next_hi_max = min(next_hi_max, gaps[toppled - 1] - 1)
        if next_hi_max < 1:
            return None
        heights[toppled - 1] = random.randint(1, next_hi_max)

        extra_max = min(3, max_height - gaps[overshoot_idx] - heights[toppled - 1])
        if extra_max < 1:
            return None
        heights[overshoot_idx] = gaps[overshoot_idx] + heights[toppled - 1] + random.randint(1, extra_max)

    for idx in range(toppled - 1):
        if idx == overshoot_idx:
            continue
        heights[idx] = gaps[idx]

    if heights[toppled - 1] == 0:
        if toppled < total_bars:
            stop_gap = gaps[toppled - 1]
            if stop_gap <= 1:
                return None
            heights[toppled - 1] = random.randint(1, stop_gap - 1)
        else:
            heights[toppled - 1] = random.randint(1, min(max_height, 6))

    for idx in range(toppled, total_bars):
        heights[idx] = random.randint(1, max_height)

    return heights


def _abstract_case() -> tuple[int, list[int], list[int], int, str] | None:
    total_bars = random.randint(MIN_BARS, MAX_BARS)
    branch, toppled = _branch_choice(total_bars)
    height = random.randint(12, MAX_H) if branch == "overshoot" else random.randint(MIN_H, MAX_H)
    max_height = height - 2

    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(total_bars - 1)]
    heights = _heights_for_case(max_height, gaps, total_bars, toppled, branch)
    if heights is None:
        return None

    positions = [0]
    for gap in gaps:
        positions.append(positions[-1] + gap)

    farthest = max(max(positions), max(positions[idx] + heights[idx] for idx in range(toppled)))
    required_width = farthest + 3
    if required_width > 30:
        return None
    return height, positions, heights, toppled, branch


def _materialize_case(
    height: int, positions: list[int], heights: list[int], direction: int, width: int, max_reach: int
) -> np.ndarray:
    near_margin = random.randint(1, width - max_reach - 2)
    if direction == 1:
        ordered_cols = [near_margin + x for x in positions]
    else:
        ordered_cols = [width - 1 - near_margin - x for x in positions]

    grid = _make_grid(height, width)
    ground_r = height - 1
    for idx, col in enumerate(ordered_cols):
        color = PUSHER if idx == 0 else BAR
        _draw_bar(grid, ground_r, col, heights[idx], color)
    return grid


def _build_candidate() -> tuple[np.ndarray, str, int] | None:
    abstract = _abstract_case()
    if abstract is None:
        return None

    height, positions, heights, toppled, branch = abstract
    max_reach = max(max(positions), max(positions[idx] + heights[idx] for idx in range(toppled)))
    min_width = max_reach + 3
    width = random.randint(min_width, min(30, min_width + 5))
    direction = random.choice([1, -1])
    grid = _materialize_case(height, positions, heights, direction, width, max_reach)
    return grid, branch, toppled


def _visible_tail_exists(output: np.ndarray, ordered: list[dict[str, int]], direction: int, toppled: int) -> bool:
    source = ordered[toppled - 2]
    target = ordered[toppled - 1]
    row = output.shape[0] - 2
    source_color = int(source["color"])
    limit = int(target["col"]) + direction * int(target["height"])
    tail_start = limit + direction
    end = int(source["col"]) + direction * int(source["height"])

    cells = []
    step = 1 if direction == 1 else -1
    c = tail_start
    while (c <= end if direction == 1 else c >= end):
        if 0 <= c < output.shape[1]:
            cells.append(int(output[row, c]))
        c += step
    return any(cell == source_color for cell in cells)


def _validate_generated(grid: np.ndarray, branch: str, expected_toppled: int) -> bool:
    try:
        output, ordered, direction, toppled = _simulate(grid)
    except AssertionError:
        return False

    if toppled != expected_toppled:
        return False
    if np.array_equal(output, grid):
        return False

    if branch == "stop_exact" and toppled >= len(ordered):
        return False
    if branch == "all_exact" and toppled != len(ordered):
        return False
    if branch == "overshoot":
        if toppled < 2:
            return False
        if not _visible_tail_exists(output, ordered, direction, toppled):
            return False

    return np.array_equal(main(grid), output)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, branch, toppled = built
        if _validate_generated(grid, branch, toppled):
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")

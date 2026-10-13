from __future__ import annotations

from collections import Counter
import random

import numpy as np


SCENARIOS = (
    "all",
    "break",
    "immediate_break",
    "overreach_break",
)

SCENARIO_WEIGHTS = (0.2, 0.3, 0.2, 0.3)
MAX_GENERATION_ATTEMPTS = 500

MIN_HEIGHT = 10
MAX_HEIGHT = 18
MIN_WIDTH = 14
MAX_WIDTH = 30
MIN_GAP = 2
MAX_GAP = 5
MAX_LEFT_PILLARS = 2
MAX_RIGHT_PILLARS = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _ground_description(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    ground = grid[h - 1]
    counts = Counter(map(int, ground))
    bg = _get_bg(grid)
    assert bg not in counts or counts[bg] == 0, "ground row must be full"
    assert len(counts) == 2, "ground row must contain sand and one marker"

    marker_candidates = [color for color, count in counts.items() if count == 1]
    sand_candidates = [color for color, count in counts.items() if count == w - 1]
    assert len(marker_candidates) == 1 and len(sand_candidates) == 1

    marker_color = int(marker_candidates[0])
    sand_color = int(sand_candidates[0])
    marker_col = int(np.flatnonzero(ground == marker_color)[0])
    return bg, sand_color, marker_color, marker_col


def _pillar_heights(grid: np.ndarray, sand_color: int) -> dict[int, int]:
    h, w = grid.shape
    above = grid[: h - 1]
    assert np.all((above == sand_color) | (above == _get_bg(grid))), "only sand may float above the ground"
    return {c: int(np.count_nonzero(above[:, c] == sand_color)) for c in range(w)}


def _toppled_columns(columns: list[int], heights: dict[int, int], marker_col: int) -> set[int]:
    start = columns.index(marker_col)
    toppled = {marker_col}
    current = start
    while current + 1 < len(columns):
        col = columns[current]
        nxt = columns[current + 1]
        if heights[col] >= nxt - col:
            toppled.add(nxt)
            current += 1
        else:
            break
    return toppled


def _render_output(
    shape: tuple[int, int],
    bg: int,
    ground_row: np.ndarray,
    sand_color: int,
    columns: list[int],
    heights: dict[int, int],
    toppled: set[int],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[h - 1] = ground_row
    base_row = h - 2

    for col in columns:
        height = heights[col]
        assert 1 <= height <= h - 1
        if col in toppled:
            assert col + height <= w
            out[base_row, col : col + height] = sand_color
        else:
            out[h - 1 - height : h - 1, col] = sand_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating sand pixel into its column, then topple the marked pile rightward as a domino chain.

    Input structure:
    - The background is dominant, and the bottom row is a full-width sand ground with exactly one differently colored marker cell.
    - Above the ground, sand pixels of the same color as the ground are scattered in separate columns.
    - The marked column contains sand above it; after dropping, non-empty sand columns are isolated and have distinct heights.

    Rule:
    - In each column, let every floating sand pixel fall straight down onto the ground or onto earlier sand in that same column, forming a one-cell-wide solid pillar.
    - The pillar above the marker topples first: erase that vertical pillar and draw a horizontal bar of the same color on the row just above the ground, starting at the pillar's column and extending right by its original height.
    - Moving only through the next pillar to the right each time, topple that next pillar iff the current fallen bar reaches the column immediately to its left; continue until the first failure, and leave all untouched pillars upright.

    Key insight:
    - A pillar's dropped height is also its rightward reach after toppling, so the chain is determined by local height-versus-gap comparisons and stops permanently at the first gap that is too large.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, sand_color, _marker_color, marker_col = _ground_description(g)
    heights = _pillar_heights(g, sand_color)
    columns = [c for c, height in heights.items() if height > 0]

    assert marker_col in columns
    assert len(columns) == len({heights[c] for c in columns}), "pillar heights must be distinct"
    assert all(columns[i + 1] - columns[i] >= 2 for i in range(len(columns) - 1)), "pillars must be isolated"

    toppled = _toppled_columns(columns, heights, marker_col)
    return _render_output(g.shape, bg, g[g.shape[0] - 1].copy(), sand_color, columns, heights, toppled)


def _choose_colors() -> tuple[int, int, int]:
    bg, sand, marker = random.sample(range(10), 3)
    return int(bg), int(sand), int(marker)


def _sample_columns(mode: str, width: int) -> tuple[list[int], list[int]]:
    for _ in range(200):
        min_right = 3 if mode in {"break", "overreach_break"} else 2
        right_count = random.randint(min_right, MAX_RIGHT_PILLARS)
        left_count = random.randint(0, MAX_LEFT_PILLARS)

        right_gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(right_count - 1)]
        left_gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(left_count)]

        right_span = sum(right_gaps)
        left_span = sum(left_gaps)
        if left_span + right_span + 1 > width:
            continue

        marker_min = left_span
        marker_max = width - 1 - right_span
        if marker_min > marker_max:
            continue

        bias_max = max(marker_min, min(marker_max, width // 2))
        marker_col = random.randint(marker_min, bias_max)

        right_cols = [marker_col]
        for gap in right_gaps:
            right_cols.append(right_cols[-1] + gap)

        left_cols_rev = []
        current = marker_col
        for gap in left_gaps:
            current -= gap
            left_cols_rev.append(current)
        left_cols = list(reversed(left_cols_rev))

        if all(0 <= c < width for c in left_cols + right_cols):
            return left_cols, right_cols
    raise RuntimeError("failed to sample pillar columns")


def _height_domains(
    height: int,
    width: int,
    left_cols: list[int],
    right_cols: list[int],
    mode: str,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    all_cols = left_cols + right_cols
    domains: dict[int, list[int]] = {}
    meta: dict[str, int] = {}
    for col in all_cols:
        max_h = min(height - 1, width - col)
        domains[col] = list(range(1, max_h + 1))

    if mode == "break":
        meta["break_idx"] = random.randint(1, len(right_cols) - 2)

    later_success_idx: int | None = None
    if mode == "immediate_break" and len(right_cols) >= 3:
        later_success_idx = 1
    elif mode == "break":
        if meta["break_idx"] + 1 < len(right_cols) - 1:
            later_success_idx = meta["break_idx"] + 1
    elif mode == "overreach_break" and len(right_cols) >= 4:
        later_success_idx = 2

    if later_success_idx is not None:
        meta["later_success_idx"] = later_success_idx

    for idx, col in enumerate(right_cols):
        low = 1
        high = min(height - 1, width - col)
        if idx < len(right_cols) - 1:
            gap = right_cols[idx + 1] - col
            if mode == "all":
                low = max(low, gap)
            elif mode == "immediate_break":
                if idx == 0:
                    high = min(high, gap - 1)
            elif mode == "break":
                break_idx = meta["break_idx"]
                if idx < break_idx:
                    low = max(low, gap)
                elif idx == break_idx:
                    high = min(high, gap - 1)
            elif mode == "overreach_break":
                if idx == 1:
                    high = min(high, gap - 1)

            if later_success_idx is not None and idx == later_success_idx:
                low = max(low, gap)

        if mode == "overreach_break" and idx == 0:
            low = max(low, right_cols[2] - right_cols[0])

        if low > high:
            raise ValueError("empty height domain")
        domains[col] = list(range(low, high + 1))

    return domains, meta


def _assign_heights(domains: dict[int, list[int]]) -> dict[int, int] | None:
    cols = sorted(domains, key=lambda col: (len(domains[col]), col))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def rec(i: int) -> bool:
        if i == len(cols):
            return True
        col = cols[i]
        values = domains[col][:]
        random.shuffle(values)
        for value in values:
            if value in used:
                continue
            used.add(value)
            assignment[col] = value
            if rec(i + 1):
                return True
            used.remove(value)
            del assignment[col]
        return False

    return assignment if rec(0) else None


def _scatter_column(grid: np.ndarray, col: int, count: int, sand_color: int) -> None:
    rows = sorted(random.sample(range(grid.shape[0] - 1), count))
    for row in rows:
        grid[row, col] = sand_color


def _build_input(
    height: int,
    width: int,
    bg: int,
    sand_color: int,
    marker_color: int,
    left_cols: list[int],
    right_cols: list[int],
    heights: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    grid[height - 1, :] = sand_color
    grid[height - 1, right_cols[0]] = marker_color
    for col in left_cols + right_cols:
        _scatter_column(grid, col, heights[col], sand_color)
    return grid


def _validate_scene(
    grid: np.ndarray,
    mode: str,
    left_cols: list[int],
    right_cols: list[int],
    heights: dict[int, int],
    meta: dict[str, int],
) -> bool:
    h, w = grid.shape
    bg, sand_color, marker_color, marker_col = _ground_description(grid)
    if marker_col != right_cols[0]:
        return False
    if int(grid[h - 1, marker_col]) != marker_color:
        return False

    parsed_heights = _pillar_heights(grid, sand_color)
    active_cols = [c for c, v in parsed_heights.items() if v > 0]
    if active_cols != left_cols + right_cols:
        return False
    if any(parsed_heights[c] != heights[c] for c in active_cols):
        return False
    if len({parsed_heights[c] for c in active_cols}) != len(active_cols):
        return False
    if any(active_cols[i + 1] - active_cols[i] < 2 for i in range(len(active_cols) - 1)):
        return False
    if parsed_heights[marker_col] <= 0:
        return False

    sand_total = int(np.count_nonzero(grid == sand_color))
    marker_total = int(np.count_nonzero(grid == marker_color))
    bg_total = int(np.count_nonzero(grid == bg))
    if marker_total != 1 or bg_total <= sand_total:
        return False

    gaps = [right_cols[i + 1] - right_cols[i] for i in range(len(right_cols) - 1)]
    toppled = _toppled_columns(active_cols, parsed_heights, marker_col)
    toppled_right = [c for c in right_cols if c in toppled]

    if mode == "all":
        if toppled_right != right_cols:
            return False
    elif mode == "immediate_break":
        if toppled_right != [right_cols[0]]:
            return False
        if len(right_cols) >= 3:
            idx = meta["later_success_idx"]
            if parsed_heights[right_cols[idx]] < gaps[idx]:
                return False
    elif mode == "break":
        break_idx = meta["break_idx"]
        expected = right_cols[: break_idx + 1]
        if toppled_right != expected:
            return False
        if len(right_cols) > break_idx + 2 and "later_success_idx" in meta:
            idx = meta["later_success_idx"]
            if parsed_heights[right_cols[idx]] < gaps[idx]:
                return False
    elif mode == "overreach_break":
        if toppled_right != right_cols[:2]:
            return False
        if parsed_heights[right_cols[0]] < right_cols[2] - right_cols[0]:
            return False
        if parsed_heights[right_cols[1]] >= right_cols[2] - right_cols[1]:
            return False
        if len(right_cols) >= 4 and "later_success_idx" in meta:
            idx = meta["later_success_idx"]
            if parsed_heights[right_cols[idx]] < gaps[idx]:
                return False

    expected = _render_output(grid.shape, bg, grid[h - 1].copy(), sand_color, active_cols, parsed_heights, toppled)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _sample_candidate() -> np.ndarray | None:
    mode = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    bg, sand_color, marker_color = _choose_colors()

    try:
        left_cols, right_cols = _sample_columns(mode, width)
        domains, meta = _height_domains(height, width, left_cols, right_cols, mode)
    except (RuntimeError, ValueError):
        return None

    heights = _assign_heights(domains)
    if heights is None:
        return None

    grid = _build_input(height, width, bg, sand_color, marker_color, left_cols, right_cols, heights)
    if not _validate_scene(grid, mode, left_cols, right_cols, heights, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sand-domino puzzle")

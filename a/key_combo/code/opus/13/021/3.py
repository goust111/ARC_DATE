from __future__ import annotations

import random

import numpy as np


BG = 0
OBSTACLE = 5
PALETTE_CHOICES = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (10, 15)
OBSTACLE_COUNT_RANGE = (2, 4)
OBSTACLE_LEN_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400


def _find_run(row: np.ndarray, col: int, color: int) -> tuple[int, int]:
    left = int(col)
    while left - 1 >= 0 and int(row[left - 1]) == color:
        left -= 1
    right = int(col)
    width = int(row.shape[0])
    while right + 1 < width and int(row[right + 1]) == color:
        right += 1
    return left, right


def _detour_target(col: int, left: int, right: int, width: int) -> tuple[int, str]:
    candidates: list[tuple[int, int, int, str]] = []
    if left - 1 >= 0:
        new_col = left - 1
        candidates.append((col - new_col, 0, new_col, "left"))
    if right + 1 < width:
        new_col = right + 1
        candidates.append((new_col - col, 1, new_col, "right"))
    assert candidates, "obstacle must have at least one free outside side"
    distance, _prefer_right, new_col, side = min(candidates, key=lambda item: (item[0], -item[1]))
    if len(candidates) == 2 and candidates[0][0] == candidates[1][0]:
        return new_col, "tie"
    return new_col, side


def _trace_events(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    palette = g[0]
    gap_cols = np.flatnonzero(palette == BG)
    assert gap_cols.size == 1
    col = int(gap_cols[0])

    events: list[dict[str, int]] = []
    row = 1
    while row < height:
        if row + 1 < height and int(g[row + 1, col]) == OBSTACLE:
            left, right = _find_run(g[row + 1], col, OBSTACLE)
            new_col, kind = _detour_target(col, left, right, width)
            events.append(
                {
                    "path_row": row,
                    "obstacle_row": row + 1,
                    "from_col": col,
                    "left": left,
                    "right": right,
                    "to_col": new_col,
                    "kind": {"left": 0, "right": 1, "tie": 2}[kind],
                    "horiz_color": int(palette[new_col]),
                }
            )
            col = new_col
            row += 2
        else:
            row += 1
    return events


def main(grid: np.ndarray) -> np.ndarray:
    """Route a falling ray from a palette-row gap around gray bars, coloring each detour by the target palette column.

    Input structure:
    - The background is black. Row 0 is a palette row with exactly one background gap and non-background colors elsewhere.
    - Rows below contain only black background plus 2-4 horizontal gray obstacle segments on distinct rows.
    - The gap column is the ray's launch column, and the palette colors act as a lookup table for later detours.

    Rule:
    - Let `c_start` be the gap column in row 0. The vertical ray color is the palette color immediately left of the gap.
    - Starting at row 1, draw the ray straight downward in column `c_start`.
    - Whenever the next row contains a gray obstacle at the current ray column, find that obstacle segment's outside left and outside right columns, choose the nearer one (breaking ties to the right), draw a horizontal segment on the row above the obstacle in the palette color of the chosen target column, then continue vertically from the row below the obstacle in that new column using the original vertical ray color.
    - Keep the palette row and all gray obstacle pixels unchanged.

    Key insight:
    - The palette row has two roles at once: its single gap defines where the ray starts, and its column colors are looked up dynamically to recolor each horizontal detour after a reroute.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 3 and width >= 3

    palette = g[0]
    gap_cols = np.flatnonzero(palette == BG)
    assert gap_cols.size == 1, "expected exactly one palette gap"
    gap_col = int(gap_cols[0])
    assert 0 < gap_col < width - 1, "gap must not touch a side edge"
    assert np.all(palette[np.arange(width) != gap_col] != BG), "palette may only contain one background gap"

    ray_color = int(palette[gap_col - 1])
    out = g.copy()

    col = gap_col
    row = 1
    while row < height:
        assert int(g[row, col]) != OBSTACLE, "ray should only occupy non-obstacle cells"
        out[row, col] = ray_color
        if row + 1 < height and int(g[row + 1, col]) == OBSTACLE:
            left, right = _find_run(g[row + 1], col, OBSTACLE)
            new_col, _ = _detour_target(col, left, right, width)
            detour_color = int(palette[new_col])
            c0, c1 = sorted((col, new_col))
            out[row, c0 : c1 + 1] = detour_color
            col = new_col
            row += 2
        else:
            row += 1

    return out


def _choose_palette(width: int) -> tuple[np.ndarray, int]:
    gap_col = random.randint(2, width - 3)
    palette = np.full(width, BG, dtype=int)
    distinct = random.sample(PALETTE_CHOICES, random.randint(3, 5))
    non_gap_cols = [c for c in range(width) if c != gap_col]
    for c in non_gap_cols:
        palette[c] = random.choice(distinct)
    forced_cols = random.sample(non_gap_cols, 3)
    for c, color in zip(forced_cols, distinct[:3]):
        palette[c] = color
    return palette, gap_col


def _choose_obstacle_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(2, height - 1))
    for _ in range(120):
        rows = sorted(random.sample(candidates, count))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1)):
            return rows
    return None


def _covering_segments(
    width: int,
    current_col: int,
    gap_col: int,
    desired: str,
    need_inner_target: bool,
) -> list[tuple[int, int, int, str]]:
    options: list[tuple[int, int, int, str]] = []
    for length in range(OBSTACLE_LEN_RANGE[0], OBSTACLE_LEN_RANGE[1] + 1):
        for left in range(0, width - length + 1):
            right = left + length - 1
            if not (left <= current_col <= right):
                continue
            new_col, kind = _detour_target(current_col, left, right, width)
            if new_col == gap_col:
                continue
            if need_inner_target and not (1 <= new_col <= width - 2):
                continue
            if desired != "any" and kind != desired:
                continue
            options.append((left, right, new_col, kind))
    return options


def _noncovering_segments(width: int, current_col: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for length in range(OBSTACLE_LEN_RANGE[0], OBSTACLE_LEN_RANGE[1] + 1):
        for left in range(0, width - length + 1):
            right = left + length - 1
            if left <= current_col <= right:
                continue
            options.append((left, right))
    return options


def _build_grid(
    height: int,
    width: int,
    palette: np.ndarray,
    obstacle_rows: list[int],
    segments: dict[int, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[0] = palette
    for row in obstacle_rows:
        left, right = segments[row]
        grid[row, left : right + 1] = OBSTACLE
    return grid


def _attempt_input() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    obstacle_count = random.randint(*OBSTACLE_COUNT_RANGE)
    obstacle_rows = _choose_obstacle_rows(height, obstacle_count)
    if obstacle_rows is None:
        return None

    palette, gap_col = _choose_palette(width)
    ray_color = int(palette[gap_col - 1])

    detour_count = random.randint(1, obstacle_count)
    detour_indices = {0}
    remaining = list(range(1, obstacle_count))
    if detour_count > 1:
        detour_indices.update(random.sample(remaining, detour_count - 1))

    preferred_first = random.choice(("left", "right", "tie"))
    preferred_other = [random.choice(("left", "right", "tie", "any")) for _ in range(obstacle_count)]
    preferred_other[0] = preferred_first

    current_col = gap_col
    segments: dict[int, tuple[int, int]] = {}
    planned_kinds: list[str] = []

    for idx, row in enumerate(obstacle_rows):
        if idx in detour_indices:
            future_detours = sum(1 for later in range(idx + 1, obstacle_count) if later in detour_indices)
            desired = preferred_other[idx]
            options = _covering_segments(
                width,
                current_col,
                gap_col,
                desired,
                need_inner_target=future_detours > 0,
            )
            if not options and desired != "any":
                options = _covering_segments(
                    width,
                    current_col,
                    gap_col,
                    "any",
                    need_inner_target=future_detours > 0,
                )
            if not options and future_detours > 0:
                options = _covering_segments(width, current_col, gap_col, "any", need_inner_target=False)
            if not options:
                return None
            left, right, new_col, kind = random.choice(options)
            segments[row] = (left, right)
            current_col = new_col
            planned_kinds.append(kind)
        else:
            options = _noncovering_segments(width, current_col)
            if not options:
                return None
            segments[row] = random.choice(options)

    grid = _build_grid(height, width, palette, obstacle_rows, segments)
    events = _trace_events(grid)
    if len(events) != len(detour_indices):
        return None
    if not any(event["from_col"] == gap_col for event in events):
        return None
    if not any(event["horiz_color"] != ray_color for event in events):
        return None
    if preferred_first != "any":
        realized_first = {0: "left", 1: "right", 2: "tie"}[events[0]["kind"]]
        if realized_first != preferred_first:
            return None
    if obstacle_count >= 3 and len(detour_indices) >= 2:
        horiz_colors = {event["horiz_color"] for event in events}
        if len(horiz_colors) < 2:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    # Require visible path pixels below the palette that are not just obstacles.
    trail = out[1:] != grid[1:]
    if int(trail.sum()) < height // 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-ray detour input")

from __future__ import annotations

import random

import numpy as np


BG = 0
OBSTACLE = 5
MARKER_COLORS = (1, 2, 3, 4)

HEIGHT_RANGE = (7, 14)
WIDTH_RANGE = (10, 17)
MAX_GENERATION_ATTEMPTS = 500


def _first_obstacle_col(row: np.ndarray) -> int | None:
    obstacle_cols = np.flatnonzero(row[1:-1] == OBSTACLE)
    if len(obstacle_cols) == 0:
        return None
    return int(obstacle_cols[0] + 1)


def _oracle_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        left = int(g[r, 0])
        right = int(g[r, W - 1])
        if left != right:
            continue
        stop = W - 1
        first_obstacle = _first_obstacle_col(g[r])
        if first_obstacle is not None:
            stop = first_obstacle
        visible = g[r, 1:stop]
        filled = out[r, 1:stop]
        filled[visible == BG] = left
    return out


def _full_fill_shortcut(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r in range(g.shape[0]):
        if int(g[r, 0]) == int(g[r, -1]):
            segment = out[r, 1:-1]
            segment[g[r, 1:-1] == BG] = int(g[r, 0])
    return out


def _left_ray_shortcut(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        color = int(g[r, 0])
        stop = W - 1
        first_obstacle = _first_obstacle_col(g[r])
        if first_obstacle is not None:
            stop = first_obstacle
        segment = out[r, 1:stop]
        segment[g[r, 1:stop] == BG] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the left-to-right visible part of rows whose boundary markers match, stopping at the first gray obstacle.

    Input structure:
    - The background is black (0).
    - Every row has a left border marker at column 0 and a right border marker at column W-1, each colored red(2), green(3), yellow(4), or blue(1).
    - Interior cells are background except that some marker-matching rows contain one or two gray(5) obstacle pixels.

    Rule:
    - Copy the input grid.
    - For each row, compare the left and right border markers.
    - If the two markers differ, leave that row unchanged.
    - If the two markers match with color C, scan from column 1 toward the right.
    - Recolor each visited background cell to C, but stop immediately when the first gray obstacle is reached; that obstacle and everything to its right stay unchanged.

    Key insight:
    - Endpoint color equality only authorizes the row; the actual filled span is determined by the first obstacle seen by the left-to-right ray, so matching rows can produce either full-width or truncated fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert W >= 3

    out = g.copy()
    for r in range(H):
        left = int(g[r, 0])
        right = int(g[r, W - 1])
        if left != right:
            continue
        for c in range(1, W - 1):
            cell = int(g[r, c])
            if cell == OBSTACLE:
                break
            if cell == BG:
                out[r, c] = left
    return out


def _sample_distinct_pair() -> tuple[int, int]:
    left = random.choice(MARKER_COLORS)
    right_choices = [color for color in MARKER_COLORS if color != left]
    return left, random.choice(right_choices)


def _sample_obstacle_positions(width: int, profile: str) -> list[int]:
    if profile == "early":
        low = 2
        high = min(width - 4, max(2, width // 2))
    elif profile == "late":
        low = max(3, width // 2)
        high = width - 4
    else:
        low = 2
        high = width - 4
    if low > high:
        low = high = width - 4
    first = random.randint(low, high)
    positions = [first]
    can_add_second = first + 2 <= width - 2
    if can_add_second and random.random() < 0.4:
        positions.append(random.randint(first + 2, width - 2))
    return positions


def _classify_rows(grid: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    match_with_obstacle: list[int] = []
    match_clean: list[int] = []
    mismatch: list[int] = []
    for r in range(g.shape[0]):
        left = int(g[r, 0])
        right = int(g[r, -1])
        assert left in MARKER_COLORS and right in MARKER_COLORS
        interior_colors = set(map(int, g[r, 1:-1]))
        assert interior_colors.issubset({BG, OBSTACLE})
        if left == right:
            if OBSTACLE in interior_colors:
                match_with_obstacle.append(r)
            else:
                match_clean.append(r)
        else:
            assert OBSTACLE not in interior_colors
            mismatch.append(r)
    return match_with_obstacle, match_clean, mismatch


def _validate_generated_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30 or min(g.shape) < 3:
        return False

    match_with_obstacle, match_clean, mismatch = _classify_rows(g)
    if len(match_with_obstacle) < 2 or len(match_clean) < 1 or len(mismatch) < 2:
        return False

    out = main(g)
    if not np.array_equal(out, _oracle_transform(g)):
        return False
    if np.array_equal(out, g):
        return False
    if np.array_equal(out, _full_fill_shortcut(g)):
        return False
    if np.array_equal(out, _left_ray_shortcut(g)):
        return False

    shared_match_colors = {int(g[r, 0]) for r in match_with_obstacle}.intersection(
        {int(g[r, 0]) for r in match_clean}
    )
    if not shared_match_colors:
        return False
    mismatch_colors = {int(g[r, 0]) for r in mismatch}.union({int(g[r, -1]) for r in mismatch})
    if not mismatch_colors.intersection({int(g[r, 0]) for r in match_with_obstacle + match_clean}):
        return False

    H, W = g.shape
    for r in range(H):
        left = int(g[r, 0])
        right = int(g[r, W - 1])
        if left != right:
            if not np.array_equal(out[r], g[r]):
                return False
            continue

        first_obstacle = _first_obstacle_col(g[r])
        if first_obstacle is None:
            expected = np.full(W - 2, left, dtype=int)
            if not np.array_equal(out[r, 1:-1], expected):
                return False
        else:
            if not np.all(out[r, 1:first_obstacle] == left):
                return False
            if not np.array_equal(out[r, first_obstacle:], g[r, first_obstacle:]):
                return False
            if not np.all(g[r, first_obstacle + 1 : -1] == out[r, first_obstacle + 1 : -1]):
                return False
        if int(out[r, 0]) != left or int(out[r, W - 1]) != right:
            return False

    alt = g.copy()
    mismatch_row = mismatch[0]
    alt[mismatch_row, -1] = alt[mismatch_row, 0]
    if np.array_equal(main(alt), out):
        return False

    obstacle_row = match_with_obstacle[0]
    obstacle_col = _first_obstacle_col(g[obstacle_row])
    if obstacle_col is None or obstacle_col >= W - 3:
        return False
    if int(g[obstacle_row, obstacle_col + 1]) != BG:
        return False
    alt = g.copy()
    alt[obstacle_row, obstacle_col] = BG
    alt[obstacle_row, obstacle_col + 1] = OBSTACLE
    if np.array_equal(main(alt), out):
        return False

    clean_row = match_clean[0]
    probe_col = random.randint(2, W - 3)
    alt = g.copy()
    alt[clean_row, probe_col] = OBSTACLE
    if np.array_equal(main(alt), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*HEIGHT_RANGE)
    W = random.randint(*WIDTH_RANGE)
    if H < 5 or W < 6:
        return None

    grid = np.zeros((H, W), dtype=int)
    rows = list(range(H))
    random.shuffle(rows)

    mismatch_count = random.randint(2, H - 3)
    matched_count = H - mismatch_count
    obstacle_count = random.randint(2, matched_count - 1)
    clean_count = matched_count - obstacle_count

    obstacle_rows = rows[:obstacle_count]
    clean_rows = rows[obstacle_count : obstacle_count + clean_count]
    mismatch_rows = rows[obstacle_count + clean_count :]

    shared_color = random.choice(MARKER_COLORS)
    grid[obstacle_rows[0], [0, W - 1]] = shared_color
    grid[clean_rows[0], [0, W - 1]] = shared_color

    for r in obstacle_rows[1:]:
        color = random.choice(MARKER_COLORS)
        grid[r, [0, W - 1]] = color
    for r in clean_rows[1:]:
        color = random.choice(MARKER_COLORS)
        grid[r, [0, W - 1]] = color

    left, right = shared_color, random.choice([c for c in MARKER_COLORS if c != shared_color])
    grid[mismatch_rows[0], [0, W - 1]] = (left, right)
    left = random.choice([c for c in MARKER_COLORS if c != shared_color])
    grid[mismatch_rows[1], [0, W - 1]] = (left, shared_color)
    for r in mismatch_rows[2:]:
        grid[r, [0, W - 1]] = _sample_distinct_pair()

    profiles = ["early", "late"] + ["random"] * max(0, obstacle_count - 2)
    random.shuffle(profiles)
    for r, profile in zip(obstacle_rows, profiles):
        for c in _sample_obstacle_positions(W, profile):
            grid[r, c] = OBSTACLE

    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

MAX_GEN_ATTEMPTS = 500
_PROFILE_BAG: list[str] = []


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_bar(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert len(coords) > 0, "expected a nonempty grey bar"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "expected a one-column grey bar"
    bar_col = int(cols[0])
    rows = np.sort(coords[:, 0])
    bar_height = len(rows)
    assert np.array_equal(rows, np.arange(bar_height)), "expected a top-anchored contiguous grey bar"
    return bar_col, bar_height


def _first_obstacle_rows(grid: np.ndarray) -> dict[int, int]:
    rows_by_col: dict[int, int] = {}
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == ORANGE)
        if rows.size:
            rows_by_col[int(c)] = int(rows[0])
    return rows_by_col


def _resolve_column(
    target: int,
    height: int,
    obstacle_row: int | None,
    carry_mode: str,
) -> tuple[int, int, str]:
    if target <= 0:
        return 0, 0, "stop"

    if obstacle_row is not None and obstacle_row < target:
        actual = int(obstacle_row)
        shortfall = target - actual
        carry = shortfall if carry_mode in {"obstacle_only", "all_shortfall"} else 0
        return actual, carry, "obstacle"

    actual = min(target, height)
    shortfall = max(0, target - actual)
    carry = shortfall if carry_mode == "all_shortfall" else 0
    reason = "floor" if target > height else "free"
    return int(actual), int(carry), reason


def _apply_rule(
    grid: np.ndarray,
    *,
    carry_mode: str = "obstacle_only",
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    allowed = {BG, GREY, ORANGE}
    colors = set(map(int, np.unique(g)))
    assert colors <= allowed, "input must use only black, grey, and orange"
    assert _get_bg(g) == BG, "expected black background"

    height, width = map(int, g.shape)
    bar_col, bar_height = _find_bar(g)
    assert 0 < bar_col < width - 1, "expected the bar to have space on both sides"
    assert not np.any(g[:, bar_col] == ORANGE), "obstacles must stay on the sides of the bar"

    obstacle_rows = _first_obstacle_rows(g)
    out = g.copy()
    left_trace: list[dict[str, int | str | None]] = []
    right_trace: list[dict[str, int | str | None]] = []

    carry = 0
    for d in range(1, bar_col + 1):
        col = bar_col - d
        base = bar_height + 2 * d
        target = base + carry
        obstacle_row = obstacle_rows.get(col)
        actual, new_carry, reason = _resolve_column(target, height, obstacle_row, carry_mode)
        if actual > 0:
            out[:actual, col] = TEAL
        left_trace.append(
            {
                "d": d,
                "col": col,
                "base": base,
                "carry_in": carry,
                "target": target,
                "obstacle_row": obstacle_row,
                "actual": actual,
                "carry_out": new_carry,
                "reason": reason,
            }
        )
        carry = new_carry

    carry = 0
    for d in range(1, width - bar_col):
        col = bar_col + d
        base = bar_height - 2 * d
        target = base + carry
        if target <= 0:
            right_trace.append(
                {
                    "d": d,
                    "col": col,
                    "base": base,
                    "carry_in": carry,
                    "target": target,
                    "obstacle_row": obstacle_rows.get(col),
                    "actual": 0,
                    "carry_out": 0,
                    "reason": "stop",
                }
            )
            break

        obstacle_row = obstacle_rows.get(col)
        actual, new_carry, reason = _resolve_column(target, height, obstacle_row, carry_mode)
        if actual > 0:
            out[:actual, col] = PINK
        right_trace.append(
            {
                "d": d,
                "col": col,
                "base": base,
                "carry_in": carry,
                "target": target,
                "obstacle_row": obstacle_row,
                "actual": actual,
                "carry_out": new_carry,
                "reason": reason,
            }
        )
        carry = new_carry

    out[g != BG] = g[g != BG]
    if return_trace:
        return out, {"left": left_trace, "right": right_trace, "bar_col": bar_col, "bar_height": bar_height}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the top grey bar sideways while conserving obstacle-blocked height into farther columns.

    Input structure:
    - The grid has a black background, exactly one one-pixel-wide grey vertical bar, and orange obstacle pixels.
    - The grey bar is contiguous, starts at the top edge, and stays unchanged.
    - Obstacles lie only in columns to the left or right of the bar.

    Rule:
    - For the column d steps left of the bar, start with target height h + 2*d plus the carry from the previous nearer-left column; fill teal from the top, but stop immediately above that column's first orange obstacle.
    - For the column d steps right of the bar, start with target height h - 2*d plus the carry from the previous nearer-right column; fill pink from the top, and stop when the target is no longer positive.
    - If an obstacle truncates a column, the blocked amount becomes carry for the next farther column on that same side; if the grid bottom clips the fill, that clipped amount is discarded and does not carry.

    Key insight:
    - The obstacle does not merely cap a column locally: only obstacle-caused losses are conserved and transferred outward, which can keep the right side alive past its normal fade-out.
    """
    return _apply_rule(grid, carry_mode="obstacle_only", return_trace=False)


def _build_grid(
    height: int,
    width: int,
    bar_col: int,
    bar_height: int,
    obstacles: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[:bar_height, bar_col] = GREY
    for c, r in obstacles.items():
        if 0 <= r < height and 0 <= c < width:
            assert c != bar_col
            grid[r, c] = ORANGE
    return grid


def _obstacles_are_singletons(grid: np.ndarray) -> bool:
    rows, cols = np.where(grid == ORANGE)
    obstacle_set = {(int(r), int(c)) for r, c in zip(rows, cols, strict=False)}
    for r, c in obstacle_set:
        if (r - 1, c) in obstacle_set or (r + 1, c) in obstacle_set:
            return False
        if (r, c - 1) in obstacle_set or (r, c + 1) in obstacle_set:
            return False
    return True


def _next_profile() -> str:
    global _PROFILE_BAG
    if not _PROFILE_BAG:
        _PROFILE_BAG = [
            "tight_right_floor",
            "left_visible_carry",
            "double_right_continue",
            "tight_right_floor",
            "left_visible_carry",
            "double_right_continue",
        ]
        random.shuffle(_PROFILE_BAG)
    return _PROFILE_BAG.pop()


def _profile_dims(profile: str) -> tuple[int, int, int, int]:
    if profile == "tight_right_floor":
        bar_height = random.randint(7, 9)
        height = random.randint(bar_height + 1, min(bar_height + 3, 13))
        left_cols = random.randint(4, 6)
        natural_right = (bar_height - 1) // 2
        right_cols = natural_right + 3 + random.randint(0, 1)
        return height, bar_height, left_cols, right_cols

    if profile == "left_visible_carry":
        bar_height = random.randint(5, 7)
        height = random.randint(bar_height + 5, min(bar_height + 8, 18))
        left_cols = random.randint(4, 6)
        natural_right = (bar_height - 1) // 2
        right_cols = natural_right + 2 + random.randint(0, 1)
        return height, bar_height, left_cols, right_cols

    bar_height = random.randint(6, 8)
    height = random.randint(bar_height + 3, min(bar_height + 5, 15))
    left_cols = random.randint(4, 6)
    natural_right = (bar_height - 1) // 2
    right_cols = natural_right + 3 + random.randint(0, 1)
    return height, bar_height, left_cols, right_cols


def _event_map(events: list[dict[str, int | str | None]]) -> dict[int, dict[str, int | str | None]]:
    return {int(event["d"]): event for event in events}


def _pick_left_obstacles(
    profile: str,
    height: int,
    bar_height: int,
    bar_col: int,
    left_cols: int,
) -> dict[int, int] | None:
    if profile == "left_visible_carry":
        for _ in range(200):
            row1 = random.randint(1, min(height - 3, bar_height))
            carry1 = bar_height + 2 - row1
            if carry1 < 2:
                continue
            low = bar_height + 5
            high = min(height - 1, bar_height + 3 + carry1)
            if low > high:
                continue
            row2 = random.randint(low, high)
            obstacles = {
                bar_col - 1: row1,
                bar_col - 2: row2,
            }
            if left_cols >= 4 and random.random() < 0.5:
                row4 = random.randint(1, height - 2)
                if row4 != row2:
                    obstacles[bar_col - 4] = row4
            return obstacles
        return None

    row1 = random.randint(1, min(height - 2, bar_height + 1))
    obstacles = {bar_col - 1: row1}
    if left_cols >= 3 and random.random() < 0.75:
        row3_choices = [r for r in range(1, height - 1) if r != row1]
        obstacles[bar_col - 3] = random.choice(row3_choices)
    return obstacles


def _pick_right_obstacles(
    profile: str,
    height: int,
    bar_height: int,
    bar_col: int,
    right_cols: int,
) -> dict[int, int] | None:
    natural_right = (bar_height - 1) // 2
    if profile == "double_right_continue":
        chain = list(range(1, natural_right + 2))
        allowed_rows = [0, 1, 2]
    elif profile == "tight_right_floor":
        chain = list(range(1, natural_right + 1))
        allowed_rows = [0, 1, 2]
    else:
        chain = list(range(1, natural_right + 1))
        allowed_rows = [1, 2, 3]

    if chain and max(chain) > right_cols:
        return None

    width = bar_col + 1 + right_cols
    for _ in range(300):
        rows = [random.choice(allowed_rows) for _ in chain]
        if any(rows[i] == rows[i + 1] for i in range(len(rows) - 1)):
            continue

        obstacles = {bar_col + d: row for d, row in zip(chain, rows, strict=False)}
        if right_cols >= chain[-1] + 2 and random.random() < 0.4:
            far_col = bar_col + chain[-1] + 2
            far_rows = [r for r in range(1, height - 1) if r != rows[-1]]
            if far_rows:
                obstacles[far_col] = random.choice(far_rows)

        grid = _build_grid(height, width, bar_col, bar_height, obstacles)
        _, trace = _apply_rule(grid, return_trace=True)
        right_events = [event for event in trace["right"] if event["reason"] != "stop"]
        continued = [event for event in right_events if int(event["base"]) <= 0 and int(event["actual"]) > 0]
        if not continued:
            continue
        if profile == "tight_right_floor":
            if not any(
                int(event["base"]) <= 0 and event["reason"] == "floor" and int(event["actual"]) == height
                for event in right_events
            ):
                continue
        if profile == "double_right_continue" and len(continued) < 2:
            continue
        return obstacles
    return None


def _validate_sample(grid: np.ndarray, profile: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _obstacles_are_singletons(grid):
        return False

    bar_col, _bar_height = _find_bar(grid)
    if not np.any(grid[:, :bar_col] == ORANGE):
        return False
    if not np.any(grid[:, bar_col + 1 :] == ORANGE):
        return False

    correct, trace = _apply_rule(grid, return_trace=True)
    no_carry, trace_no = _apply_rule(grid, carry_mode="none", return_trace=True)
    all_shortfall = _apply_rule(grid, carry_mode="all_shortfall", return_trace=False)

    if np.array_equal(correct, grid):
        return False
    if PINK not in correct or TEAL not in correct:
        return False
    if np.array_equal(correct, no_carry):
        return False

    left_events = trace["left"]
    right_events = [event for event in trace["right"] if event["reason"] != "stop"]
    if not any(event["reason"] == "obstacle" for event in left_events):
        return False
    if not any(event["reason"] == "obstacle" for event in right_events):
        return False
    if not any(event["reason"] == "floor" for event in left_events + right_events):
        return False
    if not any(int(event["base"]) <= 0 and int(event["actual"]) > 0 for event in right_events):
        return False

    if profile == "tight_right_floor":
        if np.array_equal(correct, all_shortfall):
            return False
        if not any(
            int(event["base"]) <= 0 and event["reason"] == "floor" and int(event["actual"]) == grid.shape[0]
            for event in right_events
        ):
            return False

    if profile == "double_right_continue":
        if sum(1 for event in right_events if int(event["base"]) <= 0 and int(event["actual"]) > 0) < 2:
            return False

    if profile == "left_visible_carry":
        left_map = _event_map(left_events)
        left_no_map = _event_map(trace_no["left"])
        if not any(int(left_map[d]["actual"]) > int(left_no_map[d]["actual"]) for d in left_map):
            return False

    return True


def _generate_attempt(profile: str) -> np.ndarray | None:
    height, bar_height, left_cols, right_cols = _profile_dims(profile)
    bar_col = left_cols
    width = left_cols + 1 + right_cols

    left_obstacles = _pick_left_obstacles(profile, height, bar_height, bar_col, left_cols)
    right_obstacles = _pick_right_obstacles(profile, height, bar_height, bar_col, right_cols)
    if left_obstacles is None or right_obstacles is None:
        return None

    obstacles = {}
    obstacles.update(left_obstacles)
    obstacles.update(right_obstacles)
    grid = _build_grid(height, width, bar_col, bar_height, obstacles)
    if not _validate_sample(grid, profile):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        profile = _next_profile()
        grid = _generate_attempt(profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for conserved irrigation")

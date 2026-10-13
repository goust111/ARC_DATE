from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ModuleNotFoundError:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color


_BG = Color.BLACK
_FALLING = Color.BLUE
_LEFT_COLOR = Color.RED
_RIGHT_COLOR = Color.GREEN
_OBSTACLE = Color.GREY

_HEIGHT_RANGE = (11, 15)
_WIDTH_RANGE = (11, 15)
_TOP_ROWS = (0, 1, 2)
_GEN_RETRIES = 300


def _row_space_count(grid: np.ndarray, row: int, col: int, step: int) -> int:
    """Count contiguous background cells from the obstacle outward until a blocker or boundary."""
    width = int(grid.shape[1])
    c = col + step
    count = 0
    while 0 <= c < width and int(grid[row, c]) == _BG:
        count += 1
        c += step
    return count


def _detour_direction(grid: np.ndarray, obstacle_row: int, obstacle_col: int) -> int:
    left_space = _row_space_count(grid, obstacle_row, obstacle_col, -1)
    right_space = _row_space_count(grid, obstacle_row, obstacle_col, 1)
    assert left_space != right_space, "detour direction must be uniquely determined"
    return -1 if left_space > right_space else 1


def _sorted_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    starts = [(int(r), int(c)) for r, c in np.argwhere(grid == _FALLING)]
    return sorted(starts, key=lambda rc: (-rc[0], rc[1]))


def _simulate_one(
    static_grid: np.ndarray,
    settled_grid: np.ndarray,
    start: tuple[int, int],
) -> tuple[int, int, int]:
    """Simulate one falling pixel against fixed obstacles plus already settled pixels."""
    height, width = map(int, settled_grid.shape)
    row, col = start
    color = _FALLING
    detours = 0

    while True:
        if row == height - 1:
            return row, col, color

        below = int(settled_grid[row + 1, col])
        if below == _BG:
            row += 1
            continue

        if below == _OBSTACLE:
            direction = _detour_direction(static_grid, row + 1, col)
            next_col = col + direction
            assert 0 <= next_col < width, "detour stepped outside the grid"
            assert int(settled_grid[row, next_col]) == _BG, "detour lane must be empty"
            col = next_col
            color = _LEFT_COLOR if direction < 0 else _RIGHT_COLOR
            detours += 1
            assert detours <= 1, "legal inputs contain at most one obstacle per path"
            continue

        return row, col, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop blue pixels downward, route around one gray obstacle by wider-side space, and recolor by detour direction.

    Input structure:
    - A black background contains blue falling pixels scattered near the top, with at most one blue pixel per column.
    - Gray obstacle pixels sit in the middle rows, and the lower rows are obstacle-free.
    - Blue pixels and gray obstacles are the only non-background objects.

    Rule:
    - Remove the blue pixels from their starting cells and process them from lower starts to higher starts so settled pixels can block later ones.
    - Each blue pixel falls one row at a time in its current column.
    - If the next cell downward is a gray obstacle, inspect that obstacle's row and compare the contiguous background-space counts immediately to its left and right, stopping each count at the nearest non-background cell or the grid boundary; sidestep one cell toward the wider side, then keep falling.
    - A pixel that never detours stays blue, a left detour recolors it red, and a right detour recolors it green.
    - Stop when the pixel reaches the bottom row or when the next cell downward is already occupied by a settled non-background pixel; keep all gray obstacles unchanged.

    Key insight:
    - The color is not tied to a fixed side or obstacle shape: you must measure obstacle-row free space in the puzzle's own row-wise reference frame, and settled pixels matter because multiple paths can converge and stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({_BG, _FALLING, _OBSTACLE}), "expected only black, blue, and gray"

    out = g.copy()
    out[out == _FALLING] = _BG

    starts = _sorted_starts(g)
    for start in starts:
        end_row, end_col, end_color = _simulate_one(g, out, start)
        assert int(out[end_row, end_col]) == _BG, "two particles cannot claim the same resting cell"
        out[end_row, end_col] = end_color

    return out


def _pick_unused_column(width: int, forbidden: set[int], *, margin: int = 0) -> int | None:
    candidates = [
        col
        for col in range(margin, width - margin)
        if col not in forbidden
    ]
    return random.choice(candidates) if candidates else None


def _place_open_open_detour(
    grid: np.ndarray,
    obstacle_row: int,
    obstacle_col: int,
    direction: str,
) -> None:
    """Create a nontrivial detour where both adjacent cells are open but one side has more room."""
    height, width = map(int, grid.shape)
    assert 0 <= obstacle_row < height
    assert 0 <= obstacle_col < width
    grid[obstacle_row, obstacle_col] = _OBSTACLE

    if direction == "left":
        assert obstacle_col >= 2, "left-open detour needs at least two cells to the left boundary"
        tight_col = obstacle_col + 2
        if tight_col < width:
            grid[obstacle_row, tight_col] = _OBSTACLE
        else:
            grid[obstacle_row, obstacle_col + 1] = _OBSTACLE
    else:
        assert direction == "right"
        assert obstacle_col <= width - 3, "right-open detour needs at least two cells to the right boundary"
        tight_col = obstacle_col - 2
        if tight_col >= 0:
            grid[obstacle_row, tight_col] = _OBSTACLE
        else:
            grid[obstacle_row, obstacle_col - 1] = _OBSTACLE


def _place_blocked_open_detour(
    grid: np.ndarray,
    obstacle_row: int,
    obstacle_col: int,
    direction: str,
) -> None:
    """Create a simpler branch where only the chosen side is open."""
    grid[obstacle_row, obstacle_col] = _OBSTACLE
    if direction == "left":
        grid[obstacle_row, obstacle_col + 1] = _OBSTACLE
    else:
        assert direction == "right"
        grid[obstacle_row, obstacle_col - 1] = _OBSTACLE


def _role_direction(role: dict) -> str | None:
    if role["kind"] == "detour":
        return str(role["direction"])
    return None


def _check_generated_grid(grid: np.ndarray, roles: list[dict]) -> bool:
    height, width = map(int, grid.shape)
    if not (max(height, width) <= 30):
        return False
    if not np.all(grid[-2:, :] != _OBSTACLE):
        return False

    blue_coords = np.argwhere(grid == _FALLING)
    if len(blue_coords) != len(roles):
        return False

    blue_cols = [int(c) for _, c in blue_coords]
    if len(blue_cols) != len(set(blue_cols)):
        return False

    directions = [_role_direction(role) for role in roles if role["kind"] == "detour"]
    if not {"left", "right"}.issubset(set(directions)):
        return False

    if not any(role.get("mode") == "open_open" for role in roles):
        return False

    for role in roles:
        if role["kind"] != "detour":
            continue
        obstacle_row = int(role["obstacle_row"])
        obstacle_col = int(role["start_col"])
        left_space = _row_space_count(grid, obstacle_row, obstacle_col, -1)
        right_space = _row_space_count(grid, obstacle_row, obstacle_col, 1)
        if left_space == right_space:
            return False

        chosen = -1 if role["direction"] == "left" else 1
        if role["mode"] == "open_open":
            if int(grid[obstacle_row, obstacle_col - 1]) != _BG:
                return False
            if int(grid[obstacle_row, obstacle_col + 1]) != _BG:
                return False
        else:
            blocked_col = obstacle_col - chosen
            open_col = obstacle_col + chosen
            if int(grid[obstacle_row, blocked_col]) != _OBSTACLE:
                return False
            if int(grid[obstacle_row, open_col]) != _BG:
                return False

        final_col = obstacle_col + chosen
        if np.any(grid[obstacle_row + 1 :, final_col] == _OBSTACLE):
            return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False

    moved = np.count_nonzero((out == _FALLING) | (out == _LEFT_COLOR) | (out == _RIGHT_COLOR))
    if moved != len(roles):
        return False

    if np.count_nonzero(out == _LEFT_COLOR) < 1:
        return False
    if np.count_nonzero(out == _RIGHT_COLOR) < 1:
        return False
    if np.count_nonzero(out == _FALLING) < 1:
        return False

    # The shared landing column is the anti-shortcut case: one pixel must stack above another.
    shared_target_col = next(int(role["target_col"]) for role in roles if role["label"] == "shared_fall")
    occupied_rows = [int(r) for r in np.where(out[:, shared_target_col] != _BG)[0]]
    if len(occupied_rows) < 2:
        return False
    if occupied_rows[-1] != height - 1 or occupied_rows[-2] != height - 2:
        return False

    return True


def _build_case() -> tuple[np.ndarray, list[dict]] | None:
    height = random.randint(*_HEIGHT_RANGE)
    width = random.randint(*_WIDTH_RANGE)
    if width < 11:
        return None

    grid = np.zeros((height, width), dtype=int)
    roles: list[dict] = []

    shared_target = random.randint(3, width - 4)
    shared_direction = random.choice(("left", "right"))
    shared_detour_col = shared_target + 1 if shared_direction == "left" else shared_target - 1

    used_blue_cols = {shared_target, shared_detour_col}

    other_candidates = [
        col
        for col in range(2, width - 2)
        if min(abs(col - shared_target), abs(col - shared_detour_col)) >= 4
    ]
    if not other_candidates:
        return None
    other_detour_col = random.choice(other_candidates)
    other_direction = "right" if shared_direction == "left" else "left"
    used_blue_cols.add(other_detour_col)

    extra_kind = None
    extra_col = None
    extra_direction = None
    extra_candidates = [
        col
        for col in range(1, width - 1)
        if all(abs(col - used) >= 3 for used in used_blue_cols)
    ]
    if extra_candidates and random.random() < 0.6:
        extra_kind = random.choice(("fall", "blocked_detour"))
        extra_col = random.choice(extra_candidates)
        if extra_kind == "blocked_detour":
            extra_direction = random.choice(("left", "right"))
            if extra_direction == "left" and extra_col >= width - 1:
                extra_kind = "fall"
            if extra_direction == "right" and extra_col <= 0:
                extra_kind = "fall"

    detour_count = 2 + (1 if extra_kind == "blocked_detour" else 0)
    obstacle_rows = random.sample(list(range(4, height - 2)), detour_count)
    shared_obstacle_row = obstacle_rows[0]
    other_obstacle_row = obstacle_rows[1]
    extra_obstacle_row = obstacle_rows[2] if extra_kind == "blocked_detour" else None

    shared_fall_row = random.choice((1, 2))
    shared_detour_row = 0
    other_detour_row = random.choice(_TOP_ROWS)
    extra_row = random.choice(_TOP_ROWS) if extra_kind is not None else None

    grid[shared_fall_row, shared_target] = _FALLING
    roles.append(
        {
            "label": "shared_fall",
            "kind": "fall",
            "start_row": shared_fall_row,
            "start_col": shared_target,
            "target_col": shared_target,
        }
    )

    grid[shared_detour_row, shared_detour_col] = _FALLING
    _place_open_open_detour(grid, shared_obstacle_row, shared_detour_col, shared_direction)
    roles.append(
        {
            "label": "shared_detour",
            "kind": "detour",
            "mode": "open_open",
            "direction": shared_direction,
            "start_row": shared_detour_row,
            "start_col": shared_detour_col,
            "obstacle_row": shared_obstacle_row,
            "target_col": shared_target,
        }
    )

    grid[other_detour_row, other_detour_col] = _FALLING
    _place_open_open_detour(grid, other_obstacle_row, other_detour_col, other_direction)
    roles.append(
        {
            "label": "opposite_detour",
            "kind": "detour",
            "mode": "open_open",
            "direction": other_direction,
            "start_row": other_detour_row,
            "start_col": other_detour_col,
            "obstacle_row": other_obstacle_row,
            "target_col": other_detour_col + (-1 if other_direction == "left" else 1),
        }
    )

    if extra_kind == "fall" and extra_col is not None and extra_row is not None:
        grid[extra_row, extra_col] = _FALLING
        roles.append(
            {
                "label": "extra_fall",
                "kind": "fall",
                "start_row": extra_row,
                "start_col": extra_col,
                "target_col": extra_col,
            }
        )
    elif extra_kind == "blocked_detour" and extra_col is not None and extra_row is not None and extra_obstacle_row is not None:
        if extra_direction == "left" and extra_col >= width - 1:
            return None
        if extra_direction == "right" and extra_col <= 0:
            return None
        grid[extra_row, extra_col] = _FALLING
        _place_blocked_open_detour(grid, extra_obstacle_row, extra_col, extra_direction)
        roles.append(
            {
                "label": "extra_detour",
                "kind": "detour",
                "mode": "blocked_open",
                "direction": extra_direction,
                "start_row": extra_row,
                "start_col": extra_col,
                "obstacle_row": extra_obstacle_row,
                "target_col": extra_col + (-1 if extra_direction == "left" else 1),
            }
        )

    # A few inert obstacles keep the middle visually varied without touching any path columns.
    reserved_cols = {int(role["start_col"]) for role in roles}
    reserved_cols |= {int(role["target_col"]) for role in roles}
    key_obstacle_rows = {
        int(role["obstacle_row"])
        for role in roles
        if role["kind"] == "detour"
    }
    for obstacle_row in range(4, height - 2):
        if obstacle_row in key_obstacle_rows:
            continue
        if random.random() >= 0.2:
            continue
        col = _pick_unused_column(width, reserved_cols, margin=1)
        if col is None or int(grid[obstacle_row, col]) != _BG:
            continue
        grid[obstacle_row, col] = _OBSTACLE

    return grid, roles


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, roles = built
        if _check_generated_grid(grid, roles):
            return grid
    raise RuntimeError("failed to generate a valid gravity-detour-color sample")

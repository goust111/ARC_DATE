from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LEFT_TURN_COLOR = 1   # blue
RIGHT_TURN_COLOR = 2  # red
START_COLOR = 5       # gray
PALETTE_CHOICES = (3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
GENERATION_RETRIES = 2000


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _parse_palette_and_start(grid: np.ndarray) -> tuple[list[int], int]:
    top = np.asarray(grid[0], dtype=int)
    bg = _get_bg(grid)
    c = 0
    while c < top.size and int(top[c]) == bg:
        c += 1
    palette: list[int] = []
    while c < top.size and int(top[c]) not in (bg, START_COLOR):
        palette.append(int(top[c]))
        c += 1
    start_cols = np.flatnonzero(top == START_COLOR)
    if len(start_cols) != 1 or not palette:
        raise ValueError("grid does not contain one palette run and one gray start")
    return palette, int(start_cols[0])


def _trace_path(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]], tuple[int, int, int]]:
    h, w = grid.shape
    palette, start_c = _parse_palette_and_start(grid)
    out = grid.copy()
    dir_idx = 2  # down
    palette_idx = 0
    r, c = 1, start_c
    painted: list[tuple[int, int, int]] = []
    used_turns: list[tuple[int, int, int]] = []
    seen_states: set[tuple[int, int, int, int]] = set()

    for _ in range(h * w * 4):
        if not _in_bounds((h, w), r, c):
            raise ValueError("path left the grid from an invalid intermediate state")

        state = (r, c, dir_idx, palette_idx)
        if state in seen_states:
            raise ValueError("path entered a loop")
        seen_states.add(state)

        out[r, c] = palette[palette_idx]
        painted.append((r, c, palette[palette_idx]))

        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return out, painted, used_turns, (r, c, dir_idx)

        ahead = int(grid[nr, nc])
        if ahead == RIGHT_TURN_COLOR:
            dir_idx = (dir_idx + 1) % 4
            palette_idx = (palette_idx + 1) % len(palette)
            used_turns.append((nr, nc, ahead))
            dr, dc = DIRS[dir_idx]
            r, c = r + dr, c + dc
        elif ahead == LEFT_TURN_COLOR:
            dir_idx = (dir_idx - 1) % 4
            palette_idx = (palette_idx + 1) % len(palette)
            used_turns.append((nr, nc, ahead))
            dr, dc = DIRS[dir_idx]
            r, c = r + dr, c + dc
        else:
            r, c = nr, nc

    raise ValueError("path did not leave the grid")


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the routed path from the top-edge start, turning around red and blue blockers while cycling the palette.

    Input structure:
    - The top row has a contiguous left-side run of distinct palette colors.
    - A separate gray pixel on the top edge is the unique start marker.
    - Red and blue pixels inside the grid are turn markers; other cells are background.

    Rule:
    - Start from the cell directly below the gray marker and move downward, painting with the first palette color.
    - If the next cell in the current direction is red, turn right 90 degrees; if it is blue, turn left 90 degrees.
    - The turn marker itself stays unchanged; the path continues from the side-adjacent cell in the new direction.
    - After every turn, advance to the next palette color cyclically.
    - Continue until the next step would leave the grid.

    Key insight:
    - The top-row color run is an ordered cyclic palette, while each red or blue pixel is a directional blocker that consumes one palette step and defines the next colored segment.
    """
    out, _painted, _used_turns, _end_state = _trace_path(np.asarray(grid, dtype=int))
    return out


def _clear_run_to_exit(
    shape: tuple[int, int],
    pos: tuple[int, int],
    dir_idx: int,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = shape
    r, c = pos
    dr, dc = DIRS[dir_idx]
    nr, nc = r + dr, c + dc
    while _in_bounds((h, w), nr, nc):
        if (nr, nc) in blocked:
            return False
        nr += dr
        nc += dc
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    palette_len = random.randint(3, 5)
    if w < palette_len + 6:
        return None

    palette = random.sample(PALETTE_CHOICES, palette_len)
    start_c = random.randint(palette_len + 2, w - 2)
    require_wrap = random.random() < 0.55
    if require_wrap:
        target_turns = random.randint(palette_len, palette_len + 2)
    else:
        target_turns = random.randint(2, max(2, palette_len - 1))

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for idx, color in enumerate(palette):
        grid[0, idx] = color
    grid[0, start_c] = START_COLOR

    blocked = {(0, col) for col in range(w)}
    dir_idx = 2
    palette_idx = 0
    r, c = 1, start_c
    blocked.add((r, c))

    painted = [(r, c, palette[palette_idx])]
    markers: list[tuple[int, int, int]] = []
    seen_turn_colors: list[int] = []
    steps_in_segment = 1
    desired_segment_len = random.randint(2, 5)
    exiting = False
    success = False

    for _ in range(h * w * 4):
        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        goal_met = (
            len(markers) >= target_turns
            and RIGHT_TURN_COLOR in seen_turn_colors
            and LEFT_TURN_COLOR in seen_turn_colors
        )

        if not _in_bounds((h, w), nr, nc):
            if goal_met and len(painted) >= 8:
                success = True
            break

        straight_ok = (nr, nc) not in blocked
        turn_options: list[tuple[int, int, int, int, int, int]] = []
        for turn_color, delta in ((RIGHT_TURN_COLOR, 1), (LEFT_TURN_COLOR, -1)):
            next_dir = (dir_idx + delta) % 4
            sr = r + DIRS[next_dir][0]
            sc = c + DIRS[next_dir][1]
            if (
                (nr, nc) not in blocked
                and _in_bounds((h, w), sr, sc)
                and (sr, sc) not in blocked
            ):
                turn_options.append((turn_color, next_dir, sr, sc, nr, nc))

        if goal_met and _clear_run_to_exit((h, w), (r, c), dir_idx, blocked):
            exiting = True

        if exiting and straight_ok:
            r, c = nr, nc
            blocked.add((r, c))
            painted.append((r, c, palette[palette_idx]))
            steps_in_segment += 1
            continue

        if straight_ok and steps_in_segment < desired_segment_len:
            r, c = nr, nc
            blocked.add((r, c))
            painted.append((r, c, palette[palette_idx]))
            steps_in_segment += 1
            continue

        choices: list[tuple[str, tuple[int, int, int, int, int, int] | None]] = []
        if straight_ok:
            choices.extend([("straight", None)] * 4)
        for option in turn_options:
            turn_color = option[0]
            weight = 2
            if turn_color not in seen_turn_colors:
                weight += 4
            if len(markers) < target_turns:
                weight += 2
            choices.extend([("turn", option)] * weight)
        if not choices:
            break

        action, option = random.choice(choices)
        if action == "straight":
            r, c = nr, nc
            blocked.add((r, c))
            painted.append((r, c, palette[palette_idx]))
            steps_in_segment += 1
            continue

        turn_color, dir_idx, sr, sc, tr, tc = option
        grid[tr, tc] = turn_color
        blocked.add((tr, tc))
        markers.append((tr, tc, turn_color))
        seen_turn_colors.append(turn_color)
        palette_idx = (palette_idx + 1) % palette_len
        r, c = sr, sc
        blocked.add((r, c))
        painted.append((r, c, palette[palette_idx]))
        steps_in_segment = 1
        desired_segment_len = random.randint(2, 5)

    if not success:
        return None

    expected = grid.copy()
    for rr, cc, color in painted:
        expected[rr, cc] = color

    try:
        actual, traced_painted, used_turns, _end_state = _trace_path(grid)
    except ValueError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if sorted(markers) != sorted(used_turns):
        return None
    if len(markers) < 2:
        return None
    if RIGHT_TURN_COLOR not in seen_turn_colors or LEFT_TURN_COLOR not in seen_turn_colors:
        return None
    if len({color for _r, _c, color in traced_painted}) < 3:
        return None
    if require_wrap and len(markers) < palette_len:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid turning-path instance")

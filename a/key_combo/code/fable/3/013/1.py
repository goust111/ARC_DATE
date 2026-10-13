from __future__ import annotations

from importlib.machinery import SourceFileLoader
from typing import Iterable

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_3_013_1",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
coords_where = _ARC_UTILS.coords_where
in_bounds = _ARC_UTILS.in_bounds

RIGHT_TURN = Color.RED
LEFT_TURN = Color.TEAL
STRAIGHT = Color.YELLOW
DIVIDER = Color.GRAY
START = Color.GREEN
BACKGROUND = Color.BLACK

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


def _turn(direction: tuple[int, int], command: int) -> tuple[int, int]:
    d = DIRS.index(tuple(direction))
    if command == RIGHT_TURN:
        d = (d + 1) % 4
    elif command == LEFT_TURN:
        d = (d - 1) % 4
    elif command == STRAIGHT:
        pass
    else:
        raise ValueError("unsupported command color")
    return DIRS[d]


def _find_separator(g: np.ndarray) -> tuple[int, int, int]:
    H, W = g.shape
    runs: list[tuple[int, int, int]] = []
    c = 0
    while c < W:
        col = g[:, c]
        if not np.all(col == int(col[0])):
            c += 1
            continue

        color = int(col[0])
        s = c
        while c + 1 < W and np.all(g[:, c + 1] == color):
            c += 1
        e = c
        if 0 < s and e < W - 1:
            runs.append((s, e, color))
        c += 1

    candidates = [run for run in runs if run[2] in {DIVIDER, Color.GRAY}]
    if not candidates:
        candidates = [
            run
            for run in runs
            if run[2] not in {BACKGROUND, START, RIGHT_TURN, LEFT_TURN, STRAIGHT}
        ]
    assert candidates, "expected one full-height separator run"
    return sorted(candidates, key=lambda item: (item[1] - item[0] + 1, item[0], item[2]))[0]


def _decode_program(left_panel: np.ndarray) -> list[tuple[int, int]]:
    H, W = left_panel.shape
    assert W > 0, "left panel is empty"

    cmds: list[tuple[int, int]] = []
    c = 0
    while c < W:
        col = left_panel[:, c]
        assert np.all(col == int(col[0])), "left panel stripe must be full-height"
        color = int(col[0])
        if color not in {RIGHT_TURN, LEFT_TURN, STRAIGHT}:
            raise ValueError("left panel uses unsupported stripe color")
        s = c
        while c + 1 < W and np.all(left_panel[:, c + 1] == color):
            c += 1
        e = c
        cmds.append((color, e - s + 1))
        c += 1

    return cmds


def _boundary_sides(sr: int, sc: int, panel_shape: tuple[int, int], sep_right: int, W: int) -> list[str]:
    panel_top = 0
    panel_left = sep_right + 1
    panel_bottom = panel_shape[0] - 1
    panel_right = W - 1

    if not (panel_left <= sc < W):
        raise ValueError("start is not in the right panel")

    sides: list[str] = []
    if sr == panel_top:
        sides.append("top")
    if sr == panel_bottom:
        sides.append("bottom")
    if sc == panel_left:
        sides.append("left")
    if sc == panel_right:
        sides.append("right")
    if not sides:
        raise ValueError("start must lie on the boundary of right panel")
    return sides


def _start_direction(sr: int, sc: int, panel_shape: tuple[int, int], sep_right: int, W: int) -> tuple[int, int]:
    sides = _boundary_sides(sr, sc, panel_shape, sep_right, W)
    candidates: list[tuple[int, int]] = []
    for s in sides:
        if s == "left":
            candidates.append((0, 1))
        if s == "right":
            candidates.append((0, -1))
        if s == "top":
            candidates.append((1, 0))
        if s == "bottom":
            candidates.append((-1, 0))

    if len(candidates) == 1:
        return candidates[0]

    for prefer in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        if prefer in candidates:
            return prefer
    raise ValueError("ambiguous start direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a turning path in the right panel according to the left-program stripes.

    Input structure:
    - The grid is split by a vertical full-height divider into a left panel, divider, right panel.
    - The left panel consists of full-height vertical color stripes using red, cyan/teal, and yellow.
    - Each stripe encodes one instruction: its color controls turn direction, its width encodes step count.
    - The right panel is black, with exactly one green start pixel on its boundary.

    Rule:
    - Read stripe commands from left to right on the left panel.
    - Start at the green boundary pixel in the right panel, initially pointing from boundary into the panel interior.
    - For each command, turn by stripe color (red: right turn, teal: left turn, yellow: straight), then move exactly the stripe width steps.
    - Paint every visited step cell green in the right panel.

    Key insight:
    - The left panel is a compact instruction tape: color gives turning behavior and stripe thickness gives exact step length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be non-empty 2D"

    sep_left, sep_right, _sep_color = _find_separator(g)
    left_panel = g[:, :sep_left]
    right_panel = g[:, sep_right + 1 :]

    start_coords = coords_where(right_panel == START)
    assert start_coords.shape == (1, 2), "expected exactly one green start pixel"
    sr_local, sc_local = map(int, start_coords[0])
    sr_abs = sr_local
    sc_abs = sc_local + sep_right + 1

    dr, dc = _start_direction(sr_abs, sc_abs, right_panel.shape, sep_right, g.shape[1])

    commands = _decode_program(left_panel)
    out = right_panel.copy()
    out[sr_local, sc_local] = START

    r, c = sr_local, sc_local
    visited = {(r, c)}
    for color, thickness in commands:
        dr, dc = _turn((dr, dc), color)
        for _ in range(thickness):
            r += dr
            c += dc
            if not in_bounds((right_panel.shape[0], right_panel.shape[1]), r, c):
                raise AssertionError("path leaves right panel")
            if (r, c) in visited:
                raise AssertionError("path overlaps itself")
            visited.add((r, c))
            out[int(r), int(c)] = START

    return out.astype(int)


def _paint_path_in_panel(
    panel_shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    commands: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = map(int, panel_shape)
    dr, dc = map(int, direction)
    out = np.full((H, W), BACKGROUND, dtype=int)
    out[start] = START

    visited = {start}
    path: list[tuple[int, int]] = [start]
    cmds: list[tuple[int, int]] = []
    r, c = map(int, start)

    for color, steps in commands:
        if steps <= 0:
            raise ValueError("thickness must be positive")
        dr, dc = _turn((dr, dc), int(color))
        r2, c2 = r, c
        for _ in range(int(steps)):
            r2 += dr
            c2 += dc
            if not in_bounds((H, W), r2, c2):
                raise ValueError("path exits right panel")
            if (r2, c2) in visited:
                raise ValueError("path overlaps itself")
            visited.add((r2, c2))
            path.append((r2, c2))
            out[r2, c2] = START
        r, c = r2, c2
        cmds.append((int(color), int(steps)))

    return out, path, cmds


def _choose_program(width: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if width < 3:
        return [(YELLOW, max(1, width))]

    colors = [RIGHT_TURN, LEFT_TURN, STRAIGHT]
    order = list(colors)
    rng.shuffle(order)
    remaining = int(width)
    commands: list[tuple[int, int]] = []

    for idx, color in enumerate(order):
        max_step = max(1, remaining - (len(order) - idx - 1))
        step = int(rng.integers(1, min(4, max_step) + 1))
        step = min(step, remaining - (len(order) - idx - 1))
        commands.append((int(color), step))
        remaining -= step

    while remaining > 0:
        step = int(rng.integers(1, min(4, remaining) + 1))
        prev_color = commands[-1][0]
        allowed = [c for c in colors if c != prev_color]
        color = int(rng.choice(allowed))
        commands.append((color, step))
        remaining -= step

    return commands


def _build_left_panel(commands: list[tuple[int, int]], height: int) -> np.ndarray:
    width = int(sum(step for _color, step in commands))
    panel = np.full((height, width), BACKGROUND, dtype=int)
    c = 0
    for color, steps in commands:
        panel[:, c : c + steps] = int(color)
        c += steps
    return panel


def _random_start_and_direction(
    height: int,
    width: int,
    sep_right: int,
    W: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    side = int(rng.integers(0, 4))

    if side == 0 and width >= 3:  # left boundary of right panel
        r = int(rng.integers(1, height - 1))
        c_local = 0
    elif side == 1 and width >= 3:  # top boundary
        r = 0
        c_local = int(rng.integers(1, width - 1))
    elif side == 2 and width >= 3:  # right boundary of right panel
        r = int(rng.integers(1, height - 1))
        c_local = width - 1
    else:  # bottom boundary
        r = height - 1
        c_local = int(rng.integers(1, width - 1)) if width > 2 else 0

    r_abs = int(r)
    c_abs = int(c_local + sep_right + 1)
    direction = _start_direction(r_abs, c_abs, (height, W), sep_right, W)
    side_label = ["left", "top", "right", "bottom"][side % 4]
    return (r, c_local), direction, side_label


def _compose_grid(
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    panel_sep: int = 1,
    sep_color: int = DIVIDER,
) -> np.ndarray:
    H = left_panel.shape[0]
    separator = np.full((H, panel_sep), int(sep_color), dtype=int)
    return np.concatenate([left_panel, separator, right_panel], axis=1)


def _program_from_path_constraints(
    height: int,
    right_w: int,
    sep_right: int,
    W: int,
    commands: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        start, direction, _label = _random_start_and_direction(
            height,
            right_w,
            sep_right,
            W,
            rng,
        )

        try:
            painted, path, _cmds = _paint_path_in_panel(
                (height, right_w),
                start,
                direction,
                commands,
            )
        except Exception:
            continue

        if len(path) <= 1:
            continue

        sr, sc = start
        right_input = np.full((height, right_w), BACKGROUND, dtype=int)
        right_input[sr, sc] = START
        return right_input, painted

    raise RuntimeError("failed to place a non-overlapping path on right panel")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        # 1) choose geometry
        program_width = int(rng.integers(6, 15))
        height = int(rng.integers(8, 19))
        max_right = 29 - program_width - 1
        if max_right < 6:
            continue
        right_w = int(rng.integers(6, max_right + 1))
        total_width = program_width + 1 + right_w
        if total_width > 30 or height > 30 or right_w < 4:
            continue

        # 2) build left program
        commands = _choose_program(program_width, rng)
        if int(sum(step for _c, step in commands)) != program_width:
            continue
        left_panel = _build_left_panel(commands, height)

        # 3) construct right panel with a feasible, non-overlapping path
        try:
            right_panel, expected = _program_from_path_constraints(
                height,
                right_w,
                sep_right=program_width,
                W=total_width,
                commands=commands,
                rng=rng,
            )
        except Exception:
            continue

        grid = _compose_grid(left_panel, right_panel, panel_sep=1)

        # 4) self-consistency check: parsing must recover exactly the painted path.
        if expected.shape != right_panel.shape:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if int(np.sum(expected == START)) < 2:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for strip-program puzzle")


if __name__ == "__main__":
    sample = generate_input()
    print(sample.shape, "=>", main(sample).shape)

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
ARROW = 2
GRID = 5
DIVIDER = 9
PATTERN_COLORS = (1, 3, 4, 6, 7, 8)
SIDES = ("left", "top", "right", "bottom")
GEN_RETRIES = 400


def _scan_order(n: int, side: str) -> list[tuple[int, int]]:
    if side == "left":
        return [(r, c) for c in range(n) for r in range(n)]
    if side == "right":
        return [(r, c) for c in range(n - 1, -1, -1) for r in range(n)]
    if side == "top":
        return [(r, c) for r in range(n) for c in range(n)]
    if side == "bottom":
        return [(r, c) for r in range(n - 1, -1, -1) for c in range(n)]
    raise ValueError(f"unknown side: {side}")


def _find_divider(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    if len(cols) != 1:
        raise ValueError("expected exactly one maroon divider column")
    return cols[0]


def _find_board(grid: np.ndarray, divider_col: int) -> tuple[int, int, int]:
    left = grid[:, :divider_col]
    coords = np.argwhere(left == GRID)
    if coords.size == 0:
        raise ValueError("missing grey board")

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    board = left[r0 : r1 + 1, c0 : c1 + 1]
    full_rows = [r for r in range(board.shape[0]) if np.all(board[r, :] == GRID)]
    full_cols = [c for c in range(board.shape[1]) if np.all(board[:, c] == GRID)]
    if len(full_rows) < 2 or len(full_rows) != len(full_cols):
        raise ValueError("invalid board lattice")
    if full_rows[0] != 0 or full_cols[0] != 0:
        raise ValueError("board frame must start at bbox edge")
    if full_rows[-1] != board.shape[0] - 1 or full_cols[-1] != board.shape[1] - 1:
        raise ValueError("board frame must end at bbox edge")
    if any(b - a != 4 for a, b in zip(full_rows, full_rows[1:])):
        raise ValueError("board rows must be spaced by cell+line size")
    if any(b - a != 4 for a, b in zip(full_cols, full_cols[1:])):
        raise ValueError("board cols must be spaced by cell+line size")

    n = len(full_rows) - 1
    if board.shape != (4 * n + 1, 4 * n + 1):
        raise ValueError("board bbox size does not match lattice")
    return int(r0), int(c0), int(n)


def _find_arrow_side(grid: np.ndarray, board_r0: int, board_c0: int, n: int) -> str:
    coords = np.argwhere(grid == ARROW)
    if coords.size == 0:
        raise ValueError("missing arrow")

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    board_r1 = board_r0 + 4 * n
    board_c1 = board_c0 + 4 * n

    if c1 < board_c0:
        return "left"
    if c0 > board_c1:
        return "right"
    if r1 < board_r0:
        return "top"
    if r0 > board_r1:
        return "bottom"
    raise ValueError("arrow must sit outside exactly one board side")


def _decode_cell(cell: np.ndarray):
    pts = np.argwhere(cell != BG)
    if pts.size == 0:
        return None

    colors = {int(cell[r, c]) for r, c in pts}
    if len(colors) != 1:
        raise ValueError("instruction cell must use one color")
    color = next(iter(colors))

    count = int(len(pts))
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    height = int(max_r - min_r + 1)
    width = int(max_c - min_c + 1)

    if count == 1:
        return ("pen_down", color)
    if count == 4 and height == 2 and width == 2:
        return ("pen_up", color)
    if height == 1 and width == count and count in (2, 3):
        return ("move_right", count, color)
    if width == 1 and height == count and count in (2, 3):
        return ("move_down", count, color)
    raise ValueError("unrecognized instruction glyph")


def _commands_from_grid(grid: np.ndarray, forced_side: str | None = None):
    divider_col = _find_divider(grid)
    board_r0, board_c0, n = _find_board(grid, divider_col)
    side = forced_side or _find_arrow_side(grid, board_r0, board_c0, n)

    commands_by_cell: dict[tuple[int, int], tuple] = {}
    for r in range(n):
        for c in range(n):
            sr = board_r0 + 1 + 4 * r
            sc = board_c0 + 1 + 4 * c
            cell = grid[sr : sr + 3, sc : sc + 3]
            cmd = _decode_cell(cell)
            if cmd is not None:
                commands_by_cell[(r, c)] = cmd

    ordered = [commands_by_cell[pos] for pos in _scan_order(n, side) if pos in commands_by_cell]
    return divider_col, ordered


def _execute_commands(commands: Iterable[tuple], canvas_shape: tuple[int, int]) -> np.ndarray:
    canvas = np.full(canvas_shape, BG, dtype=int)
    h, w = canvas_shape
    row = 0
    col = 0
    pen_down = False

    for cmd in commands:
        kind = cmd[0]
        if kind == "pen_down":
            pen_down = True
            if 0 <= row < h and 0 <= col < w:
                canvas[row, col] = int(cmd[1])
            continue
        if kind == "pen_up":
            pen_down = False
            continue

        steps = int(cmd[1])
        color = int(cmd[2])
        dr, dc = (0, 1) if kind == "move_right" else (1, 0)
        for _ in range(steps):
            row += dr
            col += dc
            if pen_down and 0 <= row < h and 0 <= col < w:
                canvas[row, col] = color

    return canvas


def _solve(grid: np.ndarray, forced_side: str | None = None) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    divider_col, commands = _commands_from_grid(g, forced_side=forced_side)
    canvas = _execute_commands(commands, (g.shape[0], g.shape[1] - divider_col - 1))
    g[:, divider_col + 1 :] = canvas
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a grey lattice board using the unique red arrow as the scan-order cue,
    decode cell glyphs into drawing commands, and execute them on the blank
    canvas to the right of the maroon divider.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_move(rng: random.Random) -> tuple:
    color = rng.choice(PATTERN_COLORS)
    steps = rng.choice((2, 3))
    if rng.random() < 0.5:
        return ("move_right", steps, color)
    return ("move_down", steps, color)


def _generate_command_sequence(rng: random.Random, max_cells: int) -> list[tuple] | None:
    for _ in range(120):
        commands: list[tuple] = []
        draw_moves = 0
        total_right = 0
        total_down = 0
        segments = rng.randint(1, 3)

        for seg_idx in range(segments):
            for _ in range(rng.randint(0, 2)):
                move = _random_move(rng)
                commands.append(move)
                if move[0] == "move_right":
                    total_right += int(move[1])
                else:
                    total_down += int(move[1])

            commands.append(("pen_down", rng.choice(PATTERN_COLORS)))

            for _ in range(rng.randint(1, 3 if segments > 1 else 4)):
                move = _random_move(rng)
                commands.append(move)
                draw_moves += 1
                if move[0] == "move_right":
                    total_right += int(move[1])
                else:
                    total_down += int(move[1])

            if seg_idx < segments - 1 or rng.random() < 0.65:
                commands.append(("pen_up", rng.choice(PATTERN_COLORS)))

        if len(commands) >= max_cells:
            continue
        if draw_moves < 2:
            continue
        if total_right > 16 or total_down > 16:
            continue
        return commands
    return None


def _shape_pixels(cmd: tuple, rng: random.Random) -> list[tuple[int, int]]:
    kind = cmd[0]
    if kind == "pen_down":
        return [(rng.randrange(3), rng.randrange(3))]
    if kind == "pen_up":
        r0 = rng.randrange(2)
        c0 = rng.randrange(2)
        return [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
    if kind == "move_right":
        steps = int(cmd[1])
        row = rng.randrange(3)
        c0 = rng.randrange(4 - steps)
        return [(row, c0 + dc) for dc in range(steps)]
    if kind == "move_down":
        steps = int(cmd[1])
        col = rng.randrange(3)
        r0 = rng.randrange(4 - steps)
        return [(r0 + dr, col) for dr in range(steps)]
    raise ValueError(f"unknown command: {cmd}")


def _draw_arrow(grid: np.ndarray, side: str, top: int, left: int) -> None:
    if side == "left":
        coords = [(0, 0), (1, 0), (2, 0), (1, 1)]
    elif side == "right":
        coords = [(1, 0), (0, 1), (1, 1), (2, 1)]
    elif side == "top":
        coords = [(0, 0), (0, 1), (0, 2), (1, 1)]
    elif side == "bottom":
        coords = [(0, 1), (1, 0), (1, 1), (1, 2)]
    else:
        raise ValueError(f"unknown side: {side}")
    for dr, dc in coords:
        grid[top + dr, left + dc] = ARROW


def _render_candidate(
    rng: random.Random,
    n: int,
    side: str,
    commands: list[tuple],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple], int] | None:
    board_size = 4 * n + 1
    board_top = rng.randint(4, 7)
    board_left = rng.randint(4, 7)
    gap_right = rng.randint(4, 7)
    bottom_margin = rng.randint(4, 7)

    total_right = sum(int(cmd[1]) for cmd in commands if cmd[0] == "move_right")
    total_down = sum(int(cmd[1]) for cmd in commands if cmd[0] == "move_down")
    canvas_w = total_right + rng.randint(4, 7)
    min_h = total_down + rng.randint(4, 7)

    grid_h = max(min_h, board_top + board_size + bottom_margin)
    divider_col = board_left + board_size + gap_right
    grid_w = divider_col + 1 + canvas_w

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, divider_col] = DIVIDER

    board_r1 = board_top + board_size - 1
    board_c1 = board_left + board_size - 1
    for i in range(n + 1):
        rr = board_top + 4 * i
        cc = board_left + 4 * i
        grid[rr, board_left : board_c1 + 1] = GRID
        grid[board_top : board_r1 + 1, cc] = GRID

    if side == "left":
        arrow_top = rng.randint(board_top + 1, board_r1 - 3)
        arrow_left = board_left - 3
    elif side == "right":
        arrow_top = rng.randint(board_top + 1, board_r1 - 3)
        arrow_left = board_c1 + 2
    elif side == "top":
        arrow_top = board_top - 3
        arrow_left = rng.randint(board_left + 1, board_c1 - 3)
    else:
        arrow_top = board_r1 + 2
        arrow_left = rng.randint(board_left + 1, board_c1 - 3)
    _draw_arrow(grid, side, arrow_top, arrow_left)

    order = _scan_order(n, side)
    picked_indices = sorted(rng.sample(range(n * n), len(commands)))
    placements: dict[tuple[int, int], tuple] = {}
    for cmd, idx in zip(commands, picked_indices):
        cell_r, cell_c = order[idx]
        placements[(cell_r, cell_c)] = cmd
        sr = board_top + 1 + 4 * cell_r
        sc = board_left + 1 + 4 * cell_c
        color = int(cmd[1] if cmd[0] in ("pen_down", "pen_up") else cmd[2])
        for dr, dc in _shape_pixels(cmd, rng):
            grid[sr + dr, sc + dc] = color

    return grid, placements, divider_col


def _candidate_is_good(
    grid: np.ndarray,
    n: int,
    placements: dict[tuple[int, int], tuple],
    intended_side: str,
    divider_col: int,
) -> bool:
    intended_canvas = _execute_commands(
        [placements[pos] for pos in _scan_order(n, intended_side) if pos in placements],
        (grid.shape[0], grid.shape[1] - divider_col - 1),
    )
    if int(np.count_nonzero(intended_canvas)) < 5:
        return False

    for other_side in SIDES:
        if other_side == intended_side:
            continue
        other_canvas = _execute_commands(
            [placements[pos] for pos in _scan_order(n, other_side) if pos in placements],
            intended_canvas.shape,
        )
        if np.array_equal(intended_canvas, other_canvas):
            return False

    produced = main(grid)
    if np.array_equal(produced[:, divider_col + 1 :], grid[:, divider_col + 1 :]):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(4, 6)
    side = rng.choice(SIDES)
    commands = _generate_command_sequence(rng, max_cells=n * n - 1)
    if commands is None:
        return None

    rendered = _render_candidate(rng, n, side, commands)
    if rendered is None:
        return None
    grid, placements, divider_col = rendered

    if not _candidate_is_good(grid, n, placements, side, divider_col):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

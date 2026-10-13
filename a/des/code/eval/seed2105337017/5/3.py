from __future__ import annotations

import random

import numpy as np


GREY = 5
AZURE = 8
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6

DIR_COLOR_TO_STEP = {
    RED: (0, 1),
    GREEN: (1, 0),
    YELLOW: (0, -1),
    MAGENTA: (-1, 0),
}
COMMAND_COLORS = tuple(DIR_COLOR_TO_STEP)
BACKGROUND_CHOICES = (0, 1, 7, 9)

GRID_HEIGHT_RANGE = (7, 16)
INSTR_WIDTH_RANGE = (5, 12)
CANVAS_WIDTH_RANGE = (3, 10)
COMMAND_LEN_MIN = 4
COMMAND_LEN_MAX = 24
MAX_GENERATION_ATTEMPTS = 300


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_layout(g: np.ndarray, bg: int) -> tuple[int, Coord, Coord]:
    h, w = g.shape
    candidates: list[tuple[int, Coord, Coord]] = []

    for sep in range(1, w - 1):
        sep_color = int(g[0, sep])
        if sep_color == bg or not np.all(g[:, sep] == sep_color):
            continue

        right = g[:, sep + 1 :]
        if right.size == 0 or int(right[0, 0]) != AZURE:
            continue
        right_non_bg = np.argwhere(right != bg)
        if right_non_bg.shape[0] != 1 or tuple(map(int, right_non_bg[0])) != (0, 0):
            continue

        left = g[:, :sep]
        left_azure = np.argwhere(left == AZURE)
        if left_azure.shape[0] != 1:
            continue

        instr_start = tuple(map(int, left_azure[0]))
        canvas_start = (0, sep + 1)
        candidates.append((sep, instr_start, canvas_start))

    assert len(candidates) == 1, "expected one grey separator and two azure markers"
    return candidates[0]


def _instruction_component(g: np.ndarray, sep: int, start: Coord) -> set[Coord]:
    h = g.shape[0]
    stack = [start]
    seen = {start}
    commands: set[Coord] = set()

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < sep):
                continue
            if (nr, nc) in seen:
                continue
            if int(g[nr, nc]) not in DIR_COLOR_TO_STEP:
                continue
            seen.add((nr, nc))
            commands.add((nr, nc))
            stack.append((nr, nc))

    all_commands = {
        (r, c)
        for r in range(h)
        for c in range(sep)
        if int(g[r, c]) in DIR_COLOR_TO_STEP
    }
    assert commands and commands == all_commands, "instruction colors must form one path"
    return commands


def _ordered_commands(g: np.ndarray, start: Coord, command_cells: set[Coord]) -> list[int]:
    for cell in command_cells:
        degree = sum(neigh in command_cells or neigh == start for neigh in _neighbors4(*cell))
        assert degree in (1, 2), "instruction path must not branch"

    next_cells = [cell for cell in _neighbors4(*start) if cell in command_cells]
    assert len(next_cells) == 1, "azure marker must touch exactly one path cell"

    ordered: list[int] = []
    prev = start
    cur = next_cells[0]
    while True:
        ordered.append(int(g[cur]))
        forward = [cell for cell in _neighbors4(*cur) if cell in command_cells and cell != prev]
        assert len(forward) <= 1, "instruction path traversal must be unique"
        if not forward:
            break
        prev, cur = cur, forward[0]

    assert len(ordered) == len(command_cells), "path traversal must visit every command cell"
    return ordered


def _paint_sequence(g: np.ndarray, sep: int, canvas_start: Coord, commands: list[int]) -> np.ndarray:
    h, w = g.shape
    out = np.array(g, copy=True)
    pos = canvas_start

    for color in commands:
        dr, dc = DIR_COLOR_TO_STEP[color]
        nr = pos[0] + dr
        nc = pos[1] + dc
        if 0 <= nr < h and sep + 1 <= nc < w:
            pos = (nr, nc)
            out[nr, nc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a colored path from the left instruction area and replay it as drawing steps on the right canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep, instr_start, canvas_start = _find_layout(g, bg)
    command_cells = _instruction_component(g, sep, instr_start)
    commands = _ordered_commands(g, instr_start, command_cells)
    return _paint_sequence(g, sep, canvas_start, commands)


def _path_turns(path: list[Coord]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _can_extend(cell: Coord, prev: Coord, used: set[Coord], height: int, width: int) -> bool:
    r, c = cell
    if not (0 <= r < height and 0 <= c < width) or cell in used:
        return False
    for neigh in _neighbors4(r, c):
        if neigh in used and neigh != prev:
            return False
    return True


def _future_options(cell: Coord, used: set[Coord], height: int, width: int) -> int:
    count = 0
    for neigh in _neighbors4(*cell):
        if _can_extend(neigh, cell, used, height, width):
            count += 1
    return count


def _build_path(height: int, width: int, length: int) -> list[Coord] | None:
    starts = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(starts)

    def search(path: list[Coord], used: set[Coord]) -> list[Coord] | None:
        if len(path) == length:
            return path

        cur = path[-1]
        candidates = [
            neigh
            for neigh in _neighbors4(*cur)
            if _can_extend(neigh, cur, used, height, width)
        ]
        random.shuffle(candidates)
        candidates.sort(key=lambda cell: _future_options(cell, used | {cell}, height, width))

        for nxt in candidates:
            found = search(path + [nxt], used | {nxt})
            if found is not None:
                return found
        return None

    for start in starts:
        found = search([start], {start})
        if found is None:
            continue
        turns = _path_turns(found)
        if length <= 5 or turns >= 1:
            return found
    return None


def _simulate_local_canvas(height: int, width: int, commands: list[int]) -> tuple[list[Coord], int]:
    pos = (0, 0)
    painted: list[Coord] = []
    skipped = 0

    for color in commands:
        dr, dc = DIR_COLOR_TO_STEP[color]
        nr = pos[0] + dr
        nc = pos[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            pos = (nr, nc)
            painted.append(pos)
        else:
            skipped += 1

    return painted, skipped


def _sample_enabled_colors(want_invalid: bool) -> tuple[int, ...]:
    while True:
        enabled = tuple(sorted(random.sample(COMMAND_COLORS, random.randint(2, 4))))
        if not ({RED, GREEN} & set(enabled)):
            continue
        if want_invalid and not ({YELLOW, MAGENTA} & set(enabled)):
            continue
        return enabled


def _generate_commands(height: int, width: int, length: int) -> tuple[list[int], int] | None:
    want_invalid = random.random() < 0.5
    enabled = _sample_enabled_colors(want_invalid)
    counts = {color: 0 for color in enabled}
    pos = (0, 0)
    commands: list[int] = []
    invalid_target = random.randint(1, max(1, length // 4)) if want_invalid else 0
    invalid_done = 0

    for i in range(length):
        valid: dict[int, Coord] = {}
        invalid: list[int] = []
        for color in enabled:
            dr, dc = DIR_COLOR_TO_STEP[color]
            nr = pos[0] + dr
            nc = pos[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                valid[color] = (nr, nc)
            else:
                invalid.append(color)

        remaining = length - i
        force_invalid = (
            invalid_target > 0
            and invalid
            and (remaining == invalid_target or random.random() < 0.4)
        )

        if force_invalid:
            pool = invalid
            is_valid_move = False
        elif valid:
            pool = list(valid)
            is_valid_move = True
        elif invalid:
            pool = invalid
            is_valid_move = False
        else:
            return None

        min_count = min(counts[color] for color in pool)
        least_used = [color for color in pool if counts[color] == min_count]
        unused = [color for color in least_used if counts[color] == 0]
        color = random.choice(unused or least_used)

        commands.append(color)
        counts[color] += 1
        if is_valid_move and color in valid:
            pos = valid[color]
        else:
            invalid_done += 1
            if force_invalid:
                invalid_target -= 1

    if set(commands) != set(enabled):
        return None
    if want_invalid and invalid_done == 0:
        return None
    return commands, invalid_done


def _expected_output(grid: np.ndarray, sep: int, commands: list[int]) -> np.ndarray:
    return _paint_sequence(np.asarray(grid, dtype=int), sep, (0, sep + 1), commands)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    instr_width = random.randint(*INSTR_WIDTH_RANGE)
    canvas_width = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.choice(BACKGROUND_CHOICES)

    max_commands = min(COMMAND_LEN_MAX, max(COMMAND_LEN_MIN, (height * instr_width) // 3))
    if max_commands < COMMAND_LEN_MIN:
        return None
    command_len = random.randint(COMMAND_LEN_MIN, max_commands)

    commands_info = _generate_commands(height, canvas_width, command_len)
    if commands_info is None:
        return None
    commands, skipped = commands_info

    painted, _ = _simulate_local_canvas(height, canvas_width, commands)
    distinct_painted = len(set(painted))
    if distinct_painted < min(4, max(2, command_len // 3)):
        return None

    path = _build_path(height, instr_width, command_len + 1)
    if path is None:
        return None

    grid = np.full((height, instr_width + 1 + canvas_width), bg, dtype=int)
    grid[:, instr_width] = GREY
    grid[0, instr_width + 1] = AZURE
    grid[path[0][0], path[0][1]] = AZURE
    for (r, c), color in zip(path[1:], commands):
        grid[r, c] = color

    expected = _expected_output(grid, instr_width, commands)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if _bg_color(grid) != bg:
        return None
    if skipped == 0 and random.random() < 0.25:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

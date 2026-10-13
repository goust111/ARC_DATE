from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8
PATH_COLORS = {GREEN, TEAL}
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Generation patterns: M = interior mirror turn, B = boundary bounce, E = final edge exit.
EVENT_PATTERNS = (
    ("M", "B", "E"),
    ("B", "M", "E"),
    ("M", "B", "M", "E"),
    ("B", "M", "B", "E"),
    ("M", "M", "B", "E"),
    ("B", "M", "M", "E"),
    ("M", "B", "B", "E"),
    ("B", "B", "M", "E"),
)

GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (12, 20)
START_MARGIN = 2
MIN_SEGMENT_STEPS = 2
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _is_diagonal_step(step: tuple[int, int]) -> bool:
    return step in DIAGONALS


def _ordered_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == TEAL)
    assert starts.shape == (1, 2), "expected exactly one teal start pixel"
    start = tuple(map(int, starts[0]))

    path_cells = {tuple(map(int, rc)) for rc in np.argwhere(np.isin(g, list(PATH_COLORS)))}
    assert start in path_cells

    ordered = [start]
    seen = {start}
    prev = None
    cur = start

    while True:
        nxts = []
        for nr, nc in _neighbors8(*cur):
            if not _in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) not in path_cells or (nr, nc) == prev:
                continue
            nxts.append((nr, nc))

        if prev is None:
            assert len(nxts) == 1, "start must launch a unique path"
        else:
            assert len(nxts) <= 1, "path must not branch"

        if not nxts:
            break

        nxt = nxts[0]
        assert nxt not in seen, "path must be a simple chain"
        step = (nxt[0] - cur[0], nxt[1] - cur[1])
        assert _is_diagonal_step(step), "path must use diagonal steps only"
        ordered.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert seen == path_cells, "all path-colored cells must belong to the same chain"
    assert len(ordered) >= 3, "path must have at least one segment beyond the start"
    return ordered


def _turn_records(path: list[tuple[int, int]], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    turns = []
    for idx in range(1, len(path) - 1):
        prev_cell = path[idx - 1]
        cell = path[idx]
        next_cell = path[idx + 1]
        in_dir = (cell[0] - prev_cell[0], cell[1] - prev_cell[1])
        out_dir = (next_cell[0] - cell[0], next_cell[1] - cell[1])
        assert _is_diagonal_step(in_dir) and _is_diagonal_step(out_dir)
        if in_dir == out_dir:
            continue
        r, c = cell
        turns.append(
            {
                "coord": cell,
                "in_dir": in_dir,
                "out_dir": out_dir,
                "boundary": r in (0, h - 1) or c in (0, w - 1),
            }
        )
    return turns


def _mirror_target(turn: dict) -> tuple[int, int]:
    r, c = turn["coord"]
    dr, dc = turn["in_dir"]
    return r + dr, c + dc


def _flip_kind(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> tuple[bool, bool]:
    return in_dir[0] != out_dir[0], in_dir[1] != out_dir[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden mirror cells from a diagonal light trail with mixed turn causes.

    Input structure:
    - The grid has a black background, exactly one teal start pixel, and one green
      single-cell-wide diagonal polyline leaving that start.
    - The polyline may turn at internal cells because of invisible mirrors, or at edge
      cells because of boundary bounces, and it finally terminates on the grid edge.
    - The mirrors themselves do not appear anywhere in the input.

    Rule:
    - Copy the input.
    - Trace the path from the teal start to the edge endpoint and find every turn, i.e.
      every path cell whose incoming and outgoing diagonal directions differ.
    - For each turn that lies strictly inside the grid, continue one more step in the
      incoming direction past the turn and paint that cell red.
    - Ignore turns whose turn cell is on the outer boundary row or column.

    Key insight:
    - Only internal turns correspond to hidden mirrors, and each such mirror sits in the
      cell the ray would have entered next if it had not reflected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREEN, TEAL}), "unexpected colors present"

    path = _ordered_path(g)
    turns = _turn_records(path, g.shape)
    assert turns, "expected at least one turn"

    out = g.copy()
    for turn in turns:
        if turn["boundary"]:
            continue
        tr, tc = turn["coord"]
        assert 0 < tr < g.shape[0] - 1 and 0 < tc < g.shape[1] - 1
        row_flip, col_flip = _flip_kind(turn["in_dir"], turn["out_dir"])
        assert row_flip ^ col_flip, "internal mirror reflection must flip exactly one component"
        mr, mc = _mirror_target(turn)
        assert _in_bounds(g.shape, mr, mc)
        out[mr, mc] = RED

    return out


def _boundary_event(
    cell: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, tuple[int, int], bool, bool]:
    h, w = shape
    r, c = cell
    dr, dc = direction
    row_steps = (h - 1 - r) if dr > 0 else r
    col_steps = (w - 1 - c) if dc > 0 else c
    steps = min(row_steps, col_steps)
    event_cell = (r + dr * steps, c + dc * steps)
    return steps, event_cell, row_steps == steps, col_steps == steps


def _segment_cells(
    start: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _cell_is_clear(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    prev_cell: tuple[int, int] | None,
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c):
        return False
    if cell in occupied or cell in reserved:
        return False
    adjacent = {nb for nb in _neighbors8(r, c) if nb in occupied}
    return adjacent == ({prev_cell} if prev_cell is not None else set())


def _segment_is_clear(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    trial_occupied = set(occupied)
    cells = []
    prev = start
    for cell in _segment_cells(start, direction, steps):
        if not _cell_is_clear(cell, trial_occupied, prev, reserved, shape):
            return None
        cells.append(cell)
        trial_occupied.add(cell)
        prev = cell
    return cells


def _mirror_spot_is_clear(
    mirror_cell: tuple[int, int],
    turn_cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if not _in_bounds(shape, *mirror_cell):
        return False
    if mirror_cell in occupied or mirror_cell in reserved:
        return False
    adjacent = {nb for nb in _neighbors8(*mirror_cell) if nb in occupied}
    return adjacent == {turn_cell}


def _steps_until_boundary(
    cell: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> int:
    steps, _event, _row_hit, _col_hit = _boundary_event(cell, direction, shape)
    return steps


def _append_segment(
    grid: np.ndarray,
    ordered_path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    cells: list[tuple[int, int]],
) -> None:
    for cell in cells:
        ordered_path.append(cell)
        occupied.add(cell)
        grid[cell] = GREEN


def _choose_internal_turn(
    cur: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    shape: tuple[int, int],
    need_more_room: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    max_steps, _event_cell, _row_hit, _col_hit = _boundary_event(cur, direction, shape)
    choices = []
    for steps in range(MIN_SEGMENT_STEPS, max_steps):
        segment = _segment_is_clear(cur, direction, steps, occupied, reserved, shape)
        if segment is None:
            continue
        turn_cell = segment[-1]
        tr, tc = turn_cell
        if not (0 < tr < shape[0] - 1 and 0 < tc < shape[1] - 1):
            continue
        mirror_cell = (tr + direction[0], tc + direction[1])
        if not (0 < mirror_cell[0] < shape[0] - 1 and 0 < mirror_cell[1] < shape[1] - 1):
            continue
        extended_occupied = occupied | set(segment)
        if not _mirror_spot_is_clear(mirror_cell, turn_cell, extended_occupied, reserved, shape):
            continue

        for out_dir in ((direction[0], -direction[1]), (-direction[0], direction[1])):
            needed = MIN_SEGMENT_STEPS if need_more_room else 1
            if _steps_until_boundary(turn_cell, out_dir, shape) < needed:
                continue
            first_out = (turn_cell[0] + out_dir[0], turn_cell[1] + out_dir[1])
            if not _cell_is_clear(first_out, extended_occupied, turn_cell, reserved | {mirror_cell}, shape):
                continue
            choices.append((segment, turn_cell, mirror_cell, out_dir))

    if not choices:
        return None
    return random.choice(choices)


def _sample_start(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return (
        random.randint(START_MARGIN, h - START_MARGIN - 1),
        random.randint(START_MARGIN, w - START_MARGIN - 1),
    )


def _expected_mirror_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {_mirror_target(turn) for turn in _turn_records(_ordered_path(grid), grid.shape) if not turn["boundary"]}


def _validate_turn_semantics(grid: np.ndarray, output: np.ndarray) -> bool:
    try:
        path = _ordered_path(grid)
    except AssertionError:
        return False

    turns = _turn_records(path, grid.shape)
    if not turns:
        return False

    h, w = grid.shape
    boundary_turns = 0
    interior_turns = 0
    expected_reds = set()
    for turn in turns:
        r, c = turn["coord"]
        row_flip, col_flip = _flip_kind(turn["in_dir"], turn["out_dir"])
        if turn["boundary"]:
            boundary_turns += 1
            if r in (0, h - 1) and c in (0, w - 1):
                return False
            if (r in (0, h - 1)) == (c in (0, w - 1)):
                return False
            if r in (0, h - 1):
                if not (row_flip and not col_flip):
                    return False
            else:
                if not (col_flip and not row_flip):
                    return False
        else:
            interior_turns += 1
            if row_flip == col_flip:
                return False
            target = _mirror_target(turn)
            if not (0 < target[0] < h - 1 and 0 < target[1] < w - 1):
                return False
            if grid[target] != BLACK:
                return False
            expected_reds.add(target)

    if boundary_turns == 0 or interior_turns == 0:
        return False

    actual_reds = {
        tuple(map(int, rc))
        for rc in np.argwhere((output == RED) & (grid != RED))
    }
    return actual_reds == expected_reds


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    pattern = random.choice(EVENT_PATTERNS)

    start = _sample_start(grid.shape)
    direction = random.choice(DIAGONALS)
    ordered_path = [start]
    occupied = {start}
    reserved_mirrors: set[tuple[int, int]] = set()

    grid[start] = TEAL
    cur = start
    cur_dir = direction

    for idx, event in enumerate(pattern):
        need_more_room = idx < len(pattern) - 1

        if event == "M":
            choice = _choose_internal_turn(
                cur,
                cur_dir,
                occupied,
                reserved_mirrors,
                grid.shape,
                need_more_room,
            )
            if choice is None:
                return None
            segment, turn_cell, mirror_cell, out_dir = choice
            _append_segment(grid, ordered_path, occupied, segment)
            reserved_mirrors.add(mirror_cell)
            cur = turn_cell
            cur_dir = out_dir
            continue

        steps, event_cell, row_hit, col_hit = _boundary_event(cur, cur_dir, grid.shape)
        if steps < MIN_SEGMENT_STEPS:
            return None
        if event == "B" and row_hit == col_hit:
            return None
        segment = _segment_is_clear(cur, cur_dir, steps, occupied, reserved_mirrors, grid.shape)
        if segment is None:
            return None
        _append_segment(grid, ordered_path, occupied, segment)
        cur = event_cell

        if event == "E":
            break

        if row_hit:
            out_dir = (-cur_dir[0], cur_dir[1])
        else:
            out_dir = (cur_dir[0], -cur_dir[1])
        if _steps_until_boundary(cur, out_dir, grid.shape) < (MIN_SEGMENT_STEPS if need_more_room else 1):
            return None
        first_out = (cur[0] + out_dir[0], cur[1] + out_dir[1])
        if not _cell_is_clear(first_out, occupied, cur, reserved_mirrors, grid.shape):
            return None
        cur_dir = out_dir

    if cur not in {
        (0, cur[1]),
        (h - 1, cur[1]),
        (cur[0], 0),
        (cur[0], w - 1),
    }:
        return None

    try:
        output = main(grid)
    except AssertionError:
        return None
    if not _validate_turn_semantics(grid, output):
        return None
    if not np.array_equal(output, main(grid)):
        return None
    if len(_expected_mirror_cells(grid)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-trace input")

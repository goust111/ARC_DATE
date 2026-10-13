from __future__ import annotations

import random

import numpy as np


BG = 0
START = 3
ALT = 4
BLOCK = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_SIZE = 13
MAX_SIZE = 18
MIN_TURNS = 3
MAX_TURNS = 5
MIN_TOTAL_STEPS = 15
MAX_TOTAL_STEPS = 25
MIN_FINAL_SEG = 2
MAX_SEG_LEN = 6
EXTRA_BLOCKER_RANGE = (0, 2)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perpendiculars(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (-dc, dr), (dc, -dr)


def _find_start_bar(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == START)]
    assert len(coords) == 2, "expected exactly two start-bar pixels"
    a, b = coords
    assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "start bar must be 4-connected"

    h, w = grid.shape

    def is_border(cell: tuple[int, int]) -> bool:
        r, c = cell
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    border = [cell for cell in coords if is_border(cell)]
    assert len(border) == 1, "expected exactly one border start pixel"
    border_cell = border[0]
    inner_cell = b if a == border_cell else a
    direction = (inner_cell[0] - border_cell[0], inner_cell[1] - border_cell[1])
    assert abs(direction[0]) + abs(direction[1]) == 1
    return border_cell, inner_cell, direction


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    border, inner, direction = _find_start_bar(grid)
    dr, dc = direction
    h, w = grid.shape

    positions = [border, inner]
    turn_start_indices: list[int] = []
    pending_turn_start = False
    seen_states = {(border[0], border[1], dr, dc), (inner[0], inner[1], dr, dc)}

    r, c = inner
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            break

        if int(grid[nr, nc]) == BLOCK:
            choices = []
            for tdr, tdc in _perpendiculars(dr, dc):
                tr, tc = r + tdr, c + tdc
                if _in_bounds((h, w), tr, tc) and int(grid[tr, tc]) == BG:
                    choices.append((tdr, tdc))
            assert len(choices) == 1, "each blocker collision must force one turn"
            dr, dc = choices[0]
            pending_turn_start = True
            continue

        state = (nr, nc, dr, dc)
        assert state not in seen_states, "path walk must not cycle"
        seen_states.add(state)
        positions.append((nr, nc))
        if pending_turn_start:
            turn_start_indices.append(len(positions) - 1)
            pending_turn_start = False
        r, c = nr, nc

    return positions, turn_start_indices


def _paint_output(grid: np.ndarray, reset_on_turn: bool) -> np.ndarray:
    positions, turn_start_indices = _trace_path(grid)
    out = np.array(grid, dtype=int, copy=True)

    out[positions[0]] = START
    out[positions[1]] = ALT

    if not reset_on_turn:
        for idx, cell in enumerate(positions[2:], start=2):
            out[cell] = START if idx % 2 == 0 else ALT
        return out

    turn_starts = set(turn_start_indices)
    counter = 2
    for idx, cell in enumerate(positions[2:], start=2):
        if idx in turn_starts:
            counter = 0
        out[cell] = START if counter % 2 == 0 else ALT
        counter += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the blocker-guided corridor and alternately recolor each straight segment, resetting the alternation at every turn.

    Input structure:
    - The grid is black background with teal blocker pixels and one green 2-pixel start bar.
    - Exactly one start-bar pixel lies on the grid border and the other lies one step inward.
    - The hidden corridor is a single-pixel path that goes straight through black cells, turns only when a forward blocker is hit, and finally exits the grid.

    Rule:
    - Find the border start pixel and the inward start pixel, then move from the border pixel through the inward one.
    - Keep the border start pixel green and recolor the inward start pixel yellow.
    - Continue stepping through non-blocker cells; within the current straight segment, color visited cells green, yellow, green, yellow, ...
    - When the next cell is a teal blocker, do not paint the blocker; instead turn to the unique perpendicular black cell, reset the local alternation counter, and continue.
    - Stop when the next forward step would leave the grid; all non-path background and all blockers stay unchanged.

    Key insight:
    - The path must be reconstructed from the blocker-forced turns, and the color alternation is local to each straight segment rather than one global parity pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _paint_output(g, reset_on_turn=True)


def _choose_entry(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        c = random.randint(1, w - 2)
        return (0, c), (1, c), (1, 0)
    if side == "bottom":
        c = random.randint(1, w - 2)
        return (h - 1, c), (h - 2, c), (-1, 0)
    if side == "left":
        r = random.randint(1, h - 2)
        return (r, 0), (r, 1), (0, 1)
    r = random.randint(1, h - 2)
    return (r, w - 1), (r, w - 2), (0, -1)


def _cells_along(pos: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _dist_until_exit(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    count = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return count
        count += 1


def _segment_lengths(path: list[tuple[int, int]], turn_start_indices: list[int]) -> list[int]:
    lengths: list[int] = []
    prev_start = 0
    for turn_start in turn_start_indices:
        lengths.append((turn_start - 1) - prev_start)
        prev_start = turn_start - 1
    lengths.append((len(path) - 1) - prev_start)
    return lengths


def _candidate_lengths(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> list[int]:
    max_forward = _dist_until_exit(shape, pos, direction) - 1
    if max_forward < 1:
        return []
    lo = 1
    hi = min(MAX_SEG_LEN, max_forward)
    lengths = list(range(lo, hi + 1))
    random.shuffle(lengths)
    return lengths


def _extra_blocker_sites(
    shape: tuple[int, int],
    path_cells: set[tuple[int, int]],
    blocker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    sites = []
    forbidden = set(path_cells) | set(blocker_cells)
    halo = set(forbidden)
    for r, c in forbidden:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                halo.add((rr, cc))
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell not in halo:
                sites.append(cell)
    random.shuffle(sites)
    return sites


def _build_candidate() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    shape = (h, w)
    total_turns = random.randint(MIN_TURNS, MAX_TURNS)

    border, inner, direction = _choose_entry(shape)
    base_path = [border, inner]
    path_cells = set(base_path)
    blocker_cells: set[tuple[int, int]] = set()

    def extend(
        pos: tuple[int, int],
        current_dir: tuple[int, int],
        turns_done: int,
        path: list[tuple[int, int]],
        occupied_path: set[tuple[int, int]],
        occupied_blockers: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
        steps_so_far = len(path) - 1
        turns_left = total_turns - turns_done

        if turns_left == 0:
            final_len = _dist_until_exit(shape, pos, current_dir)
            if final_len < MIN_FINAL_SEG:
                return None
            cells = _cells_along(pos, current_dir, final_len)
            if any(cell in occupied_path or cell in occupied_blockers for cell in cells):
                return None
            total_steps = steps_so_far + final_len
            if not (MIN_TOTAL_STEPS <= total_steps <= MAX_TOTAL_STEPS):
                return None
            return path + cells, occupied_blockers

        next_dirs = list(_perpendiculars(*current_dir))
        random.shuffle(next_dirs)
        for seg_len in _candidate_lengths(shape, pos, current_dir):
            future_min = (turns_left - 1) + MIN_FINAL_SEG
            if steps_so_far + seg_len + future_min > MAX_TOTAL_STEPS:
                continue

            segment_cells = _cells_along(pos, current_dir, seg_len)
            if any(cell in occupied_path or cell in occupied_blockers for cell in segment_cells):
                continue
            endpoint = segment_cells[-1]
            ahead = (endpoint[0] + current_dir[0], endpoint[1] + current_dir[1])
            if not _in_bounds(shape, *ahead):
                continue
            if ahead in occupied_path or ahead in occupied_blockers:
                continue

            for next_dir in next_dirs:
                other_dir = (-next_dir[0], -next_dir[1])
                next_cell = (endpoint[0] + next_dir[0], endpoint[1] + next_dir[1])
                if not _in_bounds(shape, *next_cell):
                    continue
                if next_cell in occupied_path or next_cell in occupied_blockers:
                    continue

                wrong_cell = (endpoint[0] + other_dir[0], endpoint[1] + other_dir[1])
                new_blockers = {ahead}
                if _in_bounds(shape, *wrong_cell):
                    if wrong_cell in occupied_path or wrong_cell in occupied_blockers:
                        continue
                    new_blockers.add(wrong_cell)

                new_path = path + segment_cells
                new_path_cells = occupied_path | set(segment_cells)
                new_blocker_cells = occupied_blockers | new_blockers
                result = extend(endpoint, next_dir, turns_done + 1, new_path, new_path_cells, new_blocker_cells)
                if result is not None:
                    return result
        return None

    built = extend(inner, direction, 0, base_path, path_cells, blocker_cells)
    if built is None:
        return None
    path, blockers = built

    grid = np.full(shape, BG, dtype=int)
    grid[border] = START
    grid[inner] = START
    for r, c in blockers:
        grid[r, c] = BLOCK

    extra_sites = _extra_blocker_sites(shape, set(path), blockers)
    for cell in extra_sites[: random.randint(*EXTRA_BLOCKER_RANGE)]:
        grid[cell] = BLOCK

    path_positions, turn_starts = _trace_path(grid)
    if path_positions != path:
        return None

    total_steps = len(path_positions) - 1
    if not (MIN_TOTAL_STEPS <= total_steps <= MAX_TOTAL_STEPS):
        return None
    if len(turn_starts) != total_turns:
        return None

    if np.array_equal(_paint_output(grid, True), _paint_output(grid, False)):
        return None

    seg_lengths = _segment_lengths(path_positions, turn_starts)
    if len(seg_lengths) != total_turns + 1:
        return None
    if any(length <= 0 for length in seg_lengths):
        return None
    if sum(length % 2 for length in seg_lengths[:-1]) == 0 and seg_lengths[-1] % 2 == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-reset sample")

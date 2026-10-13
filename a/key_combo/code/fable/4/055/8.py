import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_LEFT = {DIRS[i]: DIRS[(i - 1) % 4] for i in range(4)}
TURN_RIGHT = {DIRS[i]: DIRS[(i + 1) % 4] for i in range(4)}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _iter_line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _find_arrow_cells(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(cells) == 2, "expected exactly two green arrow cells"
    (r0, c0), (r1, c1) = cells
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "green arrow must be a domino"
    return cells[0], cells[1]


def _candidate_orientations(cells: tuple[tuple[int, int], tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    a, b = cells
    return [
        (a, (a[0] - b[0], a[1] - b[1])),
        (b, (b[0] - a[0], b[1] - a[1])),
    ]


def _simulate_path(
    grid: np.ndarray,
    head: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    h, w = grid.shape
    r, c = head
    dr, dc = direction
    painted: list[tuple[int, int]] = []
    pivots_hit = 0
    visited_states = {(r, c, dr, dc)}

    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return {
                "valid": True,
                "painted": painted,
                "pivots_hit": pivots_hit,
                "steps": len(painted) + pivots_hit,
            }

        cell = int(grid[nr, nc])
        if cell == BLACK:
            painted.append((nr, nc))
        elif cell == BLUE:
            pivots_hit += 1
            dr, dc = TURN_LEFT[(dr, dc)]
        elif cell == RED:
            pivots_hit += 1
            dr, dc = TURN_RIGHT[(dr, dc)]
        else:
            return {"valid": False}

        r, c = nr, nc
        state = (r, c, dr, dc)
        if state in visited_states:
            return {"valid": False}
        visited_states.add(state)


def _best_orientation(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], dict]:
    cells = _find_arrow_cells(grid)
    scored = []
    for head, direction in _candidate_orientations(cells):
        sim = _simulate_path(grid, head, direction)
        if not sim["valid"]:
            continue
        score = (sim["pivots_hit"], sim["steps"])
        scored.append((score, head, direction, sim))

    assert scored, "expected at least one valid arrow orientation"
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, head, direction, sim = scored[0]
    tied = [item for item in scored if item[0] == best_score]
    assert len(tied) == 1, "arrow orientation must be uniquely recoverable"
    return head, direction, sim


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the green domino arrow through colored pivots and paint the traversed background cells yellow.

    Input structure:
    - The background is black.
    - There is exactly one green domino arrow made of two orthogonally adjacent cells.
    - The remaining non-background cells are isolated blue or red pivot pixels scattered on the canvas.

    Rule:
    - Interpret the domino as a directed arrow: one green cell is the head and the ray continues forward along the domino axis.
    - Starting from the head, move one cell at a time; blue pivots turn the ray 90 degrees left and red pivots turn it 90 degrees right.
    - Continue until the ray leaves the grid, painting every traversed background cell yellow while keeping pivots and the arrow unchanged.

    Key insight:
    - Only pivots that the evolving ray actually reaches matter; each turn changes which later pivots become active, so distractor pivots stay inert.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREEN})

    _head, _direction, sim = _best_orientation(g)
    out = g.copy()
    for r, c in sim["painted"]:
        out[r, c] = YELLOW
    return out


GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
TURN_COUNT_RANGE = (2, 5)
DISTRACTOR_COUNT_RANGE = (1, 5)
SEGMENT_LENGTH_RANGE = (2, 6)
FINAL_SEGMENT_MIN = 2
GEN_ATTEMPTS = 400


def _clear_run(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        if (r, c) in occupied:
            return []
        cells.append((r, c))


def _choose_turn_sides(turn_count: int) -> list[int]:
    sides = [random.choice((-1, 1)) for _ in range(turn_count)]
    if turn_count >= 2 and len(set(sides)) == 1 and random.random() < 0.8:
        sides[random.randrange(turn_count)] *= -1
    return sides


def _carve_path(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], list[int]] | None:
    h, w = shape
    for _ in range(300):
        direction = random.choice(DIRS)
        dr, dc = direction
        head = (random.randint(1, h - 2), random.randint(1, w - 2))
        tail = (head[0] - dr, head[1] - dc)
        if not _in_bounds(shape, tail[0], tail[1]):
            continue

        occupied = {tail, head}
        path_cells: list[tuple[int, int]] = []
        pos = head
        current_dir = direction
        turn_sides = _choose_turn_sides(random.randint(*TURN_COUNT_RANGE))
        success = True

        for idx, side in enumerate(turn_sides):
            next_dir = TURN_LEFT[current_dir] if side == -1 else TURN_RIGHT[current_dir]
            clear_ahead = _clear_run(shape, pos, current_dir, occupied)
            if len(clear_ahead) < SEGMENT_LENGTH_RANGE[0]:
                success = False
                break

            max_len = min(SEGMENT_LENGTH_RANGE[1], len(clear_ahead))
            candidates = []
            for length in range(SEGMENT_LENGTH_RANGE[0], max_len + 1):
                turn_cell = clear_ahead[length - 1]
                after_turn = _clear_run(shape, turn_cell, next_dir, occupied | set(clear_ahead[:length]))
                min_needed = SEGMENT_LENGTH_RANGE[0] if idx < len(turn_sides) - 1 else FINAL_SEGMENT_MIN
                if len(after_turn) >= min_needed:
                    candidates.append(length)
            if not candidates:
                success = False
                break

            length = random.choice(candidates)
            segment = clear_ahead[:length]
            path_cells.extend(segment)
            occupied.update(segment)
            pos = segment[-1]
            current_dir = next_dir

        if not success:
            continue

        final_cells = _clear_run(shape, pos, current_dir, occupied)
        if len(final_cells) < FINAL_SEGMENT_MIN:
            continue
        path_cells.extend(final_cells)

        return path_cells, head, direction, turn_sides

    return None


def _turn_color(incoming: tuple[int, int], side: int) -> int:
    return BLUE if side == -1 else RED


def _path_to_grid(
    shape: tuple[int, int],
    path_cells: list[tuple[int, int]],
    head: tuple[int, int],
    direction: tuple[int, int],
    turn_sides: list[int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    dr, dc = direction
    tail = (head[0] - dr, head[1] - dc)
    grid[tail] = GREEN
    grid[head] = GREEN

    current_dir = direction
    pos = head
    path_index = 0
    for side in turn_sides:
        while path_index < len(path_cells):
            cell = path_cells[path_index]
            path_index += 1
            pos = cell
            next_dir = TURN_LEFT[current_dir] if side == -1 else TURN_RIGHT[current_dir]
            nr, nc = pos[0] + next_dir[0], pos[1] + next_dir[1]
            if path_index < len(path_cells) and path_cells[path_index] == (nr, nc):
                grid[pos] = _turn_color(current_dir, side)
                current_dir = next_dir
                break

    return grid


def _add_distractors(
    grid: np.ndarray,
    forbidden: Iterable[tuple[int, int]],
) -> None:
    forbidden_set = set(forbidden)
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden_set or int(grid[r, c]) != BLACK:
                continue
            if any((r + dr, c + dc) in forbidden_set for dr, dc in DIRS):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    count = min(random.randint(*DISTRACTOR_COUNT_RANGE), len(candidates))
    for r, c in candidates[:count]:
        grid[r, c] = random.choice((BLUE, RED))


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )
    carved = _carve_path(shape)
    if carved is None:
        return None

    path_cells, head, direction, turn_sides = carved
    grid = _path_to_grid(shape, path_cells, head, direction, turn_sides)
    tail = (head[0] - direction[0], head[1] - direction[1])
    _add_distractors(grid, {tail, head, *path_cells})

    try:
        best_head, best_dir, best_sim = _best_orientation(grid)
    except AssertionError:
        return None

    if best_head != head or best_dir != direction:
        return None
    if best_sim["pivots_hit"] != len(turn_sides):
        return None
    if best_sim["pivots_hit"] < 2:
        return None

    cells = _find_arrow_cells(grid)
    other = None
    for cand_head, cand_dir in _candidate_orientations(cells):
        if cand_head != head:
            other = _simulate_path(grid, cand_head, cand_dir)
            break
    if other is None or not other["valid"]:
        return grid
    if (other["pivots_hit"], other["steps"]) >= (best_sim["pivots_hit"], best_sim["steps"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-deflection input")

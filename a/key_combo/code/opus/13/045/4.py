import random
from itertools import permutations, product

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS_URDL)}

_CANDIDATE_POOLS = None
_GEN_COUNTER = 0


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIR_TO_INDEX[direction] + 1) % 4]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIR_TO_INDEX[direction] - 1) % 4]


def _sign(value: int) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS_URDL:
        yield r + dr, c + dc


def _infer_lattice(grid: np.ndarray) -> dict:
    red_coords = np.argwhere(grid == RED)
    assert len(red_coords) == 16, "expected a 4x4 red lattice"

    red_rows = sorted({int(r) for r, _c in red_coords})
    red_cols = sorted({int(c) for _r, c in red_coords})
    assert len(red_rows) == 4 and len(red_cols) == 4
    assert all(grid[r, c] == RED for r in red_rows for c in red_cols)

    row_pitchs = [b - a for a, b in zip(red_rows, red_rows[1:])]
    col_pitchs = [b - a for a, b in zip(red_cols, red_cols[1:])]
    assert len(set(row_pitchs)) == 1 and len(set(col_pitchs)) == 1
    row_pitch = row_pitchs[0]
    col_pitch = col_pitchs[0]
    assert row_pitch == col_pitch, "expected square lattice pitch"
    pitch = row_pitch

    h, w = map(int, grid.shape)
    gap_r = 4 * pitch - h
    gap_c = 4 * pitch - w
    assert gap_r == gap_c
    gap = gap_r
    cell_size = pitch - gap
    assert gap in (0, 1), "generator uses 0/1 separators"
    assert cell_size >= 4 and cell_size % 2 == 1

    row_starts = [int(r - cell_size // 2) for r in red_rows]
    col_starts = [int(c - cell_size // 2) for c in red_cols]
    assert row_starts[0] == 0 and col_starts[0] == 0
    assert all(b - a == pitch for a, b in zip(row_starts, row_starts[1:]))
    assert all(b - a == pitch for a, b in zip(col_starts, col_starts[1:]))
    assert row_starts[-1] + cell_size == h
    assert col_starts[-1] + cell_size == w

    return {
        "pitch": pitch,
        "gap": gap,
        "cell_size": cell_size,
        "row_starts": row_starts,
        "col_starts": col_starts,
        "red_set": {tuple(map(int, rc)) for rc in red_coords},
    }


def _find_cross(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    teal_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == TEAL)]
    assert teal_coords, "expected a teal cross"

    degrees = {}
    for r, c in teal_coords:
        degrees[(r, c)] = sum(
            1
            for nr, nc in _neighbors4(r, c)
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == TEAL
        )

    centers = [coord for coord, degree in degrees.items() if degree == 4]
    tips = [coord for coord, degree in degrees.items() if degree == 1]
    assert len(centers) == 1 and len(tips) == 4
    center = centers[0]

    arms = []
    for tip in tips:
        dr = _sign(tip[0] - center[0])
        dc = _sign(tip[1] - center[1])
        outward = (dr, dc)
        assert outward in DIR_TO_INDEX and abs(dr) + abs(dc) == 1
        arms.append((tip, outward))

    arms.sort(key=lambda item: DIR_TO_INDEX[item[1]])
    return center, arms


def _near_red(r: int, c: int, red_set: set[tuple[int, int]]) -> bool:
    for rr, cc in red_set:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _point_to_cell(r: int, c: int, lattice: dict) -> tuple[int, int] | None:
    cell_size = lattice["cell_size"]
    for br, rs in enumerate(lattice["row_starts"]):
        if rs <= r < rs + cell_size:
            for bc, cs in enumerate(lattice["col_starts"]):
                if cs <= c < cs + cell_size:
                    return br, bc
            return None
    return None


def _simulate_single_spiral(
    out: np.ndarray,
    start: tuple[int, int],
    initial_heading: tuple[int, int],
    lattice: dict,
) -> tuple[list[tuple[int, int]], str]:
    cur = start
    heading = initial_heading
    just_turned = False
    visited: list[tuple[int, int]] = []

    while True:
        chosen = None
        turned = False
        for direction in (_turn_right(heading), heading, _turn_left(heading)):
            nr = cur[0] + direction[0]
            nc = cur[1] + direction[1]
            if _in_bounds(out.shape, nr, nc) and out[nr, nc] == BLACK:
                chosen = direction
                turned = direction != heading
                break
        if chosen is None:
            return visited, "boxed"

        segment_len = 1 if (turned or just_turned) else 2
        heading = chosen
        just_turned = turned

        for _ in range(segment_len):
            nr = cur[0] + chosen[0]
            nc = cur[1] + chosen[1]
            if not _in_bounds(out.shape, nr, nc):
                return visited, "boundary"
            if out[nr, nc] != BLACK:
                return visited, "nonblack"
            if _near_red(nr, nc, lattice["red_set"]):
                return visited, "red"
            out[nr, nc] = TEAL
            cur = (nr, nc)
            visited.append(cur)


def _solve_with_order(
    grid: np.ndarray,
    arm_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[str, ...], tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    lattice = _infer_lattice(g)
    _center, arms = _find_cross(g)
    out = g.copy()

    if arm_order is None:
        arm_indices = tuple(range(len(arms)))
    else:
        arm_indices = tuple(arm_order)
        assert len(arm_indices) == len(arms)

    visited_all: list[tuple[int, int]] = []
    reasons: list[str] = []
    for arm_index in arm_indices:
        tip, outward = arms[arm_index]
        heading = _turn_right(outward)
        visited, reason = _simulate_single_spiral(out, tip, heading, lattice)
        visited_all.extend(visited)
        reasons.append(reason)

    touched_cells = {
        cell
        for r, c in visited_all
        if (cell := _point_to_cell(r, c, lattice)) is not None
    }
    for br, bc in touched_cells:
        mark_r = lattice["row_starts"][br] + br
        mark_c = lattice["col_starts"][bc] + bc
        assert br < lattice["cell_size"] and bc < lattice["cell_size"]
        out[mark_r, mark_c] = YELLOW

    return (
        out,
        tuple(sorted(touched_cells)),
        tuple(reasons),
        tuple(visited_all),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Grow teal spirals from a central cross until the red lattice repels them, then mark each activated cell by its lattice coordinate.

    Input structure:
    - The background is black.
    - Red pixels form a complete 4x4 lattice of single-pixel cell-center markers for equal square cells, with optional 1-pixel separators.
    - A teal plus-shaped cross sits near the middle of the grid; its four arm tips are the spiral launch points.

    Rule:
    - From each arm tip, launch a teal spiral whose initial heading is the arm's outward direction rotated 90 degrees clockwise.
    - At each step, the spiral prefers right turn, then straight, then left; a normal segment has length 2, while the first segment after a turn has length 1.
    - While traversing the chosen segment one pixel at a time, stop that spiral immediately before the first pixel that would leave the grid, hit any non-black pixel, or enter the 8-neighborhood of a red lattice marker.
    - After all spirals finish, every lattice cell touched by at least one newly drawn teal pixel receives a yellow pixel at local position (cell_row_index, cell_col_index) inside that cell; the yellow mark sits on top if the slot is already occupied.

    Key insight:
    - The red lattice has a double role: it both decides where each spiral halts and supplies the cell-address system that determines where the yellow mark is placed inside each activated cell.
    """
    return _solve_with_order(np.asarray(grid, dtype=int))[0]


def _build_input(gap: int, center: tuple[int, int], arm_lengths: tuple[int, int, int, int]) -> np.ndarray | None:
    cell_size = 5
    size = 4 * cell_size + 3 * gap
    pitch = cell_size + gap
    marker_positions = [i * pitch + cell_size // 2 for i in range(4)]

    grid = np.zeros((size, size), dtype=int)
    for r in marker_positions:
        for c in marker_positions:
            grid[r, c] = RED

    teal_cells = [center]
    for direction, length in zip(DIRS_URDL, arm_lengths):
        for step in range(1, length + 1):
            rr = center[0] + direction[0] * step
            cc = center[1] + direction[1] * step
            teal_cells.append((rr, cc))

    if any(not _in_bounds(grid.shape, r, c) for r, c in teal_cells):
        return None
    if len(set(teal_cells)) != len(teal_cells):
        return None
    if any(grid[r, c] != BLACK for r, c in teal_cells):
        return None

    for r, c in teal_cells:
        grid[r, c] = TEAL
    return grid


def _collect_candidates() -> dict[str, list[dict]]:
    candidates: list[dict] = []
    for gap in (0, 1):
        size = 20 + 3 * gap
        center_values = range(size // 2 - 3, size // 2 + 4)
        for center in product(center_values, repeat=2):
            for arm_lengths in product((2, 3), repeat=4):
                inp = _build_input(gap, center, arm_lengths)
                if inp is None:
                    continue
                try:
                    out, cells, reasons, visited = _solve_with_order(inp)
                except AssertionError:
                    continue

                if np.array_equal(inp, out):
                    continue
                if len(cells) < 3 or len(visited) < 8:
                    continue

                stable = True
                for arm_order in permutations(range(4)):
                    alt_out, _alt_cells, _alt_reasons, _alt_visited = _solve_with_order(inp, arm_order)
                    if not np.array_equal(alt_out, out):
                        stable = False
                        break
                if not stable:
                    continue

                candidates.append(
                    {
                        "gap": gap,
                        "center": tuple(map(int, center)),
                        "arm_lengths": tuple(map(int, arm_lengths)),
                        "cells": cells,
                        "reasons": reasons,
                        "visited_len": len(visited),
                    }
                )

    cellset_count = len({cand["cells"] for cand in candidates})
    assert cellset_count >= 6, "need varied activated-cell sets"

    pools = {
        "gap0_four": [cand for cand in candidates if cand["gap"] == 0 and len(cand["cells"]) == 4],
        "gap0_three": [cand for cand in candidates if cand["gap"] == 0 and len(cand["cells"]) == 3],
        "gap1_mixed": [cand for cand in candidates if cand["gap"] == 1 and len(set(cand["reasons"])) > 1],
        "gap1_pure": [cand for cand in candidates if cand["gap"] == 1 and len(set(cand["reasons"])) == 1],
    }
    assert all(pools[name] for name in pools), "expected all generator branches to exist"
    return pools


def _candidate_pools() -> dict[str, list[dict]]:
    global _CANDIDATE_POOLS
    if _CANDIDATE_POOLS is None:
        _CANDIDATE_POOLS = _collect_candidates()
    return _CANDIDATE_POOLS


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    pools = _candidate_pools()
    branch_cycle = ("gap0_four", "gap1_mixed", "gap0_three", "gap1_pure")
    branch_name = branch_cycle[_GEN_COUNTER % len(branch_cycle)]
    _GEN_COUNTER += 1

    candidate = random.choice(pools[branch_name])
    grid = _build_input(candidate["gap"], candidate["center"], candidate["arm_lengths"])
    assert grid is not None

    out, cells, reasons, _visited = _solve_with_order(grid)
    assert len(cells) >= 3
    if branch_name == "gap1_mixed":
        assert len(set(reasons)) > 1
    if branch_name == "gap1_pure":
        assert len(set(reasons)) == 1
    assert not np.array_equal(grid, out)
    assert np.array_equal(main(grid), out)
    return grid

from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PALETTE = tuple(range(1, 10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _sign_delta(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert abs(dr) == 1 and abs(dc) == 1, "path steps must be diagonal"
    return dr, dc


def _same_color_component(
    g: np.ndarray, start: tuple[int, int], color: int
) -> set[tuple[int, int]]:
    h, w = g.shape
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen:
                if int(g[rr, cc]) == int(color):
                    seen.add((rr, cc))
                    q.append((rr, cc))
    return seen


def _find_arrow(
    g: np.ndarray,
) -> tuple[int, set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    h, w = g.shape
    candidates = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            pts = []
            for r in (r0, r0 + 1):
                for c in (c0, c0 + 1):
                    if int(g[r, c]) != BG:
                        pts.append((r, c))
            if len(pts) != 3:
                continue
            colors = {int(g[r, c]) for r, c in pts}
            if len(colors) != 1:
                continue
            color = colors.pop()
            comp = _same_color_component(g, pts[0], color)
            if comp != set(pts):
                continue

            corners = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
            missing = list(corners - set(pts))
            if len(missing) != 1:
                continue
            mr, mc = missing[0]
            dr = 1 if mr == r0 else -1
            dc = 1 if mc == c0 else -1
            elbow = (mr + dr, mc + dc)
            if elbow not in set(pts):
                continue
            candidates.append((color, set(pts), elbow, (dr, dc)))

    assert len(candidates) == 1, "expected one 2x2 missing-corner L arrow"
    return candidates[0]


def _ordered_path(
    g: np.ndarray, arrow_cells: set[tuple[int, int]], elbow: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    path_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != BG)
        if (int(r), int(c)) not in arrow_cells
    }
    first = (int(elbow[0] + direction[0]), int(elbow[1] + direction[1]))
    assert first in path_cells, "path must start diagonally away from the arrow elbow"

    seq = [first]
    visited = {first}
    prev = elbow
    cur = first
    while True:
        nxts = []
        for dr, dc in DIAGONALS:
            nb = (cur[0] + dr, cur[1] + dc)
            if nb == prev or nb in visited:
                continue
            if nb in path_cells:
                nxts.append(nb)
        if not nxts:
            break
        assert len(nxts) == 1, "polyline path must be unbranched"
        prev, cur = cur, nxts[0]
        visited.add(cur)
        seq.append(cur)

    assert visited == path_cells, "all non-arrow pixels must belong to the traced path"
    er, ec = seq[-1]
    h, w = g.shape
    assert er in (0, h - 1) or ec in (0, w - 1), "path must terminate at a grid edge"
    return seq


def _wall_for_turn(
    turn: tuple[int, int], incoming: tuple[int, int], outgoing: tuple[int, int]
) -> tuple[tuple[int, int], str] | None:
    if incoming == outgoing:
        return None
    row_flip = outgoing[0] == -incoming[0] and outgoing[1] == incoming[1]
    col_flip = outgoing[0] == incoming[0] and outgoing[1] == -incoming[1]
    assert row_flip or col_flip, "turns must flip exactly one axis"
    if row_flip:
        return (turn[0] + incoming[0], turn[1]), "row"
    return (turn[0], turn[1] + incoming[1]), "col"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the invisible wall pixels that caused a colored diagonal ray to bounce.

    Input structure:
    - Black background with one monochrome 2x2 missing-corner L-shaped arrow.
    - A single unbranched diagonal polyline starts from the arrow elbow in the direction away from the missing corner.
    - The polyline is divided into single-color diagonal segments; at every 90 degree turn the segment color changes.
    - The far end of the polyline reaches the grid edge, while the wall pixels that caused the turns are absent.

    Rule:
    - Trace the path from the arrow in its missing-corner direction.
    - For each turn, compare the incoming and outgoing diagonal directions to determine which axis was flipped.
    - Paint the hidden wall in the positive neighbor cell on the incoming side of that flipped axis.
    - The wall color is inherited from the segment after the turn; the original arrow and path are preserved.

    Key insight:
    - The visible path is the afterimage of a bouncing ray, so each colored turn inversely identifies both a wall position and its inherited color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    _arrow_color, arrow_cells, elbow, arrow_dir = _find_arrow(g)
    path = _ordered_path(g, arrow_cells, elbow, arrow_dir)
    out = g.copy()

    prev = elbow
    for idx, cur in enumerate(path[:-1]):
        nxt = path[idx + 1]
        incoming = _sign_delta(prev, cur)
        outgoing = _sign_delta(cur, nxt)
        wall_info = _wall_for_turn(cur, incoming, outgoing)
        if wall_info is not None:
            wall, _axis = wall_info
            wr, wc = wall
            assert _in_bounds(g.shape, wr, wc), "inferred wall must be in bounds"
            assert int(g[wr, wc]) == BG, "inferred wall cell must be hidden in the input"
            assert int(g[cur]) == int(g[nxt]), "turn cell must use the post-bounce color"
            assert int(g[prev]) != int(g[nxt]), "a turn must switch to a new visible color"
            out[wr, wc] = int(g[nxt])
        prev = cur

    return out


def _arrow_cells(
    elbow: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _edge_distance(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    row_steps = h - 1 - r if dr == 1 else r
    col_steps = w - 1 - c if dc == 1 else c
    return int(min(row_steps, col_steps))


def _diagonal_neighbor_count(
    cells: set[tuple[int, int]], point: tuple[int, int], allowed: tuple[int, int]
) -> int:
    count = 0
    for dr, dc in DIAGONALS:
        nb = (point[0] + dr, point[1] + dc)
        if nb in cells and nb != allowed:
            count += 1
    return count


def _can_add_step(
    shape: tuple[int, int],
    cell: tuple[int, int],
    previous: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    if not _in_bounds(shape, *cell):
        return False
    if cell in path_set or cell in blocked:
        return False
    return _diagonal_neighbor_count(path_set, cell, previous) == 0


def _try_build_path(
    shape: tuple[int, int],
    elbow: tuple[int, int],
    direction: tuple[int, int],
    flips: list[str],
    arrow: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], str]], list[int]] | None:
    min_seg = 3
    min_final = 3
    path: list[tuple[int, int]] = []
    turn_walls: list[tuple[tuple[int, int], str]] = []
    turn_indices: list[int] = []
    path_set: set[tuple[int, int]] = set()
    wall_set: set[tuple[int, int]] = set()

    def add_segment_cells(
        current: tuple[int, int], vec: tuple[int, int], length: int
    ) -> list[tuple[int, int]] | None:
        segment = []
        prev = current
        for step in range(1, length + 1):
            cell = (current[0] + step * vec[0], current[1] + step * vec[1])
            blocked = arrow | wall_set
            if not _can_add_step(shape, cell, prev, path_set, blocked):
                return None
            segment.append(cell)
            prev = cell
        return segment

    def commit_segment(segment: list[tuple[int, int]]) -> None:
        for cell in segment:
            path.append(cell)
            path_set.add(cell)

    def undo_segment(segment: list[tuple[int, int]]) -> None:
        for _cell in segment:
            old = path.pop()
            path_set.remove(old)

    def rec(idx: int, current: tuple[int, int], vec: tuple[int, int]) -> bool:
        if idx == len(flips):
            length = _edge_distance(shape, current, vec)
            if length < min_final:
                return False
            segment = add_segment_cells(current, vec, length)
            if segment is None:
                return False
            commit_segment(segment)
            return True

        max_steps = _edge_distance(shape, current, vec)
        lengths = list(range(min_seg, max_steps))
        random.shuffle(lengths)
        for length in lengths:
            segment = add_segment_cells(current, vec, length)
            if segment is None:
                continue
            commit_segment(segment)

            turn = path[-1]
            flip = flips[idx]
            if flip == "row":
                wall = (turn[0] + vec[0], turn[1])
                new_vec = (-vec[0], vec[1])
            else:
                wall = (turn[0], turn[1] + vec[1])
                new_vec = (vec[0], -vec[1])

            wall_ok = (
                _in_bounds(shape, *wall)
                and wall not in wall_set
                and wall not in path_set
                and wall not in arrow
            )
            immediate = (turn[0] + new_vec[0], turn[1] + new_vec[1])
            wall_ok = wall_ok and _can_add_step(shape, immediate, turn, path_set, arrow | wall_set | {wall})
            if wall_ok:
                turn_indices.append(len(path) - 1)
                turn_walls.append((wall, flip))
                wall_set.add(wall)
                if rec(idx + 1, turn, new_vec):
                    return True
                wall_set.remove(wall)
                turn_walls.pop()
                turn_indices.pop()

            undo_segment(segment)
        return False

    if rec(0, elbow, direction):
        return path, turn_walls, turn_indices
    return None


def _choose_flips(n_bounces: int) -> list[str]:
    flips = ["row", "col"]
    while len(flips) < n_bounces:
        flips.append(random.choice(["row", "col"]))
    random.shuffle(flips)
    return flips


def _render_from_plan(
    shape: tuple[int, int],
    arrow: set[tuple[int, int]],
    path: list[tuple[int, int]],
    turn_walls: list[tuple[tuple[int, int], str]],
    turn_indices: list[int],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)

    for r, c in arrow:
        inp[r, c] = colors[0]
        out[r, c] = colors[0]

    turn_at = {idx: i for i, idx in enumerate(turn_indices)}
    color_idx = 0
    for idx, (r, c) in enumerate(path):
        if idx in turn_at:
            color_idx = turn_at[idx] + 1
        inp[r, c] = colors[color_idx]
        out[r, c] = colors[color_idx]

    for i, (wall, _flip) in enumerate(turn_walls):
        wr, wc = wall
        out[wr, wc] = colors[i + 1]
    return inp, out


def _valid_generation(inp: np.ndarray, expected: np.ndarray, flips: list[str]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(flips) < 2 or set(flips) != {"row", "col"}:
        return False

    diff = expected != inp
    if int(diff.sum()) != len(flips):
        return False
    if np.any(inp[diff] != BG):
        return False
    if np.any(expected[diff] == BG):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    direction = random.choice(DIAGONALS)
    n_bounces = random.choice([2, 2, 3, 3, 4])
    flips = _choose_flips(n_bounces)

    dr, dc = direction
    elbows = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _in_bounds((h, w), r + dr, c + dc)
        and _in_bounds((h, w), r - dr, c)
        and _in_bounds((h, w), r, c - dc)
    ]
    random.shuffle(elbows)

    for elbow in elbows[:80]:
        arrow = _arrow_cells(elbow, direction)
        plan = _try_build_path((h, w), elbow, direction, flips, arrow)
        if plan is None:
            continue
        path, turn_walls, turn_indices = plan
        colors = random.sample(PALETTE, n_bounces + 1)
        inp, expected = _render_from_plan((h, w), arrow, path, turn_walls, turn_indices, colors)
        if _valid_generation(inp, expected, flips):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-wall reconstruction input")

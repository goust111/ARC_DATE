from __future__ import annotations

from collections import Counter
from typing import NamedTuple
import random

import numpy as np


BLACK = 0
RED = 2
ARC_COLORS = list(range(10))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Hit(NamedTuple):
    cell: tuple[int, int]
    mate: tuple[int, int]
    incoming: tuple[int, int]
    outgoing: tuple[int, int]
    color: int


class Trace(NamedTuple):
    out: np.ndarray
    hits: list[Hit]
    path: list[tuple[int, int]]
    source: tuple[int, int]
    bg: int


class PlanStep(NamedTuple):
    cell: tuple[int, int]
    mate: tuple[int, int]
    incoming: tuple[int, int]
    outgoing: tuple[int, int]


class Plan(NamedTuple):
    steps: list[PlanStep]
    path: set[tuple[int, int]]
    occupied: set[tuple[int, int]]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _is_interior(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = map(int, shape)
    return 0 < r < h - 1 and 0 < c < w - 1


def _add(cell: tuple[int, int], direction: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * k, cell[1] + direction[1] * k


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _source_direction(source: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = source
    h, w = map(int, shape)
    edges = [r == 0, r == h - 1, c == 0, c == w - 1]
    assert sum(bool(x) for x in edges) == 1, "source must be on exactly one boundary edge"
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    return 0, -1


def _parse_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids are at most 30 by 30"

    bg = _get_bg(g)
    assert bg != RED, "background must differ from the red source/path color"

    red = np.argwhere(g == RED)
    assert red.shape[0] == 1, "input must contain exactly one red boundary source"
    source = tuple(map(int, red[0]))
    direction = _source_direction(source, g.shape)

    colors = sorted(set(map(int, np.unique(g))) - {int(bg), RED})
    assert colors, "expected at least one domino deflector"

    cell_to_mate: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    dominoes: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each deflector color must occupy exactly two cells"
        a, b = coords
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "deflector must be an orthogonal domino"
        assert _is_interior(g.shape, a) and _is_interior(g.shape, b), "deflectors must lie inside the field"
        cell_to_mate[a] = (b, color)
        cell_to_mate[b] = (a, color)
        dominoes.append((color, a, b))

    for i, (_color_a, a0, a1) in enumerate(dominoes):
        for _color_b, b0, b1 in dominoes[i + 1 :]:
            for p in (a0, a1):
                for q in (b0, b1):
                    assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1, "deflectors must not touch"

    return g, bg, source, direction, cell_to_mate


def _trace(grid: np.ndarray) -> Trace:
    g, bg, source, direction, cell_to_mate = _parse_grid(grid)
    out = g.copy()
    hits: list[Hit] = []
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    r, c = _add(source, direction)
    dr, dc = direction
    step_limit = int(g.shape[0] * g.shape[1] * 8)

    for _ in range(step_limit):
        if not _in_bounds(g.shape, r, c):
            return Trace(out, hits, path, source, int(bg))

        state = (r, c, dr, dc)
        assert state not in seen, "beam path must not loop"
        seen.add(state)

        cell = (r, c)
        if cell in cell_to_mate:
            mate, color = cell_to_mate[cell]
            ndr, ndc = mate[0] - r, mate[1] - c
            assert abs(ndr) + abs(ndc) == 1
            hits.append(Hit(cell, mate, (dr, dc), (ndr, ndc), int(color)))
            r, c = mate[0] + ndr, mate[1] + ndc
            dr, dc = ndr, ndc
            continue

        assert int(g[r, c]) == int(bg), "beam may pass only through background before leaving"
        out[r, c] = RED
        path.append((r, c))
        r += dr
        c += dc

    raise AssertionError("beam simulation exceeded the finite path limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a red ray through domino deflectors whose hit order determines each exit direction.

    Input structure:
    - The grid has a dominant background and one red source pixel on a non-corner boundary cell.
    - Every other foreground color forms one isolated two-cell orthogonal domino inside the field.
    - Domino colors are unique labels only; the other cell of the hit domino gives the next direction.

    Rule:
    - Launch the ray from the source perpendicular to its boundary edge.
    - Paint every background cell the ray traverses red.
    - When the ray reaches one cell X of a domino whose other cell is Y, leave the domino unchanged and continue from beyond Y in direction X-to-Y.
    - Repeat through later deflectors until the ray exits the grid.

    Key insight:
    - The outgoing direction is encoded by which cell of a two-cell deflector is hit first, not by the deflector color.
    """
    return _trace(grid).out


# ---------------- generator ----------------

_GEN_BG = BLACK
_GEN_PALETTE = [c for c in ARC_COLORS if c not in (_GEN_BG, RED)]
_GEN_MAX_ATTEMPTS = 800
_GEN_MIN_RUN_BEFORE_HIT = 2
_GEN_MAX_RUN_BEFORE_HIT = 7
_GEN_MIN_EXIT_RUN = 2
_GEN_CALL_COUNT = 0


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _edge_source(shape: tuple[int, int], edge: str) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = map(int, shape)
    if edge == "top":
        return (0, random.randint(2, w - 3)), (1, 0)
    if edge == "bottom":
        return (h - 1, random.randint(2, w - 3)), (-1, 0)
    if edge == "left":
        return (random.randint(2, h - 3), 0), (0, 1)
    if edge == "right":
        return (random.randint(2, h - 3), w - 1), (0, -1)
    raise ValueError(edge)


def _clear_new_domino(
    cells: tuple[tuple[int, int], tuple[int, int]],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    a, b = cells
    if a == b:
        return False
    if not (_is_interior(shape, a) and _is_interior(shape, b)):
        return False
    if a in forbidden or b in forbidden:
        return False
    if a in occupied or b in occupied:
        return False
    for p in cells:
        for q in occupied:
            if max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1:
                return False
    return True


def _plan_semantics_ok(steps: list[PlanStep], path: set[tuple[int, int]]) -> bool:
    if len(steps) < 3:
        return False
    straight = sum(step.incoming == step.outgoing for step in steps)
    turns = len(steps) - straight
    if straight < 1 or turns < 1:
        return False
    if len({step.outgoing for step in steps}) < 3:
        return False
    rows = {r for r, _c in path}
    cols = {c for _r, c in path}
    return len(rows) >= 4 and len(cols) >= 4 and len(path) >= 10


def _plan_path(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: tuple[int, int],
    hit_count: int,
) -> Plan | None:
    start = _add(source, direction)
    if not _in_bounds(shape, *start):
        return None

    def rec(
        current: tuple[int, int],
        incoming: tuple[int, int],
        index: int,
        occupied: set[tuple[int, int]],
        path: set[tuple[int, int]],
        states: set[tuple[int, int, int, int]],
        steps: list[PlanStep],
    ) -> Plan | None:
        outgoing_options = [incoming] + _perpendiculars(incoming)
        random.shuffle(outgoing_options)
        ray = _ray_cells(current, incoming, shape)
        distances = list(range(_GEN_MIN_RUN_BEFORE_HIT, min(len(ray), _GEN_MAX_RUN_BEFORE_HIT + 1)))
        random.shuffle(distances)

        for outgoing in outgoing_options:
            for dist in distances:
                cell = ray[dist]
                mate = _add(cell, outgoing)
                forbidden = set(path) | {source}
                if not _clear_new_domino((cell, mate), occupied, forbidden, shape):
                    continue

                before = ray[:dist]
                if any(p in occupied or p == source for p in before):
                    continue
                segment_states = {(r, c, incoming[0], incoming[1]) for r, c in before + [cell]}
                if states & segment_states:
                    continue

                occupied2 = set(occupied)
                occupied2.update((cell, mate))
                path2 = set(path)
                path2.update(before)
                states2 = set(states)
                states2.update(segment_states)
                steps2 = steps + [PlanStep(cell, mate, incoming, outgoing)]
                next_start = _add(mate, outgoing)

                if index == hit_count - 1:
                    exit_cells = _ray_cells(next_start, outgoing, shape)
                    if len(exit_cells) < _GEN_MIN_EXIT_RUN:
                        continue
                    if any(p in occupied2 or p == source for p in exit_cells):
                        continue
                    exit_states = {(r, c, outgoing[0], outgoing[1]) for r, c in exit_cells}
                    if states2 & exit_states:
                        continue
                    path_final = set(path2)
                    path_final.update(exit_cells)
                    if _plan_semantics_ok(steps2, path_final):
                        return Plan(steps2, path_final, occupied2)
                    continue

                if not _in_bounds(shape, *next_start):
                    continue
                if next_start in occupied2 or next_start == source:
                    continue
                next_state = (next_start[0], next_start[1], outgoing[0], outgoing[1])
                if next_state in states2:
                    continue
                found = rec(next_start, outgoing, index + 1, occupied2, path2, states2, steps2)
                if found is not None:
                    return found
        return None

    return rec(start, direction, 0, set(), set(), set(), [])


def _place_decoys(
    grid: np.ndarray,
    count: int,
    colors: list[int],
    occupied: set[tuple[int, int]],
    path: set[tuple[int, int]],
    source: tuple[int, int],
) -> bool:
    shape = grid.shape
    forbidden = set(path) | {source}
    orientations = [(1, 0), (0, 1)]
    for color in colors[:count]:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r in range(1, shape[0] - 1):
            for c in range(1, shape[1] - 1):
                for direction in orientations:
                    a = (r, c)
                    b = _add(a, direction)
                    if _clear_new_domino((a, b), occupied, forbidden, shape):
                        candidates.append((a, b))
        if not candidates:
            return False
        a, b = random.choice(candidates)
        grid[a] = color
        grid[b] = color
        occupied.update((a, b))
    return True


def _render_candidate(
    shape: tuple[int, int],
    source: tuple[int, int],
    plan: Plan,
    decoy_count: int,
) -> np.ndarray | None:
    needed = len(plan.steps) + decoy_count
    if needed > len(_GEN_PALETTE):
        return None
    colors = random.sample(_GEN_PALETTE, needed)

    grid = np.full(shape, _GEN_BG, dtype=int)
    grid[source] = RED
    occupied = set(plan.occupied)
    for step, color in zip(plan.steps, colors):
        grid[step.cell] = color
        grid[step.mate] = color

    if not _place_decoys(grid, decoy_count, colors[len(plan.steps) :], occupied, plan.path, source):
        return None
    return grid


def _semantic_accepts(grid: np.ndarray, plan: Plan, decoy_count: int) -> bool:
    try:
        trace = _trace(grid)
    except AssertionError:
        return False

    hit_pairs = [(hit.cell, hit.mate, hit.incoming, hit.outgoing) for hit in trace.hits]
    planned_pairs = [(step.cell, step.mate, step.incoming, step.outgoing) for step in plan.steps]
    if hit_pairs != planned_pairs:
        return False
    if not set(trace.path) == plan.path:
        return False
    if decoy_count < 1:
        return False

    changed = int(np.sum((trace.out != grid) & (grid == trace.bg)))
    if changed != len(plan.path):
        return False

    source_dir = _source_direction(trace.source, grid.shape)
    straight = grid.copy()
    for r, c in _ray_cells(_add(trace.source, source_dir), source_dir, grid.shape):
        if int(straight[r, c]) == trace.bg:
            straight[r, c] = RED
        else:
            break
    if np.array_equal(straight, trace.out):
        return False

    all_colors = set(map(int, np.unique(grid))) - {trace.bg, RED}
    hit_colors = {hit.color for hit in trace.hits}
    return len(all_colors - hit_colors) >= decoy_count and np.array_equal(main(grid), trace.out)


def _gen_attempt(edge: str) -> np.ndarray | None:
    h = random.randint(13, 24)
    w = random.randint(13, 24)
    source, direction = _edge_source((h, w), edge)
    hit_count = random.randint(3, 5)
    decoy_count = random.randint(1, min(2, len(_GEN_PALETTE) - hit_count))

    plan = _plan_path((h, w), source, direction, hit_count)
    if plan is None:
        return None
    grid = _render_candidate((h, w), source, plan, decoy_count)
    if grid is None:
        return None
    if not _semantic_accepts(grid, plan, decoy_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    edges = ["top", "right", "bottom", "left"]
    preferred = edges[_GEN_CALL_COUNT % len(edges)]
    _GEN_CALL_COUNT += 1
    edge_order = [preferred] + [edge for edge in edges if edge != preferred]

    for edge in edge_order:
        for _ in range(_GEN_MAX_ATTEMPTS):
            grid = _gen_attempt(edge)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid relay-deflector ray input")

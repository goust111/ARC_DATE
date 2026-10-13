from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components_of_nonblack(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cell = q.popleft()
                comp.append(cell)
                for nr, nc in _neighbors4(cell):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_dominoes(grid: np.ndarray) -> tuple[list[dict], int]:
    comps = _components_of_nonblack(grid)
    red_count = int(np.count_nonzero(grid == RED))
    assert red_count == 1, "expected exactly one red marker pixel"

    dominoes: list[dict] = []
    start_idx = -1
    for comp in comps:
        assert len(comp) == 2, "every foreground object must be a domino"
        a, b = comp
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "domino cells must be edge-adjacent"
        colors = (int(grid[a]), int(grid[b]))
        dominoes.append({"cells": (a, b), "colors": colors})
        if RED in colors:
            assert start_idx == -1, "red must belong to a unique start domino"
            start_idx = len(dominoes) - 1

    assert start_idx != -1, "missing red start domino"
    start = dominoes[start_idx]
    colors = start["colors"]
    assert colors.count(RED) == 1, "start domino must contain exactly one red cell"
    return dominoes, start_idx


def _trace_pipe(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    dominoes, start_idx = _parse_dominoes(g)
    cell_to_domino: dict[tuple[int, int], int] = {}
    for idx, domino in enumerate(dominoes):
        for cell in domino["cells"]:
            cell_to_domino[cell] = idx

    start_cells = dominoes[start_idx]["cells"]
    tail = start_cells[0] if int(g[start_cells[0]]) == RED else start_cells[1]
    head = start_cells[1] if tail == start_cells[0] else start_cells[0]
    direction = _sub(head, tail)

    visited = {start_idx}
    out = g.copy()
    while True:
        pos = _add(head, direction)
        while _in_bounds(g.shape, pos):
            hit = cell_to_domino.get(pos)
            if hit is not None:
                assert hit not in visited, "a relay domino may not be hit twice"
                entry = pos
                c0, c1 = dominoes[hit]["cells"]
                other = c1 if entry == c0 else c0
                new_direction = _sub(other, entry)
                assert abs(new_direction[0]) + abs(new_direction[1]) == 1, "invalid relay domino"
                assert direction[0] * new_direction[0] + direction[1] * new_direction[1] == 0, (
                    "hit domino must be perpendicular to the incoming ray"
                )
                visited.add(hit)
                head = other
                direction = new_direction
                break
            if out[pos] == BLACK:
                out[pos] = GREY
            pos = _add(pos, direction)
        else:
            return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Start from the red-marked domino and draw the full relay pipe that turns through hit dominoes.

    Input structure:
    - A black background contains several isolated 2-cell dominoes.
    - Exactly one domino is mixed: one cell is red and the other is that domino's body color.
    - Every other domino is monochrome, and the layout guarantees each relay hit lands on a perpendicular domino.

    Rule:
    - Treat the red cell of the mixed domino as the tail and its partner cell as the head; launch a ray from the head in the tail-to-head direction.
    - While the ray travels, recolor only black cells to grey until the ray enters a domino cell.
    - The entered cell becomes the new tail, the other cell of that domino becomes the new head, and the ray continues in that domino's axis direction; repeat until the ray leaves the grid.

    Key insight:
    - Each domino acts like a 2-cell elbow pipe: the hit endpoint is the inlet and the other endpoint determines the next travel direction, so the whole path is a deterministic relay chain.
    """
    return _trace_pipe(grid)


def _steps_to_edge(shape: tuple[int, int], head: tuple[int, int], direction: tuple[int, int]) -> int:
    steps = 0
    pos = _add(head, direction)
    while _in_bounds(shape, pos):
        steps += 1
        pos = _add(pos, direction)
    return steps


def _ray_cells(shape: tuple[int, int], head: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    pos = _add(head, direction)
    while _in_bounds(shape, pos):
        cells.append(pos)
        pos = _add(pos, direction)
    return cells


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _domino_fits(
    shape: tuple[int, int],
    cells: tuple[tuple[int, int], tuple[int, int]],
    occupied_domino: set[tuple[int, int]],
) -> bool:
    cand = set(cells)
    for cell in cells:
        if not _in_bounds(shape, cell) or cell in occupied_domino:
            return False
    for cell in cells:
        for nb in _neighbors8(cell):
            if nb in occupied_domino and nb not in cand:
                return False
    return True


def _exit_is_clear(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    occupied_domino: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = _ray_cells(shape, head, direction)
    if not cells:
        return None
    if any(cell in occupied_domino or cell in occupied_path for cell in cells):
        return None
    return cells


def _build_chain(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    relays_left: int,
    occupied_domino: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> tuple[list[dict], list[list[tuple[int, int]]]] | None:
    if relays_left == 0:
        exit_cells = _exit_is_clear(shape, head, direction, occupied_domino, occupied_path)
        if exit_cells is None:
            return None
        return [], [exit_cells]

    forward_steps = _steps_to_edge(shape, head, direction)
    candidates: list[tuple[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]], tuple[int, int]]] = []
    for dist in range(2, forward_steps + 1):
        entry = (head[0] + dist * direction[0], head[1] + dist * direction[1])
        between = [
            (head[0] + k * direction[0], head[1] + k * direction[1])
            for k in range(1, dist)
        ]
        if any(cell in occupied_domino or cell in occupied_path for cell in between):
            break
        if entry in occupied_domino or entry in occupied_path:
            break
        turns = list(_perpendiculars(direction))
        random.shuffle(turns)
        for new_direction in turns:
            other = _add(entry, new_direction)
            cells = (entry, other)
            if not _domino_fits(shape, cells, occupied_domino):
                continue
            if relays_left > 1 and _steps_to_edge(shape, other, new_direction) < 2:
                continue
            candidates.append((cells, between, new_direction))

    random.shuffle(candidates)
    for cells, between, new_direction in candidates:
        new_occ_domino = occupied_domino | set(cells)
        new_occ_path = occupied_path | set(between)
        tail = cells[0]
        head2 = cells[1]
        built = _build_chain(shape, head2, new_direction, relays_left - 1, new_occ_domino, new_occ_path)
        if built is None:
            continue
        rest_dominoes, segments = built
        relay = {"tail": tail, "head": head2, "cells": cells}
        return [relay] + rest_dominoes, [between] + segments
    return None


def _place_distractors(
    shape: tuple[int, int],
    occupied_domino: set[tuple[int, int]],
    occupied_path: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    targets = [0, 1, 2, 3]
    count = random.choice(targets)
    distractors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _ in range(count):
        choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                a = (r, c)
                for d in ((0, 1), (1, 0)):
                    b = _add(a, d)
                    cells = (a, b)
                    if not _in_bounds(shape, b):
                        continue
                    if any(cell in occupied_path for cell in cells):
                        continue
                    if _domino_fits(shape, cells, occupied_domino):
                        choices.append(cells)
        if not choices:
            break
        cells = random.choice(choices)
        distractors.append(cells)
        occupied_domino |= set(cells)
    return distractors


def _render_input(
    shape: tuple[int, int],
    start_tail: tuple[int, int],
    start_head: tuple[int, int],
    relay_dominoes: list[dict],
    distractors: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    start_color = random.choice(PALETTE)
    grid[start_tail] = RED
    grid[start_head] = start_color

    for relay in relay_dominoes:
        color = random.choice(PALETTE)
        a, b = relay["cells"]
        grid[a] = color
        grid[b] = color

    for a, b in distractors:
        color = random.choice(PALETTE)
        grid[a] = color
        grid[b] = color
    return grid


def _all_dominoes_are_isolated(grid: np.ndarray) -> bool:
    comps = _components_of_nonblack(grid)
    if not comps:
        return False
    seen: set[tuple[int, int]] = set()
    for comp in comps:
        if len(comp) != 2:
            return False
        for cell in comp:
            for nb in _neighbors8(cell):
                if nb in seen and nb not in comp:
                    return False
        seen.update(comp)
    return True


def _essential_relays(inp: np.ndarray, relay_dominoes: list[dict], expected: np.ndarray) -> bool:
    for relay in relay_dominoes:
        trial = inp.copy()
        for cell in relay["cells"]:
            trial[cell] = BLACK
        try:
            out = main(trial)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return False
    return True


def _validate_layout(
    inp: np.ndarray,
    expected: np.ndarray,
    start_tail: tuple[int, int],
    start_head: tuple[int, int],
    start_direction: tuple[int, int],
    relay_dominoes: list[dict],
    segments: list[list[tuple[int, int]]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.count_nonzero(inp == RED) != 1 or np.any(inp == GREY):
        return False
    if not _all_dominoes_are_isolated(inp):
        return False
    if len(relay_dominoes) < 2:
        return False
    if not all(len(seg) >= 1 for seg in segments):
        return False

    direction = start_direction
    head = start_head
    for idx, relay in enumerate(relay_dominoes):
        seg = segments[idx]
        expected_seg = [
            (head[0] + k * direction[0], head[1] + k * direction[1])
            for k in range(1, len(seg) + 1)
        ]
        if seg != expected_seg:
            return False
        entry = relay["tail"]
        if _add(seg[-1], direction) != entry:
            return False
        new_direction = _sub(relay["head"], relay["tail"])
        if direction[0] * new_direction[0] + direction[1] * new_direction[1] != 0:
            return False
        head = relay["head"]
        direction = new_direction

    if len({(abs(d[0]), abs(d[1])) for d in [start_direction] + [_sub(r["head"], r["tail"]) for r in relay_dominoes]}) < 2:
        return False

    final_seg = segments[-1]
    if final_seg[0] != _add(head, direction):
        return False
    if _in_bounds(inp.shape, _add(final_seg[-1], direction)):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if not _essential_relays(inp, relay_dominoes, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    shape = (h, w)
    relay_count = random.randint(2, 5)

    direction = random.choice(DIRS)
    start_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            head = (r, c)
            tail = (r - direction[0], c - direction[1])
            if not _in_bounds(shape, tail):
                continue
            if _steps_to_edge(shape, head, direction) < 2:
                continue
            start_candidates.append((tail, head))
    if not start_candidates:
        return None
    start_tail, start_head = random.choice(start_candidates)
    occupied_domino = {start_tail, start_head}

    built = _build_chain(shape, start_head, direction, relay_count, occupied_domino, set())
    if built is None:
        return None
    relay_dominoes, segments = built
    path_cells = {cell for seg in segments for cell in seg}
    occupied_domino |= {cell for relay in relay_dominoes for cell in relay["cells"]}

    distractors = _place_distractors(shape, set(occupied_domino), set(path_cells))
    inp = _render_input(shape, start_tail, start_head, relay_dominoes, distractors)
    expected = inp.copy()
    for seg in segments:
        for cell in seg:
            expected[cell] = GREY

    if not _validate_layout(inp, expected, start_tail, start_head, direction, relay_dominoes, segments):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-pipe ARC input")

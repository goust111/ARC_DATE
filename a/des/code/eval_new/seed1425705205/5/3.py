from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_CHOICES = [BLACK, ORANGE, AZURE]
RUN_LEN_RANGE = (3, 8)
STUB_LEN_RANGE = (1, 3)
GRID_H_RANGE = (15, 26)
GRID_W_RANGE = (15, 26)
MAX_RUNS = 4
MAX_STRAIGHT_NODES = 3
MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _line_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    *,
    include_start: bool,
) -> list[tuple[int, int]]:
    dr, dc = direction
    first = 0 if include_start else 1
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(first, steps + 1)]


def _can_add_cells(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    allowed_contacts: dict[tuple[int, int], set[tuple[int, int]]],
) -> bool:
    if not cells:
        return False
    new_set = set(cells)
    if len(new_set) != len(cells):
        return False
    for cell in cells:
        if not _in_bounds(shape, cell) or cell in occupied:
            return False
    for idx, cell in enumerate(cells):
        allowed = set(allowed_contacts.get(cell, set()))
        if idx > 0:
            allowed.add(cells[idx - 1])
        if idx + 1 < len(cells):
            allowed.add(cells[idx + 1])
        for nb in _neighbors4(cell):
            if nb in occupied or nb in new_set:
                if nb not in allowed:
                    return False
    return True


def _source_options(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    r, c = start
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if direction == (0, 1):
        candidates = [((r - 1, c - 2), (r, c - 1)), ((r, c - 2), (r, c - 1))]
    elif direction == (0, -1):
        candidates = [((r - 1, c + 1), (r, c + 1)), ((r, c + 1), (r, c + 1))]
    elif direction == (1, 0):
        candidates = [((r - 2, c - 1), (r - 1, c)), ((r - 2, c), (r - 1, c))]
    else:
        candidates = [((r + 1, c - 1), (r + 1, c)), ((r + 1, c), (r + 1, c))]

    out: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    for (r0, c0), touch in candidates:
        cells = [(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)]
        if all(_in_bounds(shape, cell) for cell in cells):
            out.append((cells, touch))
    return out


def _find_source(grid: np.ndarray) -> set[tuple[int, int]]:
    maroon_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)}
    assert len(maroon_cells) == 4, "expected a unique 2x2 maroon source"
    rows = [r for r, _ in maroon_cells]
    cols = [c for _, c in maroon_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 1 and c1 - c0 == 1, "source must be a 2x2 block"
    expected = {(r, c) for r in (r0, r1) for c in (c0, c1)}
    assert maroon_cells == expected, "source cells must fill the entire 2x2 block"
    return maroon_cells


def _path_neighbors(
    grid: np.ndarray,
    cell: tuple[int, int],
    bg: int,
    source: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for nb in _neighbors4(cell):
        if not _in_bounds(grid.shape, nb) or nb in source:
            continue
        if int(grid[nb]) != bg:
            out.append(nb)
    return out


def _is_node(
    grid: np.ndarray,
    cell: tuple[int, int],
    bg: int,
    source: set[tuple[int, int]],
) -> bool:
    r, c = cell
    horiz = False
    vert = False
    for dr, dc in DIRS4:
        nb = (r + dr, c + dc)
        if not _in_bounds(grid.shape, nb) or nb in source or int(grid[nb]) == bg:
            continue
        if dr == 0:
            horiz = True
        else:
            vert = True
    return horiz and vert


def _trace_signal(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    source = _find_source(g)

    starts: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for sr, sc in sorted(source):
        for dr, dc in DIRS4:
            nb = (sr + dr, sc + dc)
            if not _in_bounds(g.shape, nb) or nb in source:
                continue
            if int(g[nb]) != bg:
                starts.append((nb, (dr, dc)))
    unique_starts = {(cell, direction) for cell, direction in starts}
    assert len(unique_starts) == 1, "source must touch exactly one path cell"
    pos, direction = next(iter(unique_starts))
    current_color = int(g[pos])
    visited: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    max_steps = g.shape[0] * g.shape[1] * 4
    for _ in range(max_steps):
        state = (pos, direction, current_color)
        assert state not in visited, "signal path must terminate without looping"
        visited.add(state)

        cell_color = int(g[pos])
        node = _is_node(g, pos, bg, source)
        if cell_color != current_color:
            assert node, "a color change is only allowed at an intersection node"
            turn_candidates = []
            for nd in (_left(direction), _right(direction)):
                nxt = (pos[0] + nd[0], pos[1] + nd[1])
                if _in_bounds(g.shape, nxt) and nxt not in source and int(g[nxt]) != bg:
                    turn_candidates.append(nd)
            assert len(turn_candidates) == 1, "turning nodes must have a unique perpendicular exit"
            direction = turn_candidates[0]
            current_color = cell_color

        nxt = (pos[0] + direction[0], pos[1] + direction[1])
        if not _in_bounds(g.shape, nxt) or nxt in source or int(g[nxt]) == bg:
            return pos
        pos = nxt

    raise AssertionError("signal trace exceeded step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the signal through the colored path network and mark its terminal endpoint.

    Input structure:
    - bg is the most common color.
    - Several 1-pixel-wide horizontal or vertical colored path segments appear on the grid, and some of them meet at colored intersection nodes.
    - Each node pixel keeps the color of exactly one of the two crossing paths.
    - Exactly one 2x2 maroon block is attached to one path endpoint and acts as the signal source.

    Rule:
    - Start from the unique path cell adjacent to the maroon source and move away from the source along that path.
    - Continue in the current direction until reaching a node; when the node color matches the current path color, go straight through it.
    - When the node color differs from the current path color, turn 90 degrees into the other path and continue with the node's color as the new current path color.
    - Stop when the next cell in the chosen direction is out of bounds or background, then recolor the terminal path pixel to magenta and also recolor its four-neighbor background cells to magenta.

    Key insight:
    - The moving state is not just the current position but also the current path color and travel direction, because node colors act as switches: same-color means straight, different-color means turn.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    end_r, end_c = _trace_signal(g)

    out = g.copy()
    out[end_r, end_c] = MAGENTA
    for nr, nc in _neighbors4((end_r, end_c)):
        if _in_bounds(out.shape, (nr, nc)) and int(out[nr, nc]) == bg:
            out[nr, nc] = MAGENTA
    return out


def _build_route(
    rng: random.Random,
    shape: tuple[int, int],
    total_runs: int,
) -> tuple[list[dict], list[tuple[int, int]], tuple[int, int]] | None:
    directions = DIRS4[:]
    rng.shuffle(directions)

    def search(
        point: tuple[int, int],
        direction: tuple[int, int],
        run_idx: int,
        occupied: set[tuple[int, int]],
        runs: list[dict],
        source_cells: list[tuple[int, int]],
        touch_cell: tuple[int, int],
    ) -> list[dict] | None:
        lengths = list(range(RUN_LEN_RANGE[0], RUN_LEN_RANGE[1] + 1))
        rng.shuffle(lengths)
        for steps in lengths:
            full_cells = _line_cells(point, direction, steps, include_start=True)
            add_cells = full_cells if run_idx == 0 else full_cells[1:]
            allowed = {add_cells[0]: {touch_cell if run_idx == 0 else point}}
            if not _can_add_cells(add_cells, occupied, shape, allowed):
                continue

            endpoint = full_cells[-1]
            new_occupied = occupied | set(add_cells)
            run_record = {
                "start": point,
                "end": endpoint,
                "dir": direction,
                "cells": full_cells,
                "stub": [],
            }

            if run_idx == total_runs - 1:
                return runs + [run_record]

            next_dirs = [_left(direction), _right(direction)]
            rng.shuffle(next_dirs)
            stub_lengths = list(range(STUB_LEN_RANGE[0], STUB_LEN_RANGE[1] + 1))
            rng.shuffle(stub_lengths)

            for next_dir in next_dirs:
                for stub_len in stub_lengths:
                    stub_cells = _line_cells(endpoint, direction, stub_len, include_start=False)
                    stub_allowed = {stub_cells[0]: {endpoint}}
                    if not _can_add_cells(stub_cells, new_occupied, shape, stub_allowed):
                        continue

                    next_first = (endpoint[0] + next_dir[0], endpoint[1] + next_dir[1])
                    next_allowed = {next_first: {endpoint}}
                    if not _can_add_cells([next_first], new_occupied | set(stub_cells), shape, next_allowed):
                        continue

                    extended = dict(run_record)
                    extended["stub"] = stub_cells
                    result = search(
                        endpoint,
                        next_dir,
                        run_idx + 1,
                        new_occupied | set(stub_cells),
                        runs + [extended],
                        source_cells,
                        touch_cell,
                    )
                    if result is not None:
                        return result
        return None

    candidate_starts = [
        (r, c)
        for r in range(1, shape[0] - 1)
        for c in range(1, shape[1] - 1)
    ]
    rng.shuffle(candidate_starts)

    for direction in directions:
        for start in candidate_starts:
            options = _source_options(start, direction, shape)
            rng.shuffle(options)
            for source_cells, touch_cell in options:
                occupied = set(source_cells)
                result = search(start, direction, 0, occupied, [], source_cells, touch_cell)
                if result is not None:
                    return result, source_cells, touch_cell
    return None


def _place_straight_nodes(
    rng: random.Random,
    shape: tuple[int, int],
    runs: list[dict],
    occupied: set[tuple[int, int]],
    colors: list[int],
) -> list[dict] | None:
    slots: list[tuple[int, tuple[int, int]]] = []
    for run_idx, run in enumerate(runs):
        cells = run["cells"]
        for idx in range(2, len(cells) - 2):
            slots.append((run_idx, cells[idx]))
    rng.shuffle(slots)

    decoys: list[dict] = []
    used_slots: set[tuple[int, tuple[int, int]]] = set()
    for color in colors:
        placed = False
        for slot in slots:
            if slot in used_slots:
                continue
            run_idx, node = slot
            direction = runs[run_idx]["dir"]
            perps = [_left(direction), _right(direction)]
            modes = [
                ("two", perps[0], perps[1]),
                ("one", perps[0], None),
                ("one", perps[1], None),
            ]
            rng.shuffle(modes)
            for mode, first_dir, second_dir in modes:
                if mode == "one":
                    steps = rng.randint(STUB_LEN_RANGE[0], STUB_LEN_RANGE[1])
                    cells = _line_cells(node, first_dir, steps, include_start=False)
                    allowed = {cells[0]: {node}}
                    if not _can_add_cells(cells, occupied, shape, allowed):
                        continue
                    occupied |= set(cells)
                    decoys.append({"node": node, "cells": cells, "color": color})
                    used_slots.add(slot)
                    placed = True
                    break

                steps_a = rng.randint(STUB_LEN_RANGE[0], STUB_LEN_RANGE[1])
                steps_b = rng.randint(STUB_LEN_RANGE[0], STUB_LEN_RANGE[1])
                cells_a = _line_cells(node, first_dir, steps_a, include_start=False)
                cells_b = _line_cells(node, second_dir, steps_b, include_start=False)
                allowed_a = {cells_a[0]: {node}}
                if not _can_add_cells(cells_a, occupied, shape, allowed_a):
                    continue
                allowed_b = {cells_b[0]: {node}}
                if not _can_add_cells(cells_b, occupied | set(cells_a), shape, allowed_b):
                    continue
                occupied |= set(cells_a) | set(cells_b)
                decoys.append({"node": node, "cells": cells_a + cells_b, "color": color})
                used_slots.add(slot)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return decoys


def _paint_scene(
    shape: tuple[int, int],
    bg: int,
    runs: list[dict],
    source_cells: list[tuple[int, int]],
    straight_decoys: list[dict],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)

    for run in runs:
        for r, c in run["cells"]:
            grid[r, c] = run["color"]
    for run in runs:
        for r, c in run["stub"]:
            grid[r, c] = run["color"]
    for decoy in straight_decoys:
        for r, c in decoy["cells"]:
            grid[r, c] = decoy["color"]
    for r, c in source_cells:
        grid[r, c] = MAROON
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    shape = (h, w)
    bg = rng.choice(BG_CHOICES)

    total_runs = rng.randint(1, MAX_RUNS)
    route = _build_route(rng, shape, total_runs)
    if route is None:
        return None
    runs, source_cells, _ = route

    color_pool = [c for c in range(10) if c not in {bg, MAGENTA, MAROON}]
    rng.shuffle(color_pool)
    if len(color_pool) < total_runs:
        return None
    run_colors = color_pool[:total_runs]
    for run, color in zip(runs, run_colors):
        run["color"] = color

    occupied = set(source_cells)
    for run in runs:
        occupied |= set(run["cells"])
        occupied |= set(run["stub"])

    candidate_slots = sum(max(0, len(run["cells"]) - 4) for run in runs)
    max_straight = min(MAX_STRAIGHT_NODES, candidate_slots, len(color_pool) - total_runs)
    turn_count = total_runs - 1
    if max_straight == 0:
        straight_count = 0
    elif turn_count == 0:
        straight_count = rng.randint(1, max_straight)
    elif rng.random() < 0.35:
        straight_count = 0
    else:
        target = turn_count + rng.choice([-1, 0, 1])
        straight_count = max(1, min(max_straight, target))

    straight_colors = color_pool[total_runs : total_runs + straight_count]
    straight_decoys = _place_straight_nodes(rng, shape, runs, occupied, straight_colors)
    if straight_decoys is None:
        return None

    grid = _paint_scene(shape, bg, runs, source_cells, straight_decoys)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 3")

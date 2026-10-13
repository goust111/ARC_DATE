import random
from collections import deque

import numpy as np


BG = 0
PIPE = 5
SOURCE_COLORS = (1, 2, 3, 4, 7)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
GRID_MIN = 16
GRID_MAX = 30


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _rect_border_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(left, left + width):
        cells.add((top, c))
        cells.add((top + height - 1, c))
    for r in range(top, top + height):
        cells.add((r, left))
        cells.add((r, left + width - 1))
    return cells


def _rect_all_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _non_pipe_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, PIPE):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _find_source(grid: np.ndarray) -> dict:
    candidates: list[dict] = []
    for comp in _non_pipe_components(grid):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        if r1 - r0 != 1 or c1 - c0 != 1 or len(comp["cells"]) != 4:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one 2x2 source block"
    return candidates[0]


def _is_terminal_frame(grid: np.ndarray, comp: dict, source_cells: set[tuple[int, int]]) -> bool:
    if any(cell in source_cells for cell in comp["cells"]):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return False
    if np.any(patch[1:-1, 1:-1] != BG):
        return False
    return len(comp["cells"]) == 2 * h + 2 * w - 4


def _find_terminals_and_markers(grid: np.ndarray, source_cells: set[tuple[int, int]]) -> tuple[list[dict], dict[tuple[int, int], int]]:
    terminals: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in _non_pipe_components(grid):
        cells = set(comp["cells"])
        if cells <= source_cells:
            continue
        if _is_terminal_frame(grid, comp, source_cells):
            terminals.append(
                {
                    "color": int(comp["color"]),
                    "bbox": comp["bbox"],
                    "border": cells,
                }
            )
        elif len(comp["cells"]) == 1:
            markers[comp["cells"][0]] = int(comp["color"])
    assert terminals, "expected at least one terminal frame"
    colors = [int(t["color"]) for t in terminals]
    assert len(colors) == len(set(colors)), "terminal colors must be unique"
    assert set(markers.values()) <= set(colors), "marker colors must correspond to terminal colors"
    return terminals, markers


def _subtree_terminal_colors(
    start: tuple[int, int],
    parent: tuple[int, int],
    pipe_neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    terminal_adj: dict[tuple[int, int], set[int]],
    terminals: list[dict],
    cache: dict[tuple[tuple[int, int], tuple[int, int]], set[int]],
) -> set[int]:
    key = (start, parent)
    if key in cache:
        return cache[key]
    seen = {parent}
    stack = [start]
    colors: set[int] = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for terminal_id in terminal_adj.get(cell, set()):
            colors.add(int(terminals[terminal_id]["color"]))
        for nxt in pipe_neighbors[cell]:
            if nxt not in seen:
                stack.append(nxt)
    cache[key] = colors
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source = _find_source(g)
    source_color = int(source["color"])
    source_cells = set(source["cells"])
    terminals, markers = _find_terminals_and_markers(g, source_cells)

    pipe_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == PIPE)}
    assert pipe_cells, "expected a grey pipe network"

    pipe_neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {
        cell: [nb for nb in _neighbors4(cell) if nb in pipe_cells] for cell in pipe_cells
    }
    assert max(len(nbrs) for nbrs in pipe_neighbors.values()) <= 3, "pipes must only form paths and T-junctions"

    start_pipe = {nb for cell in source_cells for nb in _neighbors4(cell) if nb in pipe_cells}
    assert len(start_pipe) == 1, "source must touch exactly one pipe endpoint"
    current = next(iter(start_pipe))
    prev: tuple[int, int] | None = None

    terminal_adj: dict[tuple[int, int], set[int]] = {}
    for terminal_id, terminal in enumerate(terminals):
        attached = set()
        for border_cell in terminal["border"]:
            for nb in _neighbors4(border_cell):
                if nb in pipe_cells:
                    attached.add(nb)
                    terminal_adj.setdefault(nb, set()).add(terminal_id)
        assert attached, "every terminal must be attached to the pipe network"

    cache: dict[tuple[tuple[int, int], tuple[int, int]], set[int]] = {}
    for _ in range(len(pipe_cells) + 5):
        adjacent_terminals = terminal_adj.get(current, set())
        if adjacent_terminals:
            assert len(adjacent_terminals) == 1, "path endpoint must identify a unique terminal"
            terminal = terminals[next(iter(adjacent_terminals))]
            r0, c0, r1, c1 = terminal["bbox"]
            out = g.copy()
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BG] = source_color
            return out

        forward = list(pipe_neighbors[current]) if prev is None else [nb for nb in pipe_neighbors[current] if nb != prev]
        assert forward, "path terminated before reaching a terminal"
        if len(forward) == 1:
            prev, current = current, forward[0]
            continue
        assert len(forward) == 2, "unexpected pipe degree while tracing"

        nearby_marker_colors = {
            color
            for marker_cell, color in markers.items()
            if max(abs(marker_cell[0] - current[0]), abs(marker_cell[1] - current[1])) == 1
        }
        assert len(nearby_marker_colors) == 1, "every T-junction must have one nearby marker"
        banned_color = next(iter(nearby_marker_colors))

        candidate_sets = [
            _subtree_terminal_colors(nxt, current, pipe_neighbors, terminal_adj, terminals, cache) for nxt in forward
        ]
        valid = [idx for idx, colors in enumerate(candidate_sets) if banned_color not in colors]
        assert len(valid) == 1, "marker color must exclude exactly one branch"
        prev, current = current, forward[valid[0]]

    raise AssertionError("failed to reach a terminal")


def _append_segment(
    pipes: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[int, int] | None:
    current = start
    for _ in range(length):
        nxt = _add(current, direction)
        if nxt in pipes or nxt in markers:
            return None
        existing_neighbors = {nb for nb in _neighbors4(nxt) if nb in pipes}
        if existing_neighbors - {current}:
            return None
        if any(nb in markers for nb in _neighbors4(nxt)):
            return None
        pipes.add(nxt)
        current = nxt
    return current


def _append_polyline(
    pipes: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    start: tuple[int, int],
    segments: list[tuple[tuple[int, int], int]],
) -> tuple[int, int] | None:
    current = start
    for direction, length in segments:
        current = _append_segment(pipes, markers, current, direction, length)
        if current is None:
            return None
    return current


def _sample_leg(first_dir: tuple[int, int], *, straight_range: tuple[int, int], turn_range: tuple[int, int]) -> tuple[list[tuple[tuple[int, int], int]], tuple[int, int]]:
    if random.random() < 0.45:
        length = random.randint(*straight_range)
        return [(first_dir, length)], first_dir
    bend_dir = random.choice(_perpendiculars(first_dir))
    first_len = random.randint(1, 3)
    second_len = random.randint(*turn_range)
    return [(first_dir, first_len), (bend_dir, second_len)], bend_dir


def _sample_blocked_leg(blocked_dir: tuple[int, int], turn_dir: tuple[int, int]) -> tuple[list[tuple[tuple[int, int], int]], tuple[int, int]]:
    if random.random() < 0.5:
        length = random.randint(2, 5)
        return [(blocked_dir, length)], blocked_dir
    first_len = random.randint(1, 3)
    second_len = random.randint(1, 3)
    return [(blocked_dir, first_len), (turn_dir, second_len)], turn_dir


def _source_cells(start_pipe: tuple[int, int], direction: tuple[int, int], offset: int) -> set[tuple[int, int]]:
    r, c = start_pipe
    if direction == (0, 1):
        top, left = r - offset, c - 2
    elif direction == (0, -1):
        top, left = r - offset, c + 1
    elif direction == (1, 0):
        top, left = r - 2, c - offset
    else:
        top, left = r + 1, c - offset
    return {(top + dr, left + dc) for dr in range(2) for dc in range(2)}


def _place_source(
    pipes: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    start_pipe: tuple[int, int],
    direction: tuple[int, int],
) -> set[tuple[int, int]] | None:
    offsets = [0, 1]
    random.shuffle(offsets)
    for offset in offsets:
        cells = _source_cells(start_pipe, direction, offset)
        if cells & pipes or any(cell in markers for cell in cells):
            continue
        if any(any(nb in markers for nb in _neighbors4(cell)) for cell in cells):
            continue
        adjacent_pipes = {pipe for cell in cells for pipe in _neighbors4(cell) if pipe in pipes}
        if adjacent_pipes == {start_pipe}:
            return cells
    return None


def _frame_geometry(
    leaf: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
    attach: int,
) -> tuple[int, int, int, int]:
    r, c = leaf
    if direction == (0, 1):
        top, left = r - attach, c + 1
    elif direction == (0, -1):
        top, left = r - attach, c - width
    elif direction == (1, 0):
        top, left = r + 1, c - attach
    else:
        top, left = r - height, c - attach
    return top, left, height, width


def _place_terminal_frame(
    pipes: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    occupied: set[tuple[int, int]],
    leaf: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(60):
        height = random.randint(3, 6)
        width = random.randint(3, 6)
        attach = random.randint(1, height - 2) if direction[0] == 0 else random.randint(1, width - 2)
        top, left, height, width = _frame_geometry(leaf, direction, height, width, attach)
        border = _rect_border_cells(top, left, height, width)
        full = _rect_all_cells(top, left, height, width)
        if full & occupied:
            continue
        adjacent_pipes = {pipe for pipe in pipes if any(nb in border for nb in _neighbors4(pipe))}
        if adjacent_pipes != {leaf}:
            continue
        if any(any(nb in border for nb in _neighbors4(marker_cell)) for marker_cell in markers):
            continue
        return (top, left, top + height - 1, left + width - 1), border, full
    return None


def _embed_layout(
    pipes: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    source_cells: set[tuple[int, int]],
    source_color: int,
    terminals: list[dict],
) -> np.ndarray | None:
    all_cells = set(pipes) | set(markers) | set(source_cells)
    for terminal in terminals:
        r0, c0, r1, c1 = terminal["bbox"]
        all_cells |= _rect_all_cells(r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    min_r, min_c, max_r, max_c = _bbox(all_cells)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    h_lo = max(GRID_MIN, box_h + 2)
    h_hi = min(GRID_MAX, box_h + 8)
    w_lo = max(GRID_MIN, box_w + 2)
    w_hi = min(GRID_MAX, box_w + 8)
    if h_lo > h_hi or w_lo > w_hi:
        return None

    height = random.randint(h_lo, h_hi)
    width = random.randint(w_lo, w_hi)
    row_shift = random.randint(1 - min_r, height - 1 - max_r)
    col_shift = random.randint(1 - min_c, width - 1 - max_c)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in pipes:
        grid[r + row_shift, c + col_shift] = PIPE
    for r, c in source_cells:
        grid[r + row_shift, c + col_shift] = source_color
    for (r, c), color in markers.items():
        grid[r + row_shift, c + col_shift] = color
    for terminal in terminals:
        color = int(terminal["color"])
        r0, c0, r1, c1 = terminal["bbox"]
        grid[r0 + row_shift : r1 + row_shift + 1, c0 + col_shift : c1 + col_shift + 1] = np.where(
            np.fromfunction(
                lambda rr, cc: (rr == 0)
                | (cc == 0)
                | (rr == (r1 - r0))
                | (cc == (c1 - c0)),
                (r1 - r0 + 1, c1 - c0 + 1),
                dtype=int,
            ),
            color,
            grid[r0 + row_shift : r1 + row_shift + 1, c0 + col_shift : c1 + col_shift + 1],
        )
    return grid


def _validate_generated(grid: np.ndarray, expected_terminal_color: int) -> bool:
    source = _find_source(grid)
    terminals, markers = _find_terminals_and_markers(grid, set(source["cells"]))
    if len(terminals) != len(markers) + 1:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False

    matches = [terminal for terminal in terminals if int(terminal["color"]) == expected_terminal_color]
    if len(matches) != 1:
        return False
    terminal = matches[0]
    r0, c0, r1, c1 = terminal["bbox"]
    expected_changed = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == BG
    }
    changed = {tuple(map(int, cell)) for cell in np.argwhere(out != grid)}
    if changed != expected_changed or not changed:
        return False
    source_color = int(source["color"])
    return all(int(out[r, c]) == source_color for r, c in changed)


def _gen_attempt() -> np.ndarray | None:
    source_color = random.choice(SOURCE_COLORS)
    terminal_palette = [color for color in range(1, 10) if color not in {PIPE, source_color}]
    random.shuffle(terminal_palette)

    junction_count = random.randint(1, 3)
    terminal_colors = terminal_palette[: junction_count + 1]

    start_pipe = (0, 0)
    start_dir = random.choice(DIR4)
    pipes: set[tuple[int, int]] = {start_pipe}
    markers: dict[tuple[int, int], int] = {}
    blocked_leaves: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    current = start_pipe
    direction = start_dir

    for junction_idx in range(junction_count):
        trunk_segments, incoming_dir = _sample_leg(direction, straight_range=(2, 5), turn_range=(2, 4))
        junction = _append_polyline(pipes, markers, current, trunk_segments)
        if junction is None:
            return None

        turn_dir = random.choice(_perpendiculars(incoming_dir))
        blocked_segments, blocked_leaf_dir = _sample_blocked_leg(incoming_dir, turn_dir)
        blocked_leaf = _append_polyline(pipes, markers, junction, blocked_segments)
        if blocked_leaf is None:
            return None

        first_blocked = _add(junction, incoming_dir)
        marker_cell = _add(junction, _sub(incoming_dir, turn_dir))
        if marker_cell in pipes or marker_cell in markers:
            return None
        marker_neighbors = {nb for nb in _neighbors4(marker_cell) if nb in pipes}
        if marker_neighbors != {first_blocked}:
            return None
        markers[marker_cell] = terminal_colors[junction_idx]

        blocked_leaves.append((blocked_leaf, blocked_leaf_dir, terminal_colors[junction_idx]))
        current = junction
        direction = turn_dir

    final_segments, final_leaf_dir = _sample_leg(direction, straight_range=(2, 5), turn_range=(1, 3))
    final_leaf = _append_polyline(pipes, markers, current, final_segments)
    if final_leaf is None:
        return None

    source_cells = _place_source(pipes, markers, start_pipe, start_dir)
    if source_cells is None:
        return None

    occupied = set(pipes) | set(markers) | set(source_cells)
    terminals: list[dict] = []
    for leaf, leaf_dir, color in blocked_leaves + [(final_leaf, final_leaf_dir, terminal_colors[-1])]:
        placed = _place_terminal_frame(pipes, markers, occupied, leaf, leaf_dir)
        if placed is None:
            return None
        bbox, border, full = placed
        occupied |= full
        terminals.append({"color": color, "bbox": bbox, "border": border})

    grid = _embed_layout(pipes, markers, source_cells, source_color, terminals)
    if grid is None:
        return None
    if not _validate_generated(grid, terminal_colors[-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network ARC input")

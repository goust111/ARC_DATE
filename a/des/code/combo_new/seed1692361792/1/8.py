import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
VALVE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_objects(grid: np.ndarray):
    valves: list[dict[str, object]] = []
    clues_by_color: dict[int, list[tuple[int, int]]] = {}
    colors = sorted(set(int(v) for v in np.unique(grid)) - {BLACK, GREY, RED})
    for color in colors:
        clues_by_color[color] = []
        for comp in _components_of_color(grid, color):
            if len(comp) == 1:
                clues_by_color[color].append(comp[0])
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 2 and ((r1 - r0 + 1, c1 - c0 + 1) in {(1, 2), (2, 1)}):
                valves.append({"color": color, "cells": comp, "bbox": (r0, c0, r1, c1)})
                continue
            raise AssertionError(f"unexpected non-pipe object with color {color}: {comp}")
    return valves, clues_by_color


def _valve_is_open(valve: dict[str, object], clues_by_color: dict[int, list[tuple[int, int]]]) -> bool:
    color = int(valve["color"])
    r0, c0, r1, c1 = valve["bbox"]
    count = 0
    for r, c in clues_by_color.get(color, []):
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            count += 1
    return count % 2 == 0


def _reachable_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sources = np.argwhere(g == RED)
    assert len(sources) == 1, f"expected exactly one red source, got {len(sources)}"
    sr, sc = map(int, sources[0])

    valves, clues_by_color = _classify_objects(g)
    passable = np.zeros((h, w), dtype=bool)
    passable[g == GREY] = True
    passable[sr, sc] = True
    for valve in valves:
        if _valve_is_open(valve, clues_by_color):
            for r, c in valve["cells"]:
                passable[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every pipe segment that liquid can reach through open valves.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-wide horizontal or vertical pipe segments.
    - Exactly one red pixel marks the liquid source on a pipe.
    - Every other non-black, non-grey, non-red object is either a 2-cell valve
      domino or a same-colored single-pixel clue near that valve.

    Rule:
    - For each valve, count same-colored single-pixel clue blocks inside the
      valve's 8-neighborhood, excluding the valve cells themselves.
    - Even counts mean the valve is open; odd counts mean the valve is closed.
    - Starting from the red source, liquid moves 4-connected through grey pipe
      pixels and through open valve pixels, but never through closed valves.
    - Recolor every reachable grey pipe pixel to red and leave all valves,
      clues, unreachable pipes, and the source unchanged.

    Key insight:
    - The puzzle combines a local parity test around each 2-cell valve with a
      global reachability flood through the pipe network.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    reachable = _reachable_mask(g)
    out = g.copy()
    out[(g == GREY) & reachable] = RED
    return out


def _grid_shape() -> tuple[int, int, int, int, int]:
    while True:
        node_rows = random.randint(2, 4)
        node_cols = random.randint(2, 4)
        spacing = random.randint(5, 7)
        margin = random.randint(2, 3)
        height = 2 * margin + (node_rows - 1) * spacing + 1
        width = 2 * margin + (node_cols - 1) * spacing + 1
        if height <= 30 and width <= 30:
            return node_rows, node_cols, spacing, margin, height, width


def _node_coords(node_rows: int, node_cols: int, spacing: int, margin: int) -> dict[tuple[int, int], tuple[int, int]]:
    coords: dict[tuple[int, int], tuple[int, int]] = {}
    for nr in range(node_rows):
        for nc in range(node_cols):
            coords[(nr, nc)] = (margin + nr * spacing, margin + nc * spacing)
    return coords


def _adjacent_edges(node_rows: int, node_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r in range(node_rows):
        for c in range(node_cols):
            if r + 1 < node_rows:
                edges.append(((r, c), (r + 1, c)))
            if c + 1 < node_cols:
                edges.append(((r, c), (r, c + 1)))
    return edges


def _spanning_tree(nodes: list[tuple[int, int]], all_edges: list[tuple[tuple[int, int], tuple[int, int]]]):
    edge_set = {}
    for a, b in all_edges:
        edge_set.setdefault(a, []).append((a, b))
        edge_set.setdefault(b, []).append((a, b))
    start = random.choice(nodes)
    visited = {start}
    tree = []
    frontier = edge_set[start][:]
    while frontier:
        edge = random.choice(frontier)
        frontier.remove(edge)
        a, b = edge
        na = b if a in visited else a
        nb = a if a in visited else b
        if na in visited and nb in visited:
            continue
        new_node = na if na not in visited else nb
        visited.add(new_node)
        tree.append(edge)
        frontier.extend(edge_set[new_node])
    assert len(tree) == len(nodes) - 1
    return tree


def _draw_pipe(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo : c_hi + 1] = GREY
    else:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo : r_hi + 1, c0] = GREY


def _edge_axis_points(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[list[int], str]:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        return list(range(c_lo + 2, c_hi - 1)), "h"
    r_lo, r_hi = sorted((r0, r1))
    return list(range(r_lo + 2, r_hi - 1)), "v"


def _clue_candidates(h: int, w: int, valve_cells: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    cells = set(valve_cells)
    r0, c0, r1, c1 = _bbox(valve_cells)
    candidates = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if (r, c) in cells or int(grid[r, c]) != BLACK:
                continue
            if any(abs(r - vr) + abs(c - vc) == 1 for vr, vc in valve_cells):
                continue
            if any(max(abs(r - vr), abs(c - vc)) <= 1 for vr, vc in valve_cells):
                candidates.append((r, c))
    return candidates


def _place_valve(
    grid: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: int,
    want_open: bool,
) -> bool:
    h, w = grid.shape
    axis_points, orient = _edge_axis_points(p0, p1)
    random.shuffle(axis_points)
    side_choices = [-1, 1]
    random.shuffle(side_choices)

    for axis in axis_points:
        for side in side_choices:
            if orient == "h":
                row = p0[0]
                base = (row, axis)
                extra = (row + side, axis)
            else:
                col = p0[1]
                base = (axis, col)
                extra = (axis, col + side)

            er, ec = extra
            if not (0 <= er < h and 0 <= ec < w):
                continue
            if int(grid[er, ec]) != BLACK:
                continue

            valve_cells = [base, extra]
            candidates = _clue_candidates(h, w, valve_cells, grid)
            if len(candidates) < 3:
                continue

            grid[base] = color
            grid[extra] = color

            parity_counts = [0, 2, 4] if want_open else [1, 3]
            parity_counts = [count for count in parity_counts if count <= len(candidates)]
            assert parity_counts
            clue_count = random.choice(parity_counts)
            random.shuffle(candidates)
            for r, c in candidates[:clue_count]:
                grid[r, c] = color
            return True
    return False


def _validate_generator_assumptions(grid: np.ndarray) -> bool:
    try:
        valves, clues_by_color = _classify_objects(grid)
        if len(np.argwhere(grid == RED)) != 1 or not valves:
            return False
        for valve in valves:
            color = int(valve["color"])
            count = sum(
                1
                for r, c in clues_by_color.get(color, [])
                if valve["bbox"][0] - 1 <= r <= valve["bbox"][2] + 1
                and valve["bbox"][1] - 1 <= c <= valve["bbox"][3] + 1
            )
            if count > 4:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    node_rows, node_cols, spacing, margin, height, width = _grid_shape()
    grid = np.full((height, width), BLACK, dtype=int)

    nodes = [(r, c) for r in range(node_rows) for c in range(node_cols)]
    coords = _node_coords(node_rows, node_cols, spacing, margin)
    all_edges = _adjacent_edges(node_rows, node_cols)
    tree_edges = _spanning_tree(nodes, all_edges)
    remaining = [edge for edge in all_edges if edge not in tree_edges]
    random.shuffle(remaining)
    extra_count = random.randint(0, min(2, len(remaining)))
    edges = tree_edges + remaining[:extra_count]

    for a, b in edges:
        _draw_pipe(grid, coords[a], coords[b])

    max_valves = min(6, len(tree_edges), len(VALVE_COLORS))
    if max_valves < 2:
        return None
    valve_count = random.randint(2, max_valves)
    valve_edges = random.sample(tree_edges, valve_count)
    valve_colors = list(VALVE_COLORS[:valve_count])
    random.shuffle(valve_colors)

    target_states = [True] * (valve_count // 2) + [False] * (valve_count - valve_count // 2)
    if all(target_states) or not any(target_states):
        target_states[0] = True
        target_states[-1] = False
    random.shuffle(target_states)

    for edge, color, want_open in zip(valve_edges, valve_colors, target_states):
        if not _place_valve(grid, coords[edge[0]], coords[edge[1]], color, want_open):
            return None

    grey_cells = [tuple(x) for x in np.argwhere(grid == GREY)]
    random.shuffle(grey_cells)
    for sr, sc in grey_cells:
        if sum(1 for nr, nc in _neighbors4(sr, sc) if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) != BLACK) >= 1:
            grid[sr, sc] = RED
            break
    else:
        return None

    if not _validate_generator_assumptions(grid):
        return None

    out = main(grid)
    reached = int(np.sum((grid == GREY) & (out == RED)))
    unreached = int(np.sum((grid == GREY) & (out == GREY)))
    if reached == 0 or unreached == 0:
        return None

    valves, clues_by_color = _classify_objects(grid)
    open_count = sum(1 for valve in valves if _valve_is_open(valve, clues_by_color))
    closed_count = len(valves) - open_count
    if open_count == 0 or closed_count == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid pipe-and-valve input")

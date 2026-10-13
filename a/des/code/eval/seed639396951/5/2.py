from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7
GREY = 5
MAROON = 9
PATH_PALETTE = (1, 2, 3, 4, 6, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATION_RETRIES = 400


def _neighbors(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    special = {bg, ORANGE, GREY, MAROON}
    mask = ~np.isin(grid, list(special))
    candidates = []
    for comp in _components(mask):
        rows = sorted(r for r, _ in comp)
        cols = {c for _, c in comp}
        if len(comp) < 2 or len(cols) != 1:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        colors = [int(grid[r, next(iter(cols))]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        r0 = rows[0]
        c0 = next(iter(cols))
        candidates.append((r0 + c0, r0, c0, colors, set(comp)))

    assert candidates, "expected a vertical priority legend in the top-left"
    _, r0, c0, colors, coords = min(candidates)
    assert r0 < grid.shape[0] // 2 and c0 < grid.shape[1] // 2, "legend must sit near the top-left"
    return colors, coords


def _extract_graph(
    grid: np.ndarray, legend_colors: list[int], legend_coords: set[tuple[int, int]]
) -> tuple[
    tuple[int, int],
    dict[tuple[int, int], str],
    dict[int, dict[str, object]],
    dict[tuple[int, int], list[int]],
]:
    orange_cells = np.argwhere(grid == ORANGE)
    assert orange_cells.shape == (1, 2), "expected one orange start"
    start = tuple(map(int, orange_cells[0]))
    h, w = grid.shape
    assert start[0] in (0, h - 1) or start[1] in (0, w - 1), "start must sit on the border"

    node_kind: dict[tuple[int, int], str] = {start: "start"}
    for r, c in np.argwhere(grid == GREY):
        node_kind[(int(r), int(c))] = "terminal"
    for r, c in np.argwhere(grid == MAROON):
        node_kind[(int(r), int(c))] = "intersection"
    assert any(kind == "intersection" for kind in node_kind.values()), "expected at least one intersection"
    assert any(kind == "terminal" for kind in node_kind.values()), "expected at least one terminal"

    edges: dict[int, dict[str, object]] = {}
    node_edges = {coord: [] for coord in node_kind}

    for color in legend_colors:
        comps = _components(grid == color)
        path_comps = [comp for comp in comps if any(cell not in legend_coords for cell in comp)]
        assert len(path_comps) == 1, f"expected exactly one path component for color {color}"
        cells = set(path_comps[0])
        endpoints = set()
        for r, c in cells:
            for nr, nc in _neighbors(grid.shape, r, c):
                if int(grid[nr, nc]) in (ORANGE, GREY, MAROON):
                    endpoints.add((nr, nc))
        assert len(endpoints) == 2, f"path color {color} must connect exactly two special nodes"
        a, b = tuple(sorted(endpoints))
        edges[color] = {"cells": cells, "endpoints": (a, b)}
        node_edges[a].append(color)
        node_edges[b].append(color)

    assert len(node_edges[start]) == 1, "start must feed into exactly one path"
    return start, node_kind, edges, node_edges


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    legend_colors, legend_coords = _find_legend(g, bg)
    start, node_kind, edges, node_edges = _extract_graph(g, legend_colors, legend_coords)

    out = g.copy()
    for edge in edges.values():
        for r, c in edge["cells"]:
            out[r, c] = bg
    out[g == MAROON] = bg

    visited_edges: set[int] = set()
    visited_intersections: set[tuple[int, int]] = set()
    lit_terminals: dict[tuple[int, int], int] = {}

    current_node = start
    current_color = node_edges[start][0]
    while True:
        assert current_color not in visited_edges, "path colors cannot be traversed twice"
        edge = edges[current_color]
        visited_edges.add(current_color)
        for r, c in edge["cells"]:
            out[r, c] = current_color

        a, b = edge["endpoints"]
        assert current_node == a or current_node == b, "current node must be an endpoint of the selected path"
        next_node = b if current_node == a else a
        next_kind = node_kind[next_node]

        if next_kind == "terminal":
            lit_terminals[next_node] = current_color
            break

        assert next_kind == "intersection", "paths may continue only through intersections"
        visited_intersections.add(next_node)
        current_node = next_node
        candidates = [
            color
            for color in legend_colors
            if color in node_edges[current_node] and color not in visited_edges
        ]
        if not candidates:
            break
        current_color = candidates[0]

    for r, c in visited_intersections:
        out[r, c] = MAROON
    for (r, c), color in lit_terminals.items():
        out[r, c] = color
    return out


def _place_legend(grid: np.ndarray, legend_colors: list[int]) -> None:
    for r, color in enumerate(legend_colors):
        assert grid[r, 0] == BG
        grid[r, 0] = color


def _place_node(grid: np.ndarray, coord: tuple[int, int], color: int) -> None:
    r, c = coord
    assert grid[r, c] == BG
    grid[r, c] = color


def _draw_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1, step):
            assert grid[r0, c] == BG
            grid[r0, c] = color
        return
    assert c0 == c1, "only axis-aligned paths are supported by the generator"
    step = 1 if r1 > r0 else -1
    for r in range(r0 + step, r1, step):
        assert grid[r, c0] == BG
        grid[r, c0] = color


def _shuffle_until(
    names: list[str], constraints: list[tuple[str, str]], rng: random.Random
) -> list[str]:
    for _ in range(2000):
        order = names[:]
        rng.shuffle(order)
        pos = {name: i for i, name in enumerate(order)}
        if all(pos[hi] < pos[lo] for hi, lo in constraints):
            return order
    raise RuntimeError("could not satisfy priority constraints")


def _build_chain_case(rng: random.Random) -> np.ndarray:
    h = rng.randint(15, 22)
    w = rng.randint(20, 28)
    grid = np.full((h, w), BG, dtype=int)

    edge_names = ["start", "i0_up", "i0_down", "bridge", "i1_up", "i1_down"]
    colors = list(PATH_PALETTE)
    rng.shuffle(colors)
    edge_color = dict(zip(edge_names, colors))

    behavior = rng.choice(["early", "deep"])
    constraints: list[tuple[str, str]] = []
    if behavior == "early":
        chosen = rng.choice(["i0_up", "i0_down"])
        other = "i0_down" if chosen == "i0_up" else "i0_up"
        constraints = [(chosen, other), (chosen, "bridge")]
    else:
        chosen = rng.choice(["i1_up", "i1_down"])
        other = "i1_down" if chosen == "i1_up" else "i1_up"
        constraints = [("bridge", "i0_up"), ("bridge", "i0_down"), (chosen, other)]

    legend_order = _shuffle_until(edge_names, constraints, rng)
    _place_legend(grid, [edge_color[name] for name in legend_order])

    row = rng.randint(8, h - 4)
    top0 = rng.randint(1, row - 2)
    bot0 = rng.randint(row + 2, h - 2)
    top1 = rng.randint(1, row - 2)
    bot1 = rng.randint(row + 2, h - 2)

    start_side = rng.choice(["left", "right"])
    if start_side == "left":
        c0 = rng.randint(4, 7)
        c1 = rng.randint(max(c0 + 5, w // 2), w - 5)
        start = (row, 0)
    else:
        c0 = rng.randint(w - 8, w - 5)
        c1 = rng.randint(4, c0 - 5)
        start = (row, w - 1)

    i0 = (row, c0)
    i1 = (row, c1)
    t0u = (top0, c0)
    t0d = (bot0, c0)
    t1u = (top1, c1)
    t1d = (bot1, c1)

    for coord, color in [
        (start, ORANGE),
        (i0, MAROON),
        (i1, MAROON),
        (t0u, GREY),
        (t0d, GREY),
        (t1u, GREY),
        (t1d, GREY),
    ]:
        _place_node(grid, coord, color)

    _draw_line(grid, start, i0, edge_color["start"])
    _draw_line(grid, i0, t0u, edge_color["i0_up"])
    _draw_line(grid, i0, t0d, edge_color["i0_down"])
    _draw_line(grid, i0, i1, edge_color["bridge"])
    _draw_line(grid, i1, t1u, edge_color["i1_up"])
    _draw_line(grid, i1, t1d, edge_color["i1_down"])
    return grid


def _build_cycle_case(rng: random.Random) -> np.ndarray:
    h = rng.randint(15, 22)
    w = rng.randint(18, 26)
    grid = np.full((h, w), BG, dtype=int)

    edge_names = ["start", "e01", "e12", "e23", "e30", "term"]
    colors = list(PATH_PALETTE)
    rng.shuffle(colors)
    edge_color = dict(zip(edge_names, colors))

    direction = rng.choice(["cw", "ccw"])
    attach = rng.choice(["i1", "i2"])
    constraints: list[tuple[str, str]] = []
    if direction == "cw":
        constraints.append(("e01", "e30"))
        constraints.append(("e12", "term") if attach == "i1" else ("e23", "term"))
    else:
        constraints.append(("e30", "e01"))
        constraints.append(("e01", "term") if attach == "i1" else ("e12", "term"))

    legend_order = _shuffle_until(edge_names, constraints, rng)
    _place_legend(grid, [edge_color[name] for name in legend_order])

    r0 = rng.randint(8, h - 7)
    r1 = rng.randint(r0 + 4, h - 3)
    c0 = rng.randint(4, 7)
    c1 = rng.randint(c0 + 4, w - 4)

    start = (r0, 0)
    i0 = (r0, c0)
    i1 = (r0, c1)
    i2 = (r1, c1)
    i3 = (r1, c0)
    term = (r0, w - 1) if attach == "i1" else (h - 1, c1)

    for coord, color in [
        (start, ORANGE),
        (i0, MAROON),
        (i1, MAROON),
        (i2, MAROON),
        (i3, MAROON),
        (term, GREY),
    ]:
        _place_node(grid, coord, color)

    _draw_line(grid, start, i0, edge_color["start"])
    _draw_line(grid, i0, i1, edge_color["e01"])
    _draw_line(grid, i1, i2, edge_color["e12"])
    _draw_line(grid, i2, i3, edge_color["e23"])
    _draw_line(grid, i3, i0, edge_color["e30"])
    _draw_line(grid, i1 if attach == "i1" else i2, term, edge_color["term"])
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _build_chain_case(rng) if rng.random() < 0.5 else _build_cycle_case(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

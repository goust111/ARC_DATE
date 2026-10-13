from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_path(component: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    cells = set(component)
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(*cur) if p in cells and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _analyze_snake(grid: np.ndarray, component: list[tuple[int, int]]) -> dict | None:
    cells = set(component)
    if len(cells) < 2:
        return None
    degrees = {
        cell: sum(1 for n in _neighbors4(*cell) if n in cells)
        for cell in cells
    }
    if any(d > 2 for d in degrees.values()):
        return None
    endpoints = sorted([cell for cell, d in degrees.items() if d == 1])
    if len(endpoints) != 2:
        return None
    order = _trace_path(component, endpoints[0])
    if len(order) != len(component):
        return None
    return {
        "cells": cells,
        "endpoints": endpoints,
        "length": len(component),
    }


def _ring_coords(side: int, offset: int) -> list[tuple[int, int]]:
    lo = offset
    hi = side - 1 - offset
    coords = [(lo, c) for c in range(lo, hi + 1)]
    coords.extend((r, hi) for r in range(lo + 1, hi + 1))
    if hi > lo:
        coords.extend((hi, c) for c in range(hi - 1, lo - 1, -1))
    if hi > lo:
        coords.extend((r, lo) for r in range(hi - 1, lo, -1))
    return coords


def _paint_ring(square: np.ndarray, offset: int, color: int) -> None:
    side = square.shape[0]
    for r, c in _ring_coords(side, offset):
        square[r, c] = color


def _analyze_rings(grid: np.ndarray, component: list[tuple[int, int]], bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(component)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != width or height < 3:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == bg):
        return None
    side = height
    outer_mask = np.zeros_like(sub, dtype=bool)
    outer_mask[0, :] = True
    outer_mask[-1, :] = True
    outer_mask[:, 0] = True
    outer_mask[:, -1] = True
    outer_colors = np.unique(sub[outer_mask])
    if len(outer_colors) != 1:
        return None
    outer_color = int(outer_colors[0])
    if side > 2:
        interior = sub[1:-1, 1:-1]
        inner_colors = np.unique(interior)
        if len(inner_colors) != 1:
            return None
        inner_color = int(inner_colors[0])
        if inner_color == outer_color:
            return None
    else:
        inner_color = None
    if len(component) != side * side:
        return None
    layers = (side + 1) // 2
    if layers < 2:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "layers": layers,
        "outer_color": outer_color,
        "inner_color": inner_color,
    }


def _parse_input(grid: np.ndarray) -> tuple[dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g != bg)
    ring_infos = []
    snake_infos = []
    for comp in comps:
        ring_info = _analyze_rings(g, comp, bg)
        if ring_info is not None:
            ring_infos.append(ring_info)
        snake_info = _analyze_snake(g, comp)
        if snake_info is not None:
            snake_infos.append(snake_info)
    assert len(ring_infos) == 1
    assert len(snake_infos) == 1
    ring = ring_infos[0]
    snake = snake_infos[0]
    assert snake["length"] == ring["layers"]
    endpoints = snake["endpoints"]
    endpoint_colors = [int(g[r, c]) for r, c in endpoints]
    assert endpoint_colors[0] != endpoint_colors[1]
    start_candidates = [p for p in endpoints if int(g[p]) == ring["outer_color"]]
    assert len(start_candidates) == 1
    snake["start"] = start_candidates[0]
    return ring, snake


def _render_output(side: int, inner_color: int, colors: list[int]) -> np.ndarray:
    out = np.full((side, side), inner_color, dtype=int)
    for offset, color in enumerate(colors):
        _paint_ring(out, offset, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject the snake's anchored traversal colors into concentric square rings from outside to inside.

    Input structure:
    - The grid has a background, one solid square made of concentric one-cell-thick layers, and one separate 4-connected snake.
    - The square's outer ring is already colored while every inner layer is gray.
    - The snake has exactly as many cells as the square has layers, exactly two endpoints, and exactly one endpoint matches the outer-ring color.

    Rule:
    - Detect the square and count its layers from the square size.
    - Start tracing the snake from the endpoint whose color equals the square's outer-ring color.
    - Use the 1st, 2nd, ... snake colors to recolor the 1st, 2nd, ... rings from outside to inside, then erase the snake and crop to the square's bounding box.

    Key insight:
    - The snake is an ordered color strip; traversal direction is fixed by the endpoint whose color agrees with the outer ring, not by the top-left endpoint.
    """
    g = np.asarray(grid, dtype=int)
    ring, snake = _parse_input(g)
    order = _trace_path(list(snake["cells"]), snake["start"])
    colors = [int(g[r, c]) for r, c in order]
    assert colors[0] == ring["outer_color"]
    inner_color = ring["inner_color"]
    assert inner_color is not None
    return _render_output(ring["side"], inner_color, colors)


def _build_path(length: int) -> list[tuple[int, int]] | None:
    start = (0, 0)
    used = {start}
    path = [start]

    def rec() -> bool:
        if len(path) == length:
            rows = [r for r, _ in path]
            cols = [c for _, c in path]
            return (max(rows) > min(rows)) and (max(cols) > min(cols))
        cur = path[-1]
        moves = _neighbors4(*cur)
        random.shuffle(moves)
        for nxt in moves:
            if nxt in used:
                continue
            if sum(1 for p in _neighbors4(*nxt) if p in used) != 1:
                continue
            used.add(nxt)
            path.append(nxt)
            if rec():
                return True
            path.pop()
            used.remove(nxt)
        return False

    if not rec():
        return None
    r0, c0, _, _ = _bbox(path)
    return [(r - r0, c - c0) for r, c in path]


def _make_snake_colors(length: int, outer_color: int, bg: int) -> list[int]:
    palette = [c for c in range(10) if c not in {bg, GREY}]
    if outer_color not in palette:
        palette.append(outer_color)
    colors = [outer_color]
    for idx in range(1, length):
        choices = [c for c in palette if c != bg and c != GREY]
        if idx == length - 1:
            choices = [c for c in choices if c != outer_color]
        if idx > 0 and random.random() < 0.35:
            choices = [c for c in choices if c == colors[random.randrange(len(colors))] or c != colors[-1]]
        color = random.choice(choices)
        if len(set(colors)) < 3 and idx < length - 1:
            rare = [c for c in choices if c not in colors and c != outer_color]
            if rare and random.random() < 0.7:
                color = random.choice(rare)
        colors.append(color)
    return colors


def _expanded(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_objects(
    ring_side: int,
    snake_cells: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    snake_r0, snake_c0, snake_r1, snake_c1 = _bbox(snake_cells)
    snake_h = snake_r1 - snake_r0 + 1
    snake_w = snake_c1 - snake_c0 + 1
    min_h = max(ring_side, snake_h) + 6
    min_w = max(ring_side, snake_w) + 6
    if min_h > 30 or min_w > 30:
        return None
    for _ in range(80):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        ring_r = random.randint(1, h - ring_side - 1)
        ring_c = random.randint(1, w - ring_side - 1)
        snake_r = random.randint(1, h - snake_h - 1)
        snake_c = random.randint(1, w - snake_w - 1)
        ring_cells = {
            (ring_r + rr, ring_c + cc)
            for rr in range(ring_side)
            for cc in range(ring_side)
        }
        translated_snake = {
            (snake_r + rr, snake_c + cc)
            for rr, cc in snake_cells
        }
        if translated_snake & _expanded(ring_cells):
            continue
        return h, w, ring_r, ring_c, snake_r, snake_c
    return None


def _expected_from_spec(side: int, inner_color: int, snake_colors: list[int]) -> np.ndarray:
    return _render_output(side, inner_color, snake_colors)


def _validate_generated_case(
    grid: np.ndarray,
    want_anchor_top_left: bool,
    expected: np.ndarray,
) -> bool:
    try:
        ring, snake = _parse_input(grid)
    except Exception:
        return False
    g = np.asarray(grid, dtype=int)
    if ring["inner_color"] != GREY:
        return False
    order = _trace_path(list(snake["cells"]), snake["start"])
    colors = [int(g[r, c]) for r, c in order]
    if colors[0] != ring["outer_color"]:
        return False
    top_left_endpoint = min(snake["endpoints"])
    if (snake["start"] == top_left_endpoint) != want_anchor_top_left:
        return False
    if snake["start"] != top_left_endpoint:
        wrong_order = _trace_path(list(snake["cells"]), top_left_endpoint)
        wrong_colors = [int(g[r, c]) for r, c in wrong_order]
        if wrong_colors == colors:
            return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(4, 7)
    ring_side = 2 * layer_count - 1 + random.randint(0, 1)
    if ring_side > 14:
        ring_side = 2 * layer_count - 1

    outer_color = random.choice([c for c in range(10) if c != GREY])
    bg = random.choice([c for c in range(10) if c not in {GREY, outer_color}])
    want_anchor_top_left = random.random() < 0.5

    snake_cells = None
    for _ in range(80):
        candidate = _build_path(layer_count)
        if candidate is None:
            continue
        if (candidate[0] < candidate[-1]) != want_anchor_top_left:
            candidate = list(reversed(candidate))
        if (candidate[0] < candidate[-1]) != want_anchor_top_left:
            continue
        snake_cells = candidate
        break
    if snake_cells is None:
        return None

    snake_colors = _make_snake_colors(layer_count, outer_color, bg)
    placement = _place_objects(ring_side, snake_cells)
    if placement is None:
        return None
    h, w, ring_r, ring_c, snake_r, snake_c = placement

    grid = np.full((h, w), bg, dtype=int)
    grid[ring_r : ring_r + ring_side, ring_c : ring_c + ring_side] = GREY
    _paint_ring(grid[ring_r : ring_r + ring_side, ring_c : ring_c + ring_side], 0, outer_color)
    for (rr, cc), color in zip(snake_cells, snake_colors):
        grid[snake_r + rr, snake_c + cc] = color

    expected = _expected_from_spec(ring_side, GREY, snake_colors)
    if not _validate_generated_case(grid, want_anchor_top_left, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-into-rings input")

from __future__ import annotations

import random

import numpy as np


BG = 0
WALL_FALLBACK = 5
ROOM_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_BAND_RANGE = (5, 8)
ROOM_H_RANGE = (2, 4)
ROOM_W_RANGE = (2, 5)
MACRO_ROWS = (2, 3)
MACRO_COLS = (2, 3)
ROOM_COUNT_RANGE = (2, 5)
MARGIN_RANGE = (1, 3)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _detect_wall_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g[g != BG], return_counts=True)
    if colors.size == 0:
        return WALL_FALLBACK
    return int(colors[counts.argmax()])


def _inside_mask(g: np.ndarray, wall: int) -> np.ndarray:
    traversable = g != wall
    mask = np.zeros_like(traversable, dtype=bool)
    for comp in _components(traversable):
        if not _touches_border(comp, g.shape[0], g.shape[1]):
            for r, c in comp:
                mask[r, c] = True
    return mask


def _room_core_mask(inside: np.ndarray) -> np.ndarray:
    h, w = inside.shape
    core = np.zeros_like(inside, dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if inside[r : r + 2, c : c + 2].all():
                core[r : r + 2, c : c + 2] = True
    return core


def _extract_rooms(g: np.ndarray, inside: np.ndarray, wall: int) -> list[dict]:
    rooms: list[dict] = []
    seed_total = int(np.count_nonzero(inside & (g != BG) & (g != wall)))

    for comp in _components(_room_core_mask(inside)):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = height * width
        if area < 4 or len(comp) != area:
            continue
        if not inside[r0 : r1 + 1, c0 : c1 + 1].all():
            continue
        seeds = [
            (r, c, int(g[r, c]))
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(g[r, c]) not in (BG, wall)
        ]
        if len(seeds) != 1:
            continue
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": int(seeds[0][2]),
            }
        )

    assert rooms and len(rooms) == seed_total, "failed to identify rectangular seed rooms"
    return rooms


def _corridor_order(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        return sorted(comp, key=lambda rc: rc[1])
    if len(cols) == 1:
        return sorted(comp, key=lambda rc: rc[0])

    cell_set = set(comp)
    graph = {cell: [n for n in _neighbors4(*cell) if n in cell_set] for cell in cell_set}
    ends = sorted([cell for cell, ns in graph.items() if len(ns) == 1])
    assert len(ends) == 2, "corridor must be a simple path"

    ordered = [ends[0]]
    prev = None
    cur = ends[0]
    while cur != ends[1]:
        nxts = [n for n in graph[cur] if n != prev]
        assert len(nxts) == 1, "corridor path must not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(comp)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room with its seed color, then split every connecting
    1-pixel corridor at the midpoint between the two adjacent rooms.
    """
    g = np.asarray(grid, dtype=int)
    wall = _detect_wall_color(g)
    inside = _inside_mask(g, wall)
    rooms = _extract_rooms(g, inside, wall)

    room_mask = np.zeros_like(inside, dtype=bool)
    room_id = -np.ones(g.shape, dtype=int)
    for idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]
        room_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        room_id[r0 : r1 + 1, c0 : c1 + 1] = idx

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = room["color"]

    corridor_mask = inside & (~room_mask)
    for comp in _components(corridor_mask):
        ordered = _corridor_order(comp)

        def adjacent_room_ids(cell: tuple[int, int]) -> set[int]:
            r, c = cell
            ids = set()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and room_id[nr, nc] >= 0:
                    ids.add(int(room_id[nr, nc]))
            return ids

        left_ids = adjacent_room_ids(ordered[0])
        right_ids = adjacent_room_ids(ordered[-1])
        assert len(left_ids) == 1 and len(right_ids) == 1, "corridor endpoints must touch rooms"
        left_room = rooms[next(iter(left_ids))]
        right_room = rooms[next(iter(right_ids))]
        assert left_room is not right_room, "corridor must connect two distinct rooms"

        split = (len(ordered) + 1) // 2
        for r, c in ordered[:split]:
            out[r, c] = left_room["color"]
        for r, c in ordered[split:]:
            out[r, c] = right_room["color"]

    return out


def _sample_tree(rows: int, cols: int, n_rooms: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    start = (random.randrange(rows), random.randrange(cols))
    chosen = {start}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(chosen) < n_rooms:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in chosen:
                    frontier.append(((r, c), (nr, nc)))
        assert frontier, "macro grid cannot grow requested room tree"
        parent, child = random.choice(frontier)
        chosen.add(child)
        edges.append((parent, child))

    return sorted(chosen), edges


def _band_spans(lengths: list[int], start: int) -> list[tuple[int, int]]:
    spans = []
    cur = start
    for length in lengths:
        spans.append((cur, cur + length - 1))
        cur += length
    return spans


def _place_room(
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    axis_row: int,
    axis_col: int,
) -> tuple[int, int, int, int]:
    rs, re = row_span
    cs, ce = col_span
    max_h = min(ROOM_H_RANGE[1], re - rs - 1)
    max_w = min(ROOM_W_RANGE[1], ce - cs - 1)
    assert max_h >= ROOM_H_RANGE[0] and max_w >= ROOM_W_RANGE[0]

    height = random.randint(ROOM_H_RANGE[0], max_h)
    width = random.randint(ROOM_W_RANGE[0], max_w)

    top_lo = max(rs + 1, axis_row - height + 1)
    top_hi = min(axis_row, re - height)
    left_lo = max(cs + 1, axis_col - width + 1)
    left_hi = min(axis_col, ce - width)
    assert top_lo <= top_hi and left_lo <= left_hi

    r0 = random.randint(top_lo, top_hi)
    c0 = random.randint(left_lo, left_hi)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _render_expected(
    inp: np.ndarray,
    rooms: dict[tuple[int, int], dict],
    corridors: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for room in rooms.values():
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = room["color"]
    for corridor in corridors:
        cells = corridor["cells"]
        split = (len(cells) + 1) // 2
        for r, c in cells[:split]:
            out[r, c] = corridor["lo_color"]
        for r, c in cells[split:]:
            out[r, c] = corridor["hi_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    macro_rows = random.randint(*MACRO_ROWS)
    macro_cols = random.randint(*MACRO_COLS)
    max_rooms = min(ROOM_COUNT_RANGE[1], macro_rows * macro_cols, len(ROOM_PALETTE))
    room_count = random.randint(ROOM_COUNT_RANGE[0], max_rooms)

    band_heights = [random.randint(*GRID_BAND_RANGE) for _ in range(macro_rows)]
    band_widths = [random.randint(*GRID_BAND_RANGE) for _ in range(macro_cols)]

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    box_h = sum(band_heights) + 2
    box_w = sum(band_widths) + 2
    h = top_margin + box_h + bottom_margin
    w = left_margin + box_w + right_margin

    box_r0 = top_margin
    box_c0 = left_margin
    inner_r0 = box_r0 + 1
    inner_c0 = box_c0 + 1

    row_spans = _band_spans(band_heights, inner_r0)
    col_spans = _band_spans(band_widths, inner_c0)
    row_axes = [random.randint(rs + 1, re - 1) for rs, re in row_spans]
    col_axes = [random.randint(cs + 1, ce - 1) for cs, ce in col_spans]

    nodes, edges = _sample_tree(macro_rows, macro_cols, room_count)
    colors = random.sample(ROOM_PALETTE, room_count)

    grid = np.full((h, w), BG, dtype=int)
    grid[box_r0 : box_r0 + box_h, box_c0 : box_c0 + box_w] = WALL_FALLBACK

    rooms: dict[tuple[int, int], dict] = {}
    for node, color in zip(nodes, colors, strict=False):
        rr, cc = node
        bbox = _place_room(row_spans[rr], col_spans[cc], row_axes[rr], col_axes[cc])
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
        rooms[node] = {"bbox": bbox, "color": color}

    corridors: list[dict] = []
    carved = np.zeros_like(grid, dtype=bool)
    for room in rooms.values():
        r0, c0, r1, c1 = room["bbox"]
        carved[r0 : r1 + 1, c0 : c1 + 1] = True

    for a, b in edges:
        ar, ac = a
        br, bc = b
        room_a = rooms[a]
        room_b = rooms[b]
        a_r0, a_c0, a_r1, a_c1 = room_a["bbox"]
        b_r0, b_c0, b_r1, b_c1 = room_b["bbox"]

        if ar == br:
            if ac > bc:
                a, b = b, a
                room_a, room_b = room_b, room_a
                a_r0, a_c0, a_r1, a_c1 = room_a["bbox"]
                b_r0, b_c0, b_r1, b_c1 = room_b["bbox"]
            row = row_axes[ar]
            start = a_c1 + 1
            stop = b_c0 - 1
            if start > stop:
                return None
            cells = [(row, c) for c in range(start, stop + 1)]
        else:
            if ar > br:
                a, b = b, a
                room_a, room_b = room_b, room_a
                a_r0, a_c0, a_r1, a_c1 = room_a["bbox"]
                b_r0, b_c0, b_r1, b_c1 = room_b["bbox"]
                ar, br = br, ar
                ac, bc = bc, ac
            col = col_axes[ac]
            start = a_r1 + 1
            stop = b_r0 - 1
            if start > stop:
                return None
            cells = [(r, col) for r in range(start, stop + 1)]

        if any(carved[r, c] for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = BG
            carved[r, c] = True

        corridors.append(
            {
                "cells": cells,
                "lo_color": room_a["color"],
                "hi_color": room_b["color"],
            }
        )

    for room in rooms.values():
        r0, c0, r1, c1 = room["bbox"]
        sr = random.randint(r0, r1)
        sc = random.randint(c0, c1)
        grid[sr, sc] = room["color"]

    expected = _render_expected(grid, rooms, corridors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-room instance")

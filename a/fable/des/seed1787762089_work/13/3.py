import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PATTERNS = (
    np.ones((3, 3), dtype=bool),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _pattern_index(mask: np.ndarray) -> int | None:
    for idx, pattern in enumerate(PATTERNS):
        if np.array_equal(mask, pattern):
            return idx
    return None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    degrees = {
        cell: sum((nr, nc) in coord_set for nr, nc in _neighbors4(*cell))
        for cell in coord_set
    }
    assert all(degree <= 2 for degree in degrees.values()), "wire must be an unbranched path"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "wire must have exactly two endpoints"

    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        nxts = [cell for cell in _neighbors4(*cur) if cell in coord_set and cell != prev]
        assert len(nxts) == 1, "wire path ordering must be unique"
        prev, cur = cur, nxts[0]
        path.append(cur)
    return path


def _count_turns(path: list[tuple[int, int]]) -> int:
    steps = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    return sum(steps[i] != steps[i - 1] for i in range(1, len(steps)))


def _infer_gray_and_tokens(grid: np.ndarray, bg: int):
    tokens = []
    non_tokens = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3:
            patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            if np.all((patch == color) | (patch == bg)):
                state = _pattern_index(patch == color)
                if state is not None:
                    tokens.append(
                        {
                            "color": color,
                            "state": state,
                            "top": int(r0),
                            "left": int(c0),
                            "cells": {(int(r), int(c)) for r, c in pts},
                        }
                    )
                    continue
        non_tokens.append(color)
    assert len(non_tokens) == 1, "expected a unique gray wire color"
    return non_tokens[0], tokens


def _find_slots(grid: np.ndarray, bg: int, gray: int):
    h, w = grid.shape
    slots = []
    for r in range(1, h - 3):
        for c in range(1, w - 3):
            if not np.all(grid[r : r + 3, c : c + 3] == bg):
                continue
            corners = [(r - 1, c - 1), (r - 1, c + 3), (r + 3, c - 1), (r + 3, c + 3)]
            if not all(int(grid[rr, cc]) == gray for rr, cc in corners):
                continue
            if not all(
                sum(
                    0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == gray
                    for nr, nc in _neighbors4(rr, cc)
                )
                == 0
                for rr, cc in corners
            ):
                continue
            ports = set()
            for rr in range(r, r + 3):
                ports.add((rr, c - 1))
                ports.add((rr, c + 3))
            for cc in range(c, c + 3):
                ports.add((r - 1, cc))
                ports.add((r + 3, cc))
            gray_ports = [
                (pr, pc)
                for pr, pc in ports
                if 0 <= pr < h and 0 <= pc < w and int(grid[pr, pc]) == gray
            ]
            if len(gray_ports) != 1:
                continue
            slots.append({"top": r, "left": c, "port": gray_ports[0], "ports": ports})
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    gray, tokens = _infer_gray_and_tokens(g, bg)
    slots = _find_slots(g, bg, gray)
    gray_components = [comp for comp in _components(g == gray) if len(comp) > 1]

    token_by_endpoint = {}
    for token in tokens:
        for cell in token["cells"]:
            for endpoint in _neighbors4(*cell):
                token_by_endpoint.setdefault(endpoint, []).append(token)

    slot_by_port = {}
    for slot in slots:
        slot_by_port.setdefault(slot["port"], []).append(slot)

    out = g.copy()
    used_tokens = set()
    used_slots = set()
    for comp in gray_components:
        path = _ordered_path(comp)
        endpoints = [path[0], path[-1]]
        endpoint_tokens = [token_by_endpoint.get(endpoint, []) for endpoint in endpoints]
        endpoint_slots = [slot_by_port.get(endpoint, []) for endpoint in endpoints]

        source_idx = next((i for i, matches in enumerate(endpoint_tokens) if matches), None)
        target_idx = next((i for i, matches in enumerate(endpoint_slots) if matches), None)
        assert source_idx is not None and target_idx is not None and source_idx != target_idx
        assert len(endpoint_tokens[source_idx]) == 1 and len(endpoint_slots[target_idx]) == 1

        token = endpoint_tokens[source_idx][0]
        slot = endpoint_slots[target_idx][0]
        used_tokens.add(token["color"])
        used_slots.add((slot["top"], slot["left"]))

        turns = _count_turns(path)
        state = (token["state"] + turns) % len(PATTERNS)
        patch = out[slot["top"] : slot["top"] + 3, slot["left"] : slot["left"] + 3]
        assert np.all(patch == bg), "slot interior must start empty"
        patch[PATTERNS[state]] = token["color"]

    assert len(used_tokens) == len(tokens), "every token must feed one wire"
    assert len(used_slots) == len(slots), "every slot must be used exactly once"
    return out


BG = 0
GRAY = 5
TOKEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ACTIVE_BAND_H = 6
BAND_GAP = 1
MAX_ATTEMPTS = 400


def _draw_pattern(grid: np.ndarray, top: int, left: int, color: int, state: int) -> None:
    patch = grid[top : top + 3, left : left + 3]
    assert np.all(patch == BG)
    patch[PATTERNS[state]] = color


def _draw_slot(grid: np.ndarray, box_top: int, box_left: int) -> tuple[int, int]:
    corners = [
        (box_top, box_left),
        (box_top, box_left + 4),
        (box_top + 4, box_left),
        (box_top + 4, box_left + 4),
    ]
    for r, c in corners:
        assert grid[r, c] == BG
        grid[r, c] = GRAY
    return box_top + 1, box_left + 1


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        cells.append((r, c))
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            cells.append((r, c))
    deduped = [cells[0]]
    for cell in cells[1:]:
        if cell != deduped[-1]:
            deduped.append(cell)
    return deduped


def _paint_wire(grid: np.ndarray, points: list[tuple[int, int]]) -> None:
    cells = _polyline_cells(points)
    assert len(cells) == len(set(cells)), "wire must not self-overlap"
    for r, c in cells:
        assert grid[r, c] == BG
        grid[r, c] = GRAY


def _token_top_choices(pattern_idx: int, band_top: int, start_row: int) -> list[int]:
    return [
        top
        for top in range(band_top, band_top + ACTIVE_BAND_H - 2)
        if 0 <= start_row - top < 3 and PATTERNS[pattern_idx][start_row - top, 2]
    ]


def _slot_box_top_choices(band_top: int, end_row: int) -> list[int]:
    return [
        box_top
        for box_top in (band_top, band_top + 1)
        if end_row == box_top + 2
    ]


def _route_rows(pattern_idx: int, turns: int, band_top: int):
    start_rows = [
        row
        for row in range(band_top, band_top + ACTIVE_BAND_H)
        if _token_top_choices(pattern_idx, band_top, row)
    ]
    end_rows = [
        row
        for row in range(band_top, band_top + ACTIVE_BAND_H)
        if _slot_box_top_choices(band_top, row)
    ]
    band_rows = list(range(band_top, band_top + ACTIVE_BAND_H))

    if turns == 0:
        common = [row for row in start_rows if row in set(end_rows)]
        assert common
        row = random.choice(common)
        return row, row, None

    if turns in (1, 2):
        pairs = [(sr, er) for sr in start_rows for er in end_rows if sr != er]
        assert pairs
        sr, er = random.choice(pairs)
        return sr, er, None

    triples = [
        (sr, er, mr)
        for sr in start_rows
        for er in end_rows
        for mr in band_rows
        if mr not in {sr, er}
    ]
    assert triples
    return random.choice(triples)


def _gray_layout_ok(grid: np.ndarray) -> bool:
    comps = _components(grid == GRAY)
    wires = [comp for comp in comps if len(comp) > 1]
    if len(wires) not in (3, 4):
        return False
    for comp in comps:
        if len(comp) == 1:
            continue
        coord_set = set(comp)
        degrees = {
            cell: sum((nr, nc) in coord_set for nr, nc in _neighbors4(*cell))
            for cell in coord_set
        }
        if any(degree > 2 for degree in degrees.values()):
            return False
        if sum(degree == 1 for degree in degrees.values()) != 2:
            return False
        if _count_turns(_ordered_path(comp)) > 3:
            return False
    return True


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    wire_count = random.choice([3, 4])
    top_margin = random.randint(0, 1)
    bottom_margin = random.randint(0, 1)
    height = top_margin + wire_count * ACTIVE_BAND_H + (wire_count - 1) * BAND_GAP + bottom_margin
    width = random.randint(24, 31)

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    colors = random.sample(TOKEN_COLORS, wire_count)
    turn_counts = random.sample([0, 1, 2, 3], wire_count)

    for index in range(wire_count):
        band_top = top_margin + index * (ACTIVE_BAND_H + BAND_GAP)
        turns = turn_counts[index]
        state = random.randrange(len(PATTERNS))
        color = colors[index]

        start_row, end_row, mid_row = _route_rows(state, turns, band_top)
        token_top = random.choice(_token_top_choices(state, band_top, start_row))
        slot_box_top = random.choice(_slot_box_top_choices(band_top, end_row))

        token_left = random.randint(1, 3)
        slot_box_left = width - 5 - random.randint(1, 2)
        start_col = token_left + 3
        end_col = slot_box_left
        assert end_col - start_col >= 4

        _draw_pattern(grid, token_top, token_left, color, state)
        _draw_pattern(expected, token_top, token_left, color, state)
        slot_top, slot_left = _draw_slot(grid, slot_box_top, slot_box_left)
        _draw_slot(expected, slot_box_top, slot_box_left)

        if turns == 0:
            points = [(start_row, start_col), (end_row, end_col)]
        elif turns == 1:
            points = [(start_row, start_col), (end_row, start_col), (end_row, end_col)]
        elif turns == 2:
            mid_col = random.randint(start_col + 1, end_col - 1)
            points = [
                (start_row, start_col),
                (start_row, mid_col),
                (end_row, mid_col),
                (end_row, end_col),
            ]
        else:
            mid_col = random.randint(start_col + 1, end_col - 1)
            points = [
                (start_row, start_col),
                (mid_row, start_col),
                (mid_row, mid_col),
                (end_row, mid_col),
                (end_row, end_col),
            ]

        _paint_wire(grid, points)
        _paint_wire(expected, points)

        target_state = (state + turns) % len(PATTERNS)
        expected[slot_top : slot_top + 3, slot_left : slot_left + 3][PATTERNS[target_state]] = color

    rotation = random.randrange(4)
    grid = np.rot90(grid, rotation)
    expected = np.rot90(expected, rotation)
    assert _gray_layout_ok(grid)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, expected = _build_scene()
        except AssertionError:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")

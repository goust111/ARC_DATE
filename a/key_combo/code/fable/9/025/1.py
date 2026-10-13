from __future__ import annotations

import math
import random

import numpy as np


BG = 0
GREY = 5
START = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]

BLOCK = 7
GAP = 1
BLOCK_STRIDE = BLOCK + GAP
ANCHOR_OFFSET = 3
MAX_ATTEMPTS = 400

DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
VECTORS = [(dr * k, dc * k) for dr, dc in DIRS for k in (1, 2, 3)]


def _is_ray(dr: int, dc: int) -> bool:
    return (dr != 0 or dc != 0) and (dr == 0 or dc == 0 or abs(dr) == abs(dc))


def _step_and_length(vector: tuple[int, int]) -> tuple[int, int, int]:
    dr, dc = map(int, vector)
    assert _is_ray(dr, dc)
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    length = max(abs(dr), abs(dc))
    return step_r, step_c, int(length)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _extract_program(grid: np.ndarray) -> list[int]:
    return [int(color) for color in grid[-1] if int(color) != BG]


def _parse_mapping(grid: np.ndarray) -> tuple[tuple[int, int], list[int], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 4

    program = _extract_program(g)
    assert program, "expected a non-empty bottom program row"
    assert all(color not in {BG, GREY, START} for color in program)

    start_coords = np.argwhere(g == START)
    assert start_coords.shape == (1, 2), "expected one unique maroon start marker"
    start = tuple(map(int, start_coords[0]))
    assert start[0] < h - 1, "start marker cannot lie on the program row"

    anchors = [tuple(map(int, rc)) for rc in np.argwhere(g[:-1] == GREY)]
    legend_pixels = []
    for r, c in np.argwhere(g[:-1] != BG):
        color = int(g[r, c])
        if color not in {GREY, START}:
            legend_pixels.append((int(r), int(c), color))

    assert anchors, "expected grey legend anchors"
    assert legend_pixels, "expected colored legend pixels"

    mapping: dict[int, tuple[int, int]] = {}
    used_anchors: set[tuple[int, int]] = set()

    for r, c, color in legend_pixels:
        assert color not in mapping, "each legend color must appear once above the program row"
        candidates = []
        for ar, ac in anchors:
            dr = int(r - ar)
            dc = int(c - ac)
            if _is_ray(dr, dc):
                candidates.append((max(abs(dr), abs(dc)), ar, ac, dr, dc))
        assert candidates, "each legend color must align with a grey anchor"
        best_len = min(length for length, _ar, _ac, _dr, _dc in candidates)
        best = [item for item in candidates if item[0] == best_len]
        assert len(best) == 1, "legend pairing must be uniquely resolved by the nearest aligned grey anchor"
        _dist, ar, ac, dr, dc = best[0]
        used_anchors.add((ar, ac))
        mapping[color] = (int(dr), int(dc))

    assert len(used_anchors) == len(anchors), "every grey anchor must belong to one legend pair"
    assert set(program).issubset(mapping), "every program color must have a legend"
    assert all(not _neighbors8(start, anchor) for anchor in anchors), "start must not touch any grey anchor"
    return start, program, mapping


def _trace_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    program: list[int],
    mapping: dict[int, tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    cur = tuple(map(int, start))
    cells: list[tuple[int, int, int]] = []
    for color in program:
        step_r, step_c, length = _step_and_length(mapping[int(color)])
        for _ in range(length):
            cur = (int(cur[0] + step_r), int(cur[1] + step_c))
            assert _in_bounds(shape, cur[0], cur[1]), "path walked out of bounds"
            cells.append((int(cur[0]), int(cur[1]), int(color)))
    return cells, cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the bottom-row color program by repeatedly applying the in-grid legend vectors from the maroon start.

    Input structure:
    - Black background. The bottom row is a program row: several colored pixels appear left-to-right, with the remaining cells black.
    - Above it sit several legend pairs. Each pair contains one grey anchor pixel and one colored pixel; for any legend color there is exactly one such colored pixel.
    - The vector from a grey anchor to its colored partner is horizontal, vertical, or a 45-degree diagonal, and there is one unique maroon start pixel away from the anchors.

    Rule:
    - Read the bottom-row colors from left to right.
    - Begin at the maroon start. For each program color, find that color's legend vector and walk it step by step, coloring each traversed cell with the current program color.
    - After each vector, the landing cell becomes the new current position. Erase every legend pair and the entire program row, but keep the maroon start marker.

    Key insight:
    - The colored legend pixel does not name a fixed action externally; together with its grey anchor it defines an in-grid displacement, and the bottom row merely gives the execution order for chaining those displacements.
    """
    g = np.asarray(grid, dtype=int)
    start, program, mapping = _parse_mapping(g)
    out = np.full_like(g, BG)
    out[start] = START
    cells, _end = _trace_path(g.shape, start, program, mapping)
    for r, c, color in cells:
        out[r, c] = int(color)
    out[start] = START
    return out


def _legend_slots(colors: list[int]) -> dict[int, tuple[int, int]]:
    ordered = sorted(int(color) for color in colors)
    cols = 2 if len(ordered) <= 4 else 3
    slots = {}
    for idx, color in enumerate(ordered):
        block_r = idx // cols
        block_c = idx % cols
        anchor_r = 1 + block_r * BLOCK_STRIDE + ANCHOR_OFFSET
        anchor_c = 1 + block_c * BLOCK_STRIDE + ANCHOR_OFFSET
        slots[int(color)] = (int(anchor_r), int(anchor_c))
    return slots


def _legend_band_size(color_count: int) -> tuple[int, int]:
    cols = 2 if int(color_count) <= 4 else 3
    rows = int(math.ceil(int(color_count) / cols))
    return rows * BLOCK_STRIDE, cols * BLOCK_STRIDE


def _sample_color_vectors() -> dict[int, tuple[int, int]] | None:
    count = random.randint(3, 5)
    colors = random.sample(PALETTE, count)
    vectors = random.sample(VECTORS, count)
    axial = sum(1 for dr, dc in vectors if dr == 0 or dc == 0)
    diagonal = sum(1 for dr, dc in vectors if dr != 0 and dc != 0)
    if axial == 0 or diagonal == 0:
        return None
    if max(max(abs(dr), abs(dc)) for dr, dc in vectors) < 2:
        return None
    return {int(color): tuple(map(int, vector)) for color, vector in zip(colors, vectors)}


def _sample_program(colors: list[int]) -> list[int]:
    length = random.randint(len(colors) + 1, min(8, len(colors) + 3))
    seq = colors[:]
    random.shuffle(seq)
    while len(seq) < length:
        choices = [color for color in colors if color != seq[-1]]
        seq.append(int(random.choice(choices)))
    return [int(color) for color in seq]


def _simulate_simple_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    program: list[int],
    mapping: dict[int, tuple[int, int]],
) -> list[tuple[int, int, int]] | None:
    try:
        cells, _end = _trace_path(shape, start, program, mapping)
    except AssertionError:
        return None

    visited = set()
    for r, c, _color in cells:
        if (int(r), int(c)) in visited or (int(r), int(c)) == tuple(start):
            return None
        visited.add((int(r), int(c)))

    directions = []
    lengths = []
    for color in program:
        vector = mapping[int(color)]
        step_r, step_c, length = _step_and_length(vector)
        directions.append((int(step_r), int(step_c)))
        lengths.append(int(length))
    if len(set(directions)) < 2:
        return None
    turns = sum(1 for i in range(len(directions) - 1) if directions[i] != directions[i + 1])
    if turns == 0:
        return None
    if max(lengths) < 2:
        return None

    rr = [int(start[0])] + [int(r) for r, _c, _color in cells]
    cc = [int(start[1])] + [int(c) for _r, c, _color in cells]
    if max(rr) == min(rr) or max(cc) == min(cc):
        return None
    return cells


def _build_input(
    shape: tuple[int, int],
    start: tuple[int, int],
    program: list[int],
    mapping: dict[int, tuple[int, int]],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)
    grid[start] = START

    for color, (anchor_r, anchor_c) in _legend_slots(list(mapping)).items():
        dr, dc = mapping[int(color)]
        grid[int(anchor_r), int(anchor_c)] = GREY
        grid[int(anchor_r + dr), int(anchor_c + dc)] = int(color)

    step = max(1, (w - len(program)) // (len(program) + 1))
    col = step
    for color in program:
        if col >= w:
            col = w - 1
        grid[h - 1, int(col)] = int(color)
        col += step
    return grid


def _validate_generated(grid: np.ndarray, path_cells: list[tuple[int, int, int]]) -> None:
    start, program, mapping = _parse_mapping(grid)
    assert len(mapping) >= 3
    assert set(program) == set(mapping), "every legend color should be exercised by the program"
    assert all(program[i] != program[i + 1] for i in range(len(program) - 1))

    output = main(grid)
    assert int(output[start]) == START
    assert np.all(output[-1] == BG), "program row must disappear"

    expected = np.full_like(grid, BG)
    expected[start] = START
    for r, c, color in path_cells:
        expected[int(r), int(c)] = int(color)
    expected[start] = START
    assert np.array_equal(output, expected)

    used_vectors = {tuple(map(int, vector)) for vector in mapping.values()}
    assert len(used_vectors) == len(mapping), "generator should keep legend vectors distinct"

    degrees: dict[tuple[int, int], int] = {}
    chain = [start] + [(int(r), int(c)) for r, c, _color in path_cells]
    for a, b in zip(chain, chain[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    assert degrees[start] == 1
    assert sum(1 for deg in degrees.values() if deg == 1) == 2
    assert all(deg == 2 for cell, deg in degrees.items() if cell not in {start, chain[-1]})


def _gen_attempt() -> np.ndarray | None:
    mapping = _sample_color_vectors()
    if mapping is None:
        return None

    colors = list(mapping)
    program = _sample_program(colors)
    if set(program) != set(colors):
        return None

    legend_h, legend_w = _legend_band_size(len(colors))
    min_h = legend_h + 6
    min_w = max(legend_w, len(program) * 2 + 1, 12)
    max_h = min(30, min_h + 10)
    max_w = min(30, min_w + 8)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    slots = _legend_slots(colors)
    for color, (anchor_r, anchor_c) in slots.items():
        dr, dc = mapping[int(color)]
        if not _in_bounds((h, w), anchor_r + dr, anchor_c + dc):
            return None

    top_limit = legend_h
    start_row_lo = top_limit + 1
    start_row_hi = h - 3
    start_col_lo = 1
    start_col_hi = w - 2
    if start_row_lo > start_row_hi or start_col_lo > start_col_hi:
        return None

    start = (random.randint(start_row_lo, start_row_hi), random.randint(start_col_lo, start_col_hi))
    for anchor in slots.values():
        if _neighbors8(start, anchor):
            return None

    path_cells = _simulate_simple_path((h, w), start, program, mapping)
    if path_cells is None:
        return None

    grid = _build_input((h, w), start, program, mapping)
    try:
        _validate_generated(grid, path_cells)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-polyline ARC input")

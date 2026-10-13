from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LINE = 5
BLOCK = 3
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

# --------------- solver ---------------

Coord = tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color not in colors:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _paint_square(grid: np.ndarray, top: Coord, color: int) -> None:
    r0, c0 = top
    grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = int(color)


def _square_cells(top: Coord) -> list[Coord]:
    r0, c0 = top
    return [(r, c) for r in range(r0, r0 + BLOCK) for c in range(c0, c0 + BLOCK)]


def _validate_square(grid: np.ndarray, comp: dict) -> dict:
    color = int(comp["color"])
    coords = comp["coords"]
    assert len(coords) == BLOCK * BLOCK, "each object must be a 3x3 square"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 + 1 == BLOCK and c1 - c0 + 1 == BLOCK, "object must have 3x3 bbox"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "square must be solid"
    return {"color": color, "top": (r0, c0), "coords": coords}


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _classify_line(coords: list[Coord]) -> str:
    if len(coords) == 1:
        return "straight"

    cells = set(coords)
    degrees: list[int] = []
    corners = 0
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}

    for r, c in cells:
        up = (r - 1, c) in cells
        down = (r + 1, c) in cells
        left = (r, c - 1) in cells
        right = (r, c + 1) in cells
        degree = int(up) + int(down) + int(left) + int(right)
        assert degree in (1, 2), "line must be a simple open path"
        degrees.append(degree)
        if (up or down) and (left or right):
            corners += 1

    assert degrees.count(1) == 2, "line must have two endpoints"
    if corners == 0:
        assert len(rows) == 1 or len(cols) == 1, "straight line must stay on one row or column"
        return "straight"
    assert corners == 1, "bent line must have exactly one corner"
    assert len(rows) > 1 and len(cols) > 1, "bent line must occupy both axes"
    return "elbow"


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, str]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    square_colors = {int(x) for x in np.unique(g) if int(x) not in (BG, LINE)}
    line_colors = {LINE} if np.any(g == LINE) else set()
    squares = [_validate_square(g, comp) for comp in _find_color_components(g, square_colors)]
    lines = _find_color_components(g, line_colors)

    assert squares and len(squares) % 2 == 0, "squares must come in pairs"
    assert len(lines) == len(squares) // 2, "each pair must have one connector"

    square_id = np.full(g.shape, -1, dtype=int)
    for idx, square in enumerate(squares):
        for r, c in square["coords"]:
            square_id[r, c] = idx

    pair_counts = [0] * len(squares)
    pairs: list[tuple[int, int, str]] = []
    for line in lines:
        touched: set[int] = set()
        for r, c in line["coords"]:
            for nr, nc in _neighbors4(r, c, g.shape[0], g.shape[1]):
                idx = int(square_id[nr, nc])
                if idx >= 0:
                    touched.add(idx)
        assert len(touched) == 2, "each connector must touch exactly two squares"
        a, b = sorted(touched)
        pair_counts[a] += 1
        pair_counts[b] += 1
        pairs.append((a, b, _classify_line(line["coords"])))

    assert all(count == 1 for count in pair_counts), "each square must belong to exactly one pair"
    return squares, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    According to the connector shape, either swap the paired squares' positions
    (straight line) or swap their colors in place (L-shaped line).
    """
    g = np.asarray(grid, dtype=int)
    squares, pairs = _parse_grid(g)

    out = g.copy()
    for square in squares:
        for r, c in square["coords"]:
            out[r, c] = BG

    # The blocks are congruent 3x3 solids, so both rule branches visibly
    # exchange the two colors between the paired locations. We still parse and
    # validate the connector shape to respect the intended mechanism.
    for a, b, kind in pairs:
        sq_a = squares[a]
        sq_b = squares[b]
        if kind == "straight":
            _paint_square(out, sq_b["top"], sq_a["color"])
            _paint_square(out, sq_a["top"], sq_b["color"])
        else:
            _paint_square(out, sq_a["top"], sq_b["color"])
            _paint_square(out, sq_b["top"], sq_a["color"])

    return out


# --------------- generator ---------------

PAIR_COUNT_CHOICES = [2, 3, 4]
STRAIGHT_GAP_MIN = 2
STRAIGHT_GAP_MAX = 8
ELBOW_OFFSET_MIN = 4
ELBOW_OFFSET_MAX = 10
PAIR_PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _pair_kinds(n_pairs: int) -> list[str]:
    n_straight = n_pairs // 2
    n_elbow = n_pairs // 2
    if n_pairs % 2 == 1:
        if random.random() < 0.5:
            n_straight += 1
        else:
            n_elbow += 1
    kinds = ["straight"] * n_straight + ["elbow"] * n_elbow
    random.shuffle(kinds)
    return kinds


def _sample_shape(n_pairs: int) -> tuple[int, int]:
    base = 12 + 4 * n_pairs
    return random.randint(base, base + 6), random.randint(base, base + 6)


def _candidate_cells(pair: dict) -> list[Coord]:
    return pair["a_cells"] + pair["b_cells"] + pair["line_cells"]


def _can_place(pair: dict, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cells = _candidate_cells(pair)
    if len(set(cells)) != len(cells):
        return False

    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _make_straight_pair(h: int, w: int, color_a: int, color_b: int) -> dict | None:
    horizontal = random.random() < 0.5
    if horizontal:
        max_gap = min(STRAIGHT_GAP_MAX, w - 2 * BLOCK)
        if max_gap < STRAIGHT_GAP_MIN:
            return None
        gap = random.randint(STRAIGHT_GAP_MIN, max_gap)
        total_w = 2 * BLOCK + gap
        r = random.randint(0, h - BLOCK)
        c = random.randint(0, w - total_w)
        a_top = (r, c)
        b_top = (r, c + BLOCK + gap)
        line_cells = [(r + 1, cc) for cc in range(c + BLOCK, c + BLOCK + gap)]
    else:
        max_gap = min(STRAIGHT_GAP_MAX, h - 2 * BLOCK)
        if max_gap < STRAIGHT_GAP_MIN:
            return None
        gap = random.randint(STRAIGHT_GAP_MIN, max_gap)
        total_h = 2 * BLOCK + gap
        r = random.randint(0, h - total_h)
        c = random.randint(0, w - BLOCK)
        a_top = (r, c)
        b_top = (r + BLOCK + gap, c)
        line_cells = [(rr, c + 1) for rr in range(r + BLOCK, r + BLOCK + gap)]

    return {
        "kind": "straight",
        "a_top": a_top,
        "b_top": b_top,
        "a_color": color_a,
        "b_color": color_b,
        "a_cells": _square_cells(a_top),
        "b_cells": _square_cells(b_top),
        "line_cells": line_cells,
    }


def _make_elbow_pair(h: int, w: int, color_a: int, color_b: int) -> dict | None:
    row_limit = min(ELBOW_OFFSET_MAX, h - BLOCK)
    col_limit = min(ELBOW_OFFSET_MAX, w - BLOCK)
    if row_limit < ELBOW_OFFSET_MIN or col_limit < ELBOW_OFFSET_MIN:
        return None

    row_offset = random.choice((-1, 1)) * random.randint(ELBOW_OFFSET_MIN, row_limit)
    col_offset = random.choice((-1, 1)) * random.randint(ELBOW_OFFSET_MIN, col_limit)
    a_r_min = max(0, -row_offset)
    a_r_max = h - BLOCK - max(0, row_offset)
    a_c_min = max(0, -col_offset)
    a_c_max = w - BLOCK - max(0, col_offset)
    if a_r_min > a_r_max or a_c_min > a_c_max:
        return None

    a_top = (random.randint(a_r_min, a_r_max), random.randint(a_c_min, a_c_max))
    b_top = (a_top[0] + row_offset, a_top[1] + col_offset)
    route = random.choice(("hv", "vh"))

    if route == "hv":
        row = a_top[0] + 1
        col = b_top[1] + 1
        start = (row, a_top[1] + BLOCK if col > a_top[1] + 1 else a_top[1] - 1)
        elbow = (row, col)
        end = (b_top[0] - 1, col) if row < b_top[0] + 1 else (b_top[0] + BLOCK, col)
    else:
        row = b_top[0] + 1
        col = a_top[1] + 1
        start = (a_top[0] + BLOCK if row > a_top[0] + 1 else a_top[0] - 1, col)
        elbow = (row, col)
        end = (row, b_top[1] - 1) if col < b_top[1] + 1 else (row, b_top[1] + BLOCK)

    line_cells = _segment_cells(start, elbow) + _segment_cells(elbow, end)[1:]
    return {
        "kind": "elbow",
        "a_top": a_top,
        "b_top": b_top,
        "a_color": color_a,
        "b_color": color_b,
        "a_cells": _square_cells(a_top),
        "b_cells": _square_cells(b_top),
        "line_cells": line_cells,
    }


def _paint_pair_input(grid: np.ndarray, pair: dict) -> None:
    _paint_square(grid, pair["a_top"], pair["a_color"])
    _paint_square(grid, pair["b_top"], pair["b_color"])
    for r, c in pair["line_cells"]:
        grid[r, c] = LINE


def _expected_output(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        for r, c in pair["a_cells"] + pair["b_cells"]:
            out[r, c] = BG

    for pair in pairs:
        if pair["kind"] == "straight":
            _paint_square(out, pair["b_top"], pair["a_color"])
            _paint_square(out, pair["a_top"], pair["b_color"])
        else:
            _paint_square(out, pair["a_top"], pair["b_color"])
            _paint_square(out, pair["b_top"], pair["a_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    h, w = _sample_shape(n_pairs)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    kinds = _pair_kinds(n_pairs)
    colors = random.sample(PALETTE, 2 * n_pairs)
    pairs: list[dict] = []

    for i, kind in enumerate(kinds):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        placed = False
        for _ in range(PAIR_PLACE_ATTEMPTS):
            if kind == "straight":
                pair = _make_straight_pair(h, w, color_a, color_b)
            else:
                pair = _make_elbow_pair(h, w, color_a, color_b)
            if pair is None or not _can_place(pair, occupied):
                continue
            _paint_pair_input(grid, pair)
            for r, c in _candidate_cells(pair):
                occupied[r, c] = True
            pairs.append(pair)
            placed = True
            break
        if not placed:
            return None

    oracle = _expected_output(grid, pairs)
    if not np.array_equal(main(grid), oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")

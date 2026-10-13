from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNER_MODES = ("tl", "tr", "bl", "br")

MIN_SIZE = 10
MAX_SIZE = 20
MIN_PAIRS = 2
MAX_PAIRS = 6
BOX_SPAN_MIN = 2
BOX_SPAN_MAX = 8
BOX_GAP = 1
GEN_MAX_ATTEMPTS = 400
PAIR_PLACE_ATTEMPTS = 200

Coord = tuple[int, int]
Box = tuple[int, int, int, int]


def _corner_mode(grid: np.ndarray) -> str:
    h, w = grid.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    grey_modes = [mode for mode, (r, c) in corners.items() if int(grid[r, c]) == GREY]
    assert len(grey_modes) == 1, "expected exactly one grey corner marker"
    for mode, (r, c) in corners.items():
        if mode != grey_modes[0]:
            assert int(grid[r, c]) == BLACK, "the other three corners must stay black"
    assert int(np.count_nonzero(grid == GREY)) == 1, "grey should appear only as the corner marker"
    return grey_modes[0]


def _turn_from_box(box: Box, mode: str) -> Coord:
    r0, c0, r1, c1 = box
    if mode == "tl":
        return r0, c0
    if mode == "tr":
        return r0, c1
    if mode == "bl":
        return r1, c0
    if mode == "br":
        return r1, c1
    raise AssertionError(f"unknown corner mode: {mode}")


def _endpoints_from_box(box: Box, mode: str) -> tuple[Coord, Coord]:
    r0, c0, r1, c1 = box
    if mode in ("tl", "br"):
        return (r0, c1), (r1, c0)
    return (r0, c0), (r1, c1)


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("L-path segments must be axis-aligned")


def _path_cells(box: Box, mode: str) -> set[Coord]:
    a, b = _endpoints_from_box(box, mode)
    turn = _turn_from_box(box, mode)
    return set(_segment_cells(a, turn)) | set(_segment_cells(turn, b))


def _ordered_pairs(grid: np.ndarray) -> list[tuple[int, tuple[Coord, Coord]]]:
    seen: set[int] = set()
    pairs: list[tuple[int, tuple[Coord, Coord]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or color in seen:
                continue
            seen.add(color)
            coords = np.argwhere(grid == color)
            assert coords.shape == (2, 2), "each endpoint color must appear exactly twice"
            a = tuple(map(int, coords[0]))
            b = tuple(map(int, coords[1]))
            assert a[0] != b[0] and a[1] != b[1], "paired endpoints must be diagonal"
            pairs.append((color, (a, b)))
    assert pairs, "expected at least one endpoint pair"
    return pairs


def _paint_segment(out: np.ndarray, inp: np.ndarray, start: Coord, end: Coord, color: int) -> None:
    for r, c in _segment_cells(start, end):
        if int(inp[r, c]) in (BLACK, color):
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Connect every same-color endpoint pair with an L-shaped path chosen by the grey corner marker.

    Input structure:
    - The background is black.
    - One grey pixel appears in exactly one corner, while the other three corners stay black.
    - Every other non-black color appears exactly twice as two singleton endpoints, and the two endpoints of a pair are in different rows and different columns.

    Rule:
    - Read which corner contains the grey marker; that corner chooses the shared elbow corner of every pair's axis-aligned bounding rectangle.
    - For each same-color pair, draw the full horizontal and vertical rectangle edges that meet at that chosen elbow, including both endpoints and the elbow cell.
    - If a path passes through another pair's original endpoint, leave that endpoint's original color in place. Keep the grey corner marker unchanged.

    Key insight:
    - The same two endpoints admit two possible L-shaped connections, and the unique grey corner resolves that ambiguity globally by telling you which rectangle corner to use as the turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mode = _corner_mode(g)
    out = g.copy()
    for color, (a, b) in _ordered_pairs(g):
        r0, r1 = sorted((a[0], b[0]))
        c0, c1 = sorted((a[1], b[1]))
        box = (r0, c0, r1, c1)
        turn = _turn_from_box(box, mode)
        _paint_segment(out, g, a, turn, color)
        _paint_segment(out, g, turn, b, color)

    return out


def _boxes_too_close(a: Box, b: Box, pad: int = BOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _max_pairs_for_size(height: int, width: int) -> int:
    area = height * width
    if area < 130:
        return 3
    if area < 220:
        return 4
    return MAX_PAIRS


def _sample_box(height: int, width: int) -> Box:
    max_h = min(BOX_SPAN_MAX, height - 2)
    max_w = min(BOX_SPAN_MAX, width - 2)
    box_h = random.randint(BOX_SPAN_MIN, max_h)
    box_w = random.randint(BOX_SPAN_MIN, max_w)
    r0 = random.randint(1, height - box_h - 1)
    c0 = random.randint(1, width - box_w - 1)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    return r0, c0, r1, c1


def _layout_ok(boxes: list[Box], candidate: Box) -> bool:
    return all(not _boxes_too_close(candidate, box) for box in boxes)


def _render_input(height: int, width: int, mode: str, boxes: list[Box], colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    corner_cells = {
        "tl": (0, 0),
        "tr": (0, width - 1),
        "bl": (height - 1, 0),
        "br": (height - 1, width - 1),
    }
    mr, mc = corner_cells[mode]
    grid[mr, mc] = GREY
    for color, box in zip(colors, boxes):
        for r, c in _endpoints_from_box(box, mode):
            assert int(grid[r, c]) == BLACK
            grid[r, c] = color
    return grid


def _expected_output(inp: np.ndarray, mode: str, boxes: list[Box], colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for color, box in zip(colors, boxes):
        for r, c in _path_cells(box, mode):
            if int(inp[r, c]) in (BLACK, color):
                out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    mode = random.choice(CORNER_MODES)

    pair_cap = min(len(PAIR_COLORS), _max_pairs_for_size(height, width))
    pair_count = random.randint(MIN_PAIRS, pair_cap)
    colors = random.sample(PAIR_COLORS, pair_count)

    boxes: list[Box] = []
    for _ in range(pair_count):
        for _ in range(PAIR_PLACE_ATTEMPTS):
            candidate = _sample_box(height, width)
            if _layout_ok(boxes, candidate):
                boxes.append(candidate)
                break
        else:
            return None

    inp = _render_input(height, width, mode, boxes, colors)
    expected = _expected_output(inp, mode, boxes, colors)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

import random
from collections import Counter

import numpy as np


BLACK = 0
WALL = 8
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_SIDE_RANGE = (22, 30)
PAIR_COUNT_CHOICES = (2, 3, 4)
PAIR_COUNT_WEIGHTS = (3, 2, 1)
DIAG_LEN_RANGE = (2, 5)
AXIS_LEN_RANGE = (1, 4)
PAIR_PADDING = 1
OUTER_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PAIR_PLACEMENT_ATTEMPTS = 120


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    steps = max(abs(dr), abs(dc))
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "segments must be axis-aligned or 45-degree diagonal"
    if steps == 0:
        return [(int(a[0]), int(a[1]))]
    sr = _sign(dr)
    sc = _sign(dc)
    return [(int(a[0]) + sr * step, int(a[1]) + sc * step) for step in range(steps + 1)]


def _chain_cells(*points: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx in range(len(points) - 1):
        segment = _segment_cells(points[idx], points[idx + 1])
        if idx:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _decompose_offset(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    abs_dr = abs(int(dr))
    abs_dc = abs(int(dc))
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc, "marker offset must have unequal nonzero row/col components"

    diag_len = min(abs_dr, abs_dc)
    axis_len = abs(abs_dr - abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)
    diag_vec = (sr * diag_len, sc * diag_len)
    if abs_dr > abs_dc:
        axis_vec = (sr * axis_len, 0)
    else:
        axis_vec = (0, sc * axis_len)
    return axis_vec, diag_vec


def _pair_routes(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    axis_vec, diag_vec = _decompose_offset(b[0] - a[0], b[1] - a[1])
    axis_corner = (a[0] + axis_vec[0], a[1] + axis_vec[1])
    diag_corner = (a[0] + diag_vec[0], a[1] + diag_vec[1])
    return _chain_cells(a, axis_corner, b), _chain_cells(a, diag_corner, b)


def _route_is_clear(grid: np.ndarray, route: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BLACK for r, c in route[1:-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair along the unique clear side route of its implied parallelogram.

    Input structure:
    - The background is black and teal cells are wall segments.
    - Every other foreground color appears exactly twice as singleton marker pixels, one pair per color.
    - Within each pair, the marker offset has nonzero row and column components with unequal magnitudes.

    Rule:
    - For each pair, decompose the marker-to-marker offset into one 45-degree diagonal vector and one remaining horizontal or vertical vector.
    - These two vectors define a parallelogram with the markers as opposite corners, giving exactly two one-turn side routes between the markers.
    - Check the intermediate cells of both routes; the only valid route is the one whose intermediate cells are all black. Paint that route's intermediate cells in the pair's own color, while preserving the markers and teal walls.

    Key insight:
    - The puzzle does not ask for a shortest path search: each pair has only two geometric candidates, and the wall segments act as switches that invalidate exactly one side route.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _bg_color(g) == BLACK, "expected black background"

    colors, counts = np.unique(g, return_counts=True)
    marker_colors = [int(color) for color, count in zip(colors, counts) if int(color) not in (BLACK, WALL) and int(count) == 2]
    assert marker_colors, "expected at least one marker pair"
    assert sum(int(color not in (BLACK, WALL)) for color in colors) == len(marker_colors), "every non-wall foreground color must appear exactly twice"

    out = g.copy()
    painted: dict[tuple[int, int], int] = {}
    for color in marker_colors:
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert len(coords) == 2
        route_via_axis, route_via_diag = _pair_routes(coords[0], coords[1])
        clear_axis = _route_is_clear(g, route_via_axis)
        clear_diag = _route_is_clear(g, route_via_diag)
        assert clear_axis != clear_diag, "each pair must have exactly one clear side route"
        chosen = route_via_axis if clear_axis else route_via_diag
        for cell in chosen[1:-1]:
            prior = painted.get(cell)
            assert prior is None or prior == color, "different routes must not conflict on the same cell"
            painted[cell] = color

    for (r, c), color in painted.items():
        out[r, c] = color
    return out


def _balanced_choices(count: int, first, second) -> list:
    if count == 1:
        return [random.choice((first, second))]
    choices = [first, second]
    for _ in range(count - 2):
        choices.append(random.choice((first, second)))
    random.shuffle(choices)
    return choices


def _translate(cells: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(ar + r, ac + c) for r, c in cells]


def _spec_bbox(spec: dict) -> tuple[int, int, int, int]:
    rows = [r for r, _ in spec["footprint"]]
    cols = [c for _, c in spec["footprint"]]
    return min(rows), min(cols), max(rows), max(cols)


def _mark_padded_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    pr0 = max(0, r0 - PAIR_PADDING)
    pc0 = max(0, c0 - PAIR_PADDING)
    pr1 = min(occupied.shape[0] - 1, r1 + PAIR_PADDING)
    pc1 = min(occupied.shape[1] - 1, c1 + PAIR_PADDING)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _anchor_candidates(shape: tuple[int, int], occupied: np.ndarray, spec: dict) -> list[tuple[int, int]]:
    height, width = shape
    min_r, min_c, max_r, max_c = _spec_bbox(spec)
    row_start = OUTER_MARGIN - min_r
    row_stop = height - OUTER_MARGIN - max_r
    col_start = OUTER_MARGIN - min_c
    col_stop = width - OUTER_MARGIN - max_c
    if row_start >= row_stop or col_start >= col_stop:
        return []

    anchors: list[tuple[int, int]] = []
    for ar in range(row_start, row_stop):
        for ac in range(col_start, col_stop):
            bbox = (ar + min_r, ac + min_c, ar + max_r, ac + max_c)
            r0, c0, r1, c1 = bbox
            pr0 = max(0, r0 - PAIR_PADDING)
            pc0 = max(0, c0 - PAIR_PADDING)
            pr1 = min(height - 1, r1 + PAIR_PADDING)
            pc1 = min(width - 1, c1 + PAIR_PADDING)
            if not occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                anchors.append((ar, ac))
    return anchors


def _sample_wall_segment(route_kind: str, axis_corner: tuple[int, int], diag_corner: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    if route_kind == "axis":
        legs = [
            _segment_cells((0, 0), axis_corner)[1:],
            _segment_cells(axis_corner, goal)[1:-1],
        ]
    else:
        legs = [
            _segment_cells((0, 0), diag_corner)[1:],
            _segment_cells(diag_corner, goal)[1:-1],
        ]

    nonempty_legs = [leg for leg in legs if leg]
    assert nonempty_legs, "blocked route must provide at least one wall location"
    leg = random.choice(nonempty_legs)
    max_len = min(3, len(leg))
    wall_len = random.randint(1, max_len)
    start = random.randint(0, len(leg) - wall_len)
    return leg[start : start + wall_len]


def _pair_spec(clear_axis_first: bool, dominant_rows: bool) -> dict:
    diag_len = random.randint(*DIAG_LEN_RANGE)
    axis_len = random.randint(*AXIS_LEN_RANGE)
    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))
    if dominant_rows:
        dr = sr * (diag_len + axis_len)
        dc = sc * diag_len
    else:
        dr = sr * diag_len
        dc = sc * (diag_len + axis_len)

    goal = (dr, dc)
    axis_vec, diag_vec = _decompose_offset(dr, dc)
    axis_corner = axis_vec
    diag_corner = diag_vec

    route_via_axis = _chain_cells((0, 0), axis_corner, goal)
    route_via_diag = _chain_cells((0, 0), diag_corner, goal)
    clear_route = route_via_axis if clear_axis_first else route_via_diag
    blocked_kind = "diag" if clear_axis_first else "axis"
    wall_cells = _sample_wall_segment(blocked_kind, axis_corner, diag_corner, goal)

    footprint = list({*route_via_axis, *route_via_diag})
    return {
        "goal": goal,
        "clear_route": clear_route,
        "wall_cells": wall_cells,
        "footprint": footprint,
    }


def _place_pair(inp: np.ndarray, expected: np.ndarray, occupied: np.ndarray, color: int, spec: dict) -> bool:
    anchors = _anchor_candidates(inp.shape, occupied, spec)
    if not anchors:
        return False
    anchor = random.choice(anchors)
    goal = (anchor[0] + spec["goal"][0], anchor[1] + spec["goal"][1])
    markers = [anchor, goal]
    wall_cells = _translate(spec["wall_cells"], anchor)
    clear_middle = _translate(spec["clear_route"][1:-1], anchor)

    for r, c in markers:
        inp[r, c] = color
        expected[r, c] = color
    for r, c in wall_cells:
        inp[r, c] = WALL
        expected[r, c] = WALL
    for r, c in clear_middle:
        expected[r, c] = color

    min_r, min_c, max_r, max_c = _spec_bbox(spec)
    bbox = (anchor[0] + min_r, anchor[1] + min_c, anchor[0] + max_r, anchor[1] + max_c)
    _mark_padded_bbox(occupied, bbox)
    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(MARKER_COLORS, pair_count)
    dominant_rows_flags = _balanced_choices(pair_count, True, False)
    clear_axis_flags = _balanced_choices(pair_count, True, False)

    inp = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    tasks = list(zip(colors, dominant_rows_flags, clear_axis_flags))
    random.shuffle(tasks)
    for color, dominant_rows, clear_axis_first in tasks:
        placed = False
        for _ in range(MAX_PAIR_PLACEMENT_ATTEMPTS):
            spec = _pair_spec(clear_axis_first=clear_axis_first, dominant_rows=dominant_rows)
            if _place_pair(inp, expected, occupied, color, spec):
                placed = True
                break
        if not placed:
            return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected = built
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")

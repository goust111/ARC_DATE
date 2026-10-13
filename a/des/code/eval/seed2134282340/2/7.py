from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

PAIR_COUNT_RANGE = (3, 6)
GRID_H_MAX = 24
GRID_W_MAX = 24
GRID_MARGIN = 1
MIN_STRAIGHT_SPAN = 2
MIN_DIAG_ROW_SPAN = 2
MIN_DIAG_COL_SPAN = 2
ENDPOINT_PAD = 1

PAIR_PLACEMENT_RETRIES = 300
GEN_RETRIES = 400


def _corners(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _inclusive_range(a: int, b: int):
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _point_buffer(point: tuple[int, int], h: int, w: int, pad: int = ENDPOINT_PAD) -> set[tuple[int, int]]:
    r, c = point
    cells: set[tuple[int, int]] = set()
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
    return cells


def _horizontal_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    assert a[0] == b[0]
    r = a[0]
    c0, c1 = sorted((a[1], b[1]))
    return [(r, c) for c in range(c0, c1 + 1)]


def _vertical_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    assert a[1] == b[1]
    c = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, c) for r in range(r0, r1 + 1)]


def _nearest_corner_and_turn(
    a: tuple[int, int],
    b: tuple[int, int],
    h: int,
    w: int,
) -> tuple[tuple[int, int], str] | None:
    mid2_r = a[0] + b[0]
    mid2_c = a[1] + b[1]

    ranked: list[tuple[int, tuple[int, int]]] = []
    for corner in _corners(h, w):
        cr2 = 2 * corner[0]
        cc2 = 2 * corner[1]
        dist2 = (cr2 - mid2_r) ** 2 + (cc2 - mid2_c) ** 2
        ranked.append((dist2, corner))
    ranked.sort(key=lambda item: (item[0], item[1][0], item[1][1]))

    if len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:
        return None

    corner = ranked[0][1]
    dx2 = abs(2 * corner[1] - mid2_c)
    dy2 = abs(2 * corner[0] - mid2_r)
    if dx2 == dy2:
        return None

    # If the corner is horizontally closer to the midpoint, go vertical first.
    turn = "vh" if dx2 < dy2 else "hv"
    return corner, turn


def _diagonal_path(
    a: tuple[int, int],
    b: tuple[int, int],
    turn: str,
) -> list[tuple[int, int]]:
    left, right = sorted((a, b), key=lambda p: (p[1], p[0]))
    if turn == "hv":
        cells = [(left[0], c) for c in _inclusive_range(left[1], right[1])]
        cells.extend((r, right[1]) for r in _inclusive_range(left[0], right[0]) if r != left[0])
        return cells
    assert turn == "vh"
    cells = [(r, left[1]) for r in _inclusive_range(left[0], right[0])]
    cells.extend((right[0], c) for c in _inclusive_range(left[1], right[1]) if c != left[1])
    return cells


def _pair_record(
    color: int,
    a: tuple[int, int],
    b: tuple[int, int],
    h: int,
    w: int,
) -> dict:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    distance2 = dr * dr + dc * dc

    if a[0] == b[0]:
        path = _horizontal_path(a, b)
        kind = "h"
        turn = None
    elif a[1] == b[1]:
        path = _vertical_path(a, b)
        kind = "v"
        turn = None
    else:
        nearest = _nearest_corner_and_turn(a, b, h, w)
        assert nearest is not None, "diagonal pair must have a unique nearest corner and a strict axis preference"
        _, turn = nearest
        path = _diagonal_path(a, b, turn)
        kind = "d"

    r0, r1 = sorted((a[0], b[0]))
    c0, c1 = sorted((a[1], b[1]))
    return {
        "color": int(color),
        "points": tuple(sorted((a, b))),
        "kind": kind,
        "turn": turn,
        "distance2": int(distance2),
        "bbox": (r0, c0, r1, c1),
        "path": path,
    }


def _pair_sort_key(pair: dict):
    r0, c0, r1, c1 = pair["bbox"]
    return (-pair["distance2"], r0, c0, r1, c1)


def _extract_pairs(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must be at least 3x3"

    corner_cells = _corners(h, w)
    marker_color = int(g[corner_cells[0]])
    assert marker_color != BLACK, "corner markers must be non-black"
    for cell in corner_cells[1:]:
        assert int(g[cell]) == marker_color, "all four corner markers must share one color"

    marker_mask = np.zeros_like(g, dtype=bool)
    for r, c in corner_cells:
        marker_mask[r, c] = True
    assert int(np.count_nonzero(g == marker_color)) == 4, "marker color must appear only at the four corners"

    color_to_points: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere((g != BLACK) & (~marker_mask)):
        color = int(g[r, c])
        assert color != marker_color, "non-corner points cannot use the marker color"
        color_to_points.setdefault(color, []).append((int(r), int(c)))

    assert 3 <= len(color_to_points) <= 6, "expected 3 to 6 color pairs"

    pairs: list[dict] = []
    for color, points in color_to_points.items():
        assert len(points) == 2, "each non-marker color must appear exactly twice"
        pairs.append(_pair_record(color, points[0], points[1], h, w))

    return marker_color, pairs


def _render_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in sorted(pairs, key=_pair_sort_key):
        for r, c in pair["path"]:
            out[r, c] = pair["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color point pair with a shape determined by its relative orientation.

    Horizontal pairs become horizontal segments, vertical pairs become vertical
    segments, and diagonal pairs become L-shapes. For diagonal pairs, the
    nearest corner marker to the pair midpoint decides whether the L goes
    horizontal-then-vertical or vertical-then-horizontal.
    """
    g = np.asarray(grid, dtype=int)
    _, pairs = _extract_pairs(g)
    return _render_output(g, pairs)


def _valid_candidate(
    pair: dict,
    h: int,
    w: int,
    occupied_path: set[tuple[int, int]],
    blocked_endpoints: set[tuple[int, int]],
    used_distances: set[int],
) -> bool:
    a, b = pair["points"]
    if pair["distance2"] in used_distances:
        return False
    if a in blocked_endpoints or b in blocked_endpoints:
        return False

    path = pair["path"]
    if len(path) != len(set(path)):
        return False
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in path):
        return False
    if any(cell in occupied_path for cell in path):
        return False

    return True


def _sample_horizontal_pair(
    rng: random.Random,
    color: int,
    h: int,
    w: int,
    occupied_path: set[tuple[int, int]],
    blocked_endpoints: set[tuple[int, int]],
    used_distances: set[int],
) -> dict | None:
    for _ in range(PAIR_PLACEMENT_RETRIES):
        r = rng.randint(GRID_MARGIN, h - GRID_MARGIN - 1)
        c0 = rng.randint(GRID_MARGIN, w - GRID_MARGIN - 1 - MIN_STRAIGHT_SPAN)
        c1 = rng.randint(c0 + MIN_STRAIGHT_SPAN, w - GRID_MARGIN - 1)
        pair = _pair_record(color, (r, c0), (r, c1), h, w)
        if _valid_candidate(pair, h, w, occupied_path, blocked_endpoints, used_distances):
            return pair
    return None


def _sample_vertical_pair(
    rng: random.Random,
    color: int,
    h: int,
    w: int,
    occupied_path: set[tuple[int, int]],
    blocked_endpoints: set[tuple[int, int]],
    used_distances: set[int],
) -> dict | None:
    for _ in range(PAIR_PLACEMENT_RETRIES):
        c = rng.randint(GRID_MARGIN, w - GRID_MARGIN - 1)
        r0 = rng.randint(GRID_MARGIN, h - GRID_MARGIN - 1 - MIN_STRAIGHT_SPAN)
        r1 = rng.randint(r0 + MIN_STRAIGHT_SPAN, h - GRID_MARGIN - 1)
        pair = _pair_record(color, (r0, c), (r1, c), h, w)
        if _valid_candidate(pair, h, w, occupied_path, blocked_endpoints, used_distances):
            return pair
    return None


def _sample_diagonal_pair(
    rng: random.Random,
    color: int,
    desired_turn: str,
    h: int,
    w: int,
    occupied_path: set[tuple[int, int]],
    blocked_endpoints: set[tuple[int, int]],
    used_distances: set[int],
) -> dict | None:
    for _ in range(PAIR_PLACEMENT_RETRIES):
        a = (
            rng.randint(GRID_MARGIN, h - GRID_MARGIN - 1),
            rng.randint(GRID_MARGIN, w - GRID_MARGIN - 1),
        )
        b = (
            rng.randint(GRID_MARGIN, h - GRID_MARGIN - 1),
            rng.randint(GRID_MARGIN, w - GRID_MARGIN - 1),
        )
        if a == b or a[0] == b[0] or a[1] == b[1]:
            continue
        if abs(a[0] - b[0]) < MIN_DIAG_ROW_SPAN or abs(a[1] - b[1]) < MIN_DIAG_COL_SPAN:
            continue

        nearest = _nearest_corner_and_turn(a, b, h, w)
        if nearest is None or nearest[1] != desired_turn:
            continue

        pair = _pair_record(color, a, b, h, w)
        if _valid_candidate(pair, h, w, occupied_path, blocked_endpoints, used_distances):
            return pair
    return None


def _sample_pair(
    rng: random.Random,
    color: int,
    branch: str,
    h: int,
    w: int,
    occupied_path: set[tuple[int, int]],
    blocked_endpoints: set[tuple[int, int]],
    used_distances: set[int],
) -> dict | None:
    if branch == "h":
        return _sample_horizontal_pair(rng, color, h, w, occupied_path, blocked_endpoints, used_distances)
    if branch == "v":
        return _sample_vertical_pair(rng, color, h, w, occupied_path, blocked_endpoints, used_distances)
    if branch == "d_hv":
        return _sample_diagonal_pair(rng, color, "hv", h, w, occupied_path, blocked_endpoints, used_distances)
    assert branch == "d_vh"
    return _sample_diagonal_pair(rng, color, "vh", h, w, occupied_path, blocked_endpoints, used_distances)


def _make_branch_plan(rng: random.Random, pair_count: int) -> list[str]:
    if pair_count == 3:
        branches = ["h", "v", rng.choice(["d_hv", "d_vh"])]
    else:
        branches = ["h", "v", "d_hv", "d_vh"]
        extras = pair_count - 4
        for _ in range(extras):
            branches.append(rng.choice(["h", "v", "d_hv", "d_vh"]))
    rng.shuffle(branches)
    return branches


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    return _render_output(grid, pairs)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    min_h = 8 + 2 * pair_count
    min_w = 8 + 2 * pair_count
    if min_h > GRID_H_MAX or min_w > GRID_W_MAX:
        return None

    h = rng.randint(min_h, GRID_H_MAX)
    w = rng.randint(min_w, GRID_W_MAX)
    branches = _make_branch_plan(rng, pair_count)
    colors = rng.sample(PALETTE_COLORS, pair_count)

    occupied_path: set[tuple[int, int]] = set()
    blocked_endpoints: set[tuple[int, int]] = set()
    used_distances: set[int] = set()
    pairs: list[dict] = []

    # Diagonal placements are more constrained, so place them first.
    branch_order = sorted(branches, key=lambda branch: (0 if branch.startswith("d_") else 1, branch))
    # When branch labels repeat, assign colors by walking the original plan.
    available_colors: dict[str, list[int]] = {}
    for branch, color in zip(branches, colors):
        available_colors.setdefault(branch, []).append(color)

    for branch in branch_order:
        color = available_colors[branch].pop()
        pair = _sample_pair(rng, color, branch, h, w, occupied_path, blocked_endpoints, used_distances)
        if pair is None:
            return None
        pairs.append(pair)
        occupied_path.update(pair["path"])
        used_distances.add(pair["distance2"])
        for point in pair["points"]:
            blocked_endpoints.update(_point_buffer(point, h, w))

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in _corners(h, w):
        grid[r, c] = GREY
    for pair in pairs:
        for r, c in pair["points"]:
            grid[r, c] = pair["color"]

    try:
        marker_color, parsed = _extract_pairs(grid)
    except AssertionError:
        return None

    if marker_color != GREY or len(parsed) != pair_count:
        return None

    expected = _expected_output(grid, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    kinds = [pair["kind"] for pair in parsed]
    if "h" not in kinds or "v" not in kinds or "d" not in kinds:
        return None

    turns = [pair["turn"] for pair in parsed if pair["kind"] == "d"]
    if pair_count >= 4 and {"hv", "vh"} - set(turns):
        return None

    if len({pair["distance2"] for pair in parsed}) != pair_count:
        return None

    if int(np.count_nonzero(grid == GREY)) != 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

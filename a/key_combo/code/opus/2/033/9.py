from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

Coord = tuple[int, int]

GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _adjacent_to_color(g: np.ndarray, pos: Coord, color: int) -> bool:
    r, c = pos
    return any(
        _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == int(color)
        for nr, nc in _neighbors4(r, c)
    )


def _marker_pos(g: np.ndarray) -> Coord:
    pts = np.argwhere(g == RED)
    assert pts.shape == (1, 2), "expected exactly one red marker"
    return tuple(map(int, pts[0]))


def _sightline_to_obstacle(g: np.ndarray, start: Coord) -> tuple[Coord, Coord]:
    hits: list[tuple[Coord, Coord]] = []
    sr, sc = start

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = sr + dr, sc + dc
        last_bg: Coord | None = None
        while _in_bounds(g.shape, r, c):
            value = int(g[r, c])
            if value == GREEN:
                assert last_bg is not None, "marker must not touch the obstacle directly"
                hits.append(((dr, dc), last_bg))
                break
            assert value == BG, "the sightline to the obstacle must cross only black background"
            last_bg = (r, c)
            r += dr
            c += dc

    assert len(hits) == 1, "expected one unique cardinal sightline to the green obstacle"
    return hits[0]


def _candidate_turns(slide_dir: Coord) -> list[Coord]:
    dr, dc = slide_dir
    if dr != 0:
        return [(dr, -1), (dr, 1)]  # left first for tie-breaks
    return [(-1, dc), (1, dc)]  # up first for tie-breaks


def _walk_branch(g: np.ndarray, start: Coord, turn: Coord) -> list[Coord]:
    h, w = g.shape
    r, c = start
    path: list[Coord] = []

    while True:
        nr, nc = r + turn[0], c + turn[1]
        if not _in_bounds(g.shape, nr, nc):
            break
        if nr in (0, h - 1) or nc in (0, w - 1):
            break

        value = int(g[nr, nc])
        if value == BLUE:
            break
        if value != BG:
            break
        if not _adjacent_to_color(g, (nr, nc), GREEN):
            break

        path.append((nr, nc))
        r, c = nr, nc

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the red marker to the endpoint of the longer obstacle-adjacent contour walk after blue contour barriers truncate the candidate branches.

    Input structure:
    - The background is black.
    - One irregular 4-connected green obstacle region sits away from the border.
    - One red marker pixel has exactly one horizontal or vertical black sightline to the green region.
    - One or two blue pixels lie on black cells that are 4-adjacent to the green obstacle's outside contour.

    Rule:
    - Slide the red marker along its unique sightline to the last black cell before the green obstacle.
    - From that landing cell, follow the two diagonal contour directions that keep each walked cell black and 4-adjacent to green.
    - Stop a branch before stepping onto a blue pixel, before leaving green adjacency, or before stepping onto the grid edge.
    - Compare the two valid branch lengths; choose the longer branch, and if they tie choose the branch whose first step is upward or leftward.
    - Erase the original marker and repaint the chosen branch endpoint red, keeping the green obstacle and blue barrier pixels unchanged.

    Key insight:
    - The destination is not set by distance alone: the marker first slides to the obstacle, then blue contour blockers can shorten one contour branch enough to change which endpoint is optimal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= int(np.sum(g == BLUE)) <= 2, "expected one or two blue contour barriers"
    assert np.any(g == GREEN), "expected a green obstacle"

    marker = _marker_pos(g)
    slide_dir, landing = _sightline_to_obstacle(g, marker)
    assert _adjacent_to_color(g, landing, GREEN), "landing point must touch the obstacle"

    turns = _candidate_turns(slide_dir)
    branches = [_walk_branch(g, landing, turn) for turn in turns]
    best = 0
    if len(branches[1]) > len(branches[0]):
        best = 1

    dest = branches[best][-1] if branches[best] else landing
    out = g.copy()
    out[marker] = BG
    out[dest] = RED
    return out


def _transform_grid(g: np.ndarray, rotations: int, mirror: bool) -> np.ndarray:
    out = np.rot90(g, int(rotations))
    if mirror:
        out = np.fliplr(out)
    return out.copy()


def _sample_branch_lengths() -> tuple[int, int]:
    short_len = random.choice([1, 2, 2, 2, 3, 3, 4])
    long_len = short_len + random.randint(1, min(3, 5 - short_len))
    if random.random() < 0.5:
        return long_len, short_len
    return short_len, long_len


def _build_left_boundary(up_len: int, down_len: int) -> tuple[list[int], int, int]:
    h_min = max(6, up_len + down_len + 3)
    h = random.randint(h_min, h_min + 2)
    start_row = random.randint(up_len + 1, h - down_len - 2)
    start_left = random.randint(1, 3)

    lefts = [0] * h
    lefts[start_row] = start_left
    for k in range(1, up_len + 1):
        lefts[start_row - k] = start_left + k
    for k in range(1, down_len + 1):
        lefts[start_row + k] = start_left + k

    for r in range(start_row - up_len - 1, -1, -1):
        lefts[r] = max(0, lefts[r + 1] - random.choice([0, 1]))
    for r in range(start_row + down_len + 1, h):
        lefts[r] = max(0, lefts[r - 1] - random.choice([0, 1]))

    return lefts, start_row, start_left


def _choose_effective_lengths(raw_pref: int, raw_other: int) -> tuple[int, int]:
    assert raw_pref != raw_other and min(raw_pref, raw_other) >= 1
    raw = [raw_pref, raw_other]
    long_idx = 0 if raw_pref > raw_other else 1
    short_idx = 1 - long_idx
    long_len = raw[long_idx]
    short_len = raw[short_idx]

    outcome = random.choices(["flip", "tie", "keep"], weights=[4, 3, 3])[0]
    want_two = random.random() < 0.5
    eff = raw[:]

    if outcome == "flip":
        if want_two and short_len >= 2:
            eff[short_idx] = random.randint(1, short_len - 1)
            eff[long_idx] = random.randint(0, eff[short_idx] - 1)
        else:
            eff[long_idx] = random.randint(0, short_len - 1)
        return int(eff[0]), int(eff[1])

    if outcome == "tie":
        if want_two and short_len >= 2:
            common = random.randint(1, short_len - 1)
            eff = [common, common]
        else:
            eff[long_idx] = short_len
        return int(eff[0]), int(eff[1])

    if want_two and short_len >= 2:
        eff[short_idx] = random.randint(0, short_len - 1)
        lower = eff[short_idx] + 1
        upper = long_len - 1
        if lower <= upper:
            eff[long_idx] = random.randint(lower, upper)
            return int(eff[0]), int(eff[1])

    if long_len - short_len >= 2 and random.random() < 0.5:
        eff[long_idx] = random.randint(short_len + 1, long_len - 1)
    else:
        eff[short_idx] = random.randint(0, short_len - 1)
    return int(eff[0]), int(eff[1])


def _gen_attempt() -> np.ndarray | None:
    up_len, down_len = _sample_branch_lengths()
    lefts, start_row, start_left = _build_left_boundary(up_len, down_len)
    h = len(lefts)

    thickness = random.randint(2, 3)
    local_w = max(lefts) + thickness
    area = sum(local_w - left for left in lefts)
    if not (15 <= area <= 30):
        return None

    marker_col = random.randint(1, 3)
    sight_gap = random.randint(2, 5)
    landing_col = marker_col + sight_gap
    obstacle_c0 = landing_col + 1 - start_left
    if obstacle_c0 < 2:
        return None

    top_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    right_margin = random.randint(2, 5)
    rows = top_margin + h + bottom_margin
    cols = obstacle_c0 + local_w + right_margin
    if max(rows, cols) > 22:
        return None

    obstacle_r0 = top_margin
    marker_pos = (obstacle_r0 + start_row, marker_col)
    landing = (marker_pos[0], landing_col)

    g = np.full((rows, cols), BG, dtype=int)
    for rr, left in enumerate(lefts):
        g[obstacle_r0 + rr, obstacle_c0 + left : obstacle_c0 + local_w] = GREEN

    if int(g[landing]) != BG or not _adjacent_to_color(g, landing, GREEN):
        return None

    g[marker_pos] = RED
    slide_dir, landing_check = _sightline_to_obstacle(g, marker_pos)
    if slide_dir != (0, 1) or landing_check != landing:
        return None

    turns = _candidate_turns(slide_dir)
    paths = [_walk_branch(g, landing, turn) for turn in turns]
    raw = [len(path) for path in paths]
    if raw[0] == raw[1] or min(raw) < 1:
        return None

    eff_pref, eff_other = _choose_effective_lengths(raw[0], raw[1])
    barrier_positions: list[Coord] = []
    for path, eff_len in zip(paths, [eff_pref, eff_other]):
        if eff_len < len(path):
            barrier_positions.append(path[eff_len])

    if not (1 <= len(barrier_positions) <= 2):
        return None
    if len(set(barrier_positions)) != len(barrier_positions):
        return None

    inp = g.copy()
    for pos in barrier_positions:
        inp[pos] = BLUE

    rotations = random.randrange(4)
    mirror = random.random() < 0.5
    inp = _transform_grid(inp, rotations, mirror)

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-barrier ARC input")

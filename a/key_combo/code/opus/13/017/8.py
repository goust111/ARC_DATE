from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SEPARATOR = 7
BG = 0
KEY = 1
MARKER_COLORS = (2, 3, 4)

TILE_COUNT = 4
TILE_SIZE = 2
TILE_STRIDE = 3
GRID_SIZE = 13
SUBCELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
LANES = tuple(range(len(SUBCELLS)))

MAX_GENERATION_ATTEMPTS = 400


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 1 + TILE_STRIDE * tile_r, 1 + TILE_STRIDE * tile_c


def _tile_cell(tile_r: int, tile_c: int, lane: int) -> tuple[int, int]:
    r0, c0 = _tile_origin(tile_r, tile_c)
    dr, dc = SUBCELLS[lane]
    return r0 + dr, c0 + dc


def _parse_lattice(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    assert h == w == GRID_SIZE, "expected a 13x13 lattice"
    assert np.all(g[0, :] == SEPARATOR) and np.all(g[-1, :] == SEPARATOR), "expected separator boundary rows"
    assert np.all(g[:, 0] == SEPARATOR) and np.all(g[:, -1] == SEPARATOR), "expected separator boundary cols"

    row_runs = _runs([not np.all(g[r, :] == SEPARATOR) for r in range(h)])
    col_runs = _runs([not np.all(g[:, c] == SEPARATOR) for c in range(w)])
    assert len(row_runs) == len(col_runs) == TILE_COUNT, "expected a 4x4 tile lattice"
    assert all(end - start == TILE_SIZE for start, end in row_runs), "expected 2-row tile interiors"
    assert all(end - start == TILE_SIZE for start, end in col_runs), "expected 2-col tile interiors"

    for (start0, end0), (start1, end1) in zip(row_runs, row_runs[1:]):
        assert start1 - end0 == 1, "expected single separator rows between tile bands"
    for (start0, end0), (start1, end1) in zip(col_runs, col_runs[1:]):
        assert start1 - end0 == 1, "expected single separator cols between tile bands"
    return row_runs, col_runs


def _color_order(g: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in MARKER_COLORS and color not in seen:
                seen.add(color)
                order.append(color)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert the top-left key tile and use the revealed active subcells to decide which same-color endpoint lanes get connected across the tile lattice.

    Input structure:
    - The grid is a 4x4 lattice of 2x2 tiles separated by one-cell-wide orange rows and columns, with orange also forming the outer border.
    - The top-left tile is a special key tile containing only black and blue cells.
    - Every other tile is otherwise black and may contain at most one red, green, or yellow endpoint marker at a specific 2x2 subcell position.

    Rule:
    - Invert the key tile cell-by-cell: blue becomes black and black becomes blue.
    - Treat the blue cells in that inverted key as the active subcell lanes.
    - For each endpoint color, group its markers by shared subcell position. In each active group, fill the full span between markers that share a tile row and also the full span between markers that share a tile column; inactive groups do not connect.

    Key insight:
    - The visible blue pattern in the key tile is the complement of the real connection mask, so the usable lanes only appear after inverting that one 2x2 tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_runs, col_runs = _parse_lattice(g)

    key_r0, key_r1 = row_runs[0]
    key_c0, key_c1 = col_runs[0]
    key_tile = g[key_r0:key_r1, key_c0:key_c1]
    assert set(map(int, np.unique(key_tile))).issubset({BG, KEY}), "expected only black/blue in the key tile"
    inverted_key = np.where(key_tile == KEY, BG, KEY)
    active_lanes = {
        lane
        for lane, (dr, dc) in enumerate(SUBCELLS)
        if int(inverted_key[dr, dc]) == KEY
    }

    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    per_color_counts: Counter[int] = Counter()
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            if (tile_r, tile_c) == (0, 0):
                continue
            tile = g[r0:r1, c0:c1]
            markers: list[tuple[int, int]] = []
            for lane, (dr, dc) in enumerate(SUBCELLS):
                color = int(tile[dr, dc])
                if color == BG:
                    continue
                assert color in MARKER_COLORS, "expected only endpoint colors outside the key tile"
                markers.append((lane, color))
            assert len(markers) <= 1, "each non-key tile may contain at most one endpoint"
            if markers:
                lane, color = markers[0]
                groups[(color, lane)].append((tile_r, tile_c))
                per_color_counts[color] += 1

    assert groups, "expected at least one endpoint group"
    assert all(count >= 2 for count in per_color_counts.values()), "each endpoint color must appear at least twice"

    out = g.copy()
    out[key_r0:key_r1, key_c0:key_c1] = inverted_key

    for color in _color_order(g):
        for lane in LANES:
            points = groups.get((color, lane), [])
            if len(points) < 2 or lane not in active_lanes:
                continue

            by_row: dict[int, list[int]] = defaultdict(list)
            by_col: dict[int, list[int]] = defaultdict(list)
            for tile_r, tile_c in points:
                by_row[tile_r].append(tile_c)
                by_col[tile_c].append(tile_r)

            for tile_r, tile_cols in by_row.items():
                if len(tile_cols) < 2:
                    continue
                for tile_c in range(min(tile_cols), max(tile_cols) + 1):
                    r, c = _tile_cell(tile_r, tile_c, lane)
                    out[r, c] = color

            for tile_c, tile_rows in by_col.items():
                if len(tile_rows) < 2:
                    continue
                for tile_r in range(min(tile_rows), max(tile_rows) + 1):
                    r, c = _tile_cell(tile_r, tile_c, lane)
                    out[r, c] = color

    return out


def _make_candidate(shape: str, lane: int, endpoints, footprint):
    return {
        "shape": shape,
        "lane": int(lane),
        "endpoints": tuple(sorted(endpoints)),
        "footprint": frozenset(footprint),
    }


def _all_candidates() -> list[dict]:
    seen: set[tuple[str, int, tuple[tuple[int, int], ...]]] = set()
    candidates: list[dict] = []

    for lane in LANES:
        for tile_r in range(TILE_COUNT):
            for start_c, end_c in ((0, 2), (1, 3), (0, 3)):
                footprint = {(tile_r, tile_c) for tile_c in range(start_c, end_c + 1)}
                if (0, 0) in footprint:
                    continue
                endpoints = ((tile_r, start_c), (tile_r, end_c))
                key = ("H", lane, tuple(sorted(endpoints)))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_make_candidate("H", lane, endpoints, footprint))

        for tile_c in range(TILE_COUNT):
            for start_r, end_r in ((0, 2), (1, 3), (0, 3)):
                footprint = {(tile_r, tile_c) for tile_r in range(start_r, end_r + 1)}
                if (0, 0) in footprint:
                    continue
                endpoints = ((start_r, tile_c), (end_r, tile_c))
                key = ("V", lane, tuple(sorted(endpoints)))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_make_candidate("V", lane, endpoints, footprint))

        for corner_r in range(TILE_COUNT):
            for corner_c in range(TILE_COUNT):
                for end_c in range(TILE_COUNT):
                    if end_c == corner_c or abs(end_c - corner_c) + 1 < 3:
                        continue
                    for end_r in range(TILE_COUNT):
                        if end_r == corner_r or abs(end_r - corner_r) + 1 < 3:
                            continue
                        footprint = {
                            (corner_r, tile_c)
                            for tile_c in range(min(corner_c, end_c), max(corner_c, end_c) + 1)
                        }
                        footprint.update(
                            (tile_r, corner_c)
                            for tile_r in range(min(corner_r, end_r), max(corner_r, end_r) + 1)
                        )
                        if (0, 0) in footprint:
                            continue
                        endpoints = ((corner_r, corner_c), (corner_r, end_c), (end_r, corner_c))
                        key = ("L", lane, tuple(sorted(endpoints)))
                        if key not in seen:
                            seen.add(key)
                            candidates.append(_make_candidate("L", lane, endpoints, footprint))
    return candidates


GEN_CANDIDATES = _all_candidates()


def _pick_candidate(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_shape[candidate["shape"]].append(candidate)
    shape = random.choice(sorted(by_shape))
    return random.choice(by_shape[shape])


def _overlap_ok(first: dict, second: dict) -> bool:
    if first["lane"] != second["lane"]:
        return False
    overlap = set(first["footprint"]) & set(second["footprint"])
    if not overlap:
        return False
    first_endpoints = set(first["endpoints"])
    second_endpoints = set(second["endpoints"])
    if overlap & first_endpoints:
        return False
    if overlap & second_endpoints:
        return False
    if first_endpoints & set(second["footprint"]):
        return False
    if second_endpoints & set(first["footprint"]):
        return False
    return True


def _render_input(active_lanes: set[int], plans: list[dict]) -> np.ndarray:
    g = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for cut in range(0, GRID_SIZE, TILE_STRIDE):
        g[cut, :] = SEPARATOR
        g[:, cut] = SEPARATOR

    for lane in LANES:
        r, c = _tile_cell(0, 0, lane)
        g[r, c] = BG if lane in active_lanes else KEY

    for plan in plans:
        for tile_r, tile_c in plan["endpoints"]:
            r, c = _tile_cell(tile_r, tile_c, plan["lane"])
            assert int(g[r, c]) == BG, "endpoint tiles must be unique"
            g[r, c] = plan["color"]
    return g


def _render_expected(inp: np.ndarray, active_lanes: set[int], plans: list[dict]) -> np.ndarray:
    out = inp.copy()
    for lane in LANES:
        r, c = _tile_cell(0, 0, lane)
        out[r, c] = KEY if lane in active_lanes else BG

    plan_by_color = {plan["color"]: plan for plan in plans}
    for color in _color_order(inp):
        plan = plan_by_color[color]
        if plan["lane"] not in active_lanes:
            continue
        for tile_r, tile_c in plan["footprint"]:
            r, c = _tile_cell(tile_r, tile_c, plan["lane"])
            out[r, c] = color
    return out


def _choose_disjoint_candidate(lanes: list[int], used_footprint: set[tuple[int, int]]) -> dict | None:
    pool = [
        candidate
        for candidate in GEN_CANDIDATES
        if candidate["lane"] in lanes and set(candidate["footprint"]).isdisjoint(used_footprint)
    ]
    return _pick_candidate(pool)


def _attempt_instance() -> np.ndarray | None:
    active_count = random.randint(1, 3)
    active_lanes = set(random.sample(LANES, active_count))
    inactive_lanes = [lane for lane in LANES if lane not in active_lanes]
    overlap_case = random.random() < 0.55

    primary_lane = random.choice(sorted(active_lanes))
    primary = _pick_candidate([c for c in GEN_CANDIDATES if c["lane"] == primary_lane])
    if primary is None:
        return None

    blocked_lane = random.choice(inactive_lanes)
    blocked = _choose_disjoint_candidate([blocked_lane], set(primary["footprint"]))
    if blocked is None:
        return None

    if overlap_case:
        third_pool = [
            candidate
            for candidate in GEN_CANDIDATES
            if candidate["lane"] == primary_lane
            and _overlap_ok(primary, candidate)
            and set(candidate["footprint"]).isdisjoint(set(blocked["footprint"]))
        ]
        third = _pick_candidate(third_pool)
    else:
        lane_preferences: list[list[int]] = []
        other_active = sorted(active_lanes - {primary_lane})
        other_inactive = [lane for lane in inactive_lanes if lane != blocked_lane]
        if other_active:
            lane_preferences.append(other_active)
        lane_preferences.append([primary_lane])
        if other_inactive:
            lane_preferences.append(other_inactive)
        lane_preferences.append([blocked_lane])

        third = None
        used = set(primary["footprint"]) | set(blocked["footprint"])
        for lanes in lane_preferences:
            third = _choose_disjoint_candidate(lanes, used)
            if third is not None:
                break
    if third is None:
        return None

    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    plans = []
    for color, candidate in zip(colors, (primary, blocked, third)):
        plan = dict(candidate)
        plan["color"] = color
        plans.append(plan)

    counts = Counter(plan["color"] for plan in plans)
    assert counts == Counter({2: 1, 3: 1, 4: 1})

    inp = _render_input(active_lanes, plans)
    expected = _render_expected(inp, active_lanes, plans)

    if not np.array_equal(main(inp), expected):
        return None

    for color in MARKER_COLORS:
        assert int(np.count_nonzero(inp == color)) >= 2, "each endpoint color needs at least two markers"
    assert np.any(inp != expected), "output must differ from input"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

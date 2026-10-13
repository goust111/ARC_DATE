from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

GATE_COLORS = {BLUE, RED, YELLOW}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

MIX_MAP = {
    frozenset([RED]): RED,
    frozenset([BLUE]): BLUE,
    frozenset([YELLOW]): YELLOW,
    frozenset([RED, BLUE]): MAGENTA,
    frozenset([RED, YELLOW]): ORANGE,
    frozenset([BLUE, YELLOW]): GREEN,
    frozenset([RED, BLUE, YELLOW]): MAROON,
}
COMBO_LIST = list(MIX_MAP.keys())

MAX_GEN_ATTEMPTS = 300
MAX_PATH_ATTEMPTS = 250


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if _in_bounds(h, w, rr, cc):
            out.append((rr, cc))
    return out


def _station_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _station_approach(center: tuple[int, int], side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = center
    if side == "top":
        return (r - 1, c), (r - 2, c)
    if side == "bottom":
        return (r + 1, c), (r + 2, c)
    if side == "left":
        return (r, c - 1), (r, c - 2)
    if side == "right":
        return (r, c + 1), (r, c + 2)
    raise ValueError(f"unknown side: {side}")


def _find_station_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != GREY:
                continue
            if bool((grid[r - 1 : r + 2, c - 1 : c + 2] == GREY).all()):
                centers.append((r, c))
    return centers


def _trace_path_from_station(
    grid: np.ndarray,
    center: tuple[int, int],
    all_station_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    station = set(_station_cells(center))

    attachment_map: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in station:
        for rr, cc in _neighbors4(h, w, sr, sc):
            if (rr, cc) in station:
                continue
            if int(grid[rr, cc]) != GREY:
                continue
            attachment_map[(rr, cc)] = (sr, sc)

    if len(attachment_map) != 1:
        return None

    current, prev = next(iter(attachment_map.items()))
    path = [current]
    seen = {current, prev}

    while True:
        r, c = current
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            return path

        nxts = [
            (rr, cc)
            for rr, cc in _neighbors4(h, w, r, c)
            if int(grid[rr, cc]) == GREY and (rr, cc) != prev and (rr, cc) not in all_station_cells
        ]
        if len(nxts) != 1:
            return None

        nxt = nxts[0]
        if nxt in seen:
            return None
        path.append(nxt)
        seen.add(nxt)
        prev, current = current, nxt


def _gate_colors_on_path(grid: np.ndarray, path: list[tuple[int, int]]) -> frozenset[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for r, c in path:
        if 0 < r < h - 1:
            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            if up == down and up in GATE_COLORS:
                colors.add(up)
        if 0 < c < w - 1:
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            if left == right and left in GATE_COLORS:
                colors.add(left)
    return frozenset(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each terminal square's center according to the set of gate colors on its path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    centers = _find_station_centers(g)
    all_station_cells: set[tuple[int, int]] = set()
    for center in centers:
        all_station_cells.update(_station_cells(center))

    for center in centers:
        path = _trace_path_from_station(g, center, all_station_cells)
        if path is None:
            continue
        mix = MIX_MAP.get(_gate_colors_on_path(g, path))
        if mix is not None:
            out[center] = int(mix)

    return out


def _expand_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 != r1 and c0 != c1:
            raise ValueError("polyline segments must be axis-aligned")
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        steps = max(abs(r1 - r0), abs(c1 - c0))
        for step in range(1, steps + 1):
            cells.append((r0 + dr * step, c0 + dc * step))
    return cells


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(h, w, rr, cc):
                    out.add((rr, cc))
    return out


def _sample_combo(rng: np.random.Generator, count: int) -> list[frozenset[int]]:
    order = list(rng.permutation(len(COMBO_LIST)))
    return [COMBO_LIST[idx] for idx in order[:count]]


def _random_points_for_path(
    rng: np.random.Generator,
    h: int,
    w: int,
    end: tuple[int, int],
    start_side: str,
    mode: str,
) -> list[tuple[int, int]] | None:
    end_r, end_c = end

    if start_side in ("top", "bottom"):
        start_r = 0 if start_side == "top" else h - 1
        start_c = int(rng.integers(1, w - 1))
        start = (start_r, start_c)
        if mode == "one":
            return [start, (end_r, start_c), end]
        if start_c == end_c:
            return None
        mids = [row for row in range(1, h - 1) if row != end_r]
        if not mids:
            return None
        mid_r = mids[int(rng.integers(len(mids)))]
        return [start, (mid_r, start_c), (mid_r, end_c), end]

    start_c = 0 if start_side == "left" else w - 1
    start_r = int(rng.integers(1, h - 1))
    start = (start_r, start_c)
    if mode == "one":
        return [start, (start_r, end_c), end]
    if start_r == end_r:
        return None
    mids = [col for col in range(1, w - 1) if col != end_c]
    if not mids:
        return None
    mid_c = mids[int(rng.integers(len(mids)))]
    return [start, (start_r, mid_c), (end_r, mid_c), end]


def _usable_gate_positions(
    path_cells: list[tuple[int, int]],
    station_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]]:
    path_set = set(path_cells)
    usable: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for idx in range(1, len(path_cells) - 1):
        prev_r, prev_c = path_cells[idx - 1]
        cur_r, cur_c = path_cells[idx]
        next_r, next_c = path_cells[idx + 1]

        if prev_r == cur_r == next_r:
            side_cells = ((cur_r - 1, cur_c), (cur_r + 1, cur_c))
        elif prev_c == cur_c == next_c:
            side_cells = ((cur_r, cur_c - 1), (cur_r, cur_c + 1))
        else:
            continue

        if not all(_in_bounds(h, w, rr, cc) for rr, cc in side_cells):
            continue
        if any(cell in station_cells or cell in path_set or cell in blocked for cell in side_cells):
            continue
        usable.append((idx, side_cells))
    return usable


def _build_path_object(
    rng: np.random.Generator,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    combo: frozenset[int],
) -> dict | None:
    margin = 5
    if h < 2 * margin + 1 or w < 2 * margin + 1:
        return None

    center = (
        int(rng.integers(margin, h - margin)),
        int(rng.integers(margin, w - margin)),
    )
    station = set(_station_cells(center))
    approach_side = SIDES[int(rng.integers(len(SIDES)))]
    station_touch, end = _station_approach(center, approach_side)

    path_mode = "two" if float(rng.random()) < 0.45 else "one"
    start_side = SIDES[int(rng.integers(len(SIDES)))]
    points = _random_points_for_path(rng, h, w, end, start_side, path_mode)
    if points is None:
        return None

    path_cells = _expand_polyline(points)
    if len(path_cells) < 7:
        return None
    if len(set(path_cells)) != len(path_cells):
        return None
    if path_cells[-1] != end:
        return None

    ring = {
        (rr, cc)
        for rr in range(center[0] - 2, center[0] + 3)
        for cc in range(center[1] - 2, center[1] + 3)
        if _in_bounds(h, w, rr, cc)
    }
    if any(cell in ring for cell in path_cells[:-1]):
        return None
    if any(cell in station for cell in path_cells):
        return None
    if any(cell in blocked for cell in station):
        return None
    if any(cell in blocked for cell in path_cells):
        return None

    usable = _usable_gate_positions(path_cells, station, blocked, h, w)
    if len(usable) < len(combo):
        return None

    max_gate_count = min(len(usable), len(combo) + 2)
    gate_count = int(rng.integers(len(combo), max_gate_count + 1))
    gate_order = list(rng.permutation(len(usable)))[:gate_count]
    chosen_slots = [usable[idx] for idx in gate_order]

    gate_colors = list(combo)
    while len(gate_colors) < gate_count:
        colors = list(combo)
        gate_colors.append(colors[int(rng.integers(len(colors)))])
    gate_colors = [gate_colors[idx] for idx in rng.permutation(len(gate_colors))]

    gate_cells: dict[tuple[int, int], int] = {}
    for (_, side_cells), color in zip(chosen_slots, gate_colors):
        for cell in side_cells:
            if cell in gate_cells:
                return None
            gate_cells[cell] = int(color)

    actual = set(station) | set(path_cells) | set(gate_cells)
    if any(cell in blocked for cell in gate_cells):
        return None
    forbidden = _dilate(actual, h, w)

    return {
        "center": center,
        "combo": combo,
        "station_touch": station_touch,
        "path_cells": path_cells,
        "gate_cells": gate_cells,
        "actual": actual,
        "forbidden": forbidden,
    }


def _place_path_object(grid: np.ndarray, spec: dict) -> None:
    for r, c in _station_cells(spec["center"]):
        grid[r, c] = GREY
    for r, c in spec["path_cells"]:
        grid[r, c] = GREY
    for (r, c), color in spec["gate_cells"].items():
        grid[r, c] = int(color)


def _validate_generated_grid(grid: np.ndarray, specs: list[dict]) -> bool:
    centers = _find_station_centers(grid)
    expected_centers = {tuple(spec["center"]) for spec in specs}
    if set(centers) != expected_centers:
        return False

    all_station_cells: set[tuple[int, int]] = set()
    for center in centers:
        all_station_cells.update(_station_cells(center))

    for spec in specs:
        center = tuple(spec["center"])
        path = _trace_path_from_station(grid, center, all_station_cells)
        if path is None:
            return False
        if tuple(path[-1]) not in {(0, path[-1][1]), (grid.shape[0] - 1, path[-1][1]), (path[-1][0], 0), (path[-1][0], grid.shape[1] - 1)}:
            return False
        if _gate_colors_on_path(grid, path) != spec["combo"]:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    max_paths = 5 if min(h, w) >= 23 else 4
    path_count = int(rng.integers(2, max_paths + 1))

    grid = np.full((h, w), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for combo in _sample_combo(rng, path_count):
        placed = False
        for _ in range(MAX_PATH_ATTEMPTS):
            spec = _build_path_object(rng, h, w, blocked, combo)
            if spec is None:
                continue
            _place_path_object(grid, spec)
            blocked |= spec["forbidden"]
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated_grid(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")

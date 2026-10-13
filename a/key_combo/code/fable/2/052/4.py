import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _wall_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for nr, nc in neighbors4(r, c):
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in (BLUE, RED):
            out.append((nr, nc))
    return out


def _scan_gap_from_endpoint(
    grid: np.ndarray, r: int, c: int
) -> tuple[int, int, int, int, int] | None:
    neighbors = _wall_neighbors(grid, r, c)
    if len(neighbors) != 1:
        return None

    wr, wc = neighbors[0]
    dr, dc = r - wr, c - wc
    rr, cc = r + dr, c + dc
    width = 0
    while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BLACK:
        rr += dr
        cc += dc
        width += 1

    if width == 0 or not in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != RED:
        return None

    partner_neighbors = _wall_neighbors(grid, rr, cc)
    if len(partner_neighbors) != 1 or partner_neighbors[0] != (rr + dr, cc + dc):
        return None

    if (rr, cc) < (r, c):
        r, c, rr, cc = rr, cc, r, c
    return r, c, rr, cc, width


def _find_gap_pairs(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    pairs: dict[tuple[int, int, int, int], tuple[int, int, int, int, int]] = {}
    for r, c in np.argwhere(grid == RED):
        pair = _scan_gap_from_endpoint(grid, int(r), int(c))
        if pair is not None:
            pairs[pair[:4]] = pair

    red_points = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    used = set()
    for r0, c0, r1, c1, _width in pairs.values():
        used.add((r0, c0))
        used.add((r1, c1))
    assert used == red_points, "every red marker must belong to exactly one straight opening"
    return sorted(pairs.values())


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo, hi + 1)]


def _border_reachable_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == BLACK:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Seal only the globally narrowest marked openings, then fill the cavities that become fully enclosed.

    Input structure:
    - The background is black.
    - Blue one-cell-thick walls form several cavity outlines, with one or more straight gaps on their borders.
    - Each gap is marked by a red endpoint pixel at each wall endpoint; the two endpoints are collinear.

    Rule:
    - Measure every opening width as the number of black cells strictly between its two red endpoints.
    - Find the global minimum width and seal only those openings by painting the whole endpoint-to-endpoint segment blue.
    - Recompute enclosure after sealing: any black region that can no longer reach the canvas boundary by 4-neighbor steps is filled green.

    Key insight:
    - Filling is a second-stage consequence of selective sealing: a cavity turns green only if every one of its openings belongs to the global narrowest-width class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pairs = _find_gap_pairs(g)
    assert pairs, "expected at least one marked opening"
    min_width = min(width for _r0, _c0, _r1, _c1, width in pairs)

    out = g.copy()
    for r0, c0, r1, c1, width in pairs:
        if width != min_width:
            continue
        for r, c in _segment_cells(r0, c0, r1, c1):
            out[r, c] = BLUE

    reachable = _border_reachable_background(out)
    enclosed = (out == BLACK) & ~reachable
    out[enclosed] = GREEN
    return out


def _sample_room(role: str, min_width: int) -> dict:
    if role == "chain":
        opening_count = random.choice((2, 2, 3))
        widths = [min_width] * opening_count
    elif role == "mixed":
        opening_count = random.choice((2, 3))
        widths = [min_width]
        for _ in range(opening_count - 1):
            widths.append(random.randint(min_width + 1, min_width + 3))
        random.shuffle(widths)
    elif role == "all_wide":
        opening_count = random.choice((1, 2))
        widths = [random.randint(min_width + 1, min_width + 3) for _ in range(opening_count)]
    elif role == "single_min":
        widths = [min_width]
    else:
        raise ValueError(f"unknown role: {role}")

    sides = random.sample(SIDES, len(widths))
    max_h_gap = max([width for side, width in zip(sides, widths) if side in ("left", "right")] or [1])
    max_w_gap = max([width for side, width in zip(sides, widths) if side in ("top", "bottom")] or [1])
    height_min = max(6, max_h_gap + 4)
    width_min = max(6, max_w_gap + 4)
    height = random.randint(height_min, min(10, height_min + 2))
    width = random.randint(width_min, min(10, width_min + 2))

    return {
        "role": role,
        "height": height,
        "width": width,
        "opening_specs": [{"side": side, "width": width} for side, width in zip(sides, widths)],
    }


def _choose_layout(rooms: list[dict]) -> tuple[int, int, list[tuple[dict, tuple[int, int]]]] | None:
    slots = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], len(rooms))
    random.shuffle(rooms)
    assignments = list(zip(rooms, slots))

    row_requirements = [4, 4]
    col_requirements = [4, 4]
    for room, (sr, sc) in assignments:
        row_requirements[sr] = max(row_requirements[sr], room["height"] + random.randint(2, 4))
        col_requirements[sc] = max(col_requirements[sc], room["width"] + random.randint(2, 4))

    row_gap = random.randint(2, 3)
    col_gap = random.randint(2, 3)
    grid_h = row_requirements[0] + row_requirements[1] + row_gap
    grid_w = col_requirements[0] + col_requirements[1] + col_gap
    if grid_h > 30 or grid_w > 30:
        return None

    row_starts = [0, row_requirements[0] + row_gap]
    col_starts = [0, col_requirements[0] + col_gap]
    for room, (sr, sc) in assignments:
        r_band = row_starts[sr]
        c_band = col_starts[sc]
        band_h = row_requirements[sr]
        band_w = col_requirements[sc]
        room["r0"] = random.randint(r_band + 1, r_band + band_h - room["height"] - 1)
        room["c0"] = random.randint(c_band + 1, c_band + band_w - room["width"] - 1)
    return grid_h, grid_w, assignments


def _realize_openings(room: dict) -> None:
    r0 = room["r0"]
    c0 = room["c0"]
    r1 = r0 + room["height"] - 1
    c1 = c0 + room["width"] - 1
    room["bbox"] = (r0, c0, r1, c1)
    room["openings"] = []

    for spec in room["opening_specs"]:
        side = spec["side"]
        width = spec["width"]
        if side == "top":
            start = random.randint(c0 + 2, c1 - width - 1)
            endpoints = ((r0, start - 1), (r0, start + width))
            gap_cells = [(r0, c) for c in range(start, start + width)]
        elif side == "bottom":
            start = random.randint(c0 + 2, c1 - width - 1)
            endpoints = ((r1, start - 1), (r1, start + width))
            gap_cells = [(r1, c) for c in range(start, start + width)]
        elif side == "left":
            start = random.randint(r0 + 2, r1 - width - 1)
            endpoints = ((start - 1, c0), (start + width, c0))
            gap_cells = [(r, c0) for r in range(start, start + width)]
        elif side == "right":
            start = random.randint(r0 + 2, r1 - width - 1)
            endpoints = ((start - 1, c1), (start + width, c1))
            gap_cells = [(r, c1) for r in range(start, start + width)]
        else:
            raise ValueError(f"unknown side: {side}")

        room["openings"].append(
            {
                "side": side,
                "width": width,
                "endpoints": endpoints,
                "gap_cells": gap_cells,
            }
        )


def _draw_case(assignments: list[tuple[dict, tuple[int, int]]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for room, _slot in assignments:
        _realize_openings(room)
        r0, c0, r1, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = BLUE
        grid[r1, c0 : c1 + 1] = BLUE
        grid[r0 : r1 + 1, c0] = BLUE
        grid[r0 : r1 + 1, c1] = BLUE
        for opening in room["openings"]:
            for r, c in opening["gap_cells"]:
                grid[r, c] = BLACK
            for r, c in opening["endpoints"]:
                grid[r, c] = RED
    return grid


def _expected_output(grid: np.ndarray, rooms: list[dict]) -> np.ndarray:
    min_width = min(opening["width"] for room in rooms for opening in room["openings"])
    out = grid.copy()
    for room in rooms:
        for opening in room["openings"]:
            if opening["width"] != min_width:
                continue
            (r0, c0), (r1, c1) = opening["endpoints"]
            for r, c in _segment_cells(r0, c0, r1, c1):
                out[r, c] = BLUE
        if all(opening["width"] == min_width for opening in room["openings"]):
            r0, c0, r1, c1 = room["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = GREEN
    return out


def _validate_case(grid: np.ndarray, rooms: list[dict]) -> bool:
    openings = [opening for room in rooms for opening in room["openings"]]
    widths = [opening["width"] for opening in openings]
    if len(set(widths)) < 2:
        return False

    min_width = min(widths)
    if not any(
        len(room["openings"]) >= 2 and all(opening["width"] == min_width for opening in room["openings"])
        for room in rooms
    ):
        return False
    if not any(
        any(opening["width"] == min_width for opening in room["openings"])
        and any(opening["width"] > min_width for opening in room["openings"])
        for room in rooms
    ):
        return False

    expected_pairs = set()
    for opening in openings:
        (r0, c0), (r1, c1) = opening["endpoints"]
        if (r1, c1) < (r0, c0):
            r0, c0, r1, c1 = r1, c1, r0, c0
        expected_pairs.add((r0, c0, r1, c1, opening["width"]))
    try:
        actual_pairs = set(_find_gap_pairs(grid))
    except AssertionError:
        return False
    if actual_pairs != expected_pairs:
        return False

    out = main(grid)
    expected = _expected_output(grid, rooms)
    if not np.array_equal(out, expected):
        return False
    if not np.any(out == GREEN):
        return False
    if not np.any(out == RED):
        return False

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        fillable = all(opening["width"] == min_width for opening in room["openings"])
        if fillable:
            if not np.all(interior == GREEN):
                return False
        elif not np.all(interior == BLACK):
            return False

        for opening in room["openings"]:
            endpoint_a, endpoint_b = opening["endpoints"]
            if opening["width"] == min_width:
                for r, c in _segment_cells(endpoint_a[0], endpoint_a[1], endpoint_b[0], endpoint_b[1]):
                    if int(out[r, c]) != BLUE:
                        return False
            else:
                if int(out[endpoint_a]) != RED or int(out[endpoint_b]) != RED:
                    return False
                for r, c in opening["gap_cells"]:
                    if int(out[r, c]) != BLACK:
                        return False

    return True


def _sample_roles() -> list[str]:
    roles = ["chain", "mixed"]
    if random.random() < 0.7:
        roles.append(random.choice(("all_wide", "single_min", "chain", "mixed")))
    return roles


def _gen_attempt() -> np.ndarray | None:
    min_width = random.randint(1, 3)
    rooms = [_sample_room(role, min_width) for role in _sample_roles()]
    layout = _choose_layout(rooms)
    if layout is None:
        return None
    grid_h, grid_w, assignments = layout
    grid = _draw_case(assignments, (grid_h, grid_w))
    if not _validate_case(grid, rooms):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid narrow-seal cavity input")

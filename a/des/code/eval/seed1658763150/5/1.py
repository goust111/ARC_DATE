from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r) + dr, int(c) + dc) for dr, dc in DIRS4]


def _components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = {start}
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _mask_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    coords = {tuple(map(int, pos)) for pos in np.argwhere(mask)}
    return _components(coords)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _flood_from_border(walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not walls[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not walls[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                stack.append((nr, nc))
    return reachable


def _enclosed_after_closing(walls: np.ndarray, gap: tuple[int, int]) -> np.ndarray:
    closed = walls.copy()
    closed[int(gap[0]), int(gap[1])] = True
    reachable = _flood_from_border(closed)
    return (~closed) & (~reachable)


def _analyze_region(
    g: np.ndarray, grey_comp: set[tuple[int, int]]
) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = _bbox(grey_comp)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(g.shape[0] - 1, r1 + 1)
    pc1 = min(g.shape[1] - 1, c1 + 1)
    patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    walls = patch == GREY

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for lr in range(patch.shape[0]):
        for lc in range(patch.shape[1]):
            if walls[lr, lc]:
                continue
            if not any(
                0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and walls[nr, nc]
                for nr, nc in _neighbors4(lr, lc)
            ):
                continue

            enclosed = _enclosed_after_closing(walls, (lr, lc))
            if not bool(enclosed.any()):
                continue

            seed_mask = enclosed & (patch != BLACK) & (patch != GREY)
            seed_colors = np.unique(patch[seed_mask])
            if seed_colors.size != 1:
                continue

            inside_dirs = []
            for dr, dc in DIRS4:
                nr = lr + dr
                nc = lc + dc
                if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and enclosed[nr, nc]:
                    inside_dirs.append((dr, dc))
            if len(inside_dirs) != 1:
                continue

            black_count = int(np.count_nonzero(enclosed & (patch == BLACK)))
            if black_count <= 0:
                continue

            enclosed_area = int(np.count_nonzero(enclosed))
            step = (-inside_dirs[0][0], -inside_dirs[0][1])
            candidates.append(
                (
                    enclosed_area,
                    black_count,
                    int(seed_colors[0]),
                    pr0 + lr,
                    pc0 + lc,
                    step[0] * 10 + step[1],
                )
            )

    assert candidates, "expected one almost-closed grey region with a valid gap"
    best_area = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "gap closure should be unique"
    _area, black_count, seed_color, gap_r, gap_c, step_code = best[0]
    step = {
        -10: (-1, 0),
        10: (1, 0),
        -1: (0, -1),
        1: (0, 1),
    }[step_code]
    return gap_r, gap_c, step[0], step[1], black_count, seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one outward ray from each grey almost-closed region.

    Each region has a single interior colored seed and a single one-pixel gap in
    the grey boundary. Close the gap conceptually, count the enclosed black cells,
    then paint a seed-colored ray of that length starting at the gap and moving
    outward through the opening.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_components = _mask_components(g == GREY)
    assert grey_components, "expected at least one grey boundary region"

    out = g.copy()
    for comp in grey_components:
        gap_r, gap_c, dr, dc, length, color = _analyze_region(g, comp)
        for k in range(length):
            rr = gap_r + dr * k
            cc = gap_c + dc * k
            assert _in_bounds(g.shape, rr, cc), "ray must remain inside the grid"
            assert int(g[rr, cc]) == BLACK, "input ray path must be empty"
            assert int(out[rr, cc]) == BLACK, "generated rays must not overlap"
            out[rr, cc] = int(color)
    return out


# --------------- generator ---------------

GRID_MIN = 16
GRID_MAX = 26
OBJECT_COUNT_RANGE = (1, 4)
INTERIOR_SIZE_RANGE = (2, 9)
POLYOMINO_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 400


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r, min_c, _max_r, _max_c = _bbox(cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    return len(_components(cells)) == 1


def _grow_polyomino(size: int) -> set[tuple[int, int]] | None:
    for _ in range(POLYOMINO_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in _neighbors4(r, c)
                    if nbr not in cells
                }
            )
            weights = [
                1 + sum((nr, nc) in cells for nr, nc in _neighbors4(fr, fc))
                for fr, fc in frontier
            ]
            chosen = random.choices(frontier, weights=weights, k=1)[0]
            cells.add(chosen)

        cells = _normalize(cells)
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(rows) > 5 or len(cols) > 5:
            continue
        return cells
    return None


def _boundary_of(interior: set[tuple[int, int]]) -> set[tuple[int, int]]:
    outer: set[tuple[int, int]] = set()
    for r, c in interior:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                outer.add((r + dr, c + dc))
    return outer - interior


def _gap_candidates(
    interior: set[tuple[int, int]],
    boundary: set[tuple[int, int]],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = step
    out: list[tuple[int, int]] = []
    for br, bc in boundary:
        if sum((nr, nc) in interior for nr, nc in _neighbors4(br, bc)) != 1:
            continue
        if (br - dr, bc - dc) not in interior:
            continue
        if (br + dr, bc + dc) in interior or (br + dr, bc + dc) in boundary:
            continue
        if not _is_connected(boundary - {(br, bc)}):
            continue
        out.append((br, bc))
    return out


def _pad_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    padded.add((rr, cc))
    return padded


def _sample_room(step: tuple[int, int], color: int) -> dict[str, object] | None:
    for _ in range(POLYOMINO_ATTEMPTS):
        interior_size = random.randint(*INTERIOR_SIZE_RANGE)
        interior = _grow_polyomino(interior_size)
        if interior is None:
            continue

        boundary = _boundary_of(interior)
        gaps = _gap_candidates(interior, boundary, step)
        if not gaps:
            continue

        gap = random.choice(gaps)
        seed = random.choice(sorted(interior))
        combined = interior | boundary
        normalized = _normalize(combined)
        min_r, min_c, _max_r, _max_c = _bbox(combined)
        shift_r = -min_r
        shift_c = -min_c

        interior_n = {(r + shift_r, c + shift_c) for r, c in interior}
        boundary_n = {(r + shift_r, c + shift_c) for r, c in boundary}
        gap_n = (gap[0] + shift_r, gap[1] + shift_c)
        seed_n = (seed[0] + shift_r, seed[1] + shift_c)
        black_count = interior_size - 1

        return {
            "color": int(color),
            "step": tuple(step),
            "interior": interior_n,
            "boundary": boundary_n - {gap_n},
            "room_cells": interior_n | boundary_n,
            "gap": gap_n,
            "seed": seed_n,
            "length": int(black_count),
            "shape": (max(r for r, _c in normalized) + 1, max(c for _r, c in normalized) + 1),
        }
    return None


def _place_room(
    room: dict[str, object],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    box_h, box_w = room["shape"]
    positions = [(r0, c0) for r0 in range(1, h - box_h) for c0 in range(1, w - box_w)]
    random.shuffle(positions)

    room_cells = room["room_cells"]
    gap_r, gap_c = room["gap"]
    dr, dc = room["step"]
    length = room["length"]

    for r0, c0 in positions:
        ray = [(r0 + gap_r + dr * k, c0 + gap_c + dc * k) for k in range(length)]
        if any(not _in_bounds(shape, rr, cc) for rr, cc in ray):
            continue

        absolute_room = {(r0 + r, c0 + c) for r, c in room_cells}
        footprint = absolute_room | set(ray)
        if footprint & blocked:
            continue
        return r0, c0
    return None


def _draw_room(inp: np.ndarray, expected: np.ndarray, room: dict[str, object], r0: int, c0: int) -> set[tuple[int, int]]:
    color = int(room["color"])
    for r, c in room["boundary"]:
        inp[r0 + r, c0 + c] = GREY
        expected[r0 + r, c0 + c] = GREY
    seed_r, seed_c = room["seed"]
    inp[r0 + seed_r, c0 + seed_c] = color
    expected[r0 + seed_r, c0 + seed_c] = color

    gap_r, gap_c = room["gap"]
    dr, dc = room["step"]
    length = int(room["length"])
    ray_cells = set()
    for k in range(length):
        rr = r0 + gap_r + dr * k
        cc = c0 + gap_c + dc * k
        expected[rr, cc] = color
        ray_cells.add((rr, cc))

    absolute_room = {(r0 + r, c0 + c) for r, c in room["room_cells"]}
    return absolute_room | ray_cells


def _generate_once() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    obj_max = OBJECT_COUNT_RANGE[1] if min(h, w) >= 20 else 3
    obj_count = random.randint(OBJECT_COUNT_RANGE[0], obj_max)
    colors = random.sample(SEED_COLORS, obj_count)

    rooms: list[dict[str, object]] = []
    for color in colors:
        step = random.choice(DIRS4)
        room = _sample_room(step, color)
        if room is None:
            return None
        rooms.append(room)
    random.shuffle(rooms)

    inp = np.full((h, w), BLACK, dtype=int)
    expected = inp.copy()
    blocked: set[tuple[int, int]] = set()

    for room in rooms:
        pos = _place_room(room, blocked, (h, w))
        if pos is None:
            return None
        placed = _draw_room(inp, expected, room, *pos)
        blocked |= _pad_cells(placed, (h, w))

    if len(_mask_components(inp == GREY)) != obj_count:
        return None
    if int(np.count_nonzero((inp != BLACK) & (inp != GREY))) != obj_count:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid sample")

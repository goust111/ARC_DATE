from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

MIN_SIDE = 14
MAX_SIDE = 26
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mirror_positions(r: int, c: int, axis_r: int, axis_c: int) -> list[tuple[int, int]]:
    return [
        (r, 2 * axis_c - c),
        (2 * axis_r - r, c),
        (2 * axis_r - r, 2 * axis_c - c),
    ]


def _orbit_signature(r: int, c: int, shape: tuple[int, int], axis_r: int, axis_c: int) -> tuple[bool, bool]:
    h, w = shape
    rr = 2 * axis_r - r
    cc = 2 * axis_c - c
    return (0 <= rr < h, 0 <= cc < w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orange pixel by how many companions it has in its two-axis mirror orbit, where the two hidden axes are specified by marker counts.

    Input structure:
    - The grid has black background, orange foreground pixels, one teal marker clump, and one grey marker clump.
    - The teal pixel count gives a vertical axis column and the grey pixel count gives a horizontal axis row, both counted from 1.
    - The markers sit on the border and stay separate from the orange motif.

    Rule:
    - Let c be the teal count and r be the grey count; the hidden axes are column c and row r in 1-based coordinates.
    - For every orange pixel, inspect its vertical reflection, horizontal reflection, and combined reflection; ignore reflected positions that leave the grid.
    - If 3 of those reflected positions are orange, recolor the pixel red; if 1 or 2 are orange, recolor it yellow; if 0 are orange, recolor it blue.
    - Keep the teal and grey marker pixels unchanged, and keep black background black.

    Key insight:
    - The edge markers do not point to the axes geometrically; only their pixel counts reveal the hidden row and column that define the mirror orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = map(int, g.shape)
    teal_count = int(np.count_nonzero(g == TEAL))
    grey_count = int(np.count_nonzero(g == GREY))
    assert 1 <= teal_count <= w, "teal marker count must specify a valid column"
    assert 1 <= grey_count <= h, "grey marker count must specify a valid row"

    axis_c = teal_count - 1
    axis_r = grey_count - 1

    out = g.copy()
    for r, c in np.argwhere(g == ORANGE):
        m = 0
        for rr, cc in _mirror_positions(int(r), int(c), axis_r, axis_c):
            if _in_bounds((h, w), rr, cc) and int(g[rr, cc]) == ORANGE:
                m += 1
        out[r, c] = RED if m == 3 else YELLOW if m in (1, 2) else BLUE
    return out


def _connected_cluster(cells: list[tuple[int, int]], boundary_cells: list[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    allowed = set(cells)
    start = random.choice(boundary_cells)
    chosen = {start}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in allowed and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        chosen.add(random.choice(frontier))
    return chosen


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        comps += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return comps


def _allowed_domain(h: int, w: int) -> tuple[range, range]:
    return range(3, h - 3), range(3, w - 3)


def _orbit_catalog(h: int, w: int, axis_r: int, axis_c: int) -> dict[str, list[tuple[tuple[int, int], ...]]]:
    rows, cols = _allowed_domain(h, w)
    row_set = set(rows)
    col_set = set(cols)
    catalog: dict[str, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    seen: set[frozenset[tuple[int, int]]] = set()
    for r in rows:
        for c in cols:
            if r == axis_r or c == axis_c:
                continue
            rr = 2 * axis_r - r
            cc = 2 * axis_c - c

            if 0 <= rr < h and rr not in row_set:
                continue
            if 0 <= cc < w and cc not in col_set:
                continue

            row_in = 0 <= rr < h
            col_in = 0 <= cc < w

            orbit = {(r, c)}
            if row_in:
                orbit.add((rr, c))
            if col_in:
                orbit.add((r, cc))
            if row_in and col_in:
                orbit.add((rr, cc))

            key = frozenset(orbit)
            if key in seen:
                continue
            seen.add(key)

            if row_in and col_in:
                kind = "full"
            elif row_in:
                kind = "row_only"
            elif col_in:
                kind = "col_only"
            else:
                kind = "none"
            catalog[kind].append(tuple(sorted(orbit)))
    return catalog


def _far_enough(existing: set[tuple[int, int]], orbit: tuple[tuple[int, int], ...], gap: int) -> bool:
    for r, c in orbit:
        for rr, cc in existing:
            if max(abs(r - rr), abs(c - cc)) <= gap:
                return False
    return True


def _pick_orbit(
    candidates: list[tuple[tuple[int, int], ...]],
    used: set[frozenset[tuple[int, int]]],
    occupied: set[tuple[int, int]],
    gap: int,
) -> tuple[tuple[int, int], ...] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for orbit in shuffled:
        key = frozenset(orbit)
        if key in used:
            continue
        if _far_enough(occupied, orbit, gap):
            used.add(key)
            return orbit
    return None


def _paint_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], keep: int) -> set[tuple[int, int]]:
    chosen = set(random.sample(list(orbit), keep))
    for r, c in chosen:
        grid[r, c] = ORANGE
    return chosen


def _place_markers(grid: np.ndarray, axis_r: int, axis_c: int) -> tuple[str, str] | None:
    h, w = grid.shape

    teal_side = random.choice(["top", "bottom"])
    grey_side = random.choice(["left", "right"])

    if teal_side == "top":
        teal_cells = [(r, c) for r in (0, 1) for c in range(2, w - 2)]
        teal_boundary = [(0, c) for c in range(2, w - 2)]
    else:
        teal_cells = [(r, c) for r in (h - 2, h - 1) for c in range(2, w - 2)]
        teal_boundary = [(h - 1, c) for c in range(2, w - 2)]

    if grey_side == "left":
        grey_cells = [(r, c) for r in range(2, h - 2) for c in (0, 1)]
        grey_boundary = [(r, 0) for r in range(2, h - 2)]
    else:
        grey_cells = [(r, c) for r in range(2, h - 2) for c in (w - 2, w - 1)]
        grey_boundary = [(r, w - 1) for r in range(2, h - 2)]

    teal_cluster = _connected_cluster(teal_cells, teal_boundary, axis_c + 1)
    grey_cluster = _connected_cluster(grey_cells, grey_boundary, axis_r + 1)
    if teal_cluster is None or grey_cluster is None:
        return None

    for r, c in teal_cluster:
        grid[r, c] = TEAL
    for r, c in grey_cluster:
        grid[r, c] = GREY
    return teal_side, grey_side


def _reflected_mask(mask: np.ndarray, axis_r: int, axis_c: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    row_ref = np.zeros_like(mask, dtype=bool)
    col_ref = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        rr = 2 * axis_r - r
        cc = 2 * axis_c - c
        if 0 <= rr < h:
            row_ref[rr, c] = True
        if 0 <= cc < w:
            col_ref[r, cc] = True
    return row_ref, col_ref


def _orange_case_summary(mask: np.ndarray, axis_r: int, axis_c: int) -> set[tuple[bool, bool, int]]:
    h, w = mask.shape
    cases = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        row_in, col_in = _orbit_signature(r, c, (h, w), axis_r, axis_c)
        m = 0
        for rr, cc in _mirror_positions(r, c, axis_r, axis_c):
            if _in_bounds((h, w), rr, cc) and mask[rr, cc]:
                m += 1
        cases.add((row_in, col_in, m))
    return cases


def _validate_scene(grid: np.ndarray, axis_r: int, axis_c: int) -> bool:
    h, w = grid.shape
    if axis_r == h // 2 and h % 2 == 1:
        return False
    if axis_c == w // 2 and w % 2 == 1:
        return False

    orange_mask = grid == ORANGE
    teal_mask = grid == TEAL
    grey_mask = grid == GREY

    if np.any(orange_mask[:3, :]) or np.any(orange_mask[-3:, :]) or np.any(orange_mask[:, :3]) or np.any(orange_mask[:, -3:]):
        return False
    if int(teal_mask.sum()) != axis_c + 1 or int(grey_mask.sum()) != axis_r + 1:
        return False
    if _count_components(teal_mask) != 1 or _count_components(grey_mask) != 1:
        return False

    row_ref, col_ref = _reflected_mask(orange_mask, axis_r, axis_c)
    if np.array_equal(orange_mask, row_ref) or np.array_equal(orange_mask, col_ref):
        return False

    cases = _orange_case_summary(orange_mask, axis_r, axis_c)
    required = {
        (True, True, 3),
        (True, True, 2),
        (True, False, 1),
        (False, True, 1),
    }
    if not required.issubset(cases):
        return False
    if not any(m == 0 for _, _, m in cases):
        return False

    out = main(grid)
    output_colors = set(map(int, np.unique(out)))
    if not {BLUE, YELLOW, RED, TEAL, GREY, BLACK}.issubset(output_colors | {BLACK}):
        return False
    if BLUE not in output_colors or YELLOW not in output_colors or RED not in output_colors:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)

    valid_axis_rows = [r for r in range(4, h - 4) if not (h % 2 == 1 and r == h // 2)]
    valid_axis_cols = [c for c in range(4, w - 4) if not (w % 2 == 1 and c == w // 2)]
    if not valid_axis_rows or not valid_axis_cols:
        return None

    axis_r = random.choice(valid_axis_rows)
    axis_c = random.choice(valid_axis_cols)
    catalog = _orbit_catalog(h, w, axis_r, axis_c)
    if not catalog["full"] or not catalog["row_only"] or not catalog["col_only"]:
        return None

    blue_pool = []
    for kind in ("none", "full", "row_only", "col_only"):
        blue_pool.extend((kind, orbit) for orbit in catalog[kind])
    if not blue_pool:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    used: set[frozenset[tuple[int, int]]] = set()
    occupied: set[tuple[int, int]] = set()

    for kind, keep in (("full", 4), ("full", 3), ("row_only", 2), ("col_only", 2)):
        orbit = _pick_orbit(catalog[kind], used, occupied, gap=1)
        if orbit is None:
            return None
        occupied |= _paint_orbit(grid, orbit, keep)

    random.shuffle(blue_pool)
    blue_orbit = None
    for kind, orbit in blue_pool:
        key = frozenset(orbit)
        if key in used:
            continue
        if _far_enough(occupied, orbit, 1):
            used.add(key)
            blue_orbit = orbit
            break
    if blue_orbit is None:
        blue_orbit = _pick_orbit(catalog["full"], used, occupied, gap=0)
        if blue_orbit is None:
            return None
    occupied |= _paint_orbit(grid, blue_orbit, 1)

    extra_count = random.randint(0, 3)
    extra_options = [
        ("full", 1),
        ("full", 2),
        ("full", 3),
        ("row_only", 1),
        ("col_only", 1),
        ("none", 1),
    ]
    for _ in range(extra_count):
        random.shuffle(extra_options)
        placed = False
        for kind, keep in extra_options:
            if not catalog[kind]:
                continue
            orbit = _pick_orbit(catalog[kind], used, occupied, gap=1)
            if orbit is None or keep > len(orbit):
                continue
            occupied |= _paint_orbit(grid, orbit, keep)
            placed = True
            break
        if not placed:
            break

    if _place_markers(grid, axis_r, axis_c) is None:
        return None
    if not _validate_scene(grid, axis_r, axis_c):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/10/015/4")

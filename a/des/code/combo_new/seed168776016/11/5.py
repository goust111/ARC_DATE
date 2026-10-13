import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

GEN_RETRIES = 500
GRID_MIN = 15
GRID_MAX = 20
MIRROR_LENGTHS = (3, 5)


def _in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _seed_cells(grid):
    pts = np.argwhere((grid != BLACK) & (grid != GREY) & (grid != MAROON))
    return [(int(r), int(c), int(grid[r, c])) for r, c in pts]


def _detect_mirrors(grid):
    g = np.asarray(grid, dtype=int)
    grey_comps = _components(g == GREY)
    markers = {tuple(map(int, pt)) for pt in np.argwhere(g == MAROON)}
    mirrors = []
    by_cell = {}

    for comp in grey_comps:
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        cells = set(comp)

        if len(rows) == 1:
            orientation = "h"
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert len(comp) == c1 - c0 + 1
            assert all((row, c) in cells for c in range(c0, c1 + 1))
            adjacent = []
            for mr, mc in markers:
                if (mr == row - 1 or mr == row + 1) and c0 <= mc <= c1 and (row, mc) in cells:
                    adjacent.append((mr, mc))
            assert len(adjacent) == 1
            marker = adjacent[0]
            marker_offset = (marker[0] - row, 0)
            assert marker_offset in (UP, DOWN)
            length = c1 - c0 + 1
            mirror = {
                "orientation": orientation,
                "row": row,
                "c0": c0,
                "c1": c1,
                "length": length,
                "marker": marker,
                "reflect_from": (-marker_offset[0], 0),
                "cells": tuple(sorted(comp)),
            }
        else:
            assert len(cols) == 1
            orientation = "v"
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert len(comp) == r1 - r0 + 1
            assert all((r, col) in cells for r in range(r0, r1 + 1))
            adjacent = []
            for mr, mc in markers:
                if (mc == col - 1 or mc == col + 1) and r0 <= mr <= r1 and (mr, col) in cells:
                    adjacent.append((mr, mc))
            assert len(adjacent) == 1
            marker = adjacent[0]
            marker_offset = (0, marker[1] - col)
            assert marker_offset in (LEFT, RIGHT)
            length = r1 - r0 + 1
            mirror = {
                "orientation": orientation,
                "col": col,
                "r0": r0,
                "r1": r1,
                "length": length,
                "marker": marker,
                "reflect_from": (0, -marker_offset[1]),
                "cells": tuple(sorted(comp)),
            }

        idx = len(mirrors)
        mirrors.append(mirror)
        for cell in comp:
            by_cell[cell] = idx

    return mirrors, by_cell


def _mirror_response(mirror, hit_cell, direction):
    dr, dc = direction
    if mirror["orientation"] == "h":
        if dc != 0:
            return "parallel", None
        if (dr, dc) != mirror["reflect_from"]:
            return "blocked_nonreflect", None
        idx = int(hit_cell[1] - mirror["c0"])
        if idx < mirror["length"] // 2:
            return "reflect", LEFT
        if idx > mirror["length"] // 2:
            return "reflect", RIGHT
        return "center_hit", None

    if dr != 0:
        return "parallel", None
    if (dr, dc) != mirror["reflect_from"]:
        return "blocked_nonreflect", None
    idx = int(hit_cell[0] - mirror["r0"])
    if idx < mirror["length"] // 2:
        return "reflect", UP
    if idx > mirror["length"] // 2:
        return "reflect", DOWN
    return "center_hit", None


def _trace_ray(grid, seeds_by_pos, markers, mirrors, mirror_by_cell, start, color, direction):
    g = np.asarray(grid, dtype=int)
    r, c = start
    dr, dc = direction
    last_black = None
    reflected = False
    touched = []

    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            return {
                "start": start,
                "color": color,
                "direction": direction,
                "status": "boundary",
                "endpoint": last_black,
                "reflected": reflected,
                "touched_mirrors": tuple(touched),
            }

        cell = int(g[nr, nc])
        if cell == BLACK:
            r, c = nr, nc
            last_black = (r, c)
            continue

        if cell == GREY:
            mid = int(mirror_by_cell[(nr, nc)])
            touched.append(mid)
            if reflected:
                return {
                    "start": start,
                    "color": color,
                    "direction": direction,
                    "status": "second_mirror",
                    "endpoint": last_black,
                    "reflected": True,
                    "touched_mirrors": tuple(touched),
                }

            response, new_dir = _mirror_response(mirrors[mid], (nr, nc), (dr, dc))
            if response == "reflect":
                reflected = True
                dr, dc = new_dir
                continue
            if response == "blocked_nonreflect":
                return {
                    "start": start,
                    "color": color,
                    "direction": direction,
                    "status": "blocked_nonreflect",
                    "endpoint": None,
                    "reflected": False,
                    "touched_mirrors": tuple(touched),
                }
            return {
                "start": start,
                "color": color,
                "direction": direction,
                "status": response,
                "endpoint": None,
                "reflected": False,
                "touched_mirrors": tuple(touched),
            }

        if (nr, nc) in markers:
            return {
                "start": start,
                "color": color,
                "direction": direction,
                "status": "marker",
                "endpoint": last_black,
                "reflected": reflected,
                "touched_mirrors": tuple(touched),
            }

        if (nr, nc) in seeds_by_pos:
            return {
                "start": start,
                "color": color,
                "direction": direction,
                "status": "seed",
                "endpoint": last_black,
                "reflected": reflected,
                "touched_mirrors": tuple(touched),
            }

        return {
            "start": start,
            "color": color,
            "direction": direction,
            "status": "other_obstacle",
            "endpoint": last_black,
            "reflected": reflected,
            "touched_mirrors": tuple(touched),
        }


def _simulate_all(grid):
    g = np.asarray(grid, dtype=int)
    mirrors, mirror_by_cell = _detect_mirrors(g)
    seeds = _seed_cells(g)
    seeds_by_pos = {(r, c): color for r, c, color in seeds}
    markers = {tuple(map(int, pt)) for pt in np.argwhere(g == MAROON)}

    traces = []
    out = g.copy()
    for r, c, color in seeds:
        for direction in DIRS:
            trace = _trace_ray(
                g,
                seeds_by_pos=seeds_by_pos,
                markers=markers,
                mirrors=mirrors,
                mirror_by_cell=mirror_by_cell,
                start=(r, c),
                color=color,
                direction=direction,
            )
            traces.append(trace)
            if trace["endpoint"] is not None:
                er, ec = trace["endpoint"]
                out[er, ec] = color

    return out, traces, mirrors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored seed's four rays through one-sided grey mirrors and place same-color copies at the final stopping cells.

    Input structure:
    - The grid has a black background, several singleton colored seed pixels, and several grey straight mirror segments.
    - Each mirror is a horizontal or vertical contiguous grey line segment with one adjacent maroon marker that indicates the reflective side.
    - Seed colors are distinct and the mirrors, markers, and seeds remain present in the output.

    Rule:
    - From every seed, cast one ray in each cardinal direction through black cells only.
    - On the first mirror hit, a ray reflects by 90 degrees only if it arrives from that mirror's maroon-marked side; otherwise that ray is absorbed and creates no copy.
    - A reflected ray can continue only once more until it reaches the boundary, a seed, or another mirror; the output paints only the last black cell reached before that final stop.

    Key insight:
    - The maroon pixel does not mark a target; it marks which side of the grey segment behaves like a mirror, so the solver must infer both valid reflection side and reflected direction from the hit position along the segment.
    """
    out, _traces, _mirrors = _simulate_all(grid)
    return out


def _chebyshev_ok(existing, proposed):
    proposed = set(proposed)
    for r, c in proposed:
        if (r, c) in existing:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in existing and (rr, cc) not in proposed:
                    return False
    return True


def _ray_cells_to_impact(start, direction, distance):
    r, c = start
    dr, dc = direction
    cells = []
    for step in range(1, distance):
        cells.append((r + dr * step, c + dc * step))
    return cells


def _half_indices(length, want_left_or_up):
    mid = length // 2
    if want_left_or_up:
        return tuple(range(mid))
    return tuple(range(mid + 1, length))


def _open_run(shape, start, direction, occupied, limit=None):
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return steps
        if (r, c) in occupied:
            return steps
        steps += 1
        if limit is not None and steps >= limit:
            return steps


def _plan_targeted_mirror(rng, shape, start, direction, kind, occupied, reserved):
    h, w = shape
    sr, sc = start
    dr, dc = direction
    reflective = kind == "reflect"

    lengths = list(MIRROR_LENGTHS)
    rng.shuffle(lengths)
    distances = list(range(2, 7))
    rng.shuffle(distances)
    halves = [True, False]
    rng.shuffle(halves)

    for length in lengths:
        for distance in distances:
            for want_low_half in halves:
                if direction in (UP, DOWN):
                    mr = sr + dr * distance
                    idx_choices = list(_half_indices(length, want_low_half))
                    rng.shuffle(idx_choices)
                    for idx in idx_choices:
                        c0 = sc - idx
                        c1 = c0 + length - 1
                        midpoint = c0 + length // 2
                        marker_offset = (-dr, 0) if reflective else (dr, 0)
                        marker = (mr + marker_offset[0], midpoint)
                        segment = {(mr, c) for c in range(c0, c1 + 1)}
                        proposed = set(segment) | {marker}
                        corridor = set(_ray_cells_to_impact(start, direction, distance))
                        pre_contact = (mr - dr, sc)
                        reflected_dir = LEFT if want_low_half else RIGHT

                        if c0 < 0 or c1 >= w or not _in_bounds(shape, *marker):
                            continue
                        if not all(_in_bounds(shape, r, c) for r, c in proposed):
                            continue
                        if corridor & occupied:
                            continue
                        if proposed & reserved:
                            continue
                        if not _chebyshev_ok(occupied, proposed):
                            continue
                        if reflective and _open_run(shape, pre_contact, reflected_dir, occupied | proposed, limit=2) < 2:
                            continue
                        return {
                            "segment": segment,
                            "marker": marker,
                            "reserved": corridor,
                            "direction": direction,
                        }
                else:
                    mc = sc + dc * distance
                    idx_choices = list(_half_indices(length, want_low_half))
                    rng.shuffle(idx_choices)
                    for idx in idx_choices:
                        r0 = sr - idx
                        r1 = r0 + length - 1
                        midpoint = r0 + length // 2
                        marker_offset = (0, -dc) if reflective else (0, dc)
                        marker = (midpoint, mc + marker_offset[1])
                        segment = {(r, mc) for r in range(r0, r1 + 1)}
                        proposed = set(segment) | {marker}
                        corridor = set(_ray_cells_to_impact(start, direction, distance))
                        pre_contact = (sr, mc - dc)
                        reflected_dir = UP if want_low_half else DOWN

                        if r0 < 0 or r1 >= h or not _in_bounds(shape, *marker):
                            continue
                        if not all(_in_bounds(shape, r, c) for r, c in proposed):
                            continue
                        if corridor & occupied:
                            continue
                        if proposed & reserved:
                            continue
                        if not _chebyshev_ok(occupied, proposed):
                            continue
                        if reflective and _open_run(shape, pre_contact, reflected_dir, occupied | proposed, limit=2) < 2:
                            continue
                        return {
                            "segment": segment,
                            "marker": marker,
                            "reserved": corridor,
                            "direction": direction,
                        }

    return None


def _place_seeds(rng, shape, count):
    h, w = shape
    colors = list(rng.choice(SEED_COLOR_POOL, size=count, replace=False))
    seeds = []
    occupied = set()

    for color in colors:
        placed = False
        for _ in range(300):
            r = int(rng.integers(2, h - 2))
            c = int(rng.integers(2, w - 2))
            if any(r == rr or c == cc for rr, cc, _ in seeds):
                continue
            if any(max(abs(r - rr), abs(c - cc)) < 4 for rr, cc, _ in seeds):
                continue
            if not _chebyshev_ok(occupied, {(r, c)}):
                continue
            seeds.append((r, c, int(color)))
            occupied.add((r, c))
            placed = True
            break
        if not placed:
            return None

    return seeds


def _validate_example(grid):
    try:
        out, traces, mirrors = _simulate_all(grid)
    except AssertionError:
        return False

    seeds = _seed_cells(grid)
    if not (2 <= len(seeds) <= 4):
        return False
    if len(mirrors) < 2:
        return False
    if not np.array_equal(out.shape, grid.shape):
        return False

    statuses = {trace["status"] for trace in traces}
    if "blocked_nonreflect" not in statuses:
        return False
    if not any(trace["reflected"] and trace["endpoint"] is not None for trace in traces):
        return False
    if not any((not trace["reflected"]) and trace["status"] in {"boundary", "seed"} and trace["endpoint"] is not None for trace in traces):
        return False
    if statuses & {"marker", "parallel", "center_hit", "other_obstacle"}:
        return False

    occupied = {tuple(map(int, pt)) for pt in np.argwhere(grid != BLACK)}
    endpoints = set()
    mirror_hits = [0] * len(mirrors)
    for trace in traces:
        for mid in trace["touched_mirrors"]:
            mirror_hits[int(mid)] += 1
        endpoint = trace["endpoint"]
        if endpoint is None:
            continue
        if endpoint in endpoints:
            return False
        endpoints.add(endpoint)
        if endpoint in occupied:
            return False

    if any(hit == 0 for hit in mirror_hits):
        return False
    if len(endpoints) < 4:
        return False
    return True


def _gen_attempt(rng):
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BLACK, dtype=int)

    seed_count = int(rng.integers(2, 5))
    seeds = _place_seeds(rng, grid.shape, seed_count)
    if seeds is None:
        return None

    occupied = {(r, c) for r, c, _ in seeds}
    reserved = set()
    for r, c, color in seeds:
        grid[r, c] = color

    extra = int(rng.integers(0, 3))
    kinds = ["reflect", "block"]
    for _ in range(extra):
        kinds.append(str(rng.choice(["reflect", "block"])))
    rng.shuffle(kinds)

    used_launches = set()
    for kind in kinds:
        placed = False
        seed_order = list(range(len(seeds)))
        rng.shuffle(seed_order)
        for si in seed_order:
            dirs = list(DIRS)
            rng.shuffle(dirs)
            for direction in dirs:
                key = (si, direction)
                if key in used_launches:
                    continue
                plan = _plan_targeted_mirror(
                    rng,
                    shape=grid.shape,
                    start=(seeds[si][0], seeds[si][1]),
                    direction=direction,
                    kind=kind,
                    occupied=occupied,
                    reserved=reserved,
                )
                if plan is None:
                    continue

                for r, c in plan["segment"]:
                    grid[r, c] = GREY
                mr, mc = plan["marker"]
                grid[mr, mc] = MAROON
                occupied |= set(plan["segment"])
                occupied.add(plan["marker"])
                reserved |= set(plan["reserved"])
                used_launches.add(key)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray example")

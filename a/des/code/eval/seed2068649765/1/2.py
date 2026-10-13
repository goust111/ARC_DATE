import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = [UP, RIGHT, DOWN, LEFT]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}

GEN_ATTEMPTS = 300
PATH_BUILD_ATTEMPTS = 120
PATH_PLACE_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(max(counts, key=counts.get))


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(x) for x in exclude}
    return [int(c) for c in np.unique(grid) if int(c) not in excluded]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rotate_left(d: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_IDX[d] - 1) % 4]


def _rotate_right(d: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(DIR_TO_IDX[d] + 1) % 4]


def _turn_kind(d_in: tuple[int, int], d_out: tuple[int, int]) -> str:
    delta = (DIR_TO_IDX[d_out] - DIR_TO_IDX[d_in]) % 4
    assert delta in (1, 3), "corners must be quarter turns"
    return "right" if delta == 1 else "left"


def _trace_order_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    coords = {tuple(map(int, p)) for p in np.argwhere(grid == int(color))}
    assert coords, "path color must be present"

    degrees: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        deg = sum((r + dr, c + dc) in coords for dr, dc in DIRS)
        assert 1 <= deg <= 2, "path must be a single-width non-branching polyline"
        degrees[(r, c)] = int(deg)
        if deg == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in endpoints:
        nr, nc = next((r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in coords)
        d = (nr - r, nc - c)
        if (nr + d[0], nc + d[1]) in coords:
            start_candidates.append(((r, c), d))

    assert len(start_candidates) == 1, "expected a unique 2-cell start marker"
    start, start_dir = start_candidates[0]

    ordered = [start]
    prev: tuple[int, int] | None = None
    curr = start
    while True:
        nexts = [
            (curr[0] + dr, curr[1] + dc)
            for dr, dc in DIRS
            if (curr[0] + dr, curr[1] + dc) in coords and (curr[0] + dr, curr[1] + dc) != prev
        ]
        if not nexts:
            break
        assert len(nexts) == 1, "path traversal must be unambiguous"
        nxt = nexts[0]
        ordered.append(nxt)
        prev, curr = curr, nxt

    assert len(ordered) == len(coords), "path must be a single connected component"
    assert ordered[1] == (start[0] + start_dir[0], start[1] + start_dir[1])
    return ordered


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()
    meta = {
        "paths": 0,
        "rays": 0,
        "truncated": 0,
        "left_turns": 0,
        "right_turns": 0,
    }

    for color in _colors_present(g, exclude={bg}):
        ordered = _trace_order_for_color(g, color)
        meta["paths"] += 1
        current_dir = (ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1])
        segment_len = 0

        for i in range(1, len(ordered)):
            step_dir = (ordered[i][0] - ordered[i - 1][0], ordered[i][1] - ordered[i - 1][1])
            assert step_dir == current_dir
            segment_len += 1

            if i == len(ordered) - 1:
                break

            next_dir = (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
            if next_dir == step_dir:
                continue

            turn = _turn_kind(step_dir, next_dir)
            ray_dir = _rotate_right(step_dir) if turn == "left" else _rotate_left(step_dir)
            corner_r, corner_c = ordered[i]

            drawn = 0
            for k in range(1, segment_len + 1):
                rr = corner_r + ray_dir[0] * k
                cc = corner_c + ray_dir[1] * k
                if not _in_bounds(g.shape, rr, cc) or int(out[rr, cc]) != int(bg):
                    break
                out[rr, cc] = int(color)
                drawn += 1

            meta["rays"] += 1
            meta["truncated"] += int(drawn < segment_len)
            meta["left_turns"] += int(turn == "left")
            meta["right_turns"] += int(turn == "right")
            segment_len = 0
            current_dir = next_dir

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk each colored orthogonal path from its 2-cell start marker.

    At every corner, determine whether the walk makes a left or right turn.
    Emit a same-color ray from the outside of that bend, with length equal to
    the just-traversed straight segment length excluding the corner cell itself.
    Rays stop early at boundaries or occupied cells.
    """
    return _solve(grid, return_meta=False)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_path_spec(rng: random.Random) -> dict[str, object] | None:
    for _ in range(PATH_BUILD_ATTEMPTS):
        num_segments = rng.randint(2, 5)
        dirs = [rng.choice(DIRS)]
        for _ in range(num_segments - 1):
            if rng.random() < 0.5:
                dirs.append(_rotate_left(dirs[-1]))
            else:
                dirs.append(_rotate_right(dirs[-1]))

        steps = [rng.randint(2, 4)]
        for _ in range(num_segments - 2):
            steps.append(rng.randint(2, 4))
        steps.append(1)

        cells = [(0, 0)]
        occupied = {(0, 0)}
        ok = True
        for d, step_count in zip(dirs, steps):
            for _ in range(step_count):
                nr = cells[-1][0] + d[0]
                nc = cells[-1][1] + d[1]
                if (nr, nc) in occupied:
                    ok = False
                    break
                for adr, adc in DIRS:
                    adj = (nr + adr, nc + adc)
                    if adj in occupied and adj != cells[-1]:
                        ok = False
                        break
                if not ok:
                    break
                cells.append((nr, nc))
                occupied.add((nr, nc))
            if not ok:
                break
        if not ok:
            continue

        rays: list[dict[str, object]] = []
        ray_cells: set[tuple[int, int]] = set()
        prefix = 0
        for i in range(num_segments - 1):
            prefix += steps[i]
            corner = cells[prefix]
            turn = _turn_kind(dirs[i], dirs[i + 1])
            ray_dir = _rotate_right(dirs[i]) if turn == "left" else _rotate_left(dirs[i])
            one_ray = []
            for k in range(1, steps[i] + 1):
                rr = corner[0] + ray_dir[0] * k
                cc = corner[1] + ray_dir[1] * k
                if (rr, cc) in occupied or (rr, cc) in ray_cells:
                    ok = False
                    break
                one_ray.append((rr, cc))
            if not ok:
                break
            ray_cells.update(one_ray)
            rays.append(
                {
                    "corner": corner,
                    "dir": ray_dir,
                    "length": steps[i],
                    "cells": tuple(one_ray),
                }
            )
        if not ok:
            continue

        output_cells = set(occupied) | ray_cells
        r0, c0, r1, c1 = _bbox(output_cells)
        if (r1 - r0 + 1) > 11 or (c1 - c0 + 1) > 11:
            continue

        return {
            "path_cells": tuple(cells),
            "path_set": set(occupied),
            "rays": tuple(rays),
            "path_bbox": _bbox(set(occupied)),
            "output_bbox": (r0, c0, r1, c1),
        }
    return None


def _future_cells_for_shift(
    spec: dict[str, object], shape: tuple[int, int], shift_r: int, shift_c: int
) -> tuple[set[tuple[int, int]] | None, set[tuple[int, int]] | None, list[int]]:
    path_cells = {
        (r + shift_r, c + shift_c)
        for r, c in spec["path_cells"]  # type: ignore[index]
    }
    if not all(_in_bounds(shape, r, c) for r, c in path_cells):
        return None, None, []

    future = set(path_cells)
    drawn_counts: list[int] = []
    for ray in spec["rays"]:  # type: ignore[index]
        corner_r, corner_c = ray["corner"]  # type: ignore[index]
        ray_dir = ray["dir"]  # type: ignore[index]
        drawn = 0
        for k in range(1, int(ray["length"]) + 1):  # type: ignore[index]
            rr = int(corner_r) + shift_r + int(ray_dir[0]) * k
            cc = int(corner_c) + shift_c + int(ray_dir[1]) * k
            if not _in_bounds(shape, rr, cc):
                break
            future.add((rr, cc))
            drawn += 1
        drawn_counts.append(drawn)
    return path_cells, future, drawn_counts


def _reserve_with_gap(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    reserved = set(cells)
    for r, c in list(cells):
        for dr, dc in DIRS:
            rr = r + dr
            cc = c + dc
            if _in_bounds(shape, rr, cc):
                reserved.add((rr, cc))
    return reserved


def _place_full_path(
    spec: dict[str, object],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    r0, c0, r1, c1 = spec["output_bbox"]  # type: ignore[index]
    row_lo = -int(r0)
    row_hi = shape[0] - 1 - int(r1)
    col_lo = -int(c0)
    col_hi = shape[1] - 1 - int(c1)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PATH_PLACE_ATTEMPTS):
        shift_r = rng.randint(row_lo, row_hi)
        shift_c = rng.randint(col_lo, col_hi)
        path_cells, future, drawn = _future_cells_for_shift(spec, shape, shift_r, shift_c)
        assert path_cells is not None and future is not None
        if future & forbidden:
            continue
        if any(drawn_i != int(ray["length"]) for drawn_i, ray in zip(drawn, spec["rays"])):  # type: ignore[index]
            continue
        return path_cells, future
    return None


def _place_truncated_path(
    spec: dict[str, object],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    rays = list(spec["rays"])  # type: ignore[index]
    candidate_ids = [i for i, ray in enumerate(rays) if int(ray["length"]) >= 2]
    rng.shuffle(candidate_ids)
    if not candidate_ids:
        return None

    path_r0, path_c0, path_r1, path_c1 = spec["path_bbox"]  # type: ignore[index]
    for ray_idx in candidate_ids:
        ray = rays[ray_idx]
        corner_r, corner_c = ray["corner"]  # type: ignore[index]
        ray_dir = ray["dir"]  # type: ignore[index]
        length = int(ray["length"])
        for _ in range(PATH_PLACE_ATTEMPTS):
            avail = rng.randint(1, length - 1)
            if ray_dir == UP:
                shift_r = avail - int(corner_r)
            elif ray_dir == DOWN:
                shift_r = shape[0] - 1 - avail - int(corner_r)
            elif ray_dir == LEFT:
                shift_c = avail - int(corner_c)
            else:
                shift_c = shape[1] - 1 - avail - int(corner_c)

            if ray_dir in (UP, DOWN):
                if not (-int(path_r0) <= shift_r <= shape[0] - 1 - int(path_r1)):
                    continue
                col_lo = -int(path_c0)
                col_hi = shape[1] - 1 - int(path_c1)
                if col_lo > col_hi:
                    continue
                shift_c = rng.randint(col_lo, col_hi)
            else:
                if not (-int(path_c0) <= shift_c <= shape[1] - 1 - int(path_c1)):
                    continue
                row_lo = -int(path_r0)
                row_hi = shape[0] - 1 - int(path_r1)
                if row_lo > row_hi:
                    continue
                shift_r = rng.randint(row_lo, row_hi)

            path_cells, future, drawn = _future_cells_for_shift(spec, shape, shift_r, shift_c)
            assert path_cells is not None and future is not None
            if future & forbidden:
                continue
            if not (1 <= drawn[ray_idx] < length):
                continue
            return path_cells, future
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    path_count = rng.choice([1, 2, 3])
    need_truncation = bool(rng.randrange(2))

    min_size = 15 + 3 * (path_count - 1)
    max_size = 21 + 3 * (path_count - 1)
    h = rng.randint(min_size, min(max_size, 28))
    w = rng.randint(min_size, min(max_size, 28))
    colors = rng.sample(PALETTE, path_count)

    grid = np.full((h, w), BG, dtype=int)
    reserved_future: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    modes = ["truncated"] if need_truncation else []
    while len(modes) < path_count:
        modes.append("full")

    for color, mode in zip(colors, modes):
        placed = False
        for _ in range(PATH_BUILD_ATTEMPTS):
            spec = _build_path_spec(rng)
            if spec is None:
                continue
            if mode == "truncated":
                placement = _place_truncated_path(spec, grid.shape, forbidden, rng)
            else:
                placement = _place_full_path(spec, grid.shape, forbidden, rng)
            if placement is None:
                continue

            path_cells, future = placement
            for r, c in path_cells:
                grid[r, c] = int(color)
            reserved_future |= future
            forbidden = _reserve_with_gap(reserved_future, grid.shape)
            placed = True
            break
        if not placed:
            return None

    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return None

    if meta["paths"] != path_count or meta["rays"] < path_count:
        return None
    if need_truncation and meta["truncated"] == 0:
        return None
    if not need_truncation and meta["truncated"] != 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

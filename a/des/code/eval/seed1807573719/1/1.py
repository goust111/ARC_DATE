from __future__ import annotations

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
MAX_GENERATION_ATTEMPTS = 500
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 23)
TREE_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_RANGE = (2, 6)
MIN_TRUNK_LEN = 4
LONG_TRUNK_LEN = 7
MIN_SEED_CHEBYSHEV_GAP = 1
MAX_SEGMENT_PLACEMENT_ATTEMPTS = 80
MAX_SEED_PLACEMENT_ATTEMPTS = 120
MAX_SIDE_ASSIGNMENT_ATTEMPTS = 60
MAX_MASK_DENSITY = 0.48

_RNG = np.random.default_rng()


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_boundary(h: int, w: int, r: int, c: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _perpendicular_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _analyze_color(grid: np.ndarray, color: int) -> dict:
    h, w = grid.shape
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"color {color} is absent"

    boundary_mask = (
        (coords[:, 0] == 0)
        | (coords[:, 0] == h - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == w - 1)
    )
    boundary = coords[boundary_mask]
    interior = coords[~boundary_mask]

    assert len(interior) == 1, f"color {color} must have exactly one interior seed"
    assert len(boundary) >= 2, f"color {color} must have a border segment of length >= 2"
    seed = tuple(map(int, interior[0]))

    side_candidates: list[str] = []
    if np.all(boundary[:, 0] == 0):
        side_candidates.append("top")
    if np.all(boundary[:, 0] == h - 1):
        side_candidates.append("bottom")
    if np.all(boundary[:, 1] == 0):
        side_candidates.append("left")
    if np.all(boundary[:, 1] == w - 1):
        side_candidates.append("right")

    assert len(side_candidates) == 1, f"color {color} border must lie on one edge"
    side = side_candidates[0]

    if side in {"top", "bottom"}:
        line = np.sort(boundary[:, 1])
    else:
        line = np.sort(boundary[:, 0])
    assert int(line[-1] - line[0] + 1) == len(line), f"color {color} border must be contiguous"

    return {"color": int(color), "seed": seed, "side": side}


def _grow_tree(
    out: np.ndarray,
    color: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    h, w = out.shape
    trunk: list[tuple[int, int]] = []
    painted = {tuple(map(int, seed))}
    branch_truncated = False

    r, c = seed
    dr, dc = direction
    while True:
        nr, nc = int(r + dr), int(c + dc)
        if not _in_bounds(h, w, nr, nc):
            break
        if int(out[nr, nc]) == int(color):
            break
        out[nr, nc] = int(color)
        trunk.append((nr, nc))
        painted.add((nr, nc))
        r, c = nr, nc
        if _is_boundary(h, w, nr, nc):
            break

    for idx, (ar, ac) in enumerate(trunk, start=1):
        if (idx - 1) % 3 != 0:
            continue
        for bdr, bdc in _perpendicular_dirs(direction):
            painted_this_branch = 0
            for dist in range(1, idx + 1):
                nr, nc = int(ar + bdr * dist), int(ac + bdc * dist)
                if not _in_bounds(h, w, nr, nc):
                    branch_truncated = True
                    break
                if int(out[nr, nc]) == int(color):
                    branch_truncated = True
                    break
                out[nr, nc] = int(color)
                painted.add((nr, nc))
                painted_this_branch += 1
            if painted_this_branch < idx:
                branch_truncated = True

    return {
        "cells": painted,
        "trunk_len": len(trunk),
        "branch_truncated": branch_truncated,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one fractal tree per colored seed.

    Input structure:
    - Background is black (0).
    - Each non-black color appears as one interior seed pixel plus one border segment on a single edge.

    Rule:
    - The matching border segment determines the growth direction: grow away from that edge.
    - Draw the trunk from the seed until the next step would leave the grid or hit a same-color pixel.
    - On trunk indices 1, 4, 7, ... (counting from the first trunk pixel after the seed), draw
      two perpendicular branches of that index length, stopping before out-of-bounds or same-color cells.
    - Process seeds in top-to-bottom, then left-to-right order; later trees overwrite earlier colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background"

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    trees = [_analyze_color(g, color) for color in colors]
    trees.sort(key=lambda tree: (tree["seed"][0], tree["seed"][1]))

    out = g.copy()
    for tree in trees:
        direction = DIR_BY_SIDE[tree["side"]]
        _grow_tree(out, tree["color"], tree["seed"], direction)
    return out


def _paint_segment(grid: np.ndarray, color: int, side: str, start: int, length: int) -> None:
    if side == "top":
        grid[0, start : start + length] = int(color)
    elif side == "bottom":
        grid[-1, start : start + length] = int(color)
    elif side == "left":
        grid[start : start + length, 0] = int(color)
    else:
        grid[start : start + length, -1] = int(color)


def _edge_axis_length(h: int, w: int, side: str) -> int:
    return (w - 2) if side in {"top", "bottom"} else (h - 2)


def _place_side_intervals(axis_len: int, count: int) -> list[tuple[int, int]] | None:
    min_len = SEGMENT_LENGTH_RANGE[0]
    max_len = min(SEGMENT_LENGTH_RANGE[1], axis_len)
    if axis_len < min_len or count * min_len + max(0, count - 1) > axis_len:
        return None

    for _ in range(MAX_SEGMENT_PLACEMENT_ATTEMPTS):
        intervals: list[tuple[int, int]] = []
        ok = True
        for _ in range(count):
            length = int(_RNG.integers(min_len, max_len + 1))
            starts = []
            for start in range(1, axis_len - length + 2):
                end = start + length - 1
                if all(end < prev_start - 1 or start > prev_end + 1 for prev_start, prev_end in intervals):
                    starts.append(start)
            if not starts:
                ok = False
                break
            start = int(starts[int(_RNG.integers(0, len(starts)))])
            intervals.append((start, start + length - 1))
        if ok:
            intervals.sort()
            return intervals
    return None


def _sample_sides(count: int) -> list[str] | None:
    for _ in range(MAX_SIDE_ASSIGNMENT_ATTEMPTS):
        sides = [
            SIDES[int(_RNG.integers(0, 2))],
            SIDES[2 + int(_RNG.integers(0, 2))],
        ]
        while len(sides) < count:
            sides.append(SIDES[int(_RNG.integers(0, len(SIDES)))])
        perm = _RNG.permutation(len(sides))
        shuffled = [sides[int(i)] for i in perm]
        side_counts = {side: shuffled.count(side) for side in SIDES}
        feasible = True
        for side, side_count in side_counts.items():
            axis_len = _edge_axis_length(14, 14, side)
            min_needed = side_count * SEGMENT_LENGTH_RANGE[0] + max(0, side_count - 1)
            if min_needed > axis_len:
                feasible = False
                break
        if feasible:
            return shuffled
    return None


def _seed_candidates(h: int, w: int, side: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            trunk_len = _trunk_len(h, w, (r, c), side)
            if trunk_len >= MIN_TRUNK_LEN:
                candidates.append((r, c))
    return candidates


def _trunk_len(h: int, w: int, seed: tuple[int, int], side: str) -> int:
    r, c = seed
    if side == "top":
        return h - 1 - r
    if side == "bottom":
        return r
    if side == "left":
        return w - 1 - c
    return c


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= MIN_SEED_CHEBYSHEV_GAP


def _place_seeds(h: int, w: int, sides: list[str]) -> list[tuple[int, int]] | None:
    seeds: list[tuple[int, int]] = []
    candidates_by_side = {side: _seed_candidates(h, w, side) for side in SIDES}
    if any(not candidates_by_side[side] for side in set(sides)):
        return None

    order = list(_RNG.permutation(len(sides)))
    seeds = [(-1, -1)] * len(sides)
    chosen: list[tuple[int, int]] = []

    for index in order:
        side = sides[index]
        candidates = candidates_by_side[side].copy()
        _RNG.shuffle(candidates)
        placed = False
        for seed in candidates:
            if any(_too_close(seed, other) for other in chosen):
                continue
            seeds[index] = seed
            chosen.append(seed)
            placed = True
            break
        if not placed:
            return None
    return seeds


def _make_empty_tree_mask(h: int, w: int, seed: tuple[int, int], side: str) -> dict:
    canvas = np.zeros((h, w), dtype=int)
    canvas[seed] = 1
    return _grow_tree(canvas, 1, seed, DIR_BY_SIDE[side])


def _has_overlap(tree_specs: list[dict], h: int, w: int) -> bool:
    ordered = sorted(tree_specs, key=lambda spec: (spec["seed"][0], spec["seed"][1]))
    masks = [_make_empty_tree_mask(h, w, spec["seed"], spec["side"])["cells"] for spec in ordered]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i] & masks[j]:
                return True
    return False


def _scene_has_long_trunk(tree_specs: list[dict], h: int, w: int) -> bool:
    return any(_trunk_len(h, w, spec["seed"], spec["side"]) >= LONG_TRUNK_LEN for spec in tree_specs)


def _scene_has_truncated_branch(tree_specs: list[dict], h: int, w: int) -> bool:
    for spec in tree_specs:
        info = _make_empty_tree_mask(h, w, spec["seed"], spec["side"])
        if info["branch_truncated"]:
            return True
    return False


def _density_ok(out: np.ndarray) -> bool:
    density = float(np.count_nonzero(out != BG)) / float(out.size)
    return density <= MAX_MASK_DENSITY


def _build_scene(h: int, w: int, force_overlap: bool) -> np.ndarray | None:
    tree_count = int(_RNG.integers(TREE_COUNT_RANGE[0], TREE_COUNT_RANGE[1]))
    sides = _sample_sides(tree_count)
    if sides is None:
        return None

    side_counts = {side: sides.count(side) for side in SIDES}
    intervals_by_side: dict[str, list[tuple[int, int]]] = {}
    for side, count in side_counts.items():
        if count == 0:
            intervals_by_side[side] = []
            continue
        intervals = _place_side_intervals(_edge_axis_length(h, w, side), count)
        if intervals is None:
            return None
        intervals_by_side[side] = intervals

    seeds = None
    for _ in range(MAX_SEED_PLACEMENT_ATTEMPTS):
        seeds = _place_seeds(h, w, sides)
        if seeds is not None:
            break
    if seeds is None:
        return None

    colors = _RNG.choice(np.arange(1, 10), size=tree_count, replace=False)
    grid = np.full((h, w), BG, dtype=int)
    side_offsets = {side: 0 for side in SIDES}
    tree_specs: list[dict] = []

    for idx, color in enumerate(colors):
        side = sides[idx]
        seed = seeds[idx]
        start, end = intervals_by_side[side][side_offsets[side]]
        side_offsets[side] += 1
        _paint_segment(grid, int(color), side, int(start), int(end - start + 1))
        grid[seed] = int(color)
        tree_specs.append({"color": int(color), "seed": seed, "side": side})

    if not _scene_has_long_trunk(tree_specs, h, w):
        return None
    if not _scene_has_truncated_branch(tree_specs, h, w):
        return None

    has_overlap = _has_overlap(tree_specs, h, w)
    if force_overlap != has_overlap:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not _density_ok(out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
        w = int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
        force_overlap = bool(int(_RNG.integers(0, 2)))
        grid = _build_scene(h, w, force_overlap)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC tree-growth input")

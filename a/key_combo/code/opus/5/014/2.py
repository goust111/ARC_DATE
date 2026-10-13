from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


BG = 0
OBSTACLE = 5
RING_PALETTE = (1, 2, 3, 4, 6, 7)
RING_ORDER_CW = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
)
CORNER_ORDER_CW = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_ORDER_CCW = ((0, 1), (1, 0), (2, 1), (1, 2))

GRID_MIN = 13
GRID_MAX = 19
RING_COUNT_MIN = 4
RING_COUNT_MAX = 6
GEN_RETRIES = 400


def _apply_cycle(out_patch: np.ndarray, src_patch: np.ndarray, positions: tuple[tuple[int, int], ...]) -> None:
    values = [int(src_patch[r, c]) for r, c in positions]
    for idx, (r, c) in enumerate(positions):
        prev = values[(idx - 1) % len(values)]
        out_patch[r, c] = prev


def _classify_ring_patch(patch: np.ndarray, bg: int) -> str | None:
    if patch.shape != (3, 3):
        return None

    if any(int(patch[r, c]) in (bg, OBSTACLE) for r, c in CORNER_ORDER_CW):
        return None

    top = int(patch[0, 1])
    right = int(patch[1, 2])
    bottom = int(patch[2, 1])
    left = int(patch[1, 0])
    center = int(patch[1, 1])
    sides = (top, right, bottom, left)
    if any(side == bg for side in sides):
        return None
    if center not in (bg, OBSTACLE):
        return None

    if top == OBSTACLE and bottom == OBSTACLE and right != OBSTACLE and left != OBSTACLE:
        return "vertical"
    if left == OBSTACLE and right == OBSTACLE and top != OBSTACLE and bottom != OBSTACLE:
        return "horizontal"
    if center != bg:
        return None
    if any(side == OBSTACLE for side in sides):
        return None
    return "complete"


def _find_rings(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    rings: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(g.shape[0] - 2):
        for c0 in range(g.shape[1] - 2):
            kind = _classify_ring_patch(g[r0 : r0 + 3, c0 : c0 + 3], bg)
            if kind is None:
                continue
            if (r0, c0) in seen:
                continue
            seen.add((r0, c0))
            rings.append(
                {
                    "top_left": (r0, c0),
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "kind": kind,
                }
            )
    return sorted(rings, key=lambda ring: ring["top_left"])


def _transform_complete_ring(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    _apply_cycle(out, patch, CORNER_ORDER_CW)
    _apply_cycle(out, patch, SIDE_ORDER_CCW)
    return out


def _transform_split_ring(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "horizontal":
        blocked = {(1, 0), (1, 2)}
    elif kind == "vertical":
        blocked = {(0, 1), (2, 1)}
    else:
        raise AssertionError(kind)

    out = patch.copy()
    cycle = tuple(pos for pos in RING_ORDER_CW if pos not in blocked)
    _apply_cycle(out, patch, cycle)
    return out


def _transform_ring_patch(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "complete":
        return _transform_complete_ring(patch)
    return _transform_split_ring(patch, kind)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate colors around 3x3 rings, but collapse split rings from two four-cycles into one six-cycle.

    Input structure:
    - The grid has a black background, several separated 3x3 ring objects, and one or two grey straight obstacle segments.
    - A complete ring has eight colored outer cells around a black center.
    - A split ring has grey obstacle cells on either the middle row or the middle column, breaking the ring across that axis.

    Rule:
    - Detect every 3x3 ring footprint.
    - If the ring is complete, cycle its corner colors one step clockwise and its side-center colors one step counterclockwise.
    - If the ring is split by a grey horizontal or vertical obstacle, keep the obstacle cells fixed and cycle the other six ring cells one step clockwise along the remaining perimeter order.
    - Keep obstacle pixels outside the rings and all background pixels unchanged.

    Key insight:
    - Grey obstacles do not merely recolor two cells; they change the ring topology from two independent 4-cycles to one connected 6-cycle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    rings = _find_rings(g, bg=bg)
    assert rings, "expected at least one ring"

    for ring in rings:
        r0, c0 = ring["top_left"]
        patch = g[r0 : r0 + 3, c0 : c0 + 3]
        paste(out, _transform_ring_patch(patch, ring["kind"]), r0, c0)

    return out


def _line_candidate(shape: tuple[int, int], orientation: str, targets: tuple[tuple[int, int], ...]) -> dict | None:
    h, w = shape
    if orientation == "horizontal":
        row = targets[0][0]
        cols = sorted(c for _r, c in targets)
        rr = row + 1
        start = cols[0] - 1
        end = cols[-1] + 3
        if start < 0 or end >= w:
            return None
        cells = tuple((rr, c) for c in range(start, end + 1))
        allowed = {
            (r + 1, c + dc)
            for r, c in targets
            for dc in range(3)
        }
        kind = "horizontal"
    elif orientation == "vertical":
        col = targets[0][1]
        rows = sorted(r for r, _c in targets)
        cc = col + 1
        start = rows[0] - 1
        end = rows[-1] + 3
        if start < 0 or end >= h:
            return None
        cells = tuple((r, cc) for r in range(start, end + 1))
        allowed = {
            (r + dr, c + 1)
            for r, c in targets
            for dr in range(3)
        }
        kind = "vertical"
    else:
        raise AssertionError(orientation)

    return {
        "orientation": kind,
        "targets": tuple(sorted(targets)),
        "cells": cells,
        "allowed": allowed,
    }


def _candidate_is_valid(grid: np.ndarray, candidate: dict) -> bool:
    for r, c in candidate["cells"]:
        value = int(grid[r, c])
        if value == BG:
            continue
        if (r, c) not in candidate["allowed"]:
            return False
    return True


def _enumerate_candidates(grid: np.ndarray, positions: list[tuple[int, int]]) -> list[dict]:
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in positions:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    candidates: list[dict] = []
    for row, col_values in rows.items():
        ordered = sorted(col_values)
        for width in (2, 1):
            for i in range(len(ordered) - width + 1):
                targets = tuple((row, ordered[j]) for j in range(i, i + width))
                candidate = _line_candidate(grid.shape, "horizontal", targets)
                if candidate is not None and _candidate_is_valid(grid, candidate):
                    candidates.append(candidate)

    for col, row_values in cols.items():
        ordered = sorted(row_values)
        for height in (2, 1):
            for i in range(len(ordered) - height + 1):
                targets = tuple((ordered[j], col) for j in range(i, i + height))
                candidate = _line_candidate(grid.shape, "vertical", targets)
                if candidate is not None and _candidate_is_valid(grid, candidate):
                    candidates.append(candidate)

    return candidates


def _compatible_pair(a: dict, b: dict, ring_count: int) -> bool:
    targets_a = set(a["targets"])
    targets_b = set(b["targets"])
    if targets_a & targets_b:
        return False
    if len(targets_a | targets_b) >= ring_count:
        return False
    return set(a["cells"]).isdisjoint(b["cells"])


def _make_ring_patch() -> np.ndarray:
    while True:
        patch = np.full((3, 3), BG, dtype=int)
        unique_count = random.randint(4, len(RING_PALETTE))
        chosen = random.sample(RING_PALETTE, unique_count)
        values = chosen + [random.choice(chosen) for _ in range(8 - unique_count)]
        random.shuffle(values)
        for (r, c), value in zip(RING_ORDER_CW, values):
            patch[r, c] = value

        split_h = patch.copy()
        split_h[1, :] = OBSTACLE
        split_v = patch.copy()
        split_v[:, 1] = OBSTACLE
        if np.array_equal(_transform_complete_ring(patch), patch):
            continue
        if np.array_equal(_transform_split_ring(split_h, "horizontal"), split_h):
            continue
        if np.array_equal(_transform_split_ring(split_v, "vertical"), split_v):
            continue
        return patch


def _slot_values(limit: int) -> list[int]:
    return list(range(1, limit - 2, 4))


def _sample_positions() -> tuple[tuple[int, int], list[tuple[int, int]]]:
    while True:
        rows = random.randint(GRID_MIN, GRID_MAX)
        cols = random.randint(GRID_MIN, GRID_MAX)
        row_slots = _slot_values(rows)
        col_slots = _slot_values(cols)
        all_slots = [(r, c) for r in row_slots for c in col_slots]
        count = random.randint(RING_COUNT_MIN, RING_COUNT_MAX)
        if len(all_slots) < count or len(row_slots) < 2 or len(col_slots) < 2:
            continue
        return (rows, cols), random.sample(all_slots, count)


def _choose_obstacles(grid: np.ndarray, positions: list[tuple[int, int]]) -> list[dict]:
    candidates = _enumerate_candidates(grid, positions)
    assert candidates, "expected at least one valid obstacle candidate"

    multi = [cand for cand in candidates if len(cand["targets"]) == 2]
    single = [cand for cand in candidates if len(cand["targets"]) == 1]
    pair_candidates = [
        (a, b)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if _compatible_pair(a, b, len(positions))
    ]
    mixed_pairs = [pair for pair in pair_candidates if pair[0]["orientation"] != pair[1]["orientation"]]

    if pair_candidates and random.random() < 0.45:
        pool = mixed_pairs if mixed_pairs and random.random() < 0.7 else pair_candidates
        return list(random.choice(pool))

    weighted = multi * 3 + single
    return [random.choice(weighted)]


def _obstacle_component_ok(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return (r0 == r1 or c0 == c1) and int(component["area"]) >= 5


def _planned_output(grid: np.ndarray, split_kinds: dict[tuple[int, int], str]) -> np.ndarray:
    out = grid.copy()
    for (r0, c0), kind in split_kinds.items():
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        paste(out, _transform_ring_patch(patch, kind), r0, c0)
    return out


def _attempt_generate() -> np.ndarray | None:
    shape, positions = _sample_positions()
    grid = np.full(shape, BG, dtype=int)
    for r0, c0 in positions:
        paste(grid, _make_ring_patch(), r0, c0)

    obstacles = _choose_obstacles(grid, positions)
    split_kinds = {pos: "complete" for pos in positions}
    for obstacle in obstacles:
        for r, c in obstacle["cells"]:
            grid[r, c] = OBSTACLE
        for pos in obstacle["targets"]:
            split_kinds[pos] = obstacle["orientation"]

    detected = _find_rings(grid, bg=BG)
    if len(detected) != len(positions):
        return None

    detected_map = {tuple(ring["top_left"]): ring["kind"] for ring in detected}
    if set(detected_map) != set(positions):
        return None
    if any(detected_map[pos] != split_kinds[pos] for pos in positions):
        return None
    if not any(kind == "complete" for kind in detected_map.values()):
        return None
    if not any(kind != "complete" for kind in detected_map.values()):
        return None

    obstacle_components = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[OBSTACLE])
    if len(obstacle_components) != len(obstacles):
        return None
    if not all(_obstacle_component_ok(component) for component in obstacle_components):
        return None

    expected = _planned_output(grid, detected_map)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split-ring obstacle instance")

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

BORDER_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GAP_SPECS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)

MAX_GEN_RETRIES = 400
RING_PLACEMENT_TRIES = 300
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)
RING_COUNT_RANGE = (1, 4)
ONLINE_DOT_RANGE = (1, 3)
OFFLINE_DOT_RANGE = (0, 2)


def _ring_cells(center: tuple[int, int], gap_offset: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (r + dr, c + dc)
        for dr, dc in BORDER_OFFSETS
        if (dr, dc) != gap_offset
    ]


def _find_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue

            border_vals = [int(g[r + dr, c + dc]) for dr, dc in BORDER_OFFSETS]
            non_bg = [v for v in border_vals if v != BG]
            if len(non_bg) != 7:
                continue
            color = non_bg[0]
            if any(v != color for v in non_bg):
                continue
            if any(v not in (BG, color) for v in border_vals):
                continue

            missing = []
            for name, (dr, dc) in GAP_SPECS:
                if int(g[r + dr, c + dc]) == BG:
                    missing.append((name, dr, dc))
            if len(missing) != 1:
                continue

            name, gap_dr, gap_dc = missing[0]
            ok = True
            for dr, dc in BORDER_OFFSETS:
                expected = BG if (dr, dc) == (gap_dr, gap_dc) else color
                if int(g[r + dr, c + dc]) != expected:
                    ok = False
                    break
            if not ok:
                continue

            rings.append(
                {
                    "center": (r, c),
                    "color": color,
                    "gap_name": name,
                    "gap": (r + gap_dr, c + gap_dc),
                    "scan_dir": (gap_dr, gap_dc),
                    "cells": _ring_cells((r, c), (gap_dr, gap_dc)),
                }
            )

    rings.sort(key=lambda ring: (ring["center"][0], ring["center"][1], ring["color"]))
    return rings


def _scan_positions(ring: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    gap_r, gap_c = ring["gap"]
    dr, dc = ring["scan_dir"]
    pts: list[tuple[int, int]] = []
    r = gap_r + dr
    c = gap_c + dc
    while 0 <= r < h and 0 <= c < w:
        pts.append((r, c))
        r += dr
        c += dc
    return pts


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rings = _find_rings(g)
    ring_cells = {cell for ring in rings for cell in ring["cells"]}

    for ring in rings:
        absorbed: list[tuple[int, int]] = []
        for r, c in _scan_positions(ring, g.shape):
            if (r, c) in ring_cells:
                continue
            if int(g[r, c]) == int(ring["color"]):
                absorbed.append((r, c))

        if absorbed:
            cr, cc = ring["center"]
            out[cr, cc] = int(ring["color"])
            for r, c in absorbed:
                out[r, c] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each 3x3 ring has one missing side-center cell that points along a scanline.
    Same-colored dots on that scanline are absorbed; if at least one is found,
    the ring's center is filled with the ring color while the gap stays open.
    """
    return _transform(grid)


def _ring_zone_clear(grid: np.ndarray, center: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = center
    r0 = max(0, r - 2)
    r1 = min(h, r + 3)
    c0 = max(0, c - 2)
    c1 = min(w, c + 3)
    return not np.any(grid[r0:r1, c0:c1] != BG)


def _paint_ring(grid: np.ndarray, center: tuple[int, int], color: int, gap_offset: tuple[int, int]) -> None:
    for rr, cc in _ring_cells(center, gap_offset):
        grid[rr, cc] = color


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _dot_clear(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != BG:
        return False
    for rr, cc in _neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
            return False
    return True


def _max_rings_for_shape(h: int, w: int) -> int:
    by_area = max(1, (h * w) // 45)
    return min(RING_COUNT_RANGE[1], by_area, len(COLORS))


def _shuffled_cells(shape: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    return cells


def _place_online_dots(
    grid: np.ndarray,
    ring: dict,
    rng: random.Random,
    min_count: int,
) -> list[tuple[int, int]]:
    candidates = _scan_positions(ring, grid.shape)
    rng.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    target = rng.randint(min_count, min(ONLINE_DOT_RANGE[1], max(min_count, len(candidates))))

    for r, c in candidates:
        if _dot_clear(grid, r, c):
            grid[r, c] = int(ring["color"])
            placed.append((r, c))
            if len(placed) == target:
                break

    return placed


def _place_offline_dots(
    grid: np.ndarray,
    ring: dict,
    rng: random.Random,
) -> list[tuple[int, int]]:
    blocked = set(_scan_positions(ring, grid.shape))
    placed: list[tuple[int, int]] = []
    target = rng.randint(*OFFLINE_DOT_RANGE)

    for r, c in _shuffled_cells(grid.shape, rng):
        if (r, c) in blocked:
            continue
        if _dot_clear(grid, r, c):
            grid[r, c] = int(ring["color"])
            placed.append((r, c))
            if len(placed) == target:
                break

    return placed


def _validate_generated(
    grid: np.ndarray,
    expected_rings: Sequence[dict],
    absorbed_by_color: Dict[int, set[tuple[int, int]]],
    offline_by_color: Dict[int, set[tuple[int, int]]],
) -> bool:
    found = _find_rings(grid)
    if len(found) != len(expected_rings):
        return False

    found_by_color = {int(ring["color"]): ring for ring in found}
    expected_colors = {int(ring["color"]) for ring in expected_rings}
    if set(found_by_color) != expected_colors:
        return False

    out = main(grid)
    for ring in expected_rings:
        color = int(ring["color"])
        center = ring["center"]
        absorbed = absorbed_by_color[color]
        offline = offline_by_color[color]

        center_expected = color if absorbed else BG
        if int(out[center]) != center_expected:
            return False

        for r, c in absorbed:
            if int(out[r, c]) != BG:
                return False
        for r, c in offline:
            if int(out[r, c]) != color:
                return False

        line = set(_scan_positions(found_by_color[color], grid.shape))
        ring_cells = set(found_by_color[color]["cells"])
        counted = {
            (r, c)
            for r, c in line
            if (r, c) not in ring_cells and int(grid[r, c]) == color
        }
        if counted != absorbed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    max_rings = _max_rings_for_shape(h, w)
    ring_count = rng.randint(RING_COUNT_RANGE[0], max_rings)
    colors = rng.sample(COLORS, ring_count)
    rings: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(RING_PLACEMENT_TRIES):
            center = (rng.randint(1, h - 2), rng.randint(1, w - 2))
            gap_name, gap_offset = rng.choice(GAP_SPECS)
            if not _ring_zone_clear(grid, center):
                continue

            _paint_ring(grid, center, color, gap_offset)
            ring = {
                "center": center,
                "color": color,
                "gap_name": gap_name,
                "gap": (center[0] + gap_offset[0], center[1] + gap_offset[1]),
                "scan_dir": gap_offset,
                "cells": _ring_cells(center, gap_offset),
            }
            rings.append(ring)
            placed = True
            break
        if not placed:
            return None

    feasible_hits = [
        idx
        for idx, ring in enumerate(rings)
        if any(_dot_clear(grid, r, c) for r, c in _scan_positions(ring, grid.shape))
    ]
    if not feasible_hits:
        return None

    hit_flags = [False] * ring_count
    if ring_count == 1:
        hit_flags[feasible_hits[0]] = True
    else:
        for idx in feasible_hits:
            hit_flags[idx] = rng.random() < 0.55
        if not any(hit_flags):
            hit_flags[rng.choice(feasible_hits)] = True
        if all(hit_flags):
            hit_flags[rng.choice(feasible_hits)] = False

    absorbed_by_color: Dict[int, set[tuple[int, int]]] = {}
    offline_by_color: Dict[int, set[tuple[int, int]]] = {}
    hit_by_color = {
        int(rings[idx]["color"]): bool(hit_flags[idx]) for idx in range(len(rings))
    }
    process_rings = list(rings)
    rng.shuffle(process_rings)

    for ring in process_rings:
        color = int(ring["color"])
        absorbed: list[tuple[int, int]] = []
        if hit_by_color[color]:
            absorbed = _place_online_dots(grid, ring, rng, min_count=1)
            if not absorbed:
                return None
        absorbed_by_color[color] = set(absorbed)
        offline_by_color[color] = set(_place_offline_dots(grid, ring, rng))

    if sum(len(v) for v in absorbed_by_color.values()) == 0:
        return None

    rings_sorted = sorted(rings, key=lambda ring: ring["color"])
    if not _validate_generated(grid, rings_sorted, absorbed_by_color, offline_by_color):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

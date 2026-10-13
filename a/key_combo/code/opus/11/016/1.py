import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
OBSTACLE = 4
MARKER = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8

FILL_SEQUENCE = (BLUE, RED, GREEN, ORANGE, MAGENTA, CYAN)

SIZE_OPTIONS = (11, 13, 15, 17)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_edge_markers(grid: np.ndarray) -> dict[str, list[int]]:
    h, w = grid.shape
    top = [int(c) for c in range(1, w - 1) if int(grid[0, c]) == MARKER]
    bottom = [int(c) for c in range(1, w - 1) if int(grid[h - 1, c]) == MARKER]
    left = [int(r) for r in range(1, h - 1) if int(grid[r, 0]) == MARKER]
    right = [int(r) for r in range(1, h - 1) if int(grid[r, w - 1]) == MARKER]
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def _emit_ray(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> str:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    while True:
        nr, nc = r + dr, c + dc
        if nr <= 0 or nr >= h - 1 or nc <= 0 or nc >= w - 1:
            return "edge"
        val = int(grid[nr, nc])
        if val == MARKER:
            return "ray"
        if val == OBSTACLE:
            return "obstacle"
        grid[nr, nc] = MARKER
        r, c = nr, nc


def _simulate_rays(grid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    markers = _extract_edge_markers(out)

    reasons: list[str] = []
    for c in sorted(markers["top"]):
        reasons.append(_emit_ray(out, (0, c), (1, 0)))
    for c in sorted(markers["bottom"]):
        reasons.append(_emit_ray(out, (h - 1, c), (-1, 0)))
    for r in sorted(markers["left"]):
        reasons.append(_emit_ray(out, (r, 0), (0, 1)))
    for r in sorted(markers["right"]):
        reasons.append(_emit_ray(out, (r, w - 1), (0, -1)))
    return out, reasons


def _zero_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            min_r, min_c = r, c
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr < min_r or (cr == min_r and cc < min_c):
                    min_r, min_c = cr, cc
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "cells": cells,
                    "area": len(cells),
                    "anchor": (min_r, min_c),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit inward rays from edge markers, then fill the enclosed black regions by area rank.

    Input structure:
    - A black background grid contains only four groups of gray edge markers and one central yellow 3x3 obstacle.
    - Top and bottom markers sit on the first and last row; left and right markers sit on the first and last column.
    - No corner is marked, and the obstacle acts as a fixed barrier during ray growth.

    Rule:
    - Keep every gray marker and every yellow obstacle cell unchanged.
    - Emit gray rays inward from edge markers: top then bottom from left to right, then left then right from top to bottom.
    - Each ray extends cell by cell until the next cell would be an earlier gray ray, a yellow obstacle cell, or the opposite border cell; that stopping cell is not painted.
    - After all rays are drawn, find every 4-connected black region, sort regions by increasing area and then by topmost-leftmost cell, and fill them with colors [blue, red, green, orange, magenta, cyan] cyclically.

    Key insight:
    - The region boundaries are not predefined: they emerge only after simulating the ordered ray collisions, so area ranking must happen after the ray pass.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    markers = _extract_edge_markers(g)
    for side, coords in markers.items():
        assert 2 <= len(coords) <= 4, f"expected 2-4 markers on {side}"

    obstacle_cells = np.argwhere(g == OBSTACLE)
    assert obstacle_cells.size > 0, "expected a yellow obstacle"

    ray_grid, _reasons = _simulate_rays(g)
    regions = _zero_components(ray_grid)
    regions.sort(key=lambda comp: (int(comp["area"]), tuple(comp["anchor"])))

    out = ray_grid.copy()
    for idx, comp in enumerate(regions):
        color = FILL_SEQUENCE[idx % len(FILL_SEQUENCE)]
        for r, c in comp["cells"]:
            out[r, c] = color
    return out


def _choose_edge_positions(
    span_len: int,
    count: int,
    rng: random.Random,
    *,
    preferred: tuple[int, ...] = (),
    must_include: tuple[int, ...] = (),
    min_gap: int = 2,
) -> list[int]:
    preferred_set = {int(p) for p in preferred if 1 <= int(p) <= span_len - 2}
    must_include_set = {int(p) for p in must_include if 1 <= int(p) <= span_len - 2}
    candidates = []
    weights = []
    for combo in itertools.combinations(range(1, span_len - 1), count):
        if any(b - a < min_gap for a, b in zip(combo, combo[1:])):
            continue
        if not must_include_set.issubset(combo):
            continue
        score = 1 + 3 * len(preferred_set.intersection(combo))
        if combo[0] <= 2 or combo[-1] >= span_len - 3:
            score += 1
        candidates.append(combo)
        weights.append(score)
    assert candidates, "no valid edge-marker placements"
    return list(rng.choices(candidates, weights=weights, k=1)[0])


def _sample_opposite_edges(
    span_len: int,
    count_a: int,
    count_b: int,
    rng: random.Random,
    preferred: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    first = _choose_edge_positions(span_len, count_a, rng, preferred=preferred)
    shared_count = 0
    if rng.random() < 0.65:
        shared_count = rng.randint(0, min(len(first), count_b, 2))
    must_share = tuple(sorted(rng.sample(first, shared_count))) if shared_count else ()
    second = _choose_edge_positions(
        span_len,
        count_b,
        rng,
        preferred=preferred,
        must_include=must_share,
    )
    return sorted(first), sorted(second)


def _has_area_tie(areas: list[int]) -> bool:
    return len(set(areas)) < len(areas)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(SIZE_OPTIONS)
    w = rng.choice(SIZE_OPTIONS)
    grid = np.zeros((h, w), dtype=int)

    r0 = h // 2 - 1
    c0 = w // 2 - 1
    grid[r0 : r0 + 3, c0 : c0 + 3] = OBSTACLE

    center_cols = (c0, c0 + 1, c0 + 2)
    center_rows = (r0, r0 + 1, r0 + 2)

    col_preference = rng.choice(
        (
            center_cols,
            (1, c0 + 1, c0 + 2),
            (c0, c0 + 1, w - 2),
            (1, c0 + 1, w - 2),
        )
    )
    row_preference = rng.choice(
        (
            center_rows,
            (1, r0 + 1, r0 + 2),
            (r0, r0 + 1, h - 2),
            (1, r0 + 1, h - 2),
        )
    )

    top_cols, bottom_cols = _sample_opposite_edges(w, 2, 2, rng, col_preference)
    left_rows, right_rows = _sample_opposite_edges(h, 2, 2, rng, row_preference)

    for c in top_cols:
        grid[0, c] = MARKER
    for c in bottom_cols:
        grid[h - 1, c] = MARKER
    for r in left_rows:
        grid[r, 0] = MARKER
    for r in right_rows:
        grid[r, w - 1] = MARKER

    ray_grid, reasons = _simulate_rays(grid)
    comps = _zero_components(ray_grid)
    region_count = len(comps)
    if not (3 <= region_count <= 6):
        return None

    areas = [int(comp["area"]) for comp in comps]
    if min(areas) < 3:
        return None
    if max(areas) - min(areas) < 2:
        return None

    reason_set = set(reasons)
    if "obstacle" not in reason_set:
        return None
    if len(reason_set) < 2:
        return None

    if rng.random() < 0.25 and not _has_area_tie(areas):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition input")

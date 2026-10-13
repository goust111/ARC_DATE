from __future__ import annotations

import random
from collections import deque

import numpy as np


ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
NON_BLACK = list(range(1, 10))

GEN_HEIGHT_RANGE = (14, 30)
GEN_WIDTH_RANGE = (14, 30)
GEN_CROSS_COUNT_RANGE = (2, 2)
GEN_ATTEMPTS = 700


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cross_pixels(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_coords(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == 0:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and grid[nr, nc] != 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_crosses(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    crosses: list[dict] = []
    claimed: set[tuple[int, int]] = set()

    for comp in _component_coords(g):
        assert len(comp) == 5, "each component must have 5 pixels"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 2 and c1 - c0 == 2, "each cross must fit in a 3x3 box"

        center = (r0 + 1, c0 + 1)
        expected_pixels = set(_cross_pixels(center))
        assert set(comp) == expected_pixels, "component must be a plus-shaped cross"

        arm_coords = [(center[0] - 1, center[1]), (center[0] + 1, center[1]), (center[0], center[1] - 1), (center[0], center[1] + 1)]
        arm_colors = {int(g[r, c]) for r, c in arm_coords}
        assert len(arm_colors) == 1, "all four arms must share one color"
        arm_color = arm_colors.pop()
        center_color = int(g[center])
        assert arm_color != 0 and center_color != 0 and arm_color != center_color

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) in expected_pixels:
                    continue
                assert g[rr, cc] == 0, "cross diagonals must stay background"

        claimed.update(comp)
        crosses.append(
            {
                "center": center,
                "arm_color": arm_color,
                "center_color": center_color,
                "pixels": tuple(sorted(expected_pixels)),
            }
        )

    assert claimed == set(map(tuple, np.argwhere(g != 0))), "all foreground pixels must belong to crosses"
    assert crosses, "expected at least one cross"
    return crosses


def _trace_rays(crosses: list[dict], shape: tuple[int, int]) -> list[dict]:
    all_other_pixels = []
    for i, _cross in enumerate(crosses):
        others = []
        for j, other in enumerate(crosses):
            if i == j:
                continue
            others.extend(other["pixels"])
        all_other_pixels.append(tuple(others))

    rays: list[dict] = []
    for i, cross in enumerate(crosses):
        cr, cc = cross["center"]
        for dr, dc in ORTH_DIRS:
            cells = []
            step = 2
            reason = "boundary"
            while True:
                rr = cr + dr * step
                cc2 = cc + dc * step
                if not _in_bounds(shape, rr, cc2):
                    break
                if any(max(abs(rr - orr), abs(cc2 - occ)) <= 1 for orr, occ in all_other_pixels[i]):
                    reason = "neighbor"
                    break
                cells.append((rr, cc2))
                step += 1
            rays.append(
                {
                    "cross": i,
                    "family": "orth",
                    "direction": (dr, dc),
                    "color": cross["arm_color"],
                    "cells": tuple(cells),
                    "reason": reason,
                }
            )
        for dr, dc in DIAG_DIRS:
            cells = []
            step = 1
            reason = "boundary"
            while True:
                rr = cr + dr * step
                cc2 = cc + dc * step
                if not _in_bounds(shape, rr, cc2):
                    break
                if any(max(abs(rr - orr), abs(cc2 - occ)) <= 1 for orr, occ in all_other_pixels[i]):
                    reason = "neighbor"
                    break
                cells.append((rr, cc2))
                step += 1
            rays.append(
                {
                    "cross": i,
                    "family": "diag",
                    "direction": (dr, dc),
                    "color": cross["center_color"],
                    "cells": tuple(cells),
                    "reason": reason,
                }
            )
    return rays


def _paint_crosses(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for cross in crosses:
        r, c = cross["center"]
        out[r, c] = cross["center_color"]
        out[r - 1, c] = cross["arm_color"]
        out[r + 1, c] = cross["arm_color"]
        out[r, c - 1] = cross["arm_color"]
        out[r, c + 1] = cross["arm_color"]
    return out


def _render_from_rays(shape: tuple[int, int], crosses: list[dict], rays: list[dict]) -> np.ndarray:
    out = _paint_crosses(shape, crosses)
    for ray in rays:
        for r, c in ray["cells"]:
            out[r, c] = ray["color"]
    return out


def _solve_with_rules(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    crosses = _extract_crosses(grid)
    rays = _trace_rays(crosses, grid.shape)
    return _render_from_rays(grid.shape, crosses, rays), crosses, rays


def _render_to_edge(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    out = _paint_crosses(shape, crosses)
    for cross in crosses:
        cr, cc = cross["center"]
        for dr, dc in ORTH_DIRS:
            step = 2
            while True:
                rr = cr + dr * step
                cc2 = cc + dc * step
                if not _in_bounds(shape, rr, cc2):
                    break
                out[rr, cc2] = cross["arm_color"]
                step += 1
        for dr, dc in DIAG_DIRS:
            step = 1
            while True:
                rr = cr + dr * step
                cc2 = cc + dc * step
                if not _in_bounds(shape, rr, cc2):
                    break
                out[rr, cc2] = cross["center_color"]
                step += 1
    return out


def _render_fixed_5x5(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    out = _paint_crosses(shape, crosses)
    for cross in crosses:
        cr, cc = cross["center"]
        for dr, dc in ORTH_DIRS:
            rr = cr + dr * 2
            cc2 = cc + dc * 2
            if _in_bounds(shape, rr, cc2):
                out[rr, cc2] = cross["arm_color"]
        for dr, dc in DIAG_DIRS:
            for step in (1, 2):
                rr = cr + dr * step
                cc2 = cc + dc * step
                if _in_bounds(shape, rr, cc2):
                    out[rr, cc2] = cross["center_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each two-color cross into orthogonal arm rays and diagonal center rays, stopping one Chebyshev ring before any other cross.

    Input structure:
    - A black background contains one or more separated 3x3 plus-shaped crosses.
    - In each cross, the four arm pixels share one color and the center pixel has a different color.
    - No other foreground objects are present.

    Rule:
    - Keep every original cross pixel unchanged.
    - From each arm endpoint, extend the arm color outward along the four orthogonal directions.
    - From each center, extend the center color along the four diagonals.
    - Before painting a candidate ray cell, stop that ray if the cell is within Chebyshev distance 1 of any original pixel from a different cross; otherwise paint it and continue until blocked or until the grid edge.

    Key insight:
    - Other crosses act as the blockers: each ray grows freely except where another cross's one-cell Chebyshev safety halo cuts it off.
    """
    out, _crosses, _rays = _solve_with_rules(np.asarray(grid, dtype=int))
    return out


def _sample_any_center(h: int, w: int) -> tuple[int, int]:
    return random.randint(1, h - 2), random.randint(1, w - 2)


def _sample_edge_center(h: int, w: int) -> tuple[int, int]:
    if random.random() < 0.5:
        r = random.choice([1, 2, h - 3, h - 2])
        c = random.randint(1, w - 2)
    else:
        r = random.randint(1, h - 2)
        c = random.choice([1, 2, w - 3, w - 2])
    return r, c


def _crosses_are_separate(centers: list[tuple[int, int]]) -> bool:
    pixel_sets = [set(_cross_pixels(center)) for center in centers]
    for i in range(len(pixel_sets)):
        for j in range(i + 1, len(pixel_sets)):
            for ra, ca in pixel_sets[i]:
                for rb, cb in pixel_sets[j]:
                    if max(abs(ra - rb), abs(ca - cb)) <= 1:
                        return False
    return True


def _sample_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    assert count == 2, "generator uses a two-cross layout to satisfy the no-ray-overlap guarantee"

    horizontal = random.random() < 0.5
    if horizontal:
        anchor = random.choice([1, 2, h - 3, h - 2]) if random.random() < 0.6 else random.randint(3, h - 4)
        row_gap = random.choice([-2, -1, 1, 2])
        r1, r2 = anchor, anchor + row_gap
        if not (1 <= r2 <= h - 2):
            return None
        left_cols = list(range(1, max(2, w // 4)))
        right_cols = list(range(min(w - 2, (3 * w) // 4), w - 1))
        if not left_cols or not right_cols:
            return None
        c1 = random.choice(left_cols)
        c2 = random.choice(right_cols)
        centers = [(r1, c1), (r2, c2)]
    else:
        anchor = random.choice([1, 2, w - 3, w - 2]) if random.random() < 0.6 else random.randint(3, w - 4)
        col_gap = random.choice([-2, -1, 1, 2])
        c1, c2 = anchor, anchor + col_gap
        if not (1 <= c2 <= w - 2):
            return None
        top_rows = list(range(1, max(2, h // 4)))
        bottom_rows = list(range(min(h - 2, (3 * h) // 4), h - 1))
        if not top_rows or not bottom_rows:
            return None
        r1 = random.choice(top_rows)
        r2 = random.choice(bottom_rows)
        centers = [(r1, c1), (r2, c2)]

    if random.random() < 0.5:
        centers.reverse()
    if _crosses_are_separate(centers):
        return centers
    return None


def _sample_colors(count: int) -> list[tuple[int, int]]:
    pairs = []
    arm_pool = NON_BLACK.copy()
    random.shuffle(arm_pool)
    for i in range(count):
        arm = arm_pool[i % len(arm_pool)]
        center_choices = [color for color in NON_BLACK if color != arm]
        center = random.choice(center_choices)
        pairs.append((arm, center))
    random.shuffle(pairs)
    return pairs


def _build_input(shape: tuple[int, int], centers: list[tuple[int, int]], color_pairs: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for (r, c), (arm_color, center_color) in zip(centers, color_pairs):
        grid[r, c] = center_color
        grid[r - 1, c] = arm_color
        grid[r + 1, c] = arm_color
        grid[r, c - 1] = arm_color
        grid[r, c + 1] = arm_color
    return grid


def _ray_overlap_free(rays: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for ray in rays:
        for cell in ray["cells"]:
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out, crosses, rays = _solve_with_rules(inp)
    except AssertionError:
        return False

    if not _ray_overlap_free(rays):
        return False

    total_added = sum(len(ray["cells"]) for ray in rays)
    orth_added = sum(len(ray["cells"]) for ray in rays if ray["family"] == "orth")
    diag_added = sum(len(ray["cells"]) for ray in rays if ray["family"] == "diag")
    if total_added < 10 or orth_added == 0 or diag_added == 0:
        return False

    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, _render_to_edge(inp.shape, crosses)):
        return False
    if np.array_equal(out, _render_fixed_5x5(inp.shape, crosses)):
        return False

    if not any(ray["reason"] == "neighbor" for ray in rays):
        return False
    if not any(ray["reason"] == "boundary" for ray in rays):
        return False
    if not any(ray["family"] == "orth" and ray["reason"] == "neighbor" for ray in rays):
        return False
    per_cross_lengths: dict[int, list[int]] = {}
    per_cross_reasons: dict[int, set[str]] = {}
    for ray in rays:
        per_cross_lengths.setdefault(ray["cross"], []).append(len(ray["cells"]))
        per_cross_reasons.setdefault(ray["cross"], set()).add(ray["reason"])

    if not any(len(set(lengths)) >= 3 for lengths in per_cross_lengths.values()):
        return False
    if not any(reasons == {"boundary", "neighbor"} or reasons == {"neighbor", "boundary"} for reasons in per_cross_reasons.values()):
        return False

    # Verify the stop condition itself, not only the final image.
    for ray in rays:
        cells = list(ray["cells"])
        if ray["reason"] != "neighbor":
            continue
        if ray["family"] == "orth":
            start_step = 2
        else:
            start_step = 1
        cross = crosses[ray["cross"]]
        dr, dc = ray["direction"]
        stop_step = start_step + len(cells)
        stop_cell = (cross["center"][0] + dr * stop_step, cross["center"][1] + dc * stop_step)
        blockers = []
        for i, other in enumerate(crosses):
            if i != ray["cross"]:
                blockers.extend(other["pixels"])
        if not any(max(abs(stop_cell[0] - rr), abs(stop_cell[1] - cc)) <= 1 for rr, cc in blockers):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    count = random.randint(*GEN_CROSS_COUNT_RANGE)
    centers = _sample_centers(h, w, count)
    if centers is None:
        return None
    color_pairs = _sample_colors(count)
    inp = _build_input((h, w), centers, color_pairs)
    if _semantic_checks(inp):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-blocking ARC input")

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
WALL = 5

COLOR_ORDER = (RED, GREEN)
PERIOD_POOL = (2, 3, 4, 5, 6)
GRID_RANGE = (18, 24)
WALL_MARGIN = 4
MAX_GEN_ATTEMPTS = 500


Coord = tuple[int, int]


@dataclass(frozen=True)
class RayCandidate:
    start: Coord
    second: Coord
    delta: Coord
    period: int
    markers: tuple[Coord, Coord]
    extra: tuple[Coord, ...]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_passable(passable: np.ndarray, start: Coord) -> np.ndarray:
    seen = np.zeros_like(passable, dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _partition_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blue_coords = np.argwhere(grid == BLUE)
    assert blue_coords.shape[0] == 1, "expected a unique blue indicator"
    start = tuple(map(int, blue_coords[0]))

    wall_mask = grid == WALL
    passable = ~wall_mask
    comps = _components(passable)
    assert len(comps) == 2, f"expected exactly two non-wall regions, got {len(comps)}"

    inside = _flood_passable(passable, start)
    outside = passable & ~inside
    assert inside.any() and outside.any(), "indicator must split passable space into inside/outside"
    return inside, outside


def _infer_ray(points: np.ndarray, shape: tuple[int, int]) -> tuple[Coord, Coord, int]:
    assert points.shape == (2, 2)
    (r0, c0), (r1, c1) = [tuple(map(int, pt)) for pt in points]
    h, w = shape

    if r0 == r1:
        left, right = sorted([(r0, c0), (r1, c1)], key=lambda rc: rc[1])
        period = right[1] - left[1]
        left_dist = left[1]
        right_dist = (w - 1) - right[1]
        assert period > 0 and left_dist != right_dist, "horizontal pair must have a unique edge-near endpoint"
        if left_dist < right_dist:
            return left, (0, period), period
        return right, (0, -period), period

    if c0 == c1:
        top, bottom = sorted([(r0, c0), (r1, c1)], key=lambda rc: rc[0])
        period = bottom[0] - top[0]
        top_dist = top[0]
        bottom_dist = (h - 1) - bottom[0]
        assert period > 0 and top_dist != bottom_dist, "vertical pair must have a unique edge-near endpoint"
        if top_dist < bottom_dist:
            return top, (period, 0), period
        return bottom, (-period, 0), period

    raise AssertionError("paired points must share a row or a column")


def _ray_proposals(grid: np.ndarray, region_mask: np.ndarray, start: Coord, delta: Coord) -> list[Coord]:
    dr, dc = delta
    rr, cc = start[0] + dr, start[1] + dc
    out: list[Coord] = []
    while _in_bounds(grid.shape, rr, cc):
        if region_mask[rr, cc] and grid[rr, cc] == BG:
            out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays inside the wall-defined partition chosen by the blue indicator.

    Input structure:
    - The background is black, with a one-pixel grey wall that forms a 4-connected
      edge-to-edge barrier and splits the grid into exactly two non-wall regions.
    - One blue indicator pixel lies in one region and marks it as the inside region.
    - Each region contains exactly one red pair and one green pair; each same-color
      pair shares a row or a column.

    Rule:
    - Flood-fill from the blue indicator with 4-connectivity, blocked by the grey wall;
      the reached cells are inside and the remaining non-wall cells are outside.
    - In each region, each same-color pair defines a ray axis and a period equal to the
      gap between the two points along their shared row or column.
    - Start from the endpoint closer to the relevant grid edge and keep stepping by that
      period away from the edge. Only background cells in the same region are filled.
    - Original wall, indicator, and source points stay unchanged. If two ray fills want
      the same background cell, the smaller period wins.

    Key insight:
    - The wall is both the partition detector and the gate on propagation: it determines
      which same-color points pair together and restricts each periodic ray to its own region.
    """
    g = np.asarray(grid, dtype=int)
    inside, outside = _partition_regions(g)
    out = g.copy()

    proposals: dict[Coord, tuple[int, int]] = {}
    for region_mask in (inside, outside):
        for color in COLOR_ORDER:
            pts = np.argwhere(region_mask & (g == color))
            assert pts.shape[0] == 2, f"expected exactly two points of color {color} in each region"
            start, delta, period = _infer_ray(pts, g.shape)
            for pos in _ray_proposals(g, region_mask, start, delta):
                prev = proposals.get(pos)
                if prev is None or period < prev[0]:
                    proposals[pos] = (period, color)

    for (r, c), (_period, color) in proposals.items():
        out[r, c] = color
    return out


def _runs_1d(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = arr.shape[0]
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and arr[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _candidate_extra(start: Coord, delta: Coord, region_mask: np.ndarray) -> tuple[Coord, ...]:
    dr, dc = delta
    rr, cc = start[0] + 2 * dr, start[1] + 2 * dc
    out: list[Coord] = []
    while _in_bounds(region_mask.shape, rr, cc):
        if region_mask[rr, cc]:
            out.append((rr, cc))
        rr += dr
        cc += dc
    return tuple(out)


def _horizontal_candidates(region_mask: np.ndarray, period: int, side: str) -> list[RayCandidate]:
    h, w = region_mask.shape
    out: list[RayCandidate] = []
    for r in range(h):
        for c0, c1 in _runs_1d(region_mask[r]):
            if side == "left":
                for start_c in range(c0, min(c0 + 2, c1) + 1):
                    second_c = start_c + period
                    if second_c > c1 or start_c + 2 * period > c1:
                        continue
                    if start_c >= (w - 1) - second_c:
                        continue
                    start = (r, start_c)
                    delta = (0, period)
                    extra = _candidate_extra(start, delta, region_mask)
                    if extra:
                        out.append(RayCandidate(start, (r, second_c), delta, period, (start, (r, second_c)), extra))
            else:
                for start_c in range(c1, max(c1 - 2, c0) - 1, -1):
                    second_c = start_c - period
                    if second_c < c0 or start_c - 2 * period < c0:
                        continue
                    if (w - 1) - start_c >= second_c:
                        continue
                    start = (r, start_c)
                    delta = (0, -period)
                    extra = _candidate_extra(start, delta, region_mask)
                    if extra:
                        out.append(RayCandidate(start, (r, second_c), delta, period, (start, (r, second_c)), extra))
    random.shuffle(out)
    return out


def _vertical_candidates(region_mask: np.ndarray, period: int, side: str) -> list[RayCandidate]:
    h, w = region_mask.shape
    out: list[RayCandidate] = []
    for c in range(w):
        for r0, r1 in _runs_1d(region_mask[:, c]):
            if side == "top":
                for start_r in range(r0, min(r0 + 2, r1) + 1):
                    second_r = start_r + period
                    if second_r > r1 or start_r + 2 * period > r1:
                        continue
                    if start_r >= (h - 1) - second_r:
                        continue
                    start = (start_r, c)
                    delta = (period, 0)
                    extra = _candidate_extra(start, delta, region_mask)
                    if extra:
                        out.append(RayCandidate(start, (second_r, c), delta, period, (start, (second_r, c)), extra))
            else:
                for start_r in range(r1, max(r1 - 2, r0) - 1, -1):
                    second_r = start_r - period
                    if second_r < r0 or start_r - 2 * period < r0:
                        continue
                    if (h - 1) - start_r >= second_r:
                        continue
                    start = (start_r, c)
                    delta = (-period, 0)
                    extra = _candidate_extra(start, delta, region_mask)
                    if extra:
                        out.append(RayCandidate(start, (second_r, c), delta, period, (start, (second_r, c)), extra))
    random.shuffle(out)
    return out


def _mean_col(mask: np.ndarray) -> float:
    return float(np.argwhere(mask)[:, 1].mean())


def _build_vertical_wall(h: int, w: int) -> np.ndarray | None:
    cols = [random.randint(WALL_MARGIN, w - 1 - WALL_MARGIN)]
    for _r in range(1, h):
        prev = cols[-1]
        options = [cc for cc in range(prev - 2, prev + 3) if WALL_MARGIN <= cc <= w - 1 - WALL_MARGIN]
        weights = [4 if cc == prev else 3 if abs(cc - prev) == 1 else 1 for cc in options]
        cols.append(random.choices(options, weights=weights, k=1)[0])

    grid = np.zeros((h, w), dtype=int)
    wall_cells: set[Coord] = {(0, cols[0])}
    for r in range(h - 1):
        c0, c1 = cols[r], cols[r + 1]
        for c in range(min(c0, c1), max(c0, c1) + 1):
            wall_cells.add((r, c))
        wall_cells.add((r + 1, c1))

    for r, c in wall_cells:
        grid[r, c] = WALL

    passable = grid != WALL
    comps = _components(passable)
    if len(comps) != 2:
        return None
    sizes = sorted(comp.shape[0] for comp in comps)
    if sizes[0] < h * 3:
        return None
    return grid


def _choose_ray_pair(
    region_mask: np.ndarray,
    horiz_period: int,
    vert_period: int,
    horiz_side: str,
    want_conflict: bool,
) -> tuple[RayCandidate, RayCandidate] | None:
    vertical_side = random.choice(("top", "bottom"))
    h_candidates = _horizontal_candidates(region_mask, horiz_period, horiz_side)
    v_candidates = _vertical_candidates(region_mask, vert_period, vertical_side)
    if not h_candidates or not v_candidates:
        return None

    best: list[tuple[RayCandidate, RayCandidate]] = []
    for h_ray in h_candidates:
        h_markers = set(h_ray.markers)
        h_extra = set(h_ray.extra)
        for v_ray in v_candidates:
            if h_markers & set(v_ray.markers):
                continue
            overlap = h_extra & set(v_ray.extra)
            if want_conflict and not overlap:
                continue
            if not want_conflict and overlap:
                continue
            best.append((h_ray, v_ray))

    if not best:
        return None
    return random.choice(best)


def _apply_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _conflict_cells(grid: np.ndarray) -> int:
    inside, outside = _partition_regions(grid)
    overlaps = 0
    fills: list[dict[Coord, int]] = []
    for region_mask in (inside, outside):
        for color in COLOR_ORDER:
            pts = np.argwhere(region_mask & (grid == color))
            start, delta, period = _infer_ray(pts, grid.shape)
            ray_map = {pos: period for pos in _ray_proposals(grid, region_mask, start, delta)}
            fills.append(ray_map)
    for i in range(len(fills)):
        for j in range(i + 1, len(fills)):
            overlaps += len(set(fills[i]) & set(fills[j]))
    return overlaps


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    base = _build_vertical_wall(h, w)
    if base is None:
        return None

    passable = base != WALL
    comps = _components(passable)
    if len(comps) != 2:
        return None

    comp_masks = []
    for comp in comps:
        mask = np.zeros_like(base, dtype=bool)
        for r, c in comp:
            mask[int(r), int(c)] = True
        comp_masks.append(mask)

    comp_masks.sort(key=_mean_col)
    left_mask, right_mask = comp_masks
    region_masks = [left_mask, right_mask]
    horiz_sides = ["left", "right"]

    periods = random.sample(PERIOD_POOL, 4)
    conflict_mode = random.choices(("one", "both", "none"), weights=(6, 2, 2), k=1)[0]
    conflict_flags = [False, False]
    if conflict_mode == "one":
        conflict_flags[random.randrange(2)] = True
    elif conflict_mode == "both":
        conflict_flags = [True, True]

    placements: list[tuple[int, RayCandidate, int, RayCandidate]] = []
    occupied: set[Coord] = set()
    period_idx = 0
    for idx, region_mask in enumerate(region_masks):
        horiz_color = random.choice(COLOR_ORDER)
        vert_color = GREEN if horiz_color == RED else RED
        horiz_period = periods[period_idx]
        vert_period = periods[period_idx + 1]
        period_idx += 2

        pair = _choose_ray_pair(
            region_mask,
            horiz_period,
            vert_period,
            horiz_sides[idx],
            conflict_flags[idx],
        )
        if pair is None:
            return None
        h_ray, v_ray = pair
        if occupied & (set(h_ray.markers) | set(v_ray.markers)):
            return None
        occupied |= set(h_ray.markers) | set(v_ray.markers)
        placements.append((horiz_color, h_ray, vert_color, v_ray))

    grid = base.copy()
    ray_avoid: set[Coord] = set()
    for horiz_color, h_ray, vert_color, v_ray in placements:
        for r, c in h_ray.markers:
            grid[r, c] = horiz_color
        for r, c in v_ray.markers:
            grid[r, c] = vert_color
        ray_avoid.update(h_ray.markers)
        ray_avoid.update(v_ray.markers)
        ray_avoid.update(h_ray.extra)
        ray_avoid.update(v_ray.extra)

    blue_region = random.choice(region_masks)
    blue_choices = [(int(r), int(c)) for r, c in np.argwhere(blue_region & (grid == BG)) if (int(r), int(c)) not in ray_avoid]
    if not blue_choices:
        return None
    br, bc = random.choice(blue_choices)
    grid[br, bc] = BLUE

    grid = _apply_symmetry(grid)
    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 6:
        return None
    if conflict_mode != "none" and _conflict_cells(grid) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partitioned periodic-ray puzzle")

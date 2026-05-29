# Combo variant of b9e38dc0 (index 28)
# combo_spec: D9/1+D11/2
# Name: 仅输出射线区域+无孤立像素

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis for the fused variant
# - Concrete behavior trace:
#   Original task fills the interior and the exterior ray wedge on the full canvas, then lets extra
#   non-wall pixels block or participate in a second effect. In this combo there are no such extra
#   pixels at all, so the only active structure is the wall + interior seed. Example: if a U-like wall
#   opens upward, the original solver would leave the wall and seed in place and paint both the inside
#   cavity and the upward wedge. The fused solver still computes the same wedge, but discards the wall,
#   discards the interior fill, and crops to the minimal bbox of the exterior wedge only.
# - Complexity calibration:
#   The variant is simplified by removing obstacles, so the generator keeps the four cardinal radiate
#   directions and both arm-angle modes (1:1 and 1:2). The crop is therefore still visually diverse:
#   different samples produce short tall strips, wide horizontal strips, and jagged wedges with bg
#   corners inside the cropped bbox.
# - Assumption audit:
#   The original `main()` assumed other non-bg colors might be obstacles; that is no longer true, so the
#   solver now assumes the only non-wall color is the fill seed. It still relies on the wall being the
#   largest non-bg object, on the chosen wall component having two degree-1 endpoints, and on the opening
#   line separating interior cells from exterior ray cells. The generator enforces each of these.
# - Placement legality:
#   The wall builder from the base task is preserved so the wall remains 8-connected, bounded, and has an
#   open side with at least 3 cells to the radiate edge. The seed blob is placed strictly in the cavity
#   near the closed end, never overlapping the wall or leaving the enclosure.
# - Visual observability / neatness:
#   Because the output contains only the external ray strip, the change from input to output is obvious:
#   the sprite vanishes and only the predicted wedge remains. The generator rejects tiny or weakly cropped
#   cases so the crop itself is human-visible across a few examples.
# - New implicit assumptions:
#   The ray-only crop assumes the flood reaches cells strictly beyond the endpoint row/col in the radiate
#   direction. The generator validates this by requiring a minimum ray area and a substantial area
#   reduction after cropping.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _pick_wall_and_fill(g: np.ndarray) -> tuple[int, int, list[set[tuple[int, int]]]]:
    bg = int(get_bg(g))
    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert vals, "expected at least one non-background color"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall component"

    other_colors = [c for c in vals if c != wall]
    assert other_colors, "expected a non-wall fill color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    return wall, fill, comps


def _trace_ray_mask(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    bg = int(get_bg(g))
    wall, fill, comps = _pick_wall_and_fill(g)

    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))

    comp_mask = np.zeros_like(g, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) >= 2, "expected at least two wall endpoints"
    if len(endpoints) > 2:
        best = None
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                a, b = endpoints[i], endpoints[j]
                d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                if best is None or d > best[0]:
                    best = (d, a, b)
        assert best is not None
        endpoints = [best[1], best[2]]

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]

    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad != (0, 0), "expected a cardinal radiate direction"

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += d[0]
                cc += d[1]
            elif rad[0] != 0:
                rr += d[0]
            else:
                cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q = []
    for r, c in fill_pos:
        if passable[r, c]:
            seen[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)

    row_idx = np.arange(h)[:, None]
    col_idx = np.arange(w)[None, :]
    endpoint_rows = [r for r, _ in endpoints]
    endpoint_cols = [c for _, c in endpoints]
    if rad == (-1, 0):
        ray_side = row_idx < min(endpoint_rows)
    elif rad == (1, 0):
        ray_side = row_idx > max(endpoint_rows)
    elif rad == (0, -1):
        ray_side = col_idx < min(endpoint_cols)
    else:
        ray_side = col_idx > max(endpoint_cols)

    ray_mask = filled_mask & ray_side
    assert ray_mask.any(), "expected non-empty exterior ray region"
    return bg, fill, ray_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict only the external ray wedge emitted from a partial enclosure.

    Input structure:
    - The background is the most common color.
    - The dominant non-background color is an 8-connected wall with one open side.
    - One smaller non-wall color forms a seed blob inside the cavity near the closed end.

    Rule:
    - Flood-fill from the seed through the cavity and outward through the opening, using the wall and the
      two endpoint rays as the boundaries of the reachable region.
    - Ignore the cavity fill in the final answer and keep only the exterior ray region beyond the opening.
    - Crop the result to the minimal bounding box of that exterior region; the wall and seed do not appear.

    Key insight:
    - The two degree-1 wall endpoints determine both the cardinal radiate direction and the exact slanted
      ray boundaries, so the output is the cropped wedge outside the endpoint line rather than the full scene.
    """
    bg, fill, ray_mask = _trace_ray_mask(np.array(grid, dtype=int))
    coords = np.argwhere(ray_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    submask = ray_mask[r0 : r1 + 1, c0 : c1 + 1]
    out[submask] = int(fill)
    return out


# --------------- generator ---------------

# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Validation
MIN_RAY_PIXELS = 12
MIN_CROP_REDUCTION = 0.35
MIN_OUTPUT_SIDE = 3
GEN_RETRIES = 300


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False):
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
            chosen_rows = [1]
            if n_side_steps > 1:
                chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
            chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        depth = random.randint(DEPTH_HORIZ[0], min(DEPTH_HORIZ[1], w - 6))
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    elif gap == 6:
        left_bottom, right_bottom = 2, 4
    else:
        left_bottom, right_bottom = random.choice([(2, 5), (3, 4)])

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < WALL_PIXELS_MIN:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            ) >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int]:
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)

    interior = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        neighbors = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        for q in random.sample(neighbors, len(neighbors)):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    bg, wall_color, fill_color = random.sample(range(10), 3)
    grid = np.full((h, w), int(bg), dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = int(wall_color)

    interior = _local_interior(local_wall, depth)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = int(fill_color)

    try:
        out = main(grid)
    except Exception:
        return None

    ray_pixels = int((out == fill_color).sum())
    crop_reduction = 1.0 - (out.shape[0] * out.shape[1]) / float(h * w)
    if ray_pixels < MIN_RAY_PIXELS:
        return None
    if min(out.shape) < MIN_OUTPUT_SIDE:
        return None
    if out.shape == grid.shape:
        return None
    if crop_reduction < MIN_CROP_REDUCTION:
        return None
    if np.all(out == fill_color):
        return None
    if len(np.unique(grid)) != 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")

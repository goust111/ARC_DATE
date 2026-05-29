# Combo variant of b9e38dc0 (index 40)
# combo_spec: D4/2+D16/2
# Name: 最大色块触发+像素数编码射线长度

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior:
#    - Original: one interior fill color is taken wholesale as the seed family, and
#      radiation keeps going until the grid edge.
#    - New: the enclosure contains several colored families. Each family is one
#      multi-pixel component plus same-color singleton markers. Only the largest
#      4-connected component among all interior families is allowed to seed the fill.
#      After that winner is known, only singleton components of the same color are
#      counted, and that count caps how many outward layers of radiation survive.
#    - Example: if red has a 5-cell blob + 3 red singletons, and blue has a 2-cell
#      blob + 1 blue singleton closer to the wall, red still wins because 5 > 2.
#      The output uses red and keeps only 3 outward layers beyond the opening.
#
# b) Complexity calibration:
#    - The source comparison is visible because at least two candidate colors are
#      present, and a smaller competitor is intentionally placed as close or closer
#      to the wall than the winner. That prevents the task from collapsing back to
#      the original "nearest color wins" heuristic.
#    - Ray length uses only 2-4 singleton markers. This keeps the count discrete and
#      visually readable while still producing distinct frontier lengths.
#
# c) Hidden assumptions affected by the fusion:
#    - Original code assumed one fill color and treated all pixels of that color as
#      passable seeds. That is no longer valid: non-winning components of the same
#      color must stay inert, so the solver now seeds only the winning component.
#    - Original code assumed rays extend to the border. That is no longer valid:
#      the solver now computes the infinite wedge first and then clips outward layers
#      using the singleton count.
#    - Original code implicitly let "nearest color to wall" decide the source. The
#      generator now guarantees that this heuristic disagrees with the largest-blob
#      rule in most accepted samples.
#
# d) Placement legality:
#    - Generator places all candidate blobs and singleton markers on interior cells
#      returned by the wall construction, with spacing checks so no components merge,
#      overlap, or leave the enclosure.
#
# e/f/g) Observability and cleanliness:
#    - Distinct colors label candidate families, so a human can see which blob is
#      larger and which same-color dots belong to it.
#    - The output color equals the winning family color, and the outward reach varies
#      with the count of same-color dots. This gives a direct, structural cue rather
#      than any hidden dependence on numeric color ids.
#    - No extra obstacle palette is generated, because that would clutter the same
#      enclosure region and obscure the intended largest-blob + count-to-length logic.
# ---------------------------------------------------------------------------


# -- Configuration constants -------------------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (9, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5
OPEN_END_MARGIN = 4

WINNER_BLOB_SIZES = [4, 5]
LOSER_BLOB_SIZES = [1, 2, 3]
WINNER_RAY_COUNTS = [2, 3, 4]
LOSER_MARKER_COUNTS = [1, 2]
GROUP_COUNT_CHOICES = [2]

MIN_NEW_FILL = 12
MIN_LENGTH_DIFF = 8
MIN_SOURCE_DIFF = 8
GEN_RETRIES = 500


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _coord_set(coords: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _wall_geometry(mask: np.ndarray) -> dict:
    h, w = mask.shape
    comps = [_coord_set(pts) for pts in components_mask(mask, connectivity=8)]
    assert comps, "expected at least one wall component"
    wall_comp = max(comps, key=len)

    comp_mask = np.zeros_like(mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = sorted((r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1)
    assert len(endpoints) >= 2, "expected at least two degree-1 wall endpoints"

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
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    return {
        "comp": wall_comp,
        "mask": comp_mask,
        "points": list(wall_comp),
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
    }


def _build_barrier(geom: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    barrier = np.array(geom["mask"], copy=True)
    rad = geom["rad"]

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
            else:
                if rad[0] != 0:
                    rr += d[0]
                else:
                    cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(geom["endpoints"], geom["out_dirs"], geom["tip_12"]):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)
    return barrier


def _min_dist_to_wall(points: Iterable[tuple[int, int]], wall_points: list[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in points:
        for wr, wc in wall_points:
            d = abs(r - wr) + abs(c - wc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return int(best)


def _collect_components(grid: np.ndarray, bg: int, wall: int, wall_points: list[tuple[int, int]]) -> list[dict]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, wall)):
        for pts in components_mask(grid == color, connectivity=4):
            ptset = _coord_set(pts)
            if not ptset:
                continue
            comps.append(
                {
                    "color": int(color),
                    "pts": ptset,
                    "size": int(len(ptset)),
                    "dist": _min_dist_to_wall(ptset, wall_points),
                    "anchor": min(ptset),
                }
            )
    return comps


def _pick_largest_component(comps: list[dict]) -> dict:
    assert comps, "expected at least one interior candidate component"
    return sorted(comps, key=lambda info: (-info["size"], info["dist"], info["anchor"]))[0]


def _pick_nearest_color(grid: np.ndarray, bg: int, wall: int, wall_points: list[tuple[int, int]]) -> int:
    other_colors = [int(v) for v in np.unique(grid) if int(v) not in (bg, wall)]
    assert other_colors, "expected at least one non-wall color"

    def score(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(grid == color))]
        return _min_dist_to_wall(pts, wall_points), len(pts)

    return min(other_colors, key=score)


def _bfs_fill(passable: np.ndarray, seeds: set[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q = list(seeds)
    for r, c in q:
        if 0 <= r < h and 0 <= c < w and passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _clip_by_ray_length(
    filled_mask: np.ndarray, endpoints: list[tuple[int, int]], rad: tuple[int, int], ray_len: int
) -> np.ndarray:
    if rad == (0, 0):
        return np.array(filled_mask, copy=True)
    dr, dc = rad
    rows, cols = np.indices(filled_mask.shape)
    frontier = max(r * dr + c * dc for r, c in endpoints)
    depth = rows * dr + cols * dc - frontier
    return filled_mask & (depth <= int(ray_len))


def _apply_shadows(
    out: np.ndarray, filled_mask: np.ndarray, obstacles: np.ndarray, rad: tuple[int, int], bg: int
) -> np.ndarray:
    h, w = out.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    # Shadow holes are restored to background below.
    bg_restore = []
    if rad != (0, 0):
        dr, dc = rad
        if dr != 0 and dc == 0:
            for c in range(w):
                rows = range(h) if dr > 0 else range(h - 1, -1, -1)
                bg_restore.append([(r, c) for r in rows])
        elif dc != 0 and dr == 0:
            for r in range(h):
                cols = range(w) if dc > 0 else range(w - 1, -1, -1)
                bg_restore.append([(r, c) for c in cols])
        elif dr == dc:
            for k in range(-(w - 1), h):
                coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                bg_restore.append(coords)
        else:
            for k in range(h + w - 1):
                coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                bg_restore.append(coords)

    for coords in bg_restore:
        shadow_process(coords)
    return filled_mask


def _solve_core(
    grid: np.ndarray,
    *,
    source_mode: str = "largest_component",
    finite_rays: bool = True,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    geom = _wall_geometry(g == wall)
    wall_points = geom["points"]

    comps = _collect_components(g, bg, wall, wall_points)
    assert comps, "expected at least one candidate component inside the wall"

    if source_mode == "largest_component":
        trigger = _pick_largest_component(comps)
        fill_color = int(trigger["color"])
        trigger_pts = set(trigger["pts"])
        trigger_mask = np.zeros_like(g, dtype=bool)
        for r, c in trigger_pts:
            trigger_mask[r, c] = True
        ray_len = sum(
            1 for info in comps if info["color"] == fill_color and info["size"] == 1 and info is not trigger
        )
    elif source_mode == "nearest_color":
        fill_color = _pick_nearest_color(g, bg, wall, wall_points)
        trigger_mask = g == fill_color
        trigger_pts = {(int(r), int(c)) for r, c in zip(*np.where(trigger_mask))}
        trigger = {
            "color": fill_color,
            "size": int(trigger_mask.sum()),
        }
        ray_len = max(h, w)
    else:
        raise ValueError(f"unknown source_mode: {source_mode}")

    barrier = _build_barrier(geom, g.shape)
    passable = (~barrier) & ((g == bg) | trigger_mask)
    seen = _bfs_fill(passable, trigger_pts)
    filled_mask = seen & (g == bg)

    if finite_rays and source_mode == "largest_component":
        filled_mask = _clip_by_ray_length(filled_mask, geom["endpoints"], geom["rad"], ray_len)

    out = g.copy()
    out[filled_mask] = fill_color

    # Only colors outside the candidate-family mechanism count as shadow casters.
    candidate_colors = {int(info["color"]) for info in comps}
    obstacles = np.zeros_like(g, dtype=bool)
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, wall)):
        if color in candidate_colors:
            continue
        obstacles |= g == color

    if obstacles.any():
        filled_copy = np.array(filled_mask, copy=True)
        _apply_shadows(out, filled_copy, obstacles, geom["rad"], int(bg))
        filled_mask = filled_copy

    meta = {
        "bg": int(bg),
        "wall": int(wall),
        "fill_color": int(fill_color),
        "ray_len": int(ray_len),
        "trigger_size": int(trigger["size"]),
        "rad": tuple(int(x) for x in geom["rad"]),
        "candidate_count": int(len(comps)),
        "same_color_singletons": int(
            sum(1 for info in comps if info["color"] == fill_color and info["size"] == 1)
        ),
    }
    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the wall-bounded cavity from only the largest interior blob, then emit a finite outward fan.

    Input structure:
    - bg is the dominant color. The most frequent non-bg color is a single wall component shaped like an open funnel.
    - Inside the funnel are several colored candidate families: each has one interior blob and some same-colored singleton pixels.

    Rule:
    - Compare all interior 4-connected blobs and keep only the largest one as the fill trigger; smaller blobs are ignored.
    - Flood-fill the reachable cavity and outward radiation with that winner color, but keep only as many outward layers as the number of same-colored singleton markers belonging to the winner.

    Key insight:
    - The task now requires two linked inferences: first identify which colored family wins by blob size, then count only that family's isolated same-color dots to determine the radiation length.
    """
    return _solve_core(grid, source_mode="largest_component", finite_rays=True)


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
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
            )
            >= 2
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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
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


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int] | None:
    if rad == (-1, 0):
        lo_r, hi_r = OPEN_END_MARGIN, h - depth - 3
        lo_c, hi_c = 0, w - gap - 1
    elif rad == (1, 0):
        lo_r, hi_r = depth + 2, h - OPEN_END_MARGIN - 1
        lo_c, hi_c = 0, w - gap - 1
    elif rad == (0, -1):
        lo_r, hi_r = 0, h - gap - 1
        lo_c, hi_c = OPEN_END_MARGIN, w - depth - 3
    else:
        lo_r, hi_r = 0, h - gap - 1
        lo_c, hi_c = depth + 2, w - OPEN_END_MARGIN - 1
    if lo_r > hi_r or lo_c > hi_c:
        return None
    return random.randint(lo_r, hi_r), random.randint(lo_c, hi_c)


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


def _dist_map_to_wall(cells: Iterable[tuple[int, int]], wall: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    out = {}
    for cell in cells:
        out[cell] = min(abs(cell[0] - wr) + abs(cell[1] - wc) for wr, wc in wall)
    return out


def _cheb_clear(cell: tuple[int, int], occupied: set[tuple[int, int]], gap: int = 0) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > gap for r, c in occupied)


def _grow_blob_exact(seed: tuple[int, int], allowed: set[tuple[int, int]], target: int) -> set[tuple[int, int]] | None:
    blob = {seed}
    frontier = [seed]
    while frontier and len(blob) < target:
        p = frontier.pop(random.randrange(len(frontier)))
        nbrs = [q for q in neighbors4(*p) if q in allowed and q not in blob]
        random.shuffle(nbrs)
        for q in nbrs:
            blob.add(q)
            frontier.append(q)
            if len(blob) >= target:
                break
        if not nbrs and p in frontier:
            frontier.remove(p)
    return blob if len(blob) == target else None


def _place_blob(
    free_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target: int,
    *,
    preferred_rows: tuple[int, int] | None = None,
    require_dist: int | None = None,
    dist_map: dict[tuple[int, int], int] | None = None,
) -> set[tuple[int, int]] | None:
    seeds = list(free_cells)
    random.shuffle(seeds)
    for seed in seeds:
        if not _cheb_clear(seed, occupied):
            continue
        if preferred_rows is not None and not (preferred_rows[0] <= seed[0] <= preferred_rows[1]):
            continue
        if require_dist is not None and (dist_map is None or dist_map[seed] != require_dist):
            continue
        allowed = {
            cell
            for cell in free_cells
            if _cheb_clear(cell, occupied)
            and (preferred_rows is None or preferred_rows[0] <= cell[0] <= preferred_rows[1])
        }
        blob = _grow_blob_exact(seed, allowed, target)
        if blob is None:
            continue
        if require_dist is not None and dist_map is not None and min(dist_map[p] for p in blob) > require_dist:
            continue
        return blob
    return None


def _pick_markers(
    free_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    same_color_cells: set[tuple[int, int]],
    count: int,
) -> set[tuple[int, int]] | None:
    center_r = sum(r for r, _ in same_color_cells) / len(same_color_cells)
    center_c = sum(c for _, c in same_color_cells) / len(same_color_cells)
    candidates = list(free_cells)
    random.shuffle(candidates)
    candidates.sort(key=lambda p: (abs(p[0] - center_r) + abs(p[1] - center_c), p))
    chosen: set[tuple[int, int]] = set()
    for cell in candidates:
        if not _cheb_clear(cell, occupied | chosen):
            continue
        if any(abs(cell[0] - r) + abs(cell[1] - c) <= 1 for r, c in same_color_cells | chosen):
            continue
        chosen.add(cell)
        if len(chosen) == count:
            return chosen
    return None


def _place_candidate_families(local_wall: set[tuple[int, int]], depth: int):
    interior = _local_interior(local_wall, depth)
    interior_set = set(interior)
    if len(interior_set) < 12:
        return None

    dist_map = _dist_map_to_wall(interior_set, local_wall)
    deep_rows = (max(1, depth - 5), depth - 1)
    occupied: set[tuple[int, int]] = set()

    winner_size = random.choice(WINNER_BLOB_SIZES)
    winner_markers = random.choice(WINNER_RAY_COUNTS)
    group_count = random.choice(GROUP_COUNT_CHOICES)
    loser_specs = []
    for _ in range(group_count - 1):
        loser_specs.append((random.choice(LOSER_BLOB_SIZES), random.choice(LOSER_MARKER_COUNTS)))

    # Place one bait group close to a wall so the original nearest-color heuristic would prefer it.
    free = interior_set - occupied
    bait_blob = _place_blob(free, occupied, loser_specs[0][0], preferred_rows=deep_rows, require_dist=1, dist_map=dist_map)
    if bait_blob is None:
        return None
    occupied |= bait_blob

    free = interior_set - occupied
    winner_blob = _place_blob(free, occupied, winner_size, dist_map=dist_map)
    if winner_blob is None:
        return None
    occupied |= winner_blob

    other_blobs = []
    for size, _markers in loser_specs[1:]:
        free = interior_set - occupied
        blob = _place_blob(free, occupied, size, preferred_rows=deep_rows, dist_map=dist_map)
        if blob is None:
            return None
        occupied |= blob
        other_blobs.append(blob)

    groups = [{"blob": winner_blob, "markers": winner_markers, "role": "winner"}]
    groups.append({"blob": bait_blob, "markers": loser_specs[0][1], "role": "bait"})
    for blob, (_size, markers) in zip(other_blobs, loser_specs[1:]):
        groups.append({"blob": blob, "markers": markers, "role": "other"})

    marker_groups = []
    for group in groups:
        free = interior_set - occupied
        markers = _pick_markers(free, occupied, set(group["blob"]), int(group["markers"]))
        if markers is None:
            return None
        occupied |= markers
        marker_groups.append(markers)

    placed = []
    for group, markers in zip(groups, marker_groups):
        placed.append(
            {
                "blob": set(group["blob"]),
                "markers": set(markers),
                "role": group["role"],
                "blob_size": len(group["blob"]),
                "marker_count": len(markers),
                "min_wall_dist": min(dist_map[p] for p in group["blob"]),
            }
        )

    winner = next(info for info in placed if info["role"] == "winner")
    if any(info["blob_size"] >= winner["blob_size"] for info in placed if info["role"] != "winner"):
        return None
    if placed[1]["min_wall_dist"] > winner["min_wall_dist"]:
        return None
    return placed


def _render_groups(
    grid: np.ndarray,
    groups: list[dict],
    colors: list[int],
    rad: tuple[int, int],
    anchor: tuple[int, int],
    bg: int,
) -> dict:
    assert len(groups) == len(colors)
    meta = {}
    for group, color in zip(groups, colors):
        cells = group["blob"] | group["markers"]
        for u, v in cells:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return {}
            grid[r, c] = color
        if group["role"] == "winner":
            meta = {
                "winner_color": int(color),
                "winner_blob_size": int(group["blob_size"]),
                "winner_markers": int(group["marker_count"]),
            }
    return meta


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)
    if anchor is None:
        return None

    groups = _place_candidate_families(local_wall, depth)
    if groups is None:
        return None

    palette = random.sample(range(10), 2 + len(groups))
    bg, wall_color = palette[:2]
    group_colors = palette[2:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    render_meta = _render_groups(grid, groups, group_colors, rad, anchor, bg)
    if not render_meta:
        return None

    try:
        out, new_meta = _solve_core(grid, source_mode="largest_component", finite_rays=True, return_meta=True)
        inf_out, _ = _solve_core(grid, source_mode="largest_component", finite_rays=False, return_meta=True)
        old_out, old_meta = _solve_core(grid, source_mode="nearest_color", finite_rays=False, return_meta=True)
    except Exception:
        return None

    if new_meta["fill_color"] != render_meta["winner_color"]:
        return None
    if new_meta["trigger_size"] != render_meta["winner_blob_size"]:
        return None
    if new_meta["ray_len"] != render_meta["winner_markers"]:
        return None
    if old_meta["fill_color"] == render_meta["winner_color"]:
        return None

    new_fill = (out == render_meta["winner_color"]) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None

    length_diff = int(np.count_nonzero(out != inf_out))
    source_diff = int(np.count_nonzero(out != old_out))
    if length_diff < MIN_LENGTH_DIFF or source_diff < MIN_SOURCE_DIFF:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input for b9e38dc0")

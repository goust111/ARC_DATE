# Combo variant of b9e38dc0 (index 21)
# combo_spec: D11/0+D19/2
# Name: 孤立像素异色+颜色匹配关联
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    - Keep the original visual language: each sprite is a partial enclosure whose
#      interior source blob flood-fills outward through the opening.
#    - Make the fusion visible by using two sprites with different opening
#      directions and different border colors.  Each sprite also has a distinct
#      interior source color.
#    - Add one isolated singleton per sprite, but give that singleton the border
#      color instead of the source color.  The singleton is placed near the other
#      sprite so "nearest wall" is misleading on purpose.
#    - In the output, every singleton moves along the direction of the sprite
#      whose border color it matches, and its whole trail is painted in the
#      singleton's own color.  Because singleton color == border color != source
#      color, the trail is visibly different from the flood color.
# b) Complexity calibration:
#    - One sprite would trivialize color pairing; two sprites are the minimum that
#      make pairing and misdirection observable.
#    - Exactly one vertical and one horizontal sprite are used in each sample, so
#      the two possible singleton motions differ clearly without creating too many
#      cases.
#    - The marker count is fixed at two, which keeps the scene readable.
# c) Hidden assumptions audited from the original solver:
#    - Original code assumed a single dominant wall color.  That breaks here, so
#      the new solver detects wall components by geometry (large 8-connected
#      components with exactly two degree-1 endpoints), not by frequency.
#    - Original code assumed the closest non-wall color to a wall is the unique
#      source.  That can still work only if markers are excluded first, so the new
#      solver ignores singleton components whose color matches a wall color.
#    - Original flood-fill logic depends on each source blob being close to only
#      one wall.  The generator enforces wide separation so nearest-source
#      assignment remains valid.
# d) Placement legality:
#    - Generator rejects any overlap between walls, seeds, fill zones, and marker
#      launch rays that would create accidental collisions or out-of-bounds writes.
#    - Marker starts are background cells, 8-isolated from existing objects, and
#      their travel rays do not hit any wall cells.
# e) Visual observability:
#    - Across examples, humans can see that a singleton shares color with a wall
#      but not with the filled region it influences.  The two sprites have
#      different directions, so the association affects a clearly observable part
#      of the output rather than only an internal label.
# f) Visual cleanliness:
#    - Only two sprites are used, their nominal fill regions are forced not to
#      overlap, and each marker ray avoids walls.  This keeps the canvas legible
#      even though one colored trail may cross another sprite's fill fan.
# g) Inferability of color change:
#    - Marker color is never equal to its source color.  Instead it always equals
#      one border color, and every sample repeats the same cue twice.  The scene
#      therefore teaches the rule "pair singleton to matching border color" rather
#      than leaving color changes arbitrary.

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# --- Shared geometry configuration -------------------------------------------

GRID_H = (14, 18)
GRID_W = (16, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 8)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.45
OPEN_END_MARGIN = 3
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4]
SEED_DEPTH_LOOKBACK = 3

WALL_DETECT_MIN = 12
MIN_NEW_FILL_PER_SPRITE = 14
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(
    depth: int,
    start_v: int,
    target_v: int,
    extra_steps: int,
    tip_12: bool = False,
) -> list[tuple[int, int]]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows: list[int] = []
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
        max_depth = min(DEPTH_HORIZ[1], w - 6)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
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
    interior: list[tuple[int, int]] = []
    for u in range(1, depth):
        vals = by_row.get(u)
        if vals is None or len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target:
                    break
    return blob


def _comp_mask(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _wall_info_from_points(
    points: set[tuple[int, int]],
    color: int,
    shape: tuple[int, int],
) -> dict[str, object]:
    h, w = shape
    comp_mask = _comp_mask(points, shape)

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in points if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) == 2, "wall must have exactly two endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad in {(-1, 0), (1, 0), (0, -1), (0, 1)}, "radiate direction must be cardinal"

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
            else:
                if rad[0] != 0:
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

    return {
        "color": int(color),
        "points": set(points),
        "comp_mask": comp_mask,
        "barrier": barrier,
        "rad": rad,
        "endpoints": endpoints,
    }


def _mono_components(grid: np.ndarray, colors: list[int], connectivity: int = 4) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in colors:
        for comp in components_mask(grid == color, connectivity=connectivity):
            pts = [(int(r), int(c)) for r, c in comp]
            comps.append({"color": int(color), "points": pts})
    return comps


def _dist_between(a: list[tuple[int, int]] | set[tuple[int, int]], b: list[tuple[int, int]] | set[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in a:
        for rr, cc in b:
            d = abs(r - rr) + abs(c - cc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _reachable_bg_from_seeds(
    bg_mask: np.ndarray,
    barrier: np.ndarray,
    seed_points: list[tuple[int, int]],
) -> np.ndarray:
    passable = (~barrier) & bg_mask
    seen = np.zeros_like(passable, dtype=bool)
    q = deque()
    for r, c in seed_points:
        if 0 <= r < passable.shape[0] and 0 <= c < passable.shape[1] and not barrier[r, c]:
            if not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1]:
                if passable[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
    return seen


def _line_cells(start: tuple[int, int], d: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    coords = []
    while 0 <= rr < h and 0 <= cc < w:
        coords.append((rr, cc))
        rr += d[0]
        cc += d[1]
    return coords


def _detect_scene(grid: np.ndarray):
    g = np.array(grid, copy=False)
    bg = int(get_bg(g))
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    walls: list[dict[str, object]] = []
    wall_cells: set[tuple[int, int]] = set()
    wall_by_color: dict[int, dict[str, object]] = {}
    for color in colors:
        for comp in components_mask(g == color, connectivity=8):
            pts = {(int(r), int(c)) for r, c in comp}
            if len(pts) < WALL_DETECT_MIN:
                continue
            if len(_degree1_pixels(pts)) != 2:
                continue
            info = _wall_info_from_points(pts, color=color, shape=g.shape)
            walls.append(info)
            wall_cells |= pts
            wall_by_color[int(color)] = info

    assert len(walls) >= 2, "expected at least two wall sprites"

    source_candidates = []
    for comp in _mono_components(g, colors, connectivity=4):
        pts = comp["points"]
        color = int(comp["color"])
        pt_set = set(pts)
        if pt_set <= wall_cells:
            continue
        if len(pts) == 1 and color in wall_by_color:
            continue
        source_candidates.append({"color": color, "points": pts})

    assert len(source_candidates) >= len(walls), "missing source blobs"

    available = source_candidates[:]
    for wall in walls:
        best_idx = min(
            range(len(available)),
            key=lambda i: (
                _dist_between(wall["points"], available[i]["points"]),
                len(available[i]["points"]),
            ),
        )
        wall["source"] = available.pop(best_idx)

    markers = []
    for color, wall in wall_by_color.items():
        for comp in components_mask(g == color, connectivity=8):
            pts = [(int(r), int(c)) for r, c in comp]
            if len(pts) == 1:
                markers.append({"color": color, "point": pts[0], "wall": wall})

    assert len(markers) >= len(walls), "missing wall-colored markers"
    return bg, walls, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each partial enclosure, then launch swapped singletons by border-color matching.

    Input structure:
    - bg is the dominant color.  There are two large wall sprites, each a partial
      enclosure with its own border color and its own interior source blob color.
    - Each border color also appears once as an isolated singleton, but that
      singleton sits near the other sprite rather than near the one it belongs to.
    - Singleton color is intentionally different from the source color; the
      matching cue is border color only.

    Rule:
    - For every wall sprite, flood-fill reachable background from the interior
      source blob and let that color radiate outward through the opening, using
      the same wedge geometry as the base task.
    - For every isolated singleton, find the wall sprite whose border color
      matches the singleton color, then extend a straight trail from that
      singleton to the grid edge in that sprite's outward direction.
    - The trail uses the singleton's own color, not the source color and not the
      background.

    Key insight:
    - Spatial proximity is now a trap: the singleton is positioned beside the
      wrong sprite.  The only stable association signal is color equality between
      the singleton and one border, and that matched border also supplies the
      motion direction.
    """
    g = np.array(grid, copy=True)
    bg, walls, markers = _detect_scene(g)

    out = g.copy()
    bg_mask = g == bg
    wall_masks = np.zeros_like(g, dtype=bool)
    for wall in walls:
        wall_masks |= wall["comp_mask"]

    for wall in walls:
        source = wall["source"]
        seed_points = list(source["points"])
        reach = _reachable_bg_from_seeds(
            bg_mask | _comp_mask(set(seed_points), g.shape),
            wall["barrier"],
            seed_points,
        )
        filled_mask = reach & bg_mask
        out[filled_mask] = int(source["color"])

    for marker in markers:
        color = int(marker["color"])
        wall = marker["wall"]
        start = marker["point"]
        for r, c in _line_cells(start, wall["rad"], g.shape):
            if wall_masks[r, c]:
                break
            out[r, c] = color

    return out


# --------------- generator ---------------------------------------------------


def _expand(points: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in points:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _render_sprite(
    grid: np.ndarray,
    local_wall: set[tuple[int, int]],
    depth: int,
    rad: tuple[int, int],
    anchor: tuple[int, int],
    wall_color: int,
    source_color: int,
):
    wall_points = {_map_local(u, v, rad, anchor) for u, v in local_wall}
    if any(grid[r, c] != get_bg(grid) for r, c in wall_points):
        return None
    for r, c in wall_points:
        grid[r, c] = wall_color

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

    blob_local = _grow_seed_blob(random.choice(seed_pool), set(interior))
    blob_points = {_map_local(u, v, rad, anchor) for u, v in blob_local}
    if any(grid[r, c] != get_bg(grid) for r, c in blob_points):
        return None
    for r, c in blob_points:
        grid[r, c] = source_color

    return {
        "wall_points": wall_points,
        "blob_points": blob_points,
        "info": _wall_info_from_points(wall_points, wall_color, grid.shape),
    }


def _nominal_fill_mask(grid: np.ndarray, wall_info: dict[str, object], blob_points: set[tuple[int, int]], bg: int) -> np.ndarray:
    bg_mask = grid == bg
    reach = _reachable_bg_from_seeds(
        bg_mask | _comp_mask(blob_points, grid.shape),
        wall_info["barrier"],
        list(blob_points),
    )
    return reach & bg_mask


def _min_dist_to_points(cell: tuple[int, int], points: set[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - rr) + abs(c - cc) for rr, cc in points)


def _marker_candidates(
    grid: np.ndarray,
    own_wall: dict[str, object],
    wrong_wall: dict[str, object],
    occupied: set[tuple[int, int]],
    wrong_fill: np.ndarray,
    all_fill: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    bg = int(get_bg(grid))
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg:
                continue
            if not wrong_fill[r, c]:
                continue
            cell = (r, c)
            if any((rr, cc) in occupied for rr, cc in _expand({cell}, grid.shape, radius=1)):
                continue
            d_wrong = _min_dist_to_points(cell, wrong_wall["points"])
            d_own = _min_dist_to_points(cell, own_wall["points"])
            if d_wrong > 5 or d_own <= d_wrong:
                continue
            ray = _line_cells(cell, own_wall["rad"], grid.shape)
            blocked = False
            for rr, cc in ray[1:]:
                if (rr, cc) in occupied:
                    blocked = True
                    break
            if blocked:
                continue
            if sum(all_fill[rr, cc] for rr, cc in ray) > max(1, len(ray) // 2):
                continue
            out.append(cell)
    random.shuffle(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    bg, wall1, wall2, src1, src2 = random.sample(range(10), 5)
    rad1 = random.choice([(-1, 0), (1, 0)])
    rad2 = random.choice([(0, -1), (0, 1)])

    built1 = _build_local_wall(rad1, h, w)
    built2 = _build_local_wall(rad2, h, w)
    if built1 is None or built2 is None:
        return None
    local1, depth1, _ = built1
    local2, depth2, _ = built2

    for _ in range(120):
        grid = np.full((h, w), bg, dtype=int)
        anchor1 = _place_anchor(rad1, h, w, depth1, max(v for _, v in local1))
        anchor2 = _place_anchor(rad2, h, w, depth2, max(v for _, v in local2))

        sprite1 = _render_sprite(grid, local1, depth1, rad1, anchor1, wall1, src1)
        if sprite1 is None:
            continue

        used = _expand(sprite1["wall_points"] | sprite1["blob_points"], grid.shape, radius=1)
        sprite2 = _render_sprite(grid, local2, depth2, rad2, anchor2, wall2, src2)
        if sprite2 is None:
            continue
        second_occ = sprite2["wall_points"] | sprite2["blob_points"]
        if used & second_occ:
            continue

        fill1 = _nominal_fill_mask(grid, sprite1["info"], sprite1["blob_points"], bg)
        fill2 = _nominal_fill_mask(grid, sprite2["info"], sprite2["blob_points"], bg)
        if fill1.sum() < MIN_NEW_FILL_PER_SPRITE or fill2.sum() < MIN_NEW_FILL_PER_SPRITE:
            continue
        if np.any(fill1 & fill2):
            continue

        occupied = sprite1["wall_points"] | sprite1["blob_points"] | sprite2["wall_points"] | sprite2["blob_points"]
        all_fill = fill1 | fill2
        cand1 = _marker_candidates(grid, sprite1["info"], sprite2["info"], occupied, fill2, all_fill)
        cand2 = _marker_candidates(grid, sprite2["info"], sprite1["info"], occupied, fill1, all_fill)
        if not cand1 or not cand2:
            continue

        placed = None
        for p1 in cand1[:20]:
            filtered2 = [
                p2
                for p2 in cand2
                if p2 != p1 and not (_expand({p1}, grid.shape, 1) & {p2})
            ]
            if not filtered2:
                continue
            placed = (p1, filtered2[0])
            break
        if placed is None:
            continue

        p1, p2 = placed
        grid[p1] = wall1
        grid[p2] = wall2

        try:
            out = main(grid)
        except Exception:
            continue

        if out[p1] != wall1 or out[p2] != wall2:
            continue
        if any(out[r, c] != wall1 for r, c in sprite1["wall_points"]):
            continue
        if any(out[r, c] != wall2 for r, c in sprite2["wall_points"]):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")

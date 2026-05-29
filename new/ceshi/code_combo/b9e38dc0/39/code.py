# Combo variant of b9e38dc0 (index 39)
# combo_spec: D5/1+D18/1
# Name: 半程停止+像素计数终止

import os
import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_cdt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
SOLVER_DIR = os.path.join(ROOT, "solvers")
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for combo D5/1 + D18/1
# ---------------------------------------
# a) Concrete behavior and trace
#    - Baseline stage 1: a wall-colored funnel encloses a small fill blob; the blob flood-fills the interior and
#      then keeps radiating outward through the opening inside the wedge bounded by the two endpoint rays.
#    - Baseline stage 2 in the task family: extra isolated singleton pixels move straight along the same cardinal
#      opening direction toward the corresponding grid edge, leaving their old cells as background.
#    - Fused stage 1 here: keep the same funnel geometry, but compute the unlimited wedge first and then truncate
#      the outward part by a numeric cap. If the seed blob has 4 pixels, only cells whose forward depth from the
#      opening is at most 4 stay filled; the old edge-reaching tail disappears.
#    - Fused stage 2 here: each isolated mover advances only `dist_to_edge // 2` cells instead of all the way to
#      the border. Example with `rad = down`: a mover 7 steps from the bottom used to land on the bottom border;
#      now it moves only 3 steps and stops mid-column, while the start and traversed cells become background.
#    - Difference summary: the same input now encodes two separate quantities. The seed-pixel count caps the wedge,
#      while each mover's own distance to the edge caps only that mover. Humans must not conflate those numbers.
# b) Complexity calibration
#    - The wall/opening detection and the notion "all movers follow the funnel's cardinal direction" are preserved,
#      so the task is still recognizable as b9e38dc0 rather than a different puzzle family.
#    - The generator avoids degenerate 1-pixel rays dominating the sample set by biasing seed blobs toward sizes
#      2-6, and it forces at least two movers with travel >= 2. This keeps both numeric rules visible in 2-4 I/O
#      examples.
#    - No extra hidden coding such as color ordering is introduced; all control is geometric or count-based.
# c) Implicit-assumption audit against the original solver skeleton
#    - `wall = most frequent non-bg color` still holds because the wall stays much larger than all other colors.
#    - `fill = non-wall color closest to wall` remains valid only if movers stay far from the wall; the generator
#      enforces a taxicab distance moat around the wall when placing movers.
#    - The old ray-to-edge barrier assumed border termination. That assumption breaks under D5/1, so the solver
#      first computes the full wedge and then clips filled cells by forward depth from the opening line.
#    - The old family may contain extra colored pixels that behave as moving singletons. The new solver therefore
#      treats only 8-isolated size-1 extra components as movers; anything larger is preserved as inert clutter.
#    - The half-distance mover rule assumes a clear straight lane. Rather than adding a new collision semantics not
#      mentioned by D18/1, the generator guarantees that every chosen mover's half-path lies entirely on background
#      cells after stage 1.
# d) Placement legality
#    - Wall construction still guarantees one 8-connected funnel with exactly two endpoints.
#    - The anchor placement plus later validation ensure the opening has more free depth than the seed count, so the
#      count-based truncation actually changes the output relative to the original edge-reaching version.
#    - Mover candidates are sampled only from cells that stay background after stage 1 and whose whole half-path is
#      background, in-bounds, and non-overlapping with other movers' starts and destinations.
# e) Visual observability
#    - The truncated wedge is directly visible because the output no longer reaches the border in the radiate
#      direction, and different training examples can have different wedge depths as seed counts vary.
#    - Movers provide a second, independent cue: they all point in the same direction, but identical direction does
#      not imply identical travel because each one stops at half its own border distance.
# f) Visual cleanliness
#    - Only one large wall/fill structure and a few singleton movers are present. There are no extra obstacle colors
#      placed in the wedge, which avoids clutter obscuring the new numeric termination rules.
#    - Mover spacing constraints prevent output pixels from merging into larger blobs that would blur the notion of
#      "isolated pixels".
# g) New assumptions checked against the generator
#    - `forward_depth <= seed_count` is meaningful only if the unlimited wedge would extend farther; validation
#      explicitly rejects cases where the limited and unlimited stage-1 outputs coincide.
#    - Singleton detection for movers assumes the generator uses only isolated extra pixels. The generator does so by
#      placing movers one-by-one with Chebyshev separation.
#    - The final output paint step assumes mover destinations are unique. Candidate selection enforces unique starts
#      and unique destinations.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _component_sets(mask: np.ndarray, connectivity: int = 8) -> list[set[tuple[int, int]]]:
    return [set((int(r), int(c)) for r, c in pts) for pts in components_mask(mask, connectivity=connectivity)]


def _component_mask(shape: tuple[int, int], comp: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _outward_dir(comp_mask: np.ndarray, ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    n1 = _comp_neighbors(comp_mask, *ep)[0]
    d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
    d_use = d1
    is_12 = False
    n2s = [p for p in _comp_neighbors(comp_mask, *n1) if p != ep]
    if n2s:
        n2 = n2s[0]
        d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
        if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
            d_use = d2
            is_12 = True
    return (-d_use[0], -d_use[1]), is_12


def _trace_ray_to_edge(
    start: tuple[int, int], d: tuple[int, int], rad: tuple[int, int], is_12: bool, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    step = 0
    path: list[tuple[int, int]] = []
    while True:
        step += 1
        if not is_12 or step % 2 == 1:
            nr = rr + d[0]
            nc = cc + d[1]
        else:
            if rad[0] != 0:
                nr = rr + d[0]
                nc = cc
            else:
                nr = rr
                nc = cc + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            break
        path.append((nr, nc))
        rr, cc = nr, nc
    return path


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = _component_sets(wall_mask, connectivity=8)
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"
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
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return {"grid": g, "bg": bg, "wall": wall, "fill": fill, "fill_pos": []}

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
    comp_mask = _component_mask((h, w), wall_comp)
    endpoints = sorted([(r, c) for r, c in wall_comp if len(_comp_neighbors(comp_mask, r, c)) == 1])
    assert len(endpoints) >= 2, "expected wall component to have endpoints"
    out_info = [_outward_dir(comp_mask, ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad != (0, 0), "expected a non-zero opening direction"

    barrier = comp_mask.copy()
    ray_paths = []
    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        path = _trace_ray_to_edge(ep, d, rad, is_12, (h, w))
        ray_paths.append(path)
        for r, c in path:
            barrier[r, c] = True

    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
        "barrier": barrier,
        "ray_paths": ray_paths,
        "seed_count": len(fill_pos),
    }


def _projection(rad: tuple[int, int], r: int, c: int) -> int:
    return int(r * rad[0] + c * rad[1])


def _forward_depth(scene: dict, r: int, c: int) -> int:
    base = max(_projection(scene["rad"], rr, cc) for rr, cc in scene["endpoints"])
    return _projection(scene["rad"], r, c) - base


def _solve_stage1(grid: np.ndarray, ray_limit: int | None = None) -> tuple[np.ndarray, dict]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    if not scene["fill_pos"]:
        meta = {
            "scene": scene,
            "filled_mask": np.zeros_like(g, dtype=bool),
            "sprite_mask": np.zeros_like(g, dtype=bool),
            "external_depth": 0,
        }
        return g.copy(), meta

    passable = (~scene["barrier"]) & ((g == scene["bg"]) | (g == scene["fill"]))
    q = scene["fill_pos"][:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_full = seen & (g == scene["bg"])
    filled_mask = filled_full.copy()
    if ray_limit is not None:
        for r, c in zip(*np.where(filled_mask)):
            depth = _forward_depth(scene, int(r), int(c))
            if depth > ray_limit:
                filled_mask[int(r), int(c)] = False

    out = g.copy()
    out[filled_mask] = scene["fill"]
    sprite_mask = scene["comp_mask"] | filled_mask | (g == scene["fill"])
    external_depth = 0
    for r, c in zip(*np.where(filled_mask)):
        external_depth = max(external_depth, max(0, _forward_depth(scene, int(r), int(c))))
    meta = {"scene": scene, "filled_mask": filled_mask, "sprite_mask": sprite_mask, "external_depth": external_depth}
    return out, meta


def _singleton_movers(grid: np.ndarray, bg: int, wall: int, fill: int) -> list[tuple[int, int, int]]:
    extras = (grid != bg) & (grid != wall) & (grid != fill)
    movers = []
    for pts in components_mask(extras, connectivity=8):
        if pts.shape[0] != 1:
            continue
        r, c = pts[0]
        movers.append((int(r), int(c), int(grid[int(r), int(c)])))
    movers.sort()
    return movers


def _dist_to_edge(pos: tuple[int, int], rad: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = pos
    h, w = shape
    if rad == (-1, 0):
        return r
    if rad == (1, 0):
        return h - 1 - r
    if rad == (0, -1):
        return c
    return w - 1 - c


def _path_cells(start: tuple[int, int], rad: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    return [(r + rad[0] * k, c + rad[1] * k) for k in range(1, steps + 1)]


def _apply_movers(grid: np.ndarray, stage1_out: np.ndarray, stage1_meta: dict) -> tuple[np.ndarray, dict]:
    out = np.array(stage1_out, copy=True)
    scene = stage1_meta["scene"]
    movers = _singleton_movers(grid, scene["bg"], scene["wall"], scene["fill"])
    placements = []
    for r, c, color in movers:
        dist = _dist_to_edge((r, c), scene["rad"], grid.shape)
        steps = dist // 2
        rr = r + scene["rad"][0] * steps
        cc = c + scene["rad"][1] * steps
        placements.append({"start": (r, c), "end": (rr, cc), "color": color, "dist": dist, "steps": steps})

    for rec in placements:
        r, c = rec["start"]
        out[r, c] = scene["bg"]
    for rec in placements:
        rr, cc = rec["end"]
        out[rr, cc] = rec["color"]
    return out, {"movers": placements}


def _solve(grid: np.ndarray, return_meta: bool = False):
    stage1_out, stage1_meta = _solve_stage1(grid, ray_limit=_parse_scene(grid)["seed_count"])
    out, mover_meta = _apply_movers(grid, stage1_out, stage1_meta)
    if not return_meta:
        return out
    unlimited_out, unlimited_meta = _solve_stage1(grid, ray_limit=None)
    return out, {"stage1": stage1_meta, "stage1_unlimited": unlimited_meta, "movers": mover_meta["movers"], "unlimited_out": unlimited_out}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the funnel interior, cap the outward wedge by seed-pixel count, and move isolated pixels only halfway to the same edge.

    Input structure:
    - The most common color is background. The most frequent non-background color forms one partial wall with a
      single open side, and a small blob of another color is placed inside that enclosure near the closed end.
    - Any remaining extra non-background singletons are isolated movers sitting away from the wall.

    Rule:
    - First flood-fill the enclosure from the inner blob and let that color radiate through the opening in the same
      wedge shape as the original task.
    - Do not keep radiating to the border: only outward cells whose forward depth from the opening is at most the
      number of source pixels in the inner blob remain filled.
    - Then move each isolated singleton straight along the same opening direction by exactly half of its own current
      distance to the relevant grid edge, rounded down, and erase its old location to background.

    Key insight: Two different numeric caps govern two different object types: seed-pixel count limits the radiating
    fill, while each singleton's own edge distance limits only that singleton's travel.
    """
    return _solve(grid)


# --------------- generator ---------------

GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [2, 2, 3, 3, 4, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MIN_MOVERS = 2
MAX_MOVERS = 4
MOVER_WALL_DIST = 3
MOVER_MIN_SEP = 2
MOVER_MIN_STEPS = 2

MIN_NEW_FILL = 12
MIN_FILL_DIFF = 6
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 500


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


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall: set[tuple[int, int]], depth: int, gap: int) -> list[tuple[int, int]]:
    by_row = {}
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
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _chebyshev_far_enough(pos: tuple[int, int], placed: list[tuple[int, int]], min_sep: int) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in placed)


def _candidate_movers(base_grid: np.ndarray, stage1_out: np.ndarray, stage1_meta: dict) -> list[dict]:
    scene = stage1_meta["scene"]
    bg = scene["bg"]
    wall_mask = base_grid == scene["wall"]
    wall_dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    free = (base_grid == bg) & (stage1_out == bg)
    coords = [(int(r), int(c)) for r, c in zip(*np.where(free & (wall_dist >= MOVER_WALL_DIST)))]
    random.shuffle(coords)

    records = []
    used_ends = set()
    for start in coords:
        dist = _dist_to_edge(start, scene["rad"], base_grid.shape)
        steps = dist // 2
        if steps < MOVER_MIN_STEPS:
            continue
        path = _path_cells(start, scene["rad"], steps)
        if not path:
            continue
        if any(stage1_out[r, c] != bg for r, c in path):
            continue
        end = path[-1]
        if end in used_ends:
            continue
        records.append({"start": start, "end": end, "steps": steps, "dist": dist})
        used_ends.add(end)
    return records


def _pick_movers(records: list[dict]) -> list[dict]:
    if len(records) < MIN_MOVERS:
        return []
    random.shuffle(records)
    target_n = random.randint(MIN_MOVERS, min(MAX_MOVERS, len(records)))
    chosen = []
    used_starts: list[tuple[int, int]] = []
    used_ends: list[tuple[int, int]] = []
    for rec in records:
        if not _chebyshev_far_enough(rec["start"], used_starts, MOVER_MIN_SEP):
            continue
        if not _chebyshev_far_enough(rec["end"], used_ends, MOVER_MIN_SEP):
            continue
        chosen.append(rec)
        used_starts.append(rec["start"])
        used_ends.append(rec["end"])
        if len(chosen) >= target_n:
            break
    if len(chosen) < MIN_MOVERS:
        return []
    return chosen


def _variant_metrics(grid: np.ndarray) -> dict:
    out, meta = _solve(grid, return_meta=True)
    scene = meta["stage1"]["scene"]
    limited_fill = meta["stage1"]["filled_mask"]
    _ = out
    _, unlimited_meta = _solve_stage1(grid, ray_limit=None)
    unlimited_fill = unlimited_meta["filled_mask"]
    fill_diff = int(unlimited_fill.sum() - limited_fill.sum())
    mover_steps = [rec["steps"] for rec in meta["movers"]]
    mover_dists = [rec["dist"] for rec in meta["movers"]]
    return {
        "seed_count": scene["seed_count"],
        "limited_depth": meta["stage1"]["external_depth"],
        "unlimited_depth": unlimited_meta["external_depth"],
        "fill_diff": fill_diff,
        "movers": len(meta["movers"]),
        "mover_steps": mover_steps,
        "mover_dists": mover_dists,
    }


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    mover_colors = palette[3:]
    if len(mover_colors) == 0:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth, gap)
    interior_set = set(interior)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), interior_set)
    if len(blob) < 2:
        return None
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        limited_stage1_out, limited_stage1_meta = _solve_stage1(grid, ray_limit=len(blob))
        unlimited_stage1_out, unlimited_stage1_meta = _solve_stage1(grid, ray_limit=None)
    except Exception:
        return None

    limited_new_fill = (limited_stage1_out == fill_color) & (grid == bg)
    unlimited_new_fill = (unlimited_stage1_out == fill_color) & (grid == bg)
    fill_diff = int(unlimited_new_fill.sum() - limited_new_fill.sum())
    if int(limited_new_fill.sum()) < MIN_NEW_FILL:
        return None
    if fill_diff < MIN_FILL_DIFF:
        return None
    if unlimited_stage1_meta["external_depth"] <= len(blob):
        return None
    if limited_stage1_meta["external_depth"] != len(blob):
        return None

    records = _candidate_movers(grid, limited_stage1_out, limited_stage1_meta)
    chosen = _pick_movers(records)
    if len(chosen) < MIN_MOVERS:
        return None

    for i, rec in enumerate(chosen):
        r, c = rec["start"]
        grid[r, c] = mover_colors[i % len(mover_colors)]

    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return None

    mover_steps = [rec["steps"] for rec in meta["movers"]]
    if len(mover_steps) < MIN_MOVERS or min(mover_steps) < MOVER_MIN_STEPS:
        return None
    if len({tuple(rec["end"]) for rec in meta["movers"]}) != len(meta["movers"]):
        return None
    if np.array_equal(out, limited_stage1_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")

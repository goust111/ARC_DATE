# Variant of 8b7bacbf
# Dimension: D2 — 路径结构形态 (path)
# Change: 蜿蜒线路（meandering path）——8连通的1像素宽蜿蜒路径，使用对角线和正交步混合，从边缘种子向网格内部延伸... → 分支路径（树形结构）
#
# Design analysis
# ---------------
# variant_spec:
#   input_objects:
#     bg: most common color, still 0 or 5
#     seeds:
#       count: 1-2
#       form: unique-color single pixels on the border, usually at corners
#     line_networks:
#       count: 1 per seed
#       color: one line_color per seed, distinct from seed/frame/bg
#       structure:
#         original: one meandering 8-connected 1px path
#         variant: one meandering 8-connected 1px tree grown from the seed's first line
#         branching_rule:
#           - the trunk grows first
#           - extra branches can only sprout from bend nodes (degree-2 line pixels whose two
#             incident directions are not opposites)
#           - this is necessary because in 8-connectivity, branching from a straight segment
#             would make the new pixel touch both side-neighbors and accidentally create a cycle
#       visible_effect:
#         - multiple leaves extend into different interior regions
#         - loops can be attached on different branches, so multiple separated cavities fill
#           with the same seed color through one branched network
#     frame_loops:
#       same closed single-color rounded-rect rings as the source task
#       attached_loops:
#         - each branch gets at least one attached loop when possible
#         - each seed also gets at least one trunk-attached loop
#       distractors:
#         - 0-2 unattached loops remain possible
#
#   behavior_trace:
#     concrete_small_example:
#       - seed S at top-left corner, first line pixel L at (0,1)
#       - trunk bends through several turns and reaches an interior bend node B
#       - a new branch sprouts from B to a second leaf
#       - one frame touches the trunk, another touches the new branch
#     original_logic_at_key_step:
#       - solver finds the whole 8-connected non-bg component reachable from S
#       - because the source generator only made a single path, that traced component behaved
#         like "the line plus attached loops"
#     variant_logic_at_key_step:
#       - solver still traces the whole reachable non-bg component from S
#       - now that component is a tree rather than a chain, so cavities touching any branch are
#         also eligible and fill with S's color
#     difference:
#       - the solver must not rely on endpoint-following or on "exactly one continuation"
#       - generator must ensure genuine branching exists and matters to the filled cavities
#
#   complexity_calibration:
#     - branching count is kept small (1-3 total across the grid) so the rule stays learnable
#     - every branched seed has repeated evidence: at least one branch-attached loop, usually
#       plus a trunk loop, so the tree structure affects the output visibly
#     - colors stay local and structural: seed color fills cavities connected to that seed's
#       branched component; no hidden ordering by color id is used
#
#   main_assumption_audit:
#     - _seed_pixels:
#         assumption: seed colors are unique single border pixels
#         status: preserved by generator
#     - _seed_line_color:
#         assumption: a seed has at least one 4-neighbor non-bg pixel that belongs to its own
#         line network
#         status: preserved by placing the first line orthogonally adjacent to the seed
#     - _nonbg_component8_from:
#         assumption: all relevant branches and attached loops for one seed live in one
#         8-connected non-bg component
#         status: preserved by generator; branches use one line color and loops stay 8-adjacent
#     - cavity filtering by len(boundary_colors)==1:
#         assumption: each fillable cavity is enclosed by a single frame color
#         status: preserved exactly from the original task
#     - boundary must intersect traced component and be 8-adjacent to line_color:
#         assumption: loops are attached through the outer edge of the seed's own line network,
#         not merely near some other traced structure
#         status: preserved by the frame validator
#
#   placement_legality:
#     - new line pixels must be empty and may touch no occupied 8-neighbor except their parent
#     - branches may only start from bend nodes, which makes true 8-connected trees feasible
#     - frames require empty ring+cavity cells, no cavity pixel may touch the line, and at least
#       one ring pixel must be 8-adjacent to the intended line color
#     - seeds use different corners, so seed components do not merge
#
#   observability_and_cleanliness:
#     - branches are explicit geometric forks, not latent graph metadata
#     - loops are attached to multiple different limbs, so the output shows several separated
#       fill regions activated by one seed
#     - the generator rejects tiny/failed trees and keeps object count moderate to avoid clutter

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _seed_line_color(g: np.ndarray, seed: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    neigh_cols = []
    sr, sc = seed
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return int(Counter(neigh_cols).most_common(1)[0][0])


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _collect_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        cavities.append((region, boundary, {int(g[p]) for p in boundary}))
    return cavities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: a unique border seed labels every single-color cavity whose frame touches the
    seed's entire reachable 8-connected drawing component. In this variant that component may be a
    branching tree rather than a single chain, so loops attached to any branch are fillable.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    cavities = _collect_cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        sr, sc = seed
        seed_col = int(g[sr, sc])
        line_col = _seed_line_color(g, seed, bg)

        # Key branch-supporting choice: trace the whole non-bg component from the seed instead of
        # assuming a single endpoint-followable path. This naturally covers tree branches.
        traced = _nonbg_component8_from(g, seed, bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# ========================== CFG ==========================
GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_TRUNK_LEN = 7
TRUNK_LEN_RANGE = (9, 22)
MIN_BRANCH_LEN = 4
BRANCH_LEN_RANGE = (4, 10)
MAX_BRANCHES_PER_SEED = 2
BRANCH_ATTACH_TRIES = 40
TOTAL_BRANCHES_ONE_SEED = (1, 3)
TOTAL_BRANCHES_TWO_SEEDS = (1, 3)

FRAME_COLORS_MAX = 4
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 50
EXTRA_LOOPS_PER_SEED = (0, 2)
DISTRACTOR_SOLO_MAX = 2

DIAG_WEIGHT_ALIGNED = 4
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
TURN_BONUS = 2
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.35
PATH_STALL_LIMIT = 5

GEN_RETRIES = 400
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + ORTHO_DIRS


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1
            corner = dr in (0, H - 1) and dc in (0, W - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return True
    return False


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _try_attach_frame(
    canvas: np.ndarray,
    anchor_pixels: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    if not anchor_pixels:
        return None
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(anchor_pixels)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _bias_for(pt: tuple[int, int], h: int, w: int, rng: random.Random) -> tuple[int, int]:
    dr_bias = 1 if pt[0] < h // 2 else -1
    dc_bias = 1 if pt[1] < w // 2 else -1
    if rng.random() < 0.25:
        dr_bias *= -1
    if rng.random() < 0.25:
        dc_bias *= -1
    return dr_bias, dc_bias


def _weighted_dirs(
    dr_bias: int,
    dc_bias: int,
    prev_dir: tuple[int, int] | None,
    rng: random.Random,
) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    rng.shuffle(dirs)
    return dirs


def _can_place_line_pixel(canvas: np.ndarray, parent: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = canvas.shape
    r, c = cell
    if not _in_bounds(h, w, r, c) or int(canvas[r, c]) != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc) or (nr, nc) == parent:
            continue
        if int(canvas[nr, nc]) != -1:
            return False
    return True


def _add_line_edge(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    parent: tuple[int, int],
    child: tuple[int, int],
    line_col: int,
) -> None:
    canvas[child] = line_col
    tree_set.add(child)
    degrees.setdefault(parent, 0)
    degrees.setdefault(child, 0)
    degrees[parent] += 1
    degrees[child] += 1


def _extend_tip_path_inplace(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    tip: tuple[int, int],
    target_extra: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cur = tip
    prev_dir = None
    dr_bias, dc_bias = _bias_for(tip, h, w, rng)
    added: list[tuple[int, int]] = []
    stalls = 0
    while len(added) < target_extra and stalls < PATH_STALL_LIMIT:
        dirs = _weighted_dirs(dr_bias, dc_bias, prev_dir, rng)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            if not _can_place_line_pixel(canvas, cur, (nr, nc)):
                continue
            _add_line_edge(canvas, tree_set, degrees, cur, (nr, nc), line_col)
            cur = (nr, nc)
            added.append(cur)
            prev_dir = (ddr, ddc)
            placed = True
            break
        if placed:
            stalls = 0
            if rng.random() < 0.15:
                dr_bias, dc_bias = _bias_for(cur, h, w, rng)
        else:
            stalls += 1
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
    return added


def _line_neighbors_of(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in tree_set]


def _is_bend_node(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    neighs = _line_neighbors_of(tree_set, pt)
    if len(neighs) != 2:
        return False
    (r1, c1), (r2, c2) = neighs
    v1 = (r1 - pt[0], c1 - pt[1])
    v2 = (r2 - pt[0], c2 - pt[1])
    return not (v1[0] == -v2[0] and v1[1] == -v2[1])


def _spawn_branch_trial(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    base: tuple[int, int],
    target_len: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    trial_canvas = canvas.copy()
    trial_tree = set(tree_set)
    trial_degrees = dict(degrees)
    h, w = canvas.shape
    dr_bias, dc_bias = _bias_for(base, h, w, rng)
    for ddr, ddc in _weighted_dirs(dr_bias, dc_bias, None, rng):
        first = (base[0] + ddr, base[1] + ddc)
        if not _can_place_line_pixel(trial_canvas, base, first):
            continue
        _add_line_edge(trial_canvas, trial_tree, trial_degrees, base, first, line_col)
        path = [first]
        if target_len > 1:
            path.extend(
                _extend_tip_path_inplace(
                    trial_canvas,
                    trial_tree,
                    trial_degrees,
                    first,
                    target_len - 1,
                    line_col,
                    rng,
                )
            )
        if len(path) < MIN_BRANCH_LEN:
            continue
        canvas[:, :] = trial_canvas
        tree_set.clear()
        tree_set.update(trial_tree)
        degrees.clear()
        degrees.update(trial_degrees)
        return path
    return []


def _grow_branching_tree(
    canvas: np.ndarray,
    seed_rc: tuple[int, int],
    first_rc: tuple[int, int],
    line_col: int,
    branch_count: int,
    rng: random.Random,
) -> dict | None:
    h, w = canvas.shape
    tree_set = {first_rc}
    degrees = {first_rc: 0}

    trunk_hi = min(TRUNK_LEN_RANGE[1], max(TRUNK_LEN_RANGE[0], h + w - 6))
    trunk_target = rng.randint(TRUNK_LEN_RANGE[0], trunk_hi)
    trunk_path = [first_rc]
    trunk_path.extend(_extend_tip_path_inplace(canvas, tree_set, degrees, first_rc, trunk_target - 1, line_col, rng))
    if len(trunk_path) < MIN_TRUNK_LEN:
        return None

    segments = [trunk_path]
    for _ in range(branch_count):
        branch_hi = min(BRANCH_LEN_RANGE[1], max(BRANCH_LEN_RANGE[0], (h + w) // 2))
        branch_target = rng.randint(BRANCH_LEN_RANGE[0], branch_hi)
        placed = []
        bend_bases = [pt for pt in trunk_path if pt != first_rc and _is_bend_node(tree_set, pt)]
        bend_bases.extend(pt for seg in segments[1:] for pt in seg if _is_bend_node(tree_set, pt))
        rng.shuffle(bend_bases)
        for base in bend_bases[:BRANCH_ATTACH_TRIES]:
            placed = _spawn_branch_trial(canvas, tree_set, degrees, base, branch_target, line_col, rng)
            if placed:
                break
        if not placed:
            return None
        segments.append(placed)

    if len(tree_set) < MIN_TRUNK_LEN + branch_count * MIN_BRANCH_LEN:
        return None

    branch_points = [pt for pt, deg in degrees.items() if deg >= 3]
    if branch_count > 0 and not branch_points:
        return None

    return {
        "seed": seed_rc,
        "pixels": list(tree_set),
        "tree_set": tree_set,
        "degrees": degrees,
        "segments": segments,
        "branch_count": len(segments) - 1,
    }


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_branch_budget(num_seeds: int, rng: random.Random) -> list[int]:
    total = rng.randint(*(TOTAL_BRANCHES_ONE_SEED if num_seeds == 1 else TOTAL_BRANCHES_TWO_SEEDS))
    out = [0] * num_seeds
    for _ in range(total):
        choices = [i for i in range(num_seeds) if out[i] < MAX_BRANCHES_PER_SEED]
        if not choices:
            break
        out[rng.choice(choices)] += 1
    if sum(out) == 0:
        out[rng.randrange(num_seeds)] = 1
    return out


def _attach_seed_loops(
    canvas: np.ndarray,
    tree_info: dict,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
    segments = tree_info["segments"]
    trunk = segments[0]
    branches = segments[1:]

    required_groups = branches + [trunk]
    for group in required_groups:
        fc = rng.choice(frame_colors_pool)
        if _try_attach_frame(canvas, group, line_col, fc, rng) is None:
            return False

    extras = rng.randint(*EXTRA_LOOPS_PER_SEED)
    all_groups = segments[:]
    for _ in range(extras):
        fc = rng.choice(frame_colors_pool)
        _try_attach_frame(canvas, rng.choice(all_groups), line_col, fc, rng)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    per_seed_branches = _distribute_branch_budget(num_seeds, rng)
    lines_info = []
    total_actual_branches = 0

    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        tree_info = _grow_branching_tree(canvas, seed_rc, first_rc, line_colors[i], per_seed_branches[i], rng)
        if tree_info is None:
            return None
        total_actual_branches += int(tree_info["branch_count"])
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], tree_info))

    if total_actual_branches == 0:
        return None

    for _, _, line_col, tree_info in lines_info:
        if not _attach_seed_loops(canvas, tree_info, line_col, frame_colors_pool, rng):
            return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    # Variant guard: at least one seeded line graph must genuinely branch in 8-neighbor line-color space.
    branchy_seed_count = 0
    for _seed_col, seed_rc, line_col, _tree_info in lines_info:
        traced = _nonbg_component8_from(grid, seed_rc, bg)
        line_pts = {pt for pt in traced if int(grid[pt]) == line_col}
        degrees = {pt: sum((nr, nc) in line_pts for nr, nc in neighbors8(*pt)) for pt in line_pts}
        if any(deg >= 3 for deg in degrees.values()):
            branchy_seed_count += 1
    if branchy_seed_count == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")

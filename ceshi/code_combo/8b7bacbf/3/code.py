# Combo variant of 8b7bacbf (index 3)
# combo_spec: D2/1+D12/0
# Name: 分支路径+单种子多线路
#
# Design analysis
# ---------------
# fused_variant:
#   summary:
#     - keep the original "single-color frame loops enclosing bg cavities" objects
#     - change the seed-side wiring in two coupled ways:
#       1. exactly one border seed now launches multiple active lines
#       2. each active line is no longer a single chain; it is a thin 8-connected tree
#     - every cavity attached to one tree fills with that tree's line color, never with seed color
#
#   concrete_behavior_trace:
#     - miniature input:
#       - one seed S sits on the top edge at column c
#       - S 4-touches a left start pixel in color A and a right start pixel in color B
#       - the A line bends inward, branches once, and ends in two leaves; each leaf touches a frame
#       - the B line does the same in a different region
#     - original task execution:
#       - a seed identified one line color
#       - the solver traced one reachable non-bg component and filled touched cavities with seed color
#     - fused execution:
#       - the solver first separates the seed's two neighboring line colors
#       - for each color, it traces the full same-color 8-connected tree, which includes all branches
#       - a cavity is filled only if its frame boundary is adjacent to exactly one traced tree
#       - the fill color is that tree's line color
#     - visible difference:
#       - one seed now causes multiple fill colors
#       - one line color now fills multiple separated cavities because its line has true branches
#
#   complexity_calibration:
#     - active line count is fixed at 2; this is enough to show 1:many mapping without clutter
#     - each active line must branch at least once, so every sample visibly differs from the
#       original single-chain task
#     - each leaf segment hosts a loop, giving repeated evidence that one line color controls
#       multiple distant cavities
#     - frame colors may repeat across different line trees, preventing a shortcut such as
#       "ring color determines fill color"
#
#   old_main_assumption_audit:
#     - unique border seed pixels:
#         preserved, but now exactly one seed is expected
#     - "most common non-bg seed neighbor is the line color":
#         broken, because the seed intentionally touches two different active line colors
#     - "the entire traced non-bg component belongs to one fill label":
#         broken, because the single seed now fans out into multiple colored trees
#     - "fill with seed color":
#         broken, because output color is determined by which line tree reaches the loop
#     - "a simple path is enough":
#         broken, because each line tree can split and must be traced as a full branching component
#
#   placement_legality:
#     - the seed is placed on a non-corner border cell with exactly two orthogonal start pixels
#     - new line pixels must be empty and may touch no occupied 8-neighbor except their parent
#       and, near the root only, the shared seed
#     - branches can sprout only from bend nodes; this avoids accidental 8-connected cycles
#     - every attached frame uses empty ring/cavity cells, avoids cavity-line contact, and is
#       forbidden from touching foreign non-bg pixels
#     - leaf loops are anchored near segment tails so the "branch end -> frame ring" relation is visible
#
#   observability_and_cleanliness:
#     - the shared seed is a single obvious source, but the two active line colors immediately split
#       the scene into two explainable subsystems
#     - each subsystem has a small number of loops near leaf ends, so branch structure is readable
#     - distractors stay sparse and disconnected, adding counterexamples without obscuring the rule
#
#   new_solver_assumption_audit:
#     - `_seed_line_components(...)` assumes each active tree has a unique line color among the
#       seed's 4-neighbors; generator guarantees two distinct line colors
#     - `_boundary_touches_line(...)` assumes a fillable cavity touches exactly one active tree;
#       the frame validator enforces that rings cannot be adjacent to foreign lines
#     - the same-color flood from each seed neighbor assumes one line color forms one connected tree;
#       generator grows each active tree from one start pixel only
#     - branch detection in generator assumes degree>=3 is visible in 8-neighbor line space;
#       post-checks verify that property before accepting a sample

from __future__ import annotations

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


def _touches_border(pts: set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


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


def _same_color_component8_from(g: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    vis = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                stack.append((nr, nc))
    return vis


def _seed_line_components(g: np.ndarray, seed_rc: tuple[int, int], bg: int) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = g.shape
    by_color: dict[int, tuple[int, int]] = {}
    for nr, nc in neighbors4(*seed_rc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                by_color.setdefault(v, (nr, nc))
    return [(line_col, _same_color_component8_from(g, start, line_col)) for line_col, start in by_color.items()]


def _collect_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    cavities = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if _touches_border(region, h, w):
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


def _boundary_touches_line(boundary: set[tuple[int, int]], line_component: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in line_component:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    One border seed fans out into multiple branching line trees, and every cavity touching one
    tree is flood-filled by that tree's color.

    Input structure:
    - Background is the most frequent color.
    - Exactly one unique-color seed pixel lies on a non-corner border cell.
    - The seed 4-touches multiple different line colors; each line color forms its own 8-connected
      1-pixel-wide branching tree.
    - Closed single-color frame rings sit near branch leaves and surround background cavities.

    Rule:
    - Detect the single seed and split its outgoing structures by line color.
    - For each line color, trace the full same-color 8-connected tree reachable from the seed.
    - Any enclosed background cavity whose frame boundary is adjacent to exactly one traced tree is
      filled with that tree's line color.

    Key insight:
    - The seed is only a common origin marker. It does not supply the fill color.
    - The essential reasoning is to keep the different outgoing trees separate while still tracing
      each tree completely through all of its branches.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    seeds = _seed_pixels(g, bg)
    if len(seeds) != 1:
        return out
    seed_rc = seeds[0]

    line_infos = _seed_line_components(g, seed_rc, bg)
    if len(line_infos) < 2:
        return out

    for cav, boundary, boundary_colors in _collect_cavities(g, bg):
        if len(boundary_colors) != 1:
            continue
        touching_lines = [line_col for line_col, line_component in line_infos if _boundary_touches_line(boundary, line_component, h, w)]
        if len(touching_lines) != 1:
            continue
        fill_col = touching_lines[0]
        for r, c in cav:
            if int(out[r, c]) == bg:
                out[r, c] = fill_col
    return out


# ========================== CFG ==========================
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

ACTIVE_LINE_COUNT = 2
MIN_TRUNK_LEN = 7
TRUNK_LEN_RANGE = (9, 18)
MIN_BRANCH_LEN = 4
BRANCH_LEN_RANGE = (4, 9)
BRANCHES_PER_LINE = (1, 2)
BRANCH_ATTACH_TRIES = 50
PATH_STALL_LIMIT = 6

FRAME_COLORS_MAX = 3
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 40
EXTRA_LOOPS_PER_LINE = (0, 1)

DIAG_WEIGHT_ALIGNED = 4
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
TURN_BONUS = 2
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.35

DISTRACTOR_NETWORK_PROB = 0.35
DISTRACTOR_NETWORK_TRIES = 35
DISTRACTOR_LINE_LEN = (4, 10)
DISTRACTOR_SOLO_MAX = 1

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            on_border = dr == 0 or dc == 0 or dr == fh - 1 or dc == fw - 1
            corner = dr in (0, fh - 1) and dc in (0, fw - 1)
            if on_border and not corner:
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
        if int(canvas[rr, cc]) != -1:
            return False
    for rr, cc in cavity:
        if int(canvas[rr, cc]) != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
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
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    anchor_pixels: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
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
            if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD or r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
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
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
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


def _bias_for(pt: tuple[int, int], h: int, w: int, rng: random.Random) -> tuple[int, int]:
    dr_bias = 1 if pt[0] < h // 2 else -1
    dc_bias = 1 if pt[1] < w // 2 else -1
    if rng.random() < 0.25:
        dr_bias *= -1
    if rng.random() < 0.25:
        dc_bias *= -1
    return dr_bias, dc_bias


def _can_place_line_pixel(
    canvas: np.ndarray,
    parent: tuple[int, int],
    cell: tuple[int, int],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    h, w = canvas.shape
    r, c = cell
    if not _in_bounds(h, w, r, c) or int(canvas[r, c]) != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc) or (nr, nc) == parent or (nr, nc) in allowed_neighbors:
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
    allowed_neighbors: set[tuple[int, int]],
    initial_bias: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cur = tip
    prev_dir = None
    if initial_bias is None:
        dr_bias, dc_bias = _bias_for(tip, h, w, rng)
    else:
        dr_bias, dc_bias = initial_bias
    added: list[tuple[int, int]] = []
    stalls = 0
    while len(added) < target_extra and stalls < PATH_STALL_LIMIT:
        dirs = _weighted_dirs(dr_bias, dc_bias, prev_dir, rng)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            if not _can_place_line_pixel(canvas, cur, (nr, nc), allowed_neighbors):
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
    return [(nr, nc) for nr, nc in neighbors8(*pt) if (nr, nc) in tree_set]


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
    allowed_neighbors: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    trial_canvas = canvas.copy()
    trial_tree = set(tree_set)
    trial_degrees = dict(degrees)
    h, w = canvas.shape
    dr_bias, dc_bias = _bias_for(base, h, w, rng)
    for ddr, ddc in _weighted_dirs(dr_bias, dc_bias, None, rng):
        first = (base[0] + ddr, base[1] + ddc)
        if not _can_place_line_pixel(trial_canvas, base, first, allowed_neighbors):
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
                    allowed_neighbors,
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
    initial_bias: tuple[int, int],
    rng: random.Random,
) -> dict | None:
    h, w = canvas.shape
    allowed_neighbors = {seed_rc}
    tree_set = {first_rc}
    degrees = {first_rc: 0}

    trunk_hi = min(TRUNK_LEN_RANGE[1], max(TRUNK_LEN_RANGE[0], h + w - 8))
    trunk_target = rng.randint(TRUNK_LEN_RANGE[0], trunk_hi)
    trunk_path = [first_rc]
    trunk_path.extend(
        _extend_tip_path_inplace(
            canvas,
            tree_set,
            degrees,
            first_rc,
            trunk_target - 1,
            line_col,
            rng,
            allowed_neighbors,
            initial_bias=initial_bias,
        )
    )
    if len(trunk_path) < MIN_TRUNK_LEN:
        return None

    segments = [trunk_path]
    for _ in range(branch_count):
        branch_hi = min(BRANCH_LEN_RANGE[1], max(BRANCH_LEN_RANGE[0], (h + w) // 2))
        branch_target = rng.randint(BRANCH_LEN_RANGE[0], branch_hi)
        bend_bases = [pt for seg in segments for pt in seg if pt != first_rc and _is_bend_node(tree_set, pt) and degrees.get(pt, 0) == 2]
        rng.shuffle(bend_bases)
        placed = []
        for base in bend_bases[:BRANCH_ATTACH_TRIES]:
            placed = _spawn_branch_trial(canvas, tree_set, degrees, base, branch_target, line_col, rng, allowed_neighbors)
            if placed:
                break
        if not placed:
            return None
        segments.append(placed)

    branch_points = [pt for pt, deg in degrees.items() if deg >= 3]
    if branch_count > 0 and not branch_points:
        return None

    return {
        "tree_set": tree_set,
        "degrees": degrees,
        "segments": segments,
        "branch_count": len(segments) - 1,
    }


def _tail_anchor_pixels(segment: list[tuple[int, int]], tail_len: int = 4) -> list[tuple[int, int]]:
    if not segment:
        return []
    tail = list(reversed(segment[-min(tail_len, len(segment)) :]))
    seen = set()
    out = []
    for pt in tail:
        if pt not in seen:
            seen.add(pt)
            out.append(pt)
    return out


def _attach_leaf_loops(
    canvas: np.ndarray,
    tree_info: dict,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
    for segment in tree_info["segments"]:
        anchors = _tail_anchor_pixels(segment)
        if _try_attach_frame(canvas, anchors, line_col, rng.choice(frame_colors_pool), rng) is None:
            return False
    for _ in range(rng.randint(*EXTRA_LOOPS_PER_LINE)):
        segment = rng.choice(tree_info["segments"])
        anchors = segment[:]
        rng.shuffle(anchors)
        _try_attach_frame(canvas, anchors, line_col, rng.choice(frame_colors_pool), rng)
    return True


def _grow_simple_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    initial_bias: tuple[int, int],
) -> list[tuple[int, int]]:
    tree_set = {start}
    degrees = {start: 0}
    path = [start]
    path.extend(
        _extend_tip_path_inplace(
            canvas,
            tree_set,
            degrees,
            start,
            target_len - 1,
            line_col,
            rng,
            allowed_neighbors=set(),
            initial_bias=initial_bias,
        )
    )
    return path


def _try_place_distractor_network(canvas: np.ndarray, line_col: int, frame_colors_pool: list[int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(int(canvas[r + dr, c + dc]) != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_simple_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng, (rng.choice([-1, 1]), rng.choice([-1, 1])))
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        if _try_attach_frame(canvas, path, line_col, rng.choice(frame_colors_pool), rng) is not None:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _edge_seed_config(
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    options = []
    if w >= 10:
        c = rng.randint(3, w - 4)
        options.append(((0, c), [(0, c - 1), (0, c + 1)], [(1, -1), (1, 1)]))
        options.append(((h - 1, c), [(h - 1, c - 1), (h - 1, c + 1)], [(-1, -1), (-1, 1)]))
    if h >= 10:
        r = rng.randint(3, h - 4)
        options.append(((r, 0), [(r - 1, 0), (r + 1, 0)], [(-1, 1), (1, 1)]))
        options.append(((r, w - 1), [(r - 1, w - 1), (r + 1, w - 1)], [(-1, -1), (1, -1)]))
    return rng.choice(options)


def _component_degrees(component: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {pt: sum((nr, nc) in component for nr, nc in neighbors8(*pt)) for pt in component}


def _count_touching_cavities(
    g: np.ndarray,
    bg: int,
    line_component: set[tuple[int, int]],
) -> int:
    h, w = g.shape
    total = 0
    for _cav, boundary, boundary_colors in _collect_cavities(g, bg):
        if len(boundary_colors) != 1:
            continue
        if _boundary_touches_line(boundary, line_component, h, w):
            total += 1
    return total


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    if len(palette) < 6:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_col = palette.pop()
    line_colors = [palette.pop() for _ in range(ACTIVE_LINE_COUNT)]
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette) - 1))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    distractor_line_colors = palette[:]

    seed_rc, starts, biases = _edge_seed_config(h, w, rng)
    if int(canvas[seed_rc]) != -1 or any(int(canvas[p]) != -1 for p in starts):
        return None
    canvas[seed_rc] = seed_col

    lines_info = []
    for i, start in enumerate(starts):
        line_col = line_colors[i]
        if int(canvas[start]) != -1:
            return None
        canvas[start] = line_col
        branch_count = rng.randint(*BRANCHES_PER_LINE)
        tree_info = _grow_branching_tree(canvas, seed_rc, start, line_col, branch_count, biases[i], rng)
        if tree_info is None or tree_info["branch_count"] < 1:
            return None
        lines_info.append((line_col, tree_info))

    for line_col, tree_info in lines_info:
        if not _attach_leaf_loops(canvas, tree_info, line_col, frame_colors_pool, rng):
            return None

    if frame_colors_pool and distractor_line_colors and rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(distractor_line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    if int((grid == seed_col).sum()) != 1:
        return None

    seeds = _seed_pixels(grid, bg)
    if len(seeds) != 1 or seeds[0] != seed_rc:
        return None
    line_infos = _seed_line_components(grid, seed_rc, bg)
    if len(line_infos) != ACTIVE_LINE_COUNT:
        return None

    for line_col, line_component in line_infos:
        degrees = _component_degrees(line_component)
        if not any(deg >= 3 for deg in degrees.values()):
            return None
        if _count_touching_cavities(grid, bg, line_component) < 2:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    for line_col, _line_component in line_infos:
        if int((out == line_col).sum()) <= int((grid == line_col).sum()):
            return None
    if len({int(v) for v in np.unique(out[grid == bg]) if v != bg}) < ACTIVE_LINE_COUNT:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")

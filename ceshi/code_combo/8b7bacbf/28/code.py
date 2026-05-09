# Combo variant of 8b7bacbf (index 28)
# combo_spec: D2/1+D6/1+D9/3
# Name: 分支路径+开口框架+填充后擦除路径
#
# Design analysis
# ===============
# concretization_and_trace:
#   - Keep the original object vocabulary: most-common background, 1-2 unique border seeds,
#     one line color per seed, and rounded-rectangle frame rings with rectangular cavities.
#   - D2/1 changes the line from a single chain into a small 8-connected tree. A seed may have
#     a trunk plus 1-2 short branches, and different cavities can hang off different limbs.
#   - D9/3 is adapted to the combo setting as "true openings that are temporarily closed by the
#     path": the visible frame color component is missing 1-2 contiguous cells on one side, and
#     those missing side cells are occupied by the seed's line pixels. The cavity is therefore
#     closed in the input only because the path acts as part of the boundary.
#   - D6/1 contributes the erase-style transform, but the combo-level rule overrides what gets
#     erased: after filling all reachable path-completed cavities, erase all non-frame path pixels.
#     The final output keeps the frame rings, including their openings, and keeps the filled
#     cavity interiors; seeds and paths disappear.
#   - Small example:
#       1. A seed S on the border reaches an interior trunk pixel, branches at a bend, and one
#          branch passes through a 1px opening on the top side of a frame.
#       2. The visible frame color is broken at that opening, but the line pixel sitting there
#          completes the boundary, so the inner rectangle should still fill with S's color.
#       3. After the fill, erase the whole path tree. The output now shows an open frame with a
#          solid seed-colored interior, plus any untouched distractor frames.
#
# complexity_calibration:
#   - Branch count stays small: 1-3 total across the grid, at most 2 per seed. This is enough to
#     make "follow all branches" visible without turning the puzzle into clutter.
#   - Attached frames are always open-and-completed by the path, not a mix of many unrelated gap
#     semantics. Distractors may be closed or open, but they are not path-completed by a seed.
#   - Each branched seed gets at least one branch-attached fillable frame and one trunk-attached
#     fillable frame when possible, so the branch structure has repeated visual consequences.
#
# main_assumption_audit:
#   - Base-task bg-component cavity detection no longer works:
#       open frames would merge their cavity with the outside if we only looked at background
#       connectivity. The solver therefore switches to explicit frame-shape inference.
#   - `_seed_pixels` still assumes each seed color appears exactly once on the border.
#       The generator keeps that invariant.
#   - `_seed_line_color` still assumes an orthogonally adjacent first line pixel exists.
#       The generator preserves this by construction.
#   - `_nonbg_component8_from` now traces a whole tree plus attached frames rather than a chain.
#       This is still valid because each seed network remains one 8-connected component.
#   - New fill condition:
#       a frame is fillable only if its visible frame component matches the rounded-rectangle
#       template with a 1-2px side opening, the missing side cells are occupied by the seed's own
#       line color, and the visible frame lies inside the seed's traced component.
#   - New erase condition:
#       output reconstruction assumes every non-background non-frame pixel is part of a seed path
#       or distractor path and should disappear. The generator therefore only places seeds, paths,
#       and recognizable frames.
#
# placement_legality:
#   - Branch growth uses 8-connectivity but forbids accidental side-touching, so the line network
#     stays a true thin tree instead of collapsing into blobs or cycles.
#   - A completed open frame may only reuse the intended 1-2 gap cells from the line. Every other
#     cell inside the frame bbox must stay empty before placement, and no extra line pixel may hug
#     the cavity except through the designated gap cells.
#   - Frames keep a 1-cell border margin and may not overlap or 8-touch foreign structures, which
#     keeps bbox-based frame inference unambiguous.
#
# observability_cleanliness_and_inference:
#   - Humans can see three coupled signals directly from a few examples:
#       branching paths matter because multiple separated cavities of one seed fill;
#       visibly broken frames can still fill because a path segment sits in the opening;
#       the same path then disappears in the output, leaving the frame still open.
#   - Color roles remain structural and local: seed color becomes fill color, line color is only a
#     temporary connector, frame colors mark boundaries. No rule depends on numeric color ids.

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


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


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
    sr, sc = seed
    neigh_cols = []
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


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = (dr in (0, h - 1)) and (dc in (0, w - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _frame_sides(r0: int, c0: int, ch: int, cw: int) -> list[list[tuple[int, int]]]:
    return [
        [(r0, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + dr, c0) for dr in range(1, ch + 1)],
        [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    ]


def _is_single_short_opening(missing: set[tuple[int, int]], r0: int, c0: int, ch: int, cw: int) -> bool:
    if len(missing) == 0 or len(missing) > 2:
        return False
    for side in _frame_sides(r0, c0, ch, cw):
        idxs = [i for i, p in enumerate(side) if p in missing]
        if len(idxs) != len(missing):
            continue
        if idxs[-1] - idxs[0] + 1 == len(idxs):
            return True
    return False


def _frame_candidate_from_component(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    bg: int,
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts_set = set(pts)
    if len(pts_set) < 6:
        return None

    rows = [r for r, _ in pts_set]
    cols = [c for _, c in pts_set]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ch = r1 - r0 - 1
    cw = c1 - c0 - 1
    if ch < 2 or cw < 2:
        return None

    ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
    ring_set = set(ring)
    if not pts_set <= ring_set:
        return None

    missing = ring_set - pts_set
    if not _is_single_short_opening(missing, r0, c0, ch, cw):
        return None

    for r, c in cavity:
        if int(g[r, c]) != bg:
            return None

    frame_col = int(g[next(iter(pts_set))])
    return frame_col, set(cavity), pts_set, missing


def _all_frame_candidates(
    g: np.ndarray,
    bg: int,
) -> list[tuple[int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]]:
    candidates = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        for comp in _components_mask(g == int(col), conn8=True):
            found = _frame_candidate_from_component(g, comp, bg)
            if found is not None:
                candidates.append(found)
    return candidates


def _is_fillable_from_seed(
    g: np.ndarray,
    frame_pts: set[tuple[int, int]],
    missing: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> bool:
    if not frame_pts <= traced:
        return False
    if not missing:
        return False
    return all(pt in traced and int(g[pt]) == line_col for pt in missing)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cavities whose open frame gaps are completed by a seed's branching path, then erase paths.

    Input structure:
    - The background is the most common color.
    - One or two unique single-pixel seeds sit on the border, each with its own 8-connected line tree.
    - Single-color rounded-rectangle frames appear in the interior, but the fillable ones are visibly
      open: 1-2 contiguous side cells are missing from the frame color and are occupied by the line.

    Rule:
    - For each seed, trace its full reachable non-background component, including every branch.
    - Detect frame-color components that match a rounded rectangle with a short side opening.
    - If that opening is occupied by the traced line pixels of the seed, treat the frame as
      temporarily closed and flood-fill the inner cavity with the seed color.
    - After all such cavities are filled, erase every non-frame path pixel. The output keeps the
      frame rings in their open form and keeps the filled cavity contents.

    Key insight:
    - The cavity is not recognized from background closure anymore; it is inferred from the frame
      geometry plus the fact that the path itself completes the missing border cells. The path is a
      temporary scaffold: it enables filling, then disappears.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    frame_candidates = _all_frame_candidates(g, bg)
    for frame_col, _cavity, frame_pts, _missing in frame_candidates:
        for r, c in frame_pts:
            out[r, c] = int(frame_col)

    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        line_col = _seed_line_color(g, seed, bg)
        traced = _nonbg_component8_from(g, seed, bg)
        for _frame_col, cavity, frame_pts, missing in frame_candidates:
            if not _is_fillable_from_seed(g, frame_pts, missing, traced, line_col):
                continue
            for r, c in cavity:
                out[r, c] = seed_col

    return out


# ========================== CFG ==========================
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
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
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 90
DISTRACTOR_FRAME_TRIES = 60
ATTACHED_GAP_WEIGHTS = [(1, 4), (2, 2)]
DISTRACTOR_GAP_WEIGHTS = [(1, 3), (2, 2)]
EXTRA_LOOPS_PER_SEED = (0, 1)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_PROB = 0.35
DISTRACTOR_NETWORK_TRIES = 35
DISTRACTOR_LINE_LEN = (6, 12)

DIAG_WEIGHT_ALIGNED = 4
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
TURN_BONUS = 2
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.35
PATH_STALL_LIMIT = 5

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _weighted_choice(weighted_values: list[tuple[int, int]], rng: random.Random) -> int:
    bag = []
    for value, weight in weighted_values:
        bag.extend([int(value)] * int(weight))
    return int(rng.choice(bag))


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

    trunk_hi = min(TRUNK_LEN_RANGE[1], max(TRUNK_LEN_RANGE[0], h + w - 8))
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

    branch_points = [pt for pt, deg in degrees.items() if deg >= 3]
    if branch_count > 0 and not branch_points:
        return None

    return {
        "seed": seed_rc,
        "tree_set": tree_set,
        "degrees": degrees,
        "segments": segments,
        "branch_count": len(segments) - 1,
    }


def _segment_gap_options(segment: list[tuple[int, int]], gap_len: int) -> list[list[tuple[int, int]]]:
    if gap_len == 1:
        return [[pt] for pt in segment]
    options = []
    for a, b in zip(segment, segment[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue
        if a <= b:
            options.append([a, b])
        else:
            options.append([b, a])
    return options


def _bbox_from_gap(
    gap_cells: list[tuple[int, int]],
    side_kind: str,
    side_start: int,
    ch: int,
    cw: int,
) -> tuple[int, int]:
    if side_kind == "top":
        r0 = gap_cells[0][0]
        c0 = gap_cells[0][1] - side_start
        return r0, c0
    if side_kind == "bottom":
        r0 = gap_cells[0][0] - (ch + 1)
        c0 = gap_cells[0][1] - side_start
        return r0, c0
    if side_kind == "left":
        r0 = gap_cells[0][0] - side_start
        c0 = gap_cells[0][1]
        return r0, c0
    if side_kind == "right":
        r0 = gap_cells[0][0] - side_start
        c0 = gap_cells[0][1] - (cw + 1)
        return r0, c0
    raise ValueError(side_kind)


def _gap_side_placements(gap_cells: list[tuple[int, int]], ch: int, cw: int) -> list[tuple[str, int]]:
    gap_len = len(gap_cells)
    if gap_len == 1:
        placements = []
        for side_start in range(1, cw + 1):
            placements.append(("top", side_start))
            placements.append(("bottom", side_start))
        for side_start in range(1, ch + 1):
            placements.append(("left", side_start))
            placements.append(("right", side_start))
        return placements

    a, b = gap_cells
    if a[0] == b[0]:
        return [(side_kind, side_start) for side_kind in ("top", "bottom") for side_start in range(1, cw)]
    if a[1] == b[1]:
        return [(side_kind, side_start) for side_kind in ("left", "right") for side_start in range(1, ch)]
    return []


def _validate_completed_frame_placement(
    canvas: np.ndarray,
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    gap_cells: list[tuple[int, int]],
    line_col: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    fh, fw = ch + 2, cw + 2
    if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD or r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
        return None

    ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
    gap_set = set(gap_cells)
    ring_set = set(ring)
    if not gap_set <= ring_set:
        return None
    if any(int(canvas[r, c]) != line_col for r, c in gap_cells):
        return None

    bbox_cells = {(rr, cc) for rr in range(r0, r0 + fh) for cc in range(c0, c0 + fw)}
    for rr, cc in bbox_cells:
        if (rr, cc) in gap_set:
            continue
        if int(canvas[rr, cc]) != -1:
            return None

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if not _in_bounds(h, w, nr, nc):
                continue
            if int(canvas[nr, nc]) == line_col and (nr, nc) not in gap_set:
                return None

    kept_ring = [cell for cell in ring if cell not in gap_set]
    kept_ring_set = set(kept_ring)
    for rr, cc in kept_ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in kept_ring_set or (nr, nc) in gap_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return None

    outside_neighbors = 0
    for gr, gc in gap_cells:
        for nr, nc in neighbors8(gr, gc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in gap_set:
                continue
            if int(canvas[nr, nc]) == line_col and not (r0 <= nr < r0 + fh and c0 <= nc < c0 + fw):
                outside_neighbors += 1
    if outside_neighbors == 0:
        return None

    return kept_ring, cavity


def _try_attach_completed_frame(
    canvas: np.ndarray,
    anchor_segment: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    size_weights: list[int],
    gap_weights: list[tuple[int, int]],
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    if not anchor_segment:
        return None
    for _ in range(max_tries):
        ch = int(rng.choice(size_weights))
        cw = int(rng.choice(size_weights))
        gap_len = _weighted_choice(gap_weights, rng)
        gap_options = _segment_gap_options(anchor_segment, gap_len)
        rng.shuffle(gap_options)
        for gap_cells in gap_options:
            placements = _gap_side_placements(gap_cells, ch, cw)
            rng.shuffle(placements)
            for side_kind, side_start in placements:
                r0, c0 = _bbox_from_gap(gap_cells, side_kind, side_start, ch, cw)
                placed = _validate_completed_frame_placement(canvas, r0, c0, ch, cw, gap_cells, line_col)
                if placed is None:
                    continue
                kept_ring, cavity = placed
                for rr, cc in kept_ring:
                    canvas[rr, cc] = frame_col
                return kept_ring, cavity, gap_cells
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    *,
    size_weights: list[int],
    gap_weights: list[tuple[int, int]],
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = int(rng.choice(size_weights))
        cw = int(rng.choice(size_weights))
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return None
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        halo_ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
                    halo_ok = False
                    break
            if not halo_ok:
                break
        if not halo_ok:
            continue

        ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
        gap_len = _weighted_choice(gap_weights, rng)
        side = rng.choice(_frame_sides(r0, c0, ch, cw))
        if len(side) < gap_len:
            continue
        start = rng.randint(0, len(side) - gap_len)
        gap_cells = side[start : start + gap_len]
        kept_ring = [cell for cell in ring if cell not in set(gap_cells)]
        for rr, cc in kept_ring:
            canvas[rr, cc] = frame_col
        return kept_ring, cavity, gap_cells
    return None


def _grow_simple_path(
    canvas: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    path = [start]
    path_set = {start}
    cur = start
    dr_bias, dc_bias = _bias_for(start, h, w, rng)
    prev_dir = None
    stalls = 0
    while len(path) < target_len and stalls < PATH_STALL_LIMIT:
        placed = False
        for ddr, ddc in _weighted_dirs(dr_bias, dc_bias, prev_dir, rng):
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            if not _in_bounds(h, w, nr, nc) or int(canvas[nr, nc]) != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            canvas[cur] = line_col
            prev_dir = (ddr, ddc)
            placed = True
            break
        if placed:
            stalls = 0
        else:
            stalls += 1
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
    return path


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(int(canvas[r + dr, c + dc]) != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_simple_path(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng)
        if len(path) < 4:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        fc = rng.choice(frame_colors_pool)
        if _try_attach_completed_frame(
            canvas,
            path,
            line_col,
            fc,
            rng,
            size_weights=DISTRACTOR_CAVITY_WEIGHTS,
            gap_weights=ATTACHED_GAP_WEIGHTS,
        ) is not None:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


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


def _attach_seed_frames(
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
        if _try_attach_completed_frame(
            canvas,
            group,
            line_col,
            fc,
            rng,
            size_weights=CAVITY_SIZE_WEIGHTS,
            gap_weights=ATTACHED_GAP_WEIGHTS,
        ) is None:
            return False

    extras = rng.randint(*EXTRA_LOOPS_PER_SEED)
    for _ in range(extras):
        fc = rng.choice(frame_colors_pool)
        _try_attach_completed_frame(
            canvas,
            rng.choice(segments),
            line_col,
            fc,
            rng,
            size_weights=CAVITY_SIZE_WEIGHTS,
            gap_weights=ATTACHED_GAP_WEIGHTS,
        )
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
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
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

    for _seed_col, _seed_rc, line_col, tree_info in lines_info:
        if not _attach_seed_frames(canvas, tree_info, line_col, frame_colors_pool, rng):
            return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(
            canvas,
            rng.choice(frame_colors_pool),
            rng,
            size_weights=DISTRACTOR_CAVITY_WEIGHTS,
            gap_weights=DISTRACTOR_GAP_WEIGHTS,
        )

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        distractor_line_col = rng.choice(line_colors)
        _try_place_distractor_network(canvas, distractor_line_col, frame_colors_pool, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _seed_rc, _line_col, _tree_info in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    # Keep each seed network visually separable: a non-background component may
    # contain at most one border seed, otherwise two branch systems collapse into
    # one ambiguous object for human solvers.
    seeds = set(_seed_pixels(grid, bg))
    for comp in _components_mask(grid != bg, conn8=True):
        if sum((r, c) in seeds for r, c in comp) > 1:
            return None

    frame_candidates = _all_frame_candidates(grid, bg)
    if not frame_candidates:
        return None

    fillable_total = 0
    open_total = 0
    for _seed_col, seed_rc, line_col, _tree_info in lines_info:
        traced = _nonbg_component8_from(grid, seed_rc, bg)
        recognized_for_seed = 0
        branchy = False
        line_pts = {pt for pt in traced if int(grid[pt]) == line_col}
        for pt in line_pts:
            deg = sum((nr, nc) in line_pts for nr, nc in neighbors8(*pt))
            if deg >= 3:
                branchy = True
                break
        if not branchy:
            return None

        for _frame_col, cavity, frame_pts, missing in frame_candidates:
            if not _is_fillable_from_seed(grid, frame_pts, missing, traced, line_col):
                continue
            recognized_for_seed += 1
            fillable_total += 1
            if len(missing) > 0:
                open_total += 1
            if any(int(grid[r, c]) != bg for r, c in cavity):
                return None
        if recognized_for_seed == 0:
            return None

    if open_total == 0 or fillable_total == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    filled_cells = int(((grid == bg) & (out != bg)).sum())
    erased_cells = int(((grid != bg) & (out == bg)).sum())
    surviving_frame_cells = 0
    all_frame_pts = set()
    for _frame_col, _cavity, frame_pts, _missing in frame_candidates:
        all_frame_pts.update(frame_pts)
    for pt in all_frame_pts:
        if int(out[pt]) == int(grid[pt]):
            surviving_frame_cells += 1
    if filled_cells <= 0 or erased_cells <= 0 or surviving_frame_cells <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")

# Combo variant of 800d221b (index 15)
# combo_spec: D2/2+D7/1
# Name: hub整体着色+十字形hub

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import components_mask, get_bg, label_mask

# Design analysis for combo D2/2 + D7/1
# --------------------------------------
# a) Concrete behavior and trace
# - Original 800d221b has a 3x3 solid hub. Branch components and seed-border paths are
#   recolored to the majority interior color of the seed they touch. Only the single
#   center hub pixel changes color at the end.
# - This combo changes two linked stages:
#   1. the hub shape becomes a 5-cell cross:
#          .P.
#          PPP
#          .P.
#   2. after branch recoloring, the winning hub-adjacent branch color paints all five
#      hub cells, not just the center.
# - Small trace:
#   top seed majority = A, left seed majority = A, right seed majority = B.
#   Original task would recolor three branches/borders, then only the center hub pixel
#   becomes A while the other eight hub-square pixels stay P.
#   Here the visible core is different twice over:
#   input hub is already a cross instead of a square, and output paints the entire
#   cross A, so four extra hub pixels change compared with a center-only rule.
#
# b) Complexity calibration
# - A cross hub only has four clean non-corner exits that keep every non-hub path cell
#   at degree <= 2. Therefore the generator uses 3-4 seeds, each with exactly one path.
# - Seed interiors still use only two colors (A/B), and branch winners are forced to a
#   2-1 split for 3 seeds or a 3-1 split for 4 seeds. That keeps the hub vote obvious
#   and avoids hidden tie-breaking.
# - The rest of the scene stays in the same family as the base task: edge-attached seed
#   rectangles, a 1-pixel P border, and thin orthogonal paths with bends.
#
# c) Hidden-assumption audit of main()
# - The original "largest solid k x k P square" hub detector is invalid because the hub
#   is no longer a solid square. The solver must instead look for the unique P pixel of
#   4-neighborhood degree 4; its four orthogonal neighbors are the hub arms.
# - The original branch mask `(g == P) & (~hub)` still works once `hub` is the 5-cell
#   cross, because each seed border plus its path remains a separate 4-connected
#   component after hub removal.
# - The path-degree assertion still assumes no non-hub T-junctions. Cross-corner exits
#   would violate that by creating degree-3 cells adjacent to two hub arms, so the
#   generator must never use those corner-adjacent starts.
# - The original center-only recolor is invalid. The hub vote must now recolor every
#   cell in the hub mask.
#
# d) Placement legality
# - The hub center is placed with a radius-2 moat so the four legal exit cells
#   `(center +/- 2, center)` and `(center, center +/- 2)` are in-bounds.
# - Seed border zones are rejected if they overlap the 5x5 guard box around the hub.
# - Path carving starts from one of the four legal exit cells, never from the four
#   cross corners, so every non-hub path cell keeps degree <= 2.
#
# e) Visual observability
# - Across 2-4 examples, a human can see that the core object is not a 3x3 block but a
#   plus sign. In the outputs, that whole plus sign turns into the branch-majority
#   color rather than leaving four arm pixels unchanged as P.
# - No branch decision depends on numeric color id. Majority is computed from visible
#   seed interiors and the hub winner comes from visible counts of adjacent recolored
#   branches.
#
# f) Visual cleanliness
# - Restricting the generator to 3-4 seeds and one path per seed avoids overloading the
#   cross hub, which would otherwise become ambiguous if many paths tried to share a
#   small center.
# - The seed rectangles still live on outer edges with a gap between zones, so branch
#   colors and seed interiors remain easy to distinguish.
#
# g) Change inferability
# - The only color-changing logic is still "branch/border -> seed interior majority"
#   plus the final hub vote. The new inference burden is structural, not an extra color
#   code: spot the cross hub and notice that the full cross inherits the winning branch
#   color.
#
# h) Non-degeneracy check used after implementation
# - Generated samples are measured by two metrics:
#   1. detected hub size must be exactly 5 cells with one degree-4 center;
#   2. the number of hub pixels that change color from input to output must be 5, not 1.
# - This rejects silent regression to a square hub or to a center-only recolor rule.
#
# i) New solver assumptions that generate_input() must guarantee
# - There is exactly one degree-4 P center in the whole network.
# - Every seed region has a P frame on each non-edge side, so seed-majority lookup from
#   adjacent branch components remains valid.
# - The hub vote is visually decisive: majority branch colors are generated with a
#   strict winner, so the whole-cross recolor is unambiguous.

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Canvas / grid
GRID_SIZE_BASE = 12
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30

# Hub
HUB_GUARD_RADIUS = 2

# Seed regions
SEED_COUNT_CHOICES = [3, 4]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 500
SEED_ZONE_GAP = 1

# Path carving
MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_ASSIGNMENT_RETRIES = 40
PATH_DIR_RETRIES = 4

# Top-level generation
GEN_OUTER_RETRIES = 400


def _orth_degree(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask.astype(bool), 1)
    return (
        pad[2:, 1:-1].astype(int)
        + pad[:-2, 1:-1].astype(int)
        + pad[1:-1, 2:].astype(int)
        + pad[1:-1, :-2].astype(int)
    )


def _hub_cells_from_center(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, center)
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _hub_exit_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    return [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]


def _find_cross_hub(g: np.ndarray, P: int) -> tuple[tuple[int, int], np.ndarray]:
    H, W = g.shape
    pmask = g == int(P)
    deg = _orth_degree(pmask)
    candidates = []
    for r, c in np.argwhere(pmask & (deg == 4)):
        r = int(r)
        c = int(c)
        if not (0 < r < H - 1 and 0 < c < W - 1):
            continue
        hub_cells = _hub_cells_from_center((r, c))
        # Require an exact 5-cell plus inside the local 3x3 neighborhood so the
        # detector cannot silently accept a square-like hub.
        if all(pmask[rr, cc] for rr, cc in hub_cells) and int(np.sum(pmask[r - 1 : r + 2, c - 1 : c + 2])) == 5:
            candidates.append((r, c))
    assert len(candidates) == 1, f"expected one cross-hub center, got {len(candidates)}"
    center = candidates[0]
    hub = np.zeros((H, W), dtype=bool)
    for rr, cc in _hub_cells_from_center(center):
        hub[int(rr), int(cc)] = True
    return center, hub


def _majority_color(vals: list[int]) -> int:
    counts = Counter(int(v) for v in vals)
    return int(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors

    def comp_deg_score(mask: np.ndarray) -> float:
        comps = components_mask(mask, connectivity=4)
        if not comps:
            return -1e9
        best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
        best_sz = int(best_pts.shape[0])

        m = np.zeros(mask.shape, dtype=bool)
        m[best_pts[:, 0], best_pts[:, 1]] = True

        deg2 = 0
        deg4 = 0
        for r, c in best_pts:
            r = int(r)
            c = int(c)
            d = 0
            if r > 0 and bool(m[r - 1, c]):
                d += 1
            if r + 1 < H and bool(m[r + 1, c]):
                d += 1
            if c > 0 and bool(m[r, c - 1]):
                d += 1
            if c + 1 < W and bool(m[r, c + 1]):
                d += 1
            if d == 2:
                deg2 += 1
            elif d == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    return int(max(colors, key=lambda c: comp_deg_score(g == int(c))))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a seed-linked path network with a cross-shaped hub.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network whose core is a 5-cell
      plus-sign hub rather than a solid 3x3 block.
    - Seed regions are non-background, non-P rectangles attached to the outer grid
      edges; each seed is surrounded by a 1-pixel P frame on every side that is not on
      the grid boundary.

    Rule:
    - Infer P as the non-background color whose largest component looks most like an
      orthogonal network.
    - Detect the unique cross hub by finding the only P cell with four orthogonal P
      neighbors; exclude those five hub cells from branch-component labeling.
    - Recolor each remaining P component (path plus seed border) to the majority color
      of the seed interior it touches most.
    - Count the recolored branch pixels adjacent to the hub and repaint the entire hub
      cross with the most common adjacent branch color.

    Key insight:
    The important change is not just where the winning color is chosen but how much of
    the hub inherits it: the hub is a plus-sign anchor, and once the winning branch
    color is identified, all five hub pixels adopt that color together.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P = _infer_path_color(g, bg)
    _center, hub = _find_cross_hub(g, P)

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    all_seed_vals: list[int] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing bottom P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1)), "seed missing right P border"
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        seed_major[int(sid)] = _majority_color(vals)
        all_seed_vals.extend(vals)

    out = g.copy()

    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(P)
    p_deg = _orth_degree(pmask)
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction outside the hub"

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        sid = int(sorted(touches.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
        col = int(seed_major[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    if int(branch_n) > 0:
        for r, c in np.argwhere(hub):
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    assert all_seed_vals, "grid must contain seed regions"
    hub_color = int(sorted(touch_cols.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])
    out[hub] = int(hub_color)
    return out


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work: np.ndarray, P: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    border_cells: list[tuple[int, int]] = []
    landing_cells: list[tuple[int, int]] = []
    for cond, cells in [
        (ir0 > 0, [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < H - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < W - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[r, c] = P
                border_cells.append((r, c))
                landing_cells.append((r, c))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = P
            border_cells.append(rc)
    return border_cells, landing_cells


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge: str, H: int, W: int) -> tuple[int, int, int, int]:
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge == "top":
        ir0, ic0 = 0, random.randint(0, W - iw)
    elif edge == "bottom":
        ir0, ic0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, H - ih), 0
    else:
        ir0, ic0 = random.randint(0, H - ih), W - iw
    return ir0, ic0, ir0 + ih - 1, ic0 + iw - 1


def _fill_seed_interior(
    work: np.ndarray,
    seed_id_grid: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    majority: int,
    minority: int,
    sid: int,
) -> None:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = minority
        seed_id_grid[r, c] = sid
    for r, c in interior_cells[n_minor:]:
        work[r, c] = majority
        seed_id_grid[r, c] = sid


def _entry_ok(canvas: np.ndarray, entry: tuple[int, int], hub_cells: set[tuple[int, int]]) -> bool:
    H, W = canvas.shape
    r, c = map(int, entry)
    if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
        return False
    hub_neighbors = 0
    for dr, dc in _DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if (nr, nc) in hub_cells:
            hub_neighbors += 1
        elif canvas[nr, nc] != 0:
            return False
    return hub_neighbors == 1


def _try_carve_path(
    canvas: np.ndarray,
    P: int,
    entry: tuple[int, int],
    hub_cells: set[tuple[int, int]],
    dest_border_cells: list[tuple[int, int]],
    max_bends: int = MAX_BENDS,
) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _adjacent_to_dest(r: int, c: int) -> bool:
        return any((r + dr, c + dc) in dest_set for dr, dc in _DIRS)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev:
                continue
            v = int(canvas[nr, nc])
            if v == int(P):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    if not _entry_ok(canvas, entry, hub_cells):
        return None

    er, ec = map(int, entry)
    canvas[er, ec] = int(P)
    placed = [(er, ec)]
    if _adjacent_to_dest(er, ec):
        return placed

    init_dirs = []
    for dr, dc in _DIRS:
        nr = int(er + dr)
        nc = int(ec + dc)
        if (nr, nc) in hub_cells:
            continue
        if 0 <= nr < H and 0 <= nc < W and canvas[nr, nc] == 0:
            init_dirs.append((dr, dc))
    random.shuffle(init_dirs)
    if not init_dirs:
        canvas[er, ec] = 0
        return None

    def _attempt(initial_dir: tuple[int, int], n_bends: int) -> list[tuple[int, int]] | None:
        cur = (er, ec)
        cur_dir = initial_dir
        added: list[tuple[int, int]] = []
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr = int(cur[0] + cur_dir[0])
                nc = int(cur[1] + cur_dir[1])
                if _adjacent_to_dest(nr, nc) and _check_neighbors(nr, nc, cur, allow_dest=True):
                    canvas[nr, nc] = int(P)
                    added.append((nr, nc))
                    return placed + added
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(P)
                    added.append((nr, nc))
                    cur = (nr, nc)
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in added:
                    canvas[r, c] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = perps[0]
        for r, c in added:
            canvas[r, c] = 0
        return None

    for d in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = _attempt(d, random.randint(0, max_bends))
            if res is not None:
                return res

    for r, c in placed:
        canvas[r, c] = 0
    return None


def _carve_all_paths(
    work: np.ndarray,
    P: int,
    hub_cells: set[tuple[int, int]],
    hub_exits: list[tuple[int, int]],
    seeds: list[dict[str, object]],
) -> bool:
    baseline = work.copy()
    for _ in range(PATH_ASSIGNMENT_RETRIES):
        work[:, :] = baseline
        ok = True
        exits = random.sample(hub_exits, len(hub_exits))
        seed_order = list(range(len(seeds)))
        random.shuffle(seed_order)
        for si, entry in zip(seed_order, exits):
            path = _try_carve_path(work, P, entry, hub_cells, seeds[si]["landing_cells"])  # type: ignore[index]
            if path is None:
                ok = False
                break
        if ok:
            return True
    work[:, :] = baseline
    return False


def _planned_majorities(n: int, A: int, B: int) -> list[int]:
    if int(n) == 3:
        majors = [A, A, B] if random.random() < 0.5 else [A, B, B]
    else:
        majors = [A, A, A, B] if random.random() < 0.5 else [A, B, B, B]
    random.shuffle(majors)
    return majors


def _gen_attempt() -> np.ndarray | None:
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    cr = random.randint(max(3, H // 3), min(H - 4, 2 * H // 3))
    cc = random.randint(max(3, W // 3), min(W - 4, 2 * W // 3))
    hub_center = (cr, cc)
    hub_cells = _hub_cells_from_center(hub_center)
    hub_exits = _hub_exit_cells(hub_center)
    for r, c in hub_cells:
        g[r, c] = P

    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds: list[dict[str, object]] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]
    majorities = _planned_majorities(n_seeds_target, A, B)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < cr - HUB_GUARD_RADIUS or r0 > cr + HUB_GUARD_RADIUS or c1 < cc - HUB_GUARD_RADIUS or c0 > cc + HUB_GUARD_RADIUS)

    for _ in range(SEED_PLACEMENT_RETRIES):
        if len(seeds) >= n_seeds_target:
            break
        edge = random.choice(edges)
        ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W)
        border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)

        if _overlaps_hub(border_zone):
            continue
        if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
            continue

        majority = int(majorities[len(seeds)])
        minority = int(B if majority == A else A)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W)

        used_zones.append(border_zone)
        seeds.append(
            {
                "id": sid,
                "majority": majority,
                "minority": minority,
                "border_cells": border_cells,
                "landing_cells": landing_cells,
                "bbox": (ir0, ic0, ir1, ic1),
                "border_zone": border_zone,
            }
        )

    if len(seeds) != n_seeds_target:
        return None

    if not _carve_all_paths(work, P, hub_cells, hub_exits, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    center, hub_mask = _find_cross_hub(inp, P)
    cr, cc = map(int, center)
    if int(np.sum(hub_mask)) != 5:
        return None
    if int(np.sum((inp == P)[cr - 1 : cr + 2, cc - 1 : cc + 2])) != 5:
        return None
    if len(set(int(out[r, c]) for r, c in np.argwhere(hub_mask))) != 1:
        return None
    if int(np.sum((inp == P) & hub_mask & (out != P))) != 5:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            r = _gen_attempt()
        except Exception:
            r = None
        if r is not None:
            return r
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

# Combo variant of 800d221b (index 10)
# combo_spec: D1/0+D14/2
# Name: 弯折数决定着色规则

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for this combo variant
# ---------------------------------------------------------------------------
#
# fusion_summary:
#   - The original task recolors every branch by the majority color of the seed
#     region it reaches. D1/0 introduces the opposite choice: use the minority
#     color instead. D14/2 then ties these two choices together by making bend
#     parity the switch: even bends -> majority, odd bends -> minority.
#   - The fused task therefore cannot be solved by local adjacency alone. The
#     solver must identify the seed reached by each branch, compute that seed's
#     majority/minority pair, isolate the actual path chain, count its bends,
#     and only then choose the recolor.
#
# concrete_behavior_trace:
#   - Small example: suppose a top seed interior contains mostly red with a few
#     green cells. In the original task, any connected P-branch reaching that
#     seed would become red.
#   - In this combo, if the branch path from hub to that seed border is
#     straight or has 2 bends, it still becomes red. If the same seed is reached
#     by a 1-bend or 3-bend route, the branch becomes green instead.
#   - The visible difference is strongest when two seeds share the same majority
#     color but their path shapes have opposite bend parity; then one output
#     branch keeps the intuitive majority color while the other flips.
#
# complexity_calibration:
#   - To keep the parity rule observable, generation uses exactly four seeds,
#     one per grid edge, and exactly one branch per seed. This avoids ambiguous
#     multi-path seed components while still leaving enough repeated structure
#     for humans to compare odd and even cases in one sample.
#   - Seeds still use only two interior colors, so "majority vs minority" stays
#     visually recoverable from a few examples instead of exploding into many
#     arbitrary color mappings.
#   - Samples are rejected unless they contain both bend parities, both majority
#     colors, and at least one same-majority / different-parity pair. That keeps
#     the fused rule from collapsing back into the original majority-only task
#     or the pure minority-only inversion.
#
# hidden_assumption_audit:
#   - Original solver assumption: recolor depends only on which seed a branch
#     touches. This is no longer sufficient; the solver now must separate seed
#     border cells from the actual route cells and count direction changes.
#   - Original branch labeling (`all P except hub`) still works, but only if
#     each seed has one branch. The generator enforces that so one connected
#     component corresponds to exactly one seed border plus one path chain.
#   - New solver assumption: within a branch component, the non-border path cells
#     form a simple chain with one hub-adjacent endpoint and one seed-border-
#     adjacent endpoint. The generator preserves the original no-T-junction path
#     carving and uses non-corner landing cells so this remains true.
#
# placement_legality_and_cleanliness:
#   - Seed interiors touch the grid edge, keep their original P frame on the
#     non-edge sides, and are separated by gap-expanded bounding boxes.
#   - Paths are carved on a work canvas that already contains the hub and seed
#     borders, so they cannot overlap seeds, cannot leave bounds, and cannot
#     form unintended branch junctions.
#   - The visual layering stays clean: bg, one path color P, and two seed colors.
#     The only extra reasoning burden is counting branch bends, which is a
#     human-visible geometric property rather than a hidden numeric trick.
#
# observability_and_non_degeneracy:
#   - A valid sample must exhibit:
#       1. both even-bend and odd-bend branches,
#       2. both majority seed colors,
#       3. at least one branch whose output differs from the original-majority
#          rule, and
#       4. at least one pair of seeds with the same majority but different
#          output due solely to bend parity.
#   - The helper `_fusion_metrics` is shared by generation-time validation and
#     post-hoc testing so we can measure whether the combo behavior is actually
#     present instead of silently regressing to the base task.
#
# post_change_assumptions_checked_against_main:
#   - `main()` assumes the path color is the most network-like non-bg color; the
#     generator keeps exactly one such color.
#   - `main()` assumes the hub is the unique largest solid all-P square; the hub
#     stays 3x3 and paths are 1-pixel wide, so no competing P square appears.
#   - `main()` assumes every seed interior contains a strict majority and a
#     strict minority; generator enforces minority count in [6%, 40%) and avoids
#     ties.
#   - `main()` assumes every branch touches at least one recolored hub side; with
#     four seeds and one branch per seed, the hub is always touched.


GRID_SIZE_BASE = 13
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 15
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT = 4
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 3
MINORITY_FRAC_LO = 0.10
MINORITY_FRAC_HI = 0.35
SEED_PLACEMENT_RETRIES = 300
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_ROUND_RETRIES = 80
PATH_DIR_RETRIES = 4

GEN_OUTER_RETRIES = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _comp_deg_score(mask: np.ndarray) -> float:
    H, W = mask.shape
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


def _largest_solid_square_size(mask: np.ndarray) -> int:
    H, W = mask.shape
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = 0
    for k in range(2, int(min(H, W)) + 1):
        found = False
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    best = int(k)
                    found = True
                    break
            if found:
                break
    return int(best)


def _find_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    return int(
        max(
            colors,
            key=lambda c: (
                _largest_solid_square_size(g == int(c)),
                _comp_deg_score(g == int(c)),
            ),
        )
    )


def _find_hub(mask_p: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, W = mask_p.shape
    ps = mask_p.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in range(2, int(min(H, W)) + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found
    assert best is not None

    k, r0, c0 = best
    hub = np.zeros(mask_p.shape, dtype=bool)
    hub[int(r0) : int(r0 + k), int(c0) : int(c0 + k)] = True
    return hub, (int(k), int(r0), int(c0))


def _adjacent_to_mask(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask, 1)
    return pad[2:, 1:-1] | pad[:-2, 1:-1] | pad[1:-1, 2:] | pad[1:-1, :-2]


def _seed_stats(g: np.ndarray, bg: int, p_color: int) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(p_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    stats: dict[int, dict[str, object]] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(p_color) for c in range(cn, cx + 1))
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(p_color) for c in range(cn, cx + 1))
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(p_color) for r in range(rn, rx + 1))
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(p_color) for r in range(rn, rx + 1))

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        cnt = Counter(vals).most_common()
        assert len(cnt) >= 2, "seed must contain majority and minority colors"
        majority = int(cnt[0][0])
        minority = int(cnt[-1][0])
        assert int(cnt[0][1]) > int(cnt[-1][1]), "seed majority must be strict"
        stats[int(sid)] = {
            "majority": int(majority),
            "minority": int(minority),
            "bbox": (int(rn), int(cn), int(rx), int(cx)),
        }
    return seed_lab, stats


def _neighbors_matching(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    H, W = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
            out.append((int(rr), int(cc)))
    return out


def _ordered_path_cells(path_mask: np.ndarray, hub: np.ndarray, border_mask: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert pts, "branch path must contain non-border cells"

    start_cells = []
    end_cells = []
    path_degrees: dict[tuple[int, int], int] = {}
    border_neighbors_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        path_nbs = _neighbors_matching(path_mask, int(r), int(c))
        path_degrees[(int(r), int(c))] = int(len(path_nbs))
        assert len(path_nbs) <= 2, "path chain must stay degree <= 2"
        hub_nbs = _neighbors_matching(hub, int(r), int(c))
        border_nbs = _neighbors_matching(border_mask, int(r), int(c))
        border_neighbors_map[(int(r), int(c))] = list(border_nbs)
        if hub_nbs and len(path_nbs) <= 1:
            start_cells.append(((int(r), int(c)), hub_nbs))
        if border_nbs and len(path_nbs) <= 1:
            end_cells.append(((int(r), int(c)), border_nbs))

    assert len(start_cells) == 1, "branch must have one hub-side endpoint"
    assert len(end_cells) == 1, "branch must have one seed-side endpoint"

    start, hub_nbs = start_cells[0]
    end, border_nbs = end_cells[0]
    assert len(hub_nbs) == 1
    assert len(border_nbs) == 1
    hub_neighbor = tuple(map(int, hub_nbs[0]))
    border_neighbor = tuple(map(int, border_nbs[0]))

    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        options = [nb for nb in _neighbors_matching(path_mask, int(cur[0]), int(cur[1])) if nb != prev]
        assert len(options) == 1, "path ordering must be unique"
        prev = cur
        cur = tuple(map(int, options[0]))
        ordered.append(cur)

    end_border_nbs = border_neighbors_map[end]
    if len(end_border_nbs) > 1:
        prev_cell = ordered[-2] if len(ordered) >= 2 else hub_neighbor
        in_dir = (int(end[0] - prev_cell[0]), int(end[1] - prev_cell[1]))
        aligned = []
        for nb in end_border_nbs:
            out_dir = (int(nb[0] - end[0]), int(nb[1] - end[1]))
            if out_dir == in_dir:
                aligned.append(tuple(map(int, nb)))
        if len(aligned) == 1:
            border_neighbor = aligned[0]
        else:
            assert len(end_border_nbs) == 1, "branch endpoint must have unambiguous border contact"

    return ordered, hub_neighbor, border_neighbor


def _count_bends(path_mask: np.ndarray, hub: np.ndarray, border_mask: np.ndarray) -> int:
    ordered_path, hub_neighbor, border_neighbor = _ordered_path_cells(path_mask, hub, border_mask)
    full_chain = [hub_neighbor, *ordered_path, border_neighbor]
    dirs = []
    for a, b in zip(full_chain, full_chain[1:]):
        dirs.append((int(b[0] - a[0]), int(b[1] - a[1])))
    bends = 0
    for prev_dir, cur_dir in zip(dirs, dirs[1:]):
        if tuple(prev_dir) != tuple(cur_dir):
            bends += 1
    return int(bends)


def _analyze_branches(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    p_color = int(_find_path_color(g, bg))

    hub, (k, r0, c0) = _find_hub(g == int(p_color))
    seed_lab, seed_info = _seed_stats(g, bg, p_color)

    out = g.copy()
    branch_mask = (g == int(p_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(p_color)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    adj_seed = _adjacent_to_mask(seed_lab > 0)
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    branches: list[dict[str, int]] = []
    for bid in range(1, int(branch_n) + 1):
        bmask = branch_lab == int(bid)
        pts = np.argwhere(bmask)
        touches = Counter()
        for r, c in pts:
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        assert touches, "branch must touch a seed"
        sid = int(touches.most_common(1)[0][0])
        rn, cn, rx, cx = seed_info[int(sid)]["bbox"]
        frame = np.zeros_like(bmask, dtype=bool)
        er0 = max(0, int(rn - 1))
        ec0 = max(0, int(cn - 1))
        er1 = min(H - 1, int(rx + 1))
        ec1 = min(W - 1, int(cx + 1))
        frame[int(er0) : int(er1) + 1, int(ec0) : int(ec1) + 1] = True
        frame[int(rn) : int(rx) + 1, int(cn) : int(cx) + 1] = False
        border_mask = bmask & frame
        path_only_mask = bmask & (~border_mask)
        bends = int(_count_bends(path_only_mask, hub, border_mask))
        majority = int(seed_info[int(sid)]["majority"])
        minority = int(seed_info[int(sid)]["minority"])
        color = int(majority if bends % 2 == 0 else minority)
        out[bmask] = int(color)
        branches.append(
            {
                "branch_id": int(bid),
                "seed_id": int(sid),
                "bends": int(bends),
                "parity": int(bends % 2),
                "majority": int(majority),
                "minority": int(minority),
                "color": int(color),
            }
        )

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in DIRS4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    center_r = int(r0 + k // 2)
    center_c = int(c0 + k // 2)
    out[center_r, center_c] = int(touch_cols.most_common(1)[0][0])
    return {
        "bg": int(bg),
        "p_color": int(p_color),
        "hub": hub,
        "hub_box": (int(k), int(r0), int(c0)),
        "seed_lab": seed_lab,
        "seed_info": seed_info,
        "branches": branches,
        "out": out,
    }


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    analysis = _analyze_branches(grid)
    branches = analysis["branches"]
    parities = Counter(int(b["parity"]) for b in branches)
    majority_colors = Counter(int(b["majority"]) for b in branches)
    changed_vs_original = sum(int(b["color"]) != int(b["majority"]) for b in branches)

    same_majority_diff_parity = False
    same_parity_diff_majority = False
    for i, bi in enumerate(branches):
        for bj in branches[i + 1 :]:
            if int(bi["majority"]) == int(bj["majority"]) and int(bi["parity"]) != int(bj["parity"]):
                same_majority_diff_parity = True
            if int(bi["parity"]) == int(bj["parity"]) and int(bi["majority"]) != int(bj["majority"]):
                same_parity_diff_majority = True

    return {
        "branch_count": int(len(branches)),
        "parities": {int(k): int(v) for k, v in parities.items()},
        "majorities": {int(k): int(v) for k, v in majority_colors.items()},
        "changed_vs_original": int(changed_vs_original),
        "same_majority_diff_parity": bool(same_majority_diff_parity),
        "same_parity_diff_majority": bool(same_parity_diff_majority),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network using seed colors plus bend parity.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid
      square hub near the middle and one branch leading to each seed frame.
    - Each seed is a rectangular region of two colors surrounded by a P border
      on the sides that are not flush with the grid edge.

    Rule:
    - Infer P and the hub, then identify which seed each non-hub P component
      reaches.
    - For every seed, compute both the majority interior color and the minority
      interior color.
    - Within each branch component, ignore the seed-border cells, count the
      bends of the actual path chain from hub to seed, and recolor the whole
      branch to majority if the bend count is even or minority if it is odd.
    - Recolor only the hub center pixel to the most common color among the
      recolored branch pixels that touch the hub.

    Key insight:
    The touched seed still tells you which two candidate colors matter, but it
    no longer determines the answer alone; the branch geometry encodes a parity
    bit that chooses between the seed's majority and minority color.
    """
    return np.array(_analyze_branches(grid)["out"], dtype=int)


def _try_carve_path(canvas: np.ndarray, p_color: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set((int(r), int(c)) for r, c in dest_border_cells)

    def check_neighbors(r: int, c: int, prev: tuple[int, int] | None, allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != 0:
            return False
        for dr, dc in DIRS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            if (nr, nc) == start:
                continue
            v = int(canvas[nr, nc])
            if v == int(p_color):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in DIRS4:
        nr = int(sr + dr)
        nc = int(sc + dc)
        if 0 <= nr < H and 0 <= nc < W and int(canvas[nr, nc]) == 0:
            init_dirs.append((int(dr), int(dc)))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    def attempt(initial_dir: tuple[int, int]) -> list[tuple[int, int]] | None:
        target_bends = random.randint(0, MAX_BENDS)
        path: list[tuple[int, int]] = []
        cur = (int(sr), int(sc))
        cur_dir = (int(initial_dir[0]), int(initial_dir[1]))
        for seg_i in range(target_bends + 1):
            is_last = seg_i == target_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr = int(cur[0] + cur_dir[0])
                nc = int(cur[1] + cur_dir[1])
                if any((nr + dr, nc + dc) in dest_set for dr, dc in DIRS4) and check_neighbors(nr, nc, cur, allow_dest=True):
                    canvas[nr, nc] = int(p_color)
                    path.append((int(nr), int(nc)))
                    return path
                if check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(p_color)
                    path.append((int(nr), int(nc)))
                    cur = (int(nr), int(nc))
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in path:
                    canvas[int(r), int(c)] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = (int(perps[0][0]), int(perps[0][1]))
        for r, c in path:
            canvas[int(r), int(c)] = 0
        return None

    for initial_dir in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = attempt(initial_dir)
            if res is not None:
                return res
    return None


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (int(ir0), max(0, int(ic0 - 1)), int(ir1 + 1), min(W - 1, int(ic1 + 1)))
    if edge == "bottom":
        return (max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), int(ir1), min(W - 1, int(ic1 + 1)))
    if edge == "left":
        return (max(0, int(ir0 - 1)), int(ic0), min(H - 1, int(ir1 + 1)), int(ic1 + 1))
    return (max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), min(H - 1, int(ir1 + 1)), int(ic1))


def _place_seed_border(work: np.ndarray, p_color: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
                work[int(r), int(c)] = int(p_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(p_color)
            border_cells.append((int(rc[0]), int(rc[1])))
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
        ir0 = 0
        ic0 = random.randint(1, max(1, W - iw - 1))
    elif edge == "bottom":
        ir0 = int(H - ih)
        ic0 = random.randint(1, max(1, W - iw - 1))
    elif edge == "left":
        ir0 = random.randint(1, max(1, H - ih - 1))
        ic0 = 0
    else:
        ir0 = random.randint(1, max(1, H - ih - 1))
        ic0 = int(W - iw)
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int, sid: int) -> None:
    interior_cells = [(int(r), int(c)) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(round(n_int * MINORITY_FRAC_LO)))
    max_minor = min(n_int // 2, max(min_minor, int(round(n_int * MINORITY_FRAC_HI))))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[int(r), int(c)] = int(minority)
        seed_id_grid[int(r), int(c)] = int(sid)
    for r, c in interior_cells[n_minor:]:
        work[int(r), int(c)] = int(majority)
        seed_id_grid[int(r), int(c)] = int(sid)


def _carve_one_path_per_seed(work: np.ndarray, p_color: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict[str, object]]) -> bool:
    used_hub_starts: set[tuple[int, int]] = set()
    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        success = False
        for _ in range(PATH_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, p_color, start, list(seeds[si]["landing_cells"]))
            if path is not None:
                used_hub_starts.add(start)
                success = True
                break
        if not success:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (SEED_COUNT - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, p_color, color_a, color_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)
    hr0 = random.randint(max(3, H // 3 - 1), min(H - 6, 2 * H // 3 - 2))
    hc0 = random.randint(max(3, W // 3 - 1), min(W - 6, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(p_color)

    hub_perimeter = [
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds: list[dict[str, object]] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]
    majority_plan = [color_a, color_a, color_b, color_b]
    random.shuffle(majority_plan)

    def overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    for edge, majority in zip(edges, majority_plan):
        placed = False
        minority = int(color_b if majority == color_a else color_a)
        for _ in range(SEED_PLACEMENT_RETRIES):
            ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W)
            border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
            if overlaps_hub(border_zone):
                continue
            if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
                continue

            sid = len(seeds) + 1
            _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
            border_cells, landing_cells = _place_seed_border(work, p_color, ir0, ic0, ir1, ic1, H, W)
            used_zones.append(border_zone)
            seeds.append(
                {
                    "id": int(sid),
                    "majority": int(majority),
                    "minority": int(minority),
                    "border_cells": border_cells,
                    "landing_cells": landing_cells,
                    "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                    "edge": str(edge),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not _carve_one_path_per_seed(work, p_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    if int(Counter(int(x) for x in inp.flatten()).most_common(1)[0][0]) != int(bg):
        return None

    try:
        analysis = _analyze_branches(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if len(analysis["branches"]) != SEED_COUNT:
        return None
    if set(metrics["parities"]) != {0, 1}:
        return None
    if len(metrics["majorities"]) != 2:
        return None
    if int(metrics["changed_vs_original"]) == 0:
        return None
    if not bool(metrics["same_majority_diff_parity"]):
        return None
    if not bool(metrics["same_parity_diff_majority"]):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            candidate = _gen_attempt()
        except Exception:
            candidate = None
        if candidate is not None:
            return np.array(candidate, dtype=int)
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

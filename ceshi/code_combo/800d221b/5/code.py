# Combo variant of 800d221b (index 5)
# combo_spec: D1/3+D3/1
# Name: 颜色交换+双向传播
#
# Design analysis
# a) Concrete behavior trace:
#    - In the base task, a seed interior keeps its two-color mix, while the whole connected P structure
#      leading to that seed (branch + local P frame) turns into the seed majority color.
#    - In this combo, the seed is split into three coordinated roles:
#      1. branch pixels take the seed's original majority color,
#      2. the seed interior swaps majority/minority colors,
#      3. the seed's P border takes the seed's original minority color.
#    - Example: if a seed interior is mostly A with a few B cells, then output becomes:
#      branch -> A, border -> B, interior A/B swapped so the former A area becomes B and the former B area becomes A.
#      This differs from the base task because the border no longer matches the branch, and the interior no longer stays fixed.
# b) Complexity calibration:
#    - Only two seed colors are used across the whole grid, as in the original family, so humans can track the swap.
#    - The generator forces both majority choices to appear among seeds, giving repeated evidence that "branch follows
#      original majority while border follows original minority" is a structural rule, not a color-id coincidence.
#    - Paths remain simple chains and seeds remain rectangles, so the new difficulty comes from rule fusion, not clutter.
# c) Hidden-assumption audit:
#    - Path-color inference still works because the input still contains a single large network-like P component.
#    - Hub detection via largest solid all-P square still holds because only the input is analyzed there.
#    - Seed extraction via non-bg/non-P connected components still holds because interiors remain rectangular color regions.
#    - New assumption: border cells must be recoverable from each seed bbox. The generator preserves the canonical
#      one-cell P frame on every non-edge side, including corners when both adjacent sides exist.
#    - New assumption: a branch component should touch border cells of exactly one seed. The generator keeps seed zones
#      separated and carves simple non-branching paths, so each branch endpoint has an unambiguous owning seed.
# d) Placement legality:
#    - Seeds are placed in disjoint edge-touching zones with a configurable gap and hub-overlap rejection.
#    - Paths are carved one pixel wide, in bounds, without entering seeds or existing P cells except at the intended border.
#    - The hub stays a central 3x3 square; seeds and paths are only accepted when the solver can process the resulting input.
# e) Visual observability:
#    - The combo is visible in three separate places: branches, borders, and seed interiors.
#    - Because branch and border now intentionally disagree around each seed, the output cannot be mistaken for the base task.
#    - Reusing only two seed colors across multiple seeds gives enough repeated support in 2-4 examples.
# f) Visual cleanliness:
#    - The border frame is still a one-pixel ring and the branch remains outside it, so the three transformed regions stay legible.
#    - Seed zones are spaced apart to avoid merged frames or overlapping path neighborhoods.
# g) Change inferability:
#    - The color change is not keyed to raw color ids. It is keyed to structural roles:
#      original majority inside the seed, original minority inside the seed, and the seed's surrounding P frame.
#    - The generator ensures the original majority varies across seeds, which makes the fused rule inferable from structure alone.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _largest_all_p_square(g: np.ndarray, P: int) -> tuple[int, int, int]:
    H, W = g.shape
    mP = g == int(P)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

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
    return best


def _seed_border_cells(rn: int, rx: int, cn: int, cx: int, H: int, W: int) -> list[tuple[int, int]]:
    border = []
    if rn > 0:
        border.extend((rn - 1, c) for c in range(cn, cx + 1))
    if rx < H - 1:
        border.extend((rx + 1, c) for c in range(cn, cx + 1))
    if cn > 0:
        border.extend((r, cn - 1) for r in range(rn, rx + 1))
    if cx < W - 1:
        border.extend((r, cx + 1) for r in range(rn, rx + 1))
    if rn > 0 and cn > 0:
        border.append((rn - 1, cn - 1))
    if rn > 0 and cx < W - 1:
        border.append((rn - 1, cx + 1))
    if rx < H - 1 and cn > 0:
        border.append((rx + 1, cn - 1))
    if rx < H - 1 and cx < W - 1:
        border.append((rx + 1, cx + 1))
    return border


def _extract_seed_info(g: np.ndarray, bg: int, P: int):
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0

    border_owner = np.zeros((H, W), dtype=int)
    seed_info: dict[int, dict[str, object]] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())

        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing bot P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1)), "seed missing right P border"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        freq = Counter(vals).most_common()
        assert len(freq) >= 2, "seed interior must contain majority and minority colors"
        assert freq[0][1] > freq[1][1], "seed interior must have a strict majority"
        major = int(freq[0][0])
        minor = int(freq[-1][0])

        border_cells = _seed_border_cells(rn, rx, cn, cx, H, W)
        for br, bc in border_cells:
            assert int(g[br, bc]) == int(P), "seed border must be color P in input"
            assert int(border_owner[br, bc]) in (0, sid), "seed borders must stay disjoint"
            border_owner[br, bc] = int(sid)

        seed_info[int(sid)] = {
            "pts": pts,
            "major": major,
            "minor": minor,
            "bbox": (rn, cn, rx, cx),
            "border_cells": border_cells,
        }
    return seed_lab, border_owner, seed_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a path network by fusing seed-majority transfer with interior/border inversion.

    Input structure:
    - bg is the most common color.
    - A thin orthogonal P network contains a solid square hub and outward path branches.
    - Edge-touching seed rectangles have non-P interiors and a one-cell P frame on every non-edge side.

    Rule:
    - Infer P from the most network-like non-bg component and locate the largest solid all-P hub square.
    - For each seed rectangle, read its original majority/minority interior colors, then swap those two colors inside the seed.
    - Recolor every branch attached to that seed with the seed's original majority color, while recoloring that seed's P frame with its original minority color.
    - Recolor the hub center pixel with the color of the majority-colored branches that touch the hub most.

    Key insight:
    - The seed controls three synchronized changes at once: majority flows outward to the branch, minority flows outward to the frame, and the seed interior flips those same two roles.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P = _infer_path_color(g, bg)
    k, r0, c0 = _largest_all_p_square(g, P)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)

    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    seed_lab, border_owner, seed_info = _extract_seed_info(g, bg, P)
    out = g.copy()

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    border_mask = border_owner > 0

    for sid, info in seed_info.items():
        major = int(info["major"])
        minor = int(info["minor"])
        for r, c in info["pts"]:
            r = int(r)
            c = int(c)
            out[r, c] = minor if int(g[r, c]) == major else major
        for r, c in info["border_cells"]:
            out[int(r), int(c)] = int(minor)

    branch_mask = (g == int(P)) & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    branch_color: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(border_owner[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        assert touches, "every branch must attach to a seed border"
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_info[int(sid)]["major"])
        branch_color[int(bid)] = int(col)
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    bid = int(branch_lab[rr, cc])
                    if bid != 0:
                        touch_cols[int(branch_color[int(bid)])] += 1

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    assert touch_cols, "hub must be touched by recolored branches"
    out[int(cr), int(cc)] = int(touch_cols.most_common(1)[0][0])
    return out


# --------------- generator ---------------

# Canvas / grid
GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

# Hub
HUB_SIZE = 3

# Seed regions
SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1

# Path carving
MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_FIRST_ROUND_RETRIES = 60
PATH_EXTRA_ROUND_RETRIES = 30
PATH_DIR_RETRIES = 3

# Top-level generation
GEN_OUTER_RETRIES = 300


def _try_carve_path(canvas, P, start, dest_border_cells, max_bends=MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r, c, prev, allow_dest):
        if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = canvas[nr, nc]
            if v == P:
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in _DIRS:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < H and 0 <= nc < W and canvas[nr, nc] == 0:
            init_dirs.append((dr, dc))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    n_bends = random.randint(0, max_bends)

    def _attempt(initial_dir):
        path = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                if (
                    any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS)
                    and _check_neighbors(nr, nc, cur, allow_dest=True)
                ):
                    canvas[nr, nc] = P
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = P
                    path.append((nr, nc))
                    cur = (nr, nc)
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in path:
                    canvas[r, c] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = perps[0]
        for r, c in path:
            canvas[r, c] = 0
        return None

    for d in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = _attempt(d)
            if res is not None:
                return res
    return None


def _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W):
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W):
    border_cells = []
    landing_cells = []
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


def _zones_overlap(zone, others, gap):
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge, H, W):
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


def _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid):
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


def _carve_all_paths(work, P, hub_perimeter, seeds):
    used_hub_starts = set()
    paths_per_seed = {i: 0 for i in range(len(seeds))}

    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        ok = False
        for _ in range(PATH_FIRST_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, P, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                ok = True
                break
        if not ok:
            return False

    extra_target = sum(random.randint(0, 2) for _ in seeds)
    for _ in range(extra_target * 2):
        if len(used_hub_starts) >= len(hub_perimeter):
            break
        candidates = [si for si in range(len(seeds)) if paths_per_seed[si] < MAX_PATHS_PER_SEED]
        if not candidates:
            break
        si = random.choice(candidates)
        for _ in range(PATH_EXTRA_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, P, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                break
    return True


def _fusion_metric(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, int]:
    bg = int(get_bg(inp))
    P = _infer_path_color(inp, bg)
    seed_lab, border_owner, seed_info = _extract_seed_info(inp, bg, P)

    swapped_seeds = 0
    minority_borders = 0
    branch_border_contrasts = 0
    border_mask = border_owner > 0
    branch_mask = (inp == int(P)) & (~border_mask)

    for sid, info in seed_info.items():
        major = int(info["major"])
        minor = int(info["minor"])
        pts = info["pts"]
        if all(int(out[int(r), int(c)]) == (minor if int(inp[int(r), int(c)]) == major else major) for r, c in pts):
            swapped_seeds += 1
        if all(int(out[int(r), int(c)]) == minor for r, c in info["border_cells"]):
            minority_borders += 1

        border_touch_cols = set()
        for r, c in info["border_cells"]:
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] and bool(branch_mask[rr, cc]):
                    border_touch_cols.add(int(out[rr, cc]))
        if border_touch_cols == {major} and major != minor:
            branch_border_contrasts += 1

    return swapped_seeds, minority_borders, branch_border_contrasts


def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = P

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    majority_plan = [A, B] + [random.choice([A, B]) for _ in range(n_seeds_target - 2)]
    random.shuffle(majority_plan)

    def _overlaps_hub(zone):
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    plan_i = 0
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

        majority = majority_plan[plan_i]
        minority = B if majority == A else A
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
        plan_i += 1

    if len(seeds) < 3:
        return None
    if len({int(seed["majority"]) for seed in seeds}) < 2:
        return None

    if not _carve_all_paths(work, P, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    swapped_seeds, minority_borders, branch_border_contrasts = _fusion_metric(inp, out)
    if swapped_seeds != len(seeds):
        return None
    if minority_borders != len(seeds):
        return None
    if branch_border_contrasts < len(seeds):
        return None
    return inp


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        try:
            res = _gen_attempt()
        except Exception:
            res = None
        if res is not None:
            return res
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

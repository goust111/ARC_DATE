# Combo variant of 800d221b (index 4)
# combo_spec: D1/0+D3/1
# Name: 少数色+双向传播

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


# Design analysis for the fusion variant:
# - Concrete behavior trace:
#   In the base task, a seed interior with colors {A majority, B minority} sends A onto the whole connected
#   P structure that reaches it, including the seed's P frame. Here the same geometry is preserved, but the
#   transferred color becomes B, and the seed frame must also switch from P to B. So an upper seed with a
#   mostly-red / few-green interior produces a green branch and a green frame, not a red one. The hub center
#   still votes only from the recolored branch pixels next to the hub, so the branch recolor and the border
#   recolor now agree and reinforce one minority-color interpretation.
# - Complexity calibration:
#   The geometry stays canonical: 3-5 edge seeds, one hub, 1-3 simple paths per seed, only two seed-interior
#   colors. That keeps the rule inferable from 2-4 examples while making the variant visibly different from the
#   original because every affected branch/frame flips to the less frequent seed color instead of the dominant one.
# - Hidden-assumption audit:
#   The original solver treated every non-hub P pixel as a single recolor target, which collapses branch cells and
#   seed-frame cells together. That is no longer safe because the fusion spec explicitly talks about "branch" and
#   "seed border" as two synchronized effects. The new solver therefore reconstructs each seed's frame from the
#   one-cell ring around its interior bbox, removes those border cells from the branch components, and assigns each
#   branch component by which seed frame it touches. This relies on rectangular seed interiors, side-border landing
#   points, and no T-junctions outside hub/frame cells; the generator preserves all of these.
# - Placement legality:
#   Seeds are still placed on grid edges with a one-cell P frame only on non-edge sides, zones stay separated,
#   the hub is kept away from seed zones, and paths carve only through empty background cells without overlapping
#   seeds, hub, or existing paths.
# - Visual observability / cleanliness:
#   The output keeps only four colors total. The minority color appears simultaneously on the path and on the seed
#   frame, giving a repeated local cue instead of a single subtle recolor. Multiple seeds reuse the same two
#   interior colors so humans can compare majority-vs-minority behavior across repeated structures.
# - Fusion verification target:
#   A generated sample is considered non-degenerate if the combo output differs from the original majority-based
#   output on at least one recolored P cell. Because every placed seed has distinct majority/minority colors and
#   every seed owns at least one border/path attachment, the generator is built so this difference should hold on
#   every sample.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network using each seed's minority color.

    Input structure:
    - `bg` is the most frequent color.
    - One non-bg color `P` forms a thin orthogonal network with a solid square hub.
    - Seed interiors are rectangular mixed-color regions (non-bg, non-P) that sit on grid edges and are wrapped
      by a 1-pixel P frame on the non-edge sides.

    Rule:
    - Infer `P`, locate the largest solid all-P square hub, and recover each seed interior plus its surrounding P frame.
    - For every path branch leaving the hub, find which seed frame it touches and recolor that branch to the seed's
      minority interior color.
    - Recolor the touched seed frame itself to the same minority color, then set the hub center pixel to the color
      with the most branch pixels adjacent to the hub.

    Key insight:
    - The decisive color is the less frequent color inside each seed, and it propagates in two places at once:
      outward onto the branch and inward onto the seed's own P border.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = sorted({int(x) for x in np.unique(g)} - {bg})
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
            deg = 0
            if r > 0 and bool(m[r - 1, c]):
                deg += 1
            if r + 1 < H and bool(m[r + 1, c]):
                deg += 1
            if c > 0 and bool(m[r, c - 1]):
                deg += 1
            if c + 1 < W and bool(m[r, c + 1]):
                deg += 1
            if deg == 2:
                deg2 += 1
            elif deg == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    P = int(max(colors, key=lambda c: comp_deg_score(g == int(c))))

    mP = g == int(P)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = None
    for k in range(2, int(min(H, W)) + 1):
        found = None
        for r0 in range(0, H - k + 1):
            r1 = r0 + k - 1
            for c0 in range(0, W - k + 1):
                c1 = c0 + k - 1
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found
    assert best is not None

    k, r0, c0 = map(int, best)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r1 + 1, c0 : c1 + 1] = True

    seed_mask = (g != bg) & (g != P)
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0, "grid must contain seed regions"

    seed_minor: dict[int, int] = {}
    seed_border_cells: dict[int, set[tuple[int, int]]] = {}
    border_to_seed: dict[tuple[int, int], int] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())

        if rn > 0:
            assert all(int(g[rn - 1, c]) == P for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == P for c in range(cn, cx + 1)), "seed missing bot P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == P for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == P for r in range(rn, rx + 1)), "seed missing right P border"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        assert len(counts) >= 2, "seed must contain both majority and minority colors"
        minor = int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0])
        seed_minor[int(sid)] = int(minor)

        border_cells: set[tuple[int, int]] = set()
        for rr in range(max(0, rn - 1), min(H - 1, rx + 1) + 1):
            for cc in range(max(0, cn - 1), min(W - 1, cx + 1) + 1):
                inside = rn <= rr <= rx and cn <= cc <= cx
                if inside:
                    continue
                if int(g[rr, cc]) == P:
                    border_cells.add((int(rr), int(cc)))
        assert border_cells, "seed border must be present"
        seed_border_cells[int(sid)] = border_cells
        for cell in border_cells:
            prev = border_to_seed.get(cell)
            assert prev is None or prev == int(sid), "seed borders must not overlap"
            border_to_seed[cell] = int(sid)

    border_mask = np.zeros((H, W), dtype=bool)
    for r, c in border_to_seed:
        border_mask[int(r), int(c)] = True

    out = g.copy()

    branch_mask = (g == P) & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == P
    ppad = np.pad(pmask, 1)
    p_deg = (
        ppad[2:, 1:-1].astype(int)
        + ppad[:-2, 1:-1]
        + ppad[1:-1, 2:]
        + ppad[1:-1, :-2]
    )
    assert not np.any((p_deg > 2) & pmask & ~hub & ~border_mask), "path has T-junction"

    branch_color_by_id: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = border_to_seed.get((rr, cc))
                    if sid is not None:
                        touches[int(sid)] += 1
        assert touches, "each branch must touch a seed border"
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_minor[int(sid)])
        branch_color_by_id[int(bid)] = int(col)
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    for sid, cells in seed_border_cells.items():
        col = int(seed_minor[int(sid)])
        for r, c in cells:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    bid = int(branch_lab[rr, cc])
                    if bid != 0:
                        touch_cols[int(branch_color_by_id[int(bid)])] += 1

    assert touch_cols, "hub must be touched by at least one recolored branch"
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
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
MINORITY_FRAC_LO = 0.15
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

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _try_carve_path(canvas, P, start, dest_border_cells, max_bends=MAX_BENDS):
    """Carve one thin orthogonal path from a hub perimeter cell to a seed border."""
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
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(
                    nr, nc, cur, allow_dest=True
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
    min_minor = max(1, int(np.ceil(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(np.floor(n_int * MINORITY_FRAC_HI)))
    max_minor = min(max_minor, (n_int - 1) // 2)
    n_minor = random.randint(min_minor, max_minor)
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

    def _overlaps_hub(zone):
        zr0, zc0, zr1, zc1 = zone
        return not (zr1 < hr0 - 1 or zr0 > hr0 + HUB_SIZE or zc1 < hc0 - 1 or zc0 > hc0 + HUB_SIZE)

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

        majority = random.choice([A, B])
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

    if len(seeds) < 3:
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

    # Reject degenerate cases where the fusion output matches majority propagation everywhere.
    majority_out = _original_majority_transform(inp)
    if int(np.count_nonzero(out != majority_out)) <= 0:
        return None

    return inp


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")


def _original_majority_transform(grid: np.ndarray) -> np.ndarray:
    """Reference transform used only to reject degenerate generator samples."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = sorted({int(x) for x in np.unique(g)} - {bg})
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
            deg = 0
            if r > 0 and bool(m[int(r) - 1, int(c)]):
                deg += 1
            if int(r) + 1 < H and bool(m[int(r) + 1, int(c)]):
                deg += 1
            if c > 0 and bool(m[int(r), int(c) - 1]):
                deg += 1
            if int(c) + 1 < W and bool(m[int(r), int(c) + 1]):
                deg += 1
            if deg == 2:
                deg2 += 1
            elif deg == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    P = int(max(colors, key=lambda c: comp_deg_score(g == int(c))))

    mP = g == int(P)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = None
    for kk in range(2, int(min(H, W)) + 1):
        found = None
        for rr0 in range(0, H - kk + 1):
            rr1 = rr0 + kk - 1
            for cc0 in range(0, W - kk + 1):
                cc1 = cc0 + kk - 1
                if rect_sum(int(rr0), int(cc0), int(rr1), int(cc1)) == int(kk * kk):
                    found = (int(kk), int(rr0), int(cc0))
        if found is not None:
            best = found
    assert best is not None

    kk, rr0, cc0 = map(int, best)
    rr1, cc1 = int(rr0 + kk - 1), int(cc0 + kk - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    seed_mask = (g != bg) & (g != P)
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        seed_major[int(sid)] = int(Counter(vals).most_common(1)[0][0])

    out = g.copy()
    branch_mask = (g == P) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_major[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols
    out[int(rr0 + kk // 2), int(cc0 + kk // 2)] = int(touch_cols.most_common(1)[0][0])
    return out

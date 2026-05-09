# Combo variant of 800d221b (index 17)
# combo_spec: D3/1+D13/1
# Name: 双向传播+边框少数色
#
# Design analysis
# a) Concrete behavior trace:
#    - In the base task, each non-hub P connected component mixes a seed border and one or more attached branches, and
#      the whole component is recolored to that seed's interior majority color.
#    - In this combo, the same seed now drives two outward-facing colors at once: every branch leading away from the
#      hub takes the seed's majority color, while the seed's own P frame switches to the seed's minority color. The
#      mixed interior itself does not change.
#    - Example: if a seed interior is 7 red + 3 blue, then the old output made both its branch and frame red. The new
#      output keeps the branch red but turns that seed's frame blue. The visible difference is local and repeated on
#      every seed, so the solver must separate border cells from branch cells instead of treating all non-hub P cells
#      as one object.
#
# b) Complexity calibration:
#    - The scene family stays close to the original: one solid 3x3 hub, 3-5 edge-attached rectangular seeds, a single
#      input path color P, and thin orthogonal spokes.
#    - To keep the fusion observable, the generator still uses only two seed-interior colors per grid and forces both
#      majority orientations to appear among the seeds. This gives repeated evidence for "look inside each seed,
#      identify majority/minority locally, then split branch vs frame accordingly".
#    - Minority occupancy is kept visibly non-trivial, so the frame recolor is explainable from the input rather than
#      looking like an arbitrary palette swap.
#
# c) Hidden-assumption audit:
#    - Inferring P as the most network-like non-background color still holds because the input object family is
#      unchanged.
#    - Finding the hub as the largest solid P square still holds because the hub remains a clean 3x3 block.
#    - The original solver's biggest broken assumption is structural: labeling all non-hub P cells as one component is
#      no longer semantically correct, because a seed frame and its branches now need different colors. The revised
#      solver therefore derives each seed's exact P border from the rectangular seed bbox, colors those cells by the
#      minority color, and labels only the remaining P cells as branches.
#    - The new solver assumes each seed is still a rectangle wrapped by a 1-pixel P frame on its non-edge sides. The
#      generator preserves that exactly and keeps seed zones separated so border ownership is unambiguous.
#
# d) Placement legality:
#    - Seed interiors are placed in non-overlapping edge zones with at least one-cell separation between border zones.
#    - The P frame is added only on non-edge sides, matching the solver's border reconstruction rule.
#    - Paths are carved through free background cells only, never through other paths or seed cells, so branches stay
#      simple and do not merge unexpectedly.
#
# e) Visual observability:
#    - Every seed shows the same local motif in the output: interior stays mixed, frame shows the rarer color, and the
#      attached spoke shows the commoner color. Because multiple seeds of both majority orientations appear in one
#      grid, a human can infer the two-way propagation from a few examples.
#
# f) Visual cleanliness:
#    - The only extra visual distinction is branch-vs-frame color contrast inside the existing P network. No new
#      glyphs, markers, or overlapping annotations are introduced, so the grids remain easy to parse.
#
# g) Inferability of the color change:
#    - No step depends on numeric color ids. Majority/minority are determined by visible counts inside each seed, and
#      the generator rejects degenerate samples where both majority colors do not appear or where the combo output
#      collapses to the old "frame equals branch" behavior on the same input.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ── Tunable generation parameters ──────────────────────────────────────────
GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.18
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_FIRST_ROUND_RETRIES = 60
PATH_EXTRA_ROUND_RETRIES = 30
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 300


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain a non-background color"

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


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, int, int, int]:
    H, W = g.shape
    mP = g == int(path_color)
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

    assert best is not None, "failed to find hub square"
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, k, r0, c0


def _unique_extreme_color(counts: Counter[int], want_max: bool) -> int:
    assert counts, "expected non-empty color counts"
    extreme = max(counts.values()) if want_max else min(counts.values())
    winners = [int(col) for col, cnt in counts.items() if int(cnt) == int(extreme)]
    assert len(winners) == 1, "expected a unique majority/minority color"
    return int(winners[0])


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if r0 > 0:
        for c in range(c0, c1 + 1):
            cells.add((int(r0 - 1), int(c)))
    if r1 < H - 1:
        for c in range(c0, c1 + 1):
            cells.add((int(r1 + 1), int(c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c0 - 1)))
    if c1 < W - 1:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c1 + 1)))
    if r0 > 0 and c0 > 0:
        cells.add((int(r0 - 1), int(c0 - 1)))
    if r0 > 0 and c1 < W - 1:
        cells.add((int(r0 - 1), int(c1 + 1)))
    if r1 < H - 1 and c0 > 0:
        cells.add((int(r1 + 1), int(c0 - 1)))
    if r1 < H - 1 and c1 < W - 1:
        cells.add((int(r1 + 1), int(c1 + 1)))
    return sorted(cells)


def _analyze_grid(g: np.ndarray) -> dict:
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub, hub_size, hub_r0, hub_c0 = _find_hub(g, path_color)
    H, W = g.shape

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)

    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    border_owner = np.zeros((H, W), dtype=int)
    border_mask = np.zeros((H, W), dtype=bool)

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0, "seed component must be non-empty"
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())

        if r0 > 0:
            assert all(int(g[r0 - 1, c]) == int(path_color) for c in range(c0, c1 + 1)), "seed missing top P border"
        if r1 < H - 1:
            assert all(int(g[r1 + 1, c]) == int(path_color) for c in range(c0, c1 + 1)), "seed missing bottom P border"
        if c0 > 0:
            assert all(int(g[r, c0 - 1]) == int(path_color) for r in range(r0, r1 + 1)), "seed missing left P border"
        if c1 < W - 1:
            assert all(int(g[r, c1 + 1]) == int(path_color) for r in range(r0, r1 + 1)), "seed missing right P border"

        counts = Counter(int(g[int(r), int(c)]) for r, c in pts)
        seed_major[int(sid)] = _unique_extreme_color(counts, want_max=True)
        seed_minor[int(sid)] = _unique_extreme_color(counts, want_max=False)

        cells = _seed_border_cells(r0, c0, r1, c1, H, W)
        assert cells, "every seed must expose at least one P border cell"
        for br, bc in cells:
            assert int(g[br, bc]) == int(path_color), "reconstructed border must be color P"
            assert int(border_owner[br, bc]) in (0, int(sid)), "seed borders must not overlap"
            border_owner[br, bc] = int(sid)
            border_mask[br, bc] = True

    branch_mask = (g == int(path_color)) & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    bpad = np.pad(branch_mask, 1)
    bdeg = bpad[2:, 1:-1].astype(int) + bpad[:-2, 1:-1] + bpad[1:-1, 2:] + bpad[1:-1, :-2]
    assert not np.any((bdeg > 2) & branch_mask), "branch-only network must stay degree <= 2"

    branch_to_seed: dict[int, int] = {}
    hub_touch_counter: Counter[int] = Counter()
    seed_branch_count: Counter[int] = Counter()

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        assert pts.size > 0, "branch component must be non-empty"

        touch_seed = Counter()
        hub_contacts = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            touches_hub_here = False
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(border_owner[rr, cc])
                    if sid != 0:
                        touch_seed[int(sid)] += 1
                    if bool(hub[rr, cc]):
                        touches_hub_here = True
            if touches_hub_here:
                hub_contacts += 1

        assert touch_seed, "every branch must touch at least one seed border"
        assert len(touch_seed) == 1, "branch must belong to exactly one seed"
        sid = int(next(iter(touch_seed)))
        branch_to_seed[int(bid)] = int(sid)
        seed_branch_count[int(sid)] += 1
        assert hub_contacts >= 1, "every branch must touch the hub"
        hub_touch_counter[int(seed_major[int(sid)])] += int(hub_contacts)

    for sid in range(1, int(seed_n) + 1):
        assert int(seed_branch_count[int(sid)]) >= 1, "every seed must own at least one branch"

    return {
        "bg": int(bg),
        "path_color": int(path_color),
        "hub": hub,
        "hub_size": int(hub_size),
        "hub_r0": int(hub_r0),
        "hub_c0": int(hub_c0),
        "seed_lab": seed_lab,
        "seed_n": int(seed_n),
        "seed_major": seed_major,
        "seed_minor": seed_minor,
        "border_owner": border_owner,
        "border_mask": border_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_to_seed": branch_to_seed,
        "hub_touch_counter": hub_touch_counter,
    }


def _render_output(g: np.ndarray, info: dict, minority_border: bool) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)

    border_owner = info["border_owner"]
    for sid in range(1, int(info["seed_n"]) + 1):
        border_color = info["seed_minor"][int(sid)] if minority_border else info["seed_major"][int(sid)]
        out[border_owner == int(sid)] = int(border_color)

    branch_lab = info["branch_lab"]
    for bid in range(1, int(info["branch_n"]) + 1):
        sid = int(info["branch_to_seed"][int(bid)])
        out[branch_lab == int(bid)] = int(info["seed_major"][int(sid)])

    touch_cols = info["hub_touch_counter"]
    assert touch_cols, "hub must be touched by at least one recolored branch"
    center_col = int(touch_cols.most_common(1)[0][0])
    cr = int(info["hub_r0"] + info["hub_size"] // 2)
    cc = int(info["hub_c0"] + info["hub_size"] // 2)
    out[cr, cc] = int(center_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the path network into majority-colored spokes and minority-colored seed frames.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Mixed rectangular seed interiors (non-bg, non-P) touch the grid edge and are wrapped by a 1-pixel P frame on
      the sides that do not lie on the outer border.

    Rule:
    - Infer P from the most network-like non-background component and locate the largest solid all-P square as the hub.
    - For each seed, read the majority and minority colors from its interior only, then reconstruct that seed's exact
      P border from the seed bbox.
    - Recolor every non-hub, non-border P branch by the touching seed's majority color, recolor that seed's P border
      by the seed's minority color, and leave the seed interior unchanged.
    - Recolor the hub center pixel to the color with the most hub-adjacent recolored branch pixels.

    Key insight:
    - The old "one P component, one output color" view no longer works. The decisive step is to separate each seed's
      P frame from the rest of the network: the frame inherits the rare interior color, while the outward branch still
      inherits the common interior color.
    """
    g = np.array(grid, dtype=int)
    info = _analyze_grid(g)
    return _render_output(g, info, minority_border=True)


def _hub_vote_is_unique(counter: Counter[int]) -> bool:
    if not counter:
        return False
    top = max(counter.values())
    return sum(1 for v in counter.values() if int(v) == int(top)) == 1


def _try_carve_path(canvas: np.ndarray, path_color: int, start: tuple[int, int], dest_border_cells, max_bends: int = MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != 0:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = int(canvas[nr, nc])
            if v == int(path_color):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in _DIRS:
        nr = int(sr + dr)
        nc = int(sc + dc)
        if 0 <= nr < H and 0 <= nc < W and int(canvas[nr, nc]) == 0:
            init_dirs.append((dr, dc))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    n_bends = random.randint(0, max_bends)

    def _attempt(initial_dir: tuple[int, int]):
        path = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr = int(cur[0] + cur_dir[0])
                nc = int(cur[1] + cur_dir[1])
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, allow_dest=True):
                    canvas[nr, nc] = int(path_color)
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(path_color)
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


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work: np.ndarray, path_color: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int):
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
                work[r, c] = int(path_color)
                border_cells.append((r, c))
                landing_cells.append((r, c))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = int(path_color)
            border_cells.append(rc)
    return border_cells, landing_cells


def _zones_overlap(zone: tuple[int, int, int, int], others, gap: int) -> bool:
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


def _fill_seed_interior(work: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int):
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(np.ceil(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(np.floor(n_int * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = int(minority)
    for r, c in interior_cells[n_minor:]:
        work[r, c] = int(majority)


def _carve_all_paths(work: np.ndarray, path_color: int, hub_perimeter, seeds) -> bool:
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
            path = _try_carve_path(work, path_color, start, seeds[si]["landing_cells"])
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
            path = _try_carve_path(work, path_color, start, seeds[si]["landing_cells"])
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
    bg, path_color, color_a, color_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(path_color)

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == bg, 0, g).astype(int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    n_a_major = random.randint(1, n_seeds_target - 1)
    planned_majorities = [color_a] * n_a_major + [color_b] * (n_seeds_target - n_a_major)
    random.shuffle(planned_majorities)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    attempts = 0
    while len(seeds) < n_seeds_target and attempts < SEED_PLACEMENT_RETRIES:
        attempts += 1
        edge = random.choice(edges)
        ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W)
        border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)

        if _overlaps_hub(border_zone):
            continue
        if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
            continue

        majority = int(planned_majorities[len(seeds)])
        minority = int(color_b if majority == color_a else color_a)
        _fill_seed_interior(work, ir0, ic0, ir1, ic1, majority, minority)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W)

        used_zones.append(border_zone)
        seeds.append(
            {
                "majority": majority,
                "minority": minority,
                "border_cells": border_cells,
                "landing_cells": landing_cells,
            }
        )

    if len(seeds) < 3:
        return None

    if not _carve_all_paths(work, path_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    if int(Counter(int(x) for x in inp.flatten()).most_common(1)[0][0]) != int(bg):
        return None

    try:
        info = _analyze_grid(inp)
        combo_out = _render_output(inp, info, minority_border=True)
    except Exception:
        return None

    if len(set(info["seed_major"].values())) < 2:
        return None
    if len(set(info["seed_minor"].values())) < 2:
        return None
    if not _hub_vote_is_unique(info["hub_touch_counter"]):
        return None

    base_out = _render_output(inp, info, minority_border=False)
    border_delta = int(np.sum(combo_out[info["border_mask"]] != base_out[info["border_mask"]]))
    if border_delta != int(np.sum(info["border_mask"])):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            result = _gen_attempt()
        except Exception:
            result = None
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

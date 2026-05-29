# Combo variant of 800d221b (index 30)
# combo_spec: D1/0+D2/3+D18/0
# Name: 少数色+种子数投票+单路径
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original: one seed could feed 1-3 paths, each branch used the seed-majority color, and the hub center counted
#      hub-adjacent recolored pixels. A seed with more paths could dominate the center even if seed counts were equal.
#    - Fused variant: every seed owns exactly one P-component outside the hub (its border + one path), that component is
#      recolored to the seed's minority color, and the hub center counts connected seeds rather than adjacent pixels.
#      With only one path per seed, each branch becomes a one-vote witness for exactly one seed.
#    - Example: with three seeds whose interiors are majority colors [A, A, B], the branch colors become [B, B, A].
#      The hub center must become B because two seeds have minority B, even though the original task would have produced
#      majority-colored branches [A, A, B] and a center vote for A.
# b) Complexity calibration:
#    - The family stays readable: same hub, same edge-mounted seed rectangles, same thin orthogonal paths.
#    - To make the new rule visible, the generator forces an odd seed count and uses both seed-majority assignments, so
#      branch colors split 2:1 or 3:2 instead of collapsing to a single color.
# c) Hidden-assumption audit for main():
#    - Path-color inference still assumes the largest network-like non-background component is P; the generator preserves
#      a unique 3x3 hub plus thin chains, so this remains stable.
#    - Hub detection still assumes the unique largest solid all-P square is the hub; seed frames and paths stay 1-pixel
#      thick, so no competing square is created.
#    - Seed parsing assumes each mixed-color rectangle is surrounded by P on non-edge sides; placement keeps the original
#      border geometry intact.
#    - Branch labeling now assumes one non-hub P-component per seed. This is new and is guaranteed by generating exactly
#      one path for each seed and keeping seed border zones disjoint.
#    - Minority-color inference assumes each seed interior contains exactly two colors with unequal counts; the generator
#      enforces that directly.
# d) Placement legality:
#    - Seeds are sampled in edge zones with bbox-gap checks against each other and against the central hub.
#    - Paths are carved on free background only, reject T-junctions, and stop when adjacent to exactly one seed border.
# e) Visual observability:
#    - The key cue is that branch colors systematically disagree with the larger color inside the adjacent seed.
#    - Because the seed count is odd and both minority colors appear, the hub center visibly follows the branch-color
#      plurality by number of seeds, not hidden numeric color ids.
# f) Visual cleanliness:
#    - Only three object types remain visible: hub/path color P, two seed colors inside rectangles, and the recolored
#      output branches. No extra markers or overlays are introduced.
# g) Inferability of the change:
#    - The recolor palette stays at two seed colors, so the human only has to infer "smaller color inside seed" and
#      "one branch equals one seed vote". The generator avoids ties so the center decision is readable from 2-4 examples.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.18
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 500
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_DIR_RETRIES = 4
PATH_RETRIES_PER_SEED = 80

GEN_OUTER_RETRIES = 400

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

    assert best is not None
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r0 + k), int(c0) : int(c0 + k)] = True
    return hub, int(k), int(r0), int(c0)


def _seed_color_roles(vals: list[int]) -> tuple[int, int]:
    counts = Counter(vals)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert len(ordered) == 2, "each seed must use exactly two colors"
    assert ordered[0][1] > ordered[1][1], "seed majority/minority must be strict"
    majority = int(ordered[0][0])
    minority = int(ordered[1][0])
    return majority, minority


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-seed path network using per-seed minority colors.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid 3x3-style hub.
    - Mixed-color seed rectangles sit on grid edges; each seed interior uses exactly two colors and has a P frame on
      every non-edge side.
    - Outside the hub, each seed owns exactly one connected P branch: its frame plus one path to the hub.

    Rule:
    - Infer P from the most network-like non-background color, then find the largest solid all-P square as the hub.
    - For each seed region, identify its minority interior color.
    - Recolor the entire corresponding non-hub P branch to that minority color.
    - Recolor only the hub center pixel to the color winning a one-vote-per-seed tally over connected seed regions.

    Key insight:
    Because every seed contributes exactly one branch, each recolored branch is a direct proxy for one seed's minority
    color. The center is therefore decided by counting connected seeds, not by measuring path length or border pixels.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    path_color = _infer_path_color(g, bg)
    hub, hub_size, hr0, hc0 = _find_hub(g, path_color)

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) in SEED_COUNT_CHOICES

    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(path_color) for c in range(cn, cx + 1))
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(path_color) for c in range(cn, cx + 1))
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(path_color) for r in range(rn, rx + 1))
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(path_color) for r in range(rn, rx + 1))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        majority, minority = _seed_color_roles(vals)
        seed_major[int(sid)] = int(majority)
        seed_minor[int(sid)] = int(minority)

    out = g.copy()

    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    assert int(branch_n) == int(seed_n), "single-path variant requires one non-hub branch per seed"

    pmask = g == int(path_color)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    seed_vote = Counter()
    seen_seed_ids = set()
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touch_seed_ids = set()
        touches_hub = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touch_seed_ids.add(int(sid))
                    if bool(hub[rr, cc]):
                        touches_hub = True

        assert len(touch_seed_ids) == 1, "each branch must correspond to exactly one seed"
        assert touches_hub, "each branch must connect a seed to the hub"
        sid = int(next(iter(touch_seed_ids)))
        seen_seed_ids.add(int(sid))
        recolor = int(seed_minor[int(sid)])
        seed_vote[recolor] += 1
        for r, c in pts:
            out[int(r), int(c)] = int(recolor)

    assert seen_seed_ids == set(range(1, int(seed_n) + 1))
    vote_order = seed_vote.most_common()
    assert vote_order and (len(vote_order) == 1 or vote_order[0][1] > vote_order[1][1]), "hub vote must be strict"

    center_r = int(hr0 + hub_size // 2)
    center_c = int(hc0 + hub_size // 2)
    out[int(center_r), int(center_c)] = int(vote_order[0][0])
    return out


def _try_carve_path(canvas: np.ndarray, path_color: int, start: tuple[int, int],
                    dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
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
        nr, nc = int(sr + dr), int(sc + dc)
        if 0 <= nr < H and 0 <= nc < W and int(canvas[nr, nc]) == 0:
            init_dirs.append((int(dr), int(dc)))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    n_bends = random.randint(0, max_bends)

    def _attempt(initial_dir: tuple[int, int]) -> list[tuple[int, int]] | None:
        path = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
                if (any((int(nr + dr), int(nc + dc)) in dest_set for dr, dc in _DIRS)
                        and _check_neighbors(nr, nc, cur, allow_dest=True)):
                    canvas[nr, nc] = int(path_color)
                    path.append((int(nr), int(nc)))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(path_color)
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
            cur_dir = perps[0]
        for r, c in path:
            canvas[int(r), int(c)] = 0
        return None

    for d in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = _attempt(d)
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


def _place_seed_border(work: np.ndarray, path_color: int, ir0: int, ic0: int, ir1: int, ic1: int,
                       H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
                work[int(r), int(c)] = int(path_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(path_color)
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
        ir0, ic0 = 0, random.randint(0, W - iw)
    elif edge == "bottom":
        ir0, ic0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, H - ih), 0
    else:
        ir0, ic0 = random.randint(0, H - ih), W - iw
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int,
                        majority: int, minority: int, sid: int) -> None:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(np.ceil(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(np.floor(n_int * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    assert n_minor < n_int - n_minor
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[int(r), int(c)] = int(minority)
        seed_id_grid[int(r), int(c)] = int(sid)
    for r, c in interior_cells[n_minor:]:
        work[int(r), int(c)] = int(majority)
        seed_id_grid[int(r), int(c)] = int(sid)


def _carve_all_paths(work: np.ndarray, path_color: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict],
                     hub_center: tuple[int, int]) -> bool:
    used_hub_starts = set()
    order = list(range(len(seeds)))
    order.sort(
        key=lambda si: (
            -abs(((seeds[si]["bbox"][0] + seeds[si]["bbox"][2]) // 2) - hub_center[0])
            - abs(((seeds[si]["bbox"][1] + seeds[si]["bbox"][3]) // 2) - hub_center[1]),
            len(seeds[si]["landing_cells"]),
        )
    )
    for si in order:
        seed = seeds[si]
        target_r = int((seed["bbox"][0] + seed["bbox"][2]) // 2)
        target_c = int((seed["bbox"][1] + seed["bbox"][3]) // 2)
        ranked_starts = sorted(
            [s for s in hub_perimeter if s not in used_hub_starts],
            key=lambda rc: abs(int(rc[0]) - target_r) + abs(int(rc[1]) - target_c),
        )
        if not ranked_starts:
            return False
        ok = False
        for _ in range(PATH_RETRIES_PER_SEED):
            top_k = min(len(ranked_starts), 4)
            start = random.choice(ranked_starts[:top_k])
            path = _try_carve_path(work, path_color, start, seed["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                ok = True
                break
        if not ok:
            for start in ranked_starts[4:]:
                path = _try_carve_path(work, path_color, start, seed["landing_cells"])
                if path is not None:
                    used_hub_starts.add(start)
                    ok = True
                    break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, path_color, seed_a, seed_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)
    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0:hr0 + HUB_SIZE, hc0:hc0 + HUB_SIZE] = int(path_color)

    hub_perimeter = [
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]
    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    majority_a_count = random.choice([n_seeds_target // 2, n_seeds_target // 2 + 1])
    planned_majorities = [seed_a] * majority_a_count + [seed_b] * (n_seeds_target - majority_a_count)
    random.shuffle(planned_majorities)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

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

        majority = int(planned_majorities[len(seeds)])
        minority = int(seed_b if majority == seed_a else seed_a)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W)
        if not landing_cells:
            for r in range(ir0, ir1 + 1):
                for c in range(ic0, ic1 + 1):
                    work[int(r), int(c)] = 0
                    seed_id_grid[int(r), int(c)] = 0
            for r, c in border_cells:
                work[int(r), int(c)] = 0
            continue

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

    if not _carve_all_paths(work, path_color, hub_perimeter, seeds, (hr0 + 1, hc0 + 1)):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    center_r, center_c = hr0 + 1, hc0 + 1
    center_color = int(out[center_r, center_c])
    minority_counts = Counter(int(seed["minority"]) for seed in seeds)
    majority_counts = Counter(int(seed["majority"]) for seed in seeds)
    if len(minority_counts) < 2:
        return None
    if minority_counts.most_common(1)[0][1] == minority_counts.most_common()[-1][1]:
        return None
    if center_color != int(minority_counts.most_common(1)[0][0]):
        return None
    if center_color == int(majority_counts.most_common(1)[0][0]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            res = _gen_attempt()
        except Exception:
            res = None
        if res is not None:
            return res
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

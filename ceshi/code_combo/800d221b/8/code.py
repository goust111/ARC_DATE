# Combo variant of 800d221b (index 8)
# combo_spec: D1/0+D13/1
# Name: 少数色分支+少数色边框
#
# Design analysis
# a) Concrete behavior trace:
#    - Original task: if one seed interior contains 8 red cells and 3 blue cells, then the branch touching that
#      seed and the seed's P border are both recolored red, while the mixed seed interior stays unchanged.
#    - This combo flips both outward-facing parts at once: the same branch and the same P border are recolored blue,
#      i.e. the seed's minority color. The interior still keeps its red/blue mix. The visible effect is that the
#      rarer color leaks out from the seed and wraps the seed border and its branch together.
#    - Because branch and border now agree on the same minority color, it is correct to keep treating every non-hub
#      P connected component as one branch-plus-border object, just like the original solver did structurally.
#
# b) Complexity calibration:
#    - The object family is unchanged: one hub, 3-5 rectangular mixed seeds, thin orthogonal paths, and a single path
#      color P in the input. This preserves recognizability and avoids introducing new object classes.
#    - To make the fusion observable, the generator keeps only two seed-interior colors per grid and forces both
#      majority orientations to appear across the seeds. A seed may be A-major/B-minor or B-major/A-minor, so a
#      solver must infer minority from local seed composition rather than memorize one global output color.
#    - Minority occupancy is biased upward relative to the original generator so the minority color is visible inside
#      each seed. That keeps the "minority overflow" cue legible in 2-4 examples without adding extra mechanics.
#
# c) Hidden-assumption audit against the original solver:
#    - Inferring P from the most network-like non-background color still holds because the input format is unchanged.
#    - Hub detection by largest solid P square still holds because the hub remains a solid 3x3 P block in input.
#    - Seed detection as connected non-bg/non-P regions still holds because only the output rule changes.
#    - The original branch labeling used all non-hub P cells, which merges a seed border with every path reaching it.
#      That remains valid here and is now even more semantically aligned, since branch and border must share one
#      minority color in the output.
#    - Hub-center voting still reads colors from hub-adjacent recolored path pixels. Since every such pixel still gets
#      a seed-derived color, the vote logic remains valid.
#
# d) Placement legality:
#    - The generator keeps the original output-first construction: place hub, place non-overlapping seed zones that
#      touch grid edges, add P borders, then carve simple paths with degree <= 2 from distinct hub-boundary starts.
#    - Seed zones are separated by at least one cell, hub overlap is rejected, and path carving refuses collisions
#      with existing paths or seed cells, so branch/border objects do not merge incorrectly.
#
# e) Visual observability:
#    - In output, each seed interior remains mixed while its border and attached branch become the minority color.
#      This creates a repeated local motif: "look inside the seed, find the rarer color, and that rarer color appears
#      outside on both the frame and the spoke." That is the intended clue for human solvers.
#
# f) Visual cleanliness:
#    - No extra marks or legends are added. Regions stay separated, paths are 1-pixel wide, and only the hub center
#      changes inside the hub, so the grid remains easy to parse.
#
# g) Inferability of the color change:
#    - The rule does not depend on numeric color ids. It depends only on visible frequency inside each seed region.
#      With only two interior colors and repeated seed examples per grid, "minority drives both branch and border"
#      is structurally supported and does not collapse into arbitrary color remapping.

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


# ── Tunable generation parameters ──────────────────────────────────────────
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
MINORITY_FRAC_LO = 0.20
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
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, k, r0, c0


def _seed_stats(
    g: np.ndarray,
    bg: int,
    path_color: int,
) -> tuple[np.ndarray, int, dict[int, int], dict[int, int], list[int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    all_seed_vals: list[int] = []

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(path_color) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(path_color) for c in range(cn, cx + 1)), "seed missing bot P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(path_color) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(path_color) for r in range(rn, rx + 1)), "seed missing right P border"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        assert len(ranked) >= 2, "seed interior must contain at least two colors"
        assert ranked[0][1] > ranked[-1][1], "seed must have a strict minority color"
        seed_major[int(sid)] = int(ranked[0][0])
        seed_minor[int(sid)] = int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0])
        all_seed_vals.extend(vals)

    return seed_lab, int(seed_n), seed_major, seed_minor, all_seed_vals


def _assert_no_t_junction(g: np.ndarray, path_color: int, hub: np.ndarray, seed_lab: np.ndarray) -> None:
    pmask = g == int(path_color)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-spoke path network so the rarer seed color spills outward.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Several edge-touching seed rectangles contain a two-color interior mix and are wrapped by a P border.

    Rule:
    - Infer P from the most network-like non-background component and locate the largest solid all-P square hub.
    - For each seed interior, determine both its majority color and its minority color from the interior cells only.
    - For every non-hub P component, identify which seed it touches most and recolor that whole branch-plus-border
      component to that seed's minority color.
    - Recolor the hub center pixel to the color that appears most often on recolored branch pixels adjacent to the hub.

    Key insight:
    - The seed interiors are not simplified; only the outward P scaffold changes. The correct output comes from reading
      the rarer interior color of each seed and letting that minority color flow onto both the seed frame and its spoke.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    path_color = _infer_path_color(g, bg)
    hub, k, r0, c0 = _find_hub(g, path_color)
    seed_lab, seed_n, _seed_major, seed_minor, all_seed_vals = _seed_stats(g, bg, path_color)

    out = g.copy()
    network_mask = (g == int(path_color)) & (~hub)
    network_lab, network_n = label_mask(network_mask, connectivity=4)
    _assert_no_t_junction(g, path_color, hub, seed_lab)

    for bid in range(1, int(network_n) + 1):
        pts = np.argwhere(network_lab == int(bid))
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
        col = int(seed_minor[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(network_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    assert touch_cols, "hub must be touched by at least one recolored branch"
    assert seed_n > 0 and all_seed_vals, "grid must contain seed regions"
    out[cr, cc] = int(touch_cols.most_common(1)[0][0])
    return out


def _try_carve_path(canvas: np.ndarray, path_color: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS):
    """Carve a single 1-pixel orthogonal path from a hub boundary cell toward a seed border."""
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
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
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
    min_minor = max(1, math.ceil(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, math.floor(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = int(minority)
        seed_id_grid[r, c] = int(sid)
    for r, c in interior_cells[n_minor:]:
        work[r, c] = int(majority)
        seed_id_grid[r, c] = int(sid)


def _carve_all_paths(work: np.ndarray, path_color: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
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
    bg, path_color, seed_a, seed_b = random.sample(list(range(10)), 4)

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

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    majority_plan = [seed_a, seed_b] + [random.choice([seed_a, seed_b]) for _ in range(max(0, n_seeds_target - 2))]
    random.shuffle(majority_plan)

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

        majority = int(majority_plan[len(seeds)])
        minority = int(seed_b if majority == seed_a else seed_a)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W)

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
    if not ({seed_a, seed_b} <= {seed["majority"] for seed in seeds}):
        return None
    if not _carve_all_paths(work, path_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    diff = int(np.count_nonzero(out != inp))
    if diff <= HUB_SIZE * HUB_SIZE:
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

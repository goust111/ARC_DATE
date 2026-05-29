# Combo variant of 800d221b (index 1)
# combo_spec: D1/0+D2/0
# Name: 双重反转着色
# Design analysis:
# - Concrete trace:
#   In the base task, if three seeds have interior majorities [A, A, B], then their attached
#   path components recolor to [A, A, B], and the hub center also becomes A because A is the
#   most frequent hub-adjacent branch color. In this combo, the same seeds recolor to their
#   minority colors [B, B, A] instead, and the hub center becomes A because A is now the least
#   frequent hub-adjacent recolored branch color. Both decisions invert the original reasoning.
# - Complexity calibration:
#   The structural family stays close to the source task: 3-5 edge seeds, one thin orthogonal
#   path-network, and a 3x3 hub. To keep the inversion visible rather than noisy, the generator
#   uses only the two seed colors inside the seed interiors and enforces an uneven split of seed
#   majorities so the hub sees at least two branch colors with different frequencies.
# - Hidden-assumption audit:
#   Path-color inference, hub detection, and seed-border validation remain unchanged. The new
#   minority rule adds a fresh requirement: each seed interior must contain exactly two colors
#   with a unique minority. The new hub rule adds another: the recolored branch colors adjacent
#   to the hub must have a unique minimum count. The generator guarantees both properties.
# - Placement legality:
#   Seed zones still avoid overlap and stay away from the hub. Paths are carved from distinct
#   hub-boundary cells into free space, never crossing seeds or earlier paths, so no out-of-
#   bounds placements or T-junction violations are introduced by the combo.
# - Visual observability:
#   Humans can see that branches do not follow the dominant interior color of each seed; they
#   follow the rarer interior color. The hub center then matches the rarer of the branch colors
#   touching the hub, making the second inversion visible from only a few examples.
# - Visual cleanliness:
#   The generator keeps one path connection per seed. That avoids border/path overmerging clutter
#   while preserving enough bent routes and enough repeated evidence for both seed colors.
# - Inferability:
#   No ordering depends on palette ids. The only comparisons are visible counts inside seed
#   interiors and visible counts of hub-adjacent recolored branches.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


# ── Solver helpers ──────────────────────────────────────────────────────────

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _unique_extreme_color(counts: Counter[int], *, pick_min: bool) -> int:
    assert counts, "expected at least one counted color"
    target = min(counts.values()) if pick_min else max(counts.values())
    winners = [int(col) for col, cnt in counts.items() if int(cnt) == int(target)]
    assert len(winners) == 1, "expected a unique color count extreme"
    return int(winners[0])


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"

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


def _find_hub_mask(g: np.ndarray, P: int) -> tuple[np.ndarray, int, int, int]:
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

    assert best is not None, "failed to locate hub square"
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0):int(r0 + k), int(c0):int(c0 + k)] = True
    return hub, k, r0, c0


def _extract_seed_roles(g: np.ndarray, bg: int, P: int) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0, "empty seed component"
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

        counts = Counter(int(g[int(r), int(c)]) for r, c in pts)
        assert len(counts) == 2, "combo generator expects exactly two colors inside each seed"
        major = _unique_extreme_color(counts, pick_min=False)
        minor = _unique_extreme_color(counts, pick_min=True)
        assert int(major) != int(minor), "seed majority and minority must differ"
        seed_major[int(sid)] = int(major)
        seed_minor[int(sid)] = int(minor)

    return seed_lab, seed_major, seed_minor


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = _infer_path_color(g, bg)
    hub, k, r0, c0 = _find_hub_mask(g, P)
    seed_lab, seed_major, seed_minor = _extract_seed_roles(g, bg, P)

    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    branch_seed: dict[int, int] = {}
    branch_major_color: dict[int, int] = {}
    branch_minor_color: dict[int, int] = {}
    branch_points: dict[int, np.ndarray] = {}

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
        assert touches, "every non-hub path component must touch a seed"
        sid = _unique_extreme_color(touches, pick_min=False)
        branch_seed[int(bid)] = int(sid)
        branch_major_color[int(bid)] = int(seed_major[int(sid)])
        branch_minor_color[int(bid)] = int(seed_minor[int(sid)])
        branch_points[int(bid)] = pts

    touch_major = Counter()
    touch_minor = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    bid = int(branch_lab[rr, cc])
                    if bid != 0 and bid in branch_seed:
                        touch_major[int(branch_major_color[int(bid)])] += 1
                        touch_minor[int(branch_minor_color[int(bid)])] += 1

    return {
        "g": g,
        "bg": bg,
        "P": P,
        "hub": hub,
        "hub_size": k,
        "hub_r0": r0,
        "hub_c0": c0,
        "center": (int(r0 + k // 2), int(c0 + k // 2)),
        "seed_lab": seed_lab,
        "seed_major": seed_major,
        "seed_minor": seed_minor,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_seed": branch_seed,
        "branch_major_color": branch_major_color,
        "branch_minor_color": branch_minor_color,
        "branch_points": branch_points,
        "touch_major": touch_major,
        "touch_minor": touch_minor,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network by reversing both color choices.

    Input structure:
    - The background is the most common color in the grid.
    - One non-background color forms a thin orthogonal network with a solid square hub.
    - Rectangular seed interiors sit on the outer edges; each interior mixes exactly two colors
      and is wrapped by the path color on every non-edge side.

    Rule:
    - Identify the path color as the most network-like non-background color, then locate the
      largest solid square of that color as the hub.
    - For each non-hub path component, find which seed interior it borders and recolor the full
      component to that seed interior's minority color instead of its majority color.
    - Leave the hub intact except for its center pixel, which becomes the least frequent color
      among the recolored branch pixels that touch the hub boundary.

    Key insight:
    - The same seed interiors drive both decisions, but each step is inverted: first choose the
      rarer interior color for the attached branch, then choose the rarer hub-adjacent branch
      color for the hub center.
    """
    info = _parse_grid(grid)
    out = info["g"].copy()

    for bid, pts in info["branch_points"].items():
        col = int(info["branch_minor_color"][int(bid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    assert info["touch_minor"], "hub must touch at least one recolored branch"
    center_col = _unique_extreme_color(info["touch_minor"], pick_min=True)
    cr, cc = info["center"]
    out[int(cr), int(cc)] = int(center_col)
    return out


# ── Generator ───────────────────────────────────────────────────────────────

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
PATH_RETRIES_PER_SEED = 60
PATH_DIR_RETRIES = 3

# Top-level generation
GEN_OUTER_RETRIES = 300


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS) -> list[tuple[int, int]] | None:
    """Carve one thin orthogonal path from a hub cell to a seed border."""
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
            if v == int(P):
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
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, allow_dest=True):
                    canvas[nr, nc] = int(P)
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(P)
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


def _place_seed_border(work: np.ndarray, P: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
                work[r, c] = int(P)
                border_cells.append((r, c))
                landing_cells.append((r, c))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = int(P)
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


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int, sid: int) -> None:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
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


def _build_majority_plan(n_seeds: int, color_a: int, color_b: int) -> list[int]:
    split_choices = [k for k in range(1, n_seeds) if k != n_seeds - k]
    count_a = random.choice(split_choices)
    plan = [int(color_a)] * int(count_a) + [int(color_b)] * int(n_seeds - count_a)
    random.shuffle(plan)
    return plan


def _carve_one_path_per_seed(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
    used_hub_starts = set()
    order = list(range(len(seeds)))
    random.shuffle(order)
    for si in order:
        ok = False
        for _ in range(PATH_RETRIES_PER_SEED):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, P, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                ok = True
                break
        if not ok:
            return False
    return True


def _combo_behavior_metrics(grid: np.ndarray) -> dict:
    info = _parse_grid(grid)
    active_branches = sum(
        1
        for bid in info["branch_seed"]
        if int(info["branch_major_color"][int(bid)]) != int(info["branch_minor_color"][int(bid)])
    )
    touch_minor = info["touch_minor"]
    return {
        "branch_components": len(info["branch_seed"]),
        "active_branches": int(active_branches),
        "hub_color_count_n": len(touch_minor),
        "hub_min_count": min(touch_minor.values()) if touch_minor else 0,
        "hub_max_count": max(touch_minor.values()) if touch_minor else 0,
        "d2_active": len(touch_minor) >= 2 and min(touch_minor.values()) < max(touch_minor.values()),
    }


def _gen_attempt() -> np.ndarray | None:
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), int(bg), dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0:hr0 + HUB_SIZE, hc0:hc0 + HUB_SIZE] = int(P)
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
    majority_plan = _build_majority_plan(n_seeds_target, A, B)

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

        majority = int(majority_plan[len(seeds)])
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

    if len(seeds) < 3:
        return None

    if not _carve_one_path_per_seed(work, P, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != int(bg):
        return None

    try:
        metrics = _combo_behavior_metrics(inp)
        if metrics["active_branches"] != metrics["branch_components"]:
            return None
        if not metrics["d2_active"]:
            return None
        main(inp)
    except Exception:
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

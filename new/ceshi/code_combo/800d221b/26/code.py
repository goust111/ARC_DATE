# Combo variant of 800d221b (index 26)
# combo_spec: D2/2+D3/1
# Name: hub整体着色+双向传播
#
# Design analysis:
# - Concrete trace: in the base task, a seed with mostly color A and some color B
#   turns its attached P branch to A, while only the hub center changes. In this
#   combo, the same seed sends A outward onto the free branch chain, sends B inward
#   onto its own P frame, and the winning hub-adjacent branch color fills the whole
#   3x3 hub. The visible result is a three-layer seed structure plus a solid hub.
# - Complexity calibration: the generator keeps exactly two seed-interior colors in
#   each instance and every seed contains both with a strict majority. That makes
#   majority/minority readable by eye without relying on numeric color ids.
# - Hidden assumptions audit: the original solver could label every non-hub P cell
#   as one component family because seed borders and branches shared one output
#   color. That assumption is false here. The combo solver first reconstructs each
#   seed border from the seed bbox, recolors those border cells by minority color,
#   and only then labels the remaining P cells as branches.
# - Placement legality: the generator still uses disjoint edge-touching seed zones,
#   a centered 3x3 hub, and one-pixel orthogonal paths with unique hub starts, so
#   objects do not overlap and path degree stays <= 2 away from the hub/borders.
# - Observability and neatness: every output shows repeated examples of mixed seed
#   interiors, minority-colored seed frames, majority-colored branches, and a fully
#   recolored hub. This is visually cleaner and more legible than letting many
#   unrelated colors or overlapping decorations compete in the same area.
# - Non-degeneracy target: each generated instance has both majority colors present
#   across the seed set and a non-tied hub winner, so the combo output differs from
#   the base task both at the hub (8 extra recolored cells) and at the seed borders.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask, get_bg, label_mask

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _find_hub_square(mP: np.ndarray) -> tuple[int, int, int]:
    H, W = mP.shape
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
    return best


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if r0 > 0:
        cells.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        cells.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 > 0:
        cells.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        cells.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        cells.append((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < W - 1:
        cells.append((r0 - 1, c1 + 1))
    if r1 < H - 1 and c0 > 0:
        cells.append((r1 + 1, c0 - 1))
    if r1 < H - 1 and c1 < W - 1:
        cells.append((r1 + 1, c1 + 1))
    return cells


def _landing_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if r0 > 0:
        cells.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        cells.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 > 0:
        cells.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        cells.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch network with outward and inward seed propagation.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid 3x3 hub.
    - Each seed is a non-P, non-background rectangle of two colors, wrapped by a 1-cell P frame on non-edge sides.

    Rule:
    - Infer P as the non-background color whose largest component looks most like a path network.
    - Detect the largest solid all-P square as the hub and keep it separate from all other P cells.
    - For each seed, recolor its P frame to the seed's minority interior color and recolor every attached branch
      chain to the seed's majority interior color.
    - Count the recolored branch colors touching the hub and repaint the entire hub square to the most frequent one.

    Key insight:
    - The same seed controls two propagation directions: majority color travels outward along free branches, while
      minority color folds back onto the seed frame. The hub is the final aggregator, taking the dominant branch color
      across all of its neighbors and broadcasting it across the whole 3x3 block.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors

    P = int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))

    k, hr0, hc0 = _find_hub_square(g == int(P))
    hr1, hc1 = int(hr0 + k - 1), int(hc0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[hr0 : hr1 + 1, hc0 : hc1 + 1] = True

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0

    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
    border_owner = np.zeros((H, W), dtype=int)
    border_mask = np.zeros((H, W), dtype=bool)

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        assert len(counts) >= 2, "seed interior must contain both majority and minority colors"
        major = int(counts.most_common(1)[0][0])
        minor = int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0])
        seed_major[int(sid)] = major
        seed_minor[int(sid)] = minor

        for br, bc in _seed_border_cells(r0, c0, r1, c1, H, W):
            assert int(g[br, bc]) == int(P), "seed missing expected P border"
            assert not bool(hub[br, bc]), "seed border must stay outside the hub"
            border_owner[br, bc] = int(sid)
            border_mask[br, bc] = True

    out = g.copy()
    for r, c in np.argwhere(border_mask):
        sid = int(border_owner[int(r), int(c)])
        out[int(r), int(c)] = int(seed_minor[int(sid)])

    full_pmask = g == int(P)
    ppad = np.pad(full_pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    branch_mask = full_pmask & (~hub) & (~border_mask)
    assert not np.any((p_deg > 2) & branch_mask), "path has T-junction outside hub/borders"

    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    assert int(branch_n) > 0

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
                        touches[sid] += 1
        assert touches, "every branch must touch a seed border"
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_major[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    for r, c in np.argwhere(hub):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    hub_col = int(touch_cols.most_common(1)[0][0])
    out[hub] = int(hub_col)
    return out


# ── Generator parameters ───────────────────────────────────────────────────

GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
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


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set(dest_cells)

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
    n_bends = random.randint(0, MAX_BENDS)

    def _attempt(initial_dir: tuple[int, int]) -> list[tuple[int, int]] | None:
        path: list[tuple[int, int]] = []
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
            path = _attempt(d)
            if path is not None:
                return path
    return None


def _compute_seed_zone(edge: str, r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (r0, max(0, c0 - 1), r1 + 1, min(W - 1, c1 + 1))
    if edge == "bottom":
        return (max(0, r0 - 1), max(0, c0 - 1), r1, min(W - 1, c1 + 1))
    if edge == "left":
        return (max(0, r0 - 1), c0, min(H - 1, r1 + 1), c1 + 1)
    return (max(0, r0 - 1), max(0, c0 - 1), min(H - 1, r1 + 1), c1)


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
        r0, c0 = 0, random.randint(0, W - iw)
    elif edge == "bottom":
        r0, c0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        r0, c0 = random.randint(0, H - ih), 0
    else:
        r0, c0 = random.randint(0, H - ih), W - iw
    return int(r0), int(c0), int(r0 + ih - 1), int(c0 + iw - 1)


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, majority: int, minority: int, sid: int) -> None:
    interior = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    n_int = len(interior)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior)
    for r, c in interior[:n_minor]:
        work[r, c] = int(minority)
        seed_id_grid[r, c] = int(sid)
    for r, c in interior[n_minor:]:
        work[r, c] = int(majority)
        seed_id_grid[r, c] = int(sid)


def _choose_majority_plan(n_seeds: int, A: int, B: int) -> tuple[list[int], int]:
    winner = random.choice([A, B])
    loser = B if winner == A else A
    if n_seeds == 3:
        winner_count = 2
    elif n_seeds == 4:
        winner_count = 3
    else:
        winner_count = random.choice([3, 4])
    plan = [winner] * winner_count + [loser] * (n_seeds - winner_count)
    random.shuffle(plan)
    return plan, int(winner)


def _assign_target_paths(seeds: list[dict], winner_majority: int) -> None:
    for seed in seeds:
        seed["target_paths"] = 1
    winner_ids = [i for i, seed in enumerate(seeds) if int(seed["majority"]) == int(winner_majority)]
    extra_budget = random.randint(0, min(2, 8 - len(seeds)))
    for _ in range(extra_budget):
        candidates = [i for i in winner_ids if int(seeds[i]["target_paths"]) < MAX_PATHS_PER_SEED]
        if not candidates:
            break
        si = random.choice(candidates)
        seeds[si]["target_paths"] += 1


def _carve_all_paths(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
    used_hub_starts: set[tuple[int, int]] = set()

    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        ok = False
        for _ in range(PATH_FIRST_ROUND_RETRIES):
            avail = [cell for cell in hub_perimeter if cell not in used_hub_starts]
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

    extra_requests = [si for si, seed in enumerate(seeds) for _ in range(int(seed["target_paths"]) - 1)]
    random.shuffle(extra_requests)
    for si in extra_requests:
        ok = False
        for _ in range(PATH_EXTRA_ROUND_RETRIES):
            avail = [cell for cell in hub_perimeter if cell not in used_hub_starts]
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


def _gen_attempt() -> np.ndarray | None:
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(P)
    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds: list[dict] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]
    majority_plan, winner_majority = _choose_majority_plan(n_seeds_target, A, B)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    for major in majority_plan:
        placed = False
        for _ in range(SEED_PLACEMENT_RETRIES):
            edge = random.choice(edges)
            r0, c0, r1, c1 = _pick_seed_interior(edge, H, W)
            border_zone = _compute_seed_zone(edge, r0, c0, r1, c1, H, W)
            if _overlaps_hub(border_zone):
                continue
            if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
                continue

            minority = B if int(major) == int(A) else A
            sid = len(seeds) + 1
            _fill_seed_interior(work, seed_id_grid, r0, c0, r1, c1, int(major), int(minority), int(sid))
            border_cells = _seed_border_cells(r0, c0, r1, c1, H, W)
            landing_cells = _landing_cells(r0, c0, r1, c1, H, W)
            for br, bc in border_cells:
                work[br, bc] = int(P)

            used_zones.append(border_zone)
            seeds.append(
                {
                    "id": sid,
                    "majority": int(major),
                    "minority": int(minority),
                    "border_cells": border_cells,
                    "landing_cells": landing_cells,
                    "bbox": (r0, c0, r1, c1),
                    "border_zone": border_zone,
                }
            )
            placed = True
            break
        if not placed:
            return None

    if len(seeds) < 3:
        return None

    _assign_target_paths(seeds, winner_majority)
    if not _carve_all_paths(work, int(P), hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    hub_mask = inp == int(P)
    hk, hr, hc = _find_hub_square(hub_mask)
    assert hk == HUB_SIZE
    hub_out = out[hr : hr + HUB_SIZE, hc : hc + HUB_SIZE]
    if len(set(map(int, hub_out.flatten()))) != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

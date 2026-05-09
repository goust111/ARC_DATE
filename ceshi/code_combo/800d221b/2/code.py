# Combo variant of 800d221b (index 2)
# combo_spec: D1/0+D2/3
# Name: 少数色着色+hub按种子数投票
#
# Design analysis:
# a) Concrete behavior trace:
#    - Example with 3 seeds using colors A/B. Suppose seed minorities are [B, B, A] and the path counts are [1, 1, 3].
#    - Original task: each seed-side component recolors to the seed majority, then the hub center picks the color with
#      the most hub-adjacent branch pixels, so the 3-path seed dominates the hub vote.
#    - Combo task: each seed-side component recolors to the seed minority instead, then the hub center counts connected
#      seed regions, not hub-adjacent pixels, so color B wins 2 seeds to 1 even though color A has more paths.
#    - The visible delta is therefore two-step: branch colors invert majority/minority, and the hub can disagree with
#      the visually densest color near the hub because it follows seed-count voting.
# b) Complexity calibration:
#    - Keep the original family: one central square hub, thin orthogonal paths, edge seeds, and only two seed colors.
#    - Fix the combo generator to 3 seeds so it can reliably realize a readable 2-vs-1 seed vote against an opposite
#      3-vs-2 hub-adjacent-pixel vote.
#    - This keeps the puzzle legible while making the hub rule materially different from the source task.
# c) Hidden assumptions audited:
#    - After removing the hub, each remaining P component must correspond to exactly one seed border plus all of that
#      seed's paths. The generator preserves separate seed frames and forbids shared paths/T-junctions.
#    - Every seed must have a strict minority color. The generator fills each seed with exactly two colors and enforces
#      minority < majority.
#    - The hub vote must be unique and non-degenerate. The generator enforces seed-count winner != pixel-count winner.
# d) Placement legality:
#    - Seed border zones are checked for overlap/gap and for hub clearance.
#    - Paths use distinct hub perimeter starts, remain in bounds, never overlap seeds or other paths, and avoid T-junctions.
# e) Visual observability:
#    - Samples deliberately contain both minority colors and unequal path multiplicities, so a human can see that hub
#      color is not chosen by "which color has more path pixels near the hub".
# f) Visual cleanliness:
#    - Objects stay separated by zone gaps; paths remain 1-cell wide outside the hub; the hub is a single 3x3 block.
# g) Inferability:
#    - No rule depends on numeric color identities. The structure itself explains the mapping: inspect each seed's
#      minority color, paint its connected component with that color, then count seeds per resulting color at the hub.

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


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, W = g.shape
    mP = g == int(path_color)
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
    return hub, (k, r0, c0)


def _minority_color(vals: list[int]) -> int:
    counts = Counter(int(v) for v in vals)
    assert counts
    return int(min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0])


def _extract_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    path_color = _infer_path_color(g, bg)
    hub, (k, r0, c0) = _find_hub(g, path_color)

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) > 0

    seed_minor: dict[int, int] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
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
        seed_minor[int(sid)] = _minority_color(vals)

    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(path_color)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    branch_seed: dict[int, int] = {}
    branch_color: dict[int, int] = {}
    branch_touches_hub: dict[int, bool] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        touches_hub = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
                    if bool(hub[rr, cc]):
                        touches_hub = True
        assert touches, "each branch component must touch a seed region"
        assert touches_hub, "each branch component must connect to the hub"
        sid = int(touches.most_common(1)[0][0])
        branch_seed[int(bid)] = int(sid)
        branch_color[int(bid)] = int(seed_minor[int(sid)])
        branch_touches_hub[int(bid)] = bool(touches_hub)

    return {
        "grid": g,
        "bg": bg,
        "path_color": path_color,
        "hub": hub,
        "hub_info": (k, r0, c0),
        "seed_lab": seed_lab,
        "seed_n": int(seed_n),
        "seed_minor": seed_minor,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_seed": branch_seed,
        "branch_color": branch_color,
        "branch_touches_hub": branch_touches_hub,
    }


def _pick_counter_winner(counter: Counter[int]) -> int:
    assert counter
    return int(counter.most_common(1)[0][0])


def _behavior_signature(grid: np.ndarray) -> dict[str, object]:
    data = _extract_structure(grid)
    g = data["grid"]
    hub = data["hub"]
    branch_lab = data["branch_lab"]
    seed_minor = data["seed_minor"]
    branch_seed = data["branch_seed"]
    branch_touches_hub = data["branch_touches_hub"]
    seed_n = int(data["seed_n"])

    seed_votes = Counter()
    for sid in range(1, seed_n + 1):
        if any(int(branch_seed[bid]) == int(sid) and bool(branch_touches_hub[bid]) for bid in branch_seed):
            seed_votes[int(seed_minor[int(sid)])] += 1

    pixel_votes = Counter()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(hub[r, c]):
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    bid = int(branch_lab[rr, cc])
                    if bid != 0:
                        sid = int(branch_seed[int(bid)])
                        pixel_votes[int(seed_minor[int(sid)])] += 1

    return {
        "seed_votes": dict(seed_votes),
        "pixel_votes": dict(pixel_votes),
        "seed_winner": _pick_counter_winner(seed_votes),
        "pixel_winner": _pick_counter_winner(pixel_votes),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-path network by seed-region minorities, then vote at the hub by seed count.

    Input structure:
    - The background is the most frequent color.
    - One non-background color forms a thin orthogonal network with a solid square hub and framed seed attachments.
    - Each seed interior is a two-color region adjacent to a P-colored border/path component.

    Rule:
    - Infer the path color, locate the largest solid all-path square as the hub, and separate the remaining P components.
    - For each seed-side component, recover the seed it borders and repaint that whole component with the seed's minority color.
    - Repaint only the hub center pixel with the minority color that is represented by the largest number of connected seeds.

    Key insight: Branch density near the hub is a decoy here; the center depends on how many seeds support each minority color,
    not on how many hub-adjacent path pixels happen to carry that color.
    """
    data = _extract_structure(grid)
    g = data["grid"]
    branch_lab = data["branch_lab"]
    branch_color = data["branch_color"]
    seed_minor = data["seed_minor"]
    seed_n = int(data["seed_n"])
    k, r0, c0 = data["hub_info"]

    out = g.copy()
    for bid in range(1, int(data["branch_n"]) + 1):
        col = int(branch_color[int(bid)])
        pts = np.argwhere(branch_lab == int(bid))
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    hub_votes = Counter()
    for sid in range(1, seed_n + 1):
        connected = any(int(data["branch_seed"][bid]) == int(sid) and bool(data["branch_touches_hub"][bid]) for bid in data["branch_seed"])
        if connected:
            hub_votes[int(seed_minor[int(sid)])] += 1

    cr = int(r0 + int(k) // 2)
    cc = int(c0 + int(k) // 2)
    out[cr, cc] = int(_pick_counter_winner(hub_votes))
    return out


# --------------- generator ---------------

GRID_SIZE_BASE = 12
GRID_SIZE_PER_SEED = 4
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 500
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_MANDATORY_RETRIES = 80
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 500

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
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
        nr, nc = sr + dr, sc + dc
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
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, allow_dest=True):
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


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work: np.ndarray, P: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int):
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


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int, sid: int):
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


def _choose_edge_plan(n_seeds: int) -> list[str]:
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    if n_seeds == 3:
        plan = edges[:3]
    else:
        plan = edges + [random.choice(edges)]
        random.shuffle(plan)
    return plan


def _make_seed_specs(n_seeds: int, A: int, B: int) -> list[dict[str, int]]:
    vote_color = random.choice([A, B])
    other_color = B if vote_color == A else A
    if n_seeds == 3:
        minority_colors = [vote_color, vote_color, other_color]
        target_paths = [1, 1, 3]
    else:
        minority_colors = [vote_color, vote_color, vote_color, other_color, other_color]
        target_paths = [1, 1, 1, 2, 2]
    paired = list(zip(minority_colors, target_paths))
    random.shuffle(paired)
    specs = []
    for minority, n_paths in paired:
        majority = B if minority == A else A
        specs.append({"majority": int(majority), "minority": int(minority), "target_paths": int(n_paths)})
    return specs


def _carve_all_paths(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict[str, object]]) -> dict[int, int] | None:
    used_hub_starts = set()
    paths_per_seed = {i: 0 for i in range(len(seeds))}
    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    seed_order.sort(key=lambda si: int(seeds[si]["target_paths"]), reverse=True)

    for si in seed_order:
        need = int(seeds[si]["target_paths"])
        for _ in range(need):
            ok = False
            for _ in range(PATH_MANDATORY_RETRIES):
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
                return None
    return paths_per_seed


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
    seeds: list[dict[str, object]] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edge_plan = _choose_edge_plan(n_seeds_target)
    seed_specs = _make_seed_specs(n_seeds_target, A, B)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    for idx, spec in enumerate(seed_specs):
        edge = edge_plan[idx]
        placed = False
        for _ in range(SEED_PLACEMENT_RETRIES):
            ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W)
            border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
            if _overlaps_hub(border_zone):
                continue
            if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
                continue
            sid = len(seeds) + 1
            _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, int(spec["majority"]), int(spec["minority"]), sid)
            border_cells, landing_cells = _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W)
            used_zones.append(border_zone)
            seeds.append(
                {
                    "id": sid,
                    "majority": int(spec["majority"]),
                    "minority": int(spec["minority"]),
                    "target_paths": int(spec["target_paths"]),
                    "border_cells": border_cells,
                    "landing_cells": landing_cells,
                    "bbox": (ir0, ic0, ir1, ic1),
                    "border_zone": border_zone,
                }
            )
            placed = True
            break
        if not placed:
            return None

    carved = _carve_all_paths(work, P, hub_perimeter, seeds)
    if carved is None:
        return None
    for si, count in carved.items():
        seeds[si]["actual_paths"] = int(count)

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(inp)
        sig = _behavior_signature(inp)
    except Exception:
        return None

    if len(sig["seed_votes"]) < 2:
        return None
    if sig["seed_winner"] == sig["pixel_winner"]:
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

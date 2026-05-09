# Combo variant of 800d221b (index 32)
# combo_spec: D2/3+D7/0+D11/2
# Name: hub大小编码种子数+hub按种子数投票
#
# Design analysis:
# - Concrete trace: in the original task a fixed 3x3 hub ignores seed count, and the hub center follows the
#   color with the most recolored hub-adjacent branch pixels. In this combo, a 4x4 hub means exactly 4 seeds.
#   If three A-majority seeds each have one path while one B-majority seed has four paths, the old rule would
#   pick B by 4 adjacent branch pixels vs 3, but the combo rule must pick A by 3 seed regions vs 1.
# - Complexity calibration: the generator keeps only two seed-majority colors and uses a single multi-path
#   "heavy" seed. That is enough to make the combo visible without adding unrelated object types or hidden
#   color tables.
# - Hidden assumptions audited against the original solver:
#   * "largest solid P square is the hub" remains valid because paths stay 1-cell wide and seed borders are
#     hollow. The solver still identifies P structurally, but now it counts seed regions first and uses that
#     count as the expected hub size.
#   * "hub has one obvious center pixel" breaks for even k, so the combo defines a distinguished center cell
#     at (r0 + k//2, c0 + k//2), i.e. the lower-right cell of the central 2x2 when k is even.
#   * "hub center can be decided from adjacent recolored pixels" is intentionally replaced: the solver must
#     deduplicate by connected seed id and vote per seed region.
#   * "each non-hub P component touches exactly one seed" is preserved by generation through separated seed
#     border zones and by forbidding path-path contact away from hub/seed borders.
# - Placement legality: seeds remain edge-touching framed rectangles with a gap buffer, the hub stays near the
#   middle, and path carving rejects out-of-bounds steps, overlaps, and T-junctions away from seed borders.
# - Visual observability: hub size visibly changes across multiple sizes and equals the number of seed rectangles.
#   Samples are rejected unless seed-vote and old pixel-vote disagree, so the fused behavior is explicit.
# - Visual cleanliness: only four colors are present, paths remain thin orthogonal chains, and all extra
#   information is encoded through existing geometry rather than extra marks.
# - Inferability: a human can count seed rectangles of each majority color and see that one color can own more
#   spokes at the hub yet still lose the center vote. The solver accepts k=2..5, while the generator samples
#   a stable visible subset {3,5}: k=2 cannot visibly diverge from the old hub rule, and the 4-seed family
#   was less reliable than 3/5 under the same clean layout constraints.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


GRID_SIZE_BASE = 12
GRID_SIZE_PER_SEED = 4
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30

SEED_COUNT_CHOICES = [3, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 500
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_RETRIES_PER_TARGET = 120
PATH_DIR_RETRIES = 4

GEN_OUTER_RETRIES = 400
PRIMARY_COUNT_RETRIES = 320

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SEED_COUNT_ROTATION = [3, 4, 5]
_seed_count_rotation_index = 0


def _network_score(mask: np.ndarray) -> float:
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
    best_sz = int(best_pts.shape[0])
    H, W = mask.shape

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


def _rect_sum_table(mask: np.ndarray):
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

    return rect_sum


def _count_square_external_adjacencies(g: np.ndarray, P: int, r0: int, c0: int, k: int) -> int:
    H, W = g.shape
    r1 = int(r0 + k - 1)
    c1 = int(c0 + k - 1)
    adj = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if r0 <= rr <= r1 and c0 <= cc <= c1:
                        continue
                    if int(g[rr, cc]) == int(P):
                        adj += 1
    return int(adj)


def _find_hub(g: np.ndarray, P: int, seed_count: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, W = g.shape
    k = int(seed_count)
    assert 2 <= k <= min(H, W), "seed count must map to a valid hub size"
    mP = g == int(P)
    rect_sum = _rect_sum_table(mP)

    best = None
    for r0 in range(0, int(H - k + 1)):
        r1 = int(r0 + k - 1)
        for c0 in range(0, int(W - k + 1)):
            c1 = int(c0 + k - 1)
            if rect_sum(int(r0), int(c0), int(r1), int(c1)) != int(k * k):
                continue
            adj = _count_square_external_adjacencies(g, int(P), int(r0), int(c0), int(k))
            center_bias = -abs((2 * r0 + k) - H) - abs((2 * c0 + k) - W)
            score = (adj, center_bias)
            if best is None or score > best[0]:
                best = (score, (int(k), int(r0), int(c0)))

    assert best is not None, "hub square matching seed count not found"
    _, (k, r0, c0) = best
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r0 + k), int(c0) : int(c0 + k)] = True
    return hub, (int(k), int(r0), int(c0))


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"
    P = int(max(colors, key=lambda c: _network_score(g == int(c))))

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 2, "grid must contain at least two seed regions"

    seed_major: dict[int, int] = {}
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
        maj = int(Counter(vals).most_common(1)[0][0])
        seed_major[int(sid)] = int(maj)

    hub, (k, r0, c0) = _find_hub(g, int(P), int(seed_n))
    out = g.copy()

    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    connected_seed_ids: set[int] = set()
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touch_counts = Counter()
        touched_seed_ids: set[int] = set()
        touches_hub = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touch_counts[int(sid)] += 1
                        touched_seed_ids.add(int(sid))
                    if bool(hub[rr, cc]):
                        touches_hub = True
        assert touch_counts, "every non-hub P component must touch a seed"
        assert len(touched_seed_ids) == 1, "branch component touches multiple seed regions"
        sid = int(next(iter(touched_seed_ids)))
        col = int(seed_major[int(sid)])
        out[pts[:, 0], pts[:, 1]] = int(col)
        if touches_hub:
            connected_seed_ids.add(int(sid))

    assert len(connected_seed_ids) == int(seed_n), "all seeds must connect back to the hub"

    pixel_vote = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    pixel_vote[int(out[rr, cc])] += 1

    seed_vote = Counter(seed_major[int(sid)] for sid in sorted(connected_seed_ids))
    assert seed_vote, "hub must be touched by at least one connected seed"

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    center_col = int(seed_vote.most_common(1)[0][0])
    out[int(cr), int(cc)] = int(center_col)

    meta = {
        "bg": int(bg),
        "path_color": int(P),
        "seed_count": int(seed_n),
        "hub_size": int(k),
        "hub_origin": (int(r0), int(c0)),
        "center": (int(cr), int(cc)),
        "seed_major": dict(seed_major),
        "connected_seed_ids": sorted(connected_seed_ids),
        "seed_vote_counts": dict(seed_vote),
        "pixel_vote_counts": dict(pixel_vote),
        "seed_vote_color": int(center_col),
        "pixel_vote_color": int(pixel_vote.most_common(1)[0][0]) if pixel_vote else None,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-spoke path network whose hub size encodes how many seed regions exist.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid kxk square hub near the middle.
    - There are exactly k seed regions; each seed is a mixed-color rectangle on a grid edge, wrapped by P on its
      non-edge sides, and one seed may connect to the hub through several separate spokes.

    Rule:
    - Infer P as the non-background color with the strongest network-like component.
    - Count the seed regions first; that count is the hub side length k.
    - Locate the solid kxk P square, recolor every non-hub P component to the majority interior color of the
      unique seed region it touches, and then recolor one distinguished hub-center cell at (r0 + k//2, c0 + k//2).
    - The hub-center color is chosen by seed-region count per color, not by the number of hub-adjacent branch pixels.

    Key insight:
    The hub size is informative: it tells you how many seeds must exist. The center color follows a per-seed vote,
    so one seed with many spokes can still lose to more seeds of the other color.
    """
    out, _ = _solve_with_meta(grid)
    return out


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells, max_bends: int = MAX_BENDS):
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

    n_bends = random.randint(0, int(max_bends))

    def _attempt(initial_dir):
        path = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(n_bends + 1):
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(int(max_steps)):
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, True):
                    canvas[nr, nc] = int(P)
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, False):
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


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int):
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


def _zones_overlap(zone, others, gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge: str, H: int, W: int, *, prefer_center: bool) -> tuple[int, int, int, int]:
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge in {"top", "bottom"}:
        lo = 0
        hi = int(W - iw)
        if prefer_center:
            lo = max(0, int(W // 4 - iw // 2))
            hi = min(int(W - iw), int((3 * W) // 4 - iw // 2))
            if lo > hi:
                lo, hi = 0, int(W - iw)
        ic0 = random.randint(int(lo), int(hi))
        ir0 = 0 if edge == "top" else int(H - ih)
    else:
        lo = 0
        hi = int(H - ih)
        if prefer_center:
            lo = max(0, int(H // 4 - ih // 2))
            hi = min(int(H - ih), int((3 * H) // 4 - ih // 2))
            if lo > hi:
                lo, hi = 0, int(H - ih)
        ir0 = random.randint(int(lo), int(hi))
        ic0 = 0 if edge == "left" else int(W - iw)
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


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
):
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


def _hub_perimeter_by_edge(hr0: int, hc0: int, k: int) -> dict[str, list[tuple[int, int]]]:
    top = [(hr0, c) for c in range(hc0, hc0 + k)]
    bottom = [(hr0 + k - 1, c) for c in range(hc0, hc0 + k)]
    left = [(r, hc0) for r in range(hr0, hr0 + k)]
    right = [(r, hc0 + k - 1) for r in range(hr0, hr0 + k)]
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _preferred_starts(edge: str, hub_side_cells: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    order = [edge] + [e for e in ["top", "bottom", "left", "right"] if e != edge]
    out = []
    seen = set()
    for key in order:
        for cell in hub_side_cells[key]:
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
    return out


def _carve_paths_with_targets(work: np.ndarray, P: int, hub_side_cells: dict[str, list[tuple[int, int]]], seeds) -> bool:
    used_hub_starts = set()
    seed_order = sorted(range(len(seeds)), key=lambda si: (-int(seeds[si]["target_paths"]), si))
    for si in seed_order:
        preferred = _preferred_starts(str(seeds[si]["edge"]), hub_side_cells)
        for _path_i in range(int(seeds[si]["target_paths"])):
            ok = False
            for _ in range(PATH_RETRIES_PER_TARGET):
                avail = [s for s in preferred if s not in used_hub_starts]
                if not avail:
                    break
                start = random.choice(avail)
                path = _try_carve_path(work, int(P), start, seeds[si]["landing_cells"])
                if path is not None:
                    used_hub_starts.add(start)
                    ok = True
                    break
            if not ok:
                return False
    return True


def _build_seed_plan(n_seeds: int, color_a: int, color_b: int):
    axis = random.choice(["vertical", "horizontal"])
    if int(n_seeds) == 3:
        if axis == "vertical":
            b_edge = random.choice(["top", "bottom"])
            a_edges = ["left", "right"]
        else:
            b_edge = random.choice(["left", "right"])
            a_edges = ["top", "bottom"]
        return [
            {"edge": b_edge, "majority": int(color_b), "minority": int(color_a), "target_paths": 3, "prefer_center": True},
            {"edge": a_edges[0], "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False},
            {"edge": a_edges[1], "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False},
        ]

    if int(n_seeds) == 4:
        if axis == "vertical":
            b_edge = random.choice(["top", "bottom"])
            a_edges = ["left", "right", "bottom" if b_edge == "top" else "top"]
        else:
            b_edge = random.choice(["left", "right"])
            a_edges = ["top", "bottom", "right" if b_edge == "left" else "left"]
        return [
            {"edge": b_edge, "majority": int(color_b), "minority": int(color_a), "target_paths": 4, "prefer_center": True},
            {"edge": a_edges[0], "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False},
            {"edge": a_edges[1], "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False},
            {"edge": a_edges[2], "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False},
        ]

    assert int(n_seeds) == 5
    if axis == "vertical":
        if random.choice([False, True]):
            b_edges = [("top", 3), ("bottom", 2)]
        else:
            b_edges = [("bottom", 3), ("top", 2)]
        dup_a_edge = random.choice(["left", "right"])
        a_edges = ["left", "right", dup_a_edge]
    else:
        if random.choice([False, True]):
            b_edges = [("left", 3), ("right", 2)]
        else:
            b_edges = [("right", 3), ("left", 2)]
        dup_a_edge = random.choice(["top", "bottom"])
        a_edges = ["top", "bottom", dup_a_edge]

    plan = [
        {"edge": edge, "majority": int(color_b), "minority": int(color_a), "target_paths": int(paths), "prefer_center": True}
        for edge, paths in b_edges
    ]
    plan.extend(
        {"edge": edge, "majority": int(color_a), "minority": int(color_b), "target_paths": 1, "prefer_center": False}
        for edge in a_edges
    )
    return plan


def _gen_attempt(n_seeds_target: int | None = None):
    if n_seeds_target is None:
        n_seeds_target = int(random.choice(SEED_COUNT_CHOICES))
    else:
        n_seeds_target = int(n_seeds_target)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), int(bg), dtype=int)

    row_lo = max(2, int(H // 3 - n_seeds_target // 2))
    row_hi = min(int(H - n_seeds_target - 2), int((2 * H) // 3 - n_seeds_target // 2))
    col_lo = max(2, int(W // 3 - n_seeds_target // 2))
    col_hi = min(int(W - n_seeds_target - 2), int((2 * W) // 3 - n_seeds_target // 2))
    if row_lo > row_hi or col_lo > col_hi:
        return None

    hr0 = random.randint(row_lo, row_hi)
    hc0 = random.randint(col_lo, col_hi)
    g[hr0 : hr0 + n_seeds_target, hc0 : hc0 + n_seeds_target] = int(P)
    hub_side_cells = _hub_perimeter_by_edge(int(hr0), int(hc0), int(n_seeds_target))

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    seed_plan = _build_seed_plan(int(n_seeds_target), int(A), int(B))

    def _overlaps_hub(zone) -> bool:
        r0, c0, r1, c1 = zone
        return not (
            r1 < hr0 - 1
            or r0 > hr0 + n_seeds_target
            or c1 < hc0 - 1
            or c0 > hc0 + n_seeds_target
        )

    for sid in range(1, int(n_seeds_target) + 1):
        spec = seed_plan[sid - 1]
        edge = str(spec["edge"])
        prefer_center = bool(spec["prefer_center"])
        majority = int(spec["majority"])
        minority = int(spec["minority"])
        placed = False
        for _ in range(SEED_PLACEMENT_RETRIES):
            ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W, prefer_center=prefer_center)
            border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
            if _overlaps_hub(border_zone):
                continue
            if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
                continue
            _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
            border_cells, landing_cells = _place_seed_border(work, int(P), ir0, ic0, ir1, ic1, H, W)
            if int(spec["target_paths"]) > len(landing_cells):
                for r in range(ir0, ir1 + 1):
                    for c in range(ic0, ic1 + 1):
                        work[r, c] = 0
                        seed_id_grid[r, c] = 0
                for r, c in border_cells:
                    work[r, c] = 0
                continue
            used_zones.append(border_zone)
            seeds.append(
                {
                    "id": int(sid),
                    "edge": edge,
                    "majority": int(majority),
                    "minority": int(minority),
                    "border_cells": border_cells,
                    "landing_cells": landing_cells,
                    "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                    "border_zone": border_zone,
                    "target_paths": int(spec["target_paths"]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if not _carve_paths_with_targets(work, int(P), hub_side_cells, seeds):
        return None

    inp = np.where(work == 0, int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != int(bg):
        return None

    try:
        out, meta = _solve_with_meta(inp)
    except Exception:
        return None

    if int(meta["seed_count"]) != int(n_seeds_target):
        return None
    if int(meta["hub_size"]) != int(n_seeds_target):
        return None
    if int(meta["seed_vote_color"]) != int(A):
        return None
    if int(meta["pixel_vote_color"]) != int(B):
        return None
    if int(meta["seed_vote_color"]) == int(meta["pixel_vote_color"]):
        return None
    _ = out
    return inp


def generate_input() -> np.ndarray:
    global _seed_count_rotation_index

    ordered_counts = [int(c) for c in SEED_COUNT_CHOICES if int(c) in _SEED_COUNT_ROTATION]
    if not ordered_counts:
        ordered_counts = [int(c) for c in SEED_COUNT_CHOICES]
    pivot = int(_SEED_COUNT_ROTATION[_seed_count_rotation_index % len(_SEED_COUNT_ROTATION)])
    _seed_count_rotation_index += 1
    preferred = [pivot] if pivot in ordered_counts else []
    fallback = [c for c in ordered_counts if c not in preferred]
    attempt_plan: list[tuple[int, int]] = []
    if preferred:
        attempt_plan.append((preferred[0], PRIMARY_COUNT_RETRIES))
    remaining = max(0, GEN_OUTER_RETRIES - sum(retries for _, retries in attempt_plan))
    if fallback:
        per_fallback = max(1, remaining // len(fallback))
        for count in fallback:
            attempt_plan.append((int(count), int(per_fallback)))
        used = sum(retries for _, retries in attempt_plan)
        if used < GEN_OUTER_RETRIES:
            count, retries = attempt_plan[-1]
            attempt_plan[-1] = (count, retries + (GEN_OUTER_RETRIES - used))
    elif attempt_plan:
        count, retries = attempt_plan[0]
        attempt_plan[0] = (count, max(retries, GEN_OUTER_RETRIES))
    else:
        attempt_plan = [(int(random.choice(SEED_COUNT_CHOICES)), GEN_OUTER_RETRIES)]

    for target_count, retries in attempt_plan:
        for _ in range(int(retries)):
            try:
                r = _gen_attempt(int(target_count))
            except Exception:
                r = None
            if r is not None:
                return r
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

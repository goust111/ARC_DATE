# Combo variant of 800d221b (index 12)
# combo_spec: D1/0+D19/2
# Name: 少数色+路径标记像素
#
# Design analysis:
# - Concrete behavior: each seed-facing branch still attaches to one mixed-color seed rectangle, but the branch
#   now contains exactly one straight-path marker pixel whose color participates in the recolor rule. Example:
#   if a seed interior is mostly A with a few B cells, then a branch marker of A causes the whole branch to turn B,
#   while a branch marker of an extra instruction color M causes the whole branch to turn M.
# - Complexity calibration: letting markers use only the seed colors would collapse the combo into plain minority
#   recoloring, so the generator introduces one dedicated marker color M and forces both cases to appear in each grid:
#   at least one branch with marker == majority and at least one branch with marker == M.
# - Hidden assumptions audited from the base solver:
#   path cells are no longer all color P, so branch extraction must treat certain singleton non-P pixels as path;
#   seed detection must explicitly exclude those marker pixels; and path-color inference cannot rely on full P-network
#   connectivity because each marker splits the raw P geometry into pieces.
# - Placement legality: markers are only inserted on interior straight cells of a carved path, never on the hub,
#   never adjacent to a seed border, and never on a bend, so they remain visually separable and solver-detectable.
# - Observability and cleanliness: the instruction alphabet is tiny (binary choice: majority-colored marker vs.
#   dedicated marker color M), repeated across 3-5 branches, so a human can compare multiple examples and notice that
#   matching-the-majority flips to the minority while non-matching markers are copied literally.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _largest_square_size(mask: np.ndarray) -> int:
    H, W = mask.shape
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = 0
    for k in range(2, int(min(H, W)) + 1):
        found = False
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = True
                    best = int(k)
        if not found:
            continue
    return int(best)


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


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    return int(max(colors, key=lambda c: (_largest_square_size(g == int(c)), _comp_deg_score(g == int(c)))))


def _find_hub(g: np.ndarray, P: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, W = g.shape
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
    return hub, (int(k), int(r0), int(c0))


def _detect_markers(g: np.ndarray, bg: int, P: int, hub: np.ndarray) -> np.ndarray:
    misc_mask = (g != int(bg)) & (g != int(P))
    lab, n = label_mask(misc_mask, connectivity=4)
    marker = np.zeros(g.shape, dtype=bool)
    H, W = g.shape

    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) != 1:
            continue
        r, c = map(int, pts[0])
        if bool(hub[r, c]):
            continue
        up = r > 0 and int(g[r - 1, c]) == int(P)
        down = r + 1 < H and int(g[r + 1, c]) == int(P)
        left = c > 0 and int(g[r, c - 1]) == int(P)
        right = c + 1 < W and int(g[r, c + 1]) == int(P)
        if up and down and (not left) and (not right):
            marker[r, c] = True
        elif left and right and (not up) and (not down):
            marker[r, c] = True
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network using both seed statistics and an in-path marker.

    Input structure:
    - The background is the most common color.
    - A thin orthogonal network of color P contains a solid square hub, but each non-hub branch also has one
      single-pixel non-P marker embedded on a straight segment.
    - Mixed-color seed rectangles sit at the grid edges behind P borders; their interiors determine majority/minority.

    Rule:
    - Infer P, find the largest solid all-P square as the hub, and treat straight singleton non-P pixels between
      opposite P neighbors as branch markers rather than seed cells.
    - For each branch component, identify the adjacent seed region and compare the branch marker color to that seed's
      majority color: if they match, recolor the entire branch to the seed's minority color; otherwise recolor it to
      the marker color itself.
    - Recolor the hub center to the color appearing most often on recolored branch pixels touching the hub.

    Key insight: the marker pixel is not a decorative anomaly; it is an instruction token that either names the branch
    color directly or, when it repeats the seed's dominant color, signals a flip to the seed's minority color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P = _infer_path_color(g, bg)
    hub, (k, r0, c0) = _find_hub(g, P)
    marker_mask = _detect_markers(g, bg, P, hub)
    path_mask = (g == int(P)) | marker_mask

    seed_mask = (g != int(bg)) & (~path_mask)
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    seed_minor: dict[int, int] = {}
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
        counts = Counter(vals)
        assert int(len(counts)) == 2, "seed interior must use exactly two colors"
        major, major_n = counts.most_common(1)[0]
        minor_candidates = [int(col) for col, n in counts.items() if int(n) != int(major_n)]
        assert len(minor_candidates) == 1, "seed majority must be strict"
        seed_major[int(sid)] = int(major)
        seed_minor[int(sid)] = int(minor_candidates[0])

    out = g.copy()

    branch_mask = path_mask & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    ppad = np.pad(branch_mask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & branch_mask & ~adj_seed), "path has T-junction"

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        markers = []
        for r, c in pts:
            r = int(r)
            c = int(c)
            if bool(marker_mask[r, c]):
                markers.append((int(r), int(c)))
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        assert len(markers) == 1, "each branch must contain exactly one marker"
        sid = int(touches.most_common(1)[0][0])
        mr, mc = markers[0]
        marker_col = int(g[mr, mc])
        if int(marker_col) == int(seed_major[int(sid)]):
            col = int(seed_minor[int(sid)])
        else:
            col = int(marker_col)
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
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1

# Path carving
MAX_BENDS = 6
SEGMENT_LEN_MIN = 2
SEGMENT_LEN_MAX = 6
MIN_PATH_CELLS = 4
PATH_FIRST_ROUND_RETRIES = 60
PATH_DIR_RETRIES = 3

# Top-level generation
GEN_OUTER_RETRIES = 400
WORK_FREE = -1


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != int(WORK_FREE):
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
            if v != int(WORK_FREE):
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in _DIRS:
        nr, nc = int(sr + dr), int(sc + dc)
        if 0 <= nr < H and 0 <= nc < W and int(canvas[nr, nc]) == int(WORK_FREE):
            init_dirs.append((int(dr), int(dc)))
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
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS):
                    if _check_neighbors(nr, nc, cur, allow_dest=True):
                        if int(len(path) + 1) < MIN_PATH_CELLS:
                            blocked = True
                            break
                        canvas[nr, nc] = int(P)
                        path.append((int(nr), int(nc)))
                        return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(P)
                    path.append((int(nr), int(nc)))
                    cur = (int(nr), int(nc))
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in path:
                    canvas[int(r), int(c)] = int(WORK_FREE)
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = (int(perps[0][0]), int(perps[0][1]))
        for r, c in path:
            canvas[int(r), int(c)] = int(WORK_FREE)
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


def _place_seed_border(work: np.ndarray, P: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    border_cells: list[tuple[int, int]] = []
    landing_cells: list[tuple[int, int]] = []
    for cond, cells in [
        (ir0 > 0, [(int(ir0 - 1), int(c)) for c in range(ic0, ic1 + 1)]),
        (ir1 < H - 1, [(int(ir1 + 1), int(c)) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(int(r), int(ic0 - 1)) for r in range(ir0, ir1 + 1)]),
        (ic1 < W - 1, [(int(r), int(ic1 + 1)) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[int(r), int(c)] = int(P)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (int(ir0 - 1), int(ic0 - 1))),
        (ir0 > 0 and ic1 < W - 1, (int(ir0 - 1), int(ic1 + 1))),
        (ir1 < H - 1 and ic0 > 0, (int(ir1 + 1), int(ic0 - 1))),
        (ir1 < H - 1 and ic1 < W - 1, (int(ir1 + 1), int(ic1 + 1))),
    ]:
        if cond:
            work[rc[0], rc[1]] = int(P)
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
        ir0, ic0 = int(H - ih), random.randint(0, W - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, H - ih), 0
    else:
        ir0, ic0 = random.randint(0, H - ih), int(W - iw)
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
) -> None:
    interior_cells = [(int(r), int(c)) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
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


def _path_marker_candidates(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cands: list[tuple[int, int]] = []
    for i in range(1, int(len(path)) - 1):
        pr, pc = path[int(i - 1)]
        r, c = path[int(i)]
        nr, nc = path[int(i + 1)]
        if int(pr) == int(r) == int(nr) and {int(pc), int(nc)} == {int(c - 1), int(c + 1)}:
            cands.append((int(r), int(c)))
        elif int(pc) == int(c) == int(nc) and {int(pr), int(nr)} == {int(r - 1), int(r + 1)}:
            cands.append((int(r), int(c)))
    return cands


def _carve_all_paths(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
    used_hub_starts = set()

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
            if path is not None and _path_marker_candidates(path):
                used_hub_starts.add(start)
                seeds[si]["marker_path"] = list(path)
                ok = True
                break
            if path is not None:
                for r, c in path:
                    work[int(r), int(c)] = int(WORK_FREE)
        if not ok:
            return False
    return True


def _assign_marker_modes(seeds: list[dict]) -> None:
    idxs = list(range(len(seeds)))
    random.shuffle(idxs)
    seeds[idxs[0]]["marker_mode"] = "majority"
    seeds[idxs[1]]["marker_mode"] = "extra"
    for i in idxs[2:]:
        seeds[i]["marker_mode"] = random.choice(["majority", "extra"])


def _place_markers(work: np.ndarray, extra_marker: int, seeds: list[dict]) -> bool:
    for seed in seeds:
        path = seed.get("marker_path")
        if not path:
            return False
        cands = _path_marker_candidates(path)
        if not cands:
            return False
        r, c = random.choice(cands)
        mark_col = int(seed["majority"]) if seed["marker_mode"] == "majority" else int(extra_marker)
        work[int(r), int(c)] = int(mark_col)
        seed["marker_cell"] = (int(r), int(c))
        seed["marker_color"] = int(mark_col)
    return True


def _gen_attempt() -> np.ndarray | None:
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B, M = random.sample(list(range(10)), 5)

    g = np.full((H, W), int(bg), dtype=int)
    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[int(hr0) : int(hr0 + HUB_SIZE), int(hc0) : int(hc0 + HUB_SIZE)] = int(P)

    hub_perimeter = [
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), int(WORK_FREE), g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds: list[dict] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]

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

        majority = random.choice([A, B])
        minority = int(B if majority == A else A)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W)
        used_zones.append(border_zone)
        seeds.append(
            {
                "id": int(sid),
                "majority": int(majority),
                "minority": int(minority),
                "border_cells": border_cells,
                "landing_cells": landing_cells,
                "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "border_zone": border_zone,
            }
        )

    if len(seeds) < 3:
        return None

    _assign_marker_modes(seeds)
    if not _carve_all_paths(work, P, hub_perimeter, seeds):
        return None
    if not _place_markers(work, M, seeds):
        return None

    inp = np.where(work == int(WORK_FREE), int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    branch_modes = Counter(seed["marker_mode"] for seed in seeds)
    if int(branch_modes["majority"]) < 1 or int(branch_modes["extra"]) < 1:
        return None
    if not np.any(out == int(M)):
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

# Combo variant of 800d221b (index 25)
# combo_spec: D6/0+D19/2
# Name: 路径标记色匹配种子
#
# Design analysis:
# - Concrete behavior trace:
#   In the original task, a branch touching a red-majority seed is recolored red because adjacency decides.
#   In this combo, that same branch contains one non-P marker pixel in its middle. If the marker is blue and
#   another seed family in the grid has blue as its majority color, then the whole touched branch recolors blue
#   even though it physically ends at the red-majority seed. The hub center still follows the majority color of
#   the recolored branches adjacent to the hub.
# - Complexity calibration:
#   Using only two seed majority colors keeps the mapping legible. The marker uses one of those same colors, so
#   the "similarity" rule is the zero-distance case of nearest-color matching rather than an invisible numeric
#   trick. Each grid contains repeated support for both target colors and at least one branch whose marker target
#   disagrees with the physically touched seed, so the combo does not collapse back to the original task.
# - Hidden-assumption audit:
#   The original solver assumes every branch cell is color P; that breaks once a mid-branch marker is non-P.
#   The fused solver therefore separates large non-P components (seed interiors) from singleton non-P components
#   (markers), then rebuilds branch connectivity from P plus markers. It also assumes one marker per connected
#   branch component, so the generator is simplified to one path per seed to keep that invariant true.
# - Placement legality:
#   Seeds are still edge-touching framed rectangles placed in non-overlapping zones away from the central hub.
#   Paths are carved on an empty work canvas with the original no-overlap/no-T-junction checks, and markers are
#   inserted only after carving onto interior path cells, so no object crosses another object or leaves the grid.
# - Visual observability and cleanliness:
#   The marker is a single conspicuous non-P pixel embedded in an otherwise uniform P branch, making the new cue
#   easy to spot. Because at least one branch is intentionally recolored against its touched seed, the input/output
#   pairs visibly contradict the old adjacency rule and direct the solver toward the marker-based interpretation.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import ARC_RGB, components_mask, get_bg, label_mask


GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_FIRST_ROUND_RETRIES = 80
PATH_DIR_RETRIES = 3
MIN_PATH_LEN_FOR_MARKER = 3

GEN_OUTER_RETRIES = 300

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


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain non-background colors"
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _largest_p_square(g: np.ndarray, P: int) -> tuple[int, int, int]:
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

    assert best is not None, "hub square not found"
    return tuple(map(int, best))


def _seed_color_distance(a: int, b: int) -> int:
    da = ARC_RGB[int(a)].astype(int)
    db = ARC_RGB[int(b)].astype(int)
    diff = da - db
    return int(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])


def _nearest_seed_color(marker_col: int, available: list[int]) -> int:
    assert available, "no seed colors available"
    return int(min(available, key=lambda col: (_seed_color_distance(marker_col, int(col)), int(col))))


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = int(_infer_path_color(g, bg))

    k, r0, c0 = _largest_p_square(g, P)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    raw_mask = (g != int(bg)) & (g != int(P))
    raw_lab, raw_n = label_mask(raw_mask, connectivity=4)

    seed_lab = np.zeros((H, W), dtype=int)
    marker_mask = np.zeros((H, W), dtype=bool)
    seed_major: dict[int, int] = {}
    marker_colors: list[int] = []
    next_sid = 1

    for raw_id in range(1, int(raw_n) + 1):
        pts = np.argwhere(raw_lab == int(raw_id))
        area = int(pts.shape[0])
        if area == 1:
            r, c = map(int, pts[0])
            marker_mask[int(r), int(c)] = True
            marker_colors.append(int(g[int(r), int(c)]))
            continue

        rn, cn = pts.min(axis=0)
        rx, cx = pts.max(axis=0)
        rn, cn, rx, cx = map(int, (rn, cn, rx, cx))
        assert area == int((rx - rn + 1) * (cx - cn + 1)), "seed interior must be rectangular"
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing bottom P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1)), "seed missing right P border"

        sid = int(next_sid)
        next_sid += 1
        seed_lab[raw_lab == int(raw_id)] = int(sid)
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        seed_major[int(sid)] = int(Counter(vals).most_common(1)[0][0])

    assert seed_major, "grid must contain seed interiors"
    assert marker_colors, "grid must contain marker pixels"

    network_mask = ((g == int(P)) | marker_mask) & (~hub)
    branch_lab, branch_n = label_mask(network_mask, connectivity=4)

    netpad = np.pad((g == int(P)) | marker_mask, 1)
    net_deg = (
        netpad[2:, 1:-1].astype(int)
        + netpad[:-2, 1:-1]
        + netpad[1:-1, 2:]
        + netpad[1:-1, :-2]
    )
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    bad_junction = (net_deg > 2) & (((g == int(P)) | marker_mask) & (~hub) & (~adj_seed))
    assert not np.any(bad_junction), "path has T-junction"

    branch_markers: dict[int, tuple[int, int]] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        marks = [(int(r), int(c)) for r, c in pts if bool(marker_mask[int(r), int(c)])]
        assert len(marks) == 1, "each branch must contain exactly one marker"
        branch_markers[int(bid)] = marks[0]

    return {
        "grid": g,
        "bg": int(bg),
        "P": int(P),
        "hub": hub,
        "hub_box": (int(k), int(r0), int(c0)),
        "seed_lab": seed_lab,
        "seed_major": seed_major,
        "marker_mask": marker_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_markers": branch_markers,
    }


def _physical_touch_color(info: dict, bid: int) -> int:
    g = info["grid"]
    seed_lab = info["seed_lab"]
    seed_major = info["seed_major"]
    pts = np.argwhere(info["branch_lab"] == int(bid))
    touches = Counter()
    H, W = g.shape
    for r, c in pts:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                sid = int(seed_lab[rr, cc])
                if sid != 0:
                    touches[int(sid)] += 1
    assert touches, "branch must touch a seed interior through its border"
    sid = int(touches.most_common(1)[0][0])
    return int(seed_major[int(sid)])


def _fusion_mismatch_count(grid: np.ndarray) -> tuple[int, int]:
    info = _analyze_scene(grid)
    seed_colors = sorted({int(v) for v in info["seed_major"].values()})
    mismatches = 0
    for bid in range(1, int(info["branch_n"]) + 1):
        mr, mc = info["branch_markers"][int(bid)]
        marker_col = int(info["grid"][int(mr), int(mc)])
        target_col = int(_nearest_seed_color(marker_col, seed_colors))
        physical_col = int(_physical_touch_color(info, int(bid)))
        if int(target_col) != int(physical_col):
            mismatches += 1
    return int(mismatches), int(info["branch_n"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network using mid-branch marker colors.

    Input structure:
    - bg is the most common color.
    - Path color P forms a thin orthogonal network with a solid square hub.
    - Seed interiors are colored rectangles framed by P near the grid edges.
    - Each branch contains exactly one embedded non-P marker pixel somewhere away from the ends.

    Rule:
    - Infer P and the hub as in the base family, but treat marker singletons as part of the path network.
    - Separate non-P singleton markers from larger non-P seed interiors.
    - For each branch, ignore which seed it physically touches; instead, compare the branch's marker color to the
      set of seed-majority colors and recolor the whole branch with the nearest seed-majority color.
    - Recolor the hub center pixel with the most common color among the recolored branch cells touching the hub.

    Key insight:
    The branch-to-seed relationship is no longer encoded by adjacency. The decisive clue is the lone colored marker
    inside each otherwise-P branch, whose color points to the seed color family that branch should inherit.
    """
    info = _analyze_scene(np.array(grid, dtype=int))
    g = info["grid"]
    out = g.copy()

    seed_colors = sorted({int(v) for v in info["seed_major"].values()})
    for bid in range(1, int(info["branch_n"]) + 1):
        mr, mc = info["branch_markers"][int(bid)]
        marker_col = int(g[int(mr), int(mc)])
        target_col = int(_nearest_seed_color(marker_col, seed_colors))
        pts = np.argwhere(info["branch_lab"] == int(bid))
        for r, c in pts:
            out[int(r), int(c)] = int(target_col)

    touch_cols = Counter()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(info["hub"][int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(info["branch_lab"][rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    k, r0, c0 = info["hub_box"]
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[int(cr), int(cc)] = int(touch_cols.most_common(1)[0][0])
    return out


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[int(r), int(c)]) != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = int(canvas[int(nr), int(nc)])
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
        if 0 <= nr < H and 0 <= nc < W and int(canvas[int(nr), int(nc)]) == 0:
            init_dirs.append((int(dr), int(dc)))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)

    n_bends = random.randint(0, int(max_bends))

    def _attempt(initial_dir: tuple[int, int]):
        path: list[tuple[int, int]] = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(int(n_bends) + 1):
            is_last = bool(seg_i == int(n_bends))
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(int(max_steps)):
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
                if any((int(nr + dr), int(nc + dc)) in dest_set for dr, dc in _DIRS) and _check_neighbors(int(nr), int(nc), cur, allow_dest=True):
                    canvas[int(nr), int(nc)] = int(P)
                    path.append((int(nr), int(nc)))
                    return path
                if _check_neighbors(int(nr), int(nc), cur, allow_dest=False):
                    canvas[int(nr), int(nc)] = int(P)
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
            cur_dir = (int(perps[0][0]), int(perps[0][1]))
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
                work[int(r), int(c)] = int(P)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(P)
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


def _fill_seed_interior(work: np.ndarray, seed_id_grid: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int, sid: int) -> None:
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


def _choose_seed_majorities(n: int, A: int, B: int) -> list[int]:
    count_a = random.randint(1, int(n - 1))
    majors = [int(A)] * int(count_a) + [int(B)] * int(n - count_a)
    random.shuffle(majors)
    return majors


def _choose_marker_targets(seed_majors: list[int], A: int, B: int) -> list[int]:
    n = len(seed_majors)
    for _ in range(200):
        primary = random.choice([int(A), int(B)])
        secondary = int(B if primary == int(A) else A)
        targets = [int(primary)] * int(n // 2 + 1) + [int(secondary)] * int(n - (n // 2 + 1))
        random.shuffle(targets)
        if len(set(targets)) < 2:
            continue
        if not any(int(t) != int(m) for t, m in zip(targets, seed_majors)):
            continue
        if not any(int(t) == int(m) for t, m in zip(targets, seed_majors)):
            continue
        return [int(x) for x in targets]
    raise RuntimeError("could not choose non-degenerate marker targets")


def _carve_one_path_per_seed(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
    used_hub_starts = set()
    order = list(range(len(seeds)))
    random.shuffle(order)
    for si in order:
        ok = False
        for _ in range(PATH_FIRST_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, int(P), start, seeds[si]["landing_cells"])
            if path is None:
                continue
            if len(path) < MIN_PATH_LEN_FOR_MARKER:
                for r, c in path:
                    work[int(r), int(c)] = 0
                continue
            used_hub_starts.add(start)
            seeds[si]["path"] = [(int(r), int(c)) for r, c in path]
            seeds[si]["hub_start"] = start
            ok = True
            break
        if not ok:
            return False
    return True


def _marker_index(path: list[tuple[int, int]]) -> int:
    assert len(path) >= MIN_PATH_LEN_FOR_MARKER
    return int(min(len(path) - 2, max(1, len(path) // 2)))


def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), int(bg), dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(P)

    hub_perimeter = [
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds: list[dict] = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]
    planned_majors = _choose_seed_majorities(int(n_seeds_target), int(A), int(B))

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

        majority = int(planned_majors[len(seeds)])
        minority = int(B if majority == int(A) else A)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, int(P), ir0, ic0, ir1, ic1, H, W)
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

    if len(seeds) != n_seeds_target:
        return None
    if len({int(seed["majority"]) for seed in seeds}) < 2:
        return None

    if not _carve_one_path_per_seed(work, int(P), hub_perimeter, seeds):
        return None

    targets = _choose_marker_targets([int(seed["majority"]) for seed in seeds], int(A), int(B))
    for seed, target in zip(seeds, targets):
        path = seed["path"]
        mid_idx = _marker_index(path)
        mr, mc = path[int(mid_idx)]
        work[int(mr), int(mc)] = int(target)
        seed["marker"] = (int(mr), int(mc))
        seed["target"] = int(target)

    inp = np.where(work == 0, int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        mismatches, branch_n = _fusion_mismatch_count(inp)
        if branch_n != len(seeds) or mismatches < 1:
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

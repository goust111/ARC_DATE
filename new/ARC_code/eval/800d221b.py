from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import Color, components_mask, get_bg, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a path-network by adjacent seed regions.

    Input structure:
    - bg = most common color.
    - Path color P forms a thin orthogonal network with a solid square hub (NxN).
    - Seed colors appear in colored regions (non-bg, non-P) adjacent to path branches.

    Rule:
    - Infer P as the non-bg color whose largest component is most network-like.
    - Find the largest solid k×k all-P square (the hub); exclude it from recoloring.
    - Recolor each path branch to the seed-region color it touches most.
    - Recolor the hub center pixel to the seed whose branches touch the hub most.

    Key insight: The hub stays uncolored; branches inherit color from adjacent seed regions by adjacency count.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert not (not colors)

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

    P = int(max(colors, key=lambda c: comp_deg_score(g == int(c))))

    # --- hub: largest solid k×k all-P square anywhere ---
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

    best = None  # (k, r0, c0)
    max_k = int(min(H, W))
    for k in range(2, max_k + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert not (best is None)

    k, r0, c0 = map(int, best)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    # --- seed regions (non-bg, non-P) ---
    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    seed_major: dict[int, int] = {}
    all_seed_vals: list[int] = []
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
        all_seed_vals.extend(vals)

    out = g.copy()

    # --- recolor each path branch by the seed region it touches most ---
    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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

    # --- hub center pixel recolor ---
    touch_cols = Counter()
    if int(branch_n) > 0:
        for r in range(H):
            for c in range(W):
                if not bool(hub[int(r), int(c)]):
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(branch_lab[rr, cc]) != 0:
                            touch_cols[int(out[rr, cc])] += 1

    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)

    assert touch_cols, 'hub must be touched by at least one recolored branch'
    assert all_seed_vals, 'grid must contain seed regions'
    center_col = int(touch_cols.most_common(1)[0][0])
    out[int(cr), int(cc)] = int(center_col)

    return out

# --------------- generator (attempt_1) ---------------

import random

# ── Tunable generation parameters ──────────────────────────────────────────
# Canvas / grid
GRID_SIZE_BASE = 10          # base grid dimension before seed-count scaling
GRID_SIZE_PER_SEED = 3       # extra min-dimension per seed beyond 3
GRID_SIZE_MIN = 12           # absolute minimum grid dimension
GRID_SIZE_MAX = 30           # maximum grid dimension

# Hub
HUB_SIZE = 3                 # hub is HUB_SIZE x HUB_SIZE solid square

# Seed regions
SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2        # min interior width/height
SEED_INTERIOR_MAX = 4        # max interior width/height
MINORITY_FRAC_LO = 0.06      # min minority color fraction in seed interior
MINORITY_FRAC_HI = 0.40      # max minority color fraction in seed interior
SEED_PLACEMENT_RETRIES = 400 # max attempts to place all seeds
SEED_ZONE_GAP = 1            # min separation between seed border zones

# Path carving
MAX_BENDS = 6                # max L-bends per path
SEGMENT_LEN_MIN = 1          # min segment length before a bend
SEGMENT_LEN_MAX = 6          # max segment length before a bend
MAX_PATHS_PER_SEED = 3       # cap on paths connecting one seed to hub
PATH_FIRST_ROUND_RETRIES = 60   # retries for first mandatory path per seed
PATH_EXTRA_ROUND_RETRIES = 30   # retries per extra path attempt
PATH_DIR_RETRIES = 3         # attempts per initial direction in carve

# Top-level generation
GEN_OUTER_RETRIES = 300      # max _gen_attempt calls in generate_input

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ── Path carving ───────────────────────────────────────────────────────────

def _try_carve_path(canvas, P, start, dest_border_cells, max_bends=MAX_BENDS):
    """Carve a single 1-pixel-wide orthogonal path from hub `start` toward `dest_border_cells`.

    Works on a "work canvas" where 0 = free, P = path/hub, other = seed.
    Returns list of placed (r, c) cells, or None on failure.
    Mutates `canvas` in-place (cleans up on failure).
    """
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r, c, prev, allow_dest):
        """Check all 4-neighbors for T-junction violations.

        Returns True if placing a path cell at (r,c) is safe.
        Cell must be in-bounds and free (canvas == 0).
        """
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

    # Initial directions: those leading from hub perimeter into a free cell
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
            is_last = (seg_i == n_bends)
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                # Check if this is a valid landing cell (adjacent to dest first -- fast reject)
                if (any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS)
                        and _check_neighbors(nr, nc, cur, allow_dest=True)):
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
                # Failed: blocked mid-segment or didn't reach dest on last segment
                for r, c in path:
                    canvas[r, c] = 0
                return None
            # Bend perpendicular for next segment
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


# ── Seed placement ─────────────────────────────────────────────────────────

def _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W):
    """Compute the border zone bbox (including P frame) for a seed on `edge`."""
    if edge == 'top':
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    elif edge == 'bottom':
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    elif edge == 'left':
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    else:  # right
        return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W):
    """Place P border around seed interior on non-grid-edge sides.

    Returns (border_cells, landing_cells).
    Landing cells are non-corner border cells adjacent to the interior -- valid
    path attachment points.
    """
    border_cells = []
    landing_cells = []
    # Side borders (landing-eligible)
    for cond, cells in [
        (ir0 > 0,     [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < H - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0,     [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < W - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[r, c] = P
                border_cells.append((r, c))
                landing_cells.append((r, c))
    # Corner pieces (NOT landing-eligible)
    for cond, rc in [
        (ir0 > 0 and ic0 > 0,         (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1,     (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0,     (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = P
            border_cells.append(rc)
    return border_cells, landing_cells


def _zones_overlap(zone, others, gap):
    """Check if `zone` overlaps or is within `gap` of any zone in `others`."""
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge, H, W):
    """Pick random interior bbox (ir0, ic0, ir1, ic1) for a seed touching `edge`."""
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge == 'top':
        ir0, ic0 = 0, random.randint(0, W - iw)
    elif edge == 'bottom':
        ir0, ic0 = H - ih, random.randint(0, W - iw)
    elif edge == 'left':
        ir0, ic0 = random.randint(0, H - ih), 0
    else:  # right
        ir0, ic0 = random.randint(0, H - ih), W - iw
    return ir0, ic0, ir0 + ih - 1, ic0 + iw - 1


def _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid):
    """Fill seed interior cells with majority/minority mix."""
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    # Ensure majority strictly larger
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = minority
        seed_id_grid[r, c] = sid
    for r, c in interior_cells[n_minor:]:
        work[r, c] = majority
        seed_id_grid[r, c] = sid


# ── Path orchestration ─────────────────────────────────────────────────────

def _carve_all_paths(work, P, hub_perimeter, seeds):
    """Carve paths from hub perimeter to each seed. Returns True on success."""
    used_hub_starts = set()
    paths_per_seed = {i: 0 for i in range(len(seeds))}

    # Round 1: one mandatory path per seed
    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        ok = False
        for _ in range(PATH_FIRST_ROUND_RETRIES):
            avail = [s for s in hub_perimeter if s not in used_hub_starts]
            if not avail:
                break
            start = random.choice(avail)
            path = _try_carve_path(work, P, start, seeds[si]['landing_cells'])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                ok = True
                break
        if not ok:
            return False

    # Extra rounds: add more paths per seed (up to MAX_PATHS_PER_SEED)
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
            path = _try_carve_path(work, P, start, seeds[si]['landing_cells'])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                break
    return True


# ── Main generation attempt ────────────────────────────────────────────────

def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    # Place hub near center (middle third)
    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0:hr0 + HUB_SIZE, hc0:hc0 + HUB_SIZE] = P

    # Hub perimeter = all cells on the border of the HUB_SIZE x HUB_SIZE square
    hub_perimeter = [(r, c) for r in range(hr0, hr0 + HUB_SIZE)
                     for c in range(hc0, hc0 + HUB_SIZE)
                     if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)]

    # Working canvas: 0 = free bg, P = path/hub, other = seed color
    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ['top', 'bottom', 'left', 'right']

    def _overlaps_hub(zone):
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    # ── Place seeds ──
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
        seeds.append({
            'id': sid, 'majority': majority, 'minority': minority,
            'border_cells': border_cells, 'landing_cells': landing_cells,
            'bbox': (ir0, ic0, ir1, ic1), 'border_zone': border_zone,
        })

    if len(seeds) < 3:
        return None

    # ── Carve paths ──
    if not _carve_all_paths(work, P, hub_perimeter, seeds):
        return None

    # ── Finalize ──
    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        try:
            r = _gen_attempt()
        except Exception:
            r = None
        if r is not None:
            return r
    raise RuntimeError("generate_input failed after %d attempts" % GEN_OUTER_RETRIES)

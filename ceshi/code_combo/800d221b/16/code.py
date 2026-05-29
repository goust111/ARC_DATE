# Combo variant of 800d221b (index 16)
# combo_spec: D3/0+D13/0
# Name: 反向传播+边框保持P
#
# Design analysis
# a) Concrete behavior trace:
#    - Base task: each seed interior keeps its mixed colors, while the attached P branch and the seed's own P border
#      are recolored to that seed's interior majority color. The hub center then copies the winning branch color.
#    - This combo reverses the propagation direction and also freezes the seed border at P. On a small example, a
#      top-edge seed with interior colors [A, A, B, A] and a P frame used to paint its whole branch-plus-frame as A.
#      Now the branch stays P, the frame stays P, and the four interior cells all become P. Visually the mixed-color
#      seed collapses into a solid P rectangle fused with the network.
#    - The key difference is therefore not "which seed owns which branch color" anymore; it is "which non-P rectangle
#      is enclosed by a P frame". Once that rectangle is found, every interior pixel is rewritten to P.
#
# b) Complexity calibration:
#    - The object family stays the same: one 3x3 hub, 3-5 edge seeds, and thin orthogonal P paths.
#    - The output remains visually distinct from the input because each sample contains several multi-cell mixed-color
#      seed interiors that disappear into the P network. Keeping 3-5 seeds gives repeated evidence that this happens
#      to all framed interiors, not just one special case.
#    - The generator still uses exactly two seed colors inside each seed, which keeps the scene readable and avoids
#      introducing extra color semantics that the combo no longer needs.
#
# c) Hidden-assumption audit:
#    - Inferring P as the most network-like non-background color still holds; the input object family is unchanged.
#    - Detecting the hub as the largest solid P square still holds because seed interiors are non-P in the input, so
#      only the central 3x3 block is the dominant solid P square.
#    - The original solver assumed every non-hub P component was recolorable and therefore mixed branches with seed
#      borders. That assumption is invalid here, so the revised solver stops labeling non-hub P components entirely.
#      It only validates the framed non-P seed interiors, then recolors those interiors to P.
#    - The revised solver adds a new assumption: every non-P connected component is a full rectangular seed interior
#      enclosed by P on every non-edge side. The generator guarantees this by constructing seeds as filled rectangles
#      with optional missing frame only on the grid edge they touch.
#
# d) Placement legality:
#    - Seed interiors are placed in edge-touching rectangles whose frame zones are kept separated by at least one cell.
#      This prevents overlap between seed interiors, seed borders, and future paths.
#    - Paths are carved through background-only cells and may terminate only next to seed-frame landing cells, so the
#      P network remains connected, in-bounds, and free of accidental crossings into seed interiors.
#
# e) Visual observability:
#    - Across 2-4 samples, a human sees the same consistent change: every multicolor rectangle enclosed by P becomes a
#      pure P rectangle, while the long P branches do not change color at all. That is a direct, visible signature of
#      reverse propagation plus border preservation.
#
# f) Visual cleanliness:
#    - No extra markers are added. The combo actually reduces clutter in the output because mixed-color seed interiors
#      disappear, while the existing path geometry and seed framing remain easy to parse.
#
# g) Inferability:
#    - No decision depends on numeric color ids. The solver only uses visible structure: the most network-like color is
#      P, and every non-P connected rectangle bounded by P is absorbed into that P network.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------- solver helpers ---------------


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

    assert best is not None, "failed to locate hub square"
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, int(k), int(r0), int(c0)


def _extract_seed_data(
    g: np.ndarray,
    bg: int,
    path_color: int,
) -> tuple[np.ndarray, int, list[tuple[int, int, int, int]]]:
    H, W = g.shape
    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 1, "grid must contain at least one seed interior"

    bboxes: list[tuple[int, int, int, int]] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0, "empty seed component"
        rn, cn = pts.min(axis=0)
        rx, cx = pts.max(axis=0)
        rn, cn, rx, cx = int(rn), int(cn), int(rx), int(cx)

        area = int((rx - rn + 1) * (cx - cn + 1))
        assert area == int(pts.shape[0]), "seed interior must be a filled rectangle"
        assert rn == 0 or rx == H - 1 or cn == 0 or cx == W - 1, "seed interior must touch a grid edge"

        if rn > 0:
            assert np.all(g[rn - 1, cn : cx + 1] == int(path_color)), "seed missing top P frame"
        if rx < H - 1:
            assert np.all(g[rx + 1, cn : cx + 1] == int(path_color)), "seed missing bottom P frame"
        if cn > 0:
            assert np.all(g[rn : rx + 1, cn - 1] == int(path_color)), "seed missing left P frame"
        if cx < W - 1:
            assert np.all(g[rn : rx + 1, cx + 1] == int(path_color)), "seed missing right P frame"

        bboxes.append((rn, cn, rx, cx))

    return seed_lab, int(seed_n), bboxes


def _original_task_output(g: np.ndarray) -> np.ndarray:
    bg = int(get_bg(g))
    path_color = int(_infer_path_color(g, bg))
    hub, k, r0, c0 = _find_hub(g, path_color)
    seed_lab, seed_n, _ = _extract_seed_data(g, bg, path_color)

    seed_major: dict[int, int] = {}
    all_seed_vals: list[int] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        maj = int(Counter(vals).most_common(1)[0][0])
        seed_major[int(sid)] = int(maj)
        all_seed_vals.extend(vals)

    out = g.copy()
    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    H, W = g.shape
    pmask = g == int(path_color)
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
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
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

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must be touched by at least one recolored branch"
    assert all_seed_vals, "grid must contain seed regions"
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[cr, cc] = int(touch_cols.most_common(1)[0][0])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Absorb every framed seed interior into the surrounding P network.

    Input structure:
    - The background is the most common color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Mixed-color seed interiors appear as filled rectangles touching the grid edge and framed by P on every
      non-edge side.

    Rule:
    - Infer P as the most network-like non-background color and locate the solid P hub square.
    - Identify each non-P, non-background connected rectangle whose exposed sides are bordered by P.
    - Recolor every cell inside those seed interiors to P, while leaving the branches, the seed frames, and the hub
      unchanged.

    Key insight:
    The color flow is reversed relative to the base task: P does not get recolored by the seeds. Instead, every seed
    interior is swallowed by the existing P network, and its P border stays fixed.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    path_color = int(_infer_path_color(g, bg))
    _hub, _k, _r0, _c0 = _find_hub(g, path_color)
    seed_lab, seed_n, _ = _extract_seed_data(g, bg, path_color)
    assert int(seed_n) >= 1, "expected at least one seed interior"

    out = g.copy()
    out[seed_lab > 0] = int(path_color)
    return out


# --------------- generator ---------------


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


def _try_carve_path(canvas, path_color, start, dest_border_cells, max_bends=MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r, c, prev, allow_dest):
        if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
            return False
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = canvas[nr, nc]
            if v == path_color:
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
            is_last = seg_i == n_bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                if (
                    any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS)
                    and _check_neighbors(nr, nc, cur, allow_dest=True)
                ):
                    canvas[nr, nc] = path_color
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = path_color
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


def _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W):
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W):
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
                work[r, c] = path_color
                border_cells.append((r, c))
                landing_cells.append((r, c))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = path_color
            border_cells.append(rc)
    return border_cells, landing_cells


def _zones_overlap(zone, others, gap):
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge, H, W):
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


def _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid):
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


def _carve_all_paths(work, path_color, hub_perimeter, seeds):
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


def _fusion_metrics(inp: np.ndarray) -> tuple[int, int, int]:
    bg = int(get_bg(inp))
    path_color = int(_infer_path_color(inp, bg))
    combo_out = main(inp)
    base_out = _original_task_output(inp)
    seed_mask = (inp != int(bg)) & (inp != int(path_color))
    preserved_path_pixels = int(np.count_nonzero((inp == int(path_color)) & (combo_out == int(path_color)) & (base_out != int(path_color))))
    absorbed_seed_pixels = int(np.count_nonzero(seed_mask & (combo_out == int(path_color))))
    diff_vs_base = int(np.count_nonzero(combo_out != base_out))
    return preserved_path_pixels, absorbed_seed_pixels, diff_vs_base


def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, path_color, col_a, col_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)
    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = path_color

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    def _overlaps_hub(zone):
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

        majority = random.choice([col_a, col_b])
        minority = col_b if majority == col_a else col_a
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

    if not _carve_all_paths(work, path_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
        bg2 = int(get_bg(inp))
        seed_mask = (inp != int(bg2)) & (inp != int(path_color))
        if int(np.count_nonzero(out != inp)) != int(np.count_nonzero(seed_mask)):
            return None
        if not np.all(out[seed_mask] == int(path_color)):
            return None
        preserved_path_pixels, absorbed_seed_pixels, diff_vs_base = _fusion_metrics(inp)
        if preserved_path_pixels < 6:
            return None
        if absorbed_seed_pixels < 12:
            return None
        if diff_vs_base < absorbed_seed_pixels + 6:
            return None
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
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")

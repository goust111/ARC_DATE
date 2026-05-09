# Combo variant of 800d221b (index 9)
# combo_spec: D1/3+D13/1
# Name: 颜色交换+边框少数色
#
# Design analysis:
# - Concrete behavior trace:
#   A seed interior that is 3 majority-blue cells plus 1 minority-red cell used to leave its
#   interior unchanged while both its P border and the attached branch turned blue. In this combo,
#   the attached branch still turns blue, the seed border turns red, and the 3 blue / 1 red interior
#   swaps to 3 red / 1 blue. The hub-center vote still looks only at branch colors, so the center
#   remains a majority-color summary rather than a border-color summary.
# - Complexity calibration:
#   The visual contrast comes from three aligned layers around each seed: swapped interior, minority
#   border, and majority branch. The generator keeps only two seed colors globally and requires both
#   majority choices to appear across the sample set, so the rule is visible without exploding the
#   color vocabulary.
# - Hidden-assumption audit:
#   The original solver could treat each seed border and its branch as one P component because they
#   were recolored identically. That assumption breaks here, so the solver explicitly reconstructs
#   seed borders from seed bounding boxes, removes those border cells from the branch mask, and only
#   then labels branch components. This avoids mixing "border->minority" with "branch->majority".
# - Placement legality:
#   The generator preserves the original non-overlap strategy: edge-anchored rectangular seeds,
#   1-pixel P frames on non-edge sides, a central 3x3 hub, and simple carved paths that may touch
#   only their own destination border. Gap checks keep seed zones disjoint and away from the hub.
# - Visual observability:
#   Every seed is forced to contain both colors with a strict majority, and both majority colors
#   appear among the seeds. That guarantees visible interior swaps and visible border-vs-branch
#   contrast; otherwise the combo could collapse into the baseline behavior on some samples.
# - Clean visualization:
#   No extra symbols are added. The same canonical objects remain readable because the three changed
#   roles are separated spatially: interior cells, border frame, and branch path.
# - New solver assumptions guaranteed by the generator:
#   Each seed interior is a solid rectangle of exactly two colors, each non-hub P cell is either a
#   border cell or a simple branch cell, and every branch component touches exactly one seed border.

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UTIL_PATH = ROOT / "solvers" / "arc_utils.py"
SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_9", UTIL_PATH)
arc_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arc_utils
assert SPEC.loader is not None
SPEC.loader.exec_module(arc_utils)

components_mask = arc_utils.components_mask
get_bg = arc_utils.get_bg
label_mask = arc_utils.label_mask

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_path_color(g: np.ndarray, bg: int) -> int:
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


def _largest_p_square_hub(g: np.ndarray, P: int) -> tuple[int, int, int, np.ndarray]:
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

    assert best is not None
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r0 + k), int(c0) : int(c0 + k)] = True
    return int(k), int(r0), int(c0), hub


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
    return sorted(set(cells))


def _touch_seed_borders(pts: np.ndarray, border_owner: np.ndarray) -> Counter[int]:
    H, W = border_owner.shape
    touches: Counter[int] = Counter()
    for r, c in pts:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                sid = int(border_owner[rr, cc])
                if sid:
                    touches[int(sid)] += 1
    return touches


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = _find_path_color(g, bg)
    k, hr0, hc0, hub = _largest_p_square_hub(g, P)

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 1

    border_owner = np.zeros((H, W), dtype=int)
    seed_infos: list[dict] = []
    seed_by_id: dict[int, dict] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        rect = seed_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert bool(np.all(rect)), "seed interior must be a solid rectangle"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals).most_common()
        assert len(counts) == 2, "combo expects exactly two colors per seed"
        assert counts[0][1] > counts[1][1], "combo expects strict majority/minority inside seed"
        major = int(counts[0][0])
        minor = int(counts[1][0])

        border = _seed_border_cells(r0, c0, r1, c1, H, W)
        assert border, "seed must expose at least one P border cell"
        for br, bc in border:
            assert int(g[br, bc]) == int(P), "seed border must be P-colored in the input"
            assert int(border_owner[br, bc]) == 0, "seed borders must not overlap"
            border_owner[br, bc] = int(sid)

        info = {
            "id": int(sid),
            "pts": pts,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "majority": int(major),
            "minority": int(minor),
            "border": border,
        }
        seed_infos.append(info)
        seed_by_id[int(sid)] = info

    seed_border_mask = border_owner > 0
    non_hub_p = (g == int(P)) & (~hub)
    branch_mask = non_hub_p & (~seed_border_mask)
    assert np.array_equal(non_hub_p, branch_mask | seed_border_mask)

    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    branch_to_seed: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = _touch_seed_borders(pts, border_owner)
        assert touches, "every branch must touch a seed border"
        assert len(touches) == 1, "a branch component may touch only one seed border"
        branch_to_seed[int(bid)] = int(touches.most_common(1)[0][0])

    pad = np.pad(branch_mask, 1)
    branch_deg = (
        pad[2:, 1:-1].astype(int)
        + pad[:-2, 1:-1]
        + pad[1:-1, 2:]
        + pad[1:-1, :-2]
    )
    assert not np.any(branch_mask & (branch_deg > 2)), "branch network must stay degree <= 2"

    return {
        "grid": g,
        "bg": int(bg),
        "P": int(P),
        "H": int(H),
        "W": int(W),
        "hub": hub,
        "hub_size": int(k),
        "hub_r0": int(hr0),
        "hub_c0": int(hc0),
        "seed_infos": seed_infos,
        "seed_by_id": seed_by_id,
        "seed_border_mask": seed_border_mask,
        "border_owner": border_owner,
        "branch_mask": branch_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_to_seed": branch_to_seed,
    }


def _render_output(
    grid: np.ndarray,
    scene: dict,
    *,
    swap_seed_interior: bool,
    border_to_minor: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()

    for seed in scene["seed_infos"]:
        major = int(seed["majority"])
        minor = int(seed["minority"])
        if swap_seed_interior:
            for r, c in seed["pts"]:
                r = int(r)
                c = int(c)
                val = int(g[r, c])
                if val == int(major):
                    out[r, c] = int(minor)
                else:
                    assert val == int(minor)
                    out[r, c] = int(major)
        border_color = int(minor if border_to_minor else major)
        for r, c in seed["border"]:
            out[int(r), int(c)] = int(border_color)

    for bid in range(1, int(scene["branch_n"]) + 1):
        pts = np.argwhere(scene["branch_lab"] == int(bid))
        sid = int(scene["branch_to_seed"][int(bid)])
        col = int(scene["seed_by_id"][int(sid)]["majority"])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols: Counter[int] = Counter()
    for r in range(scene["H"]):
        for c in range(scene["W"]):
            if not bool(scene["hub"][int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < scene["H"] and 0 <= cc < scene["W"]:
                    bid = int(scene["branch_lab"][rr, cc])
                    if bid != 0:
                        touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    cr = int(scene["hub_r0"] + scene["hub_size"] // 2)
    cc = int(scene["hub_c0"] + scene["hub_size"] // 2)
    out[int(cr), int(cc)] = int(touch_cols.most_common(1)[0][0])
    return out


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    scene = _analyze_scene(inp)
    swapped_interior_cells = 0
    border_minority_cells = 0
    branch_majority_cells = 0
    majority_palette = set()

    for seed in scene["seed_infos"]:
        major = int(seed["majority"])
        minor = int(seed["minority"])
        majority_palette.add(int(major))
        for r, c in seed["pts"]:
            if int(out[int(r), int(c)]) != int(inp[int(r), int(c)]):
                swapped_interior_cells += 1
        for r, c in seed["border"]:
            if int(out[int(r), int(c)]) == int(minor):
                border_minority_cells += 1

    for bid in range(1, int(scene["branch_n"]) + 1):
        sid = int(scene["branch_to_seed"][int(bid)])
        major = int(scene["seed_by_id"][int(sid)]["majority"])
        pts = np.argwhere(scene["branch_lab"] == int(bid))
        for r, c in pts:
            if int(out[int(r), int(c)]) == int(major):
                branch_majority_cells += 1

    baseline = _render_output(inp, scene, swap_seed_interior=False, border_to_minor=False)
    output_diff_from_baseline = int(np.count_nonzero(out != baseline))
    return {
        "seed_count": int(len(scene["seed_infos"])),
        "swapped_interior_cells": int(swapped_interior_cells),
        "border_minority_cells": int(border_minority_cells),
        "branch_majority_cells": int(branch_majority_cells),
        "majority_palette_size": int(len(majority_palette)),
        "output_diff_from_baseline": int(output_diff_from_baseline),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a hub-and-branch path network while splitting border and branch roles.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Rectangular seed interiors sit on grid edges, use exactly two non-P colors, and are framed by P on every non-edge side.

    Rule:
    - Infer P, recover the largest solid all-P square as the hub, and reconstruct each seed's P frame from its rectangular interior.
    - Recolor every branch component to the majority color of the seed it reaches.
    - Recolor each seed frame to that seed's original minority color, and swap the seed interior's majority/minority colors.
    - Recolor only the hub center pixel to the branch color that is most common along hub-adjacent branch contacts.

    Key insight: The same seed now drives three coupled effects: branch pixels inherit its majority color, frame pixels display its minority color, and the seed interior itself flips majority and minority.
    """
    g = np.array(grid, dtype=int)
    scene = _analyze_scene(g)
    return _render_output(g, scene, swap_seed_interior=True, border_to_minor=True)


# --------------- generator (combo-aware attempt) ---------------

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
MINORITY_FRAC_LO = 0.15
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


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != 0:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
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
        nr = int(sr + dr)
        nc = int(sc + dc)
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
                nr = int(cur[0] + cur_dir[0])
                nc = int(cur[1] + cur_dir[1])
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
    border_cells: list[tuple[int, int]] = []
    landing_cells: list[tuple[int, int]] = []
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
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _fill_seed_interior(work: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int) -> None:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(round(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(round(n_int * MINORITY_FRAC_HI)))
    max_minor = min(max_minor, (n_int - 1) // 2)
    n_minor = random.randint(min_minor, max_minor)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = int(minority)
    for r, c in interior_cells[n_minor:]:
        work[r, c] = int(majority)


def _carve_all_paths(work: np.ndarray, P: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
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
            path = _try_carve_path(work, P, start, seeds[si]["landing_cells"])
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
            path = _try_carve_path(work, P, start, seeds[si]["landing_cells"])
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
    seeds = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]

    majority_plan = [A] * (n_seeds_target // 2) + [B] * (n_seeds_target - n_seeds_target // 2)
    random.shuffle(majority_plan)
    assert len(set(majority_plan)) == 2

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
        _fill_seed_interior(work, ir0, ic0, ir1, ic1, majority, minority)
        border_cells, landing_cells = _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W)

        used_zones.append(border_zone)
        seeds.append(
            {
                "majority": int(majority),
                "minority": int(minority),
                "border_cells": border_cells,
                "landing_cells": landing_cells,
                "bbox": (ir0, ic0, ir1, ic1),
                "border_zone": border_zone,
            }
        )

    if len(seeds) < 3:
        return None

    if not _carve_all_paths(work, P, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != int(bg):
        return None

    try:
        out = main(inp)
        metrics = _combo_metrics(inp, out)
    except Exception:
        return None

    if metrics["swapped_interior_cells"] == 0:
        return None
    if metrics["border_minority_cells"] == 0:
        return None
    if metrics["branch_majority_cells"] == 0:
        return None
    if metrics["majority_palette_size"] < 2:
        return None
    if metrics["output_diff_from_baseline"] == 0:
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

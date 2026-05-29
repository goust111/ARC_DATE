# Combo variant of 800d221b (index 28)
# combo_spec: D1/3+D3/1+D13/1
# Name: 颜色交换+双向传播+边框少数色
#
# Design analysis:
# a) Concrete behavior trace:
#    - Base task: a seed interior with original majority color A and minority color B sends A onto the attached
#      P structure, so the seed border and branch both become A while the interior stays unchanged.
#    - This combo splits that one effect into three linked layers. The same seed now sends original majority A onto
#      the free branch, sends original minority B onto its own P border, and swaps the interior so every old A cell
#      becomes B and every old B cell becomes A.
#    - After the swap, the seed's new majority is exactly the original minority, so the recolored border and the
#      post-swap dominant interior color match. That is the fused signal the solver must recover from the input.
#
# b) Complexity calibration:
#    - The scene keeps the original family: one 3x3 hub, 3-5 edge-attached rectangular seeds, and thin orthogonal
#      branches. The extra difficulty comes from role separation, not from extra geometry.
#    - Only two seed-interior colors are used per instance, and both majority orientations are forced to appear across
#      the seeds. Humans can compare several seeds in one example and see that "branch follows old majority, border
#      follows old minority, interior swaps" is a structural rule rather than a palette accident.
#    - Minority occupancy is kept visibly non-trivial, so the swap and minority border remain legible by eye.
#
# c) Hidden-assumption audit:
#    - Inferring P as the most network-like non-background color still holds because the input family is unchanged.
#    - Hub detection as the largest solid all-P square still holds because the hub remains a clean 3x3 block.
#    - The original solver's broken assumption is that every non-hub P cell can be recolored as one family. That is
#      false here because border cells and branch cells now receive different colors. The revised solver reconstructs
#      each seed border from the seed bbox, removes those cells from the branch mask, and labels only the remaining
#      P cells as branches.
#    - New solver assumptions: each seed interior is a full rectangle of exactly two colors with a strict majority;
#      each branch component touches exactly one seed border; each seed has at least one branch. The generator
#      constructs and filters samples to guarantee those properties.
#
# d) Placement legality:
#    - Seeds are placed in disjoint edge-touching zones with a one-cell gap between border zones and with explicit
#      hub-overlap rejection.
#    - Seed borders are added only on non-edge sides, including corners when both adjacent sides exist, matching the
#      solver's reconstruction rule.
#    - Paths are carved through free background cells only, never through existing paths, borders, or seed interiors,
#      so objects stay in bounds, non-overlapping, and degree <= 2 away from the hub.
#
# e) Visual observability:
#    - Each seed exposes three synchronized cues in the output: swapped interior, minority-colored frame, and
#      majority-colored branch. The same two seed colors recur across multiple seeds, which gives enough repeated
#      evidence for a human to infer the fused rule from a few examples.
#    - The hub center still aggregates branch colors only, so the border color does not interfere with the hub vote.
#
# f) Visual cleanliness:
#    - The changed roles stay spatially separated: interior, frame, and branch occupy different cells. No extra
#      markers are introduced, so the output remains readable even though three linked changes happen at once.
#
# g) Change inferability and non-degeneracy:
#    - No decision depends on raw color ids. Majority/minority come from visible counts inside each seed.
#    - The generator rejects samples unless the combo output differs from both the base-task output and the
#      no-swap "branch-majority + border-minority" output, and unless every seed satisfies
#      "swapped new majority == border color". That prevents silent collapse back to a weaker variant.

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UTIL_PATH = ROOT / "solvers" / "arc_utils.py"
SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_28", UTIL_PATH)
arc_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arc_utils
assert SPEC.loader is not None
SPEC.loader.exec_module(arc_utils)

components_mask = arc_utils.components_mask
get_bg = arc_utils.get_bg
label_mask = arc_utils.label_mask

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------- solver helpers ---------------

def _infer_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors, "grid must contain a non-background color"

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


def _find_hub(g: np.ndarray, path_color: int) -> tuple[int, int, int, np.ndarray]:
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

    assert best is not None, "failed to find hub square"
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return int(k), int(r0), int(c0), hub


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if r0 > 0:
        for c in range(c0, c1 + 1):
            cells.add((int(r0 - 1), int(c)))
    if r1 < H - 1:
        for c in range(c0, c1 + 1):
            cells.add((int(r1 + 1), int(c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c0 - 1)))
    if c1 < W - 1:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c1 + 1)))
    if r0 > 0 and c0 > 0:
        cells.add((int(r0 - 1), int(c0 - 1)))
    if r0 > 0 and c1 < W - 1:
        cells.add((int(r0 - 1), int(c1 + 1)))
    if r1 < H - 1 and c0 > 0:
        cells.add((int(r1 + 1), int(c0 - 1)))
    if r1 < H - 1 and c1 < W - 1:
        cells.add((int(r1 + 1), int(c1 + 1)))
    return sorted(cells)


def _analyze_scene(g: np.ndarray) -> dict:
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    path_color = _infer_path_color(g, bg)
    hub_size, hub_r0, hub_c0, hub = _find_hub(g, path_color)

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 1, "grid must contain seed interiors"

    border_owner = np.zeros((H, W), dtype=int)
    seed_infos: list[dict] = []
    seed_by_id: dict[int, dict] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size > 0, "seed component must be non-empty"
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        assert np.all(seed_mask[r0 : r1 + 1, c0 : c1 + 1]), "seed interior must be a full rectangle"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals).most_common()
        assert len(counts) == 2, "combo expects exactly two colors per seed"
        assert counts[0][1] > counts[1][1], "seed interior must have a strict majority"
        major = int(counts[0][0])
        minor = int(counts[1][0])

        border = _seed_border_cells(r0, c0, r1, c1, H, W)
        assert border, "seed must expose at least one P border cell"
        for br, bc in border:
            assert not bool(hub[int(br), int(bc)]), "seed border must stay outside hub"
            assert int(g[int(br), int(bc)]) == int(path_color), "reconstructed border must be color P"
            assert int(border_owner[int(br), int(bc)]) == 0, "seed borders must not overlap"
            border_owner[int(br), int(bc)] = int(sid)

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

    border_mask = border_owner > 0
    non_hub_p = (g == int(path_color)) & (~hub)
    branch_mask = non_hub_p & (~border_mask)
    assert np.array_equal(non_hub_p, border_mask | branch_mask), "every non-hub P cell must be border or branch"

    bpad = np.pad(branch_mask, 1)
    bdeg = bpad[2:, 1:-1].astype(int) + bpad[:-2, 1:-1] + bpad[1:-1, 2:] + bpad[1:-1, :-2]
    assert not np.any(branch_mask & (bdeg > 2)), "branch network must stay degree <= 2"

    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    branch_to_seed: dict[int, int] = {}
    hub_touch_counter: Counter[int] = Counter()
    seed_branch_count: Counter[int] = Counter()

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touch_seed = Counter()
        hub_contacts = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            touches_hub_here = False
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(border_owner[int(rr), int(cc)])
                    if sid != 0:
                        touch_seed[int(sid)] += 1
                    if bool(hub[int(rr), int(cc)]):
                        touches_hub_here = True
            if touches_hub_here:
                hub_contacts += 1

        assert touch_seed, "every branch must touch a seed border"
        assert len(touch_seed) == 1, "a branch component may touch only one seed border"
        sid = int(next(iter(touch_seed)))
        branch_to_seed[int(bid)] = int(sid)
        seed_branch_count[int(sid)] += 1
        assert hub_contacts >= 1, "every branch must touch the hub"
        hub_touch_counter[int(seed_by_id[int(sid)]["majority"])] += int(hub_contacts)

    for sid in range(1, int(seed_n) + 1):
        assert int(seed_branch_count[int(sid)]) >= 1, "every seed must own at least one branch"

    return {
        "grid": g,
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "path_color": int(path_color),
        "hub": hub,
        "hub_size": int(hub_size),
        "hub_r0": int(hub_r0),
        "hub_c0": int(hub_c0),
        "seed_infos": seed_infos,
        "seed_by_id": seed_by_id,
        "seed_n": int(seed_n),
        "border_owner": border_owner,
        "border_mask": border_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_to_seed": branch_to_seed,
        "hub_touch_counter": hub_touch_counter,
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
                out[r, c] = int(minor if int(g[r, c]) == int(major) else int(major))
        border_color = int(minor if border_to_minor else major)
        for r, c in seed["border"]:
            out[int(r), int(c)] = int(border_color)

    for bid in range(1, int(scene["branch_n"]) + 1):
        sid = int(scene["branch_to_seed"][int(bid)])
        col = int(scene["seed_by_id"][int(sid)]["majority"])
        out[scene["branch_lab"] == int(bid)] = int(col)

    touch_cols = scene["hub_touch_counter"]
    assert touch_cols, "hub must touch at least one recolored branch"
    cr = int(scene["hub_r0"] + scene["hub_size"] // 2)
    cc = int(scene["hub_c0"] + scene["hub_size"] // 2)
    out[int(cr), int(cc)] = int(touch_cols.most_common(1)[0][0])
    return out


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    scene = _analyze_scene(inp)
    swapped_interior_cells = 0
    border_new_majority_cells = 0
    branch_majority_cells = 0
    seeds_with_border_matching_swapped_majority = 0
    majority_palette = set()

    for seed in scene["seed_infos"]:
        major = int(seed["majority"])
        minor = int(seed["minority"])
        majority_palette.add(int(major))
        local_swap_changes = 0
        for r, c in seed["pts"]:
            r = int(r)
            c = int(c)
            if int(out[r, c]) != int(inp[r, c]):
                swapped_interior_cells += 1
                local_swap_changes += 1
        local_border_matches = 0
        for r, c in seed["border"]:
            if int(out[int(r), int(c)]) == int(minor):
                border_new_majority_cells += 1
                local_border_matches += 1
        if local_swap_changes == int(len(seed["pts"])) and local_border_matches == int(len(seed["border"])):
            seeds_with_border_matching_swapped_majority += 1

    for bid in range(1, int(scene["branch_n"]) + 1):
        sid = int(scene["branch_to_seed"][int(bid)])
        major = int(scene["seed_by_id"][int(sid)]["majority"])
        pts = np.argwhere(scene["branch_lab"] == int(bid))
        for r, c in pts:
            if int(out[int(r), int(c)]) == int(major):
                branch_majority_cells += 1

    baseline = _render_output(inp, scene, swap_seed_interior=False, border_to_minor=False)
    no_swap_combo = _render_output(inp, scene, swap_seed_interior=False, border_to_minor=True)
    return {
        "seed_count": int(len(scene["seed_infos"])),
        "swapped_interior_cells": int(swapped_interior_cells),
        "border_new_majority_cells": int(border_new_majority_cells),
        "branch_majority_cells": int(branch_majority_cells),
        "majority_palette_size": int(len(majority_palette)),
        "seeds_with_border_matching_swapped_majority": int(seeds_with_border_matching_swapped_majority),
        "output_diff_from_base": int(np.count_nonzero(out != baseline)),
        "output_diff_from_no_swap_combo": int(np.count_nonzero(out != no_swap_combo)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a hub-and-branch path network with linked branch, border, and interior changes.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Each seed is a two-color rectangular interior touching a grid edge, wrapped by a 1-cell P frame on the sides
      that do not lie on the outer border.

    Rule:
    - Infer P from the most network-like non-background component and locate the largest solid all-P square as the hub.
    - For each seed, read its original interior majority and minority colors, reconstruct its exact P border from the
      seed bbox, and separate those border cells from the free branch cells.
    - Recolor each branch by the seed's original majority color, recolor that seed's P border by its original
      minority color, and swap those two colors throughout the seed interior.
    - Recolor the hub center pixel to the branch color with the most hub-adjacent branch contacts.

    Key insight:
    - The border color is not arbitrary: it equals the seed's original minority color, which becomes the seed's new
      majority after the interior swap. Branches still show the old majority, so each seed creates a readable
      contrast between outward branch propagation and inward border/interior inversion.
    """
    g = np.array(grid, dtype=int)
    scene = _analyze_scene(g)
    return _render_output(g, scene, swap_seed_interior=True, border_to_minor=True)


# --------------- generator ---------------

GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.18
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


def _hub_vote_is_unique(counter: Counter[int]) -> bool:
    if not counter:
        return False
    top = max(counter.values())
    return sum(1 for v in counter.values() if int(v) == int(top)) == 1


def _try_carve_path(canvas: np.ndarray, path_color: int, start: tuple[int, int], dest_border_cells, max_bends: int = MAX_BENDS):
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


def _zones_overlap(zone: tuple[int, int, int, int], others, gap: int) -> bool:
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


def _fill_seed_interior(work: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int):
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(np.ceil(n_int * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(np.floor(n_int * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[r, c] = int(minority)
    for r, c in interior_cells[n_minor:]:
        work[r, c] = int(majority)


def _carve_all_paths(work: np.ndarray, path_color: int, hub_perimeter, seeds) -> bool:
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
    bg, path_color, color_a, color_b = random.sample(list(range(10)), 4)

    g = np.full((H, W), int(bg), dtype=int)

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
    seeds = []
    used_zones = []
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

        majority = random.choice([int(color_a), int(color_b)])
        minority = int(color_b if majority == int(color_a) else int(color_a))
        _fill_seed_interior(work, ir0, ic0, ir1, ic1, majority, minority)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W)

        used_zones.append(border_zone)
        seeds.append(
            {
                "majority": int(majority),
                "minority": int(minority),
                "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "border_cells": border_cells,
                "landing_cells": landing_cells,
            }
        )

    if len(seeds) < 3:
        return None

    if not _carve_all_paths(work, path_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != int(bg):
        return None

    try:
        scene = _analyze_scene(inp)
        out = _render_output(inp, scene, swap_seed_interior=True, border_to_minor=True)
    except Exception:
        return None

    metrics = _combo_metrics(inp, out)
    if metrics["majority_palette_size"] < 2:
        return None
    if not _hub_vote_is_unique(scene["hub_touch_counter"]):
        return None
    if metrics["seed_count"] < 3:
        return None
    if metrics["swapped_interior_cells"] <= 0:
        return None
    if metrics["branch_majority_cells"] <= 0:
        return None
    if metrics["seeds_with_border_matching_swapped_majority"] != metrics["seed_count"]:
        return None
    if metrics["output_diff_from_base"] <= 0:
        return None
    if metrics["output_diff_from_no_swap_combo"] <= 0:
        return None
    if not np.array_equal(main(inp), out):
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

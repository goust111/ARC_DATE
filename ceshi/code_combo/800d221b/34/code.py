# Combo variant of 800d221b (index 34)
# combo_spec: D3/1+D5/1+D13/1
# Name: 双向传播+多色种子+边框少数色
#
# Design analysis:
# - Concrete behavior trace:
#   In the base task, a two-color seed with counts A=8, B=3 recolors both its path branch and its surrounding
#   P border to A. In this combo, every seed interior mixes exactly three colors with strict counts such as
#   A=8, C=4, B=1. The attached branch must still become the majority color A, but the seed's P border must
#   become the minority color B. The same seed therefore drives two different recolor targets from one count
#   table, which is the actual fusion between D3/1, D5/1, and D13/1.
# - Complexity calibration:
#   The geometry stays in the original hub-plus-orthogonal-branches family so the new difficulty is local and
#   legible: each seed visibly contains three interior colors, and the output shows a deliberate split between
#   branch color and border color. Counts are strictly ordered major > middle > minor, avoiding hidden ties.
# - Hidden assumptions audited against the original solver:
#   * "non-bg, non-P cells of one seed form one rectangle" is preserved and still used.
#   * "a non-hub P component can be recolored as one piece" is broken by the combo, because border and branch
#     now need different colors. The new solver reconstructs seed borders from seed bboxes and excludes them
#     before labeling branch components.
#   * "branch ownership can be inferred by direct adjacency to seed interior" is no longer valid once border
#     cells are removed from the branch mask. Ownership is now inferred by adjacency to the reconstructed
#     border cells of each seed.
#   * "seed counting only needs a majority color" is broken. The solver now reads one three-color histogram per
#     seed and extracts majority, middle, and minority so it can assign majority to branches and minority to
#     borders.
# - Placement legality:
#   Seeds remain edge-touching framed rectangles placed in separated border zones away from the central hub.
#   Paths are carved on a free-space work canvas and reject overlaps, out-of-bounds moves, and non-seed
#   contacts, so the generated scenes satisfy the new solver's ownership assumptions.
# - Visual observability:
#   A human can compare each seed's three interior colors with the output and see that the long branch matches
#   the most frequent color while the thin frame matches the rarest color. The generator rejects samples unless
#   the fused output differs from both a "border takes majority" counterfactual and a "border takes middle"
#   counterfactual, so the three-color minority rule is visible rather than cosmetic.
# - Visual cleanliness:
#   No extra markers or legends are introduced. All information is expressed through the existing seed interior,
#   branch, border, and hub structures, keeping the image sparse and readable.
# - Change inferability:
#   The transformation depends only on visible region counts and attachment structure. It never relies on color
#   ids, numeric ordering of palette values, or hidden metadata.
# - Anti-degeneration metric:
#   For every accepted sample we compare the fused output against two counterfactual solvers on the same input:
#   one that recolors borders with the majority color, and one that recolors borders with the middle-frequency
#   color. Accepted samples must differ from both, and the grid must exhibit multiple distinct seed majorities
#   and minorities.
# - New solver assumptions guaranteed by the generator:
#   each seed uses exactly three colors with strict count ordering; each branch component touches exactly one
#   seed border zone; every seed has at least one attached branch; and the hub-adjacent branch vote has a
#   unique winner in accepted samples.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT, ROOT / "solvers"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.insert(0, extra_s)

from arc_utils import components_mask, get_bg, label_mask


GRID_SIZE_BASE = 10
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
SEED_MIN_AREA = 6
SEED_PLACEMENT_RETRIES = 500
SEED_ZONE_GAP = 1

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_FIRST_ROUND_RETRIES = 70
PATH_EXTRA_ROUND_RETRIES = 35
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _counter_winner(counter: Counter[int]) -> int:
    return int(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _network_score(mask: np.ndarray) -> float:
    h, w = mask.shape
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
        if r + 1 < h and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < w and bool(m[r, c + 1]):
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
    return int(max(colors, key=lambda c: _network_score(g == int(c))))


def _largest_p_square(g: np.ndarray, pcol: int) -> tuple[int, int, int]:
    h, w = g.shape
    m = g == int(pcol)
    ps = m.astype(int).cumsum(axis=0).cumsum(axis=1)

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
    max_k = int(min(h, w))
    for k in range(2, max_k + 1):
        found = None
        for r0 in range(0, int(h - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(w - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found
    assert best is not None, "hub square not found"
    return tuple(map(int, best))


def _border_cells_for_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> list[tuple[int, int]]:
    border_cells: list[tuple[int, int]] = []
    if r0 > 0:
        border_cells.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 < h - 1:
        border_cells.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 > 0:
        border_cells.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 < w - 1:
        border_cells.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        border_cells.append((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < w - 1:
        border_cells.append((r0 - 1, c1 + 1))
    if r1 < h - 1 and c0 > 0:
        border_cells.append((r1 + 1, c0 - 1))
    if r1 < h - 1 and c1 < w - 1:
        border_cells.append((r1 + 1, c1 + 1))
    return border_cells


def _extract_seed_info(g: np.ndarray, bg: int, pcol: int) -> tuple[np.ndarray, dict[int, dict]]:
    h, w = g.shape
    seed_mask = (g != int(bg)) & (g != int(pcol))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    info: dict[int, dict] = {}

    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        assert pts.size, "empty seed component"
        rn, cn = pts.min(axis=0)
        rx, cx = pts.max(axis=0)
        rn, cn, rx, cx = map(int, (rn, cn, rx, cx))
        assert int(pts.shape[0]) == int((rx - rn + 1) * (cx - cn + 1)), "seed interior must be rectangular"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
        assert len(ordered) == 3, "each seed must use exactly three colors"
        assert ordered[0][1] < ordered[1][1] < ordered[2][1], "seed color counts must be strictly ordered"

        border_cells = _border_cells_for_bbox(rn, cn, rx, cx, h, w)
        assert border_cells, "seed must expose at least one border cell"
        assert all(int(g[r, c]) == int(pcol) for r, c in border_cells), "seed missing P border"

        info[int(sid)] = {
            "bbox": (rn, cn, rx, cx),
            "pts": pts,
            "counts": counts,
            "ordered": ordered,
            "major": int(ordered[2][0]),
            "middle": int(ordered[1][0]),
            "minor": int(ordered[0][0]),
            "border_cells": border_cells,
        }

    assert info, "grid must contain seed interiors"
    return seed_lab, info


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    pcol = int(_infer_path_color(g, bg))

    k, r0, c0 = _largest_p_square(g, pcol)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((h, w), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    seed_lab, seed_info = _extract_seed_info(g, bg, pcol)

    border_owner = np.zeros((h, w), dtype=int)
    for sid, meta in seed_info.items():
        for r, c in meta["border_cells"]:
            border_owner[int(r), int(c)] = int(sid)
    border_mask = border_owner > 0

    branch_mask = (g == int(pcol)) & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    ppad = np.pad(g == int(pcol), 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & (g == int(pcol)) & (~hub) & (~adj_seed)), "path has T-junction"

    branch_owner: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    sid = int(border_owner[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        assert touches, "every branch component must touch a seed border"
        branch_owner[int(bid)] = int(_counter_winner(touches))

    return {
        "grid": g,
        "bg": int(bg),
        "P": int(pcol),
        "H": int(h),
        "W": int(w),
        "hub": hub,
        "hub_box": (int(k), int(r0), int(c0)),
        "seed_lab": seed_lab,
        "seed_info": seed_info,
        "border_mask": border_mask,
        "border_owner": border_owner,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_owner": branch_owner,
    }


def _hub_touch_counter(info: dict, branch_colors: dict[int, int]) -> Counter[int]:
    g = info["grid"]
    h, w = g.shape
    hub = info["hub"]
    branch_lab = info["branch_lab"]
    counts: Counter[int] = Counter()
    for r in range(h):
        for c in range(w):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    bid = int(branch_lab[rr, cc])
                    if bid != 0:
                        counts[int(branch_colors[int(bid)])] += 1
    return counts


def _solve_with_border_mode(grid: np.ndarray, border_mode: str) -> np.ndarray:
    info = _analyze_scene(grid)
    g = info["grid"]
    out = g.copy()
    seed_info = info["seed_info"]
    branch_lab = info["branch_lab"]
    branch_owner = info["branch_owner"]

    for sid, meta in seed_info.items():
        border_color = int(meta[border_mode])
        for r, c in meta["border_cells"]:
            out[int(r), int(c)] = int(border_color)

    branch_colors: dict[int, int] = {}
    for bid in range(1, int(info["branch_n"]) + 1):
        sid = int(branch_owner[int(bid)])
        col = int(seed_info[int(sid)]["major"])
        branch_colors[int(bid)] = int(col)
        out[branch_lab == int(bid)] = int(col)

    hub_counts = _hub_touch_counter(info, branch_colors)
    assert hub_counts, "hub must be touched by at least one recolored branch"
    center_col = int(_counter_winner(hub_counts))

    k, r0, c0 = info["hub_box"]
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[int(cr), int(cc)] = int(center_col)
    return out


def _behavior_metrics(inp: np.ndarray) -> dict[str, int | bool]:
    info = _analyze_scene(inp)
    fused_out = _solve_with_border_mode(inp, "minor")
    majority_border_out = _solve_with_border_mode(inp, "major")
    middle_border_out = _solve_with_border_mode(inp, "middle")

    border_cell_count = int(sum(len(meta["border_cells"]) for meta in info["seed_info"].values()))
    diff_vs_major = int(np.count_nonzero(fused_out != majority_border_out))
    diff_vs_middle = int(np.count_nonzero(fused_out != middle_border_out))
    distinct_minorities = len({int(meta["minor"]) for meta in info["seed_info"].values()})
    distinct_majors = len({int(meta["major"]) for meta in info["seed_info"].values()})
    all_three = all(len(meta["counts"]) == 3 for meta in info["seed_info"].values())

    branch_colors = {
        int(bid): int(info["seed_info"][int(sid)]["major"])
        for bid, sid in info["branch_owner"].items()
    }
    hub_counts = _hub_touch_counter(info, branch_colors)
    hub_counts_sorted = sorted(hub_counts.values(), reverse=True)
    hub_vote_margin = int(hub_counts_sorted[0] - hub_counts_sorted[1]) if len(hub_counts_sorted) > 1 else int(hub_counts_sorted[0])

    return {
        "all_seeds_use_three_colors": bool(all_three),
        "border_cell_count": int(border_cell_count),
        "distinct_seed_minorities": int(distinct_minorities),
        "distinct_seed_majors": int(distinct_majors),
        "diff_vs_major_border": int(diff_vs_major),
        "diff_vs_middle_border": int(diff_vs_middle),
        "hub_vote_margin": int(hub_vote_margin),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch path network by splitting each seed's three-color statistics across branches and borders.

    Input structure:
    - The background is the most common color.
    - One path color P forms a thin orthogonal network with a solid square hub near the middle.
    - Every seed is a rectangular interior touching a grid edge, framed by P on its non-edge sides.
    - Each seed interior mixes exactly three colors, so the same seed exposes a visible majority, middle, and minority count.

    Rule:
    - Infer P as the most path-like non-background color, then locate the largest solid P square as the hub.
    - For each seed interior, count its three colors and identify both the majority color and the minority color.
    - Recolor every non-hub path branch that leads to a seed with that seed's majority color.
    - Recolor that seed's surrounding P border with the seed's minority color, then recolor only the hub center
      pixel to the color with the most hub-adjacent recolored branch pixels.

    Key insight:
    The same seed drives two synchronized propagations in opposite directions: outward through the branch via
    its majority color, and around its frame via its minority color. Solving the task therefore requires
    separating border cells from branch cells and reading a full three-color histogram from each seed.
    """
    return _solve_with_border_mode(grid, "minor")


def _try_carve_path(canvas: np.ndarray, pcol: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]], max_bends: int = MAX_BENDS):
    h, w = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < h and 0 <= c < w) or int(canvas[r, c]) != 0:
            return False
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = int(canvas[nr, nc])
            if v == int(pcol):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in DIR4:
        nr, nc = int(sr + dr), int(sc + dc)
        if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) == 0:
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
            max_steps = max(h, w) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr, nc = int(cur[0] + cur_dir[0]), int(cur[1] + cur_dir[1])
                if any((nr + dr, nc + dc) in dest_set for dr, dc in DIR4) and _check_neighbors(nr, nc, cur, allow_dest=True):
                    canvas[nr, nc] = int(pcol)
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    canvas[nr, nc] = int(pcol)
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


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, h: int, w: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(w - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(w - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(h - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(h - 1, ir1 + 1), ic1)


def _place_seed_border(work: np.ndarray, pcol: int, ir0: int, ic0: int, ir1: int, ic1: int, h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    border_cells: list[tuple[int, int]] = []
    landing_cells: list[tuple[int, int]] = []

    for cond, cells in [
        (ir0 > 0, [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < h - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < w - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[r, c] = int(pcol)
                border_cells.append((r, c))
                landing_cells.append((r, c))

    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < w - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < h - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < h - 1 and ic1 < w - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[rc[0], rc[1]] = int(pcol)
            border_cells.append(rc)
    return border_cells, landing_cells


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge: str, h: int, w: int) -> tuple[int, int, int, int]:
    for _ in range(100):
        ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        if ih * iw < SEED_MIN_AREA:
            continue
        if edge == "top":
            ir0, ic0 = 0, random.randint(0, w - iw)
        elif edge == "bottom":
            ir0, ic0 = h - ih, random.randint(0, w - iw)
        elif edge == "left":
            ir0, ic0 = random.randint(0, h - ih), 0
        else:
            ir0, ic0 = random.randint(0, h - ih), w - iw
        return (int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1))
    raise RuntimeError("failed to pick a large-enough seed interior")


def _choose_tricolor_counts(n_cells: int) -> tuple[int, int, int]:
    triples = []
    for minor in range(1, n_cells - 1):
        for middle in range(minor + 1, n_cells - minor):
            major = int(n_cells - middle - minor)
            if major > middle:
                triples.append((major, middle, minor))
    assert triples, "seed area too small for three strict counts"
    return random.choice(triples)


def _fill_seed_interior(work: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, seed_colors: tuple[int, int, int]) -> dict[str, int]:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    random.shuffle(interior_cells)

    major_n, middle_n, minor_n = _choose_tricolor_counts(len(interior_cells))
    major_col, middle_col, minor_col = random.sample(list(seed_colors), 3)

    idx = 0
    for count, col in [
        (major_n, major_col),
        (middle_n, middle_col),
        (minor_n, minor_col),
    ]:
        for r, c in interior_cells[idx : idx + count]:
            work[r, c] = int(col)
        idx += count

    return {
        "major": int(major_col),
        "middle": int(middle_col),
        "minor": int(minor_col),
        "major_n": int(major_n),
        "middle_n": int(middle_n),
        "minor_n": int(minor_n),
    }


def _carve_all_paths(work: np.ndarray, pcol: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
    used_hub_starts: set[tuple[int, int]] = set()
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
            path = _try_carve_path(work, pcol, start, seeds[si]["landing_cells"])
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
            path = _try_carve_path(work, pcol, start, seeds[si]["landing_cells"])
            if path is not None:
                used_hub_starts.add(start)
                paths_per_seed[si] += 1
                break
    return True


def _gen_attempt():
    n_seeds_target = random.choice(SEED_COUNT_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (n_seeds_target - 3))
    h = random.randint(size_lo, GRID_SIZE_MAX)
    w = random.randint(size_lo, GRID_SIZE_MAX)
    bg, pcol, seed_a, seed_b, seed_c = random.sample(list(range(10)), 5)
    seed_colors = (int(seed_a), int(seed_b), int(seed_c))

    g = np.full((h, w), int(bg), dtype=int)

    hr0 = random.randint(max(2, h // 3 - 1), min(h - 5, 2 * h // 3 - 2))
    hc0 = random.randint(max(2, w // 3 - 1), min(w - 5, 2 * w // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(pcol)

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), 0, g).astype(int)
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
        ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, h, w)
        border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, h, w)

        if _overlaps_hub(border_zone):
            continue
        if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
            continue

        stats = _fill_seed_interior(work, ir0, ic0, ir1, ic1, seed_colors)
        border_cells, landing_cells = _place_seed_border(work, pcol, ir0, ic0, ir1, ic1, h, w)

        used_zones.append(border_zone)
        seeds.append({
            "bbox": (ir0, ic0, ir1, ic1),
            "border_zone": border_zone,
            "border_cells": border_cells,
            "landing_cells": landing_cells,
            "stats": stats,
        })

    if len(seeds) < 3:
        return None

    if not _carve_all_paths(work, pcol, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    metrics = _behavior_metrics(inp)
    if not metrics["all_seeds_use_three_colors"]:
        return None
    if int(metrics["distinct_seed_minorities"]) < 2:
        return None
    if int(metrics["distinct_seed_majors"]) < 2:
        return None
    if int(metrics["diff_vs_major_border"]) <= 0:
        return None
    if int(metrics["diff_vs_middle_border"]) <= 0:
        return None
    if int(metrics["hub_vote_margin"]) <= 0:
        return None

    assert np.array_equal(out, main(inp))
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

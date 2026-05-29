# Combo variant of 800d221b (index 27)
# combo_spec: D1/0+D2/0+D13/1
# Name: 三重少数色/反转主题
#
# Design analysis:
# - Concrete trace: in the base task, a seed whose interior is mostly A with a few B cells sends A onto both its
#   attached P branch and its P frame, and the hub center takes the most common hub-adjacent branch color. Here the
#   same seed sends B onto the frame, sends B onto every attached branch path, and the hub center takes the least
#   common hub-adjacent branch color. The whole transform therefore flips the decision direction from major/most to
#   minor/least at every edited stage.
# - Complexity calibration: seed interiors still use exactly two visible colors with a strict majority/minority split.
#   That keeps minority detection readable by eye while avoiding excessive color variety or hidden numeric ordering.
# - Hidden assumptions audit: the original code could treat every non-hub P cell as one family because seed frames
#   and branch trunks shared the same recolor. In this combo they still share the same minority target, but D13 is a
#   real part of the rule, so the solver reconstructs seed frames explicitly from each seed bbox, recolors them first,
#   and only then labels the remaining P cells as branch paths.
# - Placement legality: the generator keeps a centered 3x3 hub, disjoint edge-touching seed rectangles, non-overlap
#   between seed zones and hub, and simple orthogonal paths whose starts on the hub perimeter are unique.
# - Visual observability: every accepted sample contains both minority colors among hub-adjacent branches and those
#   colors appear with different frequencies, so "least at the hub" is distinguishable from the base "most at the hub"
#   rather than collapsing into the same output.
# - Visual neatness: only four ARC colors are used per sample, borders remain one cell thick, branch paths never
#   overlap, and the hub stays a clean 3x3 anchor with only its center pixel changing.
# - Non-degeneracy guard: the generator rejects instances where the combo hub center matches the "minority branches +
#   most-common hub vote" control condition, or where hub-adjacent branches expose only one color. This forces the D2
#   inversion to be genuinely active in the samples.

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


def _find_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _find_hub_square(pmask: np.ndarray) -> tuple[int, int, int]:
    H, W = pmask.shape
    ps = pmask.astype(int).cumsum(axis=0).cumsum(axis=1)

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


def _pick_majority_color(counts: Counter[int]) -> int:
    return int(counts.most_common(1)[0][0])


def _pick_minority_color(counts: Counter[int]) -> int:
    return int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0])


def _pick_most_common_color(counts: Counter[int]) -> int:
    return int(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _pick_least_common_color(counts: Counter[int]) -> int:
    return int(sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[0][0])


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    P = _find_path_color(g, bg)

    hub_k, hub_r0, hub_c0 = _find_hub_square(g == int(P))
    hub = np.zeros((H, W), dtype=bool)
    hub[hub_r0 : hub_r0 + hub_k, hub_c0 : hub_c0 + hub_k] = True

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
        assert len(counts) >= 2, "seed interior must contain at least two colors"
        major = _pick_majority_color(counts)
        minor = _pick_minority_color(counts)
        assert counts[major] > counts[minor], "seed minority must be strict"
        seed_major[int(sid)] = int(major)
        seed_minor[int(sid)] = int(minor)

        if r0 > 0:
            assert all(int(g[r0 - 1, c]) == int(P) for c in range(c0, c1 + 1)), "seed missing top P border"
        if r1 < H - 1:
            assert all(int(g[r1 + 1, c]) == int(P) for c in range(c0, c1 + 1)), "seed missing bottom P border"
        if c0 > 0:
            assert all(int(g[r, c0 - 1]) == int(P) for r in range(r0, r1 + 1)), "seed missing left P border"
        if c1 < W - 1:
            assert all(int(g[r, c1 + 1]) == int(P) for r in range(r0, r1 + 1)), "seed missing right P border"

        for br, bc in _seed_border_cells(r0, c0, r1, c1, H, W):
            assert int(g[br, bc]) == int(P), "seed corner missing P border"
            assert not bool(hub[br, bc]), "seed border overlaps hub"
            border_owner[br, bc] = int(sid)
            border_mask[br, bc] = True

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    branch_mask = pmask & (~hub) & (~border_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    assert int(branch_n) > 0

    branch_seed: dict[int, int] = {}
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
                        touches[int(sid)] += 1
        assert touches, "every branch must touch a seed border"
        branch_seed[int(bid)] = int(touches.most_common(1)[0][0])

    center_r = int(hub_r0 + hub_k // 2)
    center_c = int(hub_c0 + hub_k // 2)

    return {
        "g": g,
        "bg": int(bg),
        "H": int(H),
        "W": int(W),
        "P": int(P),
        "hub": hub,
        "hub_k": int(hub_k),
        "hub_r0": int(hub_r0),
        "hub_c0": int(hub_c0),
        "center": (int(center_r), int(center_c)),
        "seed_lab": seed_lab,
        "seed_n": int(seed_n),
        "seed_major": seed_major,
        "seed_minor": seed_minor,
        "border_owner": border_owner,
        "border_mask": border_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_seed": branch_seed,
    }


def _render_output(info: dict, branch_color_of_seed: dict[int, int], center_picker) -> tuple[np.ndarray, Counter[int]]:
    g = info["g"]
    H = info["H"]
    W = info["W"]
    out = g.copy()

    border_owner = info["border_owner"]
    border_mask = info["border_mask"]
    branch_lab = info["branch_lab"]
    branch_seed = info["branch_seed"]
    hub = info["hub"]

    for r, c in np.argwhere(border_mask):
        sid = int(border_owner[int(r), int(c)])
        out[int(r), int(c)] = int(branch_color_of_seed[int(sid)])

    for bid in range(1, int(info["branch_n"]) + 1):
        sid = int(branch_seed[int(bid)])
        col = int(branch_color_of_seed[int(sid)])
        pts = np.argwhere(branch_lab == int(bid))
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    hub_touch = Counter()
    for r, c in np.argwhere(hub):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                hub_touch[int(out[rr, cc])] += 1

    assert hub_touch, "hub must be touched by at least one recolored branch"
    cr, cc = info["center"]
    out[int(cr), int(cc)] = int(center_picker(hub_touch))
    return out, hub_touch


def _combo_output(info: dict) -> tuple[np.ndarray, Counter[int]]:
    branch_color = {sid: int(col) for sid, col in info["seed_minor"].items()}
    return _render_output(info, branch_color, _pick_least_common_color)


def _base_output(info: dict) -> tuple[np.ndarray, Counter[int]]:
    branch_color = {sid: int(col) for sid, col in info["seed_major"].items()}
    return _render_output(info, branch_color, _pick_most_common_color)


def _minority_most_output(info: dict) -> tuple[np.ndarray, Counter[int]]:
    branch_color = {sid: int(col) for sid, col in info["seed_minor"].items()}
    return _render_output(info, branch_color, _pick_most_common_color)


def _behavior_signature(grid: np.ndarray) -> dict:
    info = _analyze_grid(grid)
    combo_out, combo_touch = _combo_output(info)
    base_out, base_touch = _base_output(info)
    minority_most_out, _ = _minority_most_output(info)
    cr, cc = info["center"]

    branch_mask = (info["branch_lab"] != 0)
    border_mask = info["border_mask"]
    branch_diff = int(np.sum(combo_out[branch_mask] != base_out[branch_mask]))
    border_diff = int(np.sum(combo_out[border_mask] != base_out[border_mask]))
    combo_items = sorted((int(col), int(cnt)) for col, cnt in combo_touch.items())
    base_items = sorted((int(col), int(cnt)) for col, cnt in base_touch.items())

    return {
        "combo_center": int(combo_out[int(cr), int(cc)]),
        "base_center": int(base_out[int(cr), int(cc)]),
        "minority_most_center": int(minority_most_out[int(cr), int(cc)]),
        "combo_touch_counts": combo_items,
        "base_touch_counts": base_items,
        "n_combo_touch_colors": int(len(combo_touch)),
        "branch_diff_cells": int(branch_diff),
        "border_diff_cells": int(border_diff),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a hub-and-branch network under an all-minority / least-frequency theme.

    Input structure:
    - The background is the most common color in the grid.
    - One non-background color P forms a thin orthogonal network with a solid square hub at the center.
    - Each seed is a two-color interior rectangle touching an outer edge, wrapped by a 1-cell P frame on the sides
      that are not flush with the grid boundary.

    Rule:
    - Infer P from the color whose largest component looks most like the path network, then find the largest solid
      all-P square as the hub.
    - For every seed, identify its majority and minority interior colors; recolor the seed's P frame to the minority
      color and recolor every attached branch path to that same minority color.
    - Count the recolored branch colors touching the hub and set only the hub center pixel to the least frequent of
      those adjacent branch colors.

    Key insight:
    - The whole puzzle is governed by the same inverted preference: the less common seed color controls both the frame
      and the outgoing branches, and the least represented hub-neighbor branch color controls the hub center.
    """
    info = _analyze_grid(grid)
    out, _ = _combo_output(info)
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

GEN_OUTER_RETRIES = 400


def _try_carve_path(canvas: np.ndarray, P: int, start: tuple[int, int], dest_border_cells, max_bends: int = MAX_BENDS):
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or canvas[r, c] != 0:
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
            init_dirs.append((int(dr), int(dc)))
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
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, True):
                    canvas[nr, nc] = int(P)
                    path.append((int(nr), int(nc)))
                    return path
                if _check_neighbors(nr, nc, cur, False):
                    canvas[nr, nc] = int(P)
                    path.append((int(nr), int(nc)))
                    cur = (int(nr), int(nc))
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for r, c in path:
                    canvas[r, c] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = (int(perps[0][0]), int(perps[0][1]))
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
                work[r, c] = int(P)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
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
        ir0, ic0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, H - ih), 0
    else:
        ir0, ic0 = random.randint(0, H - ih), W - iw
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


def _carve_all_paths(work: np.ndarray, P: int, hub_perimeter, seeds) -> bool:
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

    for si, n_paths in paths_per_seed.items():
        seeds[si]["path_count"] = int(n_paths)
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
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == bg, 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    seeds = []
    used_zones = []
    edges = ["top", "bottom", "left", "right"]

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    majority_pattern = [A, A, B]
    if n_seeds_target >= 4:
        majority_pattern.append(B)
    if n_seeds_target >= 5:
        majority_pattern.append(random.choice([A, B]))
    random.shuffle(majority_pattern)

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

        majority = int(majority_pattern[len(seeds)])
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

    if not _carve_all_paths(work, P, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        out = main(inp)
        sig = _behavior_signature(inp)
    except Exception:
        return None

    if sig["n_combo_touch_colors"] < 2:
        return None
    if sig["combo_center"] == sig["minority_most_center"]:
        return None
    if sig["branch_diff_cells"] <= 0 or sig["border_diff_cells"] <= 0:
        return None
    if np.array_equal(out, _base_output(_analyze_grid(inp))[0]):
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

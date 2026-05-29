# Combo variant of 800d221b (index 18)
# combo_spec: D5/1+D13/1
# Name: 多色种子+边框少数色
#
# Design analysis:
# - a) Concrete behavior trace:
#   - In the base task, a seed interior has only two colors, so "majority" means "the more frequent of the two", and
#     both the seed's P frame and every attached branch are recolored to that same winner.
#   - In this fusion, each seed interior uses exactly three colors with a strict frequency order, e.g. counts 5/3/1.
#     The attached branch must still choose the most frequent color, but the seed frame must choose the least frequent
#     color. The middle color stays visible only inside the seed and acts as evidence that the solver really ranked
#     three candidates instead of solving a binary majority/minority task.
#   - Example: if a seed interior contains mostly blue, some red, and one green cell, then its branch becomes blue,
#     its frame becomes green, and the red cells remain unchanged. Under the original rule the frame would also become
#     blue, so the fusion output differs exactly on the frame pixels.
# - b) Complexity calibration:
#   - The scene keeps the canonical family: one central 3x3 hub, thin orthogonal P branches, and edge-anchored
#     rectangular seeds. Only the seed color statistics and the border recolor rule change.
#   - To avoid overcomplicating the palette, the generator uses one global pool of four seed colors and gives each seed
#     exactly three of them. That is enough to create genuine multi-candidate counting while still yielding repetition
#     across seeds and across samples.
#   - Every seed is forced to have a unique majority and a unique minority. This prevents ambiguous ties and makes the
#     border-vs-branch contrast visually obvious.
# - c) Hidden-assumption audit:
#   - The original solver's P-color inference still holds because the input still contains one dominant network-like
#     non-background color.
#   - Hub detection as the largest solid P square still holds because the hub remains a clean 3x3 block.
#   - The original recolor step assumed every non-hub P pixel could be labeled together because border and branch shared
#     one output color. That assumption fails here. The revised solver reconstructs each seed frame from the seed bbox,
#     removes those frame cells from the branch mask, then labels only the remaining branch chains.
#   - The new solver assumes that any non-hub P cell adjacent to a seed interior is frame, not free branch. The
#     generator preserves the original path-carving safety rule, so branches never run orthogonally beside seed
#     interiors except by ending on their designated frame cells.
# - d) Placement legality:
#   - Seeds are still placed as non-overlapping edge rectangles with a one-cell P frame on every non-edge side. Their
#     expanded border zones are separated by at least one cell and kept away from the hub.
#   - Paths are carved on a work canvas where seed interiors and existing P cells are obstacles. A candidate path may
#     touch only its chosen destination frame, so branches cannot cross seeds, overlap each other, or merge outside the
#     hub.
# - e) Visual observability:
#   - Humans can directly compare three layers around every seed: unchanged multi-color interior, minority-colored
#     frame, and majority-colored branch. Because the third color remains inside the seed, the output visibly encodes a
#     three-way ranking rather than a simple two-color split.
#   - The generator rejects samples unless every seed uses three colors and the fusion output differs from the original
#     majority-frame baseline. That keeps the intended behavior active in the sample family.
# - f) Visual cleanliness:
#   - No new object types are introduced. The changed roles are spatially separated: the interior stays inside the
#     rectangle, the frame is a one-pixel shell, and the branch is the external chain. This keeps the grid legible.
# - g) Inferability:
#   - No rule depends on numeric color ids. The only decisions are visible frequency comparisons inside each seed and
#     visible structural roles outside it. Using exactly three colors per seed keeps the counting task nontrivial but
#     still understandable from 2-4 examples.

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UTIL_PATH = ROOT / "solvers" / "arc_utils.py"
SPEC = importlib.util.spec_from_file_location("arc_utils_combo_800d221b_18", UTIL_PATH)
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


def _largest_p_square_hub(g: np.ndarray, path_color: int) -> tuple[int, int, int, np.ndarray]:
    H, W = g.shape
    mask = g == int(path_color)
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
    hub[int(r0) : int(r0 + k), int(c0) : int(c0 + k)] = True
    return int(k), int(r0), int(c0), hub


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if r0 > 0:
        cells.extend((int(r0 - 1), int(c)) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        cells.extend((int(r1 + 1), int(c)) for c in range(c0, c1 + 1))
    if c0 > 0:
        cells.extend((int(r), int(c0 - 1)) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        cells.extend((int(r), int(c1 + 1)) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        cells.append((int(r0 - 1), int(c0 - 1)))
    if r0 > 0 and c1 < W - 1:
        cells.append((int(r0 - 1), int(c1 + 1)))
    if r1 < H - 1 and c0 > 0:
        cells.append((int(r1 + 1), int(c0 - 1)))
    if r1 < H - 1 and c1 < W - 1:
        cells.append((int(r1 + 1), int(c1 + 1)))
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


def _unique_extreme_color(counts: Counter[int], *, choose_max: bool) -> int:
    assert counts, "expected a non-empty color counter"
    target = max(int(v) for v in counts.values()) if choose_max else min(int(v) for v in counts.values())
    winners = [int(col) for col, cnt in counts.items() if int(cnt) == int(target)]
    kind = "majority" if choose_max else "minority"
    assert len(winners) == 1, f"seed must have a unique {kind} color"
    return int(winners[0])


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    path_color = int(_find_path_color(g, bg))
    hub_size, hub_r0, hub_c0, hub = _largest_p_square_hub(g, path_color)

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 1, "grid must contain seed regions"

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

        rect = seed_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert bool(np.all(rect)), "seed interior must be a solid rectangle"

        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        assert len(counts) >= 3, "combo expects at least three colors inside each seed"
        majority = _unique_extreme_color(counts, choose_max=True)
        minority = _unique_extreme_color(counts, choose_max=False)
        middle_colors = sorted(int(col) for col in counts if int(col) not in {int(majority), int(minority)})
        assert middle_colors, "seed must contain at least one non-extreme interior color"

        border = _seed_border_cells(r0, c0, r1, c1, H, W)
        assert border, "seed must expose at least one border cell"
        for br, bc in border:
            assert int(g[br, bc]) == int(path_color), "seed border must be P-colored in the input"
            assert int(border_owner[br, bc]) == 0, "seed borders must not overlap"
            border_owner[br, bc] = int(sid)

        info = {
            "id": int(sid),
            "pts": pts,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "counts": {int(col): int(cnt) for col, cnt in counts.items()},
            "majority": int(majority),
            "minority": int(minority),
            "middle_colors": middle_colors,
            "border": border,
            "color_count": int(len(counts)),
        }
        seed_infos.append(info)
        seed_by_id[int(sid)] = info

    seed_border_mask = border_owner > 0
    non_hub_p = (g == int(path_color)) & (~hub)
    branch_mask = non_hub_p & (~seed_border_mask)
    assert np.array_equal(non_hub_p, branch_mask | seed_border_mask), "every non-hub P cell must be branch or seed border"

    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    branch_to_seed: dict[int, int] = {}
    branch_to_majority: dict[int, int] = {}
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = _touch_seed_borders(pts, border_owner)
        assert touches, "every branch must touch a seed border"
        assert len(touches) == 1, "each branch component must belong to exactly one seed"
        sid = int(next(iter(touches)))
        branch_to_seed[int(bid)] = int(sid)
        branch_to_majority[int(bid)] = int(seed_by_id[int(sid)]["majority"])

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
        "path_color": int(path_color),
        "H": int(H),
        "W": int(W),
        "hub_size": int(hub_size),
        "hub_r0": int(hub_r0),
        "hub_c0": int(hub_c0),
        "hub": hub,
        "seed_infos": seed_infos,
        "seed_by_id": seed_by_id,
        "seed_border_mask": seed_border_mask,
        "border_owner": border_owner,
        "branch_mask": branch_mask,
        "branch_lab": branch_lab,
        "branch_n": int(branch_n),
        "branch_to_seed": branch_to_seed,
        "branch_to_majority": branch_to_majority,
    }


def _render_output(grid: np.ndarray, scene: dict, *, border_to_minority: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()

    for seed in scene["seed_infos"]:
        border_color = int(seed["minority"] if border_to_minority else seed["majority"])
        for r, c in seed["border"]:
            out[int(r), int(c)] = int(border_color)

    for bid in range(1, int(scene["branch_n"]) + 1):
        branch_color = int(scene["branch_to_majority"][int(bid)])
        pts = np.argwhere(scene["branch_lab"] == int(bid))
        for r, c in pts:
            out[int(r), int(c)] = int(branch_color)

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


def _original_majority_transform(grid: np.ndarray) -> np.ndarray:
    scene = _analyze_scene(grid)
    return _render_output(grid, scene, border_to_minority=False)


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, object]:
    scene = _analyze_scene(inp)
    baseline = _original_majority_transform(inp)

    color_cardinalities: list[int] = []
    middle_color_cells = 0
    border_minority_cells = 0
    branch_majority_cells = 0
    branch_border_contrast = 0
    majority_palette: set[int] = set()
    minority_palette: set[int] = set()

    for seed in scene["seed_infos"]:
        sid = int(seed["id"])
        majority = int(seed["majority"])
        minority = int(seed["minority"])
        majority_palette.add(int(majority))
        minority_palette.add(int(minority))
        color_cardinalities.append(int(seed["color_count"]))
        for col in seed["middle_colors"]:
            middle_color_cells += int(seed["counts"][int(col)])
        for r, c in seed["border"]:
            if int(out[int(r), int(c)]) == int(minority):
                border_minority_cells += 1
        seed_branch_majority_cells = 0
        for bid in range(1, int(scene["branch_n"]) + 1):
            if int(scene["branch_to_seed"][int(bid)]) != int(sid):
                continue
            pts = np.argwhere(scene["branch_lab"] == int(bid))
            for r, c in pts:
                if int(out[int(r), int(c)]) == int(majority):
                    branch_majority_cells += 1
                    seed_branch_majority_cells += 1
        if int(majority) != int(minority) and seed_branch_majority_cells > 0:
            branch_border_contrast += len(seed["border"])

    return {
        "seed_count": int(len(scene["seed_infos"])),
        "seed_color_cardinalities": color_cardinalities,
        "multicolor_seed_count": int(sum(1 for k in color_cardinalities if int(k) >= 3)),
        "middle_color_cells": int(middle_color_cells),
        "border_minority_cells": int(border_minority_cells),
        "branch_majority_cells": int(branch_majority_cells),
        "branch_border_contrast": int(branch_border_contrast),
        "majority_palette_size": int(len(majority_palette)),
        "minority_palette_size": int(len(minority_palette)),
        "output_diff_from_baseline": int(np.count_nonzero(out != baseline)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint a hub-and-branch path network using ranked multi-color seeds.

    Input structure:
    - The background is the most frequent color.
    - One non-background color P forms a thin orthogonal network with a solid square hub.
    - Each seed is a rectangular interior touching a grid edge, wrapped by a one-pixel P frame on every non-edge side.
    - Unlike the base task, every seed interior contains at least three colors with a unique most frequent color and a
      unique least frequent color.

    Rule:
    - Infer P, recover the largest solid all-P square as the hub, and reconstruct each seed frame from its rectangular
      interior.
    - For every seed, count interior colors across all visible seed colors, not just two of them.
    - Recolor each attached branch to that seed's majority color, but recolor that seed's P frame to its minority
      color.
    - Leave seed interiors unchanged, then recolor only the hub center pixel to the branch color that is most common
      among hub-adjacent recolored branch cells.

    Key insight:
    The same seed now yields two different output colors. The branch follows the seed's most frequent interior color,
    while the surrounding frame follows the least frequent interior color, so the solver must separate frame pixels from
    true branch pixels and must rank three or more seed colors correctly.
    """
    g = np.array(grid, dtype=int)
    scene = _analyze_scene(g)
    return _render_output(g, scene, border_to_minority=True)


# --------------- generator (combo-aware attempt) ---------------

GRID_SIZE_BASE = 11
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30

HUB_SIZE = 3

SEED_COUNT_CHOICES = [3, 4, 5]
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
SEED_PLACEMENT_RETRIES = 400
SEED_ZONE_GAP = 1
SEED_COLOR_POOL_SIZE = 4

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_PATHS_PER_SEED = 3
PATH_FIRST_ROUND_RETRIES = 60
PATH_EXTRA_ROUND_RETRIES = 30
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 300


def _try_carve_path(
    canvas: np.ndarray,
    path_color: int,
    start: tuple[int, int],
    dest_border_cells: list[tuple[int, int]],
    max_bends: int = MAX_BENDS,
):
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
        return (int(ir0), max(0, int(ic0 - 1)), int(ir1 + 1), min(W - 1, int(ic1 + 1)))
    if edge == "bottom":
        return (max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), int(ir1), min(W - 1, int(ic1 + 1)))
    if edge == "left":
        return (max(0, int(ir0 - 1)), int(ic0), min(H - 1, int(ir1 + 1)), int(ic1 + 1))
    return (max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), min(H - 1, int(ir1 + 1)), int(ic1))


def _place_seed_border(
    work: np.ndarray,
    path_color: int,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    H: int,
    W: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
                work[int(r), int(c)] = int(path_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (int(ir0 - 1), int(ic0 - 1))),
        (ir0 > 0 and ic1 < W - 1, (int(ir0 - 1), int(ic1 + 1))),
        (ir1 < H - 1 and ic0 > 0, (int(ir1 + 1), int(ic0 - 1))),
        (ir1 < H - 1 and ic1 < W - 1, (int(ir1 + 1), int(ic1 + 1))),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(path_color)
            border_cells.append((int(rc[0]), int(rc[1])))
    return border_cells, landing_cells


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge: str, H: int, W: int) -> tuple[int, int, int, int]:
    while True:
        ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        if ih * iw >= 6:
            break
    if edge == "top":
        ir0, ic0 = 0, random.randint(0, W - iw)
    elif edge == "bottom":
        ir0, ic0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        ir0, ic0 = random.randint(0, H - ih), 0
    else:
        ir0, ic0 = random.randint(0, H - ih), W - iw
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _strict_three_way_counts(n_cells: int) -> tuple[int, int, int]:
    assert n_cells >= 6, "need at least six cells for unique major/mid/minor counts"
    for _ in range(200):
        n_minor = random.randint(1, max(1, n_cells // 4))
        max_mid = (n_cells - n_minor - 1) // 2
        if max_mid <= n_minor:
            continue
        n_mid = random.randint(n_minor + 1, max_mid)
        n_major = n_cells - n_minor - n_mid
        if n_major > n_mid > n_minor >= 1:
            return int(n_major), int(n_mid), int(n_minor)
    raise RuntimeError("failed to choose strict three-way counts")


def _fill_seed_interior(
    work: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    major: int,
    middle: int,
    minor: int,
) -> dict[int, int]:
    interior_cells = [(int(r), int(c)) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    random.shuffle(interior_cells)
    n_major, n_middle, n_minor = _strict_three_way_counts(len(interior_cells))
    counts = {int(major): int(n_major), int(middle): int(n_middle), int(minor): int(n_minor)}

    idx = 0
    for color, n_use in [(int(minor), n_minor), (int(middle), n_middle), (int(major), n_major)]:
        for r, c in interior_cells[idx : idx + n_use]:
            work[int(r), int(c)] = int(color)
        idx += n_use
    assert idx == len(interior_cells)
    return counts


def _carve_all_paths(work: np.ndarray, path_color: int, hub_perimeter: list[tuple[int, int]], seeds: list[dict]) -> bool:
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

    colors = random.sample(list(range(10)), 2 + SEED_COLOR_POOL_SIZE)
    bg = int(colors[0])
    path_color = int(colors[1])
    seed_palette = [int(c) for c in colors[2:]]

    g = np.full((H, W), int(bg), dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[int(hr0) : int(hr0 + HUB_SIZE), int(hc0) : int(hc0 + HUB_SIZE)] = int(path_color)

    hub_perimeter = [
        (int(r), int(c))
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    work = np.where(g == int(bg), 0, g).astype(int)
    seeds = []
    used_zones: list[tuple[int, int, int, int]] = []
    edges = ["top", "bottom", "left", "right"]

    majority_plan = [seed_palette[i % len(seed_palette)] for i in range(n_seeds_target)]
    random.shuffle(majority_plan)

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

        major = int(majority_plan[len(seeds)])
        others = [c for c in seed_palette if int(c) != int(major)]
        middle, minor = random.sample(others, 2)
        counts = _fill_seed_interior(work, ir0, ic0, ir1, ic1, int(major), int(middle), int(minor))
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1, H, W)

        used_zones.append(border_zone)
        seeds.append(
            {
                "majority": int(major),
                "middle": int(middle),
                "minority": int(minor),
                "counts": counts,
                "border_cells": border_cells,
                "landing_cells": landing_cells,
                "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                "border_zone": border_zone,
            }
        )

    if len(seeds) < 3:
        return None

    if not _carve_all_paths(work, path_color, hub_perimeter, seeds):
        return None

    inp = np.where(work == 0, int(bg), work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        out = main(inp)
        metrics = _fusion_metrics(inp, out)
    except Exception:
        return None

    if int(metrics["multicolor_seed_count"]) != len(seeds):
        return None
    if any(int(k) < 3 for k in metrics["seed_color_cardinalities"]):
        return None
    if int(metrics["middle_color_cells"]) <= 0:
        return None
    if int(metrics["border_minority_cells"]) <= 0:
        return None
    if int(metrics["branch_majority_cells"]) <= 0:
        return None
    if int(metrics["branch_border_contrast"]) <= 0:
        return None
    if int(metrics["majority_palette_size"]) < 2:
        return None
    if int(metrics["minority_palette_size"]) < 2:
        return None
    if int(metrics["output_diff_from_baseline"]) <= 0:
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

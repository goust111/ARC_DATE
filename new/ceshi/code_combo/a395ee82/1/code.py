# Combo variant of a395ee82 (index 1)
# combo_spec: D3/1+D5/1+D14/1
# Name: 孔洞翻转消失

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis
# ---------------
# a) Concrete behavior / trace
# - Backbone preserved from the base task:
#   input still contains one large template component in color T plus a detached
#   lattice of singleton pixels, with exactly one T-colored singleton marking the
#   lattice origin and all remaining singletons colored O.
# - D3/1 changes the copied object itself: the template is now a holey mask inside
#   a 3x3 or 4x3 bbox. Example 4x3 mask:
#     ###
#     #.#
#     ###
#     ##.
#   The enclosed '.' is a real hole and must stay bg in every copy.
# - D14/1 changes the per-copy geometry: for a lattice delta (-1,0), the copy is
#   vertically flipped before placement; for (0,-1), horizontally flipped; for
#   (-1,-1), both flips apply. Right/down copies keep the original orientation.
# - D5/1 changes overlap semantics and needs a supporting stride change: copies are
#   placed with step (h-1, w-1), not (h, w). This makes adjacent copies overlap by
#   one row/column band. Any pixel covered by 2+ copies disappears back to bg.
# - Combined trace on the fixed witness lattice used by the generator:
#   deltas {(-1,-1), (-1,0), (0,-1), (0,0), (0,1), (1,0)} create five visible
#   direction cases around the origin. Because the template is holey and asymmetric,
#   the reflected overlaps carve different bg notches on the left/up/upper-left
#   sides than a no-flip system would.
#
# b) Complexity calibration
# - If we kept the original stride (h,w), D5 would often be invisible because copies
#   would merely touch or separate. The reduced stride is therefore mandatory.
# - If the holey template were symmetric under H/V reflection, D14 would be visually
#   inert. We therefore enumerate all legal holey masks and keep only those whose
#   H, V, and HV reflections all change the silhouette.
# - If the lattice omitted left/up/upper-left witnesses, humans would not see all
#   reflection cases in 2-4 examples. The generator uses a fixed 6-point connected
#   lattice that always exhibits left, up, upper-left, right, and down relations.
# - The template space is intentionally small and repeated: 3x3 or 4x3 bboxes, 1-2
#   hole pixels, and the same directional lattice across samples. This keeps the
#   rule inferable instead of exploding the hypothesis space.
#
# c) Implicit-assumption review of the original main()
# - Original `templ_rel` assumed a solid or at least hole-agnostic foreground copy.
#   In the combo, we must stamp the exact filled mask only; bbox filling is wrong.
# - Original `off_r = dr * h`, `off_c = dc * w` assumed no overlap. That breaks D5;
#   we must switch to `(h-1, w-1)` to make cancellation observable.
# - Original reuse of one `templ_rel` for every copy assumed all copies had the same
#   orientation. D14 breaks that; we now precompute four reflected masks.
# - Original `out[total >= 2] = overlap_col` encoded "overlap becomes O". D5 changes
#   this to "overlap disappears", so overlapping cells must remain bg instead.
# - New parser assumption: the largest monochromatic non-bg component is still the
#   template. The generator enforces large size separation so singleton components
#   never compete with it.
#
# d) Placement legality
# - All output copies are placed fully in-bounds using the reduced stride.
# - The singleton cluster bbox stays disjoint from the input template bbox with at
#   least one bg cell gap, preserving the original two-object-family scene.
# - Singletons are also kept off the full stamp footprint (before cancellation), so
#   no input marker sits on a would-be output cell.
#
# e) Visual observability
# - Every sample contains:
#   one enclosed hole, visible cancellation bands, and explicit left/up/upper-left
#   reflection witnesses around the origin.
# - The generator rejects templates where the reflected-interference output collapses
#   to the same result as the non-reflected interference output.
# - No color-id ordering or hidden metadata is used; all cues are geometric or the
#   already-signaled T/O color roles.
#
# f) Visual cleanliness
# - Only three colors remain active: bg, T, O.
# - The lattice is a sparse singleton cluster and the template is a single component.
# - Although copies overlap in the output, overlap depth is limited by the tiny fixed
#   lattice, so the result shows clear cut-out bands instead of dense noise.
#
# g) Color / rule inferability
# - O still has one structural role: "origin-only visible pixels".
# - T still marks visible non-origin singleton coverage.
# - bg now has two readable structural roles: outside space and destructive overlap.
#   Because those bg holes form aligned interference bands, humans can infer that
#   disappearance depends on overlap, not on arbitrary recoloring.
#
# h) New main() assumptions introduced by the combo
# - The template must contain 1-2 enclosed hole pixels.
# - The lattice spacing must be uniform and nonzero so gcd-based normalization works.
# - At least one origin pixel must remain uniquely visible after cancellation, or O
#   would vanish from the output. The generator explicitly filters for this.


GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
TEMPLATE_HOLE_PIXELS = (1, 2)
GUNIT = 2
LATTICE_DELTAS = [(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
MIN_CANCEL_CELLS = 7
MIN_ORIGIN_VISIBLE = 1
MIN_TEMPLATE_VISIBLE = 12
MIN_DIRECTION_VISIBLE = 2
MIN_FLIP_GAP = 7
MIN_CANCEL_GAP = 7

SINGLETON_RETRIES = 80
GENERATE_RETRIES = 400


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _cells_to_mask(cells: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _enclosed_bg_mask(filled_mask: np.ndarray) -> np.ndarray:
    """Return bbox-internal bg cells that are not 4-connected to the bbox edge."""
    h, w = filled_mask.shape
    bg_mask = ~filled_mask.astype(bool)
    holes = np.zeros((h, w), dtype=bool)
    for coords in _mask_components(bg_mask):
        if coords.size == 0:
            continue
        touches_edge = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in coords
        )
        if touches_edge:
            continue
        holes[coords[:, 0], coords[:, 1]] = True
    return holes


def _stamp_steps(h: int, w: int) -> tuple[int, int]:
    """Reduced stride that creates one-cell interference bands between neighbors."""
    return max(1, int(h) - 1), max(1, int(w) - 1)


def _transform_cells(
    cells: list[tuple[int, int]], h: int, w: int, flip_v: bool, flip_h: bool
) -> list[tuple[int, int]]:
    """Reflect relative template cells inside their h x w bbox."""
    out: list[tuple[int, int]] = []
    for rr, cc in cells:
        nr = int(h - 1 - rr) if flip_v else int(rr)
        nc = int(w - 1 - cc) if flip_h else int(cc)
        out.append((int(nr), int(nc)))
    return out


def _transformed_lookup(
    cells: list[tuple[int, int]], h: int, w: int
) -> dict[tuple[bool, bool], list[tuple[int, int]]]:
    return {
        (flip_v, flip_h): _transform_cells(cells, h, w, flip_v, flip_h)
        for flip_v in (False, True)
        for flip_h in (False, True)
    }


def _relative_stamp_sets(
    transformed: dict[tuple[bool, bool], list[tuple[int, int]]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
    *,
    reflect: bool,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Return per-delta stamp cells in a shared relative coordinate system."""
    step_r, step_c = _stamp_steps(h, w)
    out: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for dr, dc in deltas:
        key = (bool(dr < 0) and reflect, bool(dc < 0) and reflect)
        pts = {
            (int(dr * step_r + rr), int(dc * step_c + cc))
            for rr, cc in transformed[key]
        }
        out[(int(dr), int(dc))] = pts
    return out


def _coverage_counts(
    stamp_sets: dict[tuple[int, int], set[tuple[int, int]]]
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    total: dict[tuple[int, int], int] = {}
    for pts in stamp_sets.values():
        for pt in pts:
            total[pt] = int(total.get(pt, 0)) + 1
    origin_pts = set(stamp_sets[(0, 0)])
    return total, origin_pts


def _render_relative(
    templ_col: int,
    overlap_col: int,
    total: dict[tuple[int, int], int],
    origin_pts: set[tuple[int, int]],
    *,
    cancel_overlaps: bool,
) -> dict[tuple[int, int], int]:
    rendered: dict[tuple[int, int], int] = {}
    for pt, count in total.items():
        if count == 1:
            rendered[pt] = int(overlap_col) if pt in origin_pts else int(templ_col)
        elif not cancel_overlaps:
            rendered[pt] = int(overlap_col)
    return rendered


def _combo_stats(cells: list[tuple[int, int]], h: int, w: int) -> dict[str, int]:
    transformed = _transformed_lookup(cells, h, w)
    combo_sets = _relative_stamp_sets(transformed, h, w, LATTICE_DELTAS, reflect=True)
    noflip_sets = _relative_stamp_sets(transformed, h, w, LATTICE_DELTAS, reflect=False)

    total_combo, origin_combo = _coverage_counts(combo_sets)
    total_noflip, origin_noflip = _coverage_counts(noflip_sets)

    combo_render = _render_relative(1, 2, total_combo, origin_combo, cancel_overlaps=True)
    noflip_render = _render_relative(1, 2, total_noflip, origin_noflip, cancel_overlaps=True)
    nocancel_render = _render_relative(1, 2, total_combo, origin_combo, cancel_overlaps=False)

    def _visible_for(delta: tuple[int, int]) -> int:
        return int(sum(total_combo[pt] == 1 for pt in combo_sets[delta]))

    keys_flip = set(combo_render) | set(noflip_render)
    keys_cancel = set(combo_render) | set(nocancel_render)
    return {
        "cancel_cells": int(sum(count >= 2 for count in total_combo.values())),
        "origin_visible": int(sum((count == 1) and (pt in origin_combo) for pt, count in total_combo.items())),
        "template_visible": int(sum((count == 1) and (pt not in origin_combo) for pt, count in total_combo.items())),
        "left_visible": _visible_for((0, -1)),
        "up_visible": _visible_for((-1, 0)),
        "upleft_visible": _visible_for((-1, -1)),
        "flip_gap": int(sum(combo_render.get(pt) != noflip_render.get(pt) for pt in keys_flip)),
        "cancel_gap": int(sum(combo_render.get(pt) != nocancel_render.get(pt) for pt in keys_cancel)),
    }


def _template_candidates(h: int, w: int) -> list[list[tuple[int, int]]]:
    """Enumerate legal holey templates that make the full combo visually observable."""
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))

    valid: list[list[tuple[int, int]]] = []
    for bits in range(1, 1 << total):
        cells = [(idx // w, idx % w) for idx in range(total) if (bits >> idx) & 1]
        area = len(cells)
        if area < lo or area > hi:
            continue

        mask = _cells_to_mask(cells, h, w)
        if len(_mask_components(mask)) != 1:
            continue
        if not mask[0, :].any() or not mask[h - 1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, w - 1].any():
            continue

        hole_pixels = int(_enclosed_bg_mask(mask).sum())
        if hole_pixels not in TEMPLATE_HOLE_PIXELS:
            continue

        transformed = _transformed_lookup(cells, h, w)
        base = set(map(tuple, cells))
        if any(set(transformed[key]) == base for key in [(False, True), (True, False), (True, True)]):
            continue

        stats = _combo_stats(cells, h, w)
        if stats["cancel_cells"] < MIN_CANCEL_CELLS:
            continue
        if stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
            continue
        if stats["template_visible"] < MIN_TEMPLATE_VISIBLE:
            continue
        if stats["left_visible"] < MIN_DIRECTION_VISIBLE:
            continue
        if stats["up_visible"] < MIN_DIRECTION_VISIBLE:
            continue
        if stats["upleft_visible"] < MIN_DIRECTION_VISIBLE:
            continue
        if stats["flip_gap"] < MIN_FLIP_GAP:
            continue
        if stats["cancel_gap"] < MIN_CANCEL_GAP:
            continue

        valid.append([tuple(map(int, rc)) for rc in cells])
    return valid


TEMPLATE_LIBRARY = {
    h: _template_candidates(h, TEMPLATE_W)
    for h in TEMPLATE_H_CHOICES
}


def _coverage_maps(
    shape: tuple[int, int],
    base_r: int,
    base_c: int,
    transformed: dict[tuple[bool, bool], list[tuple[int, int]]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    H, W = map(int, shape)
    step_r, step_c = _stamp_steps(h, w)
    total = np.zeros((H, W), dtype=int)
    origin = np.zeros((H, W), dtype=int)
    for dr, dc in deltas:
        stamp_cells = transformed[(int(dr) < 0, int(dc) < 0)]
        off_r = int(base_r + dr * step_r)
        off_c = int(base_c + dc * step_c)
        is_origin = int(dr) == 0 and int(dc) == 0
        for rr, cc in stamp_cells:
            R = int(off_r + rr)
            C = int(off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    origin[R, C] += 1
    return total, origin


def _render_output(
    bg: int,
    templ_col: int,
    overlap_col: int,
    total: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    out = np.full(total.shape, int(bg), dtype=int)
    out[(total == 1) & (origin == 0)] = int(templ_col)
    out[(total == 1) & (origin > 0)] = int(overlap_col)
    return out


def _footprint_points(
    base_r: int,
    base_c: int,
    transformed: dict[tuple[bool, bool], list[tuple[int, int]]],
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    step_r, step_c = _stamp_steps(h, w)
    pts: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        stamp_cells = transformed[(int(dr) < 0, int(dc) < 0)]
        for rr, cc in stamp_cells:
            pts.add((int(base_r + dr * step_r + rr), int(base_c + dc * step_c + cc)))
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp reflected copies of a holey template from a singleton lattice, canceling overlaps to bg.

    Input structure:
    - bg is the dominant background color.
    - One large monochrome template component of color T appears once; inside its 3x3 or 4x3
      bbox, 1-2 enclosed bg cells may exist as true holes.
    - A detached lattice of singleton pixels appears elsewhere: exactly one singleton is also T
      and marks the origin, while the remaining singletons share color O.

    Rule:
    - Recover coarse lattice deltas from every singleton relative to the unique T-colored origin.
    - Stamp the template's filled mask, not its full bbox, using stride (template height - 1,
      template width - 1) so neighboring copies overlap.
    - Flip a copy vertically when its lattice point is above the origin, horizontally when it is
      left of the origin, and apply both flips for upper-left points.
    - Filled pixels covered exactly once stay visible; origin-only visible pixels become O,
      non-origin visible pixels stay T, and any overlap pixel covered by two or more copies
      disappears back to bg.

    Key insight: Three cues must be fused at once: the copied object is a holey mask, the sign of
    each lattice offset selects a reflection, and overlap no longer recolors cells but erases them.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_col = int(template["color"])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))

    templ_mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(templ_col)
    hole_pixels = int(_enclosed_bg_mask(templ_mask).sum())
    assert hole_pixels in TEMPLATE_HOLE_PIXELS

    templ_rel = [tuple(map(int, rc)) for rc in np.argwhere(templ_mask)]
    transformed = _transformed_lookup(templ_rel, h, w)

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template or int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(comp["color"])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    deltas_pix = [(int(r - orr), int(c - orc)) for (r, c, _col) in singletons]
    gcd_vals = [abs(int(x)) for dr, dc in deltas_pix for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit)) for dr, dc in deltas_pix]

    total, origin_cover = _coverage_maps((H, W), r0, c0, transformed, h, w, deltas)
    return _render_output(bg, templ_col, overlap_col, total, origin_cover)


def _place_singletons(
    r0: int,
    c0: int,
    h: int,
    w: int,
    templ_cells: list[tuple[int, int]],
    bg: int,
    templ_col: int,
    overlap_col: int,
) -> np.ndarray | None:
    transformed = _transformed_lookup(templ_cells, h, w)
    occupied = _footprint_points(r0, c0, transformed, h, w, LATTICE_DELTAS)

    min_dr = min(dr for dr, _dc in LATTICE_DELTAS)
    max_dr = max(dr for dr, _dc in LATTICE_DELTAS)
    min_dc = min(dc for _dr, dc in LATTICE_DELTAS)
    max_dc = max(dc for _dr, dc in LATTICE_DELTAS)

    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * GUNIT
        sr1 = orr + max_dr * GUNIT
        sc0 = orc + min_dc * GUNIT
        sc1 = orc + max_dc * GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((int(orr + dr * GUNIT), int(orc + dc * GUNIT)) in occupied for dr, dc in LATTICE_DELTAS):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(templ_col)
        for dr, dc in LATTICE_DELTAS:
            rr = int(orr + dr * GUNIT)
            cc = int(orc + dc * GUNIT)
            grid[rr, cc] = int(templ_col) if (dr, dc) == (0, 0) else int(overlap_col)

        total, origin = _coverage_maps((GRID_SIZE, GRID_SIZE), r0, c0, transformed, h, w, LATTICE_DELTAS)
        expected = _render_output(bg, templ_col, overlap_col, total, origin)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    return None


def _gen_template() -> tuple[int, int, list[tuple[int, int]]] | None:
    valid_heights = [h for h in TEMPLATE_H_CHOICES if TEMPLATE_LIBRARY[h]]
    if not valid_heights:
        return None
    h = random.choice(valid_heights)
    cells = random.choice(TEMPLATE_LIBRARY[h])
    return h, TEMPLATE_W, cells


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    templ = _gen_template()
    if templ is None:
        return None
    h, w, templ_cells = templ

    transformed = _transformed_lookup(templ_cells, h, w)
    total, origin = _coverage_maps((GRID_SIZE, GRID_SIZE), h, w, transformed, h, w, LATTICE_DELTAS)
    out_stats = {
        "cancel_cells": int(np.sum(total >= 2)),
        "origin_visible": int(np.sum((total == 1) & (origin > 0))),
        "template_visible": int(np.sum((total == 1) & (origin == 0))),
    }
    if out_stats["cancel_cells"] < MIN_CANCEL_CELLS:
        return None
    if out_stats["origin_visible"] < MIN_ORIGIN_VISIBLE:
        return None
    if out_stats["template_visible"] < MIN_TEMPLATE_VISIBLE:
        return None

    step_r, step_c = _stamp_steps(h, w)
    min_dr = min(dr for dr, _dc in LATTICE_DELTAS)
    max_dr = max(dr for dr, _dc in LATTICE_DELTAS)
    min_dc = min(dc for _dr, dc in LATTICE_DELTAS)
    max_dc = max(dc for _dr, dc in LATTICE_DELTAS)

    r0_lo = -min_dr * step_r
    r0_hi = GRID_SIZE - max_dr * step_r - h
    c0_lo = -min_dc * step_c
    c0_hi = GRID_SIZE - max_dc * step_c - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    return _place_singletons(r0, c0, h, w, templ_cells, bg, templ_col, overlap_col)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")

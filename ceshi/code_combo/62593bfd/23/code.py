# Combo variant of 62593bfd (index 23)
# combo_spec: D1/0+D11/3
# Name: 面积分组 + 按面积排序逐对象处理

"""
Design analysis for combo D1/0 + D11/3
======================================

fused_rule:
  - Objects are still the 8-connected foreground components.
  - The grouping feature is no longer D4-canonical shape; it is object area.
  - Objects are processed one by one in descending area order.
  - To avoid hidden dependence on component scan order, same-area ties are broken by
    visible horizontal position: smaller bbox-left column c0 is processed first.
    If c0 also ties, the visible top row r0 is the final tie-breaker.
  - During that ordered scan:
      * first time an area appears -> move that object to the BOTTOM
      * later objects of the same area -> move to the TOP
    So area is both the sort key and the "seen before?" key; this is the intended
    self-referential interaction of the combo.

behavior_trace:
  small_example:
    objects:
      - A: area 9, c0=2, shape=L
      - B: area 9, c0=11, shape=T
      - C: area 6, c0=5, shape=zigzag
      - D: area 4, c0=14, shape=bar
    original_task:
      - group by D4 shape frequency
      - if all four shapes are different, all four go to the bottom
    D1-only area-grouping:
      - area 9 appears twice, so A and B both go to the top
      - C and D go to the bottom
    fused_combo:
      - sort by area desc, tie by c0: A(9,c0=2), B(9,c0=11), C(6), D(4)
      - A is first area-9 object, so area 9 is not yet seen -> A goes bottom
      - B sees area 9 already processed -> B goes top
      - C is first area-6 object -> bottom
      - D is first area-4 object -> bottom
    visible_difference:
      - compared with the source task, B rises even though no matching shape exists
      - compared with D1-only, one member of the repeated-area pair stays at bottom
        because it is the leftmost / first processed representative of that area

complexity_calibration:
  - The rule is still inferable from 2-4 ARC examples because:
      * there are only a few area categories
      * repeated-area groups are supported by 2 or 3 objects
      * same-area objects are intentionally different in silhouette, so the solver
        cannot fall back to "same shape means top"
      * the leftmost object within a repeated-area category consistently behaves as
        the bottom representative, making the order dependency visually observable
  - It is not over-simplified: the output is not just "all repeated areas go top".
    The first representative of each area remains at the bottom, so examples show
    a mix inside the same area category.

implicit_assumption_audit_main:
  - find_components(..., monochromatic=False, connectivity=8):
      assumes different objects never touch even diagonally. The generator enforces
      8-neighborhood separation in both input and output.
  - patch extraction from bbox:
      still valid because every component remains a whole object.
  - replacing shape_key with area:
      valid because the new grouping primitive is len(coords).
  - introducing ordered processing:
      requires a deterministic visible order. We use (-area, c0, r0), not hidden
      discovery order, so the rule is reproducible for humans and code alike.
  - paste():
      still assumes destination placements do not overlap. The generator constructs
      output first and checks every placement with the same no-touch predicate.

placement_legality:
  - Output-first construction is retained.
  - Repeated-area groups are placed as constrained batches:
      * one designated representative on the bottom row
      * all other members of that area on the top row
      * top members must have c0 strictly greater than the representative's c0
    This guarantees the solver's left-to-right order matches the intended sequential
    processing for that area.
  - Unique-area objects are placed on the bottom row.
  - Input scattering preserves each object's c0 and keeps all objects away from the
    edge rows when possible so the translation is visible.

visual_observability:
  - Humans can see that shape no longer controls the move, because repeated-area
    pairs are deliberately non-equivalent under D4.
  - Humans can also see that repeated-area objects do not all behave the same:
    the leftmost one stays at the bottom while later same-area objects rise.
  - No hidden signals such as color value, component discovery order, or numeric
    palette ordering are used in the logic.

visual_tidiness:
  - Objects remain monochromatic and sparse.
  - Top and bottom landing zones are separated by a large empty middle band.
  - 8-neighborhood spacing avoids accidental component merging and keeps silhouettes
    individually legible.

generator_non_degeneracy_metric:
  - We measure `repeated_area_bottom_count`: how many bottom-routed objects belong
    to an area appearing at least twice. In the D1-only variant this value is always 0.
    In this combo it must be >= 1.
  - We also measure `vs_shape_diff_count`: number of cells where this combo's output
    differs from the original shape-based output on the same input. It must be >= 1.

main_new_assumptions_and_generator_guarantees:
  - Assumption: same-area ties need a deterministic order.
    Guarantee: generator gives same-area objects distinct c0, so the c0 tie-break is
    sufficient in practice; r0 is only a robustness fallback.
  - Assumption: the first representative of a repeated area can coexist on the bottom
    with unique-area objects.
    Guarantee: output-first placement validates the exact final arrangement before
    returning a sample.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical D4-invariant shape key, used only by the generator/metrics."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        nrm = _norm(cur)
        best = nrm if best is None or nrm < best else best
        flip = _norm([(-r, c) for r, c in cur])
        best = flip if flip < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < h and 0 <= cc < w:
            mask[rr, cc] = True
    return mask


def _extract_objects(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    objs = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        objs.append(
            {
                "idx": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "pat": pat,
                "area": int(len(rel)),
                "canon": _canonical_shape(rel),
            }
        )
    return objs, bg


def _ordered_routes(objs, h_grid):
    ordered = sorted(objs, key=lambda d: (-int(d["area"]), int(d["c0"]), int(d["r0"])))
    seen_areas = set()
    routes = {}
    for d in ordered:
        h_obj = int(d["pat"].shape[0])
        area = int(d["area"])
        dest_r0 = 0 if area in seen_areas else int(h_grid - h_obj)
        routes[int(d["idx"])] = int(dest_r0)
        seen_areas.add(area)
    return routes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sequentially route objects by area.

    Rule:
    - Find each 8-connected foreground component.
    - Compute area = number of pixels in the component.
    - Process objects in descending area order.
    - Same-area ties are broken by smaller bbox-left column c0, then smaller r0.
    - If an object's area has already appeared among earlier processed objects,
      move it to the TOP (row 0); otherwise move it to the BOTTOM (row H-h).
    - Horizontal position c0 is preserved.
    """
    g = np.array(grid, dtype=int)
    h_grid, w_grid = g.shape
    objs, bg = _extract_objects(g)
    routes = _ordered_routes(objs, h_grid)

    out = np.full((h_grid, w_grid), bg, dtype=int)
    for d in objs:
        paste(out, d["pat"], int(routes[d["idx"]]), int(d["c0"]), transparent=bg)
    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if max(all_r) - min(all_r) < max_dim and max(all_c) - min(all_c) < max_dim:
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _shape_dims(coords):
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _sample_shape(area, *, forbidden_canons):
    for _ in range(400):
        shape = _grow_shape(area, 8)
        canon = _canonical_shape(shape)
        hh, ww = _shape_dims(shape)
        if canon in forbidden_canons:
            continue
        if hh < 2 or ww < 2:
            continue
        return shape, canon
    raise ValueError("shape sampling failed")


def _can_place_without_touch(grid, patch, r0, c0, bg):
    h, w = patch.shape
    h_grid, w_grid = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > h_grid or c0 + w > w_grid:
        return False

    mask = patch != bg
    window = grid[r0 : r0 + h, c0 : c0 + w]
    if np.any(window[mask] != bg):
        return False

    for pr, pc in np.argwhere(mask):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        r1 = max(0, rr - 1)
        r2 = min(h_grid, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(w_grid, cc + 2)
        if np.any(grid[r1:r2, c1:c2] != bg):
            return False
    return True


def _find_place_on_row(grid, patch, row, bg, *, min_c0=None, max_c0=None):
    h_grid, w_grid = grid.shape
    h_obj, w_obj = patch.shape
    row = int(row)
    if row < 0 or row + h_obj > h_grid:
        return None

    lo = 0 if min_c0 is None else max(0, int(min_c0))
    hi = w_grid - w_obj if max_c0 is None else min(w_grid - w_obj, int(max_c0))
    if lo > hi:
        return None

    candidates = list(range(lo, hi + 1))
    random.shuffle(candidates)
    for c0 in candidates:
        if _can_place_without_touch(grid, patch, row, c0, bg):
            return int(c0)
    return None


def _find_scatter_row(grid, patch, c0, bg):
    h_grid, _w_grid = grid.shape
    h_obj, _w_obj = patch.shape
    candidates = [r for r in range(0, h_grid - h_obj + 1) if r not in (0, h_grid - h_obj)]
    random.shuffle(candidates)
    for r0 in candidates:
        if _can_place_without_touch(grid, patch, r0, c0, bg):
            return int(r0)
    return None


def _place_repeated_group(out, group, h_grid, bg):
    bottom_obj = group[0]
    top_objs = list(group[1:])
    bottom_row = int(h_grid - bottom_obj["patch"].shape[0])
    bottom_candidates = list(range(0, out.shape[1] - bottom_obj["patch"].shape[1] + 1))
    random.shuffle(bottom_candidates)

    top_objs.sort(key=lambda d: int(d["patch"].shape[1]), reverse=True)
    for bottom_c0 in bottom_candidates:
        if not _can_place_without_touch(out, bottom_obj["patch"], bottom_row, bottom_c0, bg):
            continue
        trial = out.copy()
        paste(trial, bottom_obj["patch"], bottom_row, bottom_c0, transparent=bg)
        placements = {id(bottom_obj): int(bottom_c0)}
        ok = True
        for obj in top_objs:
            c0 = _find_place_on_row(trial, obj["patch"], 0, bg, min_c0=bottom_c0 + 1)
            if c0 is None:
                ok = False
                break
            paste(trial, obj["patch"], 0, c0, transparent=bg)
            placements[id(obj)] = int(c0)
        if ok:
            out[:, :] = trial
            bottom_obj["c0"] = int(bottom_c0)
            for obj in group[1:]:
                obj["c0"] = int(placements[id(obj)])
            return True
    return False


_CONFIGS = [
    ([2], 1),
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _shape_based_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h_grid, w_grid = g.shape
    objs, bg = _extract_objects(g)
    freq = Counter(obj["canon"] for obj in objs)
    out = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in objs:
        h_obj = int(obj["pat"].shape[0])
        dest_r0 = 0 if int(freq[obj["canon"]]) >= 2 else int(h_grid - h_obj)
        paste(out, obj["pat"], dest_r0, int(obj["c0"]), transparent=bg)
    return out


def _global_area_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h_grid, w_grid = g.shape
    objs, bg = _extract_objects(g)
    freq = Counter(int(obj["area"]) for obj in objs)
    out = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in objs:
        h_obj = int(obj["pat"].shape[0])
        dest_r0 = 0 if int(freq[obj["area"]]) >= 2 else int(h_grid - h_obj)
        paste(out, obj["pat"], dest_r0, int(obj["c0"]), transparent=bg)
    return out


def _fusion_metrics(grid: np.ndarray):
    objs, _bg = _extract_objects(grid)
    h_grid = int(grid.shape[0])
    routes = _ordered_routes(objs, h_grid)
    area_counts = Counter(int(obj["area"]) for obj in objs)
    repeated_area_bottom_count = sum(
        1
        for obj in objs
        if int(area_counts[obj["area"]]) >= 2 and int(routes[obj["idx"]]) != 0
    )
    combo_out = main(grid)
    shape_out = _shape_based_output(grid)
    global_area_out = _global_area_output(grid)
    return {
        "repeated_area_bottom_count": int(repeated_area_bottom_count),
        "vs_shape_diff_count": int(np.sum(combo_out != shape_out)),
        "vs_global_area_diff_count": int(np.sum(combo_out != global_area_out)),
    }


def _gen_attempt():
    h_grid = random.randint(19, 30)
    w_grid = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    duplicated_areas = random.sample(list(range(4, 13)), len(group_sizes))
    remaining_areas = [a for a in range(3, 13) if a not in duplicated_areas]
    unique_areas = random.sample(remaining_areas, n_unique)

    repeated_groups = []
    unique_objs = []

    for gsz, area in sorted(zip(group_sizes, duplicated_areas), key=lambda x: -x[1]):
        used_canons = set()
        group = []
        for j in range(gsz):
            shape, canon = _sample_shape(area, forbidden_canons=used_canons)
            used_canons.add(canon)
            coords = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
            patch = _coords_to_patch(coords, avail[ci], bg)
            group.append(
                {
                    "area": int(area),
                    "canon": canon,
                    "coords": coords,
                    "patch": patch,
                    "color": int(avail[ci]),
                    "is_bottom_rep": j == 0,
                }
            )
            ci += 1
        repeated_groups.append(group)

    for area in sorted(unique_areas, reverse=True):
        shape, canon = _sample_shape(area, forbidden_canons=set())
        coords = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
        patch = _coords_to_patch(coords, avail[ci], bg)
        unique_objs.append(
            {
                "area": int(area),
                "canon": canon,
                "coords": coords,
                "patch": patch,
                "color": int(avail[ci]),
            }
        )
        ci += 1

    out = np.full((h_grid, w_grid), bg, dtype=int)

    repeated_groups.sort(
        key=lambda group: (-len(group), -int(group[0]["area"]), -max(obj["patch"].shape[1] for obj in group))
    )
    for group in repeated_groups:
        if not _place_repeated_group(out, group, h_grid, bg):
            raise ValueError("repeated group placement failed")

    unique_objs.sort(key=lambda d: (-int(d["patch"].shape[1]), -int(d["area"])))
    for obj in unique_objs:
        row = int(h_grid - obj["patch"].shape[0])
        c0 = _find_place_on_row(out, obj["patch"], row, bg)
        if c0 is None:
            raise ValueError("unique output placement failed")
        paste(out, obj["patch"], row, c0, transparent=bg)
        obj["c0"] = int(c0)

    all_objs = [obj for group in repeated_groups for obj in group] + list(unique_objs)

    inp = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in all_objs:
        patch = obj["patch"]
        r0 = _find_scatter_row(inp, patch, int(obj["c0"]), bg)
        if r0 is None:
            raise ValueError("input scatter failed")
        paste(inp, patch, r0, int(obj["c0"]), transparent=bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    metrics = _fusion_metrics(inp)
    assert metrics["repeated_area_bottom_count"] >= 1, "combo collapsed to D1-only"
    assert metrics["vs_shape_diff_count"] >= 1, "combo collapsed to original shape logic"
    assert metrics["vs_global_area_diff_count"] >= 1, "combo collapsed to global area logic"

    for group in repeated_groups:
        group_canons = {obj["canon"] for obj in group}
        assert len(group_canons) == len(group), "repeated area group lost shape diversity"
        leftmost = min(int(obj["c0"]) for obj in group)
        reps = [obj for obj in group if int(obj["c0"]) == leftmost]
        assert len(reps) == 1 and reps[0]["is_bottom_rep"], "leftmost representative mismatch"

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D1/0+D11/3")

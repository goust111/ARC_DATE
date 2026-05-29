# Combo variant of 62593bfd (index 1)
# combo_spec: D1/0+D2/1
# Name: 面积分组 + 水平分流

"""
Design analysis for the combo variant
=====================================

combo_rule:
  - Objects are still monochromatic 8-connected components on a variable background.
  - Grouping changes from D4-canonical shape frequency to area frequency.
  - Routing changes from vertical top/bottom to horizontal left/right.
  - If an area's frequency is >= 2, every object with that area moves to the LEFT edge.
  - If an area's frequency is 1, that object moves to the RIGHT edge.
  - Each object keeps its input top row r0; only the column changes.

behavior_trace:
  - Example input objects:
      A: red L,    area 5, bbox top row r0=3, width 3
      B: blue zig, area 5, bbox top row r0=9, width 4
      C: green T,  area 8, bbox top row r0=15, width 3
  - Original task:
      A/B/C are grouped by D4-canonical shape, then shared shapes go to row 0 and
      unique shapes go to row H-h, preserving c0.
  - This combo:
      A and B share area 5, so both are "duplicated" even though their silhouettes
      differ; they move to col 0 while keeping rows 3 and 9.
      C has unique area 8, so it moves to col W-w while keeping row 15.
  - Visible difference:
      the solver must notice two independent facts at once:
      "same pixel count => same group" and "movement is horizontal, not vertical".

complexity_calibration:
  - The grouping cue is simpler than D4 matching but still nontrivial because the
    repeated-area objects are intentionally generated with different shapes.
  - We keep only 4-6 objects and a small number of area categories so the pattern
    remains inferable from few ARC examples.
  - Distinct colors per object make tracking translations easy, but color never
    determines the grouping.

implicit_assumption_audit_main:
  - find_components(..., monochromatic=False, connectivity=8):
      assumes different objects never touch, even diagonally.
      generator guarantee: every object is assigned its own row band with >=1 blank
      row between bands, so neither input nor output can merge components.
  - patch extraction from bbox:
      still valid because each detected component is a whole object.
  - grouping key:
      old code used canonical shape; combo uses len(coords).
      No later line depends on shape equivalence once the key is replaced.
  - preserved anchor:
      old vertical routing preserved c0; combo must preserve r0 instead.
      Therefore extraction stores r0, and placement computes dest_c0.
  - paste():
      assumes routed placements are in-bounds and non-overlapping.
      generator guarantee: widths are <= 5 and row bands are disjoint, so left/right
      placement is always legal inside W in [19, 30].

placement_legality:
  - Output-first construction is retained.
  - Each object is converted to a patch, then row-packed with one blank row between
    consecutive bands. This prevents overlap and 8-neighborhood contact regardless of
    whether the object is sent to the left or right edge.
  - Input is produced by keeping the same row bands and randomizing only c0, so
    horizontal motion remains observable while legality is preserved automatically.

visual_observability_and_cleanliness:
  - Repeated-area objects are forced to have different canonical shapes, so a human
    cannot explain the examples by "same shape goes together".
  - At least one duplicated-area object starts away from the left edge, and at least
    one unique-area object starts away from the right edge, so both branches visibly
    move.
  - One object per row band keeps the image tidy: no clutter, no layered overlaps,
    and no ambiguous component boundaries.

non_degeneracy_check:
  - On the same input, this combo should often disagree with the "shape-grouping +
    horizontal-routing" variant. The reason is deliberate: repeated-area objects share
    area but not canonical shape, so combo sends them left while shape-grouping would
    send them right as unique shapes.

main_new_assumptions:
  - Every foreground component is a whole object and its area is stable under
    translation.
  - The input contains at least one duplicated area and at least one unique area.
  - generate_input() guarantees those assumptions and validates main(input) against an
    explicitly constructed output on every sample.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical shape key invariant under the D4 dihedral group."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        norm = _norm(cur)
        best = norm if best is None or norm < best else best
        refl = _norm([(-r, c) for r, c in cur])
        best = refl if refl < best else best
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route objects horizontally by whether their area is shared.

    Rule:
    - Find each 8-connected foreground component.
    - Compute its area = number of pixels in the component.
    - Shared areas  (area frequency >= 2) -> translate to LEFT  (col 0).
    - Unique areas  (area frequency == 1) -> translate to RIGHT (col W-w).
    - Vertical position (top row r0 of bbox) is preserved.
    """
    g = np.array(grid, dtype=int)
    h_grid, w_grid = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, _r1, _c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "r0": int(r0),
                "pat": pat,
                "area_key": int(len(rel)),
            }
        )

    freq = Counter(d["area_key"] for d in extracted)
    out = np.full((h_grid, w_grid), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        _h_obj, w_obj = map(int, pat.shape)
        dest_c0 = 0 if int(freq[d["area_key"]]) >= 2 else int(w_grid - w_obj)
        paste(out, pat, int(d["r0"]), int(dest_c0), transparent=bg)

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels."""
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
    """Sample a nontrivial 8-connected shape with a fresh canonical key."""
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


def _pack_rows_with_gaps(heights, h_grid, gap=1):
    """Pack objects into row bands separated by at least one blank row."""
    n = len(heights)
    required = sum(int(h) for h in heights) + max(0, n - 1) * int(gap)
    if required > h_grid:
        raise ValueError("insufficient height for row packing")

    slack = int(h_grid - required)
    extra = [0] * (n + 1)
    for _ in range(slack):
        extra[random.randrange(n + 1)] += 1

    rows = []
    cur = extra[0]
    for i, h in enumerate(heights):
        rows.append(int(cur))
        cur += int(h)
        if i < n - 1:
            cur += int(gap) + extra[i + 1]
        else:
            cur += extra[-1]
    return rows


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _gen_attempt():
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    duplicated_areas = random.sample(list(range(4, 13)), len(group_sizes))
    remaining_areas = [a for a in range(3, 13) if a not in duplicated_areas]
    unique_areas = random.sample(remaining_areas, n_unique)

    objs = []
    for gsz, area in zip(group_sizes, duplicated_areas):
        used_canons = set()
        for _ in range(gsz):
            shape, canon = _sample_shape(area, forbidden_canons=used_canons)
            used_canons.add(canon)
            transformed = _transform_shape(
                shape, random.randint(0, 3), random.choice([False, True])
            )
            objs.append(
                {
                    "coords": transformed,
                    "color": int(avail[ci]),
                    "shared": True,
                    "area": int(area),
                    "canon": canon,
                }
            )
            ci += 1

    for area in unique_areas:
        shape, canon = _sample_shape(area, forbidden_canons=set())
        transformed = _transform_shape(
            shape, random.randint(0, 3), random.choice([False, True])
        )
        objs.append(
            {
                "coords": transformed,
                "color": int(avail[ci]),
                "shared": False,
                "area": int(area),
                "canon": canon,
            }
        )
        ci += 1

    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h_obj, w_obj = patch.shape
        obj["patch"] = patch
        obj["h"] = int(h_obj)
        obj["w"] = int(w_obj)

    min_h = max(19, sum(obj["h"] for obj in objs) + len(objs) - 1)
    if min_h > 30:
        raise ValueError("objects too tall for clean horizontal routing")
    h_grid = random.randint(min_h, 30)
    w_grid = random.randint(19, 30)

    random.shuffle(objs)
    rows = _pack_rows_with_gaps([obj["h"] for obj in objs], h_grid, gap=1)

    out = np.full((h_grid, w_grid), bg, dtype=int)
    for obj, r0 in zip(objs, rows):
        patch = obj["patch"]
        dest_c0 = 0 if obj["shared"] else w_grid - obj["w"]
        paste(out, patch, int(r0), int(dest_c0), transparent=bg)
        obj["r0"] = int(r0)
        obj["dest_c0"] = int(dest_c0)

    inp = np.full((h_grid, w_grid), bg, dtype=int)
    shared_shifted = False
    unique_shifted = False
    for obj in objs:
        patch = obj["patch"]
        c0 = random.randint(0, w_grid - obj["w"])
        paste(inp, patch, int(obj["r0"]), int(c0), transparent=bg)
        obj["c0"] = int(c0)
        if obj["shared"] and c0 != 0:
            shared_shifted = True
        if (not obj["shared"]) and c0 != w_grid - obj["w"]:
            unique_shifted = True

    if not shared_shifted or not unique_shifted:
        raise ValueError("scatter did not visibly exercise both horizontal branches")

    area_counts = Counter(obj["area"] for obj in objs)
    assert any(v >= 2 for v in area_counts.values())
    assert any(v == 1 for v in area_counts.values())

    duplicated_canons = [obj["canon"] for obj in objs if area_counts[obj["area"]] >= 2]
    if len(set(duplicated_canons)) <= 1:
        raise ValueError("duplicated-area group is not visually distinct")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")

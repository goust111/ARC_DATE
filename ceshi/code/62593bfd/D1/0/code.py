# Variant of 62593bfd
# Dimension: D1 — 分组判据——形状等价类的分类规则 (match_cond)
# Change: 计算每个对象在D4二面体群（4旋转x2翻转=8变换）下的canonical形状，统计各canonical形状的出现频次；... → 按面积（像素数）分组

"""
Design analysis for the D1 area-grouping variant
===============================================

spec_variant:
  preserved_structure:
    grid: 19-30 rows x 19-30 cols
    objects: 4-6 monochromatic 8-connected components on a variable background
    transform: vertical translation only; each object's left bbox column c0 is preserved
    destination_rows:
      duplicated_group: row 0
      unique_group: row H-h
  changed_rule:
    old_match_cond: objects were grouped by D4-canonical shape frequency
    new_match_cond: objects are grouped by pixel-count frequency
    duplicated: area occurring >= 2 times
    unique: area occurring exactly 1 time
  generator_adaptations:
    - At least one duplicated-area group must contain objects with the same area but
      different canonical shapes. This makes the new rule visually distinguishable
      from the original shape-based rule.
    - Objects are placed with an 8-neighborhood gap, not just non-overlap. The solver
      uses find_components(..., monochromatic=False, connectivity=8), so diagonal or
      edge contact would merge different-colored objects into one component.
    - Input objects are scattered away from the top/bottom destination rows so the
      translation remains observable in examples.

behavior_trace:
  concrete_example:
    input_objects:
      - red L-shape, area 5, c0 = 3
      - blue zigzag, area 5, c0 = 10
      - green T-shape, area 8, c0 = 15
    old_logic:
      - red and blue have different canonical shapes, so each shape frequency is 1
      - all three objects are "unique" and move to the bottom
    new_logic:
      - red and blue share area 5, so area-frequency(5) = 2 and both move to the top
      - green has unique area 8, so it moves to the bottom
    visible_difference:
      - the same pair that used to go downward now goes upward even though their
        silhouettes do not match under rotation/reflection

complexity_calibration:
  - The task is simpler than D4 matching because solvers only need component counting,
    not shape normalization.
  - It is not over-simplified because duplicated groups are generated with repeated
    areas but intentionally distinct silhouettes, so "same shape" is not a valid shortcut.
  - The number of area categories stays small (2-5), matching ARC's few-example regime.

implicit_assumption_audit_main:
  - find_components(..., monochromatic=False, connectivity=8):
      assumes different objects never touch in the 8-neighborhood.
      generator guarantee: enforced by _can_place_without_touch.
  - patch extraction from bbox:
      only depends on each component being internally connected; still valid.
  - grouping key construction:
      old code depended on D4-canonicalization; variant replaces it with len(coords).
      No other line in main depends on shape equivalence.
  - frequency count and routing:
      now counts repeated areas. The routing rows and preserved c0 remain valid.
  - paste():
      assumes output placements do not overlap. generator guarantee: enforced during
      output-first construction with the same no-touch placement check.

visual_observability:
  - Same-area objects are often visibly different in outline, so a human can notice
    that "top objects are not the same shape, but they do have the same size."
  - Unique-area objects stay bottom-routed, giving the contrast needed to infer
    the split from 2-3 examples.
  - Colors are unchanged and not used for grouping, avoiding hidden numeric cues.

main_new_assumptions:
  - Every foreground component is a whole object.
  - Component area is stable under the scatter/reassemble process.
  - Horizontal destinations chosen by preserved c0 remain collision-free once objects
    are separated into top vs bottom rows.
  generate_input_guarantees:
    - all objects are 8-connected and pairwise 8-disconnected
    - intended duplicated/unique area frequencies are the only area collisions present
    - output-first validation checks main(input) == expected_output on every sample
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
        n = _norm(cur)
        best = n if best is None or n < best else best
        n = _norm([(-r, c) for r, c in cur])
        best = n if n < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    m = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < h and 0 <= cc < w:
            m[rr, cc] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their area is shared.

    Rule:
    - Find each 8-connected foreground component.
    - Compute its area = number of pixels in the component.
    - Shared areas  (area frequency >= 2) → translate to TOP    (row 0).
    - Unique areas  (area frequency == 1) → translate to BOTTOM (row H-h).
    - Horizontal position (left column c0 of bounding box) is preserved.
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
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "area_key": int(len(rel)),
            }
        )

    freq = Counter(d["area_key"] for d in extracted)
    out = np.full((h_grid, w_grid), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h_obj, _w_obj = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["area_key"]]) >= 2 else int(h_grid - h_obj)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

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
    """Sample a nontrivial 8-connected shape whose canonical key is new."""
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

    mask = (patch != bg)
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

    objs = []

    for gsz, area in zip(group_sizes, duplicated_areas):
        used_canons = set()
        for _ in range(gsz):
            shape, canon = _sample_shape(area, forbidden_canons=used_canons)
            used_canons.add(canon)
            shape = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
            objs.append(
                {
                    "coords": shape,
                    "color": avail[ci],
                    "shared": True,
                    "area": int(area),
                    "canon": canon,
                }
            )
            ci += 1

    for area in unique_areas:
        shape, canon = _sample_shape(area, forbidden_canons=set())
        shape = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
        objs.append(
            {
                "coords": shape,
                "color": avail[ci],
                "shared": False,
                "area": int(area),
                "canon": canon,
            }
        )
        ci += 1

    out = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h_obj, w_obj = patch.shape
        dest_r0 = 0 if obj["shared"] else h_grid - h_obj
        placed = False
        for _ in range(500):
            c0 = random.randint(0, w_grid - w_obj)
            if _can_place_without_touch(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h_obj, _w_obj = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(500):
            r0 = random.randint(1, h_grid - h_obj - 1)
            if _can_place_without_touch(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    area_counts = Counter(obj["area"] for obj in objs)
    assert any(v >= 2 for v in area_counts.values())
    assert any(v == 1 for v in area_counts.values())

    duplicated_area_canons = [
        obj["canon"] for obj in objs if area_counts[obj["area"]] >= 2
    ]
    assert len(set(duplicated_area_canons)) > 1, "variant not visually distinct"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd area-grouping variant")

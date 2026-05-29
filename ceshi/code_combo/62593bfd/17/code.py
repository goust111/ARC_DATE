# Combo variant of 62593bfd (index 17)
# combo_spec: D1/0+D2/0+D4/2
# Name: 面积分组 + 方向反转 + 水平镜像

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
spec_combo:
  preserved_structure:
    - grid size stays within 19-30 by 19-30
    - objects are monochromatic 8-connected components on a dominant background
    - object pixels themselves are unchanged; only whole-object translation happens
  fused_changes:
    - grouping key changes from D4-canonical shape to area = pixel count
    - routing is inverted relative to the original task:
      duplicated area -> bottom, unique area -> top
    - destination column is mirrored:
      dest_c0 = W - c0 - w
  small_trace:
    - suppose W=20, and there are three objects:
      red area-5 at c0=2, width=3
      blue area-5 at c0=10, width=4
      green area-8 at c0=14, width=2
    - original task:
      if all three shapes are canonically distinct, all are shape-unique,
      so all would go to the bottom and keep c0
    - combo task:
      area frequencies are {5: 2, 8: 1}
      red/blue are duplicated-area -> bottom
      green is unique-area -> top
      mirrored columns become 20-2-3=15, 20-10-4=6, 20-14-2=4
    - visible difference:
      the solver must notice that grouping is arithmetic rather than geometric,
      the top/bottom mapping is reversed relative to the original, and every
      object's horizontal landing column is mirrored.

b) Complexity calibration
  - The variant still has only two routing bands, so it stays ARC-sized.
  - It is not over-simplified because duplicated-area groups are generated from
    visibly different shapes; "same silhouette" is not a valid shortcut.
  - The generator keeps 4-6 total objects with 1-2 duplicated-area groups and
    1-3 unique-area objects, giving repeated evidence without overcrowding.
  - Mirror fixed points are forbidden so the horizontal rule is always active.

c) Hidden-assumption audit of main()
  - find_components(..., monochromatic=False, connectivity=8):
    assumes different objects never touch even diagonally. The generator enforces
    an 8-neighborhood gap in both input and output.
  - bbox extraction and relative coords:
    still valid because each component remains a whole object.
  - grouping key:
    old code depended on _canonical_shape(rel). The combo uses len(rel) instead.
    No later line in main depends on shape equivalence anymore.
  - destination row:
    the original shared->top assumption is invalid here and must be reversed to
    duplicated-area->bottom, unique-area->top.
  - destination column:
    preserving c0 is invalid here; the combo depends on both c0 and object width w.
  - paste():
    still assumes in-bounds non-overlapping destinations. Output-first generation
    explicitly checks the mirrored placements before accepting a sample.

d) Placement legality
  - Output-first construction is retained.
  - For each object, the generator samples an input c0, computes its mirrored
    destination column, and places the object directly into the intended output band.
  - Only after a valid output arrangement exists does it scatter the same patches
    back into interior rows at their original c0 values.
  - Because bbox height/width are at most 5 and H,W are at least 19, mirrored
    destinations fit comfortably and top/bottom bands remain well separated.

e) Visual observability
  - Bottom-band objects repeat by area, not by shape, so a human sees that
    different-looking objects can still be grouped together.
  - Top-band objects have unique areas, giving contrast against the repeated
    bottom-band family.
  - All objects shift horizontally to mirrored columns, so left-right movement is
    visible on every accepted sample.
  - No hidden property such as color id, creation order, or component index is used.

f) Visual cleanliness
  - Objects are sparse, monochromatic, and 8-disconnected.
  - Only two edge bands are used in the output, with no overlays or markers.
  - Distinct colors per object help correspondence tracking without introducing a
    color-mapping rule.

g) Quantitative anti-degeneration checks
  - The generator validates main(input) == expected_output.
  - It also guarantees the three changed dimensions are active:
    1. duplicated and unique areas both exist
    2. at least one duplicated-area group contains multiple canonical shapes
    3. every object satisfies dest_c0 != c0
  - In evaluation, a useful metric is comparison against three ablations on the
    same input:
      area_effective: combo != shape-grouping + inverted + mirrored
      inversion_effective: combo != area-grouping + non-inverted + mirrored
      mirror_effective: combo != area-grouping + inverted + non-mirrored
    The generator is designed so all three booleans should be true.

h) New assumptions after editing main()
  - Every component's area is stable under scatter/reassemble.
  - Mirrored destinations computed from bbox width do not create collisions.
  - There is at least one duplicated-area object and at least one unique-area object
    if the examples are to teach the rule; generate_input() guarantees both.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
        flipped = _norm([(-r, c) for r, c in cur])
        best = flipped if flipped < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    h_obj, w_obj = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < h_obj and 0 <= cc < w_obj:
            mask[rr, cc] = True
    return mask


def _mirror_c0(w_grid, c0, w_obj):
    return int(w_grid - c0 - w_obj)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by area, reverse the top/bottom routing, and mirror columns.

    Rule:
    - Find each 8-connected foreground component.
    - Compute its area = number of pixels in the component.
    - Duplicated areas (freq >= 2) move to the BOTTOM edge, row H-h.
    - Unique areas    (freq == 1) move to the TOP edge, row 0.
    - Destination column is mirrored across the vertical axis:
      dest_c0 = W - c0 - w.
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
        patch = crop_bbox(g, tuple(map(int, comp["bbox"])))
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
        h_obj, w_obj = map(int, pat.shape)
        dest_r0 = int(h_grid - h_obj) if int(freq[d["area_key"]]) >= 2 else 0
        dest_c0 = _mirror_c0(w_grid, int(d["c0"]), w_obj)
        paste(out, pat, dest_r0, dest_c0, transparent=bg)

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly area pixels."""
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
                if (
                    max(all_r) - min(all_r) < max_dim
                    and max(all_c) - min(all_c) < max_dim
                ):
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
    """
    Sample a nontrivial 8-connected shape with a fresh canonical form.

    Requiring both bbox dimensions to be >= 2 avoids degenerate straight segments
    that make area counting and mirroring less visually informative.
    """
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
    h_obj, w_obj = patch.shape
    h_grid, w_grid = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h_obj > h_grid or c0 + w_obj > w_grid:
        return False

    mask = patch != bg
    if np.any(grid[r0 : r0 + h_obj, c0 : c0 + w_obj][mask] != bg):
        return False

    for pr, pc in np.argwhere(mask):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        r1 = max(0, rr - 1)
        r2 = min(h_grid, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(w_grid, cc + 2)
        window = grid[r1:r2, c1:c2]
        if np.any(window != bg):
            return False
    return True


def _interior_row(h_grid, h_obj):
    lo = 1
    hi = h_grid - h_obj - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


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

    group_sizes, n_unique = random.choice(_CONFIGS)
    duplicated_areas = random.sample(list(range(4, 13)), len(group_sizes))
    remaining_areas = [a for a in range(3, 13) if a not in duplicated_areas]
    unique_areas = random.sample(remaining_areas, n_unique)

    objs = []
    color_idx = 0
    all_canons = set()

    for gsz, area in zip(group_sizes, duplicated_areas):
        group_canons = set()
        for _ in range(gsz):
            shape, canon = _sample_shape(area, forbidden_canons=group_canons | all_canons)
            group_canons.add(canon)
            all_canons.add(canon)
            transformed = _transform_shape(
                shape, random.randint(0, 3), random.choice([False, True])
            )
            objs.append(
                {
                    "coords": transformed,
                    "color": int(avail[color_idx]),
                    "area": int(area),
                    "shared": True,
                    "canon": canon,
                }
            )
            color_idx += 1

    for area in unique_areas:
        shape, canon = _sample_shape(area, forbidden_canons=all_canons)
        all_canons.add(canon)
        transformed = _transform_shape(
            shape, random.randint(0, 3), random.choice([False, True])
        )
        objs.append(
            {
                "coords": transformed,
                "color": int(avail[color_idx]),
                "area": int(area),
                "shared": False,
                "canon": canon,
            }
        )
        color_idx += 1

    out = np.full((h_grid, w_grid), bg, dtype=int)
    moved_horizontally = 0
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h_obj, w_obj = patch.shape
        dest_r0 = int(h_grid - h_obj) if obj["shared"] else 0
        placed = False
        for _ in range(500):
            c0 = random.randint(0, w_grid - w_obj)
            dest_c0 = _mirror_c0(w_grid, c0, w_obj)
            if dest_c0 == c0:
                continue
            if _can_place_without_touch(out, patch, dest_r0, dest_c0, bg):
                paste(out, patch, dest_r0, dest_c0, transparent=bg)
                obj["patch"] = patch
                obj["c0"] = int(c0)
                obj["dest_c0"] = int(dest_c0)
                placed = True
                moved_horizontally += 1
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
            r0 = _interior_row(h_grid, h_obj)
            if r0 is None:
                raise ValueError("no interior rows available")
            if _can_place_without_touch(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    area_counts = Counter(obj["area"] for obj in objs)
    assert any(v >= 2 for v in area_counts.values()), "missing duplicated area"
    assert any(v == 1 for v in area_counts.values()), "missing unique area"
    assert moved_horizontally == len(objs), "mirror rule degenerated"

    duplicated_canons = [
        obj["canon"] for obj in objs if int(area_counts[obj["area"]]) >= 2
    ]
    assert len(set(duplicated_canons)) > 1, "duplicated-area shapes are not distinct"
    assert len(all_canons) == len(objs), "generator accidentally recreated a shape class"

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")

# Combo variant of 62593bfd (index 4)
# combo_spec: D2/0+D4/2
# Name: 方向反转 + 水平镜像

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

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Invariant core from the original task:
#   each foreground object is an 8-connected monochromatic component; objects are
#   grouped by D4-canonical shape, so rotations/reflections count as the same shape.
# - Combo change 1 (D2/0): the vertical assignment is reversed.
#   shared/duplicated shapes now go to the BOTTOM edge; unique shapes go to the TOP.
# - Combo change 2 (D4/2): the destination column is mirrored.
#   If an object has bbox-left column c0 and width w in a W-column grid, then
#   dest_c0 = W - c0 - w.
# - Pixel trace:
#   suppose H=18, W=20.
#   Shared object A has h=4, w=3, input c0=2.
#   Unique object B has h=3, w=5, input c0=11.
#   Original task:
#     A -> (row 0, col 2), B -> (row 15, col 11)
#   Combo task:
#     A -> (row 14, col 15), B -> (row 0, col 4)
#   Difference:
#     the objects themselves are not reflected internally; only whole-object placement
#     changes. The solver must infer both "shared vs unique swaps top/bottom" and
#     "all destination columns are mirrored".
#
# b) Complexity calibration
# - We keep the original object count/size regime (4-6 objects, 3-12 cells, bbox <= 5)
#   so the task remains physically placeable and visually parseable.
# - The generator still guarantees both categories exist:
#   at least one duplicated canonical shape family and at least one unique shape.
#   Without both categories, the direction reversal would be invisible.
# - To make the horizontal mirror observable, the generator forbids fixed-point columns
#   where c0 == W - c0 - w. Otherwise a sample could accidentally look like a pure D2
#   variant instead of the intended combo.
#
# c) Hidden-assumption audit of main()
# - find_components(..., connectivity=8, monochromatic=False):
#   assumes distinct objects do not touch, even diagonally, because touching different
#   colors would still merge into one component. The generator therefore uses an
#   8-neighborhood clearance check, not just non-overlap.
# - rel coords and _canonical_shape(rel):
#   depend only on geometry inside each bbox, not on where the object will be moved.
#   This stays valid under the combo.
# - Destination placement now depends on both category and bbox width:
#   dest_r0 uses reversed shared/unique mapping, and dest_c0 uses mirrored c0.
#   main() therefore must retain both the original c0 and the extracted patch width.
# - paste(..., transparent=bg):
#   assumes computed destinations are in-bounds and pairwise disjoint; the generator
#   guarantees legality in the mirrored output layout.
#
# d) Placement legality
# - Output-first generation is preserved, but legality is checked at the combo
#   destination: unique objects at the top band, shared objects at the bottom band,
#   with mirrored columns.
# - Input is obtained by scattering the same patches back to their pre-mirror columns
#   c0 while keeping rows strictly interior (1..H-h-1), so examples visibly require
#   both vertical motion and horizontal mirror.
# - Because H >= 19 and h <= 5, top and bottom edge bands remain well separated.
#
# e) Visual observability
# - A human comparing 2-3 examples can track repeated shapes across colors and see:
#   repeated families consistently end on the lower edge, unique shapes on the upper
#   edge, and every object's final horizontal position is the left-right mirror of its
#   input column.
# - No hidden numeric property such as color id, component order, or absolute index is
#   used in grouping or placement.
#
# f) Visual cleanliness
# - Objects remain isolated monochromatic blobs on a dominant background.
# - The only structural cues are edge band assignment and left-right mirroring; no extra
#   markers or overlapping annotations are introduced.
#
# g) Change inferability
# - Colors are preserved per object and are not part of the rule. Distinct colors per
#   object are kept so a human can track corresponding pieces across input/output while
#   inferring a purely shape-and-position based rule.
#
# h) Quantitative variant check
# - Generator-side anti-degeneracy checks compare the combo output against:
#   1) the original rule,
#   2) D2-only (reversed vertical, no mirror),
#   3) D4-only (original vertical, mirrored column).
#   Each generated sample must differ from all three, ensuring both control variables
#   are active in the sample itself rather than only in the code path.
#
# i) New assumptions after editing
# - main() now assumes mirrored destinations stay disjoint and in-bounds, and that the
#   input c0 values correspond to intended pre-mirror positions. generate_input() enforces
#   both explicitly.
# - main() still assumes every foreground component is a real object; the generator's
#   8-clearance placement check guarantees that no accidental component merging occurs.


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
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _mirror_c0(W, c0, w):
    return int(W - c0 - w)


def _extract_objects(grid: np.ndarray, bg: int):
    comps = find_components(
        grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, tuple(map(int, comp["bbox"])))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return extracted


def _render_objects(extracted, H, W, bg, *, shared_to_bottom: bool, mirror_horizontal: bool):
    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        is_shared = int(freq[d["shape_key"]]) >= 2
        if shared_to_bottom:
            dest_r0 = int(H - h) if is_shared else 0
        else:
            dest_r0 = 0 if is_shared else int(H - h)
        dest_c0 = _mirror_c0(W, d["c0"], w) if mirror_horizontal else int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out


def _apply_rule(grid: np.ndarray, *, shared_to_bottom: bool, mirror_horizontal: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    extracted = _extract_objects(g, bg)
    return _render_objects(
        extracted,
        H,
        W,
        bg,
        shared_to_bottom=shared_to_bottom,
        mirror_horizontal=mirror_horizontal,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected objects by D4-canonical shape and move them to opposite edges,
    with both the original vertical rule reversed and the destination columns mirrored.

    Rule:
    - Compute each component's canonical shape under the D4 group.
    - Shared shapes (freq >= 2) move to the BOTTOM edge, row H-h.
    - Unique shapes (freq == 1) move to the TOP edge, row 0.
    - The whole object's destination column is mirrored:
      dest_c0 = W - c0 - w.
    """
    return _apply_rule(grid, shared_to_bottom=True, mirror_horizontal=True)


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random 8-connected shape of exactly area pixels."""
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
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, c in pts)
    mc = min(c for r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[r, c] = color
    return p


def _can_place_component(grid, patch, r0, c0, bg):
    """Require in-bounds placement with 8-neighborhood clearance from existing objects."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = patch != bg
    if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg):
        return False

    occ = np.argwhere(mask)
    for pr, pc in occ:
        gr = int(r0 + pr)
        gc = int(c0 + pc)
        for nr in range(gr - 1, gr + 2):
            for nc in range(gc - 1, gc + 2):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if r0 <= nr < r0 + h and c0 <= nc < c0 + w and mask[nr - r0, nc - c0]:
                    continue
                if grid[nr, nc] != bg:
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


def _interior_row(H, h):
    lo = 1
    hi = H - h - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append({"coords": shape, "color": avail[ci], "shared": False})
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = int(H - h) if obj["shared"] else 0
        placed = False
        for _ in range(500):
            c0 = random.randint(0, W - w)
            dest_c0 = _mirror_c0(W, c0, w)
            if dest_c0 == c0:
                continue
            if _can_place_component(out, patch, dest_r0, dest_c0, bg):
                paste(out, patch, dest_r0, dest_c0, transparent=bg)
                obj["c0"] = c0
                obj["dest_c0"] = dest_c0
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
        placed = False
        for _ in range(500):
            r0 = _interior_row(H, h)
            if r0 is None:
                raise ValueError("no interior rows")
            if _can_place_component(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["r0"] = r0
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    combo = main(inp)
    original = _apply_rule(inp, shared_to_bottom=False, mirror_horizontal=False)
    d2_only = _apply_rule(inp, shared_to_bottom=True, mirror_horizontal=False)
    d4_only = _apply_rule(inp, shared_to_bottom=False, mirror_horizontal=True)

    assert np.array_equal(combo, out), "round-trip check failed"
    assert not np.array_equal(combo, original), "combo degenerated to original"
    assert not np.array_equal(combo, d2_only), "combo lost horizontal mirror cue"
    assert not np.array_equal(combo, d4_only), "combo lost direction-reversal cue"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")

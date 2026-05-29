# Variant of 62593bfd
# Dimension: D2 — 分组后的方向分配规则 (match_cond)
# Change: duplicated形状（频次>=2）的对象移到顶部（row 0），unique形状（频次==1）的对象移到底部（row... → 方向反转——duplicated移底部，unique移顶部

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

# Design analysis for the D2 inversion variant
# ------------------------------------------------------------
# a) Concrete behavior and trace
# spec:
#   invariant_parts:
#     grid: 19-30 x 19-30
#     objects: 4-6 monochromatic 8-connected components, each 3-12 pixels
#     grouping: D4-canonical shape equality, ignoring color
#     motion: vertical translation only; each object's left bbox column c0 is preserved
#   original_direction:
#     duplicated_shape: move to row 0
#     unique_shape: move to row H-h
#   variant_direction:
#     duplicated_shape: move to row H-h
#     unique_shape: move to row 0
#   pixel-level_effect:
#     repeated shapes that used to accumulate along the top edge now accumulate along
#     the bottom edge; one-off shapes do the opposite. Colors and shapes themselves do
#     not change, so the only visible difference is which category occupies the upper vs
#     lower boundary band.
#
# Small trace:
# - Suppose H=20 and there are three objects with canonical groups A, A, B. Let both A
#   objects have height 4 and B have height 3.
# - Original task: A objects go to row 0, B goes to row 17.
# - Variant:      A objects go to row 16, B goes to row 0.
# - Difference: identical grouping step, identical preserved c0, only the mapping from
#   {shared, unique} to {top, bottom} is flipped.
#
# b) Complexity calibration
# - The task still needs both a duplicated group and at least one unique shape; otherwise
#   the inversion would be unobservable.
# - We keep 4-6 objects and distinct colors per object so humans can track each piece and
#   compare repeated shapes without color ambiguity.
# - We keep D4 matching unchanged; this preserves the nontrivial "same shape under
#   rotation/reflection" reasoning while changing only the edge assignment rule.
#
# c) Hidden-assumption audit of main()
# - find_components(..., connectivity=8, monochromatic=False):
#   depends on objects not touching, because touching same- or different-colored pixels
#   would merge into one component. Generator still guarantees no overlap and no touching
#   by only placing onto pure background.
# - rel = coords - bbox origin and _canonical_shape(rel):
#   depends only on object geometry, not on destination direction. Still valid.
# - dest_r0 assignment:
#   this is the only line whose semantics change under the variant. The old "shared->top"
#   assumption is no longer valid and must be reversed.
# - paste(..., transparent=bg):
#   assumes output placements are non-overlapping. Generator continues to enforce this.
#
# d) Placement legality
# - Output-first generation is preserved. Objects are first placed in their variant edge
#   bands (unique at top, shared at bottom) with no overlap at their preserved columns.
# - Input is produced by vertical re-scattering at the same c0. We keep rows away from
#   the two border destinations so examples visibly require motion and do not degenerate
#   into already-solved placements.
# - Since each bbox height is at most 5 and H is at least 19, top and bottom bands are
#   always far apart; only same-band overlap needs checking.
#
# e) Visual observability
# - The inversion is visible from 2-3 examples because the repeated-shape family appears
#   multiple times in each grid. A solver who expects "repeated=top" from the original
#   task would be contradicted by the examples: the repeated family consistently ends up
#   on the bottom edge here.
# - No hidden numeric property such as color id or object order is used.
#
# f) Visual cleanliness
# - Objects remain isolated monochromatic blobs on a dominant background.
# - Only one organization principle is shown in the output: top band vs bottom band.
#   There are no extra markers or overlays.
#
# g) Color inference
# - Colors are preserved object-wise and are not part of grouping. Distinct colors per
#   object make correspondence tracking easier without introducing a color-mapping rule.
#
# h) Quantitative variant check used after implementation
# - Metric: compare this variant's output against the original direction mapping on the
#   same generated inputs. Because every sample contains at least one shared and one
#   unique shape and H>h, outputs should differ on every valid sample.
#
# i) New assumptions to re-check after editing
# - main() now assumes only that the input still contains both object categories if we
#   want the inversion to be observable; generate_input() guarantees that.
# - The interior-row scatter assumes there is at least one non-edge row for every object.
#   Given H in [19,30] and h in [1,5], generate_input() guarantees this.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their D4-canonical shape
    is unique or shared, with the direction reversed from the original task.

    Rule:
    - Compute each 8-connected component's canonical shape under D4.
    - Shared shapes (freq >= 2) move to the BOTTOM edge, row H-h.
    - Unique shapes (freq == 1) move to the TOP edge, row 0.
    - Horizontal position (left bbox column c0) is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = int(H - h) if int(freq[d["shape_key"]]) >= 2 else 0
        dest_c0 = int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

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


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


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
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
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
        # Keep inputs away from both target edges so the direction inversion
        # is visually active in the example pair.
        candidate_rows = list(range(1, H - h))
        if not candidate_rows:
            raise ValueError("no interior rows available")
        placed = False
        for _ in range(300):
            r0 = random.choice(candidate_rows)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D2 variant")

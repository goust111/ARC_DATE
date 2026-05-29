# Variant of 62593bfd
# Dimension: D4 — 平移规则——水平位置保持 (dst)
# Change: 每个对象仅做垂直平移，水平位置（外接矩形左上角列c0）保持不变；顶部对象dest_r0=0，底部对象dest_r0=H-... → 水平镜像——c0变为W-c0-w

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for this variant
# --------------------------------
# spec:
#   unchanged_core:
#     - Objects are still monochromatic 8-connected components on a bg grid.
#     - Grouping is still by canonical D4 shape: rotations/reflections count as the
#       same shape, so shared-vs-unique classification is unchanged.
#   changed_core:
#     - Original output kept each object's bbox-left column c0 unchanged.
#     - Variant output mirrors the destination column across the vertical axis:
#       dest_c0 = W - c0 - w, where w is the object's bbox width.
#     - The patch pixels themselves are not reflected; only the whole object's
#       position is mirrored while it is also moved vertically to top/bottom.
#   concrete_trace:
#     - Example with W=20 and an object of width w=3 at input c0=2:
#       original output column: 2
#       variant output column: 20 - 2 - 3 = 15
#     - If that object is in a shared D4 group, original destination is (0, 2)
#       while the variant destination is (0, 15).
#     - If it is unique, original destination is (H-h, 2) while the variant
#       destination is (H-h, 15).
#   complexity_calibration:
#     - To keep the new rule visible, generator forbids fixed-point placements
#       where c0 == W - c0 - w; otherwise an example can accidentally look like
#       the original task.
#     - Vertical scatter avoids top/bottom rows so examples visibly require both
#       "move vertically" and "mirror horizontally".
#     - Distinct colors per object are preserved so humans can visually track
#       each moved component without introducing a new color rule.
#   dependency_audit_for_main:
#     - D4 canonicalization still applies exactly as before.
#     - Extracting c0 alone is no longer sufficient for destination placement;
#       the mirrored column also depends on bbox width w, so main must use
#       pat.shape[1] when computing dest_c0.
#     - paste() still assumes the computed destination is in-bounds and that
#       different objects do not overlap after transformation. The generator now
#       guarantees this at mirrored columns.
#   placement_legality:
#     - Output-first generation is kept, but "first" now means choosing each
#       object's input c0 and checking the mirrored output column for legality.
#     - Objects are kept 8-disconnected in both input and output so
#       find_components(..., monochromatic=False, connectivity=8) recovers each
#       intended object instead of merging touching colors into one component.
#   observability_and_cleanliness:
#     - The only new cue is left-right mirrored destination placement, so sample
#       generation must make that cue appear on most/all objects.
#     - Objects remain sparse and non-touching, avoiding ambiguous merged blobs.
#   new_main_assumptions:
#     - Each foreground component is an intended object.
#     - Mirrored destinations do not overlap and stay in-bounds.
#     - generate_input() enforces both assumptions explicitly.


def _canonical_shape(coords):
    """Canonical shape key invariant under the D4 dihedral group
    (4 rotations × 2 reflections = 8 transforms).  Returns the
    lexicographically smallest sorted-tuple across all 8 images."""
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
        cur = [(c, -r) for r, c in cur]  # 90 deg CW
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their shape (up to
    rotation / reflection) is shared with another object.

    Rule:
    - Compute the canonical shape of each 8-connected component
      (invariant under D4: 0°/90°/180°/270° rotations × optional
      horizontal reflection).
    - Shared shapes  (canonical shape freq ≥ 2) → translate to TOP  (row 0).
    - Unique shapes  (freq == 1)                → translate to BOTTOM (row H−h).
    - Variant change: destination column is mirrored across the vertical axis,
      so dest_c0 = W - c0 - w.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
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
        h, w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        dest_c0 = _mirror_c0(W, d["c0"], w)
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out


# ---------------------- generator ----------------------

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random 8-connected shape of exactly *area* pixels,
    kept within a *max_dim* × *max_dim* bounding box."""
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
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
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
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(400):
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
    moved = 0
    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        c0 = obj["c0"]
        placed = False
        for _ in range(400):
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
        if obj["dest_c0"] != c0:
            moved += 1

    if moved != len(objs):
        raise AssertionError("variant degenerated to original horizontal placement")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd variant")

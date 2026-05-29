# Combo variant of 62593bfd (index 21)
# combo_spec: D5/4+D6/0
# Name: unique擦除 + 紧密裁剪

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste

# Design analysis for the fused variant
# -------------------------------------
# spec_variant:
#   fused_rule:
#     Keep the original D4-canonical grouping of monochromatic 8-connected objects.
#     Shared canonical shapes (freq >= 2) are still translated upward to row 0 with
#     their bbox-left column c0 preserved on the original canvas.
#     Unique canonical shapes (freq == 1) are erased instead of moving to the bottom.
#     After that erase step, crop the canvas to the minimal bbox of the retained pixels.
#
#   concrete_trace:
#     Example on a 23x22 canvas:
#       - shared blue L with bbox width 3 at c0=4
#       - shared red L-equivalent at c0=11
#       - unique green T at c0=15
#     Original task:
#       blue/red move to row 0, green moves to row H-h_green, output stays 23x22.
#     Fused task:
#       blue/red move to row 0, green disappears, then the full canvas is cropped to
#       rows 0..max(shared_heights)-1 and cols 4..(rightmost shared pixel). The answer
#       is a compact top strip containing only the duplicated objects.
#
#   complexity_calibration:
#     Keep the original object counts, sizes, D4 variability, and distinct per-object
#     colors so the task still requires shape grouping. Every sample must contain:
#       1. at least one duplicated shape group, so something survives the erase step
#       2. at least one unique object, so deletion is visible
#     The crop must be visually observable, so retained top objects are forced away
#     from the left and right borders. Vertical crop is automatic because only the top
#     band remains after erasing uniques.
#
#   hidden_assumptions_review_for_main:
#     - `find_components(..., monochromatic=False)` merges any 8-touching non-bg pixels
#       even across colors. Therefore generator must enforce an 8-neighborhood moat
#       between all objects in both input and retained-output construction.
#     - Component extraction via each bbox still works after fusion because retained
#       objects are pasted unchanged; only the unique branch is removed.
#     - The original bottom-placement branch no longer exists. Any code assuming the
#       output spans the full height becomes invalid, so cropping must happen only after
#       composing the retained top-row canvas.
#     - Tight crop assumes at least one retained object. Generator guarantees at least
#       one duplicated group in every sample, so the retained canvas is never empty.
#
#   placement_legality:
#     Output-first construction is now done on a full hidden HxW canvas containing only
#     retained shared objects at row 0. Unique objects are never placed into that output
#     canvas because they are erased. Input is then created by scattering shared objects
#     vertically while preserving their output c0, and scattering unique objects anywhere
#     in interior rows/columns, always with 8-neighbor separation.
#
#   visual_observability:
#     Across 2-4 examples a human sees three coupled signals:
#       - only repeated shapes survive
#       - surviving shapes line up at the top
#       - the answer frame shrinks to fit just those survivors
#     Unique-colored objects disappear entirely, and the compact output strip makes the
#     erase+crop fusion distinguishable from both the original task and the single erase
#     variant that still returns the full canvas.
#
#   cleanliness:
#     Retained top objects are separated by at least one background cell in the 8-neighborhood,
#     so the cropped strip remains readable. Input scattering avoids row 0 and, when possible,
#     avoids the exact output-aligned rows to make the movement/deletion pattern easier to infer.


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
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _tight_crop_nonbg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by D4-canonical shape, erase unique shapes, then tight-crop.

    - Shared shapes  (canonical shape freq >= 2) -> translate to TOP (row 0).
    - Unique shapes  (canonical shape freq == 1) -> erase completely.
    - Retained objects keep their shape, color, orientation, and left column c0.
    - Finally crop to the minimal bbox containing all retained pixels.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
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
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)
    kept = 0

    for d in extracted:
        if int(freq[d["shape_key"]]) < 2:
            continue
        paste(out, d["pat"], 0, int(d["c0"]), transparent=bg)
        kept += 1

    assert kept > 0
    return _tight_crop_nonbg(out, bg)


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` pixels."""
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
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor contact with already placed objects."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = patch != int(bg)
    if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != int(bg)):
        return False

    for pr, pc in np.argwhere(mask):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        rr0 = max(0, rr - 1)
        rr1 = min(H, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(W, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != int(bg)):
            return False
    return True


def _interior_rows(H, h):
    max_r0 = int(H - h)
    rows = list(range(1, max_r0))
    return rows if rows else list(range(0, max_r0 + 1))


def _choose_horizontal_band(W: int, max_obj_w: int) -> tuple[int, int]:
    max_margin = max(1, min(4, (W - max_obj_w - 1) // 2))
    if max_margin < 1:
        raise ValueError("grid too narrow for crop margins")
    left_margin = random.randint(1, max_margin)
    right_margin = random.randint(1, max_margin)
    if left_margin + max_obj_w + right_margin > W:
        raise ValueError("band infeasible")
    return int(left_margin), int(W - right_margin - 1)


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
                base,
                random.randint(0, 3),
                random.choice([False, True]),
            )
            objs.append(
                {
                    "coords": transformed,
                    "color": avail[ci],
                    "shared": True,
                }
            )
            ci += 1

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append(
                    {
                        "coords": shape,
                        "color": avail[ci],
                        "shared": False,
                    }
                )
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]
    assert shared_objs and unique_objs

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    max_shared_w = max(int(obj["patch"].shape[1]) for obj in shared_objs)
    band_l, band_r = _choose_horizontal_band(W, max_shared_w)

    out_full = np.full((H, W), bg, dtype=int)
    for obj in shared_objs:
        patch = obj["patch"]
        _h, w = patch.shape
        lo = int(band_l)
        hi = int(band_r - w + 1)
        if lo > hi:
            raise ValueError("shared placement band too small")
        placed = False
        for _ in range(400):
            c0 = random.randint(lo, hi)
            if _no_touch(out_full, patch, 0, c0, bg):
                paste(out_full, patch, 0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("shared output placement failed")

    out_cropped = _tight_crop_nonbg(out_full, bg)
    assert out_cropped.shape[0] < H, "vertical crop degenerated"
    assert out_cropped.shape[1] < W, "horizontal crop degenerated"

    inp = np.full((H, W), bg, dtype=int)

    for obj in shared_objs:
        patch = obj["patch"]
        h, _w = patch.shape
        rows = _interior_rows(H, h)
        random.shuffle(rows)
        placed = False
        for r0 in rows:
            if _no_touch(inp, patch, r0, int(obj["c0"]), bg):
                paste(inp, patch, r0, int(obj["c0"]), transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("shared input scatter failed")

    for obj in unique_objs:
        patch = obj["patch"]
        h, w = patch.shape
        rows = _interior_rows(H, h)
        placed = False
        for _ in range(500):
            r0 = random.choice(rows)
            c0 = random.randint(0, W - w)
            if _no_touch(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("unique input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out_cropped), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D5/4 + D6/0")

# Variant of 62593bfd
# Dimension: D5 — 对象变换——移动过程中的形状/颜色变化 (transform)
# Change: 对象在平移过程中不做任何变换：形状、颜色、朝向完全保持原样（duplicated-size移顶部，singleton-size移底部，但对象本身不变） → 所有对象做水平翻转

from __future__ import annotations

"""
Design analysis for the D5 variant
==================================

spec_variant:
  inherited_structure:
    grid: 19-30 rows x 19-30 cols
    bg: most common color
    objects: 4-6 monochromatic 8-connected components on bg
    grouping: canonical shape is still computed up to the full D4 group
    routing:
      duplicated canonical shape (freq >= 2): move to top edge
      unique canonical shape (freq == 1): move to bottom edge
    preserved_position: keep each object's bbox left column c0 unchanged

  d5_change:
    transform: when an object is pasted into its destination row, paste its
      horizontal mirror image (`np.fliplr` of the cropped object patch)
    color: unchanged
    grouping_key: unchanged; grouping is still based on the INPUT object's
      D4-canonical shape, not on the flipped output patch

  concrete_pixel_behavior:
    - Input still contains isolated monochromatic components scattered at
      different rows.
    - Output still separates "shared-shape" components to the top and
      "unique-shape" components to the bottom.
    - New visible difference: every output object is mirrored left-right
      inside its own bbox while keeping the same bbox width and same c0.
      Example:
        input patch        output patch
        100               001
        110      -->      011
        010               010
      The top/bottom routing is unchanged; only the sprite geometry is flipped.

  flow_impact_trace:
    original:
      extract object patch -> count D4 shape frequency -> choose top/bottom row
      -> paste patch unchanged
    variant:
      extract object patch -> count D4 shape frequency -> choose top/bottom row
      -> horizontally flip extracted patch -> paste flipped patch
    difference:
      only the final rendered sprite changes; classification logic does not.

  complexity_calibration:
    - A plain monochromatic object can hide the D5 change if it is horizontally
      symmetric or only 1 column wide. The generator therefore rejects such
      objects so the mirror operation is visible in nearly every object.
    - D4 grouping is retained exactly, so the task does not gain extra hidden
      branches; the only new freedom is a single global transform: horizontal
      reflection.

  implicit_assumption_audit_main:
    - `_canonical_shape(rel)`: assumes grouping rule is shape-based and D4
      invariant. Still valid because the variant changes output rendering only.
    - `pat` extracted from the object's bbox: assumes the object's cropped patch
      is the right thing to move. Still valid, and now it is also the thing we
      flip before pasting.
    - `dest_c0 = original c0`: assumes the object keeps its left bbox anchor.
      Still valid because `np.fliplr` preserves patch width.
    - `paste(out, pat, ...)`: this is the one line whose old assumption breaks.
      In the original task, the patch itself was final. In the variant, the
      patch must first become its horizontal mirror image.

  placement_legality:
    - Components are detected with `connectivity=8` and `monochromatic=False`.
      Therefore two different-colored objects may not touch even diagonally, or
      the solver would merge them into one component.
    - The generator enforces a one-cell 8-neighborhood gap for both input
      scatter and output placement.
    - Top and bottom destination bands remain non-overlapping because object
      heights stay <= 5 while the grid height stays >= 19.

  visual_observability:
    - The generator rejects shapes whose cropped patch equals its horizontal
      mirror image, ensuring the D5 transform is visible rather than degenerate.
    - Colors remain distinct per object so the human solver can track objects
      between input and output without relying on hidden metadata.

  cleanliness:
    - No markers or extra decorations are added.
    - Objects remain separated by background, so the grid still reads as a set
      of clean sprites that move to top/bottom and mirror in transit.

  post-change_main_assumptions_and_generator_guarantees:
    - `main()` now assumes every object can be safely mirrored inside its bbox;
      this always holds because `np.fliplr` preserves shape dimensions.
    - `main()` still assumes the found components are truly separate objects;
      the generator guarantees this with the no-touch placement rule.
    - The generator also guarantees that the D5 change is active by producing
      horizontally asymmetric object patches, so `main(input)` differs from the
      old no-flip output on almost every generated object.
"""

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
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _flip_patch_horiz(patch: np.ndarray) -> np.ndarray:
    return np.fliplr(np.array(patch, dtype=int)).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their shape (up to D4
    symmetry) is shared with another object, then horizontally mirror each
    object while moving it to the destination row.
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
        moved_pat = _flip_patch_horiz(d["pat"])
        h, _w = map(int, moved_pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        dest_c0 = int(d["c0"])
        paste(out, moved_pat, dest_r0, dest_c0, transparent=int(bg))

    return out


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


def _is_hflip_visible_patch(patch: np.ndarray, bg: int) -> bool:
    del bg
    return patch.shape[1] > 1 and not np.array_equal(patch, _flip_patch_horiz(patch))


def _random_visible_transform(coords, color, bg):
    candidates = []
    for rot in range(4):
        for flip in (False, True):
            transformed = _transform_shape(coords, rot, flip)
            patch = _coords_to_patch(transformed, color, bg)
            if _is_hflip_visible_patch(patch, bg):
                candidates.append((transformed, patch))
    if not candidates:
        raise ValueError("shape has no visibly flippable orientation")
    return random.choice(candidates)


def _can_place_with_gap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != int(bg):
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
        for _ in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon in canon_used:
                continue
            try:
                samples = []
                for j in range(gsz):
                    coords, patch = _random_visible_transform(base, avail[ci + j], bg)
                    samples.append((coords, patch))
            except ValueError:
                continue
            canon_used.add(canon)
            for j, (coords, patch) in enumerate(samples):
                objs.append(
                    {
                        "coords": coords,
                        "patch": patch,
                        "color": avail[ci + j],
                        "shared": True,
                    }
                )
            ci += gsz
            break
        else:
            raise ValueError("shared shape not found")

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            color = avail[ci]
            try:
                coords, patch = _random_visible_transform(shape, color, bg)
            except ValueError:
                continue
            canon_used.add(canon)
            objs.append(
                {
                    "coords": coords,
                    "patch": patch,
                    "color": color,
                    "shared": False,
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        out_patch = _flip_patch_horiz(obj["patch"])
        h, w = out_patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            if _can_place_with_gap(out, out_patch, dest_r0, c0, bg):
                paste(out, out_patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                obj["out_patch"] = out_patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
        target_r0 = 0 if obj["shared"] else H - h
        placed = False
        rows = list(range(0, H - h + 1))
        random.shuffle(rows)
        for r0 in rows:
            if r0 == target_r0:
                continue
            if _can_place_with_gap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed and _can_place_with_gap(inp, patch, target_r0, c0, bg):
            paste(inp, patch, target_r0, c0, transparent=bg)
            placed = True
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
    raise RuntimeError("failed to generate input for 62593bfd D5 variant")

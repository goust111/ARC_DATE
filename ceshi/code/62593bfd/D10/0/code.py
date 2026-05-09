# Variant of 62593bfd
# Dimension: D10 — canonical形状的等价变换群 (match_cond)
# Change: D4二面体群——包含0/90/180/270度旋转和水平翻转的8种变换，两个形状若在这8种变换下有相同的归一化坐标则视为等价 → 仅平移等价（无旋转无翻转）

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

"""
design_analysis:
  source_task: none
  variant_core:
    old_rule: objects are grouped by a D4-invariant canonical shape key
    new_rule: objects are grouped only by exact normalized coordinates; rotation and reflection no longer match
  concrete_pixel_behavior:
    - Each object is still a monochromatic 8-connected component with preserved color and preserved horizontal anchor c0.
    - Objects whose exact normalized shape appears >=2 times move to the top edge.
    - Objects whose exact normalized shape appears only once move to the bottom edge.
    - Rotated/flipped copies of the same base shape are now different groups unless their occupied coordinates are identical after translation normalization.
  behavior_trace:
    example:
      base_shape: [(0,0),(1,0),(1,1)]
      rotated_shape: [(0,0),(0,1),(1,0)]
      original_D4:
        - both shapes share one canonical key
        - frequency becomes 2
        - both objects move to top
      variant_translation_only:
        - normalized keys differ
        - each frequency becomes 1 unless an exact copy exists
        - both objects move to bottom
      visible_difference:
        - the same pair that used to be a duplicated top group under D4 now becomes unique bottom objects
  complexity_calibration:
    - Generator always includes at least one exact-duplicate group so the "move to top" branch remains present.
    - Generator also includes one "orientation trap" family: 2-3 objects sharing one D4 family but all using distinct exact orientations.
    - This keeps the variant observable: the sample set contains both top objects and bottom objects, and at least one bottom object would have been top in the original task.
  hidden_assumptions_checked:
    - main() previously depended on _canonical_shape being D4-invariant; this must change to pure translation normalization.
    - c0 preservation remains valid because output placement still uses the original bounding-box left column.
    - find_components(..., monochromatic=False, connectivity=8) merges touching objects even across colors, so generator must forbid 8-neighbor contact, not just overlap.
    - patch extraction via crop_bbox + mask remains valid because objects stay monochromatic and connected.
  placement_legality:
    - Output is built first, with top-row exact duplicates and bottom-row unique objects.
    - Placement rejects overlap and 8-neighbor touching, preventing accidental component merges.
    - Input scattering preserves each object's c0 and uses interior rows different from the solved row when possible, so movement is visible.
  visual_observability:
    - Orientation-trap shapes are required to have multiple distinct D4 images, so rotation/reflection differences are visible from 2-4 examples.
    - Colors are distinct per object, but color never influences grouping; only visible geometry does.
  cleanliness:
    - Only one layer of objects exists on the background.
    - Objects are separated by at least one background cell in the 8-neighborhood, so component boundaries are easy to read.
  solver_assumptions_after_change:
    - Every intended object is an isolated 8-connected component.
    - Preserving c0 does not send any patch out of bounds because generator places using the same width in input and output.
    - At least one exact duplicate group and at least one exact unique object exist in each generated sample.
"""


def _translation_shape_key(coords):
    """Normalize a shape by translation only."""
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _canonical_shape_d4(coords):
    """D4-invariant key kept only for generator-side anti-degeneracy checks."""
    pts = [(int(r), int(c)) for r, c in coords]
    best = None
    cur = list(pts)
    for _ in range(4):
        key = _translation_shape_key(cur)
        best = key if best is None or key < best else best
        key = _translation_shape_key([(-r, c) for r, c in cur])
        best = key if key < best else best
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
    Sort objects to top or bottom edge by whether their exact shape
    (translation only, with no rotation/reflection equivalence) is
    shared with another object.
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
                "shape_key": _translation_shape_key(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

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
                if max(all_r) - min(all_r) < max_dim and max(all_c) - min(all_c) < max_dim:
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    return list(_translation_shape_key(cells))


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    return list(_translation_shape_key(pts))


def _distinct_d4_images(coords):
    seen = {}
    for rot in range(4):
        for flip in (False, True):
            transformed = _transform_shape(coords, rot, flip)
            key = _translation_shape_key(transformed)
            if key not in seen:
                seen[key] = transformed
    return list(seen.values())


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _ in coords)
    mc = max(c for _, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _no_contact_or_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(patch != bg)
    for pr, pc in coords:
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if grid[rr, cc] != bg:
            return False
        rr0 = max(0, rr - 1)
        rr1 = min(H, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(W, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != bg):
            return False
    return True


def _pick_orientation_family(size, d4_used):
    for _ in range(300):
        area = random.randint(4, 10)
        base = _grow_shape(area, 8)
        d4_key = _canonical_shape_d4(base)
        if d4_key in d4_used:
            continue
        images = _distinct_d4_images(base)
        if len(images) < size:
            continue
        random.shuffle(images)
        return d4_key, images[:size]
    raise ValueError("orientation trap not found")


def _pick_fresh_shape(exact_used, d4_used):
    for _ in range(300):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        exact_key = _translation_shape_key(shape)
        d4_key = _canonical_shape_d4(shape)
        if exact_key in exact_used or d4_key in d4_used:
            continue
        return shape, exact_key, d4_key
    raise ValueError("fresh shape not found")


def _scatter_rows(H, h, solved_r0):
    interior = [r for r in range(1, H - h) if r != solved_r0]
    if interior:
        random.shuffle(interior)
        return interior
    rows = [r for r in range(0, H - h + 1) if r != solved_r0]
    random.shuffle(rows)
    return rows


_CONFIGS = [
    ([2], 2, 0),
    ([2], 2, 1),
    ([2], 3, 0),
    ([3], 2, 0),
    ([3], 2, 1),
    ([2, 2], 2, 0),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    dup_group_sizes, trap_size, n_extra_unique = random.choice(_CONFIGS)
    total_objects = sum(dup_group_sizes) + trap_size + n_extra_unique
    if total_objects > len(avail):
        raise ValueError("not enough colors")

    objs = []
    exact_used = set()
    d4_used = set()
    color_idx = 0

    for gsz in dup_group_sizes:
        shape, exact_key, d4_key = _pick_fresh_shape(exact_used, d4_used)
        exact_used.add(exact_key)
        d4_used.add(d4_key)
        for _ in range(gsz):
            objs.append({"coords": shape, "color": avail[color_idx], "shared": True})
            color_idx += 1

    trap_d4_key, trap_shapes = _pick_orientation_family(trap_size, d4_used)
    d4_used.add(trap_d4_key)
    for shape in trap_shapes:
        exact_key = _translation_shape_key(shape)
        if exact_key in exact_used:
            raise ValueError("trap exact collision")
        exact_used.add(exact_key)
        objs.append({"coords": shape, "color": avail[color_idx], "shared": False, "trap": True})
        color_idx += 1

    for _ in range(n_extra_unique):
        shape, exact_key, d4_key = _pick_fresh_shape(exact_used, d4_used)
        exact_used.add(exact_key)
        d4_used.add(d4_key)
        objs.append({"coords": shape, "color": avail[color_idx], "shared": False})
        color_idx += 1

    out = np.full((H, W), bg, dtype=int)
    random.shuffle(objs)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        cols = list(range(0, W - w + 1))
        random.shuffle(cols)
        for c0 in cols:
            if _no_contact_or_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                obj["patch"] = patch
                obj["dest_r0"] = dest_r0
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    random.shuffle(objs)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
        placed = False
        for r0 in _scatter_rows(H, h, obj["dest_r0"]):
            if _no_contact_or_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    exact_freq = Counter(_translation_shape_key(obj["coords"]) for obj in objs)
    d4_freq = Counter(_canonical_shape_d4(obj["coords"]) for obj in objs)
    changed = sum(
        (exact_freq[_translation_shape_key(obj["coords"])] >= 2)
        != (d4_freq[_canonical_shape_d4(obj["coords"])] >= 2)
        for obj in objs
    )
    assert changed >= trap_size, "variant behavior not expressed strongly enough"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D10 variant")

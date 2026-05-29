# Combo variant of 62593bfd (index 16)
# combo_spec: D1/3+D2/1
# Name: 仅旋转等价分组 + 水平分流

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior trace
--------------------------
- Original task:
  - Group each 8-connected object by a D4-invariant canonical shape
    (rotations + mirror).
  - duplicated -> move to top row 0, keep input c0.
  - unique -> move to bottom row H-h, keep input c0.
- Combo task:
  - Group by rotations only. Mirror images are different unless a rotation
    already matches them.
  - duplicated -> move to left edge col 0, keep input r0.
  - unique -> move to right edge col W-w, keep input r0.
- Small trace:
  - Suppose object A is an L-shape and object B is the mirrored L.
  - Under the original D4 rule, A and B share one canonical key, so both are
    duplicated; the old task would send them to the same vertical branch.
  - Under this combo, A and B have different rotation-only keys, so each is
    unique; they now go to the right edge instead.
  - A true rotation duplicate group still exists and goes to the left edge.
  - Visible consequence: handedness matters, and the routing axis is horizontal.

b) Complexity calibration
-------------------------
- If the generator used only rotation duplicates, D1/3 would be invisible.
- If the generator used only mirror pairs but no shared shapes, the left branch
  would be absent and the rule would look incomplete.
- Therefore each sample includes:
  - at least one rotation-duplicate group, so duplicated->left is supported;
  - at least one chiral mirror pair that is D4-equal but rotation-distinct, so
    the new grouping visibly disagrees with D4-based grouping;
  - at most 0-2 extra unique shapes, keeping the object count within 4-6.

c) Implicit assumption audit for main()
--------------------------------------
- The original solver stored `c0` because it preserved columns and changed rows.
  That assumption breaks here; the combo must store `r0` instead.
- The original `dest_r0 = 0 / H-h` hard-coded vertical routing; the combo must
  compute `dest_c0 = 0 / W-w`.
- Cropping the component bbox and reconstructing a sparse patch still works,
  because the combo changes only grouping semantics and destination anchor.
- `find_components(..., monochromatic=False, connectivity=8)` merges even
  differently colored objects if they touch diagonally. The generator must keep
  every object 8-disconnected in both input and output.

d) Placement legality
---------------------
- Output-first generation is retained.
- Objects are placed into separate row bands with at least one blank row
  between bands, so left/right routing never creates overlap or 8-neighbor
  contact.
- Input keeps the same row bands and only randomizes columns, preserving the
  combo rule that row position is invariant.

e) Visual observability
-----------------------
- Human-solvable signal 1: some same-family shapes that are mirror images no
  longer travel together; they split to the right edge.
- Human-solvable signal 2: true rotation duplicates still travel together, now
  to the left edge.
- Input scattering forces at least one shared object not already at the left
  edge and at least one unique object not already at the right edge, so the
  horizontal motion is visible instead of degenerate.

f) Visual tidiness
------------------
- No extra markers are added; only isolated monochrome objects on background.
- Distinct colors per object help object tracking but do not encode the rule.
- Row-band packing avoids clutter and accidental component merging.

g) New solver assumptions and generator guarantees
--------------------------------------------------
- main() assumes every foreground object is an isolated 8-connected component;
  generate_input() guarantees this by row-band separation.
- main() assumes at least one duplicated and one unique group can exist; the
  generator explicitly constructs both.
- main() itself does not require the D1 difference to be visible, but the
  generator guarantees visibility by embedding a chiral mirror pair and checking
  that a D4-horizontal ablation would yield a different output.
"""

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _normalize_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _canonical_shape_rot(coords):
    """Canonical key under C4 rotations only."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        norm = _normalize_coords(cur)
        best = norm if best is None or norm < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _canonical_shape_d4(coords):
    """Canonical key under the full D4 dihedral group."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        norm = _normalize_coords(cur)
        best = norm if best is None or norm < best else best
        refl = _normalize_coords([(-r, c) for r, c in cur])
        best = refl if refl < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _solve_with_key(grid: np.ndarray, shape_key_fn) -> np.ndarray:
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
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "r0": int(r0),
                "pat": pat,
                "shape_key": shape_key_fn(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)
    for d in extracted:
        pat = d["pat"]
        _h, w = map(int, pat.shape)
        dest_c0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(W - w)
        paste(out, pat, int(d["r0"]), dest_c0, transparent=bg)
    return out


def _relative_component_shapes(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    rels = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        rels.append([(int(r) - r0, int(c) - c0) for r, c in comp["coords"]])
    return rels


def _count_mirror_only_rights(grid: np.ndarray) -> int:
    """
    Count objects that are unique under rotation-only grouping but duplicated
    under D4 grouping. In this combo these are the key "mirror pair" objects:
    they go RIGHT here, but would go LEFT in the D4-horizontal ablation.
    """
    rels = _relative_component_shapes(grid)
    rot_keys = [_canonical_shape_rot(rel) for rel in rels]
    d4_keys = [_canonical_shape_d4(rel) for rel in rels]
    rot_freq = Counter(rot_keys)
    d4_freq = Counter(d4_keys)
    return sum(
        1
        for rot_key, d4_key in zip(rot_keys, d4_keys)
        if int(rot_freq[rot_key]) == 1 and int(d4_freq[d4_key]) >= 2
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route objects horizontally using rotation-only shape equivalence.

    Rule:
    - Compute the canonical shape of each 8-connected component using only the
      4 rotations.
    - duplicated shapes (freq >= 2) -> LEFT edge (col 0).
    - unique shapes     (freq == 1) -> RIGHT edge (col W-w).
    - Preserve each object's top row r0.
    """
    return _solve_with_key(grid, _canonical_shape_rot)


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _cc in cells] + [nr]
                all_c = [cc for _rr, cc in cells] + [nc]
                if max(all_r) - min(all_r) < max_dim and max(all_c) - min(all_c) < max_dim:
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    return list(_normalize_coords(cells))


def _transform_shape(coords, rot=0, flip=False):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    return list(_normalize_coords(pts))


def _is_chiral(coords):
    return _canonical_shape_rot(coords) != _canonical_shape_rot(
        _transform_shape(coords, flip=True)
    )


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _pack_rows_with_gaps(heights, H, gap=1):
    """Assign row bands with a blank row between neighboring objects."""
    n = len(heights)
    required = sum(int(h) for h in heights) + max(0, n - 1) * int(gap)
    if required > int(H):
        raise ValueError("insufficient height for row packing")

    slack = int(H - required)
    outer_gaps = [0] * (n + 1)
    for _ in range(slack):
        outer_gaps[random.randrange(n + 1)] += 1

    rows = []
    cur = outer_gaps[0]
    for idx, h in enumerate(heights):
        rows.append(int(cur))
        cur += int(h)
        if idx < n - 1:
            cur += int(gap) + outer_gaps[idx + 1]
        else:
            cur += outer_gaps[-1]
    return rows


_CONFIGS = [
    ([2], 0),
    ([2], 1),
    ([2], 2),
    ([3], 0),
    ([3], 1),
    ([2, 2], 0),
]


def _make_rotation_duplicate_group(group_size, rot_used, d4_used):
    for _ in range(300):
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        rot_key = _canonical_shape_rot(base)
        d4_key = _canonical_shape_d4(base)
        if rot_key in rot_used or d4_key in d4_used:
            continue
        rot_used.add(rot_key)
        d4_used.add(d4_key)
        return [
            {
                "coords": _transform_shape(base, rot=random.randint(0, 3), flip=False),
                "shared": True,
                "kind": "rotation_duplicate",
            }
            for _ in range(int(group_size))
        ]
    raise ValueError("rotation duplicate group not found")


def _make_mirror_unique_pair(rot_used, d4_used):
    for _ in range(400):
        area = random.randint(4, 12)
        base = _grow_shape(area, 8)
        if not _is_chiral(base):
            continue
        mirror = _transform_shape(base, flip=True)
        base_rot = _canonical_shape_rot(base)
        mirror_rot = _canonical_shape_rot(mirror)
        d4_key = _canonical_shape_d4(base)
        if base_rot == mirror_rot:
            continue
        if base_rot in rot_used or mirror_rot in rot_used or d4_key in d4_used:
            continue
        rot_used.add(base_rot)
        rot_used.add(mirror_rot)
        d4_used.add(d4_key)
        return [
            {
                "coords": _transform_shape(base, rot=random.randint(0, 3), flip=False),
                "shared": False,
                "kind": "mirror_unique",
            },
            {
                "coords": _transform_shape(mirror, rot=random.randint(0, 3), flip=False),
                "shared": False,
                "kind": "mirror_unique",
            },
        ]
    raise ValueError("mirror pair not found")


def _make_extra_unique(rot_used, d4_used):
    for _ in range(300):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        rot_key = _canonical_shape_rot(shape)
        d4_key = _canonical_shape_d4(shape)
        if rot_key in rot_used or d4_key in d4_used:
            continue
        rot_used.add(rot_key)
        d4_used.add(d4_key)
        return {"coords": shape, "shared": False, "kind": "extra_unique"}
    raise ValueError("extra unique shape not found")


def _gen_attempt():
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    group_sizes, n_extra_unique = random.choice(_CONFIGS)
    objs = []
    rot_used = set()
    d4_used = set()

    for group_size in group_sizes:
        objs.extend(_make_rotation_duplicate_group(group_size, rot_used, d4_used))
    objs.extend(_make_mirror_unique_pair(rot_used, d4_used))
    for _ in range(n_extra_unique):
        objs.append(_make_extra_unique(rot_used, d4_used))

    if len(objs) > len(avail):
        raise ValueError("not enough colors")

    for color, obj in zip(avail, objs):
        obj["color"] = int(color)
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = map(int, obj["patch"].shape)
        obj["h"] = int(h)
        obj["w"] = int(w)

    min_H = max(19, sum(obj["h"] for obj in objs) + len(objs) - 1)
    if min_H > 30:
        raise ValueError("objects too tall for clean horizontal routing")
    H = random.randint(min_H, 30)
    W = random.randint(19, 30)

    random.shuffle(objs)
    packed_rows = _pack_rows_with_gaps([obj["h"] for obj in objs], H, gap=1)

    out = np.full((H, W), bg, dtype=int)
    for obj, r0 in zip(objs, packed_rows):
        dest_c0 = 0 if obj["shared"] else int(W - obj["w"])
        paste(out, obj["patch"], int(r0), dest_c0, transparent=bg)
        obj["r0"] = int(r0)
        obj["dest_c0"] = int(dest_c0)

    inp = np.full((H, W), bg, dtype=int)
    shared_shifted = False
    unique_shifted = False
    for obj in objs:
        c0 = random.randint(0, W - obj["w"])
        paste(inp, obj["patch"], int(obj["r0"]), int(c0), transparent=bg)
        obj["input_c0"] = int(c0)
        if obj["shared"] and c0 != 0:
            shared_shifted = True
        if (not obj["shared"]) and c0 != int(W - obj["w"]):
            unique_shifted = True

    if not shared_shifted or not unique_shifted:
        raise ValueError("scatter did not visibly exercise both horizontal branches")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    legacy_horizontal = _solve_with_key(inp, _canonical_shape_d4)
    assert not np.array_equal(
        legacy_horizontal, out
    ), "generator degenerated to D4-horizontal behavior"
    assert _count_mirror_only_rights(inp) >= 2, "mirror-pair signal missing"

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo 16")

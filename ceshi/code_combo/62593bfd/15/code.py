# Combo variant of 62593bfd (index 15)
# combo_spec: D1/2+D5/0
# Name: 外接矩形分组 + 旋转对齐
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
#    - Base task:
#      * extract each 8-connected foreground object;
#      * group by exact D4 shape class;
#      * shared shape -> move to top unchanged;
#      * unique shape -> move to bottom unchanged.
#    - Fused task:
#      * still extract the same objects and preserve each object's left bbox
#        column c0;
#      * replace the grouping key with tight bbox size (height, width);
#      * bbox size seen >= 2 -> object belongs to the top group;
#      * bbox size seen once -> object belongs to the bottom group;
#      * top-group objects are additionally rotated to their own canonical
#        rotation pose, defined as the lexicographically smallest of the 4
#        rotation-only poses; bottom-group objects stay unchanged.
#    - Small trace:
#      * object A: a 2x3 zigzag, object B: a different 2x3 L-like shape,
#        object C: a 4x2 unique shape.
#      * original task: if A/B/C have distinct D4 canons, all three go bottom.
#      * fused task: A and B share bbox size 2x3, so both go top; each is
#        rotated to its own rotation-canonical pose before pasting. C keeps its
#        original pose and goes bottom.
#      * visible difference: "same bbox size" decides top/bottom, and top
#        objects may change orientation while bottom objects never do.
# b) Complexity calibration
#    - The D1 change makes grouping easier, so the generator compensates by
#      forcing duplicated bbox groups to contain different exact shapes. This
#      keeps the bbox rule visually necessary.
#    - The D5 change can become invisible if every top object is already in its
#      canonical rotation or rotationally symmetric. The generator therefore
#      requires at least one duplicated-size object whose input pose is a
#      non-canonical rotation and whose 4 rotations are all distinct.
#    - To keep the rule inferable from 2-4 examples, all colors remain stable
#      and no reflection branch is introduced: only top objects rotate.
# c) Implicit assumptions audited against `main()`
#    - `find_components(... monochromatic=False)` merges touching non-bg pixels
#      even across colors. The generator must guarantee 8-neighborhood
#      separation so each intended object stays isolated.
#    - `crop_bbox(...)` must still return the tight bbox; this is now the group
#      key, so generator-side shapes are grown with exact intended bbox sizes.
#    - Old `main()` assumed top objects were pasted unchanged. That no longer
#      holds, so the solver stores both the cropped patch and relative coords,
#      computes the canonical rotation for duplicated bbox groups, then rotates
#      the patch before pasting.
#    - Preserving c0 remains valid, but rotating can swap width/height. The
#      generator therefore chooses c0 using the max of input/output widths.
# d) Placement legality
#    - Output is built first, because the solver's top/bottom target rows are
#      deterministic once bbox frequencies and canonical rotations are known.
#    - A placement is legal only if the patch fits in-bounds and no foreground
#      pixel touches existing foreground in its 8-neighborhood.
#    - Input is produced by vertically scattering the same objects while keeping
#      c0 fixed and preferring rows away from the final edge rows.
# e) Visual observability
#    - Humans can see that different silhouettes sharing the same bbox still
#      move to the same edge, which exposes the bbox grouping rule.
#    - They can also compare a duplicated-size object before/after and observe
#      that only the top objects rotate into a standard pose.
# f) Cleanliness
#    - Distinct colors per object and one-cell 8-neighborhood clearance avoid
#      accidental merges and reduce clutter in the top/bottom landing zones.
# g) Color inferability
#    - No color mapping is used. Colors only track object identity.
# h) Anti-degeneration guarantees
#    - Every generated sample has at least one duplicated bbox group and at
#      least one unique bbox size.
#    - All objects have distinct D4 canonical shapes, so the fused grouping
#      cannot collapse back to the original shape-grouping rule.
#    - At least one duplicated-size object starts in a non-canonical rotation,
#      so the fused output differs from the D1-only bbox variant as well.
# i) New solver assumptions checked against generator
#    - For duplicated bbox groups, the canonical rotation pose must be defined
#      from rotations only: always true by construction.
#    - Rotated top objects must still fit at preserved c0: generator enforces
#      this with max(input_width, output_width) when placing outputs.
#    - No overlap after movement is assumed by the solver and guaranteed by the
#      output-first placement checks.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _norm_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _rotate_coords_cw(coords):
    return [(int(c), -int(r)) for r, c in coords]


def _rotation_poses(coords):
    cur = [(int(r), int(c)) for r, c in coords]
    poses = []
    for rot in range(4):
        poses.append((_norm_coords(cur), int(rot)))
        cur = _rotate_coords_cw(cur)
    return poses


def _canonical_rotation_pose(coords):
    return min(_rotation_poses(coords), key=lambda item: item[0])


def _canonical_shape(coords):
    """D4 canonical key kept for generator anti-degeneracy checks."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        n = _norm_coords(cur)
        best = n if best is None or n < best else best
        n = _norm_coords([(-int(r), int(c)) for r, c in cur])
        best = n if n < best else best
        cur = _rotate_coords_cw(cur)
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


def _rotate_patch_cw(patch, rot):
    return np.rot90(patch, -int(rot) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Measure each 8-connected component's tight bbox size.
    - Shared bbox sizes (freq >= 2) move to row 0 and are rotated into the
      lexicographically smallest rotation-only pose of that object.
    - Unique bbox sizes (freq == 1) move to the bottom unchanged.
    - The left bbox column c0 is preserved.
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
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "rel": rel,
                "pat": pat,
                "bbox_key": tuple(map(int, pat.shape)),
            }
        )

    freq = Counter(d["bbox_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        shared = int(freq[d["bbox_key"]]) >= 2
        pat = d["pat"]
        if shared:
            _std_pose, rot = _canonical_rotation_pose(d["rel"])
            pat = _rotate_patch_cw(pat, rot)
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if shared else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]

_DUP_DIMS = [
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
]
_ALL_DIMS = [(h, w) for h in range(2, 6) for w in range(2, 6)]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _touches_foreground(grid, r, c, bg):
    H, W = grid.shape
    for rr in range(max(0, int(r) - 1), min(H, int(r) + 2)):
        for cc in range(max(0, int(c) - 1), min(W, int(c) + 2)):
            if int(grid[rr, cc]) != int(bg):
                return True
    return False


def _can_place(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != bg):
        if _touches_foreground(grid, int(r0 + pr), int(c0 + pc), bg):
            return False
    return True


def _seed_path_for_bbox(h, w):
    """Random monotone 8-connected path from (0,0) to (h-1,w-1)."""
    r, c = 0, 0
    cells = {(0, 0)}
    while r < h - 1 or c < w - 1:
        moves = []
        if r < h - 1:
            moves.append((r + 1, c))
        if c < w - 1:
            moves.append((r, c + 1))
        if r < h - 1 and c < w - 1:
            moves.append((r + 1, c + 1))
        r, c = random.choice(moves)
        cells.add((r, c))
    return cells


def _grow_shape_in_bbox(h, w):
    """Grow a random 8-connected shape whose tight bbox is exactly (h,w)."""
    max_area = min(12, h * w)
    seed = _seed_path_for_bbox(h, w)
    min_area = max(3, len(seed))
    target_area = random.randint(min_area, max_area)

    cells = set(seed)
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    coords = sorted((int(r), int(c)) for r, c in cells)
    assert max(r for r, _c in coords) == h - 1
    assert max(c for _r, c in coords) == w - 1
    return coords


def _has_four_distinct_rotations(coords):
    return len({pose for pose, _rot in _rotation_poses(coords)}) == 4


def _sample_dup_object(h, w, used_canons, require_changed):
    for _ in range(500):
        coords = _grow_shape_in_bbox(h, w)
        canon = _canonical_shape(coords)
        if canon in used_canons:
            continue
        if not _has_four_distinct_rotations(coords):
            continue
        std_pose, _rot = _canonical_rotation_pose(coords)
        changed = _norm_coords(coords) != std_pose
        if require_changed and not changed:
            continue
        return {
            "input_coords": list(_norm_coords(coords)),
            "output_coords": list(std_pose),
            "canon": canon,
            "rotated": changed,
        }
    raise ValueError(f"duplicated-size shape not found for dims {(h, w)}")


def _sample_unique_object(h, w, used_canons):
    for _ in range(400):
        coords = _grow_shape_in_bbox(h, w)
        canon = _canonical_shape(coords)
        if canon in used_canons:
            continue
        return {
            "input_coords": list(_norm_coords(coords)),
            "output_coords": list(_norm_coords(coords)),
            "canon": canon,
            "rotated": False,
        }
    raise ValueError(f"unique-size shape not found for dims {(h, w)}")


def _choose_rows(H, h, forbidden):
    rows = list(range(0, int(H - h + 1)))
    preferred = [r for r in rows if r not in forbidden]
    random.shuffle(preferred)
    leftovers = [r for r in rows if r in forbidden]
    random.shuffle(leftovers)
    return preferred + leftovers


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    num_dup_groups = len(group_sizes)

    dup_dim_choices = [dims for dims in _DUP_DIMS if dims[0] <= H and dims[1] <= W]
    if len(dup_dim_choices) < num_dup_groups:
        raise ValueError("not enough duplicated dims choices")
    dup_dims = random.sample(dup_dim_choices, num_dup_groups)

    used_dims = set(dup_dims)
    unique_dim_choices = [
        dims
        for dims in _ALL_DIMS
        if dims not in used_dims and dims[0] <= H and dims[1] <= W
    ]
    if len(unique_dim_choices) < n_unique:
        raise ValueError("not enough unique dims choices")
    unique_dims = random.sample(unique_dim_choices, n_unique)

    objs = []
    used_canons = set()
    changed_shared = 0

    for group_idx, (dims, gsz) in enumerate(zip(dup_dims, group_sizes)):
        h, w = map(int, dims)
        for obj_idx in range(int(gsz)):
            sampled = _sample_dup_object(
                h,
                w,
                used_canons,
                require_changed=(group_idx == 0 and obj_idx == 0),
            )
            used_canons.add(sampled["canon"])
            objs.append(
                {
                    "shared": True,
                    "color": int(avail[ci]),
                    "bbox_key": (h, w),
                    "input_coords": sampled["input_coords"],
                    "output_coords": sampled["output_coords"],
                    "rotated": bool(sampled["rotated"]),
                }
            )
            changed_shared += int(sampled["rotated"])
            ci += 1

    for dims in unique_dims:
        h, w = map(int, dims)
        sampled = _sample_unique_object(h, w, used_canons)
        used_canons.add(sampled["canon"])
        objs.append(
            {
                "shared": False,
                "color": int(avail[ci]),
                "bbox_key": (h, w),
                "input_coords": sampled["input_coords"],
                "output_coords": sampled["output_coords"],
                "rotated": False,
            }
        )
        ci += 1

    if changed_shared == 0:
        raise ValueError("degenerate sample without visible top rotation")

    random.shuffle(objs)
    out = np.full((H, W), bg, dtype=int)

    for obj in objs:
        obj["input_patch"] = _coords_to_patch(obj["input_coords"], obj["color"], bg)
        obj["output_patch"] = _coords_to_patch(obj["output_coords"], obj["color"], bg)
        in_h, in_w = obj["input_patch"].shape
        out_h, out_w = obj["output_patch"].shape
        max_w = max(int(in_w), int(out_w))
        dest_r0 = 0 if obj["shared"] else int(H - out_h)

        placed = False
        for _ in range(500):
            c0 = random.randint(0, int(W - max_w))
            if _can_place(out, obj["output_patch"], dest_r0, c0, bg):
                paste(out, obj["output_patch"], dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["input_patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        forbidden = {0} if obj["shared"] else {int(H - h)}

        placed = False
        for r0 in _choose_rows(H, h, forbidden):
            if _can_place(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D1/2+D5/0")

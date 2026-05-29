# Combo variant of 62593bfd (index 9)
# combo_spec: D5/0+D12/0
# Name: duplicated旋转对齐 + 同组同色

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

try:
    from arc_utils import crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base task: group 8-connected objects by D4-invariant canonical shape.
#      Repeated groups move to the top, unique groups move to the bottom, and
#      each object keeps its own orientation and color.
#    - Combo behavior:
#      1. repeated-shape groups still move to the top, unique groups still move
#         to the bottom, and c0 is still preserved;
#      2. repeated-shape objects are additionally rotated so every member of the
#         group lands in the same lexicographically minimal rotation pose;
#      3. objects in the same shape group share one color, and different groups
#         use different colors, so the color visually confirms the grouping.
#    - Small trace:
#      two same-colored L-shapes that are rotations of each other move to row 0;
#      in the original task they would still look different after moving, while
#      here both become the same upright L in the output. A singleton shape with
#      its own private color still moves to the bottom unchanged.
# b) Complexity calibration:
#    - Shared groups are restricted to rotation-asymmetric shapes with 4 distinct
#      rotations, and at least one shared member starts non-canonical. This makes
#      the rotation-normalization visibly necessary instead of accidental.
#    - Group colors are assigned per shape group, not per object, so there are
#      only 2-4 non-background colors in a sample and each repeated color has at
#      least two supporting examples.
# c) Main() assumption audit:
#    - `find_components(... monochromatic=False)` from the base task is no longer
#      safe because repeated groups now reuse colors; same-color objects must stay
#      separate components, so extraction switches to `monochromatic=True`.
#    - Pasting the raw cropped patch unchanged is no longer valid for repeated
#      groups because those objects rotate before output. We therefore keep the
#      relative coords, compute the canonical rotation pose, rotate the isolated
#      patch, and then paste it.
#    - Preserving `c0` remains valid, but rotated width may differ from input
#      width; the generator must choose c0 that fits both.
# d) Placement legality:
#    - Output is built first. Shared objects use their canonical-rotation patch at
#      row 0; unique objects keep their input pose at the bottom.
#    - Input then scatters the pre-output objects vertically at the same c0.
#    - Same-color objects are never allowed to touch in 8-neighborhood; otherwise
#      monochromatic component extraction would merge them.
# e/f/g) Observability, cleanliness, inferability:
#    - Repeated colors expose the group structure, and repeated-shape members at
#      the top become literally identical after rotation, so both fused changes
#      are visible in 2-3 examples.
#    - Using one color per group reduces clutter; preventing same-color contact
#      keeps repeated-color groups readable as multiple objects.
#    - The color rule is structural rather than numeric: "same shape group => same
#      color, different group => different color", which is human-inferable.
# h) Anti-degeneration checks:
#    - Every generated sample has at least one repeated group and one unique group
#      from `_CONFIGS`.
#    - At least one repeated-group object starts in a non-canonical rotation, so
#      this combo does not collapse to the base task.
# i) New solver assumptions covered by the generator:
#    - Repeated groups are generated without reflection, so rotation-only
#      normalization is sufficient.
#    - Same-color objects remain 8-disconnected in both input and output.
#    - Canonical shapes are unique across groups, so the repeated/unique split is
#      unambiguous.


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
    """Canonical shape key invariant under the full D4 group."""
    pts = [(int(r), int(c)) for r, c in coords]
    cur = list(pts)
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


def _recolor_patch(patch, color, bg):
    out = np.full_like(patch, int(bg))
    out[patch != int(bg)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Extract monochrome 8-connected components so same-color group members stay
      separate objects.
    - Group objects by D4-invariant canonical shape.
    - Repeated groups (freq >= 2) move to row 0 and rotate into the group's
      lexicographically minimal rotation pose.
    - Unique groups (freq == 1) move to the bottom unchanged.
    - In all cases c0 is preserved, and group colors remain attached to the
      objects, yielding same-color repeated groups in the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, int(bg))
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "color": int(comp["color"]),
                "rel": rel,
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        shared = int(freq[d["shape_key"]]) >= 2
        pat = d["pat"]
        if shared:
            _std_pose, rot = _canonical_rotation_pose(d["rel"])
            pat = _rotate_patch_cw(pat, rot)
        pat = _recolor_patch(pat, d["color"], bg)
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if shared else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells within `max_dim`."""
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
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _cc in cells] + [nr]
                all_c = [cc for _rr, cc in cells] + [nc]
                if max(all_r) - min(all_r) < int(max_dim) and max(all_c) - min(all_c) < int(max_dim):
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    return list(_norm_coords(cells))


def _transform_shape(coords, rot, flip):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(rot) % 4):
        pts = _rotate_coords_cw(pts)
    if flip:
        pts = [(-int(r), int(c)) for r, c in pts]
    return list(_norm_coords(pts))


def _coords_to_patch(coords, color, bg):
    mr = max(int(r) for r, _c in coords)
    mc = max(int(c) for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if int(r0) < 0 or int(c0) < 0 or int(r0 + h) > H or int(c0 + w) > W:
        return False
    mask = patch != int(bg)
    return not np.any(grid[int(r0):int(r0) + h, int(c0):int(c0) + w][mask] != int(bg))


def _same_color_clear(grid, patch, r0, c0, bg):
    """Reject placements where this monochrome object would 8-touch same-color pixels."""
    cells = np.argwhere(patch != int(bg))
    if cells.size == 0:
        return False
    color = int(patch[tuple(cells[0])])
    H, W = grid.shape
    for dr, dc in cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(grid[r1:r2, c1:c2] == int(color)):
            return False
    return True


def _can_place(grid, patch, r0, c0, bg):
    return _no_overlap(grid, patch, r0, c0, bg) and _same_color_clear(grid, patch, r0, c0, bg)


def _has_four_distinct_rotations(coords):
    return len({pose for pose, _rot in _rotation_poses(coords)}) == 4


def _shared_shape_ok(coords):
    canon_d4 = _canonical_shape(coords)
    canon_rot, _rot = _canonical_rotation_pose(coords)
    return canon_d4 == canon_rot and _has_four_distinct_rotations(coords)


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


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
    avail = [c for c in range(10) if c != int(bg)]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    objs = []
    canon_used: set[tuple[tuple[int, int], ...]] = set()
    changed_shared = 0

    for gsz in group_sizes:
        for _ in range(400):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon in canon_used or not _shared_shape_ok(base):
                continue
            canon_used.add(canon)
            std_pose, _rot = _canonical_rotation_pose(base)
            std_coords = list(std_pose)
            group_color = int(avail[ci])
            ci += 1

            input_rots = [0, random.choice([1, 2, 3])]
            while len(input_rots) < int(gsz):
                input_rots.append(random.randint(0, 3))
            random.shuffle(input_rots)

            for rot_in in input_rots:
                input_coords = _transform_shape(std_coords, rot_in, False)
                objs.append(
                    {
                        "shared": True,
                        "color": group_color,
                        "shape_key": canon,
                        "input_coords": input_coords,
                        "output_coords": std_coords,
                        "rotated": int(rot_in) != 0,
                    }
                )
                changed_shared += int(rot_in != 0)
            break
        else:
            raise ValueError("shared shape not found")

    for _ in range(n_unique):
        for _ in range(400):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            group_color = int(avail[ci])
            ci += 1
            objs.append(
                {
                    "shared": False,
                    "color": group_color,
                    "shape_key": canon,
                    "input_coords": shape,
                    "output_coords": shape,
                    "rotated": False,
                }
            )
            break
        else:
            raise ValueError("unique shape not found")

    if changed_shared == 0:
        raise ValueError("degenerate sample without shared rotation")

    random.shuffle(objs)
    out = np.full((H, W), int(bg), dtype=int)

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
                paste(out, obj["output_patch"], dest_r0, c0, transparent=int(bg))
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), int(bg), dtype=int)
    for obj in objs:
        patch = obj["input_patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        forbidden = {0} if obj["shared"] else {int(H - h)}
        placed = False
        for r0 in _choose_rows(H, h, forbidden):
            if _can_place(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=int(bg))
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
    raise RuntimeError("failed to generate input for 62593bfd combo 9")

# Combo variant of 62593bfd (index 25)
# combo_spec: D5/0+D10/1
# Name: C4旋转等价 + duplicated旋转对齐

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - Base task: group 8-connected monochrome objects by D4 shape equivalence,
#   move duplicated groups to the top and unique groups to the bottom, with
#   pure translation only.
# - Combo change 1 (D10/1): grouping now uses C4 rotations only. A chiral L
#   and its mirror J no longer match, because reflection is forbidden.
# - Combo change 2 (D5/0): duplicated groups do not merely translate upward;
#   they are also rotated into one shared canonical C4 orientation.
# - Small trace:
#   object A = upright L, object B = same L rotated 90 deg, object C = mirror J
#   original D4 task:
#     A/B/C all share one D4 key, all move to the top, each keeps its input pose.
#   combo task:
#     A/B share one C4 key, so both move to the top and both become the same
#     canonical L orientation; C has a different C4 key, so it moves to bottom
#     unchanged. This makes both modified dimensions visible in one sample.
#
# b) Complexity calibration
# - Every generated sample contains:
#   1. one mandatory rotation-shared group of size 2 or 3, with at least one
#      member starting in a non-canonical orientation, so the top-row alignment
#      is visible;
#   2. one mandatory chiral mirror pair with the same D4 key but different C4
#      keys, so the exclusion of reflection is visible;
#   3. optionally 0-2 extra C4-unique shapes for variety while staying within
#      the original 4-6 object count.
# - Shared groups require enough distinct rotations to support their size, which
#   avoids invisible normalization on rotationally symmetric shapes.
#
# c) Implicit-assumption audit of main()
# - `find_components(..., monochromatic=False)` assumes objects never touch even
#   diagonally; otherwise different colors would merge into one component. The
#   generator enforces globally separated column bands with at least one blank
#   column between reserved bands.
# - The original solver assumed "movement is pure translation" and stored only a
#   cropped patch for later paste. That is no longer sufficient: duplicated
#   objects now need their relative coordinates so the required rotation can be
#   computed before paste.
# - Preserving `c0` is still part of the rule, but rotating a duplicated object
#   can change its width. The generator therefore reserves each object's column
#   band using `max(input_width, output_width)` so both input and output fit at
#   the same `c0`.
#
# d) Placement legality
# - Output is built first from the intended post-rule state.
# - Input is then scattered vertically while preserving each object's `c0`.
# - Reserved non-overlapping column bands guarantee no overlap, no out-of-bounds
#   rotation aftermath, and no accidental 8-neighbor merging in either grid.
#
# e) Visual observability
# - A human can infer the rule from a few examples because each sample shows both:
#   a top row where same-shape objects become identically oriented, and a bottom
#   row where a mirror-related object fails to join them.
# - No color-id arithmetic or hidden ordering is used; colors are identity cues.
#
# f) Visual cleanliness
# - Objects are single-color, disconnected, and horizontally spaced. The top and
#   bottom regions stay readable because there are only 4-6 objects and each fits
#   inside a 5x5 box.
#
# g) Color inferability
# - Colors never change and do not encode logic. Only position and orientation
#   change, so the transformation remains structurally inferable.
#
# h) Anti-degeneration guarantees
# - At least one duplicated object is rotated by the solver in every sample.
# - At least one chiral pair would have been merged under the old D4 rule but is
#   split under the new C4 rule, so the combo does not collapse to either single
#   change being visually absent.
#
# i) New solver assumptions and generator support
# - Assumption: some C4 key has multiplicity >= 2. Guaranteed by the mandatory
#   rotation-shared group.
# - Assumption: some object is unique under C4. Guaranteed by the chiral pair.
# - Assumption: the canonical orientation of each duplicated group is reachable by
#   rotation alone. Guaranteed because shared groups are constructed from one C4
#   rotation orbit, not via reflection.
# - Assumption: rotating at fixed `c0` still fits within the grid. Guaranteed by
#   width reservation during placement.

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _normalize_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _rotate_cw(coords):
    return [(int(c), -int(r)) for r, c in coords]


def _reflect_rows(coords):
    return [(-int(r), int(c)) for r, c in coords]


def _rotation_poses(coords):
    cur = [(int(r), int(c)) for r, c in coords]
    poses = []
    for rot in range(4):
        poses.append((_normalize_coords(cur), int(rot)))
        cur = _rotate_cw(cur)
    return poses


def _canonical_rotation_pose(coords):
    return min(_rotation_poses(coords), key=lambda item: item[0])


def _canonical_shape_c4(coords):
    return _canonical_rotation_pose(coords)[0]


def _canonical_shape_d4(coords):
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        key = _normalize_coords(cur)
        best = key if best is None or key < best else best
        mirror_key = _normalize_coords(_reflect_rows(cur))
        best = mirror_key if mirror_key < best else best
        cur = _rotate_cw(cur)
    return best


def _rotation_variants(coords):
    cur = [(int(r), int(c)) for r, c in coords]
    seen = set()
    out = []
    for _ in range(4):
        key = _normalize_coords(cur)
        if key not in seen:
            seen.add(key)
            out.append([tuple(cell) for cell in key])
        cur = _rotate_cw(cur)
    return out


def _is_chiral(coords):
    return _canonical_shape_c4(coords) != _canonical_shape_c4(_reflect_rows(coords))


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    H, W = map(int, shape)
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _rotate_patch_cw(patch, rot):
    return np.rot90(patch, -int(rot) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Group 8-connected components by canonical shape under C4 rotations only.
    - If a C4 shape occurs >= 2 times, move each object to the top and rotate it
      into the canonical rotation of that C4 class.
    - If a C4 shape occurs once, move the object to the bottom unchanged.
    - Preserve the left bbox column `c0` in both branches.
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
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, comp["bbox"])
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "rel": rel,
                "pat": pat,
                "shape_key": _canonical_shape_c4(rel),
            }
        )

    freq = Counter(item["shape_key"] for item in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for item in extracted:
        shared = int(freq[item["shape_key"]]) >= 2
        pat = item["pat"]
        if shared:
            _std_pose, rot = _canonical_rotation_pose(item["rel"])
            pat = _rotate_patch_cw(pat, rot)
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if shared else int(H - h)
        paste(out, pat, dest_r0, int(item["c0"]), transparent=int(bg))

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells within max_dim."""
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
    return [tuple(cell) for cell in _normalize_coords(cells)]


def _transform_shape(coords, rot=0, flip=False):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(rot) % 4):
        pts = _rotate_cw(pts)
    if flip:
        pts = _reflect_rows(pts)
    return [tuple(cell) for cell in _normalize_coords(pts)]


def _coords_to_patch(coords, color, bg):
    mr = max(int(r) for r, _c in coords)
    mc = max(int(c) for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _random_partition(total, parts):
    if parts <= 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_rot_shared_shape(group_size, used_c4, used_d4):
    for _ in range(500):
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        variants = _rotation_variants(base)
        c4 = _canonical_shape_c4(base)
        d4 = _canonical_shape_d4(base)
        if len(variants) < int(group_size):
            continue
        if c4 in used_c4 or d4 in used_d4:
            continue
        std_coords = [tuple(cell) for cell in c4]
        return std_coords, variants, c4, d4
    raise ValueError("rotation-shared base not found")


def _sample_chiral_shape(used_c4, used_d4):
    for _ in range(600):
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        if not _is_chiral(base):
            continue
        d4 = _canonical_shape_d4(base)
        key_a = _canonical_shape_c4(base)
        key_b = _canonical_shape_c4(_reflect_rows(base))
        if key_a in used_c4 or key_b in used_c4 or d4 in used_d4:
            continue
        return base, key_a, key_b, d4
    raise ValueError("chiral base not found")


def _sample_extra_unique_shape(used_c4, used_d4):
    for _ in range(500):
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        c4 = _canonical_shape_c4(base)
        d4 = _canonical_shape_d4(base)
        if c4 in used_c4 or d4 in used_d4:
            continue
        return base, c4, d4
    raise ValueError("extra unique shape not found")


_CONFIGS = [
    (2, 0),  # 4 objects total
    (2, 1),  # 5 objects total
    (2, 2),  # 6 objects total
    (3, 0),  # 5 objects total
    (3, 1),  # 6 objects total
]


def _solve_old_d4(grid: np.ndarray) -> np.ndarray:
    """Original task logic, kept only for anti-degeneration assertions."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, comp["bbox"])
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append({"c0": int(c0), "pat": pat, "shape_key": _canonical_shape_d4(rel)})
    freq = Counter(item["shape_key"] for item in extracted)
    out = np.full((H, W), int(bg), dtype=int)
    for item in extracted:
        pat = item["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[item["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(item["c0"]), transparent=int(bg))
    return out


def _gen_attempt():
    H = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    rot_group_size, n_extra_unique = random.choice(_CONFIGS)
    objs = []
    used_c4 = set()
    used_d4 = set()
    changed_shared = 0

    std_coords, rot_variants, rot_c4, rot_d4 = _sample_rot_shared_shape(
        rot_group_size, used_c4, used_d4
    )
    used_c4.add(rot_c4)
    used_d4.add(rot_d4)
    non_std = [coords for coords in rot_variants if tuple(coords) != tuple(std_coords)]
    random.shuffle(non_std)
    chosen_inputs = [list(std_coords)] + non_std[: int(rot_group_size) - 1]
    if len(chosen_inputs) < int(rot_group_size):
        raise ValueError("not enough non-canonical rotations")
    random.shuffle(chosen_inputs)
    for input_coords in chosen_inputs:
        is_rotated = _normalize_coords(input_coords) != _normalize_coords(std_coords)
        objs.append(
            {
                "shared": True,
                "role": "rot_shared",
                "color": avail[ci],
                "c4": rot_c4,
                "d4": rot_d4,
                "input_coords": [tuple(cell) for cell in input_coords],
                "output_coords": [tuple(cell) for cell in std_coords],
                "rotated": bool(is_rotated),
            }
        )
        changed_shared += int(is_rotated)
        ci += 1

    chiral_base, key_a, key_b, chiral_d4 = _sample_chiral_shape(used_c4, used_d4)
    used_c4.update([key_a, key_b])
    used_d4.add(chiral_d4)
    coords_a = random.choice(_rotation_variants(chiral_base))
    coords_b = random.choice(_rotation_variants(_transform_shape(chiral_base, flip=True)))
    objs.append(
        {
            "shared": False,
            "role": "chiral_a",
            "color": avail[ci],
            "c4": _canonical_shape_c4(coords_a),
            "d4": _canonical_shape_d4(coords_a),
            "input_coords": coords_a,
            "output_coords": coords_a,
            "rotated": False,
        }
    )
    ci += 1
    objs.append(
        {
            "shared": False,
            "role": "chiral_b",
            "color": avail[ci],
            "c4": _canonical_shape_c4(coords_b),
            "d4": _canonical_shape_d4(coords_b),
            "input_coords": coords_b,
            "output_coords": coords_b,
            "rotated": False,
        }
    )
    ci += 1

    for _ in range(n_extra_unique):
        base, c4, d4 = _sample_extra_unique_shape(used_c4, used_d4)
        used_c4.add(c4)
        used_d4.add(d4)
        coords = random.choice(_rotation_variants(base))
        objs.append(
            {
                "shared": False,
                "role": "extra_unique",
                "color": avail[ci],
                "c4": c4,
                "d4": d4,
                "input_coords": coords,
                "output_coords": coords,
                "rotated": False,
            }
        )
        ci += 1

    if changed_shared == 0:
        raise ValueError("degenerate sample without shared rotation alignment")

    for obj in objs:
        obj["input_patch"] = _coords_to_patch(obj["input_coords"], obj["color"], bg)
        obj["output_patch"] = _coords_to_patch(obj["output_coords"], obj["color"], bg)
        obj["reserve_w"] = max(int(obj["input_patch"].shape[1]), int(obj["output_patch"].shape[1]))

    reserve_widths = [int(obj["reserve_w"]) for obj in objs]
    required_w = sum(reserve_widths) + max(0, len(reserve_widths) - 1)
    if required_w > 30:
        raise ValueError("objects too wide to separate cleanly")
    W = random.randint(max(19, required_w), 30)

    order = list(range(len(objs)))
    random.shuffle(order)
    slacks = _random_partition(W - required_w, len(objs) + 1)
    cursor = int(slacks[0])
    for slot, obj_idx in enumerate(order):
        objs[obj_idx]["c0"] = int(cursor)
        cursor += int(objs[obj_idx]["reserve_w"])
        if slot < len(order) - 1:
            cursor += 1 + int(slacks[slot + 1])

    out = np.full((H, W), int(bg), dtype=int)
    for obj in objs:
        patch = obj["output_patch"]
        h, _w = map(int, patch.shape)
        dest_r0 = 0 if obj["shared"] else int(H - h)
        paste(out, patch, dest_r0, int(obj["c0"]), transparent=int(bg))
        obj["dest_r0"] = int(dest_r0)

    inp = np.full((H, W), int(bg), dtype=int)
    for obj in objs:
        patch = obj["input_patch"]
        h, _w = map(int, patch.shape)
        low = 1
        high = H - h - 1
        if low > high:
            raise ValueError("grid too short for interior scatter")
        choices = [r for r in range(low, high + 1) if r != int(obj["dest_r0"])]
        if not choices:
            raise ValueError("no non-target interior row available")
        r0 = random.choice(choices)
        paste(inp, patch, int(r0), int(obj["c0"]), transparent=int(bg))

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    assert not np.array_equal(result, _solve_old_d4(inp)), "sample degenerated to old D4 behavior"

    c4_freq = Counter(obj["c4"] for obj in objs)
    d4_freq = Counter(obj["d4"] for obj in objs)
    assert any(obj["role"] == "rot_shared" and obj["rotated"] for obj in objs)
    assert any(obj["role"] == "rot_shared" and c4_freq[obj["c4"]] >= 2 for obj in objs)
    assert any(
        obj["role"].startswith("chiral")
        and c4_freq[obj["c4"]] == 1
        and d4_freq[obj["d4"]] >= 2
        for obj in objs
    )
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")

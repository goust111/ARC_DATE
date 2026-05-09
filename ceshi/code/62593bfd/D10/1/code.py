# Variant of 62593bfd
# Dimension: D10 — canonical形状的等价变换群 (match_cond)
# Change: D4二面体群——包含0/90/180/270度旋转和水平翻转的8种变换，两个形状若在这8种变换下有相同的归一化坐标则视为... → 仅旋转等价（C4群，不含翻转）

from __future__ import annotations

# Design analysis for the D10 variant
# -----------------------------------
# a) Concrete behavior and trace
# - Visual scene is still 4-6 monochromatic 8-connected objects on a plain background.
# - Output behavior is unchanged except for the matching criterion:
#   objects whose canonical shape appears >= 2 times move to row 0;
#   objects whose canonical shape appears once move to the bottom edge;
#   each object keeps its original bbox left edge c0.
# - The only changed stage is shape grouping. In the original task, an L and its
#   mirror J collapse to the same D4 canonical key, so both count as "shared" and
#   rise to the top. In this variant, only rotations count; the same L/J pair has
#   two different C4 keys, so unless another rotated copy exists, both are unique
#   and move to the bottom instead.
# - Small trace:
#   base = {(0,0),(1,0),(2,0),(2,1)}   # L
#   mirror = {(0,1),(1,1),(2,0),(2,1)} # J
#   original D4: canonical(base) == canonical(mirror) -> freq 2 -> both TOP
#   new C4:      canonical(base) != canonical(mirror) -> freq 1 -> both BOTTOM
#   This is the observable difference the generator must force to appear.
#
# b) Complexity calibration
# - If the generator only switched the solver to C4 but still sampled most shared
#   groups via arbitrary D4 transforms, many instances would accidentally behave
#   like the original task. To prevent that degeneration, every sample contains:
#   1. at least one genuinely rotation-shared group (so TOP still exists), and
#   2. at least one chiral mirror pair with the same D4 key but different C4 keys
#      (so the new exclusion of reflection is visibly exercised).
# - Object count remains within the original 4-6 range. The generator varies the
#   total via one rotational shared group, one mandatory chirality-split pair,
#   and 0-2 extra unique shapes.
#
# c) Implicit-assumption audit of main()
# - get_bg(): still valid; the variant does not change background semantics.
# - find_components(... connectivity=8, monochromatic=False):
#   this assumes different-colored objects never touch, even diagonally, because
#   touching would merge them into one component. That assumption was already
#   present in the original solver; the new generator now enforces 8-neighbor
#   clearance explicitly so the assumption is guaranteed.
# - rel = coords - bbox origin: still valid because canonicalization is based on
#   translation-normalized coordinates only.
# - shape_key = _canonical_shape_rot(rel): this is the core D10 change. The old
#   code relied on reflections being acceptable; that premise is now false, so the
#   canonicalizer must enumerate only 4 rotations.
# - paste(...): still safe only if destination objects do not overlap. The
#   generator guarantees non-overlapping top/bottom placements.
#
# d) Placement legality
# - Output is constructed first, then input is scattered, as in the original task.
# - To make placement robust, the generator assigns globally separated column bands
#   to all objects with at least one empty column between bounding boxes. This
#   keeps c0 preserved while preventing accidental 8-neighbor contact in either
#   input or output.
# - Rows in the input are chosen strictly away from the target edge, so each object
#   visibly moves in the output.
#
# e) Visual observability
# - Humans can see the rule because each sample shows both branches:
#   a rotated family at the top and a mirror-related pair stranded at the bottom.
# - No hidden numeric ordering or color-id semantics are used; colors only
#   separate objects visually.
#
# f) Visual cleanliness
# - Objects are single-color and well separated. There are no overlays, no markers,
#   and no multi-layer encodings, so the grid stays legible.
#
# g) Color inferability
# - Colors do not change between input and output and do not encode logic. They are
#   only identity cues for tracking translated objects.
#
# 9) New assumptions introduced by the C4 solver and how generate_input() enforces them
# - Assumption: at least one canonical C4 key has multiplicity >= 2.
#   Guarantee: one mandatory rotation-shared group of size 2 or 3.
# - Assumption: at least one object is unique under C4.
#   Guarantee: the mandatory chiral pair is unique under C4 and placed at bottom.
# - Assumption: the chiral pair is not accidentally rotationally equivalent.
#   Guarantee: the base shape is required to be chiral before use.
# - Assumption: no other shape collides with the same C4 key and hides the effect.
#   Guarantee: generator tracks used C4 keys and rejects duplicates unless they are
#   the intended rotation-shared group.

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _normalize_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _rotate_cw(coords):
    return [(int(c), -int(r)) for r, c in coords]


def _reflect_rows(coords):
    return [(-int(r), int(c)) for r, c in coords]


def _canonical_shape_rot(coords):
    """Canonical key under C4 rotations only."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        key = _normalize_coords(cur)
        best = key if best is None or key < best else best
        cur = _rotate_cw(cur)
    return best


def _canonical_shape_d4(coords):
    """Canonical key under the original D4 group, kept for generator diagnostics."""
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
    """Distinct normalized shapes reachable by rotation."""
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
    return _canonical_shape_rot(coords) != _canonical_shape_rot(_reflect_rows(coords))


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    H, W = map(int, shape)
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their shape (up to rotation only)
    is shared with another object.

    Rule:
    - Compute the canonical shape of each 8-connected component under C4:
      0°/90°/180°/270° rotations only, with no reflections.
    - Shared shapes  (canonical shape freq >= 2) -> translate to TOP    (row 0).
    - Unique shapes  (freq == 1)               -> translate to BOTTOM (row H-h).
    - Horizontal position (left column c0 of the bounding box) is preserved.
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
                "shape_key": _canonical_shape_rot(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ---------------------- generator ----------------------

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* cells within max_dim x max_dim."""
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
    for _ in range(rot % 4):
        pts = _rotate_cw(pts)
    if flip:
        pts = _reflect_rows(pts)
    return [tuple(cell) for cell in _normalize_coords(pts)]


def _coords_to_patch(coords, color, bg):
    mr = max(int(r) for r, _ in coords)
    mc = max(int(c) for _, c in coords)
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


def _sample_rot_shared_shape(min_orbit, used_c4, used_d4):
    for _ in range(400):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        variants = _rotation_variants(shape)
        c4 = _canonical_shape_rot(shape)
        d4 = _canonical_shape_d4(shape)
        if len(variants) < int(min_orbit):
            continue
        if c4 in used_c4 or d4 in used_d4:
            continue
        return shape, variants, c4, d4
    raise ValueError("rotation-shared base not found")


def _sample_chiral_shape(used_c4, used_d4):
    for _ in range(500):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        if not _is_chiral(shape):
            continue
        d4 = _canonical_shape_d4(shape)
        key_a = _canonical_shape_rot(shape)
        key_b = _canonical_shape_rot(_reflect_rows(shape))
        if key_a in used_c4 or key_b in used_c4 or d4 in used_d4:
            continue
        return shape, key_a, key_b, d4
    raise ValueError("chiral base not found")


def _sample_extra_unique_shape(used_c4, used_d4):
    for _ in range(400):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        c4 = _canonical_shape_rot(shape)
        d4 = _canonical_shape_d4(shape)
        if c4 in used_c4 or d4 in used_d4:
            continue
        return shape, c4, d4
    raise ValueError("extra unique shape not found")


_CONFIGS = [
    (2, 0),  # 4 objects total: 2 rotationally shared + 2 chiral uniques
    (2, 1),  # 5 total
    (2, 2),  # 6 total
    (3, 0),  # 5 total
    (3, 1),  # 6 total
]


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

    # One mandatory group that is shared under rotation-only equivalence.
    rot_base, rot_variants, rot_c4, rot_d4 = _sample_rot_shared_shape(
        rot_group_size, used_c4, used_d4
    )
    used_c4.add(rot_c4)
    used_d4.add(rot_d4)
    random.shuffle(rot_variants)
    for coords in rot_variants[:rot_group_size]:
        objs.append(
            {
                "coords": coords,
                "color": avail[ci],
                "shared": True,
                "role": "rot_shared",
                "c4": rot_c4,
                "d4": rot_d4,
            }
        )
        ci += 1

    # One mandatory mirror pair: same D4 key, different C4 keys, so both are unique now.
    chiral_base, key_a, key_b, chiral_d4 = _sample_chiral_shape(used_c4, used_d4)
    used_c4.update([key_a, key_b])
    used_d4.add(chiral_d4)
    coords_a = random.choice(_rotation_variants(chiral_base))
    coords_b = random.choice(_rotation_variants(_transform_shape(chiral_base, flip=True)))
    objs.append(
        {
            "coords": coords_a,
            "color": avail[ci],
            "shared": False,
            "role": "chiral_a",
            "c4": _canonical_shape_rot(coords_a),
            "d4": _canonical_shape_d4(coords_a),
        }
    )
    ci += 1
    objs.append(
        {
            "coords": coords_b,
            "color": avail[ci],
            "shared": False,
            "role": "chiral_b",
            "c4": _canonical_shape_rot(coords_b),
            "d4": _canonical_shape_d4(coords_b),
        }
    )
    ci += 1

    for _ in range(n_extra_unique):
        shape, c4, d4 = _sample_extra_unique_shape(used_c4, used_d4)
        used_c4.add(c4)
        used_d4.add(d4)
        coords = random.choice(_rotation_variants(shape))
        objs.append(
            {
                "coords": coords,
                "color": avail[ci],
                "shared": False,
                "role": "extra_unique",
                "c4": c4,
                "d4": d4,
            }
        )
        ci += 1

    # Build patches before choosing width so we can reserve enough horizontal space.
    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    widths = [int(obj["patch"].shape[1]) for obj in objs]
    required_w = sum(widths) + max(0, len(widths) - 1)
    if required_w > 30:
        raise ValueError("objects too wide to separate cleanly")
    W = random.randint(max(19, required_w), 30)

    # Global separated columns: this preserves c0 while guaranteeing that objects
    # never touch in 8-neighborhood in either the input or the output.
    order = list(range(len(objs)))
    random.shuffle(order)
    slacks = _random_partition(W - required_w, len(objs) + 1)
    cursor = slacks[0]
    for slot, obj_idx in enumerate(order):
        objs[obj_idx]["c0"] = int(cursor)
        cursor += widths[obj_idx]
        if slot < len(order) - 1:
            cursor += 1 + slacks[slot + 1]

    out = np.full((H, W), int(bg), dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = map(int, patch.shape)
        dest_r0 = 0 if obj["shared"] else int(H - h)
        paste(out, patch, dest_r0, int(obj["c0"]), transparent=int(bg))
        obj["dest_r0"] = dest_r0

    inp = np.full((H, W), int(bg), dtype=int)
    for obj in objs:
        patch = obj["patch"]
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

    # Strong anti-degeneration check: under the old D4 logic, the chiral pair would
    # share a key; under the new C4 logic, they must remain separated.
    c4_freq = Counter(obj["c4"] for obj in objs)
    d4_freq = Counter(obj["d4"] for obj in objs)
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
    raise RuntimeError("failed to generate input for 62593bfd D10 variant")

# Variant of 62593bfd
# Dimension: D1 — 分组判据——形状等价类的分类规则 (match_cond)
# Change: 计算每个对象在D4二面体群（4旋转x2翻转=8变换）下的canonical形状，统计各canonical形状的出现频次；... → 按连通方向特征分组（仅旋转等价，不含翻转）

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D1 variant
# ==================================
# variant_spec:
#   original_rule:
#     - Each 8-connected foreground object is keyed by a canonical shape that is
#       invariant under the full D4 dihedral group: 4 rotations plus reflection.
#     - Shapes with frequency >= 2 move to the top; shapes with frequency == 1
#       move to the bottom; column position is preserved.
#   new_rule:
#     - Keep the same extraction, counting, and movement logic, but replace the
#       shape equivalence class with rotation-only equivalence.
#     - Canonicalization now considers only 0/90/180/270-degree rotations.
#       Mirror images are different unless one is also reachable by rotation.
#   concrete_behavior_trace:
#     - Example base L-shape:
#         A = {(0,0), (1,0), (2,0), (2,1)}
#       Its mirror:
#         B = {(0,1), (1,1), (2,0), (2,1)}
#     - Original solver:
#         canonical_d4(A) == canonical_d4(B), so if A and B both appear once,
#         they are treated as a duplicated shape class and both move to the top.
#     - Variant solver:
#         canonical_rot(A) != canonical_rot(B), so A and B are two unique shape
#         classes and both move to the bottom.
#     - Visible difference:
#         a mirrored pair that would have risen to the top in the original task
#         now remains at the bottom in this variant.
#   complexity_calibration:
#     - If the generator kept producing only rotationally symmetric shapes or only
#       exact rotation duplicates, the new rule would often look identical to the
#       old one.
#     - To prevent degeneration, every generated puzzle includes:
#         1. at least one true rotation-duplicate group (so the "top" branch is
#            still present),
#         2. at least one chiral mirror pair: same under D4, different under
#            rotation-only. This forces a visible disagreement with the old rule.
#     - Extra unique shapes are allowed but kept few (0-2) so the examples remain
#       easy to parse visually.
#   implicit_assumptions_review:
#     - main() calls find_components(..., connectivity=8, monochromatic=False).
#       That means different-colored objects would merge if they touched even
#       diagonally. The original code relied on random placement to avoid this.
#       The new generator explicitly enforces an 8-neighborhood gap so solver
#       components remain one-object-per-component.
#     - The extraction logic builds each object's patch from its bbox and relative
#       coordinates. This remains valid because the D1 change affects only the
#       shape-key function, not how objects are cropped or pasted.
#     - Frequency counting via Counter still works unchanged; only the meaning of
#       the key changes from D4-invariant to rotation-invariant.
#     - Vertical translation with preserved c0 remains valid because the output
#       transformation is still purely positional; no recoloring or reshaping is
#       introduced by this variant.
#   placement_legality:
#     - Output-first generation is preserved.
#     - Each object is placed at its final top/bottom row first, with preserved
#       own bbox width/height and no overlap or 8-neighbor contact.
#     - Input is then produced by scattering the same patches vertically while
#       preserving c0, again forbidding 8-neighbor contact.
#   visual_observability:
#     - Human solvers can infer the new rule because at least one mirrored pair
#       shares the same coarse silhouette family yet is not grouped together.
#       Across examples, top objects match by rotation only; mirrored "handedness"
#       matters.
#   visual_tidiness:
#     - Objects remain monochrome and spatially separated; colors do not encode
#       the rule. Distinct colors only help object individuation.
#   color_reasoning:
#     - No color-ordering or hidden palette semantics are used. Grouping depends
#       only on visible shape orientation structure.
#   post_change_solver_assumptions:
#     - Every foreground object is an isolated 8-connected component.
#     - At least one rotation-duplicate class exists.
#     - At least one mirrored-but-not-rotationally-equivalent pair exists, so the
#       variant is measurably different from the original D4 behavior.


def _normalize_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _canonical_shape_rot(coords):
    """Canonical shape key under rotations only."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        n = _normalize_coords(cur)
        best = n if best is None or n < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _canonical_shape_d4(coords):
    """Legacy canonical shape key under rotations + reflection."""
    cur = [(int(r), int(c)) for r, c in coords]
    best = None
    for _ in range(4):
        n = _normalize_coords(cur)
        best = n if best is None or n < best else best
        n = _normalize_coords([(-r, c) for r, c in cur])
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
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append({"c0": int(c0), "pat": pat, "shape_key": shape_key_fn(rel)})

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)
    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)
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


def _count_mirror_only_uniques(grid: np.ndarray) -> int:
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
    Sort objects to top or bottom by whether their shape matches another object
    under rotation-only equivalence.

    Rule:
    - Compute the canonical shape of each 8-connected component using only
      0°/90°/180°/270° rotations.
    - Shared shapes  (rotation-canonical freq >= 2) -> TOP.
    - Unique shapes  (rotation-canonical freq == 1) -> BOTTOM.
    - Horizontal position (left column c0 of bounding box) is preserved.
    """
    return _solve_with_key(grid, _canonical_shape_rot)


# ---------------------- generator ----------------------

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels."""
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
    return list(_normalize_coords(cells))


def _transform_shape(coords, rot=0, flip=False):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    return list(_normalize_coords(pts))


def _is_chiral(coords):
    """True iff the reflected shape is not rotation-equivalent to the original."""
    return _canonical_shape_rot(coords) != _canonical_shape_rot(
        _transform_shape(coords, flip=True)
    )


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _ in coords)
    mc = max(c for _, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _can_place_separate(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor touching so solver components stay isolated."""
    h, w = patch.shape
    H, W = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = (patch != bg)
    if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg):
        return False

    for pr, pc in np.argwhere(mask):
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        for nr in range(max(0, rr - 1), min(H, rr + 2)):
            for nc in range(max(0, cc - 1), min(W, cc + 2)):
                if grid[nr, nc] != bg:
                    return False
    return True


def _place_output_object(out, obj, bg):
    patch = obj["patch"]
    h, w = map(int, patch.shape)
    dest_r0 = int(obj["dest_r0"])
    cols = list(range(out.shape[1] - w + 1))
    random.shuffle(cols)
    for c0 in cols:
        if _can_place_separate(out, patch, dest_r0, c0, bg):
            paste(out, patch, dest_r0, c0, transparent=bg)
            obj["c0"] = int(c0)
            return
    raise ValueError("output placement failed")


def _place_input_object(inp, obj, bg):
    patch = obj["patch"]
    h, _w = map(int, patch.shape)
    c0 = int(obj["c0"])
    rows = list(range(inp.shape[0] - h + 1))
    random.shuffle(rows)
    preferred = [r0 for r0 in rows if r0 != int(obj["dest_r0"])]
    fallback = [r0 for r0 in rows if r0 == int(obj["dest_r0"])]
    for r0 in preferred + fallback:
        if _can_place_separate(inp, patch, r0, c0, bg):
            paste(inp, patch, r0, c0, transparent=bg)
            return
    raise ValueError("input scatter failed")


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
            for _ in range(group_size)
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
    H = random.randint(19, 30)
    W = random.randint(19, 30)
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
        h, _w = map(int, obj["patch"].shape)
        obj["dest_r0"] = 0 if obj["shared"] else int(H - h)

    out = np.full((H, W), bg, dtype=int)
    placement_order = list(objs)
    random.shuffle(placement_order)
    for obj in placement_order:
        _place_output_object(out, obj, bg)

    inp = np.full((H, W), bg, dtype=int)
    scatter_order = list(objs)
    random.shuffle(scatter_order)
    for obj in scatter_order:
        _place_input_object(inp, obj, bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    legacy = _solve_with_key(inp, _canonical_shape_d4)
    assert not np.array_equal(legacy, out), "generator degenerated to original D4 behavior"
    assert _count_mirror_only_uniques(inp) >= 2, "mirror-pair signal missing"

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D1/3")

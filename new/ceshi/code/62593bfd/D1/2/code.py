# Variant of 62593bfd
# Dimension: D1 — 分组判据——形状等价类的分类规则 (match_cond)
# Change: 计算每个对象在D4二面体群（4旋转x2翻转=8变换）下的canonical形状，统计各canonical形状的出现频次；... → 按外接矩形尺寸分组

# Design analysis for this variant
# =================================
# spec:
#   core_change:
#     original_group_key: D4-canonical exact shape
#     variant_group_key: bounding-box size (height, width)
#   concrete_behavior:
#     - Objects are still monochromatic 8-connected components on a uniform background.
#     - Each object still moves vertically only; its left column c0 stays fixed.
#     - The top group now means "another object has the same bbox size", not
#       "another object has the same exact shape up to rotation/reflection".
#     - Example:
#         * A 2x3 L-shape and a 2x3 zigzag are different canonical shapes.
#         * Original task: both would be unique and go to the bottom.
#         * Variant: both share bbox size (2, 3), so both go to the top.
#   flow_impact:
#     input_construction:
#       - We must deliberately generate several objects with identical bbox sizes
#         but different canonical shapes, otherwise the samples would often
#         collapse back to the original behavior.
#     solver:
#       - Component extraction, crop, and vertical translation stay unchanged.
#       - Only the grouping key changes from _canonical_shape(rel) to patch.shape.
#     output_generation:
#       - Placement logic still uses row 0 for duplicated-group objects and
#         row H-h for unique-group objects.
#   complexity_calibration:
#     - The variant is easier than the original because a solver only needs to
#       measure each object's bbox, not compare exact geometry under D4.
#     - To keep the visual distinction observable, duplicated-size groups are
#       generated from different shapes sharing the same bbox size; this makes
#       "same size, different shape, same destination" visible in few examples.
#   implicit_assumptions_review:
#     - find_components(... monochromatic=False, connectivity=8) treats any
#       touching non-background pixels as one component even across colors.
#       Therefore the generator must enforce at least one-cell 8-neighborhood
#       clearance between distinct objects in both input and output.
#     - crop_bbox(...) returns the tight bbox, so patch.shape is exactly the
#       measured grouping key. This remains valid in the variant.
#     - paste(...) can overwrite if objects collide after movement; the solver
#       assumes the intended output has non-overlapping placements. The
#       generator preserves that by output-first placement checks.
#   placement_legality:
#     - Objects are placed output-first on row 0 or row H-h with preserved c0.
#     - A placement is legal only if the patch is in bounds and no foreground
#       pixel touches existing foreground in its 8-neighborhood.
#     - Input is then built by scattering the same patches to random rows while
#       preserving c0; the same clearance rule prevents merged components.
#   visual_observability:
#     - A human can inspect examples and notice that objects with the same
#       height×width bbox align to the same edge even when their silhouettes
#       differ. This is the intended clue that replaces D4 matching.
#   cleanliness:
#     - Distinct colors per object and 8-neighborhood spacing keep components
#       visually separable; no extra markers or overlays are introduced.
#   color_rule:
#     - Colors are not used for grouping or movement decisions. They only help
#       separate objects visually.
#   new_main_assumptions:
#     - Every foreground component is intended to be an object.
#     - Bounding-box size alone determines the group.
#     - generate_input() guarantees at least one duplicated bbox size and at
#       least one unique bbox size, so both branches of the solver are exercised.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical shape key under the D4 dihedral group.

    Kept only for generator anti-degeneracy checks: duplicated bbox-size groups
    are forced to contain distinct canonical shapes so the variant is visibly
    different from the original task.
    """
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their bounding-box size is
    shared with another object.

    Rule:
    - Compute the tight bbox of each 8-connected component.
    - Group objects by bbox size (height, width).
    - Shared bbox sizes (freq >= 2) -> translate to TOP (row 0).
    - Unique bbox sizes (freq == 1) -> translate to BOTTOM (row H-h).
    - Horizontal position (left column c0 of bounding box) is preserved.
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
        bbox_key = tuple(map(int, pat.shape))

        extracted.append({"c0": int(c0), "pat": pat, "bbox_key": bbox_key})

    freq = Counter(d["bbox_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["bbox_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ---------------------- generator ----------------------

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
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _touches_foreground(grid, r, c, bg):
    H, W = grid.shape
    for rr in range(max(0, r - 1), min(H, r + 2)):
        for cc in range(max(0, c - 1), min(W, c + 2)):
            if int(grid[rr, cc]) != int(bg):
                return True
    return False


def _can_place(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if _touches_foreground(grid, rr, cc, bg):
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
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    coords = sorted((int(r), int(c)) for r, c in cells)
    assert max(r for r, _ in coords) == h - 1
    assert max(c for _, c in coords) == w - 1
    return coords


def _sample_shape_for_dims(h, w, forbidden_canons):
    for _ in range(300):
        coords = _grow_shape_in_bbox(h, w)
        canon = _canonical_shape(coords)
        if canon not in forbidden_canons:
            return coords, canon
    raise ValueError(f"shape not found for dims {(h, w)}")


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    num_dup_groups = len(group_sizes)

    dup_dim_choices = [d for d in _DUP_DIMS if d[1] <= W and d[0] <= H]
    if len(dup_dim_choices) < num_dup_groups:
        raise ValueError("not enough duplicated dims choices")
    dup_dims = random.sample(dup_dim_choices, num_dup_groups)

    used_dims = set(dup_dims)
    unique_dim_choices = [
        d for d in _ALL_DIMS if d not in used_dims and d[1] <= W and d[0] <= H
    ]
    if len(unique_dim_choices) < n_unique:
        raise ValueError("not enough unique dims choices")
    unique_dims = random.sample(unique_dim_choices, n_unique)

    objs = []
    used_canons = set()

    for dims, gsz in zip(dup_dims, group_sizes):
        h, w = map(int, dims)
        for _ in range(gsz):
            coords, canon = _sample_shape_for_dims(h, w, used_canons)
            used_canons.add(canon)
            objs.append(
                {
                    "coords": coords,
                    "color": int(avail[ci]),
                    "shared": True,
                    "bbox_key": (h, w),
                }
            )
            ci += 1

    for dims in unique_dims:
        h, w = map(int, dims)
        coords, canon = _sample_shape_for_dims(h, w, used_canons)
        used_canons.add(canon)
        objs.append(
            {
                "coords": coords,
                "color": int(avail[ci]),
                "shared": False,
                "bbox_key": (h, w),
            }
        )
        ci += 1

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            if _can_place(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(400):
            r0 = random.randint(0, H - h)
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
    for _ in range(600):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D1 bbox-size variant")

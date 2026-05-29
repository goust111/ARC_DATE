# Variant of 62593bfd
# Dimension: D5 — 对象变换——移动过程中的形状/颜色变化 (transform)
# Change: 对象在平移过程中不做任何变换：形状、颜色、朝向完全保持原样（duplicated-size移顶部，singleton-size移底部）→ duplicated对象旋转对齐为同一朝向

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D5 variant
# a) Concrete behavior:
#    - We keep the original task's D4 shape grouping and top/bottom relocation.
#    - The new visible change happens only for duplicated-shape objects:
#      after moving to row 0, each duplicated object is additionally rotated
#      into one shared "standard" orientation.
#    - The standard orientation is the lexicographically smallest pose among
#      the 4 rotations of that chirality. Generator-side, we only use shared
#      shapes whose full D4 canonical representative is reachable by rotation,
#      so "D4 grouping + rotation-only normalization" stays logically feasible.
#    - Example trace:
#      original: two same-color-distinct L-shapes, one upright and one lying
#      sideways, both move to top and keep their own orientations.
#      variant: the same two objects still move to the top, but both are
#      rotated to the same canonical L orientation before being pasted.
# b) Complexity calibration:
#    - Shared groups are built from rotation-asymmetric shapes with 4 distinct
#      rotations. Otherwise the rotation rule could become visually invisible.
#    - Each shared group contains at least one canonical-orientation member and
#      at least one non-canonical member, so examples show both "already
#      aligned" and "needs rotation" cases without introducing extra branches.
# c) Old assumptions audited:
#    - Original `main()` stored a cropped patch and pasted it unchanged. That
#      depended on "movement is pure translation". This is no longer valid for
#      duplicated objects, so we must retain the component's relative coords,
#      decide whether the group is duplicated, compute the canonical rotation,
#      rotate the patch, then paste.
#    - Preserving `c0` is still valid, but now the rotated width can differ
#      from the input width. The generator therefore chooses `c0` that fits
#      both the input orientation and the canonical output orientation.
# d) Placement legality:
#    - Output is still constructed first. Shared objects are placed at the top
#      in canonical rotation; unique objects are placed at the bottom unchanged.
#    - Input then scatters those same objects vertically while preserving `c0`.
#      Candidate rows avoid the final row when possible, to keep movement clear.
# e/f) Visual observability and neatness:
#    - Distinct colors per object keep tracking easy.
#    - Shared shapes use non-symmetric irregular silhouettes, so the top-row
#      normalization is visible from only a few examples.
#    - Top and bottom regions remain uncluttered because placement keeps
#      objects disjoint and the grid is much taller than any object.
# g) No color-remapping branch is introduced; colors stay attached to objects.
# h) Variant-specific anti-degeneration guarantee:
#    - At least one duplicated object per generated task starts in a
#      non-canonical rotation, so the new solver output differs from the
#      original task's output on that sample.
# i) New solver assumptions checked against generator:
#    - For duplicated groups, the canonical rotation must be well-defined and
#      reachable without reflection: generator enforces this.
#    - Rotated output must still fit at the same `c0`: generator enforces this.
#    - Objects must stay disconnected: output/input placement both enforce it.


def _norm_coords(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, c in pts)
    mc = min(c for r, c in pts)
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
    """Canonical D4 key used for grouping, unchanged from the base task."""
    pts = [(int(r), int(c)) for r, c in coords]
    cur = list(pts)
    best = None
    for _ in range(4):
        n = _norm_coords(cur)
        best = n if best is None or n < best else best
        n = _norm_coords([(-r, c) for r, c in cur])
        best = n if n < best else best
        cur = _rotate_coords_cw(cur)
    return best


def _mask_from_coords(coords, shape):
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _rotate_patch_cw(patch, rot):
    return np.rot90(patch, -int(rot) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Group 8-connected components by D4-invariant canonical shape.
    - Shared shapes (freq >= 2) move to row 0 and are rotated into the
      lexicographically smallest rotation of that shape.
    - Unique shapes (freq == 1) move to the bottom unchanged.
    - In both cases, the left bbox column c0 is preserved.
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
        h, w = map(int, pat.shape)
        dest_r0 = 0 if shared else int(H - h)
        dest_c0 = int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


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
    return list(_norm_coords(cells))


def _transform_shape(coords, rot, flip):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(rot) % 4):
        pts = _rotate_coords_cw(pts)
    if flip:
        pts = [(-int(r), int(c)) for r, c in pts]
    return list(_norm_coords(pts))


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _has_four_distinct_rotations(coords):
    return len({pose for pose, _rot in _rotation_poses(coords)}) == 4


def _shared_shape_ok(coords):
    canon_d4 = _canonical_shape(coords)
    canon_rot, _ = _canonical_rotation_pose(coords)
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
    avail = [c for c in range(10) if c != bg]
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
            std_pose, _ = _canonical_rotation_pose(base)
            std_coords = list(std_pose)

            input_rots = [0, random.choice([1, 2, 3])]
            while len(input_rots) < int(gsz):
                input_rots.append(random.randint(0, 3))
            random.shuffle(input_rots)

            for rot_in in input_rots:
                input_coords = _transform_shape(std_coords, rot_in, False)
                objs.append(
                    {
                        "shared": True,
                        "color": avail[ci],
                        "input_coords": input_coords,
                        "output_coords": std_coords,
                        "rotated": int(rot_in) != 0,
                    }
                )
                changed_shared += int(rot_in != 0)
                ci += 1
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
            objs.append(
                {
                    "shared": False,
                    "color": avail[ci],
                    "input_coords": shape,
                    "output_coords": shape,
                    "rotated": False,
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    if changed_shared == 0:
        raise ValueError("degenerate sample without shared rotation")

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
        for _ in range(400):
            c0 = random.randint(0, int(W - max_w))
            if _no_overlap(out, obj["output_patch"], dest_r0, c0, bg):
                paste(out, obj["output_patch"], dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["input_patch"]
        h, w = patch.shape
        c0 = int(obj["c0"])
        forbidden = {0} if obj["shared"] else {int(H - h)}
        placed = False
        for r0 in _choose_rows(H, h, forbidden):
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
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
    raise RuntimeError("failed to generate input for 62593bfd D5 variant")

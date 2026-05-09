# Variant of 62593bfd
# Dimension: D6 — 输出网格尺寸 (grid_shape)
# Change: 输出网格与输入网格尺寸完全相同（H x W），背景色填充空白区域
#   → 输出分为上下两个独立子网格：上半部分放 shared-shape 对象，下半部分放 unique-shape 对象，
#     中间插入一整行分隔色，整体高度压缩为 top_frame_height + 1 + bottom_frame_height。

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import crop_bbox, find_components, get_bg, paste


SEP_COLOR = 4

# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Original task: keep the input width W and height H; move every 8-connected monochrome object
#      either to row 0 (if its D4-canonical shape appears at least twice) or to row H-h
#      (if its shape is unique). Objects stay in the same column c0.
#    - D6 variant: keep the same D4 grouping and the same preserved c0, but do not leave the large empty
#      middle band. Instead build two compact subgrids:
#         top frame height    = max bbox height among shared-shape objects
#         bottom frame height = max bbox height among unique-shape objects
#      Then stack them vertically with one full-width separator row of color 4:
#         [top frame]
#         [444444...]
#         [bottom frame]
#      Shared objects are still flush to the top of their own frame; unique objects are still flush to the
#      bottom of their own frame. Width stays W so the preserved c0 remains visually meaningful.
#    - Small trace:
#      Suppose two shared objects have heights 3 and 2, and one unique object has height 4.
#      Original output on H=24 would place them at rows 0/0/20. Variant output instead has height 3+1+4=8.
#      The shared objects go to rows 0/0, separator is row 3, and the unique object goes to row 4+(4-4)=4.
#      The classification logic is unchanged; only the final canvas construction changes.
#
# b) Complexity calibration
#    - The variant is visually obvious because output height changes drastically: input H is still 19-30,
#      but output height is only 3-11. This prevents accidental degeneration into the original behavior.
#    - A constant separator color is used for all examples so the "two stacked frames" structure is easy
#      to infer from 2-4 examples. The separator color is reserved and never used as bg or object color.
#
# c) Hidden assumptions in the original main()
#    - `H, W = g.shape`: original code assumed output height must equal input height. This is no longer true;
#      only W is reused, while output height is recomputed from object heights.
#    - `out = np.full((H, W), bg, ...)`: invalid under D6 because the compact stacked-frame output must have
#      a new height. Replace with a computed compact height.
#    - `dest_r0 = 0 if shared else H-h`: the "bottom" destination depended on the old full-grid H.
#      Under D6, the correct destination is relative to the bottom frame: `top_h + 1 + (bottom_h - h)`.
#    - `paste(..., dest_c0=c0)`: this still works because width W is preserved. No change needed.
#
# d) Placement legality
#    - Within the top frame, shared objects must not overlap. Within the bottom frame, unique objects must
#      not overlap. Inter-category horizontal overlap is allowed in the output because the frames are separate.
#    - Input generation still scatters objects into one large HxW grid while preserving c0, and global
#      no-overlap is enforced there as well.
#
# e) Visual observability
#    - The separator is a full-width solid row, so the stacked-frame structure is immediately visible.
#    - The rule does not depend on hidden numeric properties such as color ordering; the separator color is
#      just a constant visible glyph, while the meaningful grouping still comes from visible shape repetition.
#
# f) Visual cleanliness
#    - Objects remain isolated components. The separator row never touches them because it occupies its own
#      dedicated row between two frames.
#    - The compact frames remove the large empty middle area and make the shared-vs-unique split easier to see.
#
# g) Color inference
#    - No color remapping occurs. Object colors are preserved exactly.
#    - The only new color role is the constant separator color 4, reserved exclusively for the divider row.
#
# Reference-task note
# -------------------
# The prompt cites 5545f144 as the frame-separation inspiration. In the provided workspace, a repo-wide
# search finds the official JSON for 5545f144 but no corresponding solver `.py`, so the implementation can
# only reuse the visible mechanism: separate regions/frames with an explicit divider. The actual logic of
# 5545f144 is not imported here.


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


def _extract_objects(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8,
                            monochromatic=False, return_format="dict")
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append({
            "c0": int(c0),
            "pat": pat,
            "shape_key": _canonical_shape(rel),
        })
    return g, bg, extracted


def _render_stacked_frames(extracted, bg: int, width: int) -> np.ndarray:
    freq = Counter(d["shape_key"] for d in extracted)
    shared = [d for d in extracted if int(freq[d["shape_key"]]) >= 2]
    unique = [d for d in extracted if int(freq[d["shape_key"]]) == 1]
    assert shared and unique

    top_h = max(int(d["pat"].shape[0]) for d in shared)
    bottom_h = max(int(d["pat"].shape[0]) for d in unique)
    out = np.full((top_h + 1 + bottom_h, int(width)), int(bg), dtype=int)
    out[top_h, :] = int(SEP_COLOR)

    for d in shared:
        paste(out, d["pat"], 0, int(d["c0"]), transparent=int(bg))
    for d in unique:
        h = int(d["pat"].shape[0])
        dest_r0 = top_h + 1 + (bottom_h - h)
        paste(out, d["pat"], dest_r0, int(d["c0"]), transparent=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by D4-canonical shape. Shared-shape objects go into a compact top
    frame, unique-shape objects go into a compact bottom frame, and a full-width
    separator row of color 4 is inserted between the two frames.
    """
    g, bg, extracted = _extract_objects(grid)
    _H, W = g.shape
    return _render_stacked_frames(extracted, bg, W)


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels within a small bbox."""
    deltas = ([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
              if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)])
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


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = (patch != bg)
    return not np.any(grid[r0:r0 + h, c0:c0 + w][mask] != bg)


_CONFIGS = [
    ([2], 2), ([2], 3), ([3], 1), ([3], 2), ([3], 3),
    ([2, 2], 1), ([2, 2], 2),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c not in {bg, SEP_COLOR}]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3),
                                           random.choice([False, True]))
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append({"coords": shape, "color": avail[ci], "shared": False})
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
        obj["shape_key"] = _canonical_shape(obj["coords"])

    shared = [obj for obj in objs if obj["shared"]]
    unique = [obj for obj in objs if not obj["shared"]]
    top_h = max(int(obj["patch"].shape[0]) for obj in shared)
    bottom_h = max(int(obj["patch"].shape[0]) for obj in unique)

    top_frame = np.full((top_h, W), bg, dtype=int)
    for obj in sorted(shared, key=lambda x: int(x["patch"].shape[1]), reverse=True):
        patch = obj["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(top_frame, patch, 0, c0, bg):
                paste(top_frame, patch, 0, c0, transparent=bg)
                obj["c0"] = c0
                placed = True
                break
        if not placed:
            raise ValueError("top-frame placement failed")

    bottom_frame = np.full((bottom_h, W), bg, dtype=int)
    for obj in sorted(unique, key=lambda x: int(x["patch"].shape[1]), reverse=True):
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = bottom_h - h
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(bottom_frame, patch, dest_r0, c0, bg):
                paste(bottom_frame, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                placed = True
                break
        if not placed:
            raise ValueError("bottom-frame placement failed")

    out = np.full((top_h + 1 + bottom_h, W), bg, dtype=int)
    paste(out, top_frame, 0, 0)
    out[top_h, :] = int(SEP_COLOR)
    paste(out, bottom_frame, top_h + 1, 0)

    inp = np.full((H, W), bg, dtype=int)
    for obj in sorted(objs, key=lambda x: int(x["patch"].shape[0]) * int(x["patch"].shape[1]), reverse=True):
        patch = obj["patch"]
        h, w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(400):
            r0 = random.randint(0, H - h)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["input_r0"] = r0
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
    raise RuntimeError("failed to generate input for 62593bfd D6 variant")

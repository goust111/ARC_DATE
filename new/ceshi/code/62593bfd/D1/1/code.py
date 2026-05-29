# Variant of 62593bfd
# Dimension: D1 — 分组判据——形状等价类的分类规则 (match_cond)
# Change: 计算每个对象在D4二面体群（4旋转x2翻转=8变换）下的canonical形状，统计各canonical形状的出现频次；... → 按颜色分组
#
# Design analysis
# ===============
# Reference-task note
# - The prompt asks to read the solver of source task b5ca7ac4 first.
# - In this workspace, b5ca7ac4 is present only as ARC data
#   (`ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json`); no solver file exists,
#   so that step cannot be executed and is skipped by necessity.
#
# a) Concrete behavior and trace
# - Original task:
#   each 8-connected monochrome object is classified by D4-invariant shape.
#   Shared shape => move to row 0; unique shape => move to row H-h.
# - Variant:
#   each monochrome object is classified only by its color.
#   If a color appears on >=2 disconnected objects, all of them move to row 0.
#   If a color appears on exactly 1 object, that object moves to row H-h.
# - Small trace:
#   suppose input has red L-shape at c=2, red zigzag at c=10, blue T-shape at c=16.
#   Original logic would inspect canonical shapes:
#   if all three shapes are unique, all three go to the bottom.
#   Variant logic counts colors instead:
#   red appears twice, so both red objects go to the top; blue appears once, so blue
#   goes to the bottom. Horizontal anchors stay at c=2,10,16.
# - Therefore the changed logic affects:
#   component extraction assumptions, grouping key computation, generator color
#   assignment, and placement legality for same-color separated components.
#
# b) Complexity calibration
# - If repeated colors were rare or if repeated-color objects also shared shapes, the
#   task could visually collapse back toward the original shape-grouping behavior.
# - To keep the D1 change visible, the generator makes every object's canonical shape
#   globally unique, while at least one color is reused across multiple objects.
#   This forces the top/bottom split to depend on color rather than shape.
# - The number of repeated-color groups is kept small (1-2 groups, sizes 2-3) so
#   examples remain readable and grouping evidence is repeated within each sample.
#
# c) Old main() assumptions that break under the variant
# - `find_components(... monochromatic=False)` was harmless before because objects
#   used distinct colors and never touched; with repeated colors this mode is wrong
#   in principle because grouping must be by disconnected same-color objects.
# - `shape_key = _canonical_shape(rel)` was the old classifier. It must be replaced
#   by `color_key = comp["color"]`.
# - `freq = Counter(d["shape_key"] ...)` must become per-color frequency.
# - Pasting logic can stay unchanged because the spatial transform is still pure
#   vertical translation with preserved `c0`.
#
# d) Placement legality
# - Output-first placement is retained.
# - Because objects should remain visually separate components, no two objects are
#   allowed to touch even diagonally in either input or output.
# - This is stricter than merely separating same-color objects, but it preserves the
#   original task's clean "disconnected objects on background" presentation while
#   keeping D1 as the only semantic change.
# - The generator therefore checks both:
#   1. no overlap with any existing object;
#   2. no Chebyshev-distance-1 contact with any already-placed foreground pixel.
# - Input scattering keeps each object's `c0` fixed and moves it to an interior row,
#   so the solver can reliably reconstruct the intended top/bottom result.
#
# e) Visual observability
# - Human solvers can directly see repeated colors. By making repeated-color objects
#   have different shapes, the rule "same colors travel together" becomes the only
#   stable explanation across examples.
# - No color-id ordering or hidden numeric property is used; only visible equality
#   of color matters.
#
# f) Visual neatness
# - Object count stays at 4-6, bbox stays within 5x5, and top/bottom layouts are
#   non-overlapping. Same-color objects are separated by at least one background
#   cell in every direction, so components remain visually distinct.
#
# g) Color-change inferability
# - There is no recoloring. Colors are preserved exactly from input to output.
# - The only color-based inference is repeated-color membership, and every sample
#   contains both a duplicated color and at least one unique color.
#
# 9) New main() assumptions and how the generator guarantees them
# - Assumption: objects are monochrome connected components. Generator builds every
#   object as a single-color 8-connected patch.
# - Assumption: objects remain pairwise disconnected under 8-connectivity.
#   Generator enforces all-object non-contact in both input and output.
# - Assumption: at least one duplicated color and one unique color exist. Generator
#   samples only from configs that satisfy both.
# - Assumption: top and bottom placements do not overlap. Output-first placement
#   validates this before scattering.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to the top or bottom edge by whether their color is shared.

    Rule:
    - Find monochrome 8-connected components.
    - Count how many components use each color.
    - Colors appearing on >=2 components -> move those components to TOP.
    - Colors appearing on exactly 1 component -> move that component to BOTTOM.
    - Horizontal position (left column c0 of bounding box) is preserved.
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
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append({
            "c0": int(c0),
            "pat": pat,
            "color": int(comp["color"]),
        })

    freq = Counter(d["color"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["color"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` pixels."""
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
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
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


def _component_separated(grid, patch, r0, c0, bg):
    """True iff the placed patch stays 8-disconnected from every existing object."""
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    H, W = grid.shape
    nz = np.argwhere(patch != bg)
    for pr, pc in nz:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != bg:
                    return False
    return True


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([2, 2], 1),
]


def _gen_unique_shape(canon_used):
    for _ in range(400):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        canon = _canonical_shape(shape)
        if canon not in canon_used:
            canon_used.add(canon)
            return shape
    raise ValueError("unique shape not found")


def _place_objects_on_rows(grid, objs, bg, row_getter):
    for obj in objs:
        patch = obj["patch"]
        _h, w = patch.shape
        dest_r0 = int(row_getter(obj))
        placed = False
        for _ in range(500):
            c0 = random.randint(0, grid.shape[1] - w)
            if _component_separated(grid, patch, dest_r0, c0, bg):
                paste(grid, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                placed = True
                break
        if not placed:
            raise ValueError("row placement failed")


def _scatter_input(inp, objs, bg):
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(500):
            r0 = random.randint(1, inp.shape[0] - h - 1)
            if _component_separated(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used = set()

    for gsz in group_sizes:
        color = avail[ci]
        ci += 1
        for _ in range(gsz):
            coords = _gen_unique_shape(canon_used)
            objs.append({"coords": coords, "color": color, "shared": True})

    for _ in range(n_unique):
        color = avail[ci]
        ci += 1
        coords = _gen_unique_shape(canon_used)
        objs.append({"coords": coords, "color": color, "shared": False})

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)
    _place_objects_on_rows(
        out,
        objs,
        bg,
        row_getter=lambda obj: 0 if obj["shared"] else H - obj["patch"].shape[0],
    )

    inp = np.full((H, W), bg, dtype=int)
    _scatter_input(inp, objs, bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd color-group variant")

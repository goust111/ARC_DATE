# Variant of 62593bfd
# Dimension: D12 — 颜色分配策略 (meta)
# Change: 每个对象使用不同的非背景色；颜色通常各不相同但允许不同对象使用相同颜色... → 同组对象必须同色
#
# Design analysis for the D12 variant
# - source_task: None.
# - Concrete behavior:
#   - The original task groups 8-connected monochrome objects by canonical shape
#     under D4 symmetry, then moves repeated-shape objects to the top edge and
#     unique-shape objects to the bottom edge.
#   - In this variant, color becomes an explicit visual cue for that grouping:
#     every object in the same canonical-shape group must use the same color, and
#     different groups must use different colors.
#   - Therefore the input now visibly contains repeated colors exactly on the
#     duplicated-shape groups. Unique-shape groups still have their own private
#     colors, so a human can infer "same color means same shape family".
# - Small trace:
#   - Original flow: two L-shapes in red and blue share a canonical shape, so
#     solver must compute D4-invariant keys to see they belong together.
#   - Variant flow: those same two L-shapes are both orange. The solver can now
#     group by color frequency instead of re-deriving shape equivalence from
#     geometry. If orange appears on 2 components, both orange components go to
#     the top; if teal appears on 1 component, that teal component goes to the
#     bottom.
#   - Difference: the vertical movement rule is unchanged, but the evidence for
#     grouping is now color repetition rather than shape matching difficulty.
# - Complexity calibration:
#   - The task still contains 4-6 objects with 1-2 duplicated groups and 1-3
#     unique groups, so repeated and singleton cases both remain present.
#   - Colors are not arbitrary per object anymore; instead there are only as many
#     non-background colors as shape groups. This keeps the cue strong and avoids
#     visually noisy many-color grids.
# - Implicit-assumption review for main():
#   - The original `find_components(... monochromatic=False)` depended on objects
#     being spatially separated regardless of color. In the variant, same-color
#     components must remain distinct, so `main()` switches to
#     `monochromatic=True`, making color-separated connected components the
#     natural unit of extraction.
#   - The original `shape_key` counting assumed color carries no grouping
#     information. That assumption is no longer central; the new assumption is
#     that each color corresponds to exactly one canonical-shape group. The
#     generator enforces this bijection.
#   - `paste(... transparent=bg)` remains safe because every extracted patch is
#     still monochrome on background, and objects are only translated vertically.
# - Placement legality:
#   - Output-first construction is kept.
#   - Same-color objects are never allowed to touch orthogonally or diagonally;
#     otherwise `monochromatic=True` would merge them into one component.
#   - Different-color objects may touch without breaking the solver, but normal
#     non-overlap is still enforced everywhere.
# - Visual observability:
#   - Every generated sample contains at least one repeated non-background color
#     (a duplicated group) and at least one singleton color (a unique group).
#   - Because repeated colors correspond exactly to shared-shape groups, the D12
#     change is visible in 2-3 examples without relying on hidden numeric color
#     identities.
# - Visual cleanliness:
#   - Reusing one color per group reduces color clutter.
#   - Preventing same-color contact keeps repeated-color groups readable as
#     multiple separate objects rather than accidental merged blobs.
# - New solver assumptions and generator support:
#   - Assumption: each monochromatic component is a single object.
#     Guarantee: generator only places monochrome 8-connected patches.
#   - Assumption: component count per color equals object count per group.
#     Guarantee: each group has a unique color and same-color objects do not
#     touch, so color frequency is well-defined.
#   - Assumption: there is at least one repeated-color group and one singleton
#     color group.
#     Guarantee: `_CONFIGS` always includes both duplicated and unique groups.

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
        p[int(r), int(c)] = int(color)
    return p


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` pixels."""
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


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = (patch != bg)
    return not np.any(grid[r0:r0 + h, c0:c0 + w][mask] != bg)


def _same_color_clear(grid, patch, r0, c0, bg):
    """Reject placements where this monochrome object would touch same-color pixels."""
    color_cells = np.argwhere(patch != bg)
    if color_cells.size == 0:
        return False
    color = int(patch[tuple(color_cells[0])])
    H, W = grid.shape
    for dr, dc in color_cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(grid[r1:r2, c1:c2] == color):
            return False
    return True


def _can_place(grid, patch, r0, c0, bg):
    return _no_overlap(grid, patch, r0, c0, bg) and _same_color_clear(grid, patch, r0, c0, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    D12 variant: colors become a visual cue, but shape still decides routing.

    Rule:
    - Extract 8-connected components the same way as the original task.
    - Compute each component's canonical D4-invariant shape key.
    - Shared shapes  (canonical-shape freq >= 2) -> translate to TOP.
    - Unique shapes  (canonical-shape freq == 1) -> translate to BOTTOM.
    - Horizontal position (`c0`) is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8,
                            monochromatic=False, return_format="dict")
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        extracted.append({
            "c0": int(c0),
            "pat": patch,
            "shape_key": _canonical_shape(rel),
        })

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


_CONFIGS = [
    ([2], 2), ([2], 3), ([3], 1), ([3], 2), ([3], 3),
    ([2, 2], 1), ([2, 2], 2),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
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
        group_color = avail[ci]
        ci += 1
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3),
                                           random.choice([False, True]))
            objs.append({
                "coords": transformed,
                "color": group_color,
                "shared": True,
                "shape_key": canon,
            })

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                group_color = avail[ci]
                ci += 1
                objs.append({
                    "coords": shape,
                    "color": group_color,
                    "shared": False,
                    "shape_key": canon,
                })
                break
        else:
            raise ValueError("unique shape not found")

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
                obj["c0"] = c0
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
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
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D12")

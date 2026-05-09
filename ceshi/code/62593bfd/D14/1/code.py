# Variant of 62593bfd
# Dimension: D14 — 干扰项与额外结构 (src)
# Change: 网格中仅包含背景色和待处理的连通分量对象，无干扰项或额外标记... → 增加噪声像素（孤立单像素）
#
# Design analysis
# ===============
# a) Concrete behavior / trace
#    - Original task: every non-background 8-connected component is treated as a real object.
#      Each object's D4-canonical shape is counted; repeated shapes go to the top edge, unique
#      shapes go to the bottom edge, while each object's left column stays fixed.
#    - D14 variant: the input additionally contains a few isolated single-pixel colored dots.
#      These dots are pure distractors. They never appear in the output and must not affect the
#      shared-vs-unique shape grouping. Real objects still satisfy the old size regime: area 3-12.
#    - Small trace:
#         old input comps = {shape A area 5, shape A area 5, shape B area 4, noise area 1, noise area 1}
#         original logic counts all 5 components, so the two 1-pixel noises share the same
#         canonical shape ((0,0)) and would be moved to row 0.
#         variant logic first filters to area >= 3, so only {A, A, B} are counted and moved;
#         the two single pixels are ignored completely.
#    - Flow impact: only component extraction changes in solver. Output construction for true
#      objects is unchanged. Generator changes by injecting interior isolated noise after the
#      normal object scatter step.
#
# b) Complexity calibration
#    - Noise count is kept small (2-5) so the grid is visibly different from the original task
#      but not cluttered.
#    - Using at least two noise pixels matters: it creates a clear contrast against the original
#      solver, which would treat those singleton shapes as a repeated group and move them.
#    - Noise colors are arbitrary non-background colors and may repeat. The rule depends on
#      visible size/connectivity, not on hidden color ids.
#
# c) Implicit-assumption review for original main()
#    - `find_components(... monochromatic=False, connectivity=8)` assumes every foreground
#      component is a genuine object. This fails under D14 because area-1 distractors become fake
#      objects. Fix: require `min_area=3`.
#    - `_canonical_shape(rel)` assumes the component should participate in grouping. Still valid
#      after filtering.
#    - `paste(out, pat, ...)` assumes extracted components are exactly the objects intended for
#      relocation. Still valid once the filter removes noise.
#    - `assert comps` adds a new assumption: at least one real object of area >= 3 exists.
#      Generator keeps the original object construction, so this holds.
#
# d) Placement legality
#    - Real objects are still placed output-first exactly as in the seed: top for shared, bottom
#      for unique, no overlap, fixed c0.
#    - Input scatter preserves c0 and avoids overlaps among real objects.
#    - Noise pixels are inserted only into background cells that are 8-isolated from every
#      existing foreground pixel and from previously placed noise, so each distractor remains an
#      area-1 component and never merges with a true object.
#    - Noise is restricted to interior rows 1..H-2 so the legacy solver would move it, making the
#      D14 effect observable.
#
# e) Visual observability
#    - Across examples, the user will see a few tiny colored dots vanish while the larger shapes
#      are still sorted top/bottom by canonical shape multiplicity.
#    - The rule is inferable from 2-4 examples because all surviving moved objects have area >= 3,
#      while all disappearing dots have area 1 and are spatially isolated.
#
# f) Visual cleanliness
#    - Only isolated dots are added; no extra frames, markers, or overlaps are introduced.
#    - Low noise density and 8-isolation keep the scene readable.
#
# g) Color inference
#    - No color mapping is introduced. Colors are incidental; object fate depends only on component
#      size plus the original shape-grouping rule.
#
# Source-task note
# ----------------
# The prompt asked to read the solver of source task 71e489b6 first. A repository-wide search in
# the provided workspace did not find a solver file for 71e489b6, so this variant is calibrated
# from the explicit D14 requirement itself: inject isolated singleton distractors that must be
# ignored by the solver.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


MIN_OBJECT_AREA = 3
NOISE_COUNT_RANGE = (2, 5)


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


def _solve_core(grid: np.ndarray, *, min_area: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        min_area=int(min_area),
        return_format="dict",
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
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    D14 variant of 62593bfd.

    Same as the seed task, except area-1 singleton noise pixels are ignored. Real objects remain
    the 8-connected components of area >= 3, grouped by D4-canonical shape and moved top/bottom.
    """
    return _solve_core(grid, min_area=MIN_OBJECT_AREA)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Seed behavior, kept only to reject degenerate generator samples."""
    return _solve_core(grid, min_area=1)


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
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


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
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _is_isolated_noise_slot(grid, r, c, bg):
    H, W = grid.shape
    if not (1 <= int(r) < H - 1):
        return False
    if grid[int(r), int(c)] != bg:
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] != bg:
                return False
    return True


def _add_noise_pixels(grid, bg, palette):
    H, W = grid.shape
    count = random.randint(*NOISE_COUNT_RANGE)
    noise = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(W)
            if _is_isolated_noise_slot(grid, r, c, bg)
        ]
        if not candidates:
            if len(noise) >= NOISE_COUNT_RANGE[0]:
                break
            raise ValueError("not enough isolated slots for noise")
        r, c = random.choice(candidates)
        color = random.choice(palette)
        grid[r, c] = color
        noise.append((r, c, color))
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("too little noise")
    return noise


def _count_singleton_noise(grid: np.ndarray) -> int:
    bg = int(get_bg(grid))
    comps = find_components(
        np.array(grid, dtype=int),
        bg=bg,
        connectivity=8,
        monochromatic=False,
        min_area=1,
        return_format="dict",
    )
    return sum(1 for comp in comps if int(comp["area"]) == 1)


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
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
        area = random.randint(MIN_OBJECT_AREA, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(MIN_OBJECT_AREA, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append({"coords": shape, "color": avail[ci], "shared": False})
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
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
        for _ in range(300):
            r0 = random.randint(0, H - h)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    noise = _add_noise_pixels(inp, bg, avail)
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise missing")
    if _count_singleton_noise(inp) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise merged unexpectedly")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    assert not np.array_equal(_legacy_main(inp), out), "variant degenerated to seed behavior"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D14")

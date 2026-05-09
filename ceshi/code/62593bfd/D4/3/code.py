# Variant of 62593bfd
# Dimension: D4 — 平移规则——水平位置保持 (dst)
# Change: 每个对象仅做垂直平移，水平位置（外接矩形左上角列c0）保持不变；顶部对象dest_r0=0，底部对象dest_r0=H-... → 保持行位置不变，仅做水平平移到左/右边缘

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the D4/dst variant
# --------------------------------------
# variant_spec:
#   unchanged_core:
#     - Objects are still monochromatic 8-connected components on a bg grid.
#     - Grouping is still by canonical shape under the full D4 dihedral group.
#     - There must still be at least one duplicated canonical shape and at least
#       one unique canonical shape.
#   changed_core:
#     - Original task: duplicated -> row 0, unique -> row H-h, while preserving c0.
#     - New variant: duplicated -> col 0, unique -> col W-w, while preserving r0.
#   concrete_trace:
#     - Suppose a shared 3x4 object has bbox top-left (r0=7, c0=11) in input.
#     - Original solver would output it at (0, 11).
#     - Variant solver must output it at (7, 0).
#     - Likewise a unique 2x5 object at (12, 3) would move from original
#       output (H-2, 3) to variant output (12, W-5).
#   complexity_calibration:
#     - The visual change must be obvious in 2-4 examples, so the generator
#       should avoid degenerate cases where objects already sit on their target
#       side. Input objects are therefore scattered to interior columns whenever
#       possible, forcing visible horizontal motion.
#     - To make the new rule distinguishable from the original vertical rule,
#       output rows are chosen as interior rows; the solver preserves those rows.
#   main_assumption_audit:
#     - D4 canonicalization code is still valid; the grouping property did not change.
#     - Component extraction is still valid, but we must preserve each object's r0
#       instead of c0. Reusing old c0-preservation would silently implement the
#       old task and destroy the variant.
#     - Paste destinations must now use dest_c0 chosen by duplication status and
#       dest_r0 copied from the input bbox. Any unconditional top/bottom write is wrong.
#   placement_legality:
#     - Solver assumes input components are separated under 8-connectivity because
#       find_components(..., monochromatic=False) merges touching colors. The
#       generator therefore enforces a one-cell moat in the 8-neighborhood.
#     - Output-first generation is kept, but outputs are packed onto left/right
#       edges with interior row margins and 1-row gaps on each side.
#     - Inputs preserve those rows and search for interior columns distinct from
#       the target edge, again enforcing 8-neighborhood separation.
#   observability:
#     - Human-visible invariant across examples: row bands stay fixed from input
#       to output, while duplicated shapes slide left and unique shapes slide right.
#     - No color-based ordering is used; color only labels objects.


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
    Sort objects to left or right edge by whether their shape (up to
    rotation / reflection) is shared with another object.

    Rule:
    - Compute the canonical shape of each 8-connected component
      (invariant under D4: 0°/90°/180°/270° rotations × optional reflection).
    - Shared shapes  (canonical shape freq >= 2) -> translate to LEFT  (col 0).
    - Unique shapes  (freq == 1)                 -> translate to RIGHT (col W-w).
    - Vertical position (top row r0 of bounding box) is preserved.
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
                "r0": int(r0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = int(d["r0"])
        dest_c0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(W - w)
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out


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
        p[int(r), int(c)] = int(color)
    return p


def _fits_without_touch(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    occ = np.argwhere(patch != bg)
    for pr, pc in occ:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if np.any(grid[max(0, rr - 1) : min(H, rr + 2), max(0, cc - 1) : min(W, cc + 2)] != bg):
            return False
    return True


def _pack_rows(side_objs, H):
    if not side_objs:
        return
    random.shuffle(side_objs)
    used = sum(int(obj["h"]) for obj in side_objs) + (len(side_objs) + 1)
    slack = H - used
    if slack < 0:
        raise ValueError("side stack too tall")
    gaps = [1] * (len(side_objs) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1
    r = gaps[0]
    for idx, obj in enumerate(side_objs):
        obj["r0"] = int(r)
        r += int(obj["h"]) + gaps[idx + 1]


def _candidate_input_cols(W, w, dest_c0):
    cols = [c for c in range(1, int(W - w)) if c != int(dest_c0)]
    if not cols:
        cols = [c for c in range(0, int(W - w) + 1) if c != int(dest_c0)]
    random.shuffle(cols)
    return cols


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
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    objs = []
    canon_used = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            color = avail[ci]
            ci += 1
            patch = _coords_to_patch(transformed, color, bg)
            h, w = map(int, patch.shape)
            objs.append(
                {
                    "coords": transformed,
                    "color": color,
                    "shared": True,
                    "patch": patch,
                    "h": h,
                    "w": w,
                }
            )

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            color = avail[ci]
            ci += 1
            patch = _coords_to_patch(shape, color, bg)
            h, w = map(int, patch.shape)
            objs.append(
                {
                    "coords": shape,
                    "color": color,
                    "shared": False,
                    "patch": patch,
                    "h": h,
                    "w": w,
                }
            )
            break
        else:
            raise ValueError("unique shape not found")

    left_objs = [obj for obj in objs if obj["shared"]]
    right_objs = [obj for obj in objs if not obj["shared"]]
    req_h = max(
        19,
        sum(obj["h"] for obj in left_objs) + len(left_objs) + 1,
        sum(obj["h"] for obj in right_objs) + len(right_objs) + 1,
    )
    if req_h > 30:
        raise ValueError("height budget exceeded")

    min_w = max(19, 2 * max(obj["w"] for obj in objs) + 8, 22)
    if min_w > 30:
        raise ValueError("width budget exceeded")

    H = random.randint(req_h, 30)
    W = random.randint(min_w, 30)

    _pack_rows(left_objs, H)
    _pack_rows(right_objs, H)

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        dest_c0 = 0 if obj["shared"] else int(W - obj["w"])
        if not _fits_without_touch(out, patch, obj["r0"], dest_c0, bg):
            raise ValueError("output placement failed")
        paste(out, patch, obj["r0"], dest_c0, transparent=bg)
        obj["dest_c0"] = int(dest_c0)

    inp = np.full((H, W), bg, dtype=int)
    for obj in sorted(objs, key=lambda d: (-d["h"], -d["w"])):
        patch = obj["patch"]
        placed = False
        for c0 in _candidate_input_cols(W, obj["w"], obj["dest_c0"]):
            if _fits_without_touch(inp, patch, obj["r0"], c0, bg):
                paste(inp, patch, obj["r0"], c0, transparent=bg)
                obj["input_c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    if not any(obj["input_c0"] != obj["dest_c0"] for obj in objs):
        raise ValueError("degenerate no-motion sample")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D4/3 variant")

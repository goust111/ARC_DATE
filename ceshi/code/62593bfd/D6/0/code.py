# Variant of 62593bfd
# Dimension: D6 — 输出网格尺寸 (grid_shape)
# Change: 输出网格与输入网格尺寸完全相同（H x W），背景色填充空白区域
#   → 紧密裁剪——输出仅包含有对象的区域

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste

# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   core_rule:
#     - Keep the original D4-canonical grouping logic unchanged:
#       shared shapes go to the top band, unique shapes go to the bottom band,
#       and each object keeps its original left bbox column c0 on the full canvas.
#     - New behavior is applied after that placement step:
#       return the tight crop of the placed objects instead of the whole HxW canvas.
#   concrete_trace:
#     - Example on a 24x20 canvas:
#       shared objects land at row 0 with left columns 4 and 12;
#       unique objects land at rows 20..23 with left columns 6 and 14.
#       Original task returns the full 24x20 canvas.
#       This variant first builds that same 24x20 canvas, then crops columns
#       min_c..max_c covering all non-bg pixels, producing a narrower output.
#   complexity_calibration:
#     - Because the original rule requires at least one shared-shape object group
#       at the top and at least one unique-shape object at the bottom, the union of
#       placed objects spans the full height. So D6 is visually observable mainly
#       through horizontal crop, not vertical crop.
#     - Generator therefore forces all output objects to stay strictly away from
#       the left and right borders; otherwise tight-crop could collapse back to the
#       original width and the variant would be visually weak.
#   implicit_assumptions_review:
#     - Component extraction, D4 canonicalization, and dest_r0 = 0 / H-h all remain valid.
#     - The old `out = full canvas` return relied on the old dimension invariant.
#       In this variant, that assumption is invalid; the solver must crop by the
#       bbox of all non-background pixels before returning.
#     - Cropping changes horizontal coordinates in the final answer by subtracting
#       the global minimum occupied column. This is safe because the rule only says
#       c0 is preserved on the intermediate placement canvas, not in the cropped frame.
#   placement_legality:
#     - Generator still uses output-first construction on an HxW canvas so the solver
#       can be oracle-checked with `main(inp)`.
#     - Objects are placed with 8-neighbor separation and with interior left/right
#       margins, so components stay distinct and the final crop is strictly narrower.
#   observability:
#     - Across examples, humans see the same top-vs-bottom sorting pattern as the
#       original task, but the answer frame shrinks to the occupied columns only.
#       The crop is structural and visible from the pixel geometry; it does not use
#       hidden properties such as absolute color ids.


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


def _tight_crop_nonbg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Original placement rule, followed by a tight crop around all placed objects.

    Rule:
    - Compute each component's D4-canonical shape.
    - Shared shapes  (freq >= 2) move to the TOP of the full canvas.
    - Unique shapes  (freq == 1) move to the BOTTOM of the full canvas.
    - Preserve the bbox left column c0 on that full canvas.
    - Finally crop the result to the minimal bbox containing all non-bg pixels.
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
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return _tight_crop_nonbg(out, bg)


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
        p[int(r), int(c)] = int(color)
    return p


def _placement_ok(grid, patch, r0, c0, bg, *, connectivity=8):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = patch != int(bg)
    if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != int(bg)):
        return False

    if connectivity != 8:
        return True

    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    window = grid[rr0:rr1, cc0:cc1]
    expanded = np.zeros_like(window, dtype=bool)
    expanded[(r0 - rr0) : (r0 - rr0 + h), (c0 - cc0) : (c0 - cc0 + w)] = mask

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(expanded.shape[0], expanded.shape[0] - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(expanded.shape[1], expanded.shape[1] - dc)
            shifted = np.zeros_like(expanded)
            shifted[src_r0 + dr : src_r1 + dr, src_c0 + dc : src_c1 + dc] = expanded[
                src_r0:src_r1, src_c0:src_c1
            ]
            if np.any((window != int(bg)) & shifted):
                return False
    return True


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _choose_horizontal_band(W: int, max_obj_w: int) -> tuple[int, int]:
    max_margin = max(1, min(4, (W - max_obj_w - 1) // 2))
    if max_margin < 1:
        raise ValueError("grid too narrow for crop margins")
    left_margin = random.randint(1, max_margin)
    right_margin = random.randint(1, max_margin)
    if left_margin + max_obj_w + right_margin > W:
        raise ValueError("band infeasible")
    return int(left_margin), int(W - right_margin - 1)


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
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            objs.append({"coords": transformed, "color": avail[ci], "shared": True})
            ci += 1

    for _ in range(n_unique):
        for _attempt in range(200):
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

    max_obj_w = max(int(obj["patch"].shape[1]) for obj in objs)
    band_l, band_r = _choose_horizontal_band(W, max_obj_w)

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        lo = int(band_l)
        hi = int(band_r - w + 1)
        if lo > hi:
            raise ValueError("placement band too small")
        placed = False
        for _ in range(400):
            c0 = random.randint(lo, hi)
            if _placement_ok(out, patch, dest_r0, c0, bg, connectivity=8):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    out_cropped = _tight_crop_nonbg(out, bg)
    assert out_cropped.shape[1] < W, "variant degenerated to original width"

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(500):
            r0 = random.randint(0, H - h)
            if _placement_ok(inp, patch, r0, c0, bg, connectivity=8):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out_cropped), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D6")

# Variant of 62593bfd
# Dimension: D4 — 平移规则——水平位置保持 (dst)
# Change: 每个对象仅做垂直平移，水平位置（外接矩形左上角列c0）保持不变；顶部对象dest_r0=0，底部对象dest_r0=H-... → 水平位置也重新排列（按列坐标排序紧密排列）

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for this variant
# =================================
# 0) Source-task note
#    - The prompt asks to read the solver of source task `b5ca7ac4` first.
#    - In this workspace there is no local solver file for `b5ca7ac4`; only
#      `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json`
#      exists. So the solver-read step is skipped because the requested file is
#      unavailable. The implementation below uses only the explicit variant spec:
#      same top/bottom grouping as 62593bfd, but with left-anchored horizontal
#      packing inside each side.
#
# 1) Concrete behavior and trace
#    - Input still contains 4-6 monochromatic 8-connected objects on a background.
#    - Objects are grouped by canonical shape modulo D4 exactly as in the original.
#    - Shared shapes go to the top edge; unique shapes go to the bottom edge.
#    - The changed part is the destination column:
#         original: dest_c0 = input_c0
#         variant : sort all top objects by input c0 and place them at c=0, then
#                   immediately after the previous bbox width; do the same
#                   independently for bottom objects.
#    - Small trace:
#         top objects in input have bbox-left columns [2, 9, 15] and widths [3, 4, 2].
#         original output columns: [2, 9, 15]
#         variant  output columns: [0, 3, 7]
#      The order is preserved, but the empty horizontal gaps disappear.
#
# 2) Complexity calibration
#    - If generated examples accidentally use already-compact columns, the new
#      rule would look identical to the original. Therefore the generator must
#      intentionally create positive gaps in at least one side, preferably the
#      shared/top side because it always has at least two objects.
#    - We keep all other degrees of freedom unchanged: same D4 grouping, same
#      monochrome objects, same output-first structure. This keeps the variant
#      focused on the destination-layout change instead of mixing in new cues.
#
# 3) Implicit-assumption audit of the original main()
#    - `find_components(... monochromatic=False, connectivity=8)` assumes input
#      objects do not touch, even diagonally, regardless of color. This was
#      already true logically; the new generator now enforces a 1-cell 8-neighbor
#      margin when scattering inputs so components never merge by accident.
#    - The original `dest_c0 = d["c0"]` relied on output columns already being
#      legal independently. In the variant, legality depends on cumulative widths
#      of the whole side, so generator and solver must share one more assumption:
#      total packed width of each side must fit inside the grid width.
#    - Sorting only by `c0` is ambiguous if two same-side objects share the same
#      input column. The generator avoids such ties; the solver still includes an
#      index tiebreaker so behavior stays deterministic on arbitrary valid inputs.
#
# 4) Placement legality
#    - Output legality: top and bottom are packed independently from the left.
#      Since all objects are only translated, no overlap occurs iff each side's
#      total bbox width fits in W. The generator checks this before accepting an
#      attempt.
#    - Input legality: objects keep their chosen input columns and are scattered
#      to different rows with a no-touch constraint, so the solver can recover
#      each object as a separate 8-connected component.
#
# 5) Visual observability and cleanliness
#    - The main visible cue is that gaps between same-side objects vanish in the
#      output. This is easy to infer from a few examples because the relative
#      left-to-right order stays the same while absolute columns are discarded.
#    - Colors are kept distinct per object so packed objects remain visually
#      separable even when adjacent with no background gap.
#    - No hidden use of numeric color ids is introduced; only visible geometry
#      (shape-sharing and left-to-right order) controls the transformation.
#
# 6) New main() assumptions and generator guarantees
#    - Assumption: each foreground object is an isolated 8-connected component.
#      Guarantee: `_can_place_with_margin` enforces a 1-cell 8-neighbor buffer in
#      the generated input.
#    - Assumption: each side's packed width fits in-grid.
#      Guarantee: `_choose_spread_columns` is only used when `sum(widths) <= W`,
#      and the attempt is rejected otherwise.
#    - Assumption: the horizontal-packing rule is actually exercised.
#      Guarantee: the shared/top side is forced to contain at least one positive
#      input gap, so the new output differs from the original rule.


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


def _pack_side(entries, *, top_side, H, bg):
    out = []
    cursor = 0
    for entry in sorted(entries, key=lambda d: (int(d["c0"]), int(d["idx"]))):
        pat = entry["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = 0 if top_side else int(H - h)
        out.append((pat, dest_r0, int(cursor)))
        cursor += int(w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to the top or bottom edge by D4-canonical shape frequency.

    Shared shapes  (freq >= 2) go to the top edge.
    Unique shapes  (freq == 1) go to the bottom edge.

    Variant change:
    - Original task preserved each object's input bbox-left column c0.
    - This variant reorders horizontal placement inside each side:
      after grouping to top/bottom, objects are sorted by input c0 and packed
      tightly from the left edge with no horizontal gaps.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "idx": int(idx),
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    top_entries = [d for d in extracted if int(freq[d["shape_key"]]) >= 2]
    bottom_entries = [d for d in extracted if int(freq[d["shape_key"]]) == 1]

    out = np.full((H, W), int(bg), dtype=int)
    for pat, dest_r0, dest_c0 in _pack_side(top_entries, top_side=True, H=H, bg=bg):
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))
    for pat, dest_r0, dest_c0 in _pack_side(bottom_entries, top_side=False, H=H, bg=bg):
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))
    return out


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells."""
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


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _can_place_with_margin(grid, patch, r0, c0, bg):
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    H, W = grid.shape
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(grid[r1:r2, c1:c2] != bg):
            return False
    return True


def _choose_spread_columns(widths, W, *, require_shift):
    widths = [int(w) for w in widths]
    total = sum(widths)
    slack = int(W - total)
    if slack < 0:
        raise ValueError("packed side wider than grid")
    if require_shift and slack <= 0:
        raise ValueError("no room to create a visible horizontal shift")

    # Effective slots are: leading gap + gaps between consecutive objects.
    slots = [0] * max(1, len(widths))
    if require_shift:
        slots[random.randrange(len(slots))] += 1
        slack -= 1
    for _ in range(slack):
        slots[random.randrange(len(slots))] += 1

    cols = []
    cur = int(slots[0])
    for i, w in enumerate(widths):
        cols.append(int(cur))
        cur += int(w)
        if i + 1 < len(widths):
            cur += int(slots[i + 1])
    return cols


def _render_expected_output(objs, H, W, bg):
    out = np.full((H, W), bg, dtype=int)
    shared = []
    unique = []
    for idx, obj in enumerate(objs):
        entry = {"idx": idx, "c0": int(obj["c0"]), "pat": obj["patch"]}
        (shared if obj["shared"] else unique).append(entry)
    for pat, dest_r0, dest_c0 in _pack_side(shared, top_side=True, H=H, bg=bg):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    for pat, dest_r0, dest_c0 in _pack_side(unique, top_side=False, H=H, bg=bg):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    return out


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
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
            patch = _coords_to_patch(transformed, avail[ci], bg)
            objs.append(
                {
                    "coords": transformed,
                    "patch": patch,
                    "color": avail[ci],
                    "shared": True,
                }
            )
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            transformed = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
            patch = _coords_to_patch(transformed, avail[ci], bg)
            objs.append(
                {
                    "coords": transformed,
                    "patch": patch,
                    "color": avail[ci],
                    "shared": False,
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]

    random.shuffle(shared_objs)
    random.shuffle(unique_objs)

    shared_widths = [obj["patch"].shape[1] for obj in shared_objs]
    unique_widths = [obj["patch"].shape[1] for obj in unique_objs]

    shared_cols = _choose_spread_columns(shared_widths, W, require_shift=True)
    unique_cols = _choose_spread_columns(unique_widths, W, require_shift=(len(unique_objs) > 1))

    for obj, c0 in zip(shared_objs, shared_cols):
        obj["c0"] = int(c0)
    for obj, c0 in zip(unique_objs, unique_cols):
        obj["c0"] = int(c0)

    out = _render_expected_output(objs, H, W, bg)

    inp = np.full((H, W), bg, dtype=int)
    order = sorted(objs, key=lambda obj: (-obj["patch"].shape[0], -obj["patch"].shape[1]))
    for obj in order:
        patch = obj["patch"]
        h, _w = patch.shape
        dest_r0 = 0 if obj["shared"] else int(H - h)

        row_options = [r for r in range(0, H - h + 1) if r != dest_r0]
        if not row_options:
            row_options = [dest_r0]
        random.shuffle(row_options)

        placed = False
        for r0 in row_options:
            if _can_place_with_margin(inp, patch, r0, int(obj["c0"]), bg):
                paste(inp, patch, r0, int(obj["c0"]), transparent=bg)
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
    raise RuntimeError("failed to generate input for 62593bfd D4 variant")

# Combo variant of 62593bfd (index 6)
# combo_spec: D1/0+D4/0
# Name: 面积分组 + 紧密排列

"""
Fusion design analysis
======================

This variant combines two changes on top of source task `62593bfd`.

1) Grouping key changes from D4-canonical shape to pixel area.
   - Objects with an area value that appears at least twice are the duplicated group.
   - Objects whose area appears exactly once are the unique group.
   - Duplicated-area objects go to the top edge; unique-area objects go to the bottom.

2) Horizontal destination changes from "preserve input c0" to packed placement.
   - After the top/bottom split, each side is sorted by the object's input bbox-left
     column `c0`.
   - Within each side, objects are packed from the left with no bbox-gap:
       first object at col 0, next at previous_col + previous_width, etc.

Concrete behavior trace
-----------------------
Suppose the input has four objects:
  - A: area 5, shape L,  input c0 = 2, width 3
  - B: area 5, shape zigzag, input c0 = 9, width 4
  - C: area 8, shape T, input c0 = 6, width 3
  - D: area 11, shape plus, input c0 = 15, width 5

Original task:
  - Group by D4-canonical shape frequency.
  - Preserve each object's c0.
  - If all four shapes are distinct, all go to the bottom at columns [2, 9, 6, 15].

Fusion variant:
  - Group by area frequency instead:
      area 5 appears twice, so A and B go to the top.
      areas 8 and 11 are unique, so C and D go to the bottom.
  - Then pack per side after sorting by input c0:
      top side input order is A(2), B(9)  -> output cols [0, 3]
      bottom side input order is C(6), D(15) -> output cols [0, 3]

Visible difference:
  - The top group is no longer "same shape"; it is "same area".
  - The output no longer preserves empty horizontal gaps on either side.

Complexity calibration
----------------------
The area rule is simpler than D4-matching, so the generator must keep it visually
honest:
  - duplicated-area objects are forced to have different canonical shapes, so a
    solver cannot still succeed by shape matching;
  - all canonical shapes are globally unique, so shape-based grouping would disagree
    with the fused rule on every accepted sample;
  - at least one packed side, specifically the duplicated/top side, is forced to have
    positive slack in the input columns, so packing changes the output visibly.

Implicit-assumption audit for `main()`
--------------------------------------
`find_components(..., monochromatic=False, connectivity=8)`:
  - Assumes distinct objects do not touch, even diagonally, regardless of color.
  - Generator guarantee: input placement uses a 1-cell 8-neighborhood margin.

Patch extraction via bbox + relative coordinates:
  - Assumes every recovered component is a full object. Still valid under the fusion.

`area_key = len(rel)`:
  - Replaces the original D4-canonical key. This assumes object area is invariant under
    scattering, which is true because the solver only translates objects.

Sorting by `(c0, idx)` before packing:
  - Assumes side-internal ordering is defined by visible input columns.
  - Generator guarantee: same-side input c0 values are strictly increasing, so the
    hidden `idx` tiebreaker is never semantically needed on generated samples.

Packed placement:
  - Introduces a new assumption absent from the original solver: the total packed width
    of every side must fit within the grid.
  - Generator guarantee: both side widths fit, and for audit purposes the total width
    of all objects also fits, so counterfactual packed layouts stay well-defined.

Placement legality and cleanliness
----------------------------------
The generator remains output-first:
  - sample objects first;
  - choose input columns consistent with the eventual left-to-right order;
  - define the expected fused output by area-grouping + packing;
  - scatter objects vertically to make the input while preserving those columns.

Top and bottom outputs never collide because:
  - object bbox heights are at most 5;
  - grid height is at least 19;
  - top objects sit at row 0 and bottom objects at row `H-h`.

Objects may become horizontally adjacent in the output after packing. This is intended.
Distinct colors keep neighboring objects separable to a human viewer.

Visual observability and inference
----------------------------------
The intended human-solvable cues are:
  - top-routed objects are sometimes visibly different in shape, but have the same
    number of pixels;
  - same-side objects lose their original horizontal gaps while preserving left-to-right
    order;
  - no color-id arithmetic or hidden ordering is used.

Fusion-specific anti-degeneracy checks in the generator
-------------------------------------------------------
For each accepted sample, the generator verifies:
  - fused output != area-grouping with preserved c0
    -> packing is genuinely active;
  - fused output != shape-grouping with packing
    -> area-based grouping is genuinely active.

This ensures the dataset does not silently collapse back to either single variant.
"""

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
        nrm = _norm(cur)
        best = nrm if best is None or nrm < best else best
        flipped = _norm([(-r, c) for r, c in cur])
        best = flipped if flipped < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w:
            mask[rr, cc] = True
    return mask


def _pack_side(entries, *, top_side, H):
    placements = []
    cursor = 0
    for entry in sorted(entries, key=lambda d: (int(d["c0"]), int(d["idx"]))):
        pat = entry["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = 0 if top_side else int(H - h)
        placements.append((pat, dest_r0, int(cursor)))
        cursor += int(w)
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by repeated area, then pack each side horizontally.

    Rule:
    - Find each 8-connected foreground component.
    - Compute each component's area = number of pixels.
    - Repeated area (frequency >= 2) -> top side.
    - Unique area   (frequency == 1) -> bottom side.
    - Inside each side, sort objects by input bbox-left column `c0`.
    - Pack them from the left edge with no horizontal bbox gap.
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
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "idx": int(idx),
                "c0": int(c0),
                "pat": pat,
                "area_key": int(len(rel)),
            }
        )

    freq = Counter(d["area_key"] for d in extracted)
    top_entries = [d for d in extracted if int(freq[d["area_key"]]) >= 2]
    bottom_entries = [d for d in extracted if int(freq[d["area_key"]]) == 1]

    out = np.full((H, W), bg, dtype=int)
    for pat, dest_r0, dest_c0 in _pack_side(top_entries, top_side=True, H=H):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    for pat, dest_r0, dest_c0 in _pack_side(bottom_entries, top_side=False, H=H):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    return out


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
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _ in coords)
    mc = max(c for _, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _shape_dims(coords):
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _sample_shape(area, *, forbidden_canons):
    """Sample a non-line 8-connected shape with a new canonical key."""
    for _ in range(400):
        shape = _grow_shape(area, 8)
        canon = _canonical_shape(shape)
        hh, ww = _shape_dims(shape)
        if canon in forbidden_canons:
            continue
        if hh < 2 or ww < 2:
            continue
        return shape, canon
    raise ValueError("shape sampling failed")


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
        raise ValueError("no room to create a visible packing shift")

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


def _render_output(objs, *, group_by_area, pack_horizontal, H, W, bg):
    if group_by_area:
        freq = Counter(int(obj["area"]) for obj in objs)
        goes_top = lambda obj: int(freq[int(obj["area"])]) >= 2
    else:
        freq = Counter(obj["canon"] for obj in objs)
        goes_top = lambda obj: int(freq[obj["canon"]]) >= 2

    top_entries = []
    bottom_entries = []
    for idx, obj in enumerate(objs):
        entry = {"idx": int(idx), "c0": int(obj["c0"]), "pat": obj["patch"]}
        (top_entries if goes_top(obj) else bottom_entries).append(entry)

    out = np.full((H, W), bg, dtype=int)
    if pack_horizontal:
        top_places = _pack_side(top_entries, top_side=True, H=H)
        bottom_places = _pack_side(bottom_entries, top_side=False, H=H)
    else:
        top_places = []
        for entry in top_entries:
            h, _w = map(int, entry["pat"].shape)
            top_places.append((entry["pat"], 0, int(entry["c0"])))
        bottom_places = []
        for entry in bottom_entries:
            h, _w = map(int, entry["pat"].shape)
            bottom_places.append((entry["pat"], int(H - h), int(entry["c0"])))

    for pat, dest_r0, dest_c0 in top_places:
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    for pat, dest_r0, dest_c0 in bottom_places:
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
    duplicated_areas = random.sample(list(range(4, 13)), len(group_sizes))
    remaining_areas = [a for a in range(3, 13) if a not in duplicated_areas]
    unique_areas = random.sample(remaining_areas, n_unique)

    objs = []
    used_canons = set()

    for gsz, area in zip(group_sizes, duplicated_areas):
        for _ in range(gsz):
            shape, canon = _sample_shape(area, forbidden_canons=used_canons)
            used_canons.add(canon)
            transformed = _transform_shape(
                shape, random.randint(0, 3), random.choice([False, True])
            )
            patch = _coords_to_patch(transformed, avail[ci], bg)
            objs.append(
                {
                    "coords": transformed,
                    "patch": patch,
                    "color": int(avail[ci]),
                    "area": int(area),
                    "canon": canon,
                }
            )
            ci += 1

    for area in unique_areas:
        shape, canon = _sample_shape(area, forbidden_canons=used_canons)
        used_canons.add(canon)
        transformed = _transform_shape(
            shape, random.randint(0, 3), random.choice([False, True])
        )
        patch = _coords_to_patch(transformed, avail[ci], bg)
        objs.append(
            {
                "coords": transformed,
                "patch": patch,
                "color": int(avail[ci]),
                "area": int(area),
                "canon": canon,
            }
        )
        ci += 1

    assert len(used_canons) == len(objs)

    total_width = sum(int(obj["patch"].shape[1]) for obj in objs)
    if total_width > W:
        raise ValueError("all-object packed width exceeds grid")

    area_counts = Counter(int(obj["area"]) for obj in objs)
    shared_objs = [obj for obj in objs if int(area_counts[int(obj["area"])]) >= 2]
    unique_objs = [obj for obj in objs if int(area_counts[int(obj["area"])]) == 1]

    random.shuffle(shared_objs)
    random.shuffle(unique_objs)

    shared_widths = [int(obj["patch"].shape[1]) for obj in shared_objs]
    unique_widths = [int(obj["patch"].shape[1]) for obj in unique_objs]

    shared_cols = _choose_spread_columns(shared_widths, W, require_shift=True)
    unique_cols = _choose_spread_columns(
        unique_widths, W, require_shift=(len(unique_objs) > 1)
    )

    for obj, c0 in zip(shared_objs, shared_cols):
        obj["c0"] = int(c0)
    for obj, c0 in zip(unique_objs, unique_cols):
        obj["c0"] = int(c0)

    fused_out = _render_output(
        objs, group_by_area=True, pack_horizontal=True, H=H, W=W, bg=bg
    )
    area_only_out = _render_output(
        objs, group_by_area=True, pack_horizontal=False, H=H, W=W, bg=bg
    )
    pack_only_counterfactual = _render_output(
        objs, group_by_area=False, pack_horizontal=True, H=H, W=W, bg=bg
    )

    if np.array_equal(fused_out, area_only_out):
        raise ValueError("packing dimension inactive")
    if np.array_equal(fused_out, pack_only_counterfactual):
        raise ValueError("area-grouping dimension inactive")

    inp = np.full((H, W), bg, dtype=int)
    order = sorted(objs, key=lambda obj: (-obj["patch"].shape[0], -obj["patch"].shape[1]))
    for obj in order:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        dest_r0 = 0 if int(area_counts[int(obj["area"])]) >= 2 else int(H - h)

        row_options = [r for r in range(0, H - h + 1) if r != dest_r0]
        if not row_options:
            row_options = [dest_r0]
        random.shuffle(row_options)

        placed = False
        for r0 in row_options:
            if _can_place_with_margin(inp, patch, int(r0), c0, bg):
                paste(inp, patch, int(r0), c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, fused_out), "round-trip check failed"
    assert any(v >= 2 for v in area_counts.values())
    assert any(v == 1 for v in area_counts.values())
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")

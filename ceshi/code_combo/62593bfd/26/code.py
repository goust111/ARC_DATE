# Combo variant of 62593bfd (index 26)
# combo_spec: D3/0+D13/0
# Name: 4连通 + 小面积对象

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


"""
Design analysis for the fused variant
====================================

1. Concrete behavior and trace
   Original task:
   - detect one monochromatic 8-connected object;
   - compute its D4-canonical shape;
   - shapes seen >= 2 times go to the top, unique shapes go to the bottom.

   Fused task:
   - detect monochromatic 4-connected objects instead;
   - generator restricts every true object to area 2-5;
   - same-color pixels may touch diagonally, so one visual 8-connected blob can
     split into two 4-components.

   Example trace:
   - input contains a same-color diagonal cluster made of shape A and shape B.
   - under 8-connectivity that cluster is one merged object C.
   - under 4-connectivity it becomes two objects: A and B.
   - if A also appears twice elsewhere, A is "shared" and moves to the top.
   - B is unique and moves to the bottom.
   Difference from the original logic is therefore observable in output:
   part of one same-color diagonal blob is pulled upward while the other part is
   pulled downward.

2. Complexity calibration
   - Restricting area to 2-5 makes the shape vocabulary smaller, so duplicates
     are easier to see.
   - To avoid degenerating into the old task, every generated sample contains
     one deliberate same-color diagonal-touch cluster whose 4-connected split is
     semantically necessary.
   - To avoid overcomplication, only one such special cluster is forced per
     sample; the rest are ordinary small isolated objects.

3. Implicit assumption audit of main()
   - `get_bg`: still valid because the background remains the dominant color.
   - `find_components(... connectivity=4, monochromatic=True)`:
     the old code assumed one visual object == one 8-component and even merged
     across colors by using `monochromatic=False`. That assumption is invalid
     here. The new solver must split per color and per 4-neighborhood.
   - bbox extraction and canonicalization:
     still valid because each extracted component is a real monochromatic
     object. Small-area constraints affect generation, not the shape-grouping
     rule itself.
   - `dest_r0 = 0 or H-h`, `dest_c0 = c0`:
     still valid, but the generator must now guarantee that all 4-components
     fit without overlap after splitting.

4. Placement legality
   - Output is built first from the logical 4-components.
   - The special diagonal cluster is created only in the input, by placing two
     logical objects with a precomputed diagonal-only offset: 8-touching but not
     4-touching, no overlap, and within bounds.
   - All unrelated objects are kept separated by at least one 8-neighborhood
     cell in the input so no accidental mergers occur.

5. Visual observability and neatness
   - Objects are small polyominoes (area 2-5), so repeated shapes are easy to
     spot.
   - Only one same-color split cluster is forced, which keeps the scene legible.
   - Top and bottom rows contain separated objects; random placement uses an
     8-neighbor gap to avoid visual clutter.

6. Generator guarantees for new solver assumptions
   - every intended logical object is monochromatic and 4-connected;
   - every logical object has area 2-5;
   - at least one shared shape exists (three copies of shape A);
   - at least one unique shape exists (shape B, plus optional extra uniques);
   - one copy of A and the unique B form a same-color diagonal-touch cluster in
     the input, ensuring the fused 4-connectivity behavior is actually used.
"""


AREA_MIN = 2
AREA_MAX = 5
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEI8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _canonical_shape(coords):
    """D4-invariant canonical key for a set of coordinates."""
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


def _all_transforms(coords):
    """Unique normalized D4 transforms of a shape."""
    seen = set()
    out = []
    pts = list(coords)
    cur = list(pts)
    for _ in range(4):
        for variant in (cur, [(-r, c) for r, c in cur]):
            mr = min(r for r, c in variant)
            mc = min(c for r, c in variant)
            norm = tuple(sorted((r - mr, c - mc) for r, c in variant))
            if norm not in seen:
                seen.add(norm)
                out.append(list(norm))
        cur = [(c, -r) for r, c in cur]
    return out


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
    Group monochromatic 4-connected objects by D4-canonical shape.

    Shared shapes (freq >= 2) move to the top edge, unique shapes move to the
    bottom edge, while each object's left column is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        min_area=AREA_MIN,
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


def _enumerate_shape_library():
    by_area = {1: {((0, 0),)}}
    for area in range(2, AREA_MAX + 1):
        cur = set()
        for shape in by_area[area - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in NEI4:
                    cand = set(cells)
                    cand.add((r + dr, c + dc))
                    if len(cand) == area:
                        cur.add(_canonical_shape(cand))
        by_area[area] = cur
    out = []
    for area in range(AREA_MIN, AREA_MAX + 1):
        out.extend(sorted(by_area[area]))
    return out


SHAPE_LIBRARY = _enumerate_shape_library()


def _random_shape(exclude=()):
    excluded = set(exclude)
    choices = [shape for shape in SHAPE_LIBRARY if shape not in excluded]
    return random.choice(choices)


def _random_transform(shape):
    return [tuple(x) for x in random.choice(_all_transforms(shape))]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _patch_cells(patch, bg):
    return [(int(r), int(c)) for r, c in np.argwhere(patch != int(bg))]


def _can_place_with_gap(grid, patch, r0, c0, bg):
    h, w = map(int, patch.shape)
    H, W = map(int, grid.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    cells = _patch_cells(patch, bg)
    for rr, cc in cells:
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        if grid[ar, ac] != int(bg):
            return False
        for dr, dc in NEI8:
            nr = ar + dr
            nc = ac + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != int(bg):
                return False
    return True


def _find_diagonal_offset(coords_a, coords_b, max_bbox=7):
    a = {(int(r), int(c)) for r, c in coords_a}
    b0 = {(int(r), int(c)) for r, c in coords_b}
    candidates = []
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            b = {(r + dr, c + dc) for r, c in b0}
            if a & b:
                continue
            has_8 = False
            has_4 = False
            for r1, c1 in a:
                for r2, c2 in b:
                    md = abs(r1 - r2) + abs(c1 - c2)
                    if md == 1:
                        has_4 = True
                    if max(abs(r1 - r2), abs(c1 - c2)) == 1:
                        has_8 = True
            if has_8 and not has_4:
                rows = [r for r, _ in a] + [r for r, _ in b]
                cols = [c for _, c in a] + [c for _, c in b]
                if max(rows) - min(rows) + 1 <= max_bbox and max(cols) - min(cols) + 1 <= max_bbox:
                    candidates.append((dr, dc))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_union_patch(coords_a, coords_b, offset, color, bg):
    dr, dc = offset
    a = {(int(r), int(c)) for r, c in coords_a}
    b = {(int(r + dr), int(c + dc)) for r, c in coords_b}
    cells = a | b
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    a_anchor = (-min_r, -min_c)
    b_anchor = (dr - min_r, dc - min_c)
    mr = max(r for r, c in norm_cells)
    mc = max(c for r, c in norm_cells)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in norm_cells:
        patch[int(r), int(c)] = int(color)
    return patch, a_anchor, b_anchor


def _place_fixed_c0(out, patch, dest_r0, c0, bg):
    if not _can_place_with_gap(out, patch, dest_r0, c0, bg):
        return False
    paste(out, patch, dest_r0, c0, transparent=bg)
    return True


def _choose_bundle():
    for _ in range(200):
        shared_shape = _random_shape()
        unique_shape = _random_shape(exclude={shared_shape})
        shared_bundle = _random_transform(shared_shape)
        unique_bundle = _random_transform(unique_shape)
        offset = _find_diagonal_offset(shared_bundle, unique_bundle)
        if offset is not None:
            return shared_shape, unique_shape, shared_bundle, unique_bundle, offset
    raise ValueError("failed to build diagonal bundle")


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    shared_shape, unique_shape, shared_bundle_coords, unique_bundle_coords, offset = _choose_bundle()
    extra_unique_count = random.choice([0, 1, 2])

    shared_other_1 = _random_transform(shared_shape)
    shared_other_2 = _random_transform(shared_shape)

    used_shapes = {shared_shape, unique_shape}
    extra_shapes = []
    for _ in range(extra_unique_count):
        shape = _random_shape(exclude=used_shapes)
        used_shapes.add(shape)
        extra_shapes.append(shape)

    bundle_color = colors[0]
    logical = [
        {
            "id": "shared_bundle",
            "coords": shared_bundle_coords,
            "color": bundle_color,
            "shared": True,
            "shape_key": shared_shape,
        },
        {
            "id": "shared_1",
            "coords": shared_other_1,
            "color": colors[1],
            "shared": True,
            "shape_key": shared_shape,
        },
        {
            "id": "shared_2",
            "coords": shared_other_2,
            "color": colors[2],
            "shared": True,
            "shape_key": shared_shape,
        },
        {
            "id": "unique_bundle",
            "coords": unique_bundle_coords,
            "color": bundle_color,
            "shared": False,
            "shape_key": unique_shape,
        },
    ]
    for idx, shape in enumerate(extra_shapes, start=3):
        logical.append(
            {
                "id": f"unique_{idx}",
                "coords": _random_transform(shape),
                "color": colors[idx],
                "shared": False,
                "shape_key": shape,
            }
        )

    for obj in logical:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
        obj["h"], obj["w"] = map(int, obj["patch"].shape)
        obj["dest_r0"] = 0 if obj["shared"] else int(H - obj["h"])

    out = np.full((H, W), bg, dtype=int)
    bundle_shared = next(obj for obj in logical if obj["id"] == "shared_bundle")
    bundle_unique = next(obj for obj in logical if obj["id"] == "unique_bundle")

    _, dc = offset
    c0_candidates = []
    for c0_shared in range(W - bundle_shared["w"] + 1):
        c0_unique = c0_shared + dc
        if 0 <= c0_unique <= W - bundle_unique["w"]:
            c0_candidates.append((c0_shared, c0_unique))
    random.shuffle(c0_candidates)
    placed_bundle = False
    for c0_shared, c0_unique in c0_candidates:
        if not _place_fixed_c0(out, bundle_shared["patch"], bundle_shared["dest_r0"], c0_shared, bg):
            continue
        if _place_fixed_c0(out, bundle_unique["patch"], bundle_unique["dest_r0"], c0_unique, bg):
            bundle_shared["c0"] = c0_shared
            bundle_unique["c0"] = c0_unique
            placed_bundle = True
            break
        out = np.full((H, W), bg, dtype=int)
    if not placed_bundle:
        raise ValueError("bundle output placement failed")

    remaining = [obj for obj in logical if obj["id"] not in {"shared_bundle", "unique_bundle"}]
    random.shuffle(remaining)
    for obj in remaining:
        candidates = list(range(W - obj["w"] + 1))
        random.shuffle(candidates)
        for c0 in candidates:
            if _place_fixed_c0(out, obj["patch"], obj["dest_r0"], c0, bg):
                obj["c0"] = c0
                break
        else:
            raise ValueError("output placement failed")

    union_patch, anchor_shared, anchor_unique = _build_union_patch(
        bundle_shared["coords"], bundle_unique["coords"], offset, bundle_color, bg
    )
    union_c0 = int(bundle_shared["c0"] - anchor_shared[1])
    if union_c0 != int(bundle_unique["c0"] - anchor_unique[1]):
        raise ValueError("bundle anchors inconsistent")

    inp = np.full((H, W), bg, dtype=int)
    union_rows = list(range(H - union_patch.shape[0] + 1))
    random.shuffle(union_rows)
    for union_r0 in union_rows:
        if _can_place_with_gap(inp, union_patch, union_r0, union_c0, bg):
            paste(inp, union_patch, union_r0, union_c0, transparent=bg)
            break
    else:
        raise ValueError("bundle input placement failed")

    isolated = [obj for obj in logical if obj["id"] not in {"shared_bundle", "unique_bundle"}]
    random.shuffle(isolated)
    for obj in isolated:
        rows = list(range(H - obj["h"] + 1))
        random.shuffle(rows)
        for r0 in rows:
            if _can_place_with_gap(inp, obj["patch"], r0, obj["c0"], bg):
                paste(inp, obj["patch"], r0, obj["c0"], transparent=bg)
                break
        else:
            raise ValueError("isolated input placement failed")

    result = main(inp)
    if not np.array_equal(result, out):
        raise AssertionError("round-trip check failed")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for combo variant 62593bfd/26")

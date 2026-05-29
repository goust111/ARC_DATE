# Variant of 62593bfd
# Dimension: D15 — 源-目标基数关系 (meta)
# Change: 多对二——多个对象分为两组，分别映射到顶部边缘和底部边缘两个目标区域（many:2）...
#   → 一对一——每个对象移到与其颜色对应的目标位置

from __future__ import annotations

import random

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D15 variant
# -----------------------------------
# a) Concretization and behavior trace
# - Keep the original physical regime: 19..30 grid, 4..6 monochromatic 8-connected
#   objects, area 3..12, bbox at most 5x5.
# - Add one same-color 1-pixel marker for every object. Markers live only on the
#   top edge or bottom edge. Objects stay in the interior in the input.
# - Output removes all markers and translates each object to the edge indicated by
#   its own color marker. The marker column is the object's destination left column.
#   If the marker is on row 0, the object is pasted at (0, marker_c). If the marker
#   is on row H-1, the object is pasted at (H-h, marker_c).
# - Small trace:
#   Original logic: object A and B have the same D4 shape, so both go to the top;
#   object C is unique, so it goes to the bottom; each keeps its original c0.
#   Variant logic: A goes wherever the A-colored edge marker is, B goes to the
#   B-colored marker, C goes to the C-colored marker. Shape sharing is irrelevant.
#   The flow change is therefore at "choose destination anchor", not at extraction.
#
# b) Complexity calibration
# - The task must not collapse into a trivial one-marker puzzle. We keep 4..6 objects
#   and distinct colors so every example shows repeated color-marker pairing.
# - To avoid degeneration back to the original task, the generator still creates the
#   old shared-vs-unique shape structure, but marker edges are assigned independently.
#   This guarantees the new control variable is the marker, not shape multiplicity.
#
# c) Implicit-assumption audit of the original main()
# - `find_components(... monochromatic=False)` relied on each object being isolated
#   foreground regardless of color. That breaks here because a same-color marker and
#   object would still be separate components but we need per-color pairing, so the
#   variant must switch to `monochromatic=True`.
# - `shape_key = _canonical_shape(rel)` assumed destination choice comes from shape
#   equivalence. That assumption no longer holds, so shape extraction is removed from
#   the solver path.
# - `dest_r0 = 0 if freq[...] >= 2 else H-h` assumed only two target regions exist.
#   The variant still uses top/bottom edges as physical target bands, but selection is
#   now per-object from its own marker rather than from a global two-group partition.
# - `dest_c0 = d["c0"]` assumed horizontal position is preserved from input. That is
#   false in the variant; destination column comes from the marker column instead.
#
# d) Placement legality
# - Input objects are scattered strictly inside the grid with at least two empty rows
#   above and below them. This prevents accidental 8-connect merging with top/bottom
#   markers of the same color.
# - Output placements are chosen output-first with explicit non-overlap checks.
# - Marker columns are chosen so the object's bbox fits fully inside the grid after
#   snapping to the corresponding edge.
#
# e) Visual observability
# - The same-color 1-pixel markers are visible in the input and disappear in the
#   output while their larger same-color objects move to those edge positions.
# - Destination is inferable from visible color matching, not from hidden numeric
#   color ids.
# - The generator enforces at least two marker-edge assignments that disagree with the
#   old shared/unique rule, so the new behavior is visibly different from the seed.
#
# f) Visual cleanliness
# - Markers are single isolated pixels on the edge; objects are larger interior blobs.
#   This keeps the scene readable and avoids crowded composite structures.
# - Only top/bottom markers are used. This preserves the original edge-centric look
#   while avoiding the extra anchor ambiguity that left/right markers would introduce.
#
# g) Color inference
# - There is exactly one object and one marker per non-background color. This makes the
#   pairing unambiguous from 2..4 examples without needing extra legend structures.


def _canonical_shape(coords):
    """Canonical D4-invariant shape key, kept for generator-side anti-degeneration."""
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


def _grow_shape(area, connectivity=8, max_dim=5):
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


def _is_marker_component(comp, H):
    coords = comp["coords"]
    if int(coords.shape[0]) != 1:
        return False
    r, _c = map(int, coords[0])
    return r == 0 or r == H - 1


def _classify_components(grid):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8,
                            monochromatic=True, return_format="dict")
    assert comps

    markers = {}
    objects = []

    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if _is_marker_component(comp, H):
            assert color not in markers
            markers[color] = (r0, c0)
            continue

        patch = crop_bbox(g, (r0, c0, r1, c1))
        objects.append({
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "pat": patch,
        })

    assert objects
    assert len(objects) == len(markers)
    assert {d["color"] for d in objects} == set(markers)
    return g, H, W, bg, objects, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each object with the same-color edge marker and translate it there.

    Marker rule:
    - Each non-background color appears in exactly two monochromatic components:
      one 1-pixel marker on the top/bottom edge, and one larger interior object.
    - A top-edge marker at (0, c) means: paste that color's object with its bbox
      top-left at (0, c).
    - A bottom-edge marker at (H-1, c) means: paste that color's object with its
      bbox bottom edge flush to the grid, i.e. top-left at (H-h, c).
    - Markers themselves disappear in the output.
    """
    _g, H, W, bg, objects, markers = _classify_components(grid)
    out = np.full((H, W), bg, dtype=int)

    for obj in objects:
        pat = obj["pat"]
        h, w = map(int, pat.shape)
        marker_r, marker_c = markers[obj["color"]]
        dest_r0 = 0 if marker_r == 0 else int(H - h)
        dest_c0 = int(marker_c)
        assert 0 <= dest_c0 <= W - w
        assert _no_overlap(out, pat, dest_r0, dest_c0, bg)
        paste(out, pat, dest_r0, dest_c0, transparent=bg)

    return out


_CONFIGS = [
    ([2], 2), ([2], 3), ([3], 1), ([3], 2), ([3], 3),
    ([2, 2], 1), ([2, 2], 2),
]


def _choose_target_edges(objs):
    n = len(objs)
    for _ in range(300):
        targets = [bool(random.getrandbits(1)) for _ in range(n)]
        if len(set(targets)) < 2:
            continue
        mismatches = sum(int(t != bool(obj["shared"])) for t, obj in zip(targets, objs))
        if mismatches >= 2:
            return targets
    raise ValueError("could not choose non-degenerate marker edges")


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

    target_top_flags = _choose_target_edges(objs)

    out = np.full((H, W), bg, dtype=int)
    for obj, target_top in zip(objs, target_top_flags):
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if target_top else H - h
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["target_top"] = bool(target_top)
                obj["marker"] = (0 if target_top else H - 1, c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        mr, mc = obj["marker"]
        if inp[mr, mc] != bg:
            raise ValueError("marker overlap")
        inp[mr, mc] = int(obj["color"])

    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(500):
            r0 = random.randint(2, H - h - 2)
            c0 = random.randint(0, W - w)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["input_pos"] = (r0, c0)
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
    raise RuntimeError("failed to generate input for 62593bfd D15 variant")

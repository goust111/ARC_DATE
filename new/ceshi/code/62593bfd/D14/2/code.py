# Variant of 62593bfd
# Dimension: D14 — 干扰项与额外结构 (src)
# Change: 网格中仅包含背景色和待处理的连通分量对象，无干扰项或额外标记... → 增加标记像素指示分组结果
#
# Design analysis
# - Reference-task note: the prompt cites 88e364bc as inspiration for symbol-like instructions,
#   but this repo only contains /ARC-data/.../88e364bc.json and no solver source. I therefore
#   use it only as a style cue: isolated border symbols act as instructions and disappear in output.
# - Concrete variant:
#   Input now contains the original movable objects plus exactly two 1-pixel border markers:
#   red (2) and blue (1). They sit on opposite vertical edges, away from corners. They are not
#   part of the payload; they are a legend that says the task has two destinations:
#   red = the "top-going" class, blue = the "bottom-going" class. The movable objects still split
#   by D4-canonical shape frequency exactly as in the base task. Output removes the markers and
#   only shows the rearranged objects.
# - Why this is a visible D14 change:
#   the input now has extra non-bg singleton components on the border that can be mistaken for
#   tiny objects. A solver that blindly runs the original logic on every non-bg component will
#   miscount shapes and will also paste those singleton markers into the final grid.
# - Small behavior trace:
#   original flow on a grid with shapes A,A,B and markers {red at left edge, blue at right edge}:
#   `find_components(..., monochromatic=False)` sees five components, not three, because each
#   marker is a component too. Their 1-cell shapes are "unique", so the original solver would
#   paste them onto the bottom edge. The variant flow must first strip the two instruction pixels,
#   then group only A,A,B, so A,A go to row 0 and B goes to row H-h, with no markers in output.
# - Complexity calibration:
#   only two marker colors are used, repeated across all samples. That keeps the legend learnable.
#   The markers are always singleton edge components, while movable objects keep area 3-12, so the
#   extra structure is strong enough to matter but not so unconstrained that it becomes noisy.
# - Hidden-assumption audit against the base main():
#   1. `find_components(g, bg=bg, connectivity=8, monochromatic=False, ...)` assumed every non-bg
#      component was a movable object. No longer true: markers must be removed first.
#   2. `assert comps` still holds after removing markers because the generator guarantees 4-6
#      actual objects.
#   3. Every extracted component used to contribute one D4 shape key. If markers were left in,
#      they would create extra unique shapes and change placements.
#   4. Pasting every extracted patch into the output used to be safe. With markers present, that
#      would incorrectly preserve instruction pixels in the result.
# - Placement legality:
#   the generator reserves a 1-cell side margin for all movable objects so border markers can be
#   placed cleanly on the left/right edges. A 1-cell separation rule prevents 8-connect touching
#   between different objects and between markers and objects, which keeps solver component parsing
#   unambiguous.
# - Visual observability:
#   humans can see that every input has the same two border singletons, while every output removes
#   them and only reorganizes the larger blobs. Because the marker sides vary between samples, the
#   color identity is the stable cue rather than left/right position.
# - Cleanliness:
#   markers are isolated on side edges only; objects are kept off the outer border in the input.
#   This avoids clutter near the top/bottom destination rows.
# - Color inference:
#   marker colors are fixed to blue/red only, and movable objects never use those colors. That
#   keeps the legend visually crisp; the ambiguity comes from "is this tiny thing an object or a
#   symbol?", not from a large uncontrolled recoloring scheme.
# - New solver assumptions introduced here:
#   there is exactly one blue border singleton and exactly one red border singleton; movable objects
#   have area >= 3 and stay 8-disconnected from markers. `generate_input()` explicitly enforces all
#   of these conditions.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


TOP_MARKER = 2
BOTTOM_MARKER = 1
MARKER_COLORS = {TOP_MARKER, BOTTOM_MARKER}


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


def _is_edge_cell(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in {0, int(H - 1)} or int(c) in {0, int(W - 1)}


def _find_instruction_markers(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = g.shape
    markers: dict[int, tuple[int, int]] = {}
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[TOP_MARKER, BOTTOM_MARKER],
        return_format="dict",
    )
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        if int(coords.shape[0]) != 1:
            continue
        r, c = map(int, coords[0])
        if not _is_edge_cell(r, c, H, W):
            continue
        markers[color] = (r, c)
    assert set(markers) == MARKER_COLORS
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same payload rule as the base task, but the input contains two edge markers:
    red (2) = the top-target label, blue (1) = the bottom-target label.
    Those markers are instructions only and do not appear in the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    markers = _find_instruction_markers(g, bg)
    payload = g.copy()
    for r, c in markers.values():
        payload[int(r), int(c)] = int(bg)

    comps = find_components(
        payload,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(payload, (r0, c0, r1, c1))
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
    marker_to_dest = {
        TOP_MARKER: 0,
        BOTTOM_MARKER: None,
    }
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        is_shared = int(freq[d["shape_key"]]) >= 2
        if is_shared:
            dest_r0 = int(marker_to_dest[TOP_MARKER])
        else:
            dest_r0 = int(H - h if marker_to_dest[BOTTOM_MARKER] is None else marker_to_dest[BOTTOM_MARKER])
        dest_c0 = int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

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


def _can_place_with_gap(grid, patch, r0, c0, bg, gap=1):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = (patch != bg)
    cells = np.argwhere(mask)
    for rr, cc in cells:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        r1 = max(0, gr - gap)
        r2 = min(H, gr + gap + 1)
        c1 = max(0, gc - gap)
        c2 = min(W, gc + gap + 1)
        if np.any(grid[r1:r2, c1:c2] != bg):
            return False
    return True


def _marker_candidate_rows(grid, side, bg):
    H, W = grid.shape
    c = 0 if side == "left" else int(W - 1)
    nbr_c = 1 if side == "left" else int(W - 2)
    rows = []
    for r in range(1, H - 1):
        if int(grid[r, c]) != int(bg):
            continue
        ok = True
        for rr in range(max(0, r - 1), min(H, r + 2)):
            for cc in {c, nbr_c}:
                if int(grid[rr, cc]) != int(bg):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            rows.append(int(r))
    return rows


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
    bg = random.choice([0, 5, 8, 9])
    avail = [c for c in range(10) if c not in {bg, TOP_MARKER, BOTTOM_MARKER}]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set[tuple] = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
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

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(500):
            c0 = random.randint(1, W - w - 2)
            if _can_place_with_gap(out, patch, dest_r0, c0, bg, gap=1):
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
        for _ in range(500):
            r0 = random.randint(1, H - h - 2)
            if _can_place_with_gap(inp, patch, r0, c0, bg, gap=1):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    side_for = [("left", TOP_MARKER), ("right", BOTTOM_MARKER)]
    if random.choice([False, True]):
        side_for = [("right", TOP_MARKER), ("left", BOTTOM_MARKER)]
    for side, color in side_for:
        rows = _marker_candidate_rows(inp, side, bg)
        if not rows:
            raise ValueError("marker placement failed")
        r = random.choice(rows)
        c = 0 if side == "left" else W - 1
        inp[int(r), int(c)] = int(color)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D14")

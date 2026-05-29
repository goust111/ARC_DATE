# Variant of 62593bfd
# Dimension: D14 — 干扰项与额外结构 (src)
# Change: 网格中仅包含背景色和待处理的连通分量对象，无干扰项或额外标记... → 部分对象被标记为'冻结'不移动

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the D14 variant
# ===================================
# spec_delta:
#   added_structure:
#   - Introduce one control color used only for single-pixel "freeze markers".
#   - A frozen object has exactly one control pixel in 4-neighbor contact with it.
#   - Marker pixels stay in the output together with the frozen object.
#   moved_subset:
#   - Only non-frozen monochromatic 8-connected objects participate in D4 shape grouping.
#   - Frozen objects are excluded from shape-frequency counting and remain unchanged.
# concrete_pixel_behavior:
#   input:
#   - 4-6 monochromatic 8-connected objects on a bg grid, as in the base task.
#   - 1-2 of those objects are interior frozen objects.
#   - Each frozen object has a single control pixel of a shared color directly above/below/left/right
#     of its bbox boundary, so the relation is visually obvious.
#   output:
#   - Frozen objects and their control pixels remain exactly where they were.
#   - Unmarked objects move exactly as in the base task: shared canonical shape -> top, unique -> bottom.
# example_trace:
#   - Old logic: find_components(..., monochromatic=False) treats every non-bg connected thing as payload.
#     If a red control pixel touches a blue object, the red pixel is absorbed into that component and changes
#     both its bbox and its canonical shape.
#   - New logic: extract monochromatic components first, identify the unique "all singleton and side-touching"
#     color as the control color, map each control pixel to the single adjacent object, then skip those objects
#     during shape grouping and keep them fixed in the output.
# complexity_calibration:
#   - Marker color has repeated support across samples because every frozen object uses the same control color.
#   - Only 1-2 frozen objects are used, enough to make the new behavior visible without clutter.
#   - Frozen shapes are generated with canonical forms distinct from movable shapes to avoid accidental
#     ambiguity about whether frozen objects should affect the movable frequency count.
# implicit_assumption_audit_against_original_main:
#   - `find_components(... monochromatic=False)` assumed every non-bg pixel belongs to a movable object.
#     This is invalid once control pixels exist, so the variant must switch to monochromatic extraction.
#   - `shape_key = _canonical_shape(rel)` assumed the bbox contains only object geometry.
#     With markers present, that would corrupt shape comparison unless markers are stripped first.
#   - `freq = Counter(...)` previously counted all components. Now it must count only movable objects.
#   - `paste(out, pat, dest_r0, c0, ...)` assumed every component is translated. Frozen objects must instead
#     be repainted at their original `(r0, c0)` and their control pixel preserved.
# placement_legality:
#   - Frozen packages are forced into an interior band so top/bottom destination rows remain free.
#   - Each control pixel lies on the outside of the object's bbox and touches exactly one object by 4-neighbors.
#   - Input scattering for movable objects avoids row 0 and row `H-h` so movement is always visible.
# visual_observability:
#   - The same control color repeats within each sample, and the pixel is directly attached to its frozen object.
#   - Frozen objects stay in place while visually similar unmarked objects move, so the new rule is inferable
#     from a few examples without relying on hidden numeric properties of colors.


def _canonical_shape(coords):
    """D4-invariant canonical shape key."""
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


def _neighbors4(r, c):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _find_control_color(comps):
    """Infer the control color structurally: all components are 1-pixel markers touching one object."""
    by_color: dict[int, list[int]] = {}
    for idx, comp in enumerate(comps):
        by_color.setdefault(int(comp["color"]), []).append(int(idx))

    object_coord_to_idx: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        if int(comp["area"]) < 3:
            continue
        for r, c in comp["coords"]:
            object_coord_to_idx[(int(r), int(c))] = int(idx)

    candidates = []
    for color, idxs in by_color.items():
        if not idxs or not all(int(comps[idx]["area"]) == 1 for idx in idxs):
            continue
        ok = True
        for idx in idxs:
            r, c = map(int, comps[idx]["coords"][0])
            touching = {
                object_coord_to_idx[(rr, cc)]
                for rr, cc in _neighbors4(r, c)
                if (rr, cc) in object_coord_to_idx
            }
            if len(touching) != 1:
                ok = False
                break
        if ok:
            candidates.append((len(idxs), int(color)))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Single-pixel control markers identify frozen objects.
    - Frozen objects (and their markers) stay in place.
    - Remaining objects are grouped by D4-canonical shape and moved as in the base task:
      shared -> top, unique -> bottom, preserving bbox left column.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )
    assert comps

    ctrl_color = _find_control_color(comps)

    object_coord_to_idx: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        if ctrl_color is not None and int(comp["color"]) == int(ctrl_color):
            continue
        for r, c in comp["coords"]:
            object_coord_to_idx[(int(r), int(c))] = int(idx)

    frozen_markers_by_idx: dict[int, list[tuple[int, int]]] = {}
    if ctrl_color is not None:
        for idx, comp in enumerate(comps):
            if int(comp["color"]) != int(ctrl_color):
                continue
            r, c = map(int, comp["coords"][0])
            touching = []
            for rr, cc in _neighbors4(r, c):
                obj_idx = object_coord_to_idx.get((int(rr), int(cc)))
                if obj_idx is not None:
                    touching.append(int(obj_idx))
            touching = sorted(set(touching))
            if len(touching) == 1:
                frozen_markers_by_idx.setdefault(int(touching[0]), []).append((int(r), int(c)))

    extracted = []
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        if ctrl_color is not None and color == int(ctrl_color):
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "idx": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
                "frozen": int(idx) in frozen_markers_by_idx,
                "markers": list(frozen_markers_by_idx.get(int(idx), [])),
            }
        )

    movable = [d for d in extracted if not d["frozen"]]
    freq = Counter(d["shape_key"] for d in movable)
    out = np.full((H, W), int(bg), dtype=int)

    for d in movable:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    for d in extracted:
        if not d["frozen"]:
            continue
        paste(out, d["pat"], int(d["r0"]), int(d["c0"]), transparent=int(bg))
        for r, c in d["markers"]:
            out[int(r), int(c)] = int(ctrl_color)

    return out


# ---------------------- generator ----------------------


def _grow_shape(area, connectivity=8, max_dim=5):
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


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _no_overlap_with_marker(grid, patch, r0, c0, marker_rel, bg):
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    H, W = grid.shape
    mr, mc = map(int, marker_rel)
    rr = int(r0 + mr)
    cc = int(c0 + mc)
    return 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(bg)


def _marker_candidates(coords):
    pts = set((int(r), int(c)) for r, c in coords)
    h = max(r for r, _c in pts) + 1
    w = max(c for _r, c in pts) + 1
    cand = set()
    for c in range(w):
        if (0, c) in pts:
            cand.add((-1, c))
        if (h - 1, c) in pts:
            cand.add((h, c))
    for r in range(h):
        if (r, 0) in pts:
            cand.add((r, -1))
        if (r, w - 1) in pts:
            cand.add((r, w))
    if not cand:
        raise ValueError("no marker candidate")
    return sorted(cand)


def _footprint_bounds(patch, marker_rel):
    h, w = map(int, patch.shape)
    mr, mc = map(int, marker_rel)
    rmin = min(0, mr)
    cmin = min(0, mc)
    rmax = max(h - 1, mr)
    cmax = max(w - 1, mc)
    return int(rmin), int(cmin), int(rmax), int(cmax)


_CONFIGS = [
    ([2], 1),
    ([2], 2),
    ([3], 1),
    ([3], 2),
    ([2, 2], 1),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ctrl_color = int(avail.pop())

    group_sizes, n_unique = random.choice(_CONFIGS)
    movable_count = sum(group_sizes) + int(n_unique)
    frozen_count = random.randint(1, min(2, 6 - movable_count))

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate shared canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
            color = int(avail.pop())
            objs.append(
                {
                    "coords": transformed,
                    "color": color,
                    "shared": True,
                    "frozen": False,
                }
            )

    for _ in range(n_unique):
        for _try in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                color = int(avail.pop())
                objs.append(
                    {
                        "coords": shape,
                        "color": color,
                        "shared": False,
                        "frozen": False,
                    }
                )
                break
        else:
            raise ValueError("unique shape not found")

    for _ in range(frozen_count):
        for _try in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            color = int(avail.pop())
            marker_rel = random.choice(_marker_candidates(shape))
            objs.append(
                {
                    "coords": shape,
                    "color": color,
                    "shared": False,
                    "frozen": True,
                    "marker_rel": marker_rel,
                }
            )
            break
        else:
            raise ValueError("frozen shape not found")

    out = np.full((H, W), bg, dtype=int)

    movable = [obj for obj in objs if not obj["frozen"]]
    random.shuffle(movable)
    for obj in movable:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                placed = True
                break
        if not placed:
            raise ValueError("movable output placement failed")

    frozen = [obj for obj in objs if obj["frozen"]]
    for obj in frozen:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        marker_rel = obj["marker_rel"]
        rmin, cmin, rmax, cmax = _footprint_bounds(patch, marker_rel)
        row_lo = max(6 - rmin, 0 - rmin)
        row_hi = min(H - 7 - rmax, H - 1 - rmax)
        col_lo = max(0 - cmin, 1)
        col_hi = min(W - 1 - cmax, W - patch.shape[1] - 1)
        if row_lo > row_hi or col_lo > col_hi:
            raise ValueError("no room for frozen package")

        placed = False
        for _ in range(300):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            if _no_overlap_with_marker(out, patch, r0, c0, marker_rel, bg):
                paste(out, patch, r0, c0, transparent=bg)
                mr, mc = map(int, marker_rel)
                out[int(r0 + mr), int(c0 + mc)] = int(ctrl_color)
                obj["r0"] = int(r0)
                obj["c0"] = int(c0)
                obj["patch"] = patch
                obj["marker_abs"] = (int(r0 + mr), int(c0 + mc))
                placed = True
                break
        if not placed:
            raise ValueError("frozen output placement failed")

    inp = np.full((H, W), bg, dtype=int)

    for obj in frozen:
        patch = obj["patch"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        if not _no_overlap_with_marker(inp, patch, r0, c0, obj["marker_rel"], bg):
            raise ValueError("frozen input placement failed")
        paste(inp, patch, r0, c0, transparent=bg)
        mr, mc = obj["marker_abs"]
        inp[int(mr), int(mc)] = int(ctrl_color)

    random.shuffle(movable)
    for obj in movable:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        dest_r0 = 0 if obj["shared"] else H - h
        rows = [r for r in range(1, H - h) if r != dest_r0]
        random.shuffle(rows)
        placed = False
        for r0 in rows[:300]:
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("movable input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D14 variant")

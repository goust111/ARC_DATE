# Combo variant of 62593bfd (index 14)
# combo_spec: D5/4+D14/3
# Name: unique擦除 + 冻结标记

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the combo variant
# =====================================
# fusion_rule:
#   - Add D14/3 control pixels: some objects have a 1-pixel side marker of one repeated control color.
#     Those marked objects are frozen and stay exactly in place together with their marker.
#   - Apply D5/4 erase only to the non-frozen subset:
#       duplicated canonical shape (D4-equivalent, freq >= 2 among movable objects) -> move to top row
#       unique canonical shape (freq == 1 among movable objects) -> erase completely
#   - Final output therefore contains three visible classes:
#       1. frozen packages at original interior positions,
#       2. duplicated movable objects on row 0,
#       3. unique movable objects absent.
# concrete_trace:
#   - Suppose input has:
#       blue L at c0=2, unmarked; cyan mirrored L at c0=9, unmarked; yellow T at c0=14, unmarked;
#       green zigzag at row 11 col 5 with one red side marker, marked frozen.
#   - Original 62593bfd would move blue/cyan to top and yellow to bottom, and it would also fail to
#     interpret the red marker as control structure because non-monochromatic extraction would absorb it.
#   - D14-only would keep green+red fixed, move blue/cyan to top, and move yellow to bottom.
#   - This combo keeps green+red fixed, moves blue/cyan to top, and erases yellow entirely.
#   - The two changed stages are therefore coupled: markers first partition the object set, then erase
#     applies only inside the movable partition.
# complexity_calibration:
#   - Every generated sample contains at least one frozen object, at least one duplicated movable group,
#     and at least one unique movable object. This prevents regression to either single-variant.
#   - The control color repeats across all frozen markers within a sample, giving the solver a visible
#     structural cue rather than a one-off odd pixel.
#   - Frozen shapes are forced to have canonical forms distinct from movable shapes, so the human-visible
#     grouping problem stays on the movable subset and does not get muddied by "does the frozen one count?"
# implicit_assumption_audit:
#   - Base `find_components(... monochromatic=False)` is invalid now because a control pixel 4-touching an
#     object would pollute its geometry and canonical shape. The combo must switch to monochromatic parsing.
#   - Base frequency counting assumed every foreground component participates in grouping. That no longer
#     holds: control pixels are metadata, and frozen objects are excluded from the frequency Counter.
#   - Base placement assumed every extracted component is pasted somewhere in the output. That breaks twice:
#       frozen objects keep their original `(r0, c0)`, unique movable objects are not pasted at all.
#   - New assumption introduced by this solver: the control color can be inferred as a color whose every
#     component is a singleton touching exactly one area>=3 object by 4-neighbors. The generator enforces it.
# placement_legality:
#   - Output is built first from top-row shared movable objects plus interior frozen packages.
#   - A stricter 8-neighborhood moat is enforced between different packages, except that each marker is
#     intentionally allowed to 4-touch its own frozen object because that adjacency is the signal.
#   - Frozen packages are restricted to an interior band away from row 0 and the bottom edge so their
#     "stay put" behavior is visually distinct from both movement and erasure.
# visual_observability_and_cleanliness:
#   - In each sample, the repeated control color highlights which objects are exempt from movement.
#   - Unique movable colors disappear entirely from the output, while duplicated movable colors reappear
#     together on the top row. This is inferable from 2-3 examples without using numeric color identities.
#   - Objects are kept separated by a moat, preventing clutter from ambiguous diagonal contacts.
# post_edit_assumption_review:
#   - `_find_control_color` assumes non-marker objects have area >= 3. Generator keeps all payload objects
#     within the original spec area range 3-12, so singleton components are reserved for markers only.
#   - `_extract_scene` assumes each marker touches exactly one object. Generator places one marker outside
#     the object's bbox on a side cell and enforces package-level spacing so that is always true.
#   - `main()` assumes movable duplicated objects can be pasted at row 0 without overlap. Generator assigns
#     their output c0 first and reuses it during input scatter, so this always holds.
#   - `main()` assumes frozen markers should remain visible in the output. Generator preserves them in both
#     input and output, matching the D14 control-pixel semantics.


def _canonical_shape(coords):
    """Canonical D4-invariant shape key."""
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
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _neighbors4(r, c):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _find_control_color(comps):
    """Infer the marker color as repeated singleton components each touching one object."""
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


def _extract_scene(grid: np.ndarray):
    """Parse objects, infer frozen markers, and return payload descriptors."""
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
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
            touching = {
                object_coord_to_idx[(rr, cc)]
                for rr, cc in _neighbors4(r, c)
                if (rr, cc) in object_coord_to_idx
            }
            if len(touching) == 1:
                obj_idx = next(iter(touching))
                frozen_markers_by_idx.setdefault(int(obj_idx), []).append((int(r), int(c)))

    extracted = []
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        if ctrl_color is not None and color == int(ctrl_color):
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
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

    return extracted, ctrl_color, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Side-touching singleton control pixels mark frozen objects.
    - Frozen objects and their markers remain at their original coordinates.
    - Among the remaining objects only:
        shared D4-canonical shapes -> move to row 0, preserving c0
        unique D4-canonical shapes -> erase
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    extracted, ctrl_color, bg = _extract_scene(g)

    movable = [d for d in extracted if not d["frozen"]]
    freq = Counter(d["shape_key"] for d in movable)
    out = np.full((H, W), int(bg), dtype=int)

    for d in movable:
        if int(freq[d["shape_key"]]) < 2:
            continue
        paste(out, d["pat"], 0, int(d["c0"]), transparent=int(bg))

    for d in extracted:
        if not d["frozen"]:
            continue
        paste(out, d["pat"], int(d["r0"]), int(d["c0"]), transparent=int(bg))
        for r, c in d["markers"]:
            if ctrl_color is not None:
                out[int(r), int(c)] = int(ctrl_color)

    return out


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
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _cells_from_patch(patch, r0, c0, bg):
    cells = set()
    for pr, pc in np.argwhere(patch != bg):
        cells.add((int(r0 + pr), int(c0 + pc)))
    return cells


def _package_cells(patch, r0, c0, bg, marker_rel=None):
    cells = _cells_from_patch(patch, r0, c0, bg)
    if marker_rel is not None:
        mr, mc = map(int, marker_rel)
        cells.add((int(r0 + mr), int(c0 + mc)))
    return cells


def _can_place_cells(grid, cells, bg):
    H, W = grid.shape
    cell_set = set((int(r), int(c)) for r, c in cells)
    if not cell_set:
        return False

    for r, c in cell_set:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False

    for r, c in cell_set:
        rr0 = max(0, int(r) - 1)
        rr1 = min(H, int(r) + 2)
        cc0 = max(0, int(c) - 1)
        cc1 = min(W, int(c) + 2)
        for rr in range(rr0, rr1):
            for cc in range(cc0, cc1):
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != int(bg):
                    return False
    return True


def _paint_package(grid, patch, r0, c0, bg, marker_rel=None, marker_color=None):
    paste(grid, patch, int(r0), int(c0), transparent=int(bg))
    if marker_rel is not None and marker_color is not None:
        mr, mc = map(int, marker_rel)
        grid[int(r0 + mr), int(c0 + mc)] = int(marker_color)


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


def _footprint_bounds(patch, marker_rel=None):
    h, w = map(int, patch.shape)
    if marker_rel is None:
        return 0, 0, h - 1, w - 1
    mr, mc = map(int, marker_rel)
    rmin = min(0, mr)
    cmin = min(0, mc)
    rmax = max(h - 1, mr)
    cmax = max(w - 1, mc)
    return int(rmin), int(cmin), int(rmax), int(cmax)


def _interior_rows(H, h, forbidden):
    max_r0 = int(H - h)
    rows = [r for r in range(1, max_r0) if int(r) not in {int(x) for x in forbidden}]
    if rows:
        return rows
    return [r for r in range(0, max_r0 + 1) if int(r) not in {int(x) for x in forbidden}] or list(
        range(0, max_r0 + 1)
    )


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
            transformed = _transform_shape(
                base,
                random.randint(0, 3),
                random.choice([False, True]),
            )
            objs.append(
                {
                    "coords": transformed,
                    "color": int(avail.pop()),
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
                objs.append(
                    {
                        "coords": shape,
                        "color": int(avail.pop()),
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
            objs.append(
                {
                    "coords": shape,
                    "color": int(avail.pop()),
                    "shared": False,
                    "frozen": True,
                    "marker_rel": random.choice(_marker_candidates(shape)),
                }
            )
            break
        else:
            raise ValueError("frozen shape not found")

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)

    shared_objs = [obj for obj in objs if obj["shared"] and not obj["frozen"]]
    unique_objs = [obj for obj in objs if (not obj["shared"]) and (not obj["frozen"])]
    frozen_objs = [obj for obj in objs if obj["frozen"]]
    assert shared_objs and unique_objs and frozen_objs

    random.shuffle(shared_objs)
    for obj in shared_objs:
        patch = obj["patch"]
        _h, w = patch.shape
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            cells = _package_cells(patch, 0, c0, bg)
            if _can_place_cells(out, cells, bg):
                _paint_package(out, patch, 0, c0, bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("shared output placement failed")

    for obj in frozen_objs:
        patch = obj["patch"]
        marker_rel = obj["marker_rel"]
        rmin, cmin, rmax, cmax = _footprint_bounds(patch, marker_rel)
        row_lo = max(6 - rmin, 0 - rmin)
        row_hi = min(H - 7 - rmax, H - 1 - rmax)
        col_lo = max(1 - cmin, 0 - cmin)
        col_hi = min(W - 2 - cmax, W - 1 - cmax)
        if row_lo > row_hi or col_lo > col_hi:
            raise ValueError("no room for frozen package")

        placed = False
        for _ in range(400):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            cells = _package_cells(patch, r0, c0, bg, marker_rel=marker_rel)
            if _can_place_cells(out, cells, bg):
                _paint_package(out, patch, r0, c0, bg, marker_rel=marker_rel, marker_color=ctrl_color)
                obj["r0"] = int(r0)
                obj["c0"] = int(c0)
                obj["marker_abs"] = (int(r0 + marker_rel[0]), int(c0 + marker_rel[1]))
                placed = True
                break
        if not placed:
            raise ValueError("frozen output placement failed")

    inp = np.full((H, W), bg, dtype=int)

    for obj in frozen_objs:
        patch = obj["patch"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        marker_rel = obj["marker_rel"]
        cells = _package_cells(patch, r0, c0, bg, marker_rel=marker_rel)
        if not _can_place_cells(inp, cells, bg):
            raise ValueError("frozen input placement failed")
        _paint_package(inp, patch, r0, c0, bg, marker_rel=marker_rel, marker_color=ctrl_color)

    random.shuffle(shared_objs)
    for obj in shared_objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        rows = _interior_rows(H, h, forbidden={0})
        random.shuffle(rows)
        placed = False
        for r0 in rows:
            cells = _package_cells(patch, r0, c0, bg)
            if _can_place_cells(inp, cells, bg):
                _paint_package(inp, patch, r0, c0, bg)
                obj["r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("shared input scatter failed")

    random.shuffle(unique_objs)
    for obj in unique_objs:
        patch = obj["patch"]
        h, w = patch.shape
        forbidden = {H - h}
        rows = _interior_rows(H, h, forbidden=forbidden)
        placed = False
        for _ in range(600):
            r0 = random.choice(rows)
            c0 = random.randint(0, W - w)
            cells = _package_cells(patch, r0, c0, bg)
            if _can_place_cells(inp, cells, bg):
                _paint_package(inp, patch, r0, c0, bg)
                obj["r0"] = int(r0)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("unique input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant 14")

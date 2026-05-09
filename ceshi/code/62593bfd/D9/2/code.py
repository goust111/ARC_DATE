# Variant of 62593bfd
# Dimension: D9 — 对象间的无重叠约束 (meta)
# Change: 输出中放置后的对象互不重叠；网格足够高使得顶部对象和底部对象不会碰撞
#   → 重叠检测——重叠时水平偏移避让

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input still contains 4-6 monochromatic 8-connected objects on a background.
#      Objects are grouped by D4-canonical shape exactly as in the source task.
#    - Output still sends repeated shapes to the top edge and unique shapes to the
#      bottom edge, but "keep c0 unchanged" is no longer always legal.
#    - New rule: within each destination band (top or bottom), objects are processed
#      from left to right by their original bbox c0. Each object first tries to land
#      at that original c0. If any colored pixel would overlap an already placed
#      object, the object slides right by 1 column repeatedly until the first
#      collision-free placement is found.
#    - Small trace:
#        top object A wants c0=2 and lands at c0=2.
#        top object B wants c0=4; at c0=4 it overlaps A, at c0=5 it still overlaps,
#        at c0=6 it becomes clear, so the output uses c0=6.
#      Original task would have pasted both at their raw c0 values and relied on the
#      generator to avoid overlap entirely; this variant makes the solver resolve the
#      conflict explicitly.
#
# b) Complexity calibration
#    - The visible change is the right-shift itself, so the generator must create
#      real collisions. It does so in both the top band and the bottom band, rather
#      than leaving collisions to chance.
#    - The freedom is still controlled: only two vertical destinations exist (top or
#      bottom), and the horizontal resolution is a deterministic left-to-right scan.
#      This keeps the rule inferable from a few examples.
#
# c) Original assumptions that fail under the variant
#    - In the source solver, `dest_c0 = c0` assumed output placements were already
#      non-overlapping. That is the exact assumption D9 removes.
#    - In the source solver, `paste(out, ...)` was safe immediately after computing
#      `dest_c0`. Under D9 this can overwrite existing pixels unless we search for a
#      new c0 first.
#    - Order used to be irrelevant once each object had a unique legal target. Under
#      D9 order matters, so the solver now fixes a left-to-right band order.
#
# d) Placement legality
#    - The helper `_first_clear_c0` checks bounds and overlap while scanning right.
#    - The generator uses the same helper as the solver when constructing the target
#      output, so `main(input)` and the intended output stay aligned.
#    - Input scattering additionally enforces a one-cell 8-neighborhood gap between
#      components, because `find_components(..., monochromatic=False, connectivity=8)`
#      would merge touching objects otherwise.
#
# e) Visual observability
#    - Every generated sample contains at least one shifted top object and at least
#      one shifted bottom object. The new behavior therefore appears directly in the
#      input/output pairs instead of being a rare edge case.
#    - Colors are only identity markers; no rule depends on numeric color values.
#
# f) Visual cleanliness
#    - Objects stay monochromatic and separate in the input, so component parsing is
#      stable and the scene is readable.
#    - Output collisions are resolved by horizontal sliding only; there are no extra
#      markers or overlays, so the visual language remains close to the original task.
#
# g) Color reasoning
#    - No color remapping is introduced. Shape-sharing still drives the top/bottom
#      split, and color merely tracks which object moved where.
#
# Post-change solver assumptions checked against the generator
#    - Assumption: objects are isolated components in the input.
#      Generator guarantee: `_no_contact` enforces an 8-neighbor moat.
#    - Assumption: right-shifting eventually finds room inside the grid.
#      Generator guarantee: output-first construction rejects layouts that cannot be
#      resolved by `_first_clear_c0`.
#    - Assumption: left/right order is well defined.
#      Generator guarantee: within each band, original c0 values are distinct.


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
    mask = (patch != bg)
    return not np.any(grid[r0:r0 + h, c0:c0 + w][mask] != bg)


def _no_contact(grid, patch, r0, c0, bg):
    """Strictly stronger than no-overlap: forbid 8-neighbor touching."""
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    occ = (grid != bg)
    H, W = grid.shape
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(occ[r1:r2, c1:c2]):
            return False
    return True


def _first_clear_c0(grid, patch, r0, c0, bg):
    h, w = map(int, patch.shape)
    H, W = grid.shape
    if r0 < 0 or r0 + h > H:
        raise ValueError("vertical placement out of bounds")
    cur = int(c0)
    while cur + w <= W:
        if _no_overlap(grid, patch, r0, cur, bg):
            return cur
        cur += 1
    raise ValueError("no collision-free right shift available")


def _extract_objects(grid):
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
        h, w = map(int, pat.shape)
        extracted.append(
            {
                "idx": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "h": h,
                "w": w,
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return g, H, W, bg, extracted


def _place_objects(extracted, H, W, bg):
    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)
    placed = []

    for d in extracted:
        shared = int(freq[d["shape_key"]]) >= 2
        dest_r0 = 0 if shared else int(H - d["h"])
        placed.append({**d, "shared": shared, "dest_r0": dest_r0})

    for shared_flag in (True, False):
        band = [d for d in placed if d["shared"] == shared_flag]
        band.sort(key=lambda d: (d["c0"], d["r0"], d["idx"]))
        for d in band:
            dest_c0 = _first_clear_c0(out, d["pat"], d["dest_r0"], d["c0"], bg)
            d["dest_c0"] = int(dest_c0)
            d["shift"] = int(dest_c0 - d["c0"])
            paste(out, d["pat"], d["dest_r0"], dest_c0, transparent=int(bg))

    return out, placed


def main(grid: np.ndarray) -> np.ndarray:
    g, H, W, bg, extracted = _extract_objects(grid)
    del g
    out, _ = _place_objects(extracted, H, W, bg)
    return out


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape within a small bounding box."""
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


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 2),
    ([3], 3),
    ([2, 2], 2),
]


def _make_objects(bg, avail, group_sizes, n_unique):
    ci = 0
    objs = []
    canon_used = set()

    for gsz in group_sizes:
        for _ in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon not in canon_used:
                canon_used.add(canon)
                break
        else:
            raise ValueError("shared canonical not found")

        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            patch = _coords_to_patch(transformed, avail[ci], bg)
            h, w = map(int, patch.shape)
            objs.append(
                {
                    "coords": transformed,
                    "color": int(avail[ci]),
                    "shared": True,
                    "patch": patch,
                    "h": h,
                    "w": w,
                    "shape_key": canon,
                }
            )
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon not in canon_used:
                canon_used.add(canon)
                patch = _coords_to_patch(shape, avail[ci], bg)
                h, w = map(int, patch.shape)
                objs.append(
                    {
                        "coords": shape,
                        "color": int(avail[ci]),
                        "shared": False,
                        "patch": patch,
                        "h": h,
                        "w": w,
                        "shape_key": canon,
                    }
                )
                ci += 1
                break
        else:
            raise ValueError("unique canonical not found")

    return objs


def _assign_band_columns(objs, H, W, bg):
    """Give each object a distinct original c0 that forces at least one shift."""
    if len(objs) < 2:
        raise ValueError("band too small for D9 variant")

    band = list(objs)
    random.shuffle(band)

    min_required_shifts = 1
    if sum(obj["w"] for obj in band) > W:
        raise ValueError("band too wide")

    for _ in range(400):
        origin_limit = W - sum(obj["w"] for obj in band)
        if origin_limit < 0:
            raise ValueError("band cannot fit")
        start = random.randint(0, origin_limit)
        out_band = np.full((H, W), bg, dtype=int)
        used_c0 = set()
        prev_c0 = None
        shifted = 0
        assigned = []

        try:
            for i, obj in enumerate(band):
                low = start if i == 0 else int(prev_c0 + 1)
                high = min(int(low + 2), int(W - obj["w"]))
                if low > high:
                    raise ValueError("no legal original c0")
                options = [c for c in range(low, high + 1) if c not in used_c0]
                if not options:
                    raise ValueError("duplicate c0 only")
                c0 = random.choice(options)
                dest_r0 = 0 if obj["shared"] else int(H - obj["h"])
                dest_c0 = _first_clear_c0(out_band, obj["patch"], dest_r0, c0, bg)
                paste(out_band, obj["patch"], dest_r0, dest_c0, transparent=bg)
                assigned.append((obj, c0, dest_c0))
                shifted += int(dest_c0 > c0)
                used_c0.add(c0)
                prev_c0 = c0
        except ValueError:
            continue

        if shifted >= min_required_shifts:
            for obj, c0, dest_c0 in assigned:
                obj["c0"] = int(c0)
                obj["dest_c0"] = int(dest_c0)
            return

    raise ValueError("failed to force a band collision")


def _scatter_input(objs, H, W, bg):
    inp = np.full((H, W), bg, dtype=int)
    order = list(objs)
    random.shuffle(order)

    for obj in order:
        patch = obj["patch"]
        h, w = obj["h"], obj["w"]
        c0 = obj["c0"]
        dest_r0 = 0 if obj["shared"] else int(H - h)
        placed = False
        for _ in range(500):
            r0 = random.randint(0, H - h)
            if r0 == dest_r0:
                continue
            if _no_contact(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["input_r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")
    return inp


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(22, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    group_sizes, n_unique = random.choice(_CONFIGS)
    objs = _make_objects(bg, avail, group_sizes, n_unique)

    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]
    _assign_band_columns(shared_objs, H, W, bg)
    _assign_band_columns(unique_objs, H, W, bg)

    out_expected, placed = _place_objects(
        [
            {
                "idx": i,
                "r0": 0,
                "c0": obj["c0"],
                "h": obj["h"],
                "w": obj["w"],
                "pat": obj["patch"],
                "shape_key": obj["shape_key"],
            }
            for i, obj in enumerate(objs)
        ],
        H,
        W,
        bg,
    )

    shifts_top = sum(int(d["shared"] and d["shift"] > 0) for d in placed)
    shifts_bottom = sum(int((not d["shared"]) and d["shift"] > 0) for d in placed)
    if shifts_top < 1 or shifts_bottom < 1:
        raise ValueError("variant not visibly triggered")

    for d in placed:
        objs[d["idx"]]["dest_c0"] = int(d["dest_c0"])

    inp = _scatter_input(objs, H, W, bg)
    result = main(inp)
    assert np.array_equal(result, out_expected), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D9 variant")

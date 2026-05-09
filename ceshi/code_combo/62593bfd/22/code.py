# Combo variant of 62593bfd (index 22)
# combo_spec: D7/1+D9/0
# Name: 大量对象 + 覆盖重叠
#
# Design analysis
# ===============
# a) Concrete behavior / trace
#    - Seed task: detect each 8-connected monochromatic object, group by
#      D4-canonical shape, move duplicated shapes to row 0 and unique shapes to
#      row H-h, preserving each object's left column c0. The seed generator
#      forbids output overlap, so placement order never matters.
#    - This combo keeps the same grouping rule but changes two interacting
#      aspects:
#        1. There are now 8-10 objects, so each edge band becomes crowded.
#        2. Output overlap is allowed, and later placements cover earlier ones.
#    - To make the overlap rule observable instead of accidental, the solver now
#      stamps objects in increasing c0 order within each destination band. More
#      rightward objects are painted later, so they cover leftward ones where
#      the translated shapes intersect.
#    - Small trace:
#        * Suppose two duplicated-shape objects A and B both go to the top.
#          A has c0=4, B has c0=6, and their translated pixels overlap.
#        * Seed logic would have no visible conflict because the old generator
#          never allowed that overlap, and the seed solver effectively relied on
#          component scan order.
#        * Combo logic still sends both to row 0, but now A is pasted first and
#          B second because 4 < 6. The overlap area therefore shows B's color.
#        * With 8-10 objects, several such overlaps can occur in one sample.
#
# b) Complexity calibration
#    - The variant must look materially different from the seed, not just be a
#      denser copy. The generator therefore enforces all of the following:
#        * total object count is 8-10;
#        * there is at least one duplicated canonical-shape group and at least
#          one unique shape, so both top and bottom placements occur;
#        * at least one top-band overlap witness exists;
#        * the witness is chosen so left-to-right order disagrees with the seed
#          solver's scan order on the same input, forcing a different output;
#        * overlap depth is capped to keep the picture readable.
#    - This keeps the change visually strong without turning the grid into an
#      unreadable pile of fully hidden shapes.
#
# c) Implicit-assumption review of main()
#    - `find_components(... connectivity=8, monochromatic=False)` still works,
#      but only if input objects do not touch. The generator keeps all input
#      objects 8-separated, even when colors repeat.
#    - `shape_key = _canonical_shape(rel)` still defines the top/bottom split.
#      The large-object-count change affects frequencies, not the key itself.
#    - The seed relied on a hidden assumption: output objects do not overlap, so
#      the paste order is irrelevant. That assumption no longer holds.
#    - We therefore replace the implicit component-order stamping with an
#      explicit order: sort by `c0` (tie-break by input `r0`) inside each edge
#      band, then paste with overwrite. This is the new rule the samples teach.
#
# d) Placement legality
#    - Each object remains a valid 8-connected monochromatic component of area
#      3-12 inside a bbox no larger than 5x5.
#    - Output is still built first, but now c0 values are chosen in clustered
#      left-to-right sequences so translated objects overlap legally inside the
#      grid.
#    - Input is then scattered with the same c0 values and no overlap, so the
#      solver sees all objects separately while the output exhibits coverage.
#
# e) Visual observability
#    - A human can still see the seed rule: repeated shapes end up on top, lone
#      shapes on bottom, and x-position stays fixed.
#    - The new coverage rule is also visible because crowded top objects partly
#      overwrite one another and the rightward object wins in overlap zones.
#    - No hidden numeric feature such as color-id ordering is used; only visible
#      geometry and horizontal position determine the overlay order.
#
# f) Visual cleanliness
#    - Overlap is concentrated into one or two left-to-right clusters rather
#      than one full stack, and the generator rejects cases with depth > 3.
#    - Every object must keep at least one visible pixel in the final output, so
#      no object becomes entirely impossible to infer from the sample pair.
#
# g) New assumptions after the modification
#    - The solver assumes the input still exposes each object as its own
#      component. The generator guarantees this with non-overlapping scatter.
#    - The solver assumes c0 ordering is well defined. The generator guarantees
#      unique c0 values within each destination band, so the tie-breaker almost
#      never matters.
#    - The solver assumes overlap is meaningful, not degenerate. The generator
#      verifies that the combo solver output differs from the seed solver output
#      on the same input.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


LARGE_CONFIGS = [
    ([2, 2], 4),
    ([2, 2], 5),
    ([2, 2, 2], 2),
    ([2, 2, 2], 3),
    ([2, 2, 2], 4),
    ([3, 2], 3),
    ([3, 2], 4),
    ([3, 3], 2),
    ([3, 3], 3),
    ([3, 2, 2], 2),
    ([3, 2, 2], 3),
    ([3, 3, 2], 1),
    ([3, 3, 2], 2),
    ([2, 2, 2, 2], 1),
    ([2, 2, 2, 2], 2),
]


def _canonical_shape(coords):
    """Canonical D4-invariant key for a set of relative coords."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        n1 = _norm(cur)
        best = n1 if best is None or n1 < best else best
        n2 = _norm([(-r, c) for r, c in cur])
        best = n2 if n2 < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _extract_objects(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
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
                "id": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return extracted, H, W, bg


def _render_extracted(extracted, H: int, W: int, bg: int, *, order_mode: str) -> np.ndarray:
    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    top = [d for d in extracted if int(freq[d["shape_key"]]) >= 2]
    bottom = [d for d in extracted if int(freq[d["shape_key"]]) == 1]

    if order_mode == "seed":
        ordered_top = [d for d in extracted if d in top]
        ordered_bottom = [d for d in extracted if d in bottom]
    elif order_mode == "combo":
        ordered_top = sorted(top, key=lambda d: (int(d["c0"]), int(d["r0"]), int(d["id"])))
        ordered_bottom = sorted(bottom, key=lambda d: (int(d["c0"]), int(d["r0"]), int(d["id"])))
    else:
        raise ValueError(f"unknown order_mode: {order_mode}")

    for d in ordered_top:
        pat = d["pat"]
        paste(out, pat, 0, int(d["c0"]), transparent=int(bg))
    for d in ordered_bottom:
        pat = d["pat"]
        h = int(pat.shape[0])
        paste(out, pat, int(H - h), int(d["c0"]), transparent=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected objects by D4-canonical shape as in the seed task, but
    allow translated objects to overlap. Inside each destination band, objects
    are stamped left-to-right by c0; later, more-rightward objects cover earlier
    ones.
    """
    extracted, H, W, bg = _extract_objects(grid)
    return _render_extracted(extracted, H, W, bg, order_mode="combo")


def _seed_main(grid: np.ndarray) -> np.ndarray:
    extracted, H, W, bg = _extract_objects(grid)
    return _render_extracted(extracted, H, W, bg, order_mode="seed")


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* cells."""
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
                nr = int(r + dr)
                nc = int(c + dc)
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
    return [(int(r - mr), int(c - mc)) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(int(r - mr), int(c - mc)) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((int(mr) + 1, int(mc) + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != int(bg)
    return not np.any(grid[int(r0) : int(r0 + h), int(c0) : int(c0 + w)][mask] != int(bg))


def _can_place_isolated(grid, patch, r0, c0, bg):
    """True iff patch fits and every foreground pixel stays 8-separated."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != int(bg)):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, int(rr - 1))
        r2 = min(H, int(rr + 2))
        c1 = max(0, int(cc - 1))
        c2 = min(W, int(cc + 2))
        if bool(np.any(grid[r1:r2, c1:c2] != int(bg))):
            return False
    return True


def _render_objects(objs, H: int, W: int, bg: int):
    out = np.full((H, W), int(bg), dtype=int)
    depth = np.zeros((H, W), dtype=int)
    owner = -np.ones((H, W), dtype=int)
    overlap_pixels = 0
    max_depth = 0

    ordered = sorted(
        objs,
        key=lambda o: (0 if o["shared"] else 1, int(o["c0"]), int(o["id"])),
    )
    for obj in ordered:
        patch = obj["patch"]
        h, w = patch.shape
        r0 = 0 if obj["shared"] else int(H - h)
        c0 = int(obj["c0"])
        mask = patch != int(bg)
        region_depth = depth[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
        overlap_pixels += int(np.sum(region_depth[mask] > 0))
        region_depth[mask] += 1
        max_depth = max(max_depth, int(region_depth[mask].max(initial=0)))
        owner_region = owner[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
        owner_region[mask] = int(obj["id"])
        paste(out, patch, r0, c0, transparent=int(bg))

    visible_pixels = {int(obj["id"]): int(np.sum(owner == int(obj["id"]))) for obj in objs}
    overlap_pairs = []
    for i in range(len(ordered)):
        oi = ordered[i]
        pi = oi["patch"]
        hi, wi = pi.shape
        ri = 0 if oi["shared"] else int(H - hi)
        ci = int(oi["c0"])
        mi = pi != int(bg)
        for j in range(i + 1, len(ordered)):
            oj = ordered[j]
            if bool(oi["shared"]) != bool(oj["shared"]):
                continue
            pj = oj["patch"]
            hj, wj = pj.shape
            rj = 0 if oj["shared"] else int(H - hj)
            cj = int(oj["c0"])
            rr0 = max(int(ri), int(rj))
            cc0 = max(int(ci), int(cj))
            rr1 = min(int(ri + hi), int(rj + hj))
            cc1 = min(int(ci + wi), int(cj + wj))
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            subi = mi[int(rr0 - ri) : int(rr1 - ri), int(cc0 - ci) : int(cc1 - ci)]
            subj = (pj != int(bg))[int(rr0 - rj) : int(rr1 - rj), int(cc0 - cj) : int(cc1 - cj)]
            if bool(np.any(subi & subj)):
                overlap_pairs.append((int(oi["id"]), int(oj["id"])))

    return {
        "grid": out,
        "overlap_pixels": int(overlap_pixels),
        "max_depth": int(max_depth),
        "visible_pixels": visible_pixels,
        "overlap_pairs": overlap_pairs,
    }


def _assign_band_positions(objs, H: int, W: int, bg: int, *, min_overlap_pixels: int) -> bool:
    if not objs:
        return True
    for _ in range(600):
        ordered = list(objs)
        random.shuffle(ordered)
        n = len(ordered)
        breaks = set()
        if n >= 5:
            breaks.add(random.randint(1, n - 2))
            if n >= 7 and random.random() < 0.35:
                breaks.add(random.randint(1, n - 2))

        prefix = [0]
        for i in range(n - 1):
            if i + 1 in breaks:
                step = random.randint(4, 7)
            else:
                step = random.randint(1, 3)
            prefix.append(int(prefix[-1] + step))

        max_start = min(int(W - ordered[i]["patch"].shape[1] - prefix[i]) for i in range(n))
        if max_start < 0:
            continue
        start = random.randint(0, int(max_start))
        for i, obj in enumerate(ordered):
            obj["c0"] = int(start + prefix[i])

        stats = _render_objects(ordered, H, W, bg)
        if int(stats["overlap_pixels"]) < int(min_overlap_pixels):
            continue
        if int(stats["max_depth"]) > 3:
            continue
        if min(stats["visible_pixels"].values()) < 1:
            continue
        return True
    return False


def _assign_colors(objs, bg: int):
    colors = [c for c in range(10) if c != int(bg)]
    random.shuffle(colors)
    ordered = sorted(objs, key=lambda o: (0 if o["shared"] else 1, int(o["id"])))
    for idx, obj in enumerate(ordered):
        obj["color"] = int(colors[idx % len(colors)])
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)


def _pick_witness_pair(objs, H: int, W: int, bg: int):
    stats = _render_objects(objs, H, W, bg)
    pairs = list(stats["overlap_pairs"])
    random.shuffle(pairs)
    by_id = {int(obj["id"]): obj for obj in objs}
    for a, b in pairs:
        oa = by_id[int(a)]
        ob = by_id[int(b)]
        left, right = (oa, ob) if int(oa["c0"]) < int(ob["c0"]) else (ob, oa)
        if int(left["c0"]) == int(right["c0"]):
            continue
        if int(left["color"]) == int(right["color"]):
            continue
        if not left["shared"] or not right["shared"]:
            continue
        return left, right
    return None


def _candidate_rows(H: int, h: int):
    if H - h + 1 <= 0:
        return []
    if H - h - 1 >= 1:
        return list(range(1, H - h))
    return list(range(0, H - h + 1))


def _place_witness_pair(inp, left, right, H: int, bg: int) -> bool:
    lh, _lw = left["patch"].shape
    rh, _rw = right["patch"].shape
    left_rows = _candidate_rows(H, lh)
    right_rows = _candidate_rows(H, rh)
    random.shuffle(left_rows)
    random.shuffle(right_rows)
    for rr in right_rows:
        for rl in left_rows:
            if int(rl) <= int(rr):
                continue
            tmp = inp.copy()
            if not _can_place_isolated(tmp, right["patch"], rr, int(right["c0"]), bg):
                continue
            paste(tmp, right["patch"], rr, int(right["c0"]), transparent=int(bg))
            if not _can_place_isolated(tmp, left["patch"], rl, int(left["c0"]), bg):
                continue
            paste(tmp, left["patch"], rl, int(left["c0"]), transparent=int(bg))
            inp[:] = tmp
            left["input_r0"] = int(rl)
            right["input_r0"] = int(rr)
            return True
    return False


def _scatter_input(objs, H: int, W: int, bg: int, witness_pair) -> np.ndarray | None:
    for _ in range(300):
        inp = np.full((H, W), int(bg), dtype=int)
        for obj in objs:
            obj.pop("input_r0", None)

        placed_ids = set()
        if witness_pair is not None:
            left, right = witness_pair
            if not _place_witness_pair(inp, left, right, H, bg):
                continue
            placed_ids.add(int(left["id"]))
            placed_ids.add(int(right["id"]))

        remaining = [o for o in objs if int(o["id"]) not in placed_ids]
        random.shuffle(remaining)
        remaining.sort(key=lambda o: (-int(o["patch"].shape[1]), -int(o["patch"].shape[0])))

        ok = True
        for obj in remaining:
            patch = obj["patch"]
            h, _w = patch.shape
            rows = _candidate_rows(H, int(h))
            random.shuffle(rows)
            placed = False
            for r0 in rows:
                if _can_place_isolated(inp, patch, int(r0), int(obj["c0"]), bg):
                    paste(inp, patch, int(r0), int(obj["c0"]), transparent=int(bg))
                    obj["input_r0"] = int(r0)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return inp
    return None


def _gen_attempt():
    H = random.randint(24, 30)
    W = random.randint(24, 30)
    bg = random.choice([0, 1, 5, 8, 9])

    group_sizes, n_unique = random.choice(LARGE_CONFIGS)
    objs = []
    canon_used = set()
    next_id = 0

    for gsz in group_sizes:
        for _ in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8, max_dim=5)
            canon = _canonical_shape(base)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            for _ in range(gsz):
                transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
                objs.append(
                    {
                        "id": int(next_id),
                        "coords": transformed,
                        "shared": True,
                        "shape_key": canon,
                    }
                )
                next_id += 1
            break
        else:
            raise ValueError("duplicate canonical for shared group")

    for _ in range(n_unique):
        for _ in range(300):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8, max_dim=5)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            objs.append(
                {
                    "id": int(next_id),
                    "coords": shape,
                    "shared": False,
                    "shape_key": canon,
                }
            )
            next_id += 1
            break
        else:
            raise ValueError("unique shape not found")

    _assign_colors(objs, bg)
    top_objs = [o for o in objs if o["shared"]]
    bottom_objs = [o for o in objs if not o["shared"]]

    top_target = 4 if len(top_objs) >= 4 else 2
    if not _assign_band_positions(top_objs, H, W, bg, min_overlap_pixels=top_target):
        raise ValueError("top-band placement failed")
    if not _assign_band_positions(bottom_objs, H, W, bg, min_overlap_pixels=0):
        raise ValueError("bottom-band placement failed")

    out_stats = _render_objects(objs, H, W, bg)
    out = out_stats["grid"]
    if int(out_stats["overlap_pixels"]) < int(top_target):
        raise ValueError("insufficient overlap")
    if int(out_stats["max_depth"]) > 3:
        raise ValueError("too much overlap depth")

    witness_pair = _pick_witness_pair(objs, H, W, bg)
    if witness_pair is None:
        raise ValueError("no visible witness pair")

    inp = _scatter_input(objs, H, W, bg, witness_pair)
    if inp is None:
        raise ValueError("input scatter failed")

    combo = main(inp)
    if not np.array_equal(combo, out):
        raise AssertionError("round-trip check failed")

    seed = _seed_main(inp)
    if np.array_equal(combo, seed):
        raise ValueError("combo degenerated to seed behavior")

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D7/1+D9/0")

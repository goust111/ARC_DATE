# Combo variant of 62593bfd (index 7)
# combo_spec: D3/0+D14/1
# Name: 4连通识别 + 噪声像素干扰
#
# Design analysis
# ===============
# a) Concrete behavior / trace
#    - Seed task: treat each monochromatic 8-connected object as one unit, group objects by their
#      D4-canonical shape, move repeated shapes to the top edge and unique shapes to the bottom.
#    - Combo task: object detection changes in two coupled ways.
#      1. Use monochromatic 4-connectivity, so same-color diagonal contact does not merge objects.
#      2. Ignore area-1 singleton pixels as noise.
#    - Example trace:
#         input foreground (same color): an L-shape of area 4, diagonally touching a T-shape of
#         area 5; plus three isolated single pixels elsewhere.
#         Old 8-connected logic would merge the L+T cluster into one larger object and would also
#         count the three singletons as repeated shape ((0,0)).
#         New logic splits the diagonal cluster into two real objects because they are only
#         4-disconnected, then discards the three singleton dots entirely.
#         The L and T are grouped by their own canonical shapes and moved independently; the dots
#         disappear.
#    - The important visual difference is that one same-color diagonal cluster in the input can
#      separate into two independently relocated pieces in the output, while tiny isolated dots
#      vanish.
#
# b) Complexity calibration
#    - Real objects keep the seed scale: 4-6 objects, area 3-12, within a 5x5 bbox, with at least
#      one repeated shape group and at least one unique shape.
#    - Exactly one shared-shape object and one unique-shape object are forced into a same-color
#      diagonal-touch cluster. This is enough to make 4-connectivity observable without making the
#      scene cluttered or overly combinatorial.
#    - Noise count stays small (2-5). Using at least two singletons ensures an unfiltered solver
#      would visibly move them, so the D14 effect is not degenerate.
#
# c) Implicit-assumption review for main()
#    - `find_components(... connectivity=8, monochromatic=False)` from the seed assumes every
#      foreground contact, including diagonal contact, belongs to one object. That fails here.
#      Fix: use `connectivity=4` and `monochromatic=True`.
#    - Counting every component also fails because singleton dots are distractors. Fix:
#      `min_area=3`.
#    - Extracting `c0` from each component still works, but now it refers to each 4-connected
#      piece after diagonal splits, not to an 8-connected union.
#    - `paste(out, pat, ...)` assumes output placements do not overlap. The generator still
#      constructs the output first and rejects placements that would collide.
#
# d) Placement legality
#    - Output is still generated first, so every true object has a legal target row and fixed `c0`.
#    - The special diagonal pair is chosen from one shared object and one unique object, so their
#      output rows differ; only the column relation must be preserved.
#    - Input placement preserves each object's `c0`. The paired objects are placed with a searched
#      `(dr, dc)` offset whose union is one 8-connected blob but exactly two 4-connected objects.
#    - Noise pixels are inserted only in 8-isolated background cells, so they remain singleton
#      distractors and never change real object topology.
#
# e) Visual observability
#    - A human can see one same-color diagonal cluster split into two relocated output objects,
#      which strongly signals the connectivity change.
#    - Several isolated one-pixel dots vanish completely across examples, which exposes the noise
#      filter.
#    - Using one shared object and one unique object in the diagonal pair prevents color from
#      becoming a false cue, because same color can lead to different destinations.
#
# f) Visual cleanliness
#    - Only one diagonal cluster is forced, and the rest of the objects are scattered without
#      overlap.
#    - Noise density is low and restricted to isolated cells, so the grid remains readable.
#
# g) Change inferability
#    - No color remapping is introduced. Color is incidental.
#    - The transformation is explainable from visible structure only:
#      4-connected pieces of size >= 3 are real objects, diagonal-only contact does not fuse them,
#      and singleton dots are ignored.
#
# h) New assumptions introduced by the combo main()
#    - Assumption: at least one real component of area >= 3 exists under 4-connectivity.
#      Generator guarantee: it always builds 4-6 such objects first.
#    - Assumption: the diagonal pair stays split under 4-connectivity and merged under 8.
#      Generator guarantee: `_find_split_offset` explicitly searches for an offset with that
#      property and rejects samples that do not satisfy it.
#    - Assumption: singleton dots remain isolated. Generator guarantee: `_add_noise_pixels`
#      checks 8-neighborhood emptiness before every insertion.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


MIN_OBJECT_AREA = 3
OBJECT_CONNECTIVITY = 4
NOISE_COUNT_RANGE = (2, 5)
MAX_SHAPE_DIM = 5
_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


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
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _solve_core(grid: np.ndarray, *, connectivity: int, min_area: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=int(connectivity),
        monochromatic=True,
        min_area=int(min_area),
        return_format="dict",
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver for D3/0 + D14/1.

    Real objects are monochromatic 4-connected components of area >= 3. Their D4-canonical
    shapes are counted; repeated shapes move to the top edge and unique shapes to the bottom.
    Area-1 singleton pixels are treated as noise and ignored completely.
    """
    return _solve_core(grid, connectivity=OBJECT_CONNECTIVITY, min_area=MIN_OBJECT_AREA)


def _main_without_noise_filter(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=OBJECT_CONNECTIVITY, min_area=1)


def _main_without_4conn_split(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=8, min_area=MIN_OBJECT_AREA)


def _main_seed_like(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=8, min_area=1)


def _grow_shape(area, connectivity=4, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if int(connectivity) == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if max(all_r) - min(all_r) < int(max_dim) and max(all_c) - min(all_c) < int(max_dim):
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


def _shape_dims(coords):
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _is_valid_object_shape(coords):
    h, w = _shape_dims(coords)
    area = len(coords)
    return (
        MIN_OBJECT_AREA <= int(area) <= 12
        and 2 <= int(h) <= MAX_SHAPE_DIM
        and 2 <= int(w) <= MAX_SHAPE_DIM
        and int(area) < int(h * w)
    )


def _random_object_shape():
    for _ in range(200):
        area = random.randint(MIN_OBJECT_AREA, 12)
        coords = _grow_shape(area, connectivity=OBJECT_CONNECTIVITY, max_dim=MAX_SHAPE_DIM)
        if _is_valid_object_shape(coords):
            return coords
    raise ValueError("shape generation failed")


def _coords_to_patch(coords, color, bg):
    mr = max(int(r) for r, _c in coords)
    mc = max(int(c) for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _component_areas(mask: np.ndarray, connectivity: int) -> list[int]:
    comps = find_components(
        np.array(mask, dtype=int),
        bg=0,
        connectivity=int(connectivity),
        monochromatic=True,
        min_area=1,
        return_format="dict",
    )
    return sorted(int(comp["area"]) for comp in comps)


def _find_split_offset(coords_a, coords_b):
    """
    Search a relative offset (dr, dc) such that the same-color union is one 8-connected component
    but two 4-connected components with the original areas preserved.
    """
    area_a = len(coords_a)
    area_b = len(coords_b)
    ha, wa = _shape_dims(coords_a)
    hb, wb = _shape_dims(coords_b)
    candidates = []
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            if dr == 0 and dc == 0:
                continue
            min_r = min(0, dr)
            min_c = min(0, dc)
            placed_a = {(int(r - min_r), int(c - min_c)) for r, c in coords_a}
            placed_b = {(int(r + dr - min_r), int(c + dc - min_c)) for r, c in coords_b}
            if placed_a & placed_b:
                continue
            all_pts = placed_a | placed_b
            H = max(r for r, _c in all_pts) + 1
            W = max(c for _r, c in all_pts) + 1
            grid = np.zeros((H, W), dtype=int)
            for r, c in all_pts:
                grid[r, c] = 1
            areas4 = _component_areas(grid, connectivity=4)
            areas8 = _component_areas(grid, connectivity=8)
            if areas4 == sorted([area_a, area_b]) and areas8 == [area_a + area_b]:
                candidates.append((int(dr), int(dc)))
    if not candidates:
        return None
    candidates.sort(key=lambda rc: (abs(rc[0]) + abs(rc[1]), abs(rc[0]), abs(rc[1])))
    best_len = abs(candidates[0][0]) + abs(candidates[0][1])
    close = [rc for rc in candidates if abs(rc[0]) + abs(rc[1]) == best_len]
    return random.choice(close)


def _is_isolated_noise_slot(grid, r, c, bg):
    H, W = grid.shape
    if not (1 <= int(r) < H - 1):
        return False
    if grid[int(r), int(c)] != bg:
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] != bg:
                return False
    return True


def _add_noise_pixels(grid, bg, palette):
    H, W = grid.shape
    count = random.randint(*NOISE_COUNT_RANGE)
    noise = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(W)
            if _is_isolated_noise_slot(grid, r, c, bg)
        ]
        if not candidates:
            if len(noise) >= NOISE_COUNT_RANGE[0]:
                break
            raise ValueError("not enough isolated slots for noise")
        r, c = random.choice(candidates)
        grid[int(r), int(c)] = int(random.choice(palette))
        noise.append((int(r), int(c)))
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("too little noise")
    return noise


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    real4 = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        min_area=MIN_OBJECT_AREA,
        return_format="dict",
    )
    real8 = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        min_area=MIN_OBJECT_AREA,
        return_format="dict",
    )
    all4 = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        min_area=1,
        return_format="dict",
    )
    singletons = sum(1 for comp in all4 if int(comp["area"]) == 1)
    return {
        "real4": len(real4),
        "real8": len(real8),
        "split_gain": len(real4) - len(real8),
        "singletons": int(singletons),
    }


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    group_sizes, n_unique = random.choice(_CONFIGS)
    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        for _ in range(200):
            base = _random_object_shape()
            canon = _canonical_shape(base)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            for _copy in range(gsz):
                transformed = _transform_shape(
                    base, random.randint(0, 3), random.choice([False, True])
                )
                objs.append({"coords": transformed, "shared": True, "shape_key": canon})
            break
        else:
            raise ValueError("shared shape not found")

    for _ in range(n_unique):
        for _attempt in range(200):
            shape = _random_object_shape()
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            objs.append({"coords": shape, "shared": False, "shape_key": canon})
            break
        else:
            raise ValueError("unique shape not found")

    shared_ids = [i for i, obj in enumerate(objs) if obj["shared"]]
    unique_ids = [i for i, obj in enumerate(objs) if not obj["shared"]]
    random.shuffle(shared_ids)
    random.shuffle(unique_ids)

    pair = None
    for i in shared_ids:
        for j in unique_ids:
            offset = _find_split_offset(objs[i]["coords"], objs[j]["coords"])
            if offset is not None:
                pair = (int(i), int(j), int(offset[0]), int(offset[1]))
                break
        if pair is not None:
            break
    if pair is None:
        raise ValueError("no valid diagonal split pair")

    pair_shared, pair_unique, pair_dr, pair_dc = pair
    pair_color = avail.pop()
    objs[pair_shared]["color"] = pair_color
    objs[pair_unique]["color"] = pair_color
    for obj in objs:
        if "color" not in obj:
            obj["color"] = avail.pop()

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)

    patch_a = objs[pair_shared]["patch"]
    patch_b = objs[pair_unique]["patch"]
    ha, wa = patch_a.shape
    hb, wb = patch_b.shape
    c0_low = max(0, -pair_dc)
    c0_high = min(W - wa, W - wb - pair_dc)
    if c0_low > c0_high:
        raise ValueError("paired columns do not fit")
    c0_a = random.randint(c0_low, c0_high)
    c0_b = c0_a + pair_dc
    objs[pair_shared]["c0"] = int(c0_a)
    objs[pair_unique]["c0"] = int(c0_b)
    paste(out, patch_a, 0, c0_a, transparent=bg)
    paste(out, patch_b, H - hb, c0_b, transparent=bg)

    other_ids = [i for i in range(len(objs)) if i not in {pair_shared, pair_unique}]
    random.shuffle(other_ids)
    for idx in other_ids:
        obj = objs[idx]
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(300):
            c0 = random.randint(0, W - w)
            if _no_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    candidate_rows = list(
        range(
            max(0, -pair_dr),
            min(H - ha, H - hb - pair_dr) + 1,
        )
    )
    interior_rows = [
        r0
        for r0 in candidate_rows
        if 1 <= int(r0) <= H - ha - 1 and 1 <= int(r0 + pair_dr) <= H - hb - 1
    ]
    if interior_rows:
        candidate_rows = interior_rows
    random.shuffle(candidate_rows)
    placed_pair = False
    for r0_a in candidate_rows:
        r0_b = int(r0_a + pair_dr)
        if not _no_overlap(inp, patch_a, r0_a, c0_a, bg):
            continue
        paste(inp, patch_a, r0_a, c0_a, transparent=bg)
        if _no_overlap(inp, patch_b, r0_b, c0_b, bg):
            paste(inp, patch_b, r0_b, c0_b, transparent=bg)
            placed_pair = True
            break
        inp[:, :] = bg
    if not placed_pair:
        raise ValueError("input pair placement failed")

    remaining_ids = [i for i in range(len(objs)) if i not in {pair_shared, pair_unique}]
    random.shuffle(remaining_ids)
    for idx in remaining_ids:
        obj = objs[idx]
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(300):
            r0 = random.randint(0, H - h)
            if _no_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    noise = _add_noise_pixels(inp, bg, [pair_color] + avail)
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise missing")

    metrics = _fusion_metrics(inp)
    if metrics["split_gain"] < 1:
        raise ValueError("4-connectivity effect missing")
    if metrics["singletons"] < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise effect missing")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    assert not np.array_equal(_main_without_noise_filter(inp), out), "noise-filter effect degenerated"
    assert not np.array_equal(_main_without_4conn_split(inp), out), "4-connectivity effect degenerated"
    assert not np.array_equal(_main_seed_like(inp), out), "combo degenerated to seed-like behavior"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D3/0+D14/1")

# Combo variant of 62593bfd (index 20)
# combo_spec: D1/2+D3/0+D14/1
# Name: 外接矩形分组 + 4连通 + 噪声
#
# Design analysis
# ===============
# a) Concrete behavior / trace
#    - Seed task: every non-bg 8-connected component is a real object; objects
#      are grouped by exact D4-canonical shape and then moved to top/bottom.
#    - This combo changes three points together:
#        1. Objects are the 4-connected components, not 8-connected ones.
#        2. Grouping key is tight bbox size (h, w), not exact shape.
#        3. Area-1 isolated dots are noise and must disappear.
#    - Small trace:
#        * Input contains two same-color blobs that touch only diagonally:
#          left blob bbox 2x3, right blob bbox 3x2. Under 8-connectivity they
#          would look like one larger object; under 4-connectivity they are two
#          objects. Suppose another unrelated object also has bbox 2x3.
#        * Seed-style reasoning would either merge the diagonal pair or compare
#          exact shapes, so it would not send the left blob to the same group as
#          the unrelated 2x3 object.
#        * Combo reasoning first splits by 4-connectivity, ignores singleton
#          noise dots, measures each surviving component's bbox, and moves every
#          duplicated bbox size to row 0 while unique bbox sizes go to row H-h.
#        * Therefore the diagonal pair can split and even go to opposite edges.
#
# b) Complexity calibration
#    - The variant must visibly differ from all three partial tasks, so the
#      generator enforces:
#        * at least one duplicated bbox-size group whose members have distinct
#          exact shapes, making bbox grouping observable;
#        * at least one unique bbox size, so both top and bottom branches occur;
#        * at least one same-color diagonal pair whose members are opposite-fate
#          objects (one duplicated-size, one unique-size), making 4-connectivity
#          visibly necessary;
#        * 2-5 isolated noise pixels, so filtering singletons matters.
#
# c) Implicit-assumption review of main()
#    - `find_components(... connectivity=8, min_area=1)` from the seed no
#      longer works: diagonal contact must split, and singleton noise must be
#      dropped. We replace it with connectivity=4 and min_area=3.
#    - `shape_key = _canonical_shape(rel)` is no longer the grouping key.
#      `patch.shape` becomes the key; the generator therefore guarantees a
#      repeated bbox size and a unique bbox size in every sample.
#    - `crop_bbox(...)` still provides the exact measured bbox, so the new key
#      is stable.
#    - `paste(...)` still assumes translated objects do not overlap. The
#      generator preserves this via output-first placement with 8-neighborhood
#      clearance on each destination edge.
#
# d) Placement legality
#    - Real objects are generated as 4-connected shapes of area 3-12 with exact
#      bbox sizes in 2..5 by 2..5.
#    - Output is built first: duplicated bbox groups at row 0, unique groups at
#      row H-h, c0 preserved, no overlap or 8-touch within the same edge band.
#    - Input is built next: one chosen shared object and one chosen unique
#      object are placed as a same-color diagonal pair, so they form one
#      8-connected-looking cluster but remain two 4-components. All other real
#      objects are 8-isolated from everything else.
#    - Noise is inserted only into background cells that are 8-isolated from all
#      foreground, so each dot remains a singleton distractor.
#
# e) Visual observability
#    - A human can see that tiny isolated dots vanish.
#    - A human can also see that diagonal touching does not count as one object,
#      because a same-color diagonal cluster can split and its pieces can move
#      independently.
#    - Finally, repeated bbox sizes are visible because different silhouettes
#      with the same height x width still go to the same edge.
#
# f) Visual cleanliness
#    - Outside the intentional diagonal pair, the generator keeps one-cell
#      8-neighborhood clearance between foreground items. This avoids accidental
#      merges and keeps the scene readable.
#    - Colors are only a separator aid. One pair reuses a color on purpose to
#      make the diagonal split legible, but no hidden color mapping is used.
#
# g) New assumptions after the modification
#    - Every real 4-component has area >= 3; otherwise it would be
#      indistinguishable from noise. The generator guarantees this by generating
#      each component independently before diagonal pairing.
#    - The solver assumes that bbox grouping alone determines destination. The
#      generator guarantees this is non-degenerate by making the combo output
#      differ from three partial baselines:
#        * 4-connectivity + shape grouping + noise filter
#        * 8-connectivity + bbox grouping + noise filter
#        * 4-connectivity + bbox grouping + no noise filter

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


MIN_OBJECT_AREA = 3
MAX_OBJECT_AREA = 12
NOISE_COUNT_RANGE = (2, 5)

_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]

_DUP_DIMS = [
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
]
_ALL_DIMS = [(h, w) for h in range(2, 6) for w in range(2, 6)]


def _canonical_shape(coords):
    """Canonical D4-invariant key, used only for generator anti-degeneracy."""
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
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _solve_core(grid: np.ndarray, *, connectivity: int, min_area: int, group_mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=int(connectivity),
        monochromatic=False,
        min_area=int(min_area),
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
        if group_mode == "bbox":
            key = tuple(map(int, pat.shape))
        elif group_mode == "shape":
            key = _canonical_shape(rel)
        else:
            raise ValueError(f"unknown group_mode: {group_mode}")
        extracted.append({"c0": int(c0), "pat": pat, "group_key": key})

    freq = Counter(d["group_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["group_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver for 62593bfd:
    - objects = 4-connected foreground components
    - ignore components of area < 3 (singleton noise)
    - group surviving objects by bbox size (h, w)
    - duplicated bbox sizes -> top edge, unique bbox sizes -> bottom edge
    - preserve each object's left column c0
    """
    return _solve_core(grid, connectivity=4, min_area=MIN_OBJECT_AREA, group_mode="bbox")


def _shape4_filtered(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=4, min_area=MIN_OBJECT_AREA, group_mode="shape")


def _bbox8_filtered(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=8, min_area=MIN_OBJECT_AREA, group_mode="bbox")


def _bbox4_noisy(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, connectivity=4, min_area=1, group_mode="bbox")


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _touches_foreground(grid, r, c, bg):
    H, W = grid.shape
    for rr in range(max(0, int(r) - 1), min(H, int(r) + 2)):
        for cc in range(max(0, int(c) - 1), min(W, int(c) + 2)):
            if int(grid[rr, cc]) != int(bg):
                return True
    return False


def _can_place(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if _touches_foreground(grid, rr, cc, bg):
            return False
    return True


def _seed_path_for_bbox(h, w):
    """Monotone 4-connected path from (0,0) to (h-1,w-1)."""
    r = 0
    c = 0
    cells = {(0, 0)}
    while r < h - 1 or c < w - 1:
        moves = []
        if r < h - 1:
            moves.append((r + 1, c))
        if c < w - 1:
            moves.append((r, c + 1))
        r, c = random.choice(moves)
        cells.add((r, c))
    return cells


def _grow_shape_in_bbox(h, w):
    """Grow a random 4-connected shape with exact bbox (h, w)."""
    max_area = min(MAX_OBJECT_AREA, h * w)
    seed = _seed_path_for_bbox(h, w)
    min_area = max(MIN_OBJECT_AREA, len(seed))
    target_area = random.randint(min_area, max_area)

    cells = set(seed)
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    coords = sorted((int(r), int(c)) for r, c in cells)
    assert max(r for r, _ in coords) == h - 1
    assert max(c for _, c in coords) == w - 1
    return coords


def _sample_shape_for_dims(h, w, forbidden_canons):
    for _ in range(400):
        coords = _grow_shape_in_bbox(h, w)
        canon = _canonical_shape(coords)
        if canon not in forbidden_canons:
            return coords, canon
    raise ValueError(f"shape not found for dims {(h, w)}")


def _count_components(grid: np.ndarray, *, connectivity: int, min_area: int) -> int:
    bg = int(get_bg(grid))
    comps = find_components(
        np.array(grid, dtype=int),
        bg=bg,
        connectivity=int(connectivity),
        monochromatic=False,
        min_area=int(min_area),
        return_format="dict",
    )
    return len(comps)


def _count_singletons(grid: np.ndarray) -> int:
    bg = int(get_bg(grid))
    comps = find_components(
        np.array(grid, dtype=int),
        bg=bg,
        connectivity=8,
        monochromatic=False,
        min_area=1,
        return_format="dict",
    )
    return sum(1 for comp in comps if int(comp["area"]) == 1)


def _is_isolated_noise_slot(grid, r, c, bg):
    H, W = grid.shape
    if not (1 <= int(r) < H - 1):
        return False
    if int(grid[int(r), int(c)]) != int(bg):
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg):
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
        grid[r, c] = int(random.choice(palette))
        noise.append((int(r), int(c)))
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("too little noise")
    return noise


def _choose_dims(H, W, group_sizes, n_unique):
    dup_choices = [d for d in _DUP_DIMS if d[0] <= H and d[1] <= W]
    if len(dup_choices) < len(group_sizes):
        raise ValueError("not enough duplicated dims choices")
    dup_dims = random.sample(dup_choices, len(group_sizes))

    used = set(dup_dims)
    unique_choices = [d for d in _ALL_DIMS if d not in used and d[0] <= H and d[1] <= W]
    if len(unique_choices) < n_unique:
        raise ValueError("not enough unique dims choices")
    unique_dims = random.sample(unique_choices, n_unique)
    return dup_dims, unique_dims


def _assign_colors(objs, bg):
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    shared_obj = next(obj for obj in objs if obj["shared"])
    unique_obj = next(obj for obj in objs if not obj["shared"])
    pair_color = int(palette[0])
    color_i = 1

    for obj in objs:
        if obj is shared_obj or obj is unique_obj:
            obj["color"] = pair_color
        else:
            obj["color"] = int(palette[color_i])
            color_i += 1

    return shared_obj, unique_obj, palette


def _place_output(objs, H, W, bg, pair_shared, pair_unique):
    out = np.full((H, W), bg, dtype=int)

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    left = pair_shared
    right = pair_unique
    w_left = int(left["patch"].shape[1])
    w_right = int(right["patch"].shape[1])
    c0_left = random.randint(0, W - (w_left + w_right))
    c0_right = c0_left + w_left
    left["c0"] = int(c0_left)
    right["c0"] = int(c0_right)

    paste(out, left["patch"], 0, left["c0"], transparent=bg)
    paste(out, right["patch"], H - int(right["patch"].shape[0]), right["c0"], transparent=bg)

    remaining = [obj for obj in objs if obj is not left and obj is not right]
    random.shuffle(remaining)
    for obj in remaining:
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(500):
            c0 = random.randint(0, W - w)
            if _can_place(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    return out


def _place_input(objs, H, W, bg, pair_shared, pair_unique):
    inp = np.full((H, W), bg, dtype=int)

    left = pair_shared
    right = pair_unique
    h_left = int(left["patch"].shape[0])
    h_right = int(right["patch"].shape[0])
    row_min = 1
    row_max = H - (h_left + h_right) - 2
    if row_max < row_min:
        raise ValueError("not enough room for diagonal pair")

    r0_left = random.randint(row_min, row_max)
    r0_right = r0_left + h_left
    paste(inp, left["patch"], r0_left, left["c0"], transparent=bg)
    paste(inp, right["patch"], r0_right, right["c0"], transparent=bg)
    left["input_r0"] = int(r0_left)
    right["input_r0"] = int(r0_right)

    remaining = [obj for obj in objs if obj is not left and obj is not right]
    random.shuffle(remaining)
    for obj in remaining:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(500):
            r0 = random.randint(0, H - h)
            if _can_place(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["input_r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    return inp


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])

    group_sizes, n_unique = random.choice(_CONFIGS)
    dup_dims, unique_dims = _choose_dims(H, W, group_sizes, n_unique)

    objs = []
    used_canons = set()

    for dims, gsz in zip(dup_dims, group_sizes):
        h, w = map(int, dims)
        for _ in range(gsz):
            coords, canon = _sample_shape_for_dims(h, w, used_canons)
            used_canons.add(canon)
            objs.append(
                {
                    "coords": coords,
                    "shared": True,
                    "bbox_key": (h, w),
                    "canon": canon,
                }
            )

    for dims in unique_dims:
        h, w = map(int, dims)
        coords, canon = _sample_shape_for_dims(h, w, used_canons)
        used_canons.add(canon)
        objs.append(
            {
                "coords": coords,
                "shared": False,
                "bbox_key": (h, w),
                "canon": canon,
            }
        )

    pair_shared, pair_unique, palette = _assign_colors(objs, bg)
    out = _place_output(objs, H, W, bg, pair_shared, pair_unique)
    inp = _place_input(objs, H, W, bg, pair_shared, pair_unique)

    noise = _add_noise_pixels(inp, bg, palette)
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise missing")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    real4 = _count_components(inp, connectivity=4, min_area=MIN_OBJECT_AREA)
    real8 = _count_components(inp, connectivity=8, min_area=MIN_OBJECT_AREA)
    singleton_noise = _count_singletons(inp)

    assert real4 == len(objs), "unexpected real 4-components count"
    assert real8 < real4, "4-connectivity effect missing"
    assert singleton_noise >= NOISE_COUNT_RANGE[0], "noise merged unexpectedly"
    assert not np.array_equal(_shape4_filtered(inp), out), "bbox grouping effect missing"
    assert not np.array_equal(_bbox8_filtered(inp), out), "4-connectivity effect degenerated"
    assert not np.array_equal(_bbox4_noisy(inp), out), "noise-filter effect missing"

    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo 20")

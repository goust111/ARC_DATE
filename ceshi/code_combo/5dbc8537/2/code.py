# Combo variant of 5dbc8537 (index 2)
# combo_spec: D1/1+D4/1
# Name: 旋转匹配+旋转填入

from __future__ import annotations

import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The strip-shaped base template from the original task is preserved.
#    - Its holes are no longer solid rectangles; each hole is a separate irregular 4-connected
#      polyomino with a 1-cell base border around it.
#    - Each outside sprite is the same polyomino class as one hole, but the observed source
#      orientation is inverse-rotated relative to the hole by a non-zero quarter turn.
#    - Small trace:
#      original: a 3x2 filled rectangle matches a 3x2 hole by bbox, then pastes unchanged.
#      combo: a hook-shaped sprite outside the strip may appear as a 2x3 rotation; the solver
#      must rotate it until its mask matches the hook hole, and the only legal quarter-turn is
#      determined by whether that hole sits in the top/left, middle, or bottom/right zone.
# b) Complexity calibration:
#    - Only three rotations are used: 90/180/270 degrees. This keeps the rule learnable.
#    - Every generated instance requires all three zone rotations on non-symmetric pieces, so
#      the location-to-rotation mapping has repeated visual support.
#    - Piece classes are unique up to rotation, which removes assignment ambiguity without
#      needing hidden metadata or color ids.
# c) Old assumptions that break:
#    - `hole_mask[r:r+h, c:c+w].all()` is no longer valid because holes are sparse masks.
#    - Matching by bbox alone is invalid because different rotations can swap bbox dimensions.
#    - Pasting the raw source patch is wrong because the source must be rotated to the hole.
#    - Allowing arbitrary matching rotations would still solve many D1-only instances, so this
#      combo additionally constrains the legal rotation by the hole's position inside the strip.
# d) Placement legality:
#    - The generator spaces hole bboxes by at least one base cell along the strip axis and keeps
#      every bbox one cell away from the strip boundary, so holes never overlap or merge.
#    - Outside sprites are scattered with `random_free_location(..., padding=1)`, so they remain
#      isolated from each other and from the template.
# e) Visual observability:
#    - Masks are rotation-asymmetric, so turning a piece is visible even for monochrome sprites.
#    - Many sprites are bichromatic with directional accents, so 180-degree cases remain readable.
#    - The rotation rule depends only on visible geometry: the hole center relative to the strip
#      center along the strip's long axis.
# f) Visual cleanliness:
#    - Holes are separated components instead of one tangled cavity, keeping the jigsaw readable.
#    - The strip/frame and the scattered sprites stay visually similar to the source task.
# g) Color inferability:
#    - There is no color remapping. Colors move with the rotated sprite, and palette size stays
#      moderate so shape and position remain the dominant signals.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes each hole is a separate 4-connected component; the generator enforces gaps.
#    - `main()` assumes every source sprite has exactly one hole whose mask equals one of its
#      rotated variants and that this matching variant uses the zone-prescribed angle.
#    - `main()` assumes no piece is invariant under 90/180/270-degree rotation; the generator
#      rejects such masks to keep the rotation search and the zone rule both observable.

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 8, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 5, 8
PIECE_DIM_MIN, PIECE_DIM_MAX = 2, 4
PIECE_AREA_MIN = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.16, 0.32
MIN_PALETTE_SIZE = 5
MAX_RETRIES = 500
PIECE_RETRIES = 240


def _rotate(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, int(k))


def _rotation_variants(patch: np.ndarray, mask: np.ndarray) -> list[dict]:
    uniq: list[dict] = []
    seen: set[bytes] = set()
    for k in range(4):
        rm = _rotate(mask, k)
        key = rm.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"k": int(k), "mask": rm, "patch": _rotate(patch, k)})
    return uniq


def _shape_class_signature(mask: np.ndarray) -> bytes:
    return min(_rotate(mask, k).tobytes() for k in range(4))


def _rotation_for_hole(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> int:
    """
    Map a hole's center position along the strip's long axis to a required quarter-turn count.

    top/left third -> 1, middle third -> 2, bottom/right third -> 3
    """
    if dh >= dw:
        coord = float(rr0) + float(h - 1) / 2.0
        span = float(dh)
    else:
        coord = float(cc0) + float(w - 1) / 2.0
        span = float(dw)
    cut1 = span / 3.0
    cut2 = 2.0 * span / 3.0
    if coord < cut1:
        return 1
    if coord >= cut2:
        return 3
    return 2


def _zone_index(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> int:
    return _rotation_for_hole(rr0, cc0, h, w, dh, dw) - 1


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(PIECE_RETRIES):
        h = random.randint(PIECE_DIM_MIN, PIECE_DIM_MAX)
        w = random.randint(PIECE_DIM_MIN, PIECE_DIM_MAX)
        full = h * w
        if full <= PIECE_AREA_MIN:
            continue
        target = random.randint(max(PIECE_AREA_MIN, (full + 1) // 2), full - 1)

        mask = np.zeros((h, w), dtype=bool)
        sr, sc = random.randrange(h), random.randrange(w)
        mask[sr, sc] = True
        frontier = [(int(sr), int(sc))]

        while int(mask.sum()) < target:
            nbrs: list[tuple[int, int]] = []
            for r, c in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        nbrs.append((rr, cc))
            if not nbrs:
                break
            rr, cc = random.choice(nbrs)
            mask[rr, cc] = True
            frontier.append((int(rr), int(cc)))

        if int(mask.sum()) != target:
            continue

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        mh, mw = map(int, mask.shape)
        if mh < 2 or mw < 2:
            continue
        if bool(mask.all()):
            continue
        if int(mask.sum()) < PIECE_AREA_MIN:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len({_rotate(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    return None


def _choose_colors(bg: int, base: int, count: int) -> list[tuple[int, ...]] | None:
    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette_size = random.randint(MIN_PALETTE_SIZE, len(available))
    palette = random.sample(available, palette_size)
    out: list[tuple[int, ...]] = []
    for _ in range(count):
        if random.random() < 0.7:
            a, b = random.sample(palette, 2)
            out.append((int(a), int(b)))
        else:
            out.append((int(random.choice(palette)),))
    return out


def _make_piece_patch(mask: np.ndarray, cols: tuple[int, ...], bg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.full((h, w), int(bg), dtype=int)
    out[mask] = int(cols[0])
    if len(cols) == 1:
        return out

    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    coords.sort(key=lambda rc: (5 * rc[0] + 7 * rc[1], rc[0], rc[1]))
    take = max(1, min(len(coords) - 1, (len(coords) + 2) // 4))
    for r, c in coords[:take]:
        out[int(r), int(c)] = int(cols[1])
    return out


def _place_destinations(dh: int, dw: int, pieces: list[dict], *, vertical: bool) -> list[tuple[int, int]] | None:
    sizes = [int(p["dest_mask"].shape[0] if vertical else p["dest_mask"].shape[1]) for p in pieces]
    min_need = 2 + sum(sizes) + max(0, len(pieces) - 1)
    long = int(dh if vertical else dw)
    if min_need > long:
        return None

    gaps = [1] * (len(pieces) + 1)
    for _ in range(long - min_need):
        gaps[random.randrange(len(gaps))] += 1

    pos: list[tuple[int, int]] = []
    cursor = gaps[0]
    for i, piece in enumerate(pieces):
        h, w = map(int, piece["dest_mask"].shape)
        if vertical:
            if dw - w - 1 < 1:
                return None
            rr = int(cursor)
            cc = random.randint(1, dw - w - 1)
            pos.append((rr, cc))
            cursor += h + gaps[i + 1]
        else:
            if dh - h - 1 < 1:
                return None
            rr = random.randint(1, dh - h - 1)
            cc = int(cursor)
            pos.append((rr, cc))
            cursor += w + gaps[i + 1]
    return pos


def _zone_coverage_ok(pieces: list[dict], placements: list[tuple[int, int]], dh: int, dw: int) -> bool:
    seen: set[int] = set()
    for piece, (rr, cc) in zip(pieces, placements):
        h, w = map(int, piece["dest_mask"].shape)
        seen.add(_zone_index(rr, cc, h, w, dh, dw))
    return seen == {0, 1, 2}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill irregular strip cavities by rotating each outside sprite according to the hole it matches.

    Input structure:
    - The dominant color is the background. One large non-background strip forms the template bbox.
    - Inside that strip, the empty regions are separate irregular cavities rather than solid rectangles.
    - Matching colored sprites are scattered outside the strip, each isolated as its own component.

    Rule:
    - The largest non-background connected component color identifies the strip and its tight crop.
    - Each outside sprite matches exactly one cavity under rotation, not reflection.
    - The only legal rotation is determined by the cavity's position along the strip: top/left third
      uses 90 degrees CCW, middle third uses 180 degrees, bottom/right third uses 270 degrees.
    - Rotate the sprite by that position-based amount so its mask matches the cavity, then paste it.

    Key insight:
    - Solving requires both full shape comparison and a location-derived rotation rule: bbox size
      alone is insufficient, and even a shape match is only valid when the matching rotation agrees
      with the hole's zone inside the strip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    holes: list[dict] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        h, w = int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "mask": mask,
                "area": int(mask.sum()),
                "rot": _rotation_for_hole(int(hr0), int(hc0), h, w, dh, dw),
            }
        )
    holes.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "area": int(mask.sum()),
                "mask": mask,
                "variants": _rotation_variants(patch, mask),
            }
        )

    sprites.sort(key=lambda s: (-int(s["area"]), -int(s["mask"].shape[0] * s["mask"].shape[1])))
    assert len(sprites) == len(holes)
    assert sum(int(s["area"]) for s in sprites) == int(hole_mask.sum())

    candidates: list[list[tuple[int, dict]]] = []
    for sp in sprites:
        opts: list[tuple[int, dict]] = []
        for hi, hole in enumerate(holes):
            if int(hole["area"]) != int(sp["area"]):
                continue
            need_rot = int(hole["rot"])
            for var in sp["variants"]:
                if int(var["k"]) != need_rot:
                    continue
                if var["mask"].shape == hole["mask"].shape and np.array_equal(var["mask"], hole["mask"]):
                    opts.append((int(hi), var))
                    break
        candidates.append(opts)

    assert all(candidates)
    order = sorted(range(len(sprites)), key=lambda i: (len(candidates[i]), -int(sprites[i]["area"])))
    used_holes = [False] * len(holes)
    placement: list[tuple[int, dict] | None] = [None] * len(sprites)

    def dfs(t: int) -> bool:
        if t == len(order):
            return True
        si = int(order[t])
        for hi, var in candidates[si]:
            if used_holes[hi]:
                continue
            used_holes[hi] = True
            placement[si] = (int(hi), var)
            if dfs(t + 1):
                return True
            placement[si] = None
            used_holes[hi] = False
        return False

    assert dfs(0)

    out = np.full((dh, dw), int(base_col), dtype=int)
    for si, assigned in enumerate(placement):
        assert assigned is not None
        hi, var = assigned
        hr0, hc0, _hr1, _hc1 = holes[int(hi)]["bbox"]
        paste(out, var["patch"], int(hr0), int(hc0), mask=var["mask"], overwrite=True)

    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0

    long = int(dh if vertical else dw)
    count_hi = min(SPRITE_COUNT_MAX, max(SPRITE_COUNT_MIN, (long - 2) // 3))
    if count_hi < SPRITE_COUNT_MIN:
        return None
    count = random.randint(SPRITE_COUNT_MIN, count_hi)

    bg, base = random.sample(range(1, 10), 2)
    color_sets = _choose_colors(bg, base, count)
    if color_sets is None:
        return None

    pieces: list[dict] = []
    used_classes: set[bytes] = set()
    for cols in color_sets:
        mask = None
        for _ in range(50):
            cand = _random_connected_mask()
            if cand is None:
                continue
            sig = _shape_class_signature(cand)
            if sig in used_classes:
                continue
            mask = cand
            used_classes.add(sig)
            break
        if mask is None:
            return None
        patch = _make_piece_patch(mask, cols, bg)
        variants = _rotation_variants(patch, mask)
        pieces.append(
            {
                "dest_patch": patch,
                "dest_mask": mask,
                "variants": {int(v["k"]): v for v in variants},
                "area": int(mask.sum()),
            }
        )

    random.shuffle(pieces)
    placements = _place_destinations(dh, dw, pieces, vertical=vertical)
    if placements is None:
        return None
    if not _zone_coverage_ok(pieces, placements, dh, dw):
        return None

    total_hole_area = sum(int(p["area"]) for p in pieces)
    frac = total_hole_area / float(dh * dw)
    if not (HOLE_FRAC_LO <= frac <= HOLE_FRAC_HI):
        return None

    out = np.full((dh, dw), int(base), dtype=int)
    for piece, (rr, cc) in zip(pieces, placements):
        paste(out, piece["dest_patch"], int(rr), int(cc), mask=piece["dest_mask"], overwrite=True)

    inp = np.full((H, W), int(bg), dtype=int)
    inp[int(tr0) : int(tr0 + dh), int(tc0) : int(tc0 + dw)] = int(base)
    for piece, (rr, cc) in zip(pieces, placements):
        hole = piece["dest_mask"]
        sub = inp[int(tr0 + rr) : int(tr0 + rr + hole.shape[0]), int(tc0 + cc) : int(tc0 + cc + hole.shape[1])]
        sub[hole] = int(bg)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[int(tr0) : int(tr0 + dh), int(tc0) : int(tc0 + dw)] = int(base)
    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(
        key=lambda i: (
            -int(pieces[i]["area"]),
            -int(pieces[i]["dest_mask"].shape[0] * pieces[i]["dest_mask"].shape[1]),
        )
    )
    for i in order:
        piece = pieces[i]
        rr, cc = placements[i]
        h, w = map(int, piece["dest_mask"].shape)
        rot = _rotation_for_hole(int(rr), int(cc), h, w, dh, dw)
        src = piece["variants"][(4 - int(rot)) % 4]
        assert int(src["k"]) in (1, 2, 3)
        patch = src["patch"]
        sr, sc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, int(sr), int(sc), transparent=bg)
        paste(placement, patch, int(sr), int(sc), transparent=bg)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_instance()
    if built is None:
        return None
    inp, out = built
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")

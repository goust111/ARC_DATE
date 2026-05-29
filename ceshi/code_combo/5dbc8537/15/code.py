# Combo variant of 5dbc8537 (index 15)
# combo_spec: D1/1+D2/0
# Name: 不规则精灵旋转翻转匹配

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
#    - The original task used filled rectangles both for the outside sprites and for the holes in
#      the base-colored strip. Matching was therefore "same bbox, same slot".
#    - In this combo, both sides become irregular connected polyominoes. A source sprite may be an
#      L / hook / jagged shape with background pixels inside its bbox, and the corresponding cavity
#      inside the strip has the same polyomino mask but often in a different orientation.
#    - Small trace:
#      original: a 3x2 solid block outside the strip fills a 3x2 empty rectangle inside it.
#      combo: a 3x3 hook with five occupied cells may appear outside as one orientation, while the
#      destination cavity is the rotated or mirrored hook. The output keeps the strip crop and pastes
#      the recolored hook only on the cavity mask cells after the needed dihedral transform.
# b) Complexity calibration:
#    - Using separate cavities rather than one merged hole keeps the rule legible in a few examples.
#    - Every generated shape is unique up to the full dihedral group, so shape alone determines the
#      destination cavity and the solver is not forced into arbitrary tie-breaking.
#    - Each valid generated case contains at least one rotation-required match and one flip-required
#      match; most remaining pieces are also transformed, so the new behavior is visually salient.
# c) Broken original assumptions:
#    - `hole_mask[r:r+h, c:c+w].all()` assumed holes were solid rectangles. This is false because a
#      cavity mask now contains background holes inside its bbox.
#    - The original exact-cover search marked whole rectangles as occupied. That would incorrectly
#      consume base-colored cells that belong to the frame.
#    - Pasting the raw source patch is no longer valid because the correct cavity may require a
#      different orientation of the same irregular sprite.
#    - The original "sum of sprite bbox areas == hole area" invariant is replaced by "sum of sprite
#      foreground areas == total cavity area".
# d) Placement legality:
#    - Cavities are placed with a one-cell base border inside the strip and a one-cell gap along the
#      strip direction, so they cannot overlap, touch the strip boundary, or merge into one component.
#    - Outside sprites are scattered with `random_free_location(..., padding=1)` so each source remains
#      a distinct component and stays separated from the template.
# e) Visual observability:
#    - Shapes are asymmetric and chiral (`len(dihedral_variants) == 8`), so rotation and reflection
#      produce genuinely different silhouettes instead of visually identical states.
#    - Most pieces are bichromatic with asymmetric internal color layouts, making orientation changes
#      visible not only by outline but also by color arrangement.
# f) Visual cleanliness:
#    - The strip structure from the original task is preserved, while cavities are arranged as a tidy
#      sequence rather than a dense jumble. This keeps the input readable and the output crop clean.
# g) Inferability of changes:
#    - There is no hidden color remapping. Colors move rigidly with the transformed sprite.
#    - The solver therefore depends only on observable geometry: identify the strip, extract each
#      irregular cavity, compare it to all rotated/flipped variants of each outside sprite, and paste
#      the uniquely matching orientation.


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    if flip:
        out = np.fliplr(out)
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _variant_records(patch: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Return unique dihedral variants keyed by mask only."""
    uniq: list[dict] = []
    seen_masks: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        for flip in (False, True):
            vm = _transform(mask, k, flip)
            key = _mask_key(vm)
            if key in seen_masks:
                continue
            seen_masks.add(key)
            uniq.append(
                {
                    "patch": _transform(patch, k, flip),
                    "mask": vm,
                    "k": int(k),
                    "flip": bool(flip),
                }
            )
    return uniq


def _shape_class_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_transform(mask, k, flip)) for k in range(4) for flip in (False, True))


def _relation_kind(src_mask: np.ndarray, dst_mask: np.ndarray) -> str:
    if src_mask.shape == dst_mask.shape and np.array_equal(src_mask, dst_mask):
        return "identity"
    for k in (1, 2, 3):
        rm = np.rot90(src_mask, k)
        if rm.shape == dst_mask.shape and np.array_equal(rm, dst_mask):
            return "rotation"
    for k in range(4):
        fm = np.fliplr(np.rot90(src_mask, k))
        if fm.shape == dst_mask.shape and np.array_equal(fm, dst_mask):
            return "flip"
    raise AssertionError("source and destination masks must be dihedrally related")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill irregular cavities in a strip template by reorienting scattered irregular sprites.

    Input structure:
    - The background is the most frequent color, while one large non-background color forms a strip-
      shaped template whose tight bbox spans the full grid height or width.
    - Inside that strip, the missing regions are separate irregular cavities such as hooks or jagged
      polyominoes rather than solid rectangles.
    - Matching sprites sit outside the strip as isolated irregular components, and their observed
      orientation may differ from the orientation of the cavity they belong to.

    Rule:
    - Detect the template by taking the color whose largest connected component dominates the grid.
    - Split the strip's empty region into cavity components, and split the outer foreground into sprite
      components.
    - For each sprite, compare all rotated and mirrored variants against the cavity masks; paste the
      uniquely matching orientation into the corresponding cavity and return the tight strip crop.

    Key insight:
    - Size alone no longer determines the match. The solver must compare full cavity geometry against
      the dihedral variants of each irregular sprite and use the matching transformed patch when filling
      the template.
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
    assert base_coords.size != 0
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    hole_comps = components_mask(hole_mask, connectivity=4)
    holes: list[dict] = []
    for coords in hole_comps:
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        h, w = int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    holes.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))

    out = np.full((dh, dw), int(base_col), dtype=int)

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
                "patch": patch,
                "mask": mask,
                "area": int(mask.sum()),
                "variants": _variant_records(patch, mask),
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
            match_variant = None
            for var in sp["variants"]:
                if var["mask"].shape == hole["mask"].shape and np.array_equal(var["mask"], hole["mask"]):
                    match_variant = var
                    break
            if match_variant is not None:
                opts.append((int(hi), match_variant))
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

    for si, assigned in enumerate(placement):
        assert assigned is not None
        hi, var = assigned
        hr0, hc0, _hr1, _hc1 = holes[int(hi)]["bbox"]
        paste(out, var["patch"], int(hr0), int(hc0), mask=var["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 8, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 4, 7
PIECE_DIM_MIN, PIECE_DIM_MAX = 2, 4
PIECE_AREA_MIN = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.16, 0.34
MIN_PALETTE_SIZE = 5
MAX_RETRIES = 500
PIECE_RETRIES = 240


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
        cells = [(int(sr), int(sc))]

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
            cells.append((rr, cc))

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
        if len(_variant_records(mask.astype(int), mask)) != 8:
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
        if random.random() < 0.75:
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
    secondary = int(cols[1])
    choice = random.choice(["anchor", "diag", "half"])
    chosen: list[tuple[int, int]] = []
    if choice == "anchor":
        ordered = sorted(coords, key=lambda rc: (7 * rc[0] + 11 * rc[1], rc[0], rc[1]))
        take = random.randint(1, max(1, len(ordered) // 3))
        chosen = ordered[:take]
    elif choice == "diag":
        phase = random.randint(0, 2)
        chosen = [(r, c) for r, c in coords if (2 * r + c + phase) % 3 == 0]
    else:
        if random.random() < 0.5:
            pivot = random.randint(1, h - 1)
            chosen = [(r, c) for r, c in coords if r >= pivot]
        else:
            pivot = random.randint(1, w - 1)
            chosen = [(r, c) for r, c in coords if c >= pivot]
    if not chosen or len(chosen) == len(coords):
        chosen = [sorted(coords, key=lambda rc: (rc[0], rc[1]))[0]]
    for r, c in chosen:
        out[int(r), int(c)] = secondary
    return out


def _pick_src_dst_variants(variants: list[dict], desired: str | None = None) -> tuple[dict, dict, str] | None:
    pairs: list[tuple[dict, dict, str]] = []
    for src in variants:
        for dst in variants:
            relation = _relation_kind(src["mask"], dst["mask"])
            if desired is not None and relation != desired:
                continue
            pairs.append((src, dst, relation))
    if not pairs:
        return None
    return random.choice(pairs)


def _place_destinations(
    dh: int,
    dw: int,
    pieces: list[dict],
    *,
    vertical: bool,
) -> list[tuple[int, int]] | None:
    long = int(dh if vertical else dw)
    sizes = [int(p["dest_mask"].shape[0] if vertical else p["dest_mask"].shape[1]) for p in pieces]
    min_need = 2 + sum(sizes) + max(0, len(pieces) - 1)
    if min_need > long:
        return None
    extra = long - min_need
    gaps = [1] * (len(pieces) + 1)
    for _ in range(extra):
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


def _build_instance(*, return_meta: bool = False) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray] | None:
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
    bg, base = random.sample(range(1, 10), 2)
    count_hi = min(SPRITE_COUNT_MAX, max(SPRITE_COUNT_MIN, (long - 2) // 3))
    if count_hi < SPRITE_COUNT_MIN:
        return None
    count = random.randint(SPRITE_COUNT_MIN, count_hi)
    color_sets = _choose_colors(bg, base, count)
    if color_sets is None:
        return None

    desired_relations = ["rotation", "flip"]
    while len(desired_relations) < count:
        desired_relations.append(random.choices(["rotation", "flip"], weights=[3, 2])[0])
    random.shuffle(desired_relations)

    pieces: list[dict] = []
    used_shape_classes: set[tuple[tuple[int, int], bytes]] = set()
    for cols, desired in zip(color_sets, desired_relations):
        mask = None
        for _ in range(40):
            cand = _random_connected_mask()
            if cand is None:
                continue
            sig = _shape_class_signature(cand)
            if sig in used_shape_classes:
                continue
            mask = cand
            used_shape_classes.add(sig)
            break
        if mask is None:
            return None

        canon_patch = _make_piece_patch(mask, cols, bg)
        variants = _variant_records(canon_patch, mask)
        pick = _pick_src_dst_variants(variants, desired=desired)
        if pick is None:
            return None
        src, dst, relation = pick
        pieces.append(
            {
                "source_patch": src["patch"],
                "source_mask": src["mask"],
                "dest_patch": dst["patch"],
                "dest_mask": dst["mask"],
                "relation": relation,
                "area": int(mask.sum()),
                "shape_signature": sig,
            }
        )

    random.shuffle(pieces)
    placements = _place_destinations(dh, dw, pieces, vertical=vertical)
    if placements is None:
        return None

    total_hole_area = sum(int(piece["area"]) for piece in pieces)
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
            -int(pieces[i]["source_mask"].sum()),
            -int(pieces[i]["source_patch"].shape[0] * pieces[i]["source_patch"].shape[1]),
        )
    )
    for i in order:
        patch = pieces[i]["source_patch"]
        try:
            r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        paste(inp, patch, int(r), int(c), transparent=bg)
        paste(placement, patch, int(r), int(c), transparent=bg)

    meta = {
        "vertical": bool(vertical),
        "piece_count": int(len(pieces)),
        "hole_fraction": float(frac),
        "relations": [str(piece["relation"]) for piece in pieces],
        "rotation_count": sum(piece["relation"] == "rotation" for piece in pieces),
        "flip_count": sum(piece["relation"] == "flip" for piece in pieces),
        "identity_count": sum(piece["relation"] == "identity" for piece in pieces),
        "non_identity_count": sum(piece["relation"] != "identity" for piece in pieces),
    }
    if return_meta:
        return inp, out, meta
    return inp, out


def generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(MAX_RETRIES):
        built = _build_instance(return_meta=True)
        if built is None:
            continue
        inp, out, meta = built
        if meta["rotation_count"] == 0 or meta["flip_count"] == 0 or meta["identity_count"] != 0:
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp, out, meta
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp

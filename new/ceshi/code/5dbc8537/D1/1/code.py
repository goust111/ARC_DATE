# Variant of 5dbc8537
# Dimension: D1 — 匹配条件：精灵与洞的对应方式 (match_cond)
# Change: s.bbox == d.rect — 精灵外接矩形尺寸与洞的矩形尺寸完全相等（shape_match）... → 旋转/翻转后形状匹配

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

# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior:
#    - The large base-colored strip is preserved from the original task.
#    - Inside the strip, holes are no longer axis-aligned rectangles. Each hole is a separate
#      irregular 4-connected cavity, usually an L / hook / jagged polyomino.
#    - Outside the strip, each sprite is the same polyomino, but its observed orientation may
#      differ from the hole's orientation. The output pastes the sprite into the hole after the
#      necessary rotation / reflection.
#    - Small process trace:
#      original: a 3x2 filled rectangle sprite matches a 3x2 rectangular gap purely by bbox size.
#      variant: a hook-shaped sprite with bbox 3x2 may need to become a reflected 2x3 hook before
#      it matches the cavity mask; bbox equality alone is no longer enough.
# b) Complexity calibration:
#    - Holes are separated components so the new transform rule is visually readable in 2-4 ARC
#      examples; otherwise a single merged cavity would hide the piece boundaries and make the task
#      too opaque.
#    - To avoid ambiguity, every generated shape is unique up to the full dihedral group, so shape
#      alone determines the destination hole.
#    - Each instance forces both a non-trivial rotation match and a reflection-required match when
#      possible, so the new rule is not a rare corner case.
# c) Old assumptions that break:
#    - Original candidate search relied on `hole_mask[r:r+h, c:c+w].all()`: this assumes holes are
#      solid rectangles. That is false now because cavities are sparse masks inside their bboxes.
#    - Original exact-cover occupancy over rectangles assumes sprite bbox == hole bbox. That would
#      place pixels into base-colored cells for irregular cavities.
#    - Original paste used the source patch without reorientation. That is now wrong whenever the
#      sprite must be rotated / flipped to fit the cavity.
# d) Placement legality:
#    - Generator places each cavity with a 1-cell base border inside the strip and a 1-cell base
#      gap along the strip direction, so cavities never overlap, never touch the strip boundary,
#      and never merge into one component.
#    - Outside sprites are scattered with `random_free_location(..., padding=1)` so they remain
#      isolated from each other and from the template.
# e) Visual observability:
#    - Shapes are asymmetric and usually bichromatic, so orientation changes are visible.
#    - The generator enforces at least one rotation-required piece and one flip-required piece in
#      successful instances, which makes the D1 change readable from examples.
# f) Visual cleanliness:
#    - Cavities are arranged as a clean sequence inside the strip instead of overlapping or densely
#      interleaving, keeping the template legible.
# g) Color inferability:
#    - There is no color remapping rule. Colors move with the sprite under the chosen transform,
#      which is directly observable from the outside sprite and the filled output.


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    if flip:
        out = np.fliplr(out)
    return out


def _variant_records(patch: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Unique dihedral variants keyed by mask only; generator makes this unambiguous."""
    uniq: list[dict] = []
    seen_masks: set[bytes] = set()
    for k in range(4):
        for flip in (False, True):
            pm = _transform(mask, k, flip)
            key = pm.tobytes()
            if key in seen_masks:
                continue
            seen_masks.add(key)
            uniq.append(
                {
                    "patch": _transform(patch, k, flip),
                    "mask": pm,
                    "k": int(k),
                    "flip": bool(flip),
                }
            )
    return uniq


def _shape_class_signature(mask: np.ndarray) -> bytes:
    return min(_transform(mask, k, flip).tobytes() for k in range(4) for flip in (False, True))


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
    Fill irregular cavities in a strip template by reorienting matching outside sprites.

    Input structure:
    - bg is the dominant color. A single large base-colored strip forms the template region.
    - Inside that strip, the empty cavities are separate irregular shapes rather than solid rectangles.
    - Matching colored sprites are scattered outside the strip, each as its own connected component.

    Rule:
    - The largest non-bg connected component color identifies the template strip and its tight bbox.
    - Each outside sprite matches exactly one cavity up to rotation and/or mirror reflection.
    - Reorient each sprite to the unique cavity with the same shape, then paste it into the template.

    Key insight:
    - Bounding-box size is no longer sufficient; the solver must compare full cavity masks against
      all dihedral variants of each sprite and use the matching orientation when filling the strip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- pick base color by largest CC size (excluding bg) ---
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

    # --- destination bbox = bbox of all base-color pixels ---
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
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
        m = np.zeros((h, w), dtype=bool)
        m[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        holes.append(
            {
                "bbox": (int(hr0), int(hc0), int(hr1), int(hc1)),
                "mask": m,
                "area": int(m.sum()),
            }
        )
    holes.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))

    out = np.full((dh, dw), int(base_col), dtype=int)

    # --- source sprites = connected components of non-bg pixels outside dst_bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        variants = _variant_records(patch, m)
        sprites.append({"patch": patch, "mask": m, "area": int(m.sum()), "variants": variants})

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

    # --- paste each sprite using the orientation that matches its cavity ---
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
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.16, 0.32
MIN_PALETTE_SIZE = 5
MAX_RETRIES = 400
PIECE_RETRIES = 200


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


def _transform_options(mask: np.ndarray, patch: np.ndarray) -> list[dict]:
    return _variant_records(patch, mask)


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
    dests: list[dict],
    *,
    vertical: bool,
) -> list[tuple[int, int]] | None:
    long = int(dh if vertical else dw)
    cross = int(dw if vertical else dh)
    sizes = [int(d["dest_mask"].shape[0] if vertical else d["dest_mask"].shape[1]) for d in dests]
    min_need = 2 + sum(sizes) + max(0, len(dests) - 1)
    if min_need > long:
        return None
    extra = long - min_need
    gaps = [1] * (len(dests) + 1)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    pos: list[tuple[int, int]] = []
    cursor = gaps[0]
    for i, d in enumerate(dests):
        h, w = map(int, d["dest_mask"].shape)
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
    bg, base = random.sample(range(1, 10), 2)
    count_hi = min(SPRITE_COUNT_MAX, max(SPRITE_COUNT_MIN, (long - 2) // 3))
    if count_hi < SPRITE_COUNT_MIN:
        return None
    count = random.randint(SPRITE_COUNT_MIN, count_hi)
    color_sets = _choose_colors(bg, base, count)
    if color_sets is None:
        return None

    pieces: list[dict] = []
    used_shape_classes: set[bytes] = set()
    desired_relations = ["rotation", "flip"] + [None] * max(0, count - 2)
    random.shuffle(desired_relations)
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
        variants = _transform_options(mask, canon_patch)
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
            }
        )

    random.shuffle(pieces)
    placements = _place_destinations(dh, dw, pieces, vertical=vertical)
    if placements is None:
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
            -int(pieces[i]["source_mask"].sum()),
            -int(pieces[i]["source_patch"].shape[0] * pieces[i]["source_patch"].shape[1]),
        )
    )
    for i in order:
        patch = pieces[i]["source_patch"]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, int(r), int(c), transparent=bg)
        paste(placement, patch, int(r), int(c), transparent=bg)

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
    raise RuntimeError("failed to generate valid 5dbc8537 D1 variant input")

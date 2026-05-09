# Combo variant of 58f5dbd5 (index 30)
# combo_spec: D3/1+D12/2
# Name: 可旋转motif+固定几何形状

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste

# Design analysis for combo D3/1 + D12/2
# a) Concrete behavior:
#    Original task: for each 5x5 solid color block, find the same-colored small motif elsewhere and
#    carve that exact visible mask into the centered 3x3 window of the block. Any connected 3x3-ish
#    shape worked, so the solver could treat the cropped mask literally.
#    Fused task: the visible motifs now come only from a tiny library of exact 3x3 geometric families
#    (cross, L, T, U). The displayed motif may be rotated or flipped, but the carved hole must use the
#    family's fixed canonical orientation. Example: suppose the canonical T is
#      111 / 010 / 010
#    and the input shows its right-facing rotation
#      001 / 111 / 001
#    above the same-colored 5x5 block. The old solver would carve the right-facing T. The fused solver
#    must first recognize "this is a T-family motif", normalize it back to 111/010/010, then carve that.
#    The fusion matters because D12 narrows the motif space to a recognizable typed vocabulary, while D3
#    says the input pose is not authoritative; shape family gives the identity, canonical orientation
#    gives the output stencil.
# b) Complexity calibration:
#    A free-form canonicalization rule would be too unconstrained, while a single family would collapse
#    back into a trivial lookup. The generator therefore uses only four fixed 3x3 families and requires
#    at least two distinct families per sample, with at least two matching motifs rendered in a
#    non-canonical pose whenever possible. This keeps the change visually obvious without making the
#    hypothesis space too large for 2-4 ARC examples.
# c) Implicit-assumption audit of the original main():
#    - Original Step 2 assumed the smallest same-colored component could be used directly as the carving
#      mask. This fails once the visible pose is only a transformed representative of a motif family.
#    - Original Step 3 assumed centering the recovered bbox was enough. In the fused task, centering is
#      still correct, but the recovered mask must first be mapped to a fixed family orientation.
#    - Original color matching still holds, so block detection, same-color pairing, row/column order,
#      and output packing remain valid.
#    - Because D12 restricts motifs to exact full-3x3 templates, the solver now implicitly assumes every
#      actionable motif belongs to the library up to a dihedral transform. The generator enforces that.
# d) Placement legality:
#    The source layout logic is preserved: 5x5 blocks live in one region, 3x3 motif slots in another,
#    and the two regions are separated on at least one axis. All motif families touch every row and
#    column of their 3x3 bbox, so rotating or flipping them never changes slot occupancy or causes crop
#    shrinkage, overlap, or out-of-bounds placement.
# e) Visual observability:
#    Humans can see both parts of the fused rule. D12 is visible because motifs are no longer arbitrary:
#    they repeat as a small family vocabulary of clear shapes. D3 is visible because matching motifs are
#    usually shown in non-canonical orientations, so copying the visible pose into the block would give a
#    visibly different output. Repetition across colors supports the inference that the family, not the
#    displayed pose, determines the carved stencil.
# f) Visual cleanliness:
#    No extra markers are overlaid on blocks. The only scene elements remain the separated block region
#    and motif region, so the puzzle stays uncluttered. The motif library is small and high-contrast, and
#    1-cell gaps between slots preserve legibility after rotation/flip.
# g) Change inferability:
#    There is no hidden color arithmetic or ordering by numeric color id. Colors only pair motifs to
#    blocks, while all new behavior is carried by visible geometry: family class plus orientation.
#    The generator also avoids degenerate samples by rejecting cases where canonical carving would match
#    the legacy "use visible mask directly" output.
# i) New solver assumptions and generator guarantees:
#    - Solver assumption: every matching motif crops to a 3x3 mask belonging to the fixed library under
#      rotation/flip. Generator guarantee: all matching and distractor motifs are drawn from that library.
#    - Solver assumption: each block color has one non-block component that is its clue motif. Generator
#      guarantee: distractor colors never reuse block colors.
#    - Solver assumption: canonical families are distinguishable. Generator guarantee: the family library
#      has non-overlapping dihedral orbits, and each sample uses at least two different families.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask with True at the given coords (in-bounds only)."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _touches_all_edges(mask: np.ndarray) -> bool:
    """Return whether a 3x3 motif uses every outer row/column at least once."""
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _variant_entries(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Unique dihedral variants as (variant_mask, is_identity)."""
    out: list[tuple[np.ndarray, bool]] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for flip in (False, True):
            cand = np.fliplr(rot) if flip else rot
            key = cand.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((cand.copy(), bool(k == 0 and not flip)))
    return out


FAMILY_NAMES = ("cross", "L", "T", "U")
STANDARD_MOTIFS: list[np.ndarray] = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
]

STANDARD_MOTIF_BY_VARIANT: dict[bytes, np.ndarray] = {}
FAMILY_INDEX_BY_VARIANT: dict[bytes, int] = {}
MOTIF_VARIANTS_BY_INDEX: list[list[np.ndarray]] = []
NON_IDENTITY_VARIANTS_BY_INDEX: list[list[np.ndarray]] = []
ASYMMETRIC_FAMILY_IDS: list[int] = []

for family_idx, motif in enumerate(STANDARD_MOTIFS):
    assert motif.shape == (3, 3)
    assert _touches_all_edges(motif)
    variants = _variant_entries(motif)
    MOTIF_VARIANTS_BY_INDEX.append([variant.copy() for variant, _is_identity in variants])
    non_identity = [variant.copy() for variant, is_identity in variants if not is_identity]
    NON_IDENTITY_VARIANTS_BY_INDEX.append(non_identity)
    if non_identity:
        ASYMMETRIC_FAMILY_IDS.append(int(family_idx))
    for variant, _is_identity in variants:
        key = variant.tobytes()
        assert key not in STANDARD_MOTIF_BY_VARIANT
        STANDARD_MOTIF_BY_VARIANT[key] = motif.copy()
        FAMILY_INDEX_BY_VARIANT[key] = int(family_idx)


def _canonicalize_motif(mask: np.ndarray) -> np.ndarray:
    """Map a displayed 3x3 motif to its family's canonical orientation."""
    key = np.asarray(mask, dtype=bool).tobytes()
    if key not in STANDARD_MOTIF_BY_VARIANT:
        raise ValueError("motif is not a supported rotated/flipped geometric family")
    return STANDARD_MOTIF_BY_VARIANT[key].copy()


def _family_index(mask: np.ndarray) -> int:
    """Return the family id of a 3x3 motif from the fixed library."""
    key = np.asarray(mask, dtype=bool).tobytes()
    if key not in FAMILY_INDEX_BY_VARIANT:
        raise ValueError("motif family is unknown")
    return int(FAMILY_INDEX_BY_VARIANT[key])


def _find_blocks(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Extract solid 5x5 monochrome destination blocks."""
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": int(r0), "c0": int(c0)})
    return blocks


def _find_visible_motifs(
    g: np.ndarray, bg: int, block_colors: set[int]
) -> dict[int, np.ndarray]:
    """For each block color, recover the smallest same-colored non-block component."""
    motifs: dict[int, np.ndarray] = {}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(bcol)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == int(bcol)).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if not candidates:
            raise ValueError(f"missing motif for block color {bcol}")
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs[int(bcol)] = candidates[0][4]
    return motifs


def _diagnose_input(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a generated sample really exercises the fused behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    blocks = _find_blocks(g, bg)
    block_colors = {int(b["col"]) for b in blocks}
    visible = _find_visible_motifs(g, bg, block_colors)
    canonical = {color: _canonicalize_motif(mask) for color, mask in visible.items()}

    active_transforms = 0
    canonical_difference_pixels = 0
    families: set[int] = set()
    for color in block_colors:
        vmask = visible[int(color)]
        cmask = canonical[int(color)]
        families.add(_family_index(cmask))
        if not np.array_equal(vmask, cmask):
            active_transforms += 1
            canonical_difference_pixels += int(np.count_nonzero(vmask != cmask))

    return {
        "block_count": int(len(blocks)),
        "shape_diversity": int(len(families)),
        "active_transforms": int(active_transforms),
        "canonical_difference_pixels": int(canonical_difference_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recognize each rotated/flipped geometric motif family, carve its canonical stencil into the matching blocks, and pack them.

    Input structure:
    - The background is the most frequent color.
    - One region contains several solid 5x5 monochrome blocks, each with a unique non-background color.
    - A separate region contains small same-colored 3x3 motifs drawn from a fixed library of geometric shapes such as cross, L, T, and U, and those motifs may be rotated or mirrored.

    Rule:
    - Match every motif to the 5x5 block of the same color.
    - Identify which geometric family the visible motif belongs to, convert it back to that family's standard orientation, and carve that canonical 3x3 stencil into the centered area of the block by replacing motif cells with background.
    - Pack the carved 5x5 blocks into an output grid that preserves their original row/column order, with 1-cell gaps and a 1-cell background border.

    Key insight:
    - The source motif is a transformed clue from a small typed shape vocabulary; the output depends on the family's canonical pose, not on the pose drawn in the input.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks = _find_blocks(g, bg)
    assert blocks

    # STEP 2) For each block color, recover the displayed motif and normalize it to its family standard.
    block_colors = {int(b["col"]) for b in blocks}
    visible_motifs = _find_visible_motifs(g, bg, block_colors)
    canonical_motifs = {
        int(color): _canonicalize_motif(mask)
        for color, mask in visible_motifs.items()
    }

    # STEP 3) Carve each 5x5 tile using the normalized 3x3 motif.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = canonical_motifs[int(col)]
        for rr, cc in np.argwhere(mm):
            tile[1 + int(rr), 1 + int(cc)] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x, combo D3/1 + D12/2) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

MIN_ACTIVE_TRANSFORMS = 2
MIN_SHAPE_DIVERSITY = 2
MIN_DIFFERENCE_PIXELS = 6
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    """Return whether destination and motif regions can be separated by an axis."""
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible orderings for separated destination and motif regions."""
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place vertically separated regions and return destination then motif origins."""
    if scheme == "dst_above":
        top_h, top_w, bot_h, bot_w = dh, dw, mh, mw
    else:
        top_h, top_w, bot_h, bot_w = mh, mw, dh, dw

    top_r = random.randint(OUTER_MARGIN, H - top_h - bot_h - OUTER_MARGIN - REGION_GAP)
    bot_r = random.randint(top_r + top_h + REGION_GAP, H - bot_h - OUTER_MARGIN)
    top_c = random.randint(OUTER_MARGIN, W - top_w - OUTER_MARGIN)
    bot_c = random.randint(OUTER_MARGIN, W - bot_w - OUTER_MARGIN)
    if scheme == "dst_above":
        return top_r, top_c, bot_r, bot_c
    return bot_r, bot_c, top_r, top_c


def _place_horizontal(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place horizontally separated regions and return destination then motif origins."""
    if scheme == "dst_left":
        left_h, left_w, right_h, right_w = dh, dw, mh, mw
    else:
        left_h, left_w, right_h, right_w = mh, mw, dh, dw

    left_c = random.randint(OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP)
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w - OUTER_MARGIN)
    left_r = random.randint(OUTER_MARGIN, H - left_h - OUTER_MARGIN)
    right_r = random.randint(OUTER_MARGIN, H - right_h - OUTER_MARGIN)
    if scheme == "dst_left":
        return left_r, left_c, right_r, right_c
    return right_r, right_c, left_r, left_c


def _place_regions(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> tuple[int, int, int, int] | None:
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None

    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int, int]] = []
    min_filled, max_filled = FILLED_SLOT_RANGE
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for filled in range(min_filled, max_filled + 1):
                distractors = filled - count
                if not (0 <= distractors <= count):
                    continue
                if distractors > COLOR_COUNT - 1 - count:
                    continue
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (count <= capacity <= 2 * count):
                        continue
                    if not (filled <= capacity and capacity - filled <= MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, filled))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving the original two-stage count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _choose_family_indices(count: int) -> list[int]:
    """Assign fixed geometric families while keeping visible diversity and repeated support."""
    family_pool = list(range(len(STANDARD_MOTIFS)))
    asym_pool = ASYMMETRIC_FAMILY_IDS.copy()
    random.shuffle(family_pool)
    random.shuffle(asym_pool)

    target_distinct = min(len(STANDARD_MOTIFS), count, 2 + (count >= 4) + (count >= 6))
    chosen: list[int] = []

    for idx in asym_pool:
        if len(chosen) >= min(MIN_ACTIVE_TRANSFORMS, count):
            break
        chosen.append(int(idx))

    for idx in family_pool:
        if len(set(chosen)) >= target_distinct:
            break
        if idx not in chosen:
            chosen.append(int(idx))

    while len(chosen) < count:
        chosen.append(int(random.choice(family_pool)))

    random.shuffle(chosen)
    return chosen


def _choose_canonical_masks(block_colors: list[int]) -> dict[int, np.ndarray]:
    """Assign one canonical geometric family mask to each block color."""
    family_indices = _choose_family_indices(len(block_colors))
    return {
        int(color): STANDARD_MOTIFS[int(family_idx)].copy()
        for color, family_idx in zip(block_colors, family_indices)
    }


def _visible_variant(mask: np.ndarray) -> np.ndarray:
    """Pick a displayed pose for a matching motif, preferring non-canonical transforms."""
    family_idx = _family_index(mask)
    variants = NON_IDENTITY_VARIANTS_BY_INDEX[int(family_idx)]
    if variants:
        return random.choice(variants).copy()
    return mask.copy()


def _random_display_mask() -> np.ndarray:
    """Pick any displayed motif from the fixed geometric family library."""
    family_idx = random.randrange(len(STANDARD_MOTIFS))
    return random.choice(MOTIF_VARIANTS_BY_INDEX[int(family_idx)]).copy()


def _choose_visible_masks(canonical_masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Choose the visible transformed pose for each matching motif."""
    return {
        int(color): _visible_variant(mask)
        for color, mask in canonical_masks.items()
    }


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], canonical_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[1:4, 1:4][canonical_masks[color]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int],
    distractor_colors: list[int],
    visible_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled visible motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), visible_masks[int(color)].copy()) for color in block_colors
    ]
    items.extend((int(color), _random_display_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    visible_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, visible_masks)

    for (sr, sc), (color, mask) in zip(slots, slot_items):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    visible_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and displayed motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, visible_masks
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it unless it cleanly exercises the fused rule."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    canonical_masks = _choose_canonical_masks(block_colors)
    visible_masks = _choose_visible_masks(canonical_masks)

    expected = _render_expected(bg, R, C, block_colors, canonical_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, visible_masks, placement
    )

    if not np.array_equal(main(inp), expected):
        return None

    metrics = _diagnose_input(inp)
    if metrics["shape_diversity"] < MIN_SHAPE_DIVERSITY:
        return None
    if metrics["active_transforms"] < MIN_ACTIVE_TRANSFORMS:
        return None
    if metrics["canonical_difference_pixels"] < MIN_DIFFERENCE_PIXELS:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

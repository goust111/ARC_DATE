# Variant of 58f5dbd5
# Dimension: D3 — 源对象：motif结构 (src)
# Change: 单色、8连通、3x3外接矩形内的小型motif精灵，面积4-8像素... → 可旋转/翻转的motif

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste

# Design analysis for the D3 variant
# a) Concrete behavior:
#    The input still contains solid 5x5 color blocks plus same-colored 3x3 motifs elsewhere.
#    The change is that each visible motif is no longer shown in its carving orientation. Instead,
#    it is a rotated/flipped copy of one of a few fixed asymmetric 3x3 prototypes. The output still
#    shows carved 5x5 blocks, but the holes use the prototype's standard orientation, not the visible
#    orientation in the input. Example trace: if the standard mask is
#      001 / 010 / 110
#    and the input shows its horizontal mirror
#      100 / 010 / 011,
#    the old solver would carve the mirrored hole, while the new rule must first map that mask back
#    to the standard prototype and then carve 001/010/110 at the centered 3x3 window of the block.
# b) Complexity calibration:
#    Free-form "canonical orientation" would be too unconstrained for a human to infer from 2-4
#    examples. To keep the variant observable but still nontrivial, the generator draws motifs from a
#    small fixed library of asymmetric prototypes whose dihedral orbits do not overlap. Repetition
#    across colors provides evidence that the same underlying shape can appear in different poses.
# c) Implicit-assumption audit of the original main():
#    - Step 2 originally assumed the cropped same-color motif mask was already the exact stencil.
#      That assumption fails here because the displayed pose is not authoritative.
#    - Step 3 originally carved with the raw recovered mask; now it must carve with a normalized mask.
#    - Color matching, 5x5 block detection, row/column packing, and gap/border arithmetic remain valid.
# d) Placement legality:
#    Regions, spacing, counts, and margins stay identical to the source task. Motifs still occupy 3x3
#    slots on a separate motif region, so rotated/flipped variants never exceed their slots or overlap.
# e) Visual observability:
#    Every canonical prototype is asymmetric, so rotation/flip visibly changes it. Matching motifs are
#    intentionally rendered with non-identity transforms, making the D3 change active in nearly every
#    sample instead of silently collapsing back to the original task.
# f) Visual cleanliness:
#    Nothing is overlaid inside the input; blocks and motif slots remain separated regions exactly as
#    before. The only extra reasoning burden is the pose normalization of the source motifs.
# g) Color-rule audit:
#    No hidden ordering or color arithmetic is introduced. Colors still only identify which motif
#    belongs to which 5x5 block; the new behavior is purely geometric.
# i) New solver assumptions and generator guarantees:
#    - Solver assumption: every matching motif belongs to the fixed prototype library up to a dihedral
#      transform. Generator guarantee: all matching and distractor motifs are sampled from that library.
#    - Solver assumption: matching motifs keep a full 3x3 bbox after cropping. Generator guarantee:
#      every prototype touches all four bbox edges, so every rotated/flipped copy still crops to 3x3.


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


def _variant_entries(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool, bool]]:
    """Unique dihedral variants as (variant, rot_k, flip, is_identity)."""
    out: list[tuple[np.ndarray, int, bool, bool]] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for flip in (False, True):
            cand = np.fliplr(rot) if flip else rot
            key = cand.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((cand.copy(), int(k), bool(flip), bool(k == 0 and not flip)))
    return out


STANDARD_MOTIFS: list[np.ndarray] = [
    np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

STANDARD_MOTIF_BY_VARIANT: dict[bytes, np.ndarray] = {}
VARIANT_METADATA: dict[bytes, tuple[int, int, bool, bool]] = {}
MOTIF_VARIANTS_BY_INDEX: list[list[tuple[np.ndarray, int, bool, bool]]] = []
for motif_idx, motif in enumerate(STANDARD_MOTIFS):
    variants = _variant_entries(motif)
    assert len(variants) == 8
    MOTIF_VARIANTS_BY_INDEX.append(variants)
    for variant, k, flip, is_identity in variants:
        key = variant.tobytes()
        assert key not in STANDARD_MOTIF_BY_VARIANT
        STANDARD_MOTIF_BY_VARIANT[key] = motif.copy()
        VARIANT_METADATA[key] = (int(motif_idx), int(k), bool(flip), bool(is_identity))


def _canonicalize_motif(mask: np.ndarray) -> np.ndarray:
    """Map a displayed 3x3 motif to its standard prototype orientation."""
    key = np.asarray(mask, dtype=bool).tobytes()
    if key not in STANDARD_MOTIF_BY_VARIANT:
        raise ValueError("motif is not a supported rotated/flipped prototype")
    return STANDARD_MOTIF_BY_VARIANT[key].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize rotated/flipped motifs, carve the canonical stencil into the matching blocks, and pack them.

    Input structure:
    - bg is the most common color.
    - Several solid 5x5 monochrome blocks appear in one region, each with its own non-bg color.
    - Elsewhere, each block color has a small same-colored 3x3 motif, but that motif may be shown after a rotation or a mirror flip.

    Rule:
    - Pair each motif with the solid block of the same color.
    - Convert the visible motif back to its standard orientation, center that 3x3 stencil inside the 5x5 block, and carve those cells to bg.
    - Output all carved blocks in their original row/column order, packed with 1-cell gaps and a 1-cell bg border.

    Key insight:
    - The displayed source motif is only a transformed clue; the hole pattern comes from the motif family’s canonical pose.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, recover the transformed motif and normalize it.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
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

        assert candidates
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs[int(bcol)] = _canonicalize_motif(candidates[0][4])

    # STEP 3) Carve each 5x5 tile using the normalized motif.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = motifs[int(col)]
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
        i = int(r_index[r0])
        j = int(c_index[c0])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


# --------------- generator (o5x, D3 variant) ---------------

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


def _choose_template_indices(count: int) -> list[int]:
    """Sample prototype ids with replacement while keeping some visible variety/repetition."""
    idxs = [random.randrange(len(STANDARD_MOTIFS)) for _ in range(count)]
    if count >= 2 and len(set(idxs)) == 1:
        idxs[-1] = (idxs[-1] + random.randint(1, len(STANDARD_MOTIFS) - 1)) % len(STANDARD_MOTIFS)
    if count >= 4 and len(set(idxs)) == count and count <= len(STANDARD_MOTIFS):
        idxs[-1] = idxs[0]
    return idxs


def _choose_canonical_masks(block_colors: list[int]) -> dict[int, np.ndarray]:
    """Assign one standard prototype to each block color."""
    idxs = _choose_template_indices(len(block_colors))
    return {
        int(color): STANDARD_MOTIFS[int(idx)].copy()
        for color, idx in zip(block_colors, idxs)
    }


def _non_identity_variant(mask: np.ndarray) -> np.ndarray:
    """Pick a visible non-identity dihedral variant of a standard motif."""
    motif_idx, _k, _flip, _id = VARIANT_METADATA[mask.tobytes()]
    variants = MOTIF_VARIANTS_BY_INDEX[int(motif_idx)]
    return random.choice(variants[1:])[0].copy()


def _random_display_mask() -> np.ndarray:
    """Pick any transformed motif from the fixed prototype library."""
    motif_idx = random.randrange(len(STANDARD_MOTIFS))
    return random.choice(MOTIF_VARIANTS_BY_INDEX[int(motif_idx)])[0].copy()


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
    canonical_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled visible motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), _non_identity_variant(canonical_masks[color])) for color in block_colors
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
    canonical_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, canonical_masks)

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
    canonical_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and transformed motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, canonical_masks
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
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

    expected = _render_expected(bg, R, C, block_colors, canonical_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, canonical_masks, placement
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D3 input")

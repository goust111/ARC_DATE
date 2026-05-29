# Combo variant of 58f5dbd5 (index 12)
# combo_spec: D7/3+D13/1+D14/0
# Name: 计数编码+更多方块+面积排序输出
#
# Design analysis:
# a) Concrete behavior and trace:
#    - Original task: each same-colored 3x3-ish motif is centered into its 5x5
#      block and carved literally by shape, then blocks are packed in their
#      input row/column order.
#    - Fused variant: the color match is preserved, but the motif contributes
#      only its filled-cell count. A motif of area 5 no longer punches its own
#      silhouette; instead, the block shows 5 background dots on a fixed inner
#      3x3 lattice. After all tiles are encoded this way, they are sorted by
#      motif area before being packed.
#    - Example trace: if a 3x3 block grid has spatial motif areas
#      [7, 4, 8, 5, 6, 4, 7, 8, 5], the original output would preserve that
#      slot order, while the fused output writes tiles in
#      [4, 4, 5, 5, 6, 7, 7, 8, 8]. Two different area-4 motif shapes still
#      become identical 4-dot tiles, so the transform visibly depends on count
#      rather than geometry.
# b) Complexity calibration:
#    - Block detection stays simple: solid 5x5 monochrome squares.
#    - The new count range is still 4..8, which fits comfortably inside a 9-site
#      tally lattice.
#    - Because 8-9 blocks exceed the five available area values, the generator
#      deliberately uses repeated areas. That repetition is useful: it gives the
#      solver enough evidence that equal-area but different-shape motifs collapse
#      to the same encoded tile.
#    - Sorting ties use visible input order as a stable fallback, not hidden
#      color values.
# c) Implicit assumption audit:
#    - The old code assumed "motif mask geometry matters" in the carving loop.
#      That assumption is removed; only component area survives into the output.
#    - The old packing step assumed spatial order determines slot order. That is
#      replaced by ascending area order, with input row-major position only used
#      to stabilize equal-area ties.
#    - The solver still assumes every block color has one non-block component of
#      the same color. The generator enforces that by keeping motif colors unique
#      and never adding same-color distractors.
#    - The solver infers the packed output rectangle from unique block rows and
#      columns, so the generator must use full rectangular layouts only:
#      2x4, 4x2, or 3x3.
# d) Placement legality:
#    - The destination region and motif region are still placed as disjoint
#      rectangles separated by at least one background cell, with one-cell outer
#      margins. Larger canvases are required so a 3x3 block region and a 3x3
#      motif-slot region can coexist without overlap.
# e) Visual observability:
#    - The generator rejects samples whose spatial area sequence is already
#      nondecreasing; otherwise the area-based reordering would be invisible.
#    - It also uses repeated areas with different sampled masks so a human can
#      see that motif shape disappears and only motif area remains.
# f) Visual cleanliness:
#    - Output tiles all use the same centered 3x3 dot lattice, which keeps the
#      count code regular even when many tiles are shown together.
#    - Input motifs remain isolated connected components in a clean 3x3 slot
#      grid, so color matching remains readable despite the higher object count.
# g) Change inferability:
#    - No hidden numeric property of the color IDs controls the result.
#      Matching is by same color, encoding is by visible motif area, and sorting
#      is by visibly comparable tile dot counts / motif sizes.
# new_solver_assumptions:
#    - Every block has a matching same-colored motif component.
#    - Matching motif areas are in 4..8, so the 3x3 tally lattice is sufficient.
#    - Equal-area ties may occur; they are resolved by original block order.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


COUNT_DOT_COORDS = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
]


def _is_solid_block(grid: np.ndarray, comp: dict[str, object]) -> bool:
    """Return whether the component is a solid 5x5 monochrome destination block."""
    bbox = comp["bbox"]
    if bbox_h(bbox) != 5 or bbox_w(bbox) != 5:
        return False
    color = int(comp["color"])
    return bool((crop_bbox(grid, bbox) == int(color)).all())


def _render_count_tile(color: int, bg: int, count: int) -> np.ndarray:
    """Render one encoded 5x5 tile using the fixed inner 3x3 tally lattice."""
    tile = np.full((5, 5), int(color), dtype=int)
    capped = max(0, min(int(count), len(COUNT_DOT_COORDS)))
    for rr, cc in COUNT_DOT_COORDS[:capped]:
        tile[int(rr), int(cc)] = int(bg)
    return tile


def _render_carved_tile(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    """Render the old carve-by-shape tile, used only for generator diagnostics."""
    tile = np.full((5, 5), int(color), dtype=int)
    mh, mw = map(int, mask.shape)
    off_r = (5 - mh) // 2
    off_c = (5 - mw) // 2
    for rr, cc in np.argwhere(mask):
        tile[int(off_r + rr), int(off_c + cc)] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match motifs to same-colored 5x5 blocks, encode motif area as dot counts, then sort tiles by area.

    Input structure:
    - The background is the most common color.
    - There are 8 or 9 solid 5x5 monochrome destination blocks, each in a distinct non-background color.
    - Elsewhere, each block color appears again as one smaller same-colored connected motif component inside a tidy motif-slot region.

    Rule:
    - Pair every block with the smallest same-colored non-5x5 component.
    - Ignore that motif's exact geometry and keep only its filled-cell area.
    - Replace the block with a 5x5 count-encoded tile whose inner 3x3 lattice shows that many background dots, then sort all tiles by motif area from small to large and repack them into the original rectangular block-grid shape with 1-cell gaps and a 1-cell border.

    Key insight:
    - Color still provides the match, but the motif acts as a numeric clue rather than a stencil, and the final arrangement is driven by motif area instead of the blocks' original positions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        if not _is_solid_block(g, comp):
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert blocks
    assert 8 <= len(blocks) <= 9

    # STEP 2) For each block color, identify its motif area:
    # the smallest same-colored component that is not one of the 5x5 blocks.
    motif_areas: dict[int, int] = {}
    block_colors = {int(block["col"]) for block in blocks}
    for color in block_colors:
        candidates: list[tuple[int, int, int, int]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        for comp in comps_col:
            if _is_solid_block(g, comp):
                continue
            r0, c0, r1, c1 = map(int, comp["bbox"])
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0)))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            motif_areas[int(color)] = int(candidates[0][0])
    assert len(motif_areas) == len(block_colors)

    # STEP 3) Encode each 5x5 tile and keep the matched area for sorting.
    encoded_blocks: list[dict[str, object]] = []
    for block in blocks:
        color = int(block["col"])
        area = int(motif_areas[int(color)])
        tile = _render_count_tile(color, bg, area)
        encoded_blocks.append(
            {
                "r0": int(block["r0"]),
                "c0": int(block["c0"]),
                "area": int(area),
                "tile": tile,
            }
        )

    # STEP 4) Pack the encoded tiles into the original rectangular slot grid,
    # but fill slots in ascending motif-area order with spatial order as tie-break.
    rs = sorted({int(block["r0"]) for block in blocks})
    cs = sorted({int(block["c0"]) for block in blocks})
    R = len(rs)
    C = len(cs)
    assert R * C == len(blocks)

    out_h = 1 + R * 5 + (R - 1) + 1
    out_w = 1 + C * 5 + (C - 1) + 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    ordered = sorted(
        encoded_blocks,
        key=lambda item: (int(item["area"]), int(item["r0"]), int(item["c0"])),
    )
    for idx, item in enumerate(ordered):
        rr = 1 + (idx // C) * 6
        cc = 1 + (idx % C) * 6
        out[rr : rr + 5, cc : cc + 5] = item["tile"]

    return out


# --------------- generator (combo D7/3 + D13/1 + D14/0) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (31, 36)
CANVAS_WIDTH_RANGE = (31, 36)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    9: [(3, 3)],
}
FILLED_SLOT_OPTIONS = {
    8: [8, 9],
    9: [9],
}
MAX_EMPTY_SLOTS = 1

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

# Generation retry budget.
GENERATION_RETRIES = 400

# Repeated-support area templates used to make count-only encoding observable.
AREA_TEMPLATES = {
    8: [
        [4, 4, 5, 5, 6, 7, 8, 8],
        [4, 4, 5, 6, 6, 7, 7, 8],
        [4, 5, 5, 6, 6, 7, 8, 8],
    ],
    9: [
        [4, 4, 5, 5, 6, 6, 7, 8, 8],
        [4, 4, 5, 6, 6, 7, 7, 8, 8],
        [4, 5, 5, 6, 6, 7, 7, 8, 8],
    ],
}


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(mask.sum())
        if area not in masks:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if is_contiguous(mask.astype(int), background=0, connectivity=8):
            masks[area].append(mask)
    return masks


MASKS_BY_AREA = _make_masks_by_area()


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

    top_r = random.randint(
        OUTER_MARGIN, H - top_h - bot_h - OUTER_MARGIN - REGION_GAP
    )
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

    left_c = random.randint(
        OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP
    )
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
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for Mr, Mc in MOTIF_LAYOUTS[9]:
                mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                if not _can_split(H, W, dh, dw, mh, mw):
                    continue
                for filled in FILLED_SLOT_OPTIONS[count]:
                    empty_slots = Mr * Mc - filled
                    if not (0 <= empty_slots <= MAX_EMPTY_SLOTS):
                        continue
                    if filled > COLOR_COUNT - 1:
                        continue
                    out.append((R, C, Mr, Mc, filled))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a feasible layout, keeping 8-block and 9-block cases both active."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    chosen_count = random.choice(counts)
    candidates = [cfg for cfg in configs if cfg[0] * cfg[1] == chosen_count]
    return random.choice(candidates)


def _random_mask(area: int | None = None) -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    if area is None:
        area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[int(area)]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors and optional distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _sample_area_sequence(count: int) -> list[int]:
    """Pick a repeated-support area multiset and scramble it into input spatial order."""
    areas = list(random.choice(AREA_TEMPLATES[count]))
    random.shuffle(areas)
    return [int(area) for area in areas]


def _choose_matching_specs(block_colors: list[int]) -> list[dict[str, object]]:
    """Assign each block color one motif area and one sampled mask of that area."""
    areas = _sample_area_sequence(len(block_colors))
    area_banks: dict[int, list[np.ndarray]] = {}
    for area in sorted(set(areas)):
        need = areas.count(area)
        bank = MASKS_BY_AREA[int(area)]
        chosen = random.sample(bank, need) if need <= len(bank) else [_random_mask(area) for _ in range(need)]
        random.shuffle(chosen)
        area_banks[int(area)] = [mask.copy() for mask in chosen]

    specs: list[dict[str, object]] = []
    for color, area in zip(block_colors, areas):
        specs.append(
            {
                "color": int(color),
                "area": int(area),
                "mask": area_banks[int(area)].pop(),
            }
        )
    return specs


def _ordered_specs(block_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort block specs by area, preserving input order within equal-area groups."""
    indexed = list(enumerate(block_specs))
    indexed.sort(key=lambda pair: (int(pair[1]["area"]), int(pair[0])))
    return [spec for _idx, spec in indexed]


def _render_expected(bg: int, R: int, C: int, block_specs: list[dict[str, object]]) -> np.ndarray:
    """Render the oracle output for the fused rule."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for idx, spec in enumerate(_ordered_specs(block_specs)):
        rr = OUTPUT_BORDER + (idx // C) * DST_BLOCK_STEP
        cc = OUTPUT_BORDER + (idx % C) * DST_BLOCK_STEP
        tile = _render_count_tile(int(spec["color"]), int(bg), int(spec["area"]))
        paste(expected, tile, rr, cc)
    return expected


def _render_oldstyle_sorted(bg: int, R: int, C: int, block_specs: list[dict[str, object]]) -> np.ndarray:
    """Render the area-sorted output using the original carve-by-shape transform."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for idx, spec in enumerate(_ordered_specs(block_specs)):
        rr = OUTPUT_BORDER + (idx // C) * DST_BLOCK_STEP
        cc = OUTPUT_BORDER + (idx % C) * DST_BLOCK_STEP
        tile = _render_carved_tile(int(spec["color"]), int(bg), spec["mask"])
        paste(expected, tile, rr, cc)
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_specs: list[dict[str, object]]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for idx, spec in enumerate(block_specs):
        i = idx // C
        j = idx % C
        block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(spec["color"]), dtype=int)
        paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _slot_items(
    block_specs: list[dict[str, object]], distractor_colors: list[int]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(spec["color"]), spec["mask"]) for spec in block_specs
    ]
    items.extend((int(color), _random_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_specs: list[dict[str, object]],
    distractor_colors: list[int],
) -> None:
    """Assign matching motifs and optional distractors to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    for (sr, sc), (color, mask) in zip(slots, _slot_items(block_specs, distractor_colors)):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_specs: list[dict[str, object]],
    distractor_colors: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_specs)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_specs, distractor_colors)
    return inp


def _inversion_count(values: list[int]) -> int:
    """Count out-of-order pairs relative to ascending order."""
    return sum(
        1
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if int(values[i]) > int(values[j])
    )


def _movement_count(values: list[int]) -> int:
    """Count how many items change slot after stable area sorting."""
    ordered = sorted(range(len(values)), key=lambda idx: (int(values[idx]), int(idx)))
    return sum(1 for new_idx, old_idx in enumerate(ordered) if int(new_idx) != int(old_idx))


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the fused behavior is not visible."""
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
    block_specs = _choose_matching_specs(block_colors)
    spatial_areas = [int(spec["area"]) for spec in block_specs]

    if _inversion_count(spatial_areas) < 2:
        return None
    if _movement_count(spatial_areas) < 2:
        return None

    expected = _render_expected(bg, R, C, block_specs)
    oldstyle_sorted = _render_oldstyle_sorted(bg, R, C, block_specs)
    if np.array_equal(expected, oldstyle_sorted):
        return None

    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_specs, distractor_colors, placement
    )
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

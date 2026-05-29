# Variant of 58f5dbd5
# Dimension: D19 — src-dst基数关系 (meta)
# Change: 1:1 — 每个方块恰好对应一个匹配motif... → 多对一（多个motif雕刻同一方块）

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D19 variant
# -----------------------------------
# a) Concrete behavior:
#    - The destination side is still a rectangular array of solid 5x5 monochrome blocks.
#    - The source side is still a separate rectangular array of 3x3 motif slots.
#    - The change is that each block color now appears in 2 or 3 different source motifs,
#      not just one. All same-colored motifs are centered onto the same 5x5 block and
#      carved as background holes, so the output hole pattern is the union of several masks.
#    - Small example: if color A has two motifs m1 and m2, the original task would carve
#      only one centered 3x3 mask. The variant carves both centered masks into the same
#      5x5 tile, so every cell present in m1 or m2 becomes background in the output.
#
# b) Complexity calibration:
#    - Keeping the old 3-6 block count is physically incompatible with 2-3 motifs per block
#      inside the original 18-22 by 18-24 canvas while preserving region separation.
#    - The minimal compatible adjustment is to use 3-4 blocks. Count 4 supports 2 motifs per
#      block; count 3 supports either 2 or 3 motifs per block. This keeps the original object
#      vocabulary and layout style while making the new cardinality visible.
#
# c) Hidden-assumption audit against the original main():
#    - Original Step 2 sorted same-colored components and kept only the smallest one. That
#      relied on the 1:1 property and is invalid here because additional matching motifs would
#      be silently ignored.
#    - Original `motifs: dict[int, np.ndarray]` assumed one mask per color. The variant needs
#      `dict[int, list[np.ndarray]]`.
#    - Original Step 3 carved one mask. In the variant, repeated carving is safe because all
#      writes change cells only from block color to background; later writes cannot corrupt an
#      already carved hole.
#
# d) Placement legality:
#    - Destination region placement stays unchanged: one separated rectangle of 5x5 blocks.
#    - Motif slots are expanded to capacities 8, 9, or 10 so they can hold 2-3 motifs per
#      block with at most one empty slot.
#    - The generator places only separated regions and uses fixed slot steps, so objects do
#      not overlap or go out of bounds.
#
# e) Visual observability:
#    - All matching cues remain visible in color and repetition: each block color now appears
#      once as a big square and 2-3 times as small sprites.
#    - To avoid the change degenerating into an indistinguishable full 3x3 hole, matching mask
#      sets are chosen so their union has area 6-8 (not 9) and every motif contributes at
#      least one unique carved cell.
#
# f) Visual cleanliness:
#    - Regions remain axis-separated, each motif still occupies a clean 3x3 slot, and there
#      are no same-color distractors. This keeps the motif region readable despite denser reuse.
#
# g) Color inference:
#    - No color-id arithmetic or ordering is used. Matching is purely by visible color equality.
#      Output packing order still depends only on input row/column positions of the large blocks.
#
# h) Variant metric used in validation:
#    - For generated samples we can count matching motifs per block color and compare the new
#      output against a simulated old solver that keeps only one motif per color. A healthy
#      variant should show 2-3 matches per block and frequent non-zero output differences.
#
# i) New solver assumptions and generator guarantees:
#    - Solver assumption: every non-5x5 component sharing a block color is an intended motif.
#    - Generator guarantee: block colors are unique, and only true matching motifs use those
#      colors. Distractor colors are omitted so the solver's "collect all same-color motifs"
#      rule is exact on generated inputs.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given in-bounds coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine multiple color-matched motifs into one carved block pattern and pack the carved blocks.

    Input structure:
    - The dominant color is the background.
    - Several solid 5×5 monochrome blocks appear in a rectangular arrangement, each with a unique non-background color.
    - Elsewhere, every block color appears in 2-3 small same-colored motif sprites that fit inside 3×3 boxes.

    Rule:
    - For each block color, collect all non-5×5 components of that color.
    - Center every collected motif inside the matching 5×5 block and recolor the motif cells to background; the carved result is the union of all centered motifs.
    - Pack the carved 5×5 blocks into a compact output grid preserving their original row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight:
    - Color still identifies the source-to-destination match, but the stencil for one block is now built from every same-colored motif rather than from a single sprite.
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

    # STEP 2) For each block color, collect ALL smaller same-colored components.
    motifs_by_color: dict[int, list[np.ndarray]] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, int, np.ndarray]] = []
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
            candidates.append((int(r0), int(c0), int(comp["area"]), int(h * w), int(len(rel_coords)), m))

        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
        motifs_by_color[int(bcol)] = [m for _r0, _c0, _area, _bbox_area, _n, m in candidates]

    # STEP 3) Carve each 5x5 tile by overlaying every centered motif of the same color.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        for mm in motifs_by_color.get(int(col), []):
            mh, mw = map(int, mm.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
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


# --------------- generator (o5x, D19 variant) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
}
MAX_EMPTY_SLOTS = 1

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
UNION_AREA_RANGE = (6, 8)

# Generation retry budget.
GENERATION_RETRIES = 400
MASK_SET_RETRIES = 400


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
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


MASKS_BY_AREA = _make_masks_by_area()
ALL_MATCH_MASKS = [
    m.copy()
    for area in range(MOTIF_AREA_RANGE[0], MOTIF_AREA_RANGE[1] + 1)
    for m in MASKS_BY_AREA[area]
]


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


def _motifs_per_color_options(count: int) -> list[int]:
    """Return feasible matching-motif multiplicities for the sampled block count."""
    if count == 3:
        return [2, 3]
    return [2]


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int, int]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for motifs_per_color in _motifs_per_color_options(count):
                filled = count * motifs_per_color
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (filled <= capacity <= filled + MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, motifs_per_color))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving a count-first weighting."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _k in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _choose_palette(bg: int, count: int) -> list[int]:
    """Pick distinct destination colors from the non-background palette."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    return palette[:count]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _union_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Return the boolean union of equally shaped masks."""
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= m
    return union


def _mask_set_is_valid(masks: list[np.ndarray]) -> bool:
    """Require each motif to matter and avoid degenerate full-center carve-outs."""
    signatures = {tuple(int(v) for v in m.flatten()) for m in masks}
    if len(signatures) != len(masks):
        return False

    union = _union_mask(masks)
    if not (UNION_AREA_RANGE[0] <= int(union.sum()) <= UNION_AREA_RANGE[1]):
        return False

    for idx, m in enumerate(masks):
        others = np.zeros_like(m, dtype=bool)
        for j, other in enumerate(masks):
            if j != idx:
                others |= other
        if not np.any(m & ~others):
            return False
    return True


def _choose_mask_set(count: int) -> list[np.ndarray]:
    """Sample 2-3 unique motif masks whose union stays visually informative."""
    for _ in range(MASK_SET_RETRIES):
        masks = [m.copy() for m in random.sample(ALL_MATCH_MASKS, count)]
        if _mask_set_is_valid(masks):
            return masks
    raise RuntimeError("failed to sample a valid matching motif set")


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_mask_sets: dict[int, list[np.ndarray]]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            inner = tile[motif_slice, motif_slice]
            inner[_union_mask(motif_mask_sets[color])] = bg
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
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _slot_items(
    block_colors: list[int], motif_mask_sets: dict[int, list[np.ndarray]]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = []
    for color in block_colors:
        for mask in motif_mask_sets[color]:
            items.append((int(color), mask))
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_mask_sets: dict[int, list[np.ndarray]],
) -> None:
    """Randomly assign the matching motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, motif_mask_sets)

    for (sr, sc), (color, mask) in zip(slots, slot_items):
        _paint_motif(
            grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_mask_sets: dict[int, list[np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, motif_mask_sets
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
    R, C, Mr, Mc, motifs_per_color = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors = _choose_palette(bg, count)
    motif_mask_sets = {
        color: _choose_mask_set(motifs_per_color) for color in block_colors
    }

    expected = _render_expected(bg, R, C, block_colors, motif_mask_sets)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, motif_mask_sets, placement
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D19 input")

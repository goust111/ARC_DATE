# Combo variant of 58f5dbd5 (index 22)
# combo_spec: D11/0+D15/2
# Name: 7x7方块+雕刻扩散

# Design analysis for the combo variant
# - concretization:
#   The input keeps the original two-region language: one region contains several solid destination
#   blocks, and a separate region contains small monochrome 3x3-bounded motifs whose colors match
#   those blocks plus possible distractors. The fused change is local to the destination tiles:
#   each destination is now a solid 7x7 square instead of 5x5, the matched motif is centered at
#   offset (2,2), those motif cells are carved to background, and then the carved background grows
#   by one simultaneous 4-neighbor step. A concrete trace: if a block's motif is a 3x3 plus, the
#   source task would only punch the centered plus; this combo first punches that plus inside the
#   7x7 tile and then expands it into a larger diamond/cross-like cavity within the centered 5x5
#   area. The output still packs the processed tiles by their original row/column order with a
#   1-cell border and 1-cell gaps, but the packed grid is larger because each tile is 7x7.
# - complexity calibration:
#   The two modifications reinforce each other rather than acting independently. If the blocks grew
#   to 7x7 without the second step, the change would mostly be "more empty margin"; if we kept 5x5
#   and only expanded the cavity, many shapes would hit the border immediately. Using 7x7 gives the
#   expansion visible room to grow while keeping the rule simple: exactly one matched 3x3 stencil
#   and exactly one orthogonal growth step. The generator keeps motif masks connected and touching
#   all four sides of their 3x3 bbox so the growth is reliably observable across samples.
# - implicit assumption review:
#   Every original line that hard-coded 5x5 geometry stops being valid: block detection must test
#   for solid 7x7 squares, centered carving must target a 7x7 canvas, and output packing must use
#   a step of 8 rather than 6. The old assumption "hole cells are exactly the centered motif cells"
#   also fails because the rendered cavity is now the union of the carved mask and its one-step
#   4-neighbor dilation. The solver therefore routes tile construction through helpers that first
#   place the motif in a 7x7 hole mask and only then expand from the pre-expansion seed cells.
#   Matching by "smallest same-colored non-block component" still works because the generator keeps
#   every block color unique and reserves at most one non-block motif of that color.
# - placement legality:
#   Larger blocks make the old 18-24 canvas too cramped, especially for 2x3 or 3x2 layouts, so the
#   generator enlarges the canvas range and preserves separated placement by reusing the region-split
#   strategy from the base task. Destination blocks are placed on an 8-step lattice (7 body + 1
#   gap), motif slots remain on a 4-step lattice, and both regions keep at least a 1-cell outer
#   margin plus a 1-cell separating gap. This guarantees no overlaps, no off-grid painting, and no
#   accidental same-color merging between a block and its motif.
# - visual observability / tidiness / inference:
#   A human can still infer the mapping from colors: each output tile color matches one solid input
#   block and one small same-colored motif. The extra rule is visible because the hole pattern is
#   larger than the 3x3 motif but still clearly centered inside a 7x7 square, producing a consistent
#   "carve then spread" effect across examples. The layout remains tidy because the input retains two
#   cleanly separated regions and the output contains only one processed tile per original block.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given local coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


BLOCK_SIZE = 7
BLOCK_STEP = 8
MOTIF_SIZE = 3
OUTPUT_BORDER = 1


def _is_solid_block(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    """Return True iff the bbox is a solid monochrome 7x7 destination block."""
    if bbox_h(bbox) != BLOCK_SIZE or bbox_w(bbox) != BLOCK_SIZE:
        return False
    patch = crop_bbox(grid, bbox)
    return bool((patch == int(color)).all())


def _centered_hole_mask(mask: np.ndarray | None) -> np.ndarray:
    """Center the motif mask inside a 7x7 tile and return the carved hole mask."""
    holes = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    if mask is None:
        return holes

    mh, mw = map(int, mask.shape)
    off_r = (BLOCK_SIZE - mh) // 2
    off_c = (BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < BLOCK_SIZE and 0 <= tc < BLOCK_SIZE:
            holes[tr, tc] = True
    return holes


def _expand_holes_once(hole_mask: np.ndarray) -> np.ndarray:
    """Grow holes by one simultaneous 4-neighbor step inside the tile."""
    expanded = hole_mask.copy()
    H, W = expanded.shape
    for r, c in np.argwhere(hole_mask):
        r = int(r)
        c = int(c)
        if r > 0:
            expanded[r - 1, c] = True
        if r + 1 < H:
            expanded[r + 1, c] = True
        if c > 0:
            expanded[r, c - 1] = True
        if c + 1 < W:
            expanded[r, c + 1] = True
    return expanded


def _render_tile(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """Render one 7x7 tile using carve-then-expand semantics."""
    tile = np.full((BLOCK_SIZE, BLOCK_SIZE), int(color), dtype=int)
    carved_holes = _centered_hole_mask(motif_mask)
    grown_holes = _expand_holes_once(carved_holes)
    tile[grown_holes] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match motif sprites to same-colored 7x7 blocks, carve them, then spread each cavity once.

    Input structure:
    - The background is the most common color.
    - Several non-background solid 7x7 blocks occupy one separated region.
    - Elsewhere, each block color appears on a small same-colored motif sprite; extra distractor motifs may also appear.

    Rule:
    - For each 7x7 block, find the smallest same-colored component that is not itself a solid 7x7 block.
    - Center that motif inside the block and carve those covered cells to background.
    - Expand the freshly carved hole by one orthogonal step in all four directions, then pack all processed 7x7 tiles in their original row/column order with 1-cell gaps and a 1-cell outer border.

    Key insight: The motif is a centered stencil seed, not the final hole; the visible cavity is the carved motif plus one simultaneous 4-neighbor growth step inside a larger 7x7 block.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 7x7 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        col = int(comp["color"])
        if not _is_solid_block(g, b, col):
            continue
        r0, c0, _r1, _c1 = map(int, b)
        blocks.append({"col": col, "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 7x7 blocks.
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
            if _is_solid_block(g, b, bcol):
                continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Render each 7x7 tile via carve-then-expand.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        col = int(b["col"])
        carved[(r0, c0)] = _render_tile(col, bg, motifs.get(col))

    # STEP 4) Pack processed 7x7 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(OUTPUT_BORDER + len(rs) * BLOCK_SIZE + (len(rs) - 1) + OUTPUT_BORDER)
    out_w = int(OUTPUT_BORDER + len(cs) * BLOCK_SIZE + (len(cs) - 1) + OUTPUT_BORDER)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[r0])
        j = int(c_index[c0])
        rr = int(OUTPUT_BORDER + i * BLOCK_STEP)
        cc = int(OUTPUT_BORDER + j * BLOCK_STEP)
        out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = carved[(r0, c0)]

    return out


# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (25, 32)
CANVAS_WIDTH_RANGE = (25, 34)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = BLOCK_SIZE
DST_BLOCK_STEP = BLOCK_STEP
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1

# Generation retry budget.
GENERATION_RETRIES = 500


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
    """Sample a valid layout while preserving the original count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into the input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _render_tile(color, bg, motif_masks[color])
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
    """Paint the solid 7x7 destination blocks into the input grid."""
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
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[color]) for color in block_colors
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
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, motif_masks)

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
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
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
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, placement
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

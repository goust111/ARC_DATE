# Variant of 58f5dbd5
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: motif精灵与5x5方块通过相同颜色匹配（s.color == d.color）... → 形状嵌合匹配

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)


# Design analysis
# a) Concrete behavior and trace:
#    - Input still has two separated regions: several 5x5 destination blocks and several 3x3 motif sprites.
#    - The destination blocks are no longer solid. Each block is mostly one color, but one edge-centered 3x3
#      window contains a bg notch whose occupied cells follow one motif mask.
#    - Example: a red 5x5 block may have a top-edge notch shaped like
#        111
#        010
#        011
#      while the matching sprite elsewhere may be blue with exactly that same 3x3 mask.
#      The output is not the notched input block; it is a rebuilt solid red 5x5 tile with that matched mask
#      carved in the center at offset (1,1).
#    - Original logic at the matching step: "find the small component with the same color as the block".
#      Variant logic: "extract the block's notch mask, then find the small component with the same tight shape".
#      This is the only semantic switch; packing and centered carving stay the same.
# b) Complexity calibration:
#    - Real block notch shapes are unique within a sample.
#    - Distractor motifs also use unique shapes, but none of them matches a block notch.
#    - Matching motif colors are a cyclic derangement of the block colors, so color matching is systematically
#      wrong rather than merely unhelpful. This makes the D1 change visible and prevents regression to the
#      original color-key rule.
# c) Implicit assumption review against the original main():
#    - Original block detection assumed a fully solid 5x5 patch. That no longer holds, so block detection must
#      accept 5x5 monochrome components whose patch contains only {block_color, bg} and exactly one edge-touching
#      bg component acting as the notch.
#    - Original motif lookup assumed one relevant motif per block color. That assumption is now false by design,
#      so motifs must be indexed by normalized binary shape rather than color.
#    - The carving step can still reuse the matched motif mask once found; the centered output geometry is unchanged.
# d) Placement legality:
#    - Notches are restricted to one of four side-centered 3x3 windows inside each 5x5 block: top, bottom, left,
#      or right. This keeps the color component connected, keeps the notch visibly attached to the block boundary,
#      and ensures every block still occupies a clean 5x5 bbox for solver detection.
#    - Destination and motif regions keep the original separated placement strategy, so objects do not overlap.
# e) Visual observability:
#    - Humans can see that block colors and motif colors no longer correspond.
#    - Because every input block already exposes a unique notch silhouette on its boundary, the relevant cue is a
#      visible spatial shape match rather than any hidden id.
#    - A few distractors with other unique shapes force an explicit comparison instead of a trivial one-to-one scan.
# f) Visual cleanliness:
#    - Only two object types appear in each region: notched 5x5 blocks and isolated 3x3 motifs.
#    - No extra markers, labels, or layered overlays are introduced; the scene stays close to the original task.
# g) Color-rule audit:
#    - Colors are not used for matching or ordering. They only distinguish components as visible paint.
#    - Output block order still comes from spatial row/column order, which is visible and already part of the base task.
# h) New solver assumptions and generator guarantees:
#    - Solver assumes each valid block contains exactly one edge-connected notch component with bbox at most 3x3.
#      Generator guarantees this by carving exactly one sampled 3x3 mask into exactly one side-centered slot.
#    - Solver assumes each real notch shape appears among motif candidates. Generator guarantees this by placing one
#      motif for every block mask.
#    - Solver assumes shape keys are unambiguous. Generator guarantees all occupied motif shapes are unique.


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


def _tight_mask_from_coords(coords) -> np.ndarray:
    """Normalize arbitrary coords into a tight boolean mask."""
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    rel = [(int(r) - r0, int(c) - c0) for r, c in coords]
    return mask_from_coords(rel, (int(r1 - r0 + 1), int(c1 - c0 + 1)))


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    """Hashable key for exact shape comparison."""
    h, w = map(int, mask.shape)
    bits = tuple(int(v) for v in mask.astype(int).ravel())
    return h, w, bits


def _extract_notch_mask(patch: np.ndarray, bg: int) -> np.ndarray | None:
    """Return the unique edge-touching bg notch inside a 5x5 block patch."""
    empty = (patch == int(bg))
    if not bool(empty.any()):
        return None

    notches: list[np.ndarray] = []
    for pts in components_mask(empty, connectivity=8):
        rows = pts[:, 0]
        cols = pts[:, 1]
        if not (
            bool((rows == 0).any())
            or bool((rows == patch.shape[0] - 1).any())
            or bool((cols == 0).any())
            or bool((cols == patch.shape[1] - 1).any())
        ):
            continue
        mask = _tight_mask_from_coords([(int(r), int(c)) for r, c in pts])
        h, w = map(int, mask.shape)
        if h > 3 or w > 3:
            continue
        notches.append(mask)

    if len(notches) != 1:
        return None
    return notches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match motifs to notched blocks by exact shape fit, then carve centered copies into packed output tiles.

    Input structure:
    - bg is the most common color.
    - Several 5x5 destination blocks appear as mostly solid monochrome squares, but each block already contains one bg notch touching a block edge.
    - Elsewhere are small monochrome motif sprites; the correct sprite for a block is the one whose tight shape exactly equals that block's notch, regardless of color.

    Rule:
    - Detect each 5x5 notched block and extract its notch mask.
    - Find the motif with the same normalized shape, rebuild a solid 5x5 tile of the block's color, and carve that matched motif into the tile's center.
    - Pack the carved 5x5 tiles into a compact grid that preserves the original block row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight: The notch is only a matching cue. Its original edge position is discarded; after the shape match is found, the motif is reused as a centered carving stencil.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: 5x5 mostly monochrome squares with one edge-touching bg notch.
    blocks: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if not bool(np.isin(patch, [int(bg), int(col)]).all()):
            continue
        notch = _extract_notch_mask(patch, bg)
        if notch is None:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0), "notch": notch})

    assert blocks

    # STEP 2) Index motif candidates by tight shape instead of by color.
    motifs: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) == 5 and bbox_w(b) == 5:
            continue
        r0, c0, r1, c1 = map(int, b)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h > 3 or w > 3:
            continue
        mask = _tight_mask_from_coords(comp["coords"])
        motifs[_mask_key(mask)] = mask

    # STEP 3) Rebuild and carve each 5x5 tile using the motif whose shape fits the notch.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        notch = b["notch"]
        mm = motifs.get(_mask_key(notch))
        assert mm is not None
        mh, mw = map(int, mm.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mm):
            tile[int(off_r + rr), int(off_c + cc)] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[r0])
        j = int(c_index[c0])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


# --------------- generator (o5x, D1 shape-match variant) ---------------

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
    6: [(2, 3), (3, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
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
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
NOTCH_OFFSETS = [(0, 1), (1, 0), (1, 2), (2, 1)]

# Generation retry budget.
GENERATION_RETRIES = 300


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
ALL_MASKS = [m for area in sorted(MASKS_BY_AREA) for m in MASKS_BY_AREA[area]]


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
    """Sample a valid layout while preserving the original two-stage count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _choose_unique_masks(count: int) -> list[np.ndarray]:
    """Sample pairwise-distinct motif masks."""
    return [m.copy() for m in random.sample(ALL_MASKS, count)]


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor-only colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _deranged_match_colors(block_colors: list[int]) -> list[int]:
    """Give the true matching motifs colors that never equal their own block color."""
    shift = random.randint(1, len(block_colors) - 1)
    return [int(block_colors[(i + shift) % len(block_colors)]) for i in range(len(block_colors))]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], block_masks: list[np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            motif_view = tile[
                MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
                MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
            ]
            motif_view[block_masks[idx]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray,
    bg: int,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    block_masks: list[np.ndarray],
    notch_offsets: list[tuple[int, int]],
) -> None:
    """Paint the 5x5 destination blocks with visible edge notches into the input grid."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            nr, nc = notch_offsets[idx]
            notch_view = block[nr : nr + MOTIF_SIZE, nc : nc + MOTIF_SIZE]
            notch_view[block_masks[idx]] = int(bg)
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _slot_items(
    match_colors: list[int],
    block_masks: list[np.ndarray],
    distractor_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), mask.copy()) for color, mask in zip(match_colors, block_masks)
    ]
    items.extend(
        (int(color), mask.copy()) for color, mask in zip(distractor_colors, distractor_masks)
    )
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    match_colors: list[int],
    block_masks: list[np.ndarray],
    distractor_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(match_colors, block_masks, distractor_colors, distractor_masks)

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
    match_colors: list[int],
    distractor_colors: list[int],
    block_masks: list[np.ndarray],
    distractor_masks: list[np.ndarray],
    notch_offsets: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled notched blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(
        inp, bg, dst_r, dst_c, R, C, block_colors, block_masks, notch_offsets
    )
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        match_colors,
        block_masks,
        distractor_colors,
        distractor_masks,
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
    match_colors = _deranged_match_colors(block_colors)
    sampled_masks = _choose_unique_masks(filled)
    block_masks = sampled_masks[:count]
    distractor_masks = sampled_masks[count:]
    notch_offsets = [random.choice(NOTCH_OFFSETS) for _ in range(count)]

    expected = _render_expected(bg, R, C, block_colors, block_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        match_colors,
        distractor_colors,
        block_masks,
        distractor_masks,
        notch_offsets,
        placement,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D1 variant input")

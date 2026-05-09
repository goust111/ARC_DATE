# Variant of 58f5dbd5
# Dimension: D10 — 轨迹：源对象处理 (trajectory)
# Change: motif在输出中消失（不保留源），仅在方块上留下雕刻痕迹... → 保留源motif

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# ---------------------------------------------------------------------------
# Design analysis for the D10 keep-source variant
# ---------------------------------------------------------------------------
# design_analysis:
#   a_concretization:
#     input:
#       - Keep the original scene grammar unchanged: several solid 5x5 blocks of
#         unique colors plus several 3x3 motif slots elsewhere, including one
#         matching motif per block color and optional distractor motifs.
#       - A matching motif is still a same-colored connected sprite whose mask is
#         centered inside the 5x5 block to carve bg holes.
#     output:
#       - Each packed output cell is no longer just a carved 5x5 block.
#       - It becomes a 5x9 composite: [5x5 carved block][1-col bg gap][3x3 motif panel].
#       - The motif panel shows the original source sprite in its original shape,
#         centered vertically beside the carved result, so the "tool" and the
#         "result" are both visible at once.
#     trace_example:
#       - Original: a red 3x3 cross motif matches a red 5x5 block; output keeps
#         only the carved red 5x5 with cross-shaped bg holes.
#       - Variant: the same carving happens, but the output cell also includes the
#         red cross motif to the right of that carved block.
#       - Difference: the transformation changes only the packed-output rendering,
#         not the input matching or carving mechanics.
#   b_complexity_calibration:
#     - The visual change is strong because every output object widens from 5 to 9
#       columns and explicitly displays two related objects.
#     - Matching remains color-based and one-to-one, avoiding new ambiguity.
#     - Distractor motifs remain in input so the solver still must identify the
#       correct motif/block pairs rather than simply copying all motifs.
#   c_implicit_assumption_audit:
#     - Block detection still assumes 5x5 monochrome components; unchanged.
#     - Motif extraction still assumes the smallest same-colored non-block
#       component is the tool; generator guarantees one real tool per block color
#       and no same-colored distractor.
#     - Old packing assumed every output payload was 5x5; that is invalid now.
#       The new packer must reserve width for both the carved block and the kept
#       motif, otherwise motifs would be overwritten or clipped.
#   d_placement_legality:
#     - Input placement logic stays unchanged, so source motifs and destination
#       blocks remain separated and non-overlapping.
#     - Output packing uses a fixed 5x9 composite width with 1-cell inter-item
#       gaps and a 1-cell outer border, so composites never overlap.
#   e_visual_observability:
#     - The keep-source change is inferable from 2-3 examples because the right
#       side of every packed cell consistently contains the same motif shape that
#       produced the hole pattern inside the block.
#     - No ordering or recoloring depends on hidden numeric color IDs.
#   f_visual_tidiness:
#     - Keeping the motif beside, not on top of, the block avoids clutter.
#     - The 1-column internal gap clearly separates "tool" from "result".
#   g_color_inference:
#     - Colors are preserved rather than remapped, so the human-visible clue is
#       shape correspondence: the motif shape on the right matches the bg holes in
#       the 5x5 block on the left.
#   generator_guarantees_for_new_main:
#     - Every block color has exactly one matching motif color in input.
#     - Motif masks fit within 3x3, so the fixed motif panel is sufficient.
#     - The oracle renderer uses the same composite layout as main(), preventing
#       accidental regression back to the carved-only original behavior.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then pack both result and source.

    Input structure:
    - bg is the most frequent color.
    - Several solid 5x5 monochrome blocks appear in one region, each with a unique non-bg color.
    - Elsewhere, small same-colored motif sprites appear in 3x3-sized slots, alongside optional distractor motifs of other colors.

    Rule:
    - Match each block with the smallest same-colored non-block component, center that motif inside the 5x5 block, and turn the covered cells into bg holes.
    - Pack the blocks by their original row/column order, but for each packed item keep both objects: the carved 5x5 block on the left and the original motif sprite on the right.
    - Use a 1-cell outer border, a 1-cell gap between packed items, and a 1-cell internal gap between each block and its preserved motif.

    Key insight: The motif is still a stencil for carving, but in this variant the output also displays the stencil itself beside the carved block.
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

    assert not (not blocks)

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is NOT one of the 5x5 blocks.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []  # (area, bbox_area, r0, c0, mask)

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
                    continue  # skip the big 5x5 blocks

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)

            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each 5x5 tile:
    # center the motif mask within the 5x5 and repaint those mask cells to bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = motifs.get(int(col))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved blocks plus preserved motifs into composite output cells.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    block_size = 5
    tool_size = 3
    internal_gap = 1
    composite_w = block_size + internal_gap + tool_size
    row_step = 6
    col_step = composite_w + 1

    out_h = int(1 + len(rs) * block_size + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * composite_w + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        col = int(b["col"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * row_step)
        cc = int(1 + j * col_step)
        out[rr : rr + block_size, cc : cc + block_size] = carved[(int(r0), int(c0))]

        mm = motifs.get(int(col))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            tool_r = int(rr + (block_size - mh) // 2)
            tool_c = int(cc + block_size + internal_gap + (tool_size - mw) // 2)
            motif_patch = np.full(mm.shape, int(col), dtype=int)
            paste(out, motif_patch, tool_r, tool_c, mask=mm)

    return out


# --------------- generator (o5x + keep_source output) ---------------

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
OUTPUT_INTERNAL_GAP = 1
OUTPUT_MOTIF_PANEL = 3
OUTPUT_COMPOSITE_WIDTH = DST_BLOCK_SIZE + OUTPUT_INTERNAL_GAP + OUTPUT_MOTIF_PANEL
OUTPUT_COL_STEP = OUTPUT_COMPOSITE_WIDTH + 1

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
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output for the keep-source variant."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, OUTPUT_COMPOSITE_WIDTH, OUTPUT_COL_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            mask = motif_masks[color]
            mh, mw = map(int, mask.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(mask):
                tile[int(off_r + rr), int(off_c + cc)] = bg

            base_r = OUTPUT_BORDER + i * DST_BLOCK_STEP
            base_c = OUTPUT_BORDER + j * OUTPUT_COL_STEP
            paste(expected, tile, base_r, base_c)

            motif_r = base_r + (DST_BLOCK_SIZE - mh) // 2
            motif_c = base_c + DST_BLOCK_SIZE + OUTPUT_INTERNAL_GAP + (OUTPUT_MOTIF_PANEL - mw) // 2
            motif_patch = np.full(mask.shape, color, dtype=int)
            paste(expected, motif_patch, motif_r, motif_c, mask=mask)

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
    raise RuntimeError("failed to generate a valid 58f5dbd5 input")

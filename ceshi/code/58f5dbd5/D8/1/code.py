# Variant of 58f5dbd5
# Dimension: D8 — 变换操作：motif对齐方式 (transform)
# Change: motif居中于5x5方块内（偏移量=(5-h)//2, (5-w)//2）... → 由标记像素决定对齐
from __future__ import annotations

"""
Design analysis for the D8 anchor-alignment variant.

variant_spec:
  preserved_core:
    - Input still contains several 5x5 destination blocks and a separate motif region.
    - Matching is still by color: each destination block color has one same-colored motif.
    - Output still packs the carved 5x5 results with 1-cell gaps and a 1-cell border.
  changed_dimension:
    - Original rule always centered a motif inside its 5x5 block.
    - New rule uses a visible anchor marker inside each 5x5 block.
    - The motif's geometric center is aligned to that marker, so the carved hole pattern shifts.
  concrete_pixel_form:
    - Each destination block is no longer 25 cells of one color.
    - It is now 24 cells of the block color plus 1 singleton marker pixel of a shared anchor color.
    - The marker is restricted to the block's interior 3x3 positions, so a 3x3 motif can move without clipping.
    - Motifs remain monochrome 3x3-bbox connected sprites, but generator requires the motif center cell to be filled.
      This makes the marker location visibly become a hole in the output and strengthens the cue.
  behavior_trace:
    - Original:
      A 3x3 motif with mask cell (0,0) always carved tile cell (1,1), because the whole motif was centered.
    - Variant:
      If a block's marker sits at local position (1,3), the motif center (1,1) is moved there,
      so mask cell (0,0) carves tile cell (0,2) instead.
    - Difference:
      The carve offset now depends on an observed pixel inside each block, not on a constant formula.
  complexity_calibration:
    - The anchor color is shared across all blocks, so humans can spot the special role quickly.
    - Anchor positions vary across blocks in the same puzzle, and generator rejects the degenerate all-centered case.
    - The anchor is chosen from 9 interior positions only, keeping the rule small and repeatable instead of arbitrary.
  implicit_assumption_audit:
    - Original block detection assumed a solid 5x5 monochrome square. That breaks immediately here.
      New solver instead looks for a 24-cell monochrome component with 5x5 bbox and exactly one non-bg foreign pixel.
    - Original motif skip logic ignored only solid 5x5 components. That is no longer enough because blocks are area 24.
      New solver skips 5x5 / area-24 same-colored components when collecting motif candidates.
    - Original carving used a fixed centering offset. New solver computes top-left from the detected marker location.
  placement_legality:
    - Marker positions stay in rows/cols 1..3 of the 5x5 block, so the 3x3 motif always fits fully.
    - Destination and motif regions still use the original separated placement logic, so no new overlaps are introduced.
    - Distractor motifs still use colors disjoint from block colors and the anchor color.
  visual_observability:
    - Across 3-6 blocks per sample, the same special marker color repeatedly indicates different carve offsets.
    - Because the marker cell is part of the carved motif mask, its position maps directly to a missing output cell.
    - This is visible from a few examples without relying on hidden color-id ordering.
  cleanliness:
    - Only one extra pixel is added per block, so the destination region stays readable.
    - The marker color is reserved for anchors only, avoiding mixed semantics elsewhere.
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


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


def _extract_block(comp: dict, grid: np.ndarray, bg: int) -> dict[str, int] | None:
    """Recognize a 5x5 destination block with one internal anchor marker."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5 or int(comp["area"]) != 24:
        return None

    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    if int((patch == int(col)).sum()) != 24:
        return None
    if int((patch == int(bg)).sum()) != 0:
        return None

    others = np.argwhere(patch != int(col))
    if others.shape[0] != 1:
        return None

    ar, ac = map(int, others[0])
    marker_col = int(patch[int(ar), int(ac)])
    if int(marker_col) in {int(bg), int(col)}:
        return None
    if not (1 <= int(ar) <= 3 and 1 <= int(ac) <= 3):
        return None

    r0, c0, _r1, _c1 = map(int, b)
    return {
        "col": int(col),
        "r0": int(r0),
        "c0": int(c0),
        "anchor_r": int(ar),
        "anchor_c": int(ac),
        "marker_col": int(marker_col),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into marked 5x5 blocks and pack the results.

    Input structure:
    - bg = most common color.
    - Several 5x5 destination blocks are almost solid: 24 cells of one color plus 1 special marker pixel.
    - Elsewhere, each block color has a small same-colored motif sprite; extra distractor motifs may also appear.

    Rule:
    - Match each motif to the block with the same main color.
    - Use the marker pixel inside the 5x5 block as the motif anchor location, aligning the motif's center to it.
    - Recolor the aligned motif cells to bg, then pack all carved 5x5 blocks in original row/column order with 1-cell gaps and a 1-cell border.

    Key insight:
    - The odd-colored pixel inside each block is not part of the final object; it is a positioning cue that shifts where the stencil is carved.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: 5x5 almost-monochrome squares with one anchor marker.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        block = _extract_block(comp, g, bg)
        if block is not None:
            blocks.append(block)

    assert not (not blocks)

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 24-cell 5x5 destination bodies.
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
            if bbox_h(b) == 5 and bbox_w(b) == 5 and int(comp["area"]) == 24:
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

    # STEP 3) Carve each 5x5 tile:
    # align the motif's center to the detected anchor marker and repaint those cells to bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = motifs.get(int(col))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = int(b["anchor_r"]) - mh // 2
            off_c = int(b["anchor_c"]) - mw // 2
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


# --------------- generator (o5x / D8 anchor variant) ---------------

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
CENTER_ANCHOR = (2, 2)
ANCHOR_CHOICES = [(r, c) for r in range(1, 4) for c in range(1, 4)]

# Generation retry budget.
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not bool(m[1, 1]):
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


def _place_vertical(H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str) -> tuple[int, int, int, int]:
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


def _place_horizontal(H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str) -> tuple[int, int, int, int]:
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


def _place_regions(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> tuple[int, int, int, int] | None:
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
                if distractors > COLOR_COUNT - 2 - count:
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


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int], int]:
    """Pick destination colors, distractor colors, and one reserved marker color."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    marker_color = int(palette[count + (filled - count)])
    return block_colors, distractor_colors, marker_color


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _choose_anchor_positions(count: int) -> list[tuple[int, int]]:
    """Sample block anchor positions, rejecting the degenerate all-centered case."""
    while True:
        anchors = [random.choice(ANCHOR_CHOICES) for _ in range(count)]
        if not any(anchor != CENTER_ANCHOR for anchor in anchors):
            continue
        if len(set(anchors)) < 2:
            continue
        return anchors


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            anchor_r, anchor_c = anchors[idx]
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            mm = motif_masks[color]
            off_r = int(anchor_r) - mm.shape[0] // 2
            off_c = int(anchor_c) - mm.shape[1] // 2
            for rr, cc in np.argwhere(mm):
                tile[int(off_r + int(rr)), int(off_c + int(cc))] = int(bg)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    anchors: list[tuple[int, int]],
    marker_color: int,
) -> None:
    """Paint the 5x5 destination blocks with their internal anchor markers."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            ar, ac = anchors[idx]
            block[int(ar), int(ac)] = int(marker_color)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [(int(color), motif_masks[color]) for color in block_colors]
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
    marker_color: int,
    motif_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, anchors, marker_color)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks)
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
    block_colors, distractor_colors, marker_color = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}
    anchors = _choose_anchor_positions(count)

    expected = _render_expected(bg, R, C, block_colors, motif_masks, anchors)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        marker_color,
        motif_masks,
        anchors,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D8 input")

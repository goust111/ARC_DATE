# Variant of 58f5dbd5
# Dimension: D16 — 符号编码 (symbol_instructions)
# Change: 无符号编码，所有信息直接通过颜色和空间关系表达... → 标记像素指示排列顺序

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D16 variant
# -----------------------------------
# design:
#   reference_task_97d7923e:
#     takeaway: the symbol should not solve the whole puzzle alone; it should
#       only encode an index/rank that redirects an otherwise familiar transform.
#       Here the carving-by-color logic stays intact, while the final packing order
#       is no longer read from source coordinates.
#   concrete_form:
#     - Each 5x5 destination block keeps its original solid color and still has one
#       same-colored 3x3 motif elsewhere.
#     - Directly above each block sits a dedicated 2x3 "caption window" containing
#       1..N marker pixels of one neutral instruction color. The pixels fill that
#       2x3 box in row-major prefix order, so the human-visible quantity is just the
#       count of dots.
#     - The dot count is the block's output rank. Rank 1 goes to the first output
#       slot, rank 2 to the second, etc., in row-major order over the compact R x C
#       packed output shape.
#   trace_example:
#     - Original task: with a 2x2 block layout, the upper-left source block is always
#       copied to output slot (0,0) because source row/column order is preserved.
#     - Variant: the same upper-left source block might have 4 dots above it, so after
#       carving it is sent to output slot 4 instead. Another source block with 1 dot
#       becomes the new output slot 1 even if it started in the bottom row.
#     - Difference: color still identifies which motif carves which block, but spatial
#       source order is replaced by counted symbol order.
#   complexity_calibration:
#     - The symbol has only one degree of freedom: dot count 1..N where N is the number
#       of blocks (3, 4, or 6). This is visible, countable, and repeated across all blocks.
#     - Counts are unique within a sample, avoiding ambiguous placement.
#     - The marker shape is intentionally simple and fixed-size; only the number of lit
#       cells varies. This keeps the symbolic rule legible from 2-4 examples.
#   hidden_assumptions_review:
#     - Original step 1 assumed every relevant 5x5 object was an isolated solid component.
#       That still holds because markers use a separate color and stay one bg row above.
#     - Original step 2 chose the smallest same-colored non-5x5 component as the motif.
#       That would break if markers shared block colors, so the generator now reserves a
#       distinct instruction color that never appears in motifs or blocks.
#     - Original step 4 sorted blocks by (r0, c0). That is the specific behavior D16
#       replaces; the solver now reads a local 2x3 window above each block and uses that
#       count as the destination index.
#   placement_legality:
#     - Each block row in the source now includes a 2-row marker header, a 1-row gap,
#       then the 5x5 block. This guarantees the marker window is local, clean, and
#       non-touching.
#     - The canvas is enlarged to 24..28 so the expanded block region can still be
#       separated from the motif region without overlap.
#     - Layout families remain the same (1x3, 3x1, 2x2, 2x3, 3x2); only the source
#       block region footprint changes.
#   visual_observability:
#     - A human can compare input/output and see that block color ↔ motif carving still
#       matches, but output order no longer follows source geometry. The only visible new
#       cue explaining that reorder is the count of dots above each block.
#   cleanliness:
#     - Markers live in dedicated caption windows, not mixed into motif slots or carved
#       blocks, so the three object types remain separable by both location and geometry.
#   color_inference:
#     - No color-id arithmetic is used. Colors only support the original color matching,
#       while the new order information is encoded purely by visible pixel counts.
# solver_assumptions_checked_against_generator:
#   - Every block has exactly one non-empty 2x3 caption window directly above it.
#   - Caption windows contain only bg plus one instruction color, so counting non-bg
#     pixels recovers the rank.
#   - The ranks form a permutation of 1..block_count.
#   - Source blocks still occupy a full R x C rectangle, allowing the solver to infer
#     the packed output shape from the set of distinct source rows/cols.


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


def _marker_mask(rank: int) -> np.ndarray:
    """Connected 2x3 prefix mask with exactly `rank` lit cells."""
    m = np.zeros((MARKER_H, MARKER_W), dtype=bool)
    for idx in range(int(rank)):
        m[idx // MARKER_W, idx % MARKER_W] = True
    return m


def _solid_blocks(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Find isolated solid 5x5 monochrome blocks."""
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != DST_BLOCK_SIZE or bbox_w(b) != DST_BLOCK_SIZE:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": r0, "c0": c0})
    return blocks


def _extract_motifs(g: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, np.ndarray]:
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
            if bbox_h(b) == DST_BLOCK_SIZE and bbox_w(b) == DST_BLOCK_SIZE:
                patch = crop_bbox(g, b)
                if bool((patch == int(bcol)).all()):
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
    return motifs


def _rank_from_marker_window(g: np.ndarray, bg: int, block_r0: int, block_c0: int) -> int:
    """Count non-bg marker pixels in the fixed 2x3 caption window above a block."""
    rr0 = int(block_r0 - MARKER_H - MARKER_GAP)
    rr1 = int(block_r0 - MARKER_GAP)
    cc0 = int(block_c0 + MARKER_INSET_C)
    cc1 = int(cc0 + MARKER_W)
    window = g[rr0:rr1, cc0:cc1]
    vals = sorted({int(v) for v in window.flatten() if int(v) != bg})
    assert len(vals) == 1, "each caption window must contain exactly one marker color"
    rank = int((window != bg).sum())
    assert 1 <= rank <= MARKER_H * MARKER_W, "marker count out of supported range"
    return rank


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then place the carved blocks by local dot-count captions.

    Input structure:
    - bg is the most common color.
    - Several solid 5x5 monochrome blocks appear in an R x C source arrangement.
    - Each block has a tiny 2x3 marker window directly above it; the number of marker pixels is a unique rank.
    - Elsewhere, each block color has a small same-colored motif sprite, with optional distractor motifs in other colors.

    Rule:
    - Match every block to the smallest same-colored non-block motif and use that motif as a centered carving stencil.
    - Read the count of marker pixels above each block as its destination index in the packed output.
    - Pack the carved 5x5 tiles into the usual bordered layout, filling row-major slots by rank instead of by source coordinates.

    Key insight: Color still determines which motif punches holes in which block, but the final ordering is redirected by the counted symbol above each block rather than by the block's original row/column position.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks = _solid_blocks(g, bg)
    assert blocks

    block_colors = {int(b["col"]) for b in blocks}
    motifs = _extract_motifs(g, bg, block_colors)

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    R = len(rs)
    C = len(cs)
    assert len(blocks) == R * C

    carved_by_rank: dict[int, np.ndarray] = {}
    seen_ranks: set[int] = set()
    for b in blocks:
        rank = _rank_from_marker_window(g, bg, int(b["r0"]), int(b["c0"]))
        assert rank not in seen_ranks, "marker ranks must be unique"
        seen_ranks.add(rank)

        col = int(b["col"])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                    tile[tr, tc] = bg
        carved_by_rank[rank] = tile

    assert seen_ranks == set(range(1, len(blocks) + 1))

    out_h = int(1 + R * DST_BLOCK_SIZE + (R - 1) * PACK_GAP + 1)
    out_w = int(1 + C * DST_BLOCK_SIZE + (C - 1) * PACK_GAP + 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for rank, tile in carved_by_rank.items():
        pos = int(rank - 1)
        i = int(pos // C)
        j = int(pos % C)
        rr = int(OUTPUT_BORDER + i * PACK_STEP)
        cc = int(OUTPUT_BORDER + j * PACK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = tile

    return out


# --------------- generator (o5x + D16 symbols) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (24, 28)
CANVAS_WIDTH_RANGE = (24, 28)
COLOR_COUNT = 10

# Destination block geometry in the packed output.
DST_BLOCK_SIZE = 5
PACK_GAP = 1
PACK_STEP = DST_BLOCK_SIZE + PACK_GAP
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

# Source block geometry with symbol header.
MARKER_H = 2
MARKER_W = 3
MARKER_GAP = 1
MARKER_INSET_C = 1
SRC_BLOCK_ROW_STEP = 9
SRC_BLOCK_COL_STEP = 6
SRC_BLOCK_REGION_H = MARKER_H + MARKER_GAP + DST_BLOCK_SIZE
SRC_BLOCK_REGION_W = DST_BLOCK_SIZE

# Source layouts remain the original count families.
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

# Region placement.
OUTER_MARGIN = 1
REGION_GAP = 1

# Generation retry budget.
GENERATION_RETRIES = 400


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
            dh = _spanned_size(R, SRC_BLOCK_REGION_H, SRC_BLOCK_ROW_STEP)
            dw = _spanned_size(C, SRC_BLOCK_REGION_W, SRC_BLOCK_COL_STEP)
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
    """Pick block colors, distractor colors, and one dedicated marker color."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_count = filled - count
    distractor_colors = palette[count : count + distractor_count]
    marker_color = int(palette[count + distractor_count])
    return block_colors, distractor_colors, marker_color


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _choose_ranks(count: int) -> list[int]:
    """Return a non-identity permutation of 1..count so the D16 behavior is active."""
    ranks = list(range(1, count + 1))
    while True:
        random.shuffle(ranks)
        if any(rank != idx + 1 for idx, rank in enumerate(ranks)):
            return ranks


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    ranks: list[int],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, PACK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, PACK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for src_idx, color in enumerate(block_colors):
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        tile[motif_slice, motif_slice][motif_masks[int(color)]] = bg

        pos = int(ranks[src_idx] - 1)
        i = int(pos // C)
        j = int(pos % C)
        paste(
            expected,
            tile,
            OUTPUT_BORDER + i * PACK_STEP,
            OUTPUT_BORDER + j * PACK_STEP,
        )
    return expected


def _source_block_origin(dst_r: int, dst_c: int, i: int, j: int) -> tuple[int, int]:
    """Map a source layout slot to the 5x5 block's top-left inside the marked region."""
    return (
        int(dst_r + i * SRC_BLOCK_ROW_STEP + MARKER_H + MARKER_GAP),
        int(dst_c + j * SRC_BLOCK_COL_STEP),
    )


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    ranks: list[int],
    marker_color: int,
) -> None:
    """Paint solid source blocks together with their local order markers."""
    idx = 0
    marker_patch = np.full((MARKER_H, MARKER_W), int(marker_color), dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[idx])
            block_r, block_c = _source_block_origin(dst_r, dst_c, i, j)
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, block_r, block_c)

            marker_r = block_r - MARKER_H - MARKER_GAP
            marker_c = block_c + MARKER_INSET_C
            paste(grid, marker_patch, marker_r, marker_c, mask=_marker_mask(int(ranks[idx])))
            idx += 1


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
    ranks: list[int],
    marker_color: int,
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks, markers, and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, ranks, marker_color)
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

    dst_h = _spanned_size(R, SRC_BLOCK_REGION_H, SRC_BLOCK_ROW_STEP)
    dst_w = _spanned_size(C, SRC_BLOCK_REGION_W, SRC_BLOCK_COL_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors, marker_color = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}
    ranks = _choose_ranks(count)

    expected = _render_expected(bg, R, C, block_colors, motif_masks, ranks)
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
        motif_masks,
        ranks,
        marker_color,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D16 input")

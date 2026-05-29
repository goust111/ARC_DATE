# Combo variant of 58f5dbd5 (index 32)
# combo_spec: D14/2+D16/2
# Name: 按motif位置排序+标记点指示顺序

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# -------------------------------------
# design:
#   concrete_behavior:
#     - Keep the original two-region scene: a source lattice of solid 5x5 blocks and
#       a separate region of small same-colored motifs plus optional distractors.
#     - Add a D16-style 2x3 caption window above every source block. The number of lit
#       dots in that window is a unique rank from 1..N.
#     - Apply the D14 change to the true output rule: carved tiles are packed in the
#       scan order of their matching motifs, not in the scan order of the blocks.
#     - Fusion requirement: the dot-count rank and the matching-motif scan rank must
#       encode the same permutation. Example with source block colors [A, B, C, D]:
#       if matching motifs appear in row-major order [C, A, D, B], then ranks above
#       the source blocks become A->2, B->4, C->1, D->3, and the packed output order
#       is exactly [C, A, D, B].
#   complexity_calibration:
#     - The symbolic freedom is limited to counts 1..N with N in {3, 4, 6}, matching
#       the number of blocks. This is small enough to count visually and large enough
#       to force a real permutation.
#     - The generator rejects the identity permutation, so the combo never collapses
#       back to the original "preserve block order" behavior.
#     - A dedicated marker color keeps captions visually separate from both blocks and
#       motifs, avoiding accidental same-color component ambiguity.
#   hidden_assumptions_review:
#     - Original step 4 assumed source block coordinates determine output order; this
#       is no longer valid and is used only to infer the output rectangle shape R x C.
#     - D14 already requires storing motif positions, not just motif masks. The combo
#       keeps that and additionally derives a rank from each block's local caption.
#     - D16 alone could solve the permutation from captions only. Here the solver reads
#       both captions and motif positions and asserts that they agree, so the two clues
#       are coupled rather than independent add-ons.
#     - The smallest same-colored non-5x5 component heuristic still works because
#       marker color is reserved and never reused by blocks or motifs.
#   placement_legality:
#     - Source rows use an 8-cell local footprint per block: 2 marker rows, 1 blank
#       row, then the 5x5 block. A row step of 9 and column step of 6 preserve clear
#       gaps between caption windows and neighboring objects.
#     - The canvas is enlarged relative to the original task so the taller source block
#       region can still be separated from the motif region without overlap.
#     - Motifs remain inside a regular 3x3-slot grid with at most two empty slots, so
#       row-major motif order is legible and unambiguous.
#   visual_observability:
#     - A human can see that color still pairs each motif with one block, but the final
#       packed order no longer follows block geometry. The visible explanation is that
#       motif scan order and caption counts agree on the same rearrangement.
#     - No invisible property such as numeric color id participates in the order.
#   cleanliness:
#     - Object roles stay separated by region and geometry: caption windows above
#       blocks, 5x5 solids below them, and motif sprites in a distant second region.
#       There is no stacking of multiple symbol systems onto the same cells.
# solver_assumptions_checked_against_generator:
#   - Each block color appears exactly once as a 5x5 source block and exactly once as a
#     non-block motif.
#   - Every block has one 2x3 caption window directly above it with exactly one marker
#     color and a unique count in 1..block_count.
#   - Sorting matching motifs by their top-left coordinates yields the same permutation
#     as sorting blocks by their marker counts.
#   - Source blocks still occupy a full rectangular R x C lattice, so distinct source
#     row/column coordinates suffice to infer the packed output shape.


# Canvas and palette bounds.
# Keep enough headroom for the tall 3-row source layouts so branch coverage
# does not collapse toward only the flatter cases.
CANVAS_HEIGHT_RANGE = (28, 32)
CANVAS_WIDTH_RANGE = (24, 30)
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


def _non_identity_permutation(items: list[int]) -> list[int]:
    """Return a shuffled copy that differs from the input order."""
    out = list(items)
    if len(out) <= 1:
        return out
    for _ in range(20):
        random.shuffle(out)
        if out != items:
            return out
    return list(items[1:] + items[:1])


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


def _extract_motif_info(
    g: np.ndarray, bg: int, block_colors: set[int]
) -> dict[int, dict[str, object]]:
    """For each block color, recover the smallest same-colored non-block component."""
    motifs: dict[int, dict[str, object]] = {}
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
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), r0, c0, mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            _area, _bbox_area, r0, c0, mask = candidates[0]
            motifs[int(bcol)] = {"r0": r0, "c0": c0, "mask": mask}
    return motifs


def _rank_from_marker_window(g: np.ndarray, bg: int, block_r0: int, block_c0: int) -> int:
    """Count non-bg marker pixels in the fixed 2x3 caption window above a block."""
    rr0 = int(block_r0 - MARKER_H - MARKER_GAP)
    rr1 = int(block_r0 - MARKER_GAP)
    cc0 = int(block_c0 + MARKER_INSET_C)
    cc1 = int(cc0 + MARKER_W)
    window = g[rr0:rr1, cc0:cc1]
    assert window.shape == (MARKER_H, MARKER_W)
    vals = sorted({int(v) for v in window.flatten() if int(v) != bg})
    assert len(vals) == 1, "each caption window must contain exactly one marker color"
    rank = int((window != bg).sum())
    assert 1 <= rank <= MARKER_H * MARKER_W
    return rank


def _carve_tile(bg: int, color: int, mask: np.ndarray) -> np.ndarray:
    """Center-carve a motif mask into a solid 5x5 tile."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    mh, mw = map(int, mask.shape)
    off_r = (DST_BLOCK_SIZE - mh) // 2
    off_c = (DST_BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
            tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then pack the carved tiles by a motif-driven order that is echoed by local dot captions.

    Input structure:
    - The background is the most common color.
    - Several solid 5x5 monochrome blocks appear in a rectangular source lattice.
    - Each block has a 2x3 caption window directly above it whose dot count gives a unique rank.
    - Elsewhere, each block color also appears as a smaller motif sprite; distractor motifs in other colors may be present.

    Rule:
    - Match every block to the smallest same-colored non-block motif and use that motif as a centered carving stencil.
    - Sort the carved tiles by the row-major positions of their matching motifs in the input.
    - Confirm that this motif order matches the dot-count ranks above the blocks, then place the carved tiles into the usual packed output frame with 1-cell gaps and a 1-cell border.

    Key insight:
    - The motif positions provide the actual global ordering, while the counted dots above each block repeat that same permutation locally; color only handles block↔motif pairing.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks = _solid_blocks(g, bg)
    assert blocks

    block_colors = {int(b["col"]) for b in blocks}
    motifs = _extract_motif_info(g, bg, block_colors)
    assert all(int(b["col"]) in motifs for b in blocks)

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    R = len(rs)
    C = len(cs)
    assert len(blocks) == R * C

    motif_order = sorted(
        [int(b["col"]) for b in blocks],
        key=lambda col: (int(motifs[col]["r0"]), int(motifs[col]["c0"])),
    )
    motif_rank_by_color = {int(col): int(idx + 1) for idx, col in enumerate(motif_order)}

    carved_by_rank: dict[int, np.ndarray] = {}
    seen_ranks: set[int] = set()
    for b in blocks:
        col = int(b["col"])
        rank = _rank_from_marker_window(g, bg, int(b["r0"]), int(b["c0"]))
        assert rank == motif_rank_by_color[col], "marker order must agree with motif order"
        assert rank not in seen_ranks, "marker ranks must be unique"
        seen_ranks.add(rank)

        motif_mask = motifs[col]["mask"]
        assert isinstance(motif_mask, np.ndarray)
        carved_by_rank[rank] = _carve_tile(bg, col, motif_mask)

    assert seen_ranks == set(range(1, len(blocks) + 1))

    out_h = int(2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, PACK_STEP))
    out_w = int(2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, PACK_STEP))
    out = np.full((out_h, out_w), bg, dtype=int)

    for rank, tile in carved_by_rank.items():
        pos = int(rank - 1)
        i = int(pos // C)
        j = int(pos % C)
        rr = int(OUTPUT_BORDER + i * PACK_STEP)
        cc = int(OUTPUT_BORDER + j * PACK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = tile

    return out


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


def _render_expected(
    bg: int, R: int, C: int, ordered_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output after packing tiles in motif-scan order."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, PACK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, PACK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for idx, color in enumerate(ordered_colors):
        tile = _carve_tile(bg, int(color), motif_masks[int(color)])
        i = int(idx // C)
        j = int(idx % C)
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
    marker_patch = np.full((MARKER_H, MARKER_W), int(marker_color), dtype=int)
    idx = 0
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


def _plan_motif_payloads(
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> tuple[list[tuple[int, int, int, np.ndarray]], list[int], list[int]]:
    """Place matching motifs so their scan order defines a nontrivial output permutation."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    match_count = len(block_colors)
    matching_slots = sorted(random.sample(slots, match_count))
    ordered_colors = _non_identity_permutation(block_colors)

    payloads: list[tuple[int, int, int, np.ndarray]] = []
    used_slots: set[tuple[int, int]] = set()
    for (sr, sc), color in zip(matching_slots, ordered_colors):
        payloads.append((sr, sc, int(color), motif_masks[int(color)]))
        used_slots.add((sr, sc))

    remaining_slots = [slot for slot in slots if slot not in used_slots]
    random.shuffle(remaining_slots)
    for (sr, sc), color in zip(remaining_slots, distractor_colors):
        payloads.append((sr, sc, int(color), _random_mask()))

    rank_by_color = {int(color): int(idx + 1) for idx, color in enumerate(ordered_colors)}
    ranks = [rank_by_color[int(color)] for color in block_colors]
    return payloads, ordered_colors, ranks


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    payloads: list[tuple[int, int, int, np.ndarray]],
) -> None:
    """Paint the chosen motif payloads into the visible motif slot grid."""
    for sr, sc, color, mask in payloads:
        _paint_motif(
            grid,
            motif_r + int(sr) * MOTIF_STEP,
            motif_c + int(sc) * MOTIF_STEP,
            int(color),
            mask,
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    placement: tuple[int, int, int, int],
    block_colors: list[int],
    ranks: list[int],
    marker_color: int,
    motif_payloads: list[tuple[int, int, int, np.ndarray]],
) -> np.ndarray:
    """Render the sampled destination blocks, markers, and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, ranks, marker_color)
    _paint_motif_slots(inp, motif_r, motif_c, motif_payloads)
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
    motif_payloads, ordered_colors, ranks = _plan_motif_payloads(
        Mr, Mc, block_colors, distractor_colors, motif_masks
    )

    expected = _render_expected(bg, R, C, ordered_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        placement,
        block_colors,
        ranks,
        marker_color,
        motif_payloads,
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

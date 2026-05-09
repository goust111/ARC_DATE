# Combo variant of 58f5dbd5 (index 27)
# combo_spec: D1/3+D15/0
# Name: 排列匹配+迭代雕刻

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    label_mask,
    paste,
)


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   core_change:
#     old: a solid 5x5 block matched exactly one motif by shared color, then that one
#       connected motif mask was centered and carved once.
#     new: every 5x5 block already contains two internal marker holes, and matching is
#       based on the arrangement of those holes rather than color. Each matched block then
#       receives two carving rounds from the two motif slots that share the same layout.
#   concretization:
#     - Block colors remain distinct so the packed output is easy to read, but motif colors
#       are drawn from a separate small palette and intentionally reused across different
#       layouts, making color useless for matching.
#     - The identifying layout is restricted to two internal markers inside the centered
#       3x3 zone of a block. This keeps the arrangement visible while leaving enough 3x3
#       mask degrees of freedom for later carving rounds to add new holes.
#     - Each motif slot is a full 3x3 sprite window whose foreground splits into exactly
#       two 8-disconnected parts. The sorted top-left anchors of those parts define the
#       same layout signature used by a block's preset holes.
#     - A block layout is unique within a sample and appears on exactly two motif slots,
#       so the solver groups motifs by arrangement and applies both in reading order.
#   behavior_trace:
#     - Original path: block color X -> find same-colored connected motif -> center tight
#       mask once -> carve.
#     - Combo path: block color X is irrelevant for matching -> read the block's two-hole
#       layout -> find both 3x3 motif slots whose two-part arrangement matches that layout
#       -> carve the first slot, then carve the second slot onto the already holed tile.
#     - Difference: matching now requires spatial layout reasoning, and every matched block
#       accumulates holes from multiple motifs instead of one.
#   complexity_calibration:
#     - Two marker holes are enough to create four clear, human-visible arrangements inside
#       the 3x3 center, which supports up to four destination blocks without ambiguity.
#     - The generator fixes two rounds per block. That is enough to make the multi-step
#       behavior visible while avoiding a crowded motif region full of disconnected pieces.
#     - Distractors are omitted on purpose: disconnected multi-part motifs plus arrangement
#       matching already raise the reasoning load, so extra decoys would add clutter faster
#       than they add insight.
#   implicit_assumptions_review:
#     - The original `_is_solid_block` assumption is invalid because destination blocks are
#       no longer solid. The new detector must accept 5x5 color components with border
#       intact and exactly two internal single-cell holes.
#     - The original motif extraction by monochrome connected-components is invalid because
#       one motif is now intentionally disconnected. The solver therefore recovers the motif
#       region as a full slot lattice and reads each 3x3 slot as one object.
#     - The original color-based lookup is invalid because motif colors do not correspond to
#       block colors. The new matching key is the arrangement signature only.
#     - The original single-stencil carve is invalid because each block has two matched
#       motif slots. The carving code must preserve prior holes and apply both masks.
#   placement_legality:
#     - Destination blocks still use a 6-step lattice (5 body + 1 gap), and motif slots
#       still use a 4-step lattice (3 body + 1 gap). Regions are separated along one axis.
#     - All motif slots are occupied and every motif touches all four edges of its 3x3 slot,
#       so the motif region bbox exposes the full slot lattice without needing hidden data.
#     - Block marker holes live strictly inside the centered 3x3, so the 5x5 border remains
#       solid and block detection is stable.
#   visual_observability:
#     - A human can see that block colors do not appear on motifs, so color matching fails.
#     - The two preset block holes visibly mirror the arrangement of the two disconnected
#       motif parts, making the new correspondence rule explicit across examples.
#     - Each matched layout appears twice in the motif region, and the final tile contains
#       more holes than a one-motif carve, making the iterative aspect visible as well.
#   cleanliness:
#     - The scene still has only two clean regions: a block region and a motif-slot region.
#       No extra markers, legends, or numeric encodings are introduced.
# new_main_assumptions_and_generator_guarantees:
#   - Every block has one of four allowed two-hole signatures, and those signatures are
#     unique among blocks in the same sample.
#   - Every motif slot contains exactly one monochrome 3x3 sprite with exactly two
#     disconnected foreground parts and a signature from the same allowed set.
#   - Each block signature appears on exactly two motif slots.
#   - For every block, the second carving round removes at least one cell that the first
#     round did not remove, so the multi-step behavior never collapses to a one-round task.


CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_LAYOUTS = {
    3: [(2, 3), (3, 2)],
    4: [(2, 4), (4, 2)],
}

ROUND_COUNT = 2
OUTPUT_BORDER = 1
OUTER_MARGIN = 1
REGION_GAP = 1
GENERATION_RETRIES = 400

SIGNATURES = [
    ((0, 0), (2, 2)),
    ((0, 2), (2, 0)),
    ((0, 0), (0, 2)),
    ((0, 0), (2, 0)),
]
SIGNATURE_SET = set(SIGNATURES)


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _component_anchor_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    """Return sorted component top-left anchors for a 3x3 motif mask."""
    lab, n = label_mask(mask.astype(bool), connectivity=8)
    if int(n) != 2:
        return None
    anchors: list[tuple[int, int]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            return None
        r0, c0 = pts.min(axis=0)
        anchors.append((int(r0), int(c0)))
    return tuple(sorted(anchors))


def _make_masks_by_signature() -> dict[tuple[tuple[int, int], ...], list[np.ndarray]]:
    """Enumerate 3x3 two-component masks keyed by their arrangement signature."""
    out = {sig: [] for sig in SIGNATURES}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        if int(mask.sum()) < 3:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        sig = _component_anchor_signature(mask)
        if sig in out:
            out[sig].append(mask.copy())
    return out


MASKS_BY_SIGNATURE = _make_masks_by_signature()


def _hole_signature_from_block(tile: np.ndarray, bg: int) -> tuple[tuple[int, int], ...] | None:
    """Read the two preset internal holes of a 5x5 block as a normalized signature."""
    lab, n = label_mask(tile == int(bg), connectivity=8)
    anchors: list[tuple[int, int]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == 4 or c1 == 4):
            return None
        if int(len(pts)) != 1:
            return None
        anchors.append((int(r0) - 1, int(c0) - 1))
    sig = tuple(sorted(anchors))
    return sig if sig in SIGNATURE_SET else None


def _is_block_component(comp: dict, grid: np.ndarray, bg: int) -> bool:
    """Return whether a component is a valid 5x5 destination block for this combo."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    tile = crop_bbox(grid, b)
    if not set(map(int, np.unique(tile))).issubset({int(bg), int(col)}):
        return False
    if not bool((tile[0, :] == col).all() and (tile[-1, :] == col).all()):
        return False
    if not bool((tile[:, 0] == col).all() and (tile[:, -1] == col).all()):
        return False
    return _hole_signature_from_block(tile, bg) is not None


def _carve_tile(tile: np.ndarray, motif_sequence: list[np.ndarray], bg: int) -> np.ndarray:
    """Apply the full 3x3 motif slot masks in sequence onto the block center."""
    carved = tile.copy()
    for mask in motif_sequence:
        carved[1:4, 1:4][mask] = int(bg)
    return carved


def _pack_tiles(blocks: list[dict[str, int]], tiles: dict[tuple[int, int], np.ndarray], bg: int) -> np.ndarray:
    """Pack carved 5x5 tiles preserving original row/column order."""
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for b in blocks:
        rr = int(1 + r_index[int(b["r0"])] * DST_BLOCK_STEP)
        cc = int(1 + c_index[int(b["c0"])] * DST_BLOCK_STEP)
        out[rr : rr + 5, cc : cc + 5] = tiles[(int(b["r0"]), int(b["c0"]))]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match disconnected motif layouts to pre-holed 5x5 blocks, iteratively carve both matched motifs, then pack the finished blocks.

    Input structure:
    - The dominant color is the background.
    - Several distinct-color 5x5 blocks appear in one region; each block already contains exactly two tiny background holes inside its centered 3x3 zone.
    - A separate motif region is a full lattice of occupied 3x3 slots; each slot contains one monochrome sprite split into exactly two disconnected parts.

    Rule:
    - Read each block's two internal holes as an arrangement signature.
    - Read each motif slot's two disconnected parts as the same kind of signature, and group motif slots by signature instead of by color.
    - For a block, take the two motif slots with the matching signature in reading order and carve both 3x3 masks into the block center, preserving previously carved holes; then pack all blocks in original row/column order with 1-cell gaps and a 1-cell border.

    Key insight:
    - The preset block holes are not output noise; they are the matching key that identifies which disconnected motif sequence belongs to that block, while repeated same-layout motifs mean carving must be accumulated over multiple rounds.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )

    blocks: list[dict[str, int]] = []
    block_mask = np.zeros_like(g, dtype=bool)
    for comp in comps:
        if not _is_block_component(comp, g, bg):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        sig = _hole_signature_from_block(crop_bbox(g, comp["bbox"]), bg)
        assert sig is not None
        blocks.append(
            {"col": int(comp["color"]), "r0": int(r0), "c0": int(c0), "sig": sig}
        )
        block_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert blocks

    block_sigs = [tuple(b["sig"]) for b in blocks]
    assert len(set(block_sigs)) == len(block_sigs)

    motif_only = g.copy()
    motif_only[block_mask] = int(bg)
    motif_pts = np.argwhere(motif_only != int(bg))
    assert motif_pts.size > 0
    mr0, mc0 = motif_pts.min(axis=0)
    mr1, mc1 = motif_pts.max(axis=0)
    motif_region = motif_only[int(mr0) : int(mr1) + 1, int(mc0) : int(mc1) + 1]

    mh, mw = motif_region.shape
    assert (int(mh) + 1) % MOTIF_STEP == 0
    assert (int(mw) + 1) % MOTIF_STEP == 0
    Mr = (int(mh) + 1) // MOTIF_STEP
    Mc = (int(mw) + 1) // MOTIF_STEP
    assert Mr * Mc == len(blocks) * ROUND_COUNT

    motifs_by_sig: dict[tuple[tuple[int, int], ...], list[tuple[int, int, np.ndarray]]] = {}
    for i in range(Mr):
        for j in range(Mc):
            rr = int(i * MOTIF_STEP)
            cc = int(j * MOTIF_STEP)
            patch = motif_region[rr : rr + MOTIF_SIZE, cc : cc + MOTIF_SIZE]
            mask = patch != int(bg)
            assert bool(mask.any())
            sig = _component_anchor_signature(mask)
            assert sig in SIGNATURE_SET
            motifs_by_sig.setdefault(sig, []).append((int(i), int(j), mask.copy()))

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        sig = tuple(b["sig"])
        seq_items = sorted(motifs_by_sig.get(sig, []), key=lambda item: (item[0], item[1]))
        assert len(seq_items) == ROUND_COUNT
        base_tile = crop_bbox(g, (int(b["r0"]), int(b["c0"]), int(b["r0"]) + 4, int(b["c0"]) + 4))
        carved[(int(b["r0"]), int(b["c0"]))] = _carve_tile(
            base_tile, [item[2] for item in seq_items], bg
        )

    return _pack_tiles(blocks, carved, bg)


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
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
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs_for_count(count: int) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for R, C in DST_LAYOUTS[count]:
        for Mr, Mc in MOTIF_LAYOUTS[count]:
            out.append((R, C, Mr, Mc))
    return out


def _choose_canvas_and_config() -> tuple[int, int, int, int, int, int, tuple[int, int, int, int]] | None:
    """Choose count first so 3-block and 4-block branches are both exercised regularly."""
    counts = sorted(set(DST_LAYOUTS) & set(MOTIF_LAYOUTS))
    random.shuffle(counts)
    for count in counts:
        configs = _configs_for_count(count)
        random.shuffle(configs)
        for R, C, Mr, Mc in configs:
            dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
            motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

            feasible_dims = [
                (H, W)
                for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1)
                for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1)
                if _can_split(H, W, dst_h, dst_w, motif_h, motif_w)
            ]
            if not feasible_dims:
                continue

            H, W = random.choice(feasible_dims)
            placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
            if placement is not None:
                return H, W, R, C, Mr, Mc, placement
    return None


def _choose_palette(bg: int, count: int) -> tuple[list[int], list[int]] | None:
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    if len(palette) < count + 2:
        return None
    return palette[:count], palette[count : count + 2]


def _make_block_tile(color: int, sig: tuple[tuple[int, int], ...], bg: int) -> np.ndarray:
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    for r, c in sig:
        tile[1 + int(r), 1 + int(c)] = int(bg)
    return tile


def _sample_sequence(sig: tuple[tuple[int, int], ...]) -> list[np.ndarray] | None:
    masks = MASKS_BY_SIGNATURE[sig]
    hole_mask = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    for r, c in sig:
        hole_mask[int(r), int(c)] = True
    for _ in range(200):
        first, second = random.sample(masks, 2)
        if np.array_equal(first, second):
            continue
        if not bool((first & ~second & ~hole_mask).any()):
            continue
        if not bool((second & ~first & ~hole_mask).any()):
            continue
        return [first.copy(), second.copy()]
    return None


def _sample_sequences(
    signatures: list[tuple[tuple[int, int], ...]]
) -> dict[tuple[tuple[int, int], ...], list[np.ndarray]] | None:
    out: dict[tuple[tuple[int, int], ...], list[np.ndarray]] = {}
    for sig in signatures:
        seq = _sample_sequence(sig)
        if seq is None:
            return None
        out[sig] = seq
    return out


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_specs: list[tuple[int, tuple[tuple[int, int], ...]]],
    motif_sequences: dict[tuple[tuple[int, int], ...], list[np.ndarray]],
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    for i in range(R):
        for j in range(C):
            color, sig = block_specs[i * C + j]
            tile = _make_block_tile(int(color), sig, bg)
            tile = _carve_tile(tile, motif_sequences[sig], bg)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _render_first_round_only(
    bg: int,
    R: int,
    C: int,
    block_specs: list[tuple[int, tuple[tuple[int, int], ...]]],
    motif_sequences: dict[tuple[tuple[int, int], ...], list[np.ndarray]],
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    for i in range(R):
        for j in range(C):
            color, sig = block_specs[i * C + j]
            tile = _make_block_tile(int(color), sig, bg)
            tile = _carve_tile(tile, [motif_sequences[sig][0]], bg)
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
    block_specs: list[tuple[int, tuple[tuple[int, int], ...]]],
    bg: int,
) -> None:
    for i in range(R):
        for j in range(C):
            color, sig = block_specs[i * C + j]
            tile = _make_block_tile(int(color), sig, bg)
            paste(grid, tile, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _choose_slot_payloads(
    signatures: list[tuple[tuple[int, int], ...]],
    motif_sequences: dict[tuple[tuple[int, int], ...], list[np.ndarray]],
    motif_colors: list[int],
) -> dict[int, tuple[int, np.ndarray, tuple[tuple[int, int], ...]]]:
    count = len(signatures)
    even_slots = list(range(0, 2 * count, 2))
    odd_slots = list(range(1, 2 * count, 2))
    random.shuffle(even_slots)
    random.shuffle(odd_slots)

    payloads: dict[int, tuple[int, np.ndarray, tuple[tuple[int, int], ...]]] = {}
    sig_order = signatures.copy()
    random.shuffle(sig_order)
    for sig in sig_order:
        slot_a = even_slots.pop()
        slot_b = odd_slots.pop()
        mask_a, mask_b = motif_sequences[sig]
        if random.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a
        payloads[int(slot_a)] = (int(motif_colors[0]), mask_a.copy(), sig)
        payloads[int(slot_b)] = (int(motif_colors[1]), mask_b.copy(), sig)
    return payloads


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    signatures: list[tuple[tuple[int, int], ...]],
    motif_sequences: dict[tuple[tuple[int, int], ...], list[np.ndarray]],
    motif_colors: list[int],
) -> None:
    payloads = _choose_slot_payloads(signatures, motif_sequences, motif_colors)
    assert len(payloads) == Mr * Mc
    for idx, (color, mask, _sig) in payloads.items():
        sr = idx // Mc
        sc = idx % Mc
        patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
        paste(grid, patch, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, mask=mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_specs: list[tuple[int, tuple[tuple[int, int], ...]]],
    motif_sequences: dict[tuple[tuple[int, int], ...], list[np.ndarray]],
    motif_colors: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_specs, bg)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        [sig for _color, sig in block_specs],
        motif_sequences,
        motif_colors,
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(COLOR_COUNT)

    choice = _choose_canvas_and_config()
    if choice is None:
        return None
    H, W, R, C, Mr, Mc, placement = choice

    count = R * C
    palette = _choose_palette(bg, count)
    if palette is None:
        return None
    block_colors, motif_colors = palette

    signatures = random.sample(SIGNATURES, count)
    motif_sequences = _sample_sequences(signatures)
    if motif_sequences is None:
        return None

    block_specs = list(zip(block_colors, signatures))
    expected = _render_expected(bg, R, C, block_specs, motif_sequences)
    baseline = _render_first_round_only(bg, R, C, block_specs, motif_sequences)
    if np.array_equal(expected, baseline):
        return None

    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_specs,
        motif_sequences,
        motif_colors,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

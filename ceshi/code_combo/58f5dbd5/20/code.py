# Combo variant of 58f5dbd5 (index 20)
# combo_spec: D2/0+D4/1+D13/1
# Name: 颜色频次消歧+大量干扰+更多方块

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   core_change:
#     The original task only needed color equality because each block color had one useful motif.
#     This combo scales the destination set to 8-9 blocks and turns the motif region into a large
#     same-color search problem. For every block color there is now a 5-motif pool:
#       3 copies of the true stencil shape + 2 singleton same-colored decoy shapes.
#     With 8-9 block colors, the ARC palette is nearly exhausted, so "many distractors" must be
#     realized mostly inside these same-color pools rather than by inventing many extra colors.
#     That creates stronger interference than the source task because color matching alone fails.
#   concrete_behavior:
#     - Destination blocks: either a 2x4 / 4x2 grid (8 blocks) or a 3x3 grid (9 blocks).
#     - Motif region: exactly 40 or 45 occupied 3x3 slots, arranged as a clean 5x8, 8x5, 5x9,
#       or 9x5 slot lattice depending on the sampled block count and canvas orientation.
#     - Each color pool visibly contains one repeated motif shape and two oddball same-colored
#       decoys. The output carves the repeated shape, not the smallest visible same-colored motif.
#   example_trace:
#     Suppose a sample has 9 blocks and the red color pool is:
#       [plus, plus, plus, small-L, bent-line]
#     - Original logic: scan red non-block components, sort by area, choose the smallest decoy,
#       so the red block would get the wrong hole pattern.
#     - Combo logic: group exact red masks, count frequencies, notice plus appears 3 times while
#       each decoy appears once, then carve the plus shape into the red 5x5 block.
#     - Difference: every color contributes 2 distractors, so a 9-block sample has 18 distractor
#       motifs versus 9 logical matches; the packed output still preserves block row/column order.
#   complexity_calibration:
#     Five motifs per color is intentionally moderate. It gives a clear 3-vs-1-vs-1 frequency
#     pattern and a 2x distractor budget without becoming visually chaotic. The repeated support
#     is enough for a human to infer "use the repeated same-colored shape" from a few examples.
#   hidden_assumptions_review:
#     - STEP 1 still assumes destination objects are solid 5x5 monochrome squares; generator
#       preserves that exactly.
#     - The original STEP 2 assumed at most one useful same-colored motif and picked by size.
#       That assumption is false now, so the solver must group by exact mask key and compare
#       frequencies inside each color pool.
#     - Exact mask equality is only meaningful if the same visual shape always yields the same
#       bbox-normalized boolean array. The generator enforces full 3x3 edge-touching masks so the
#       bbox is stable and there is no translation ambiguity inside a slot.
#     - STEP 4 packs blocks by unique source rows/cols, which assumes the blocks form a proper
#       rectangular lattice. The generator restricts layouts to 2x4, 4x2, or 3x3 to keep that
#       assumption true even after increasing the block count.
#     - Because distractors reuse block colors, color equality is no longer sufficient. The new
#       solver must actually exercise the frequency branch on every generated sample.
#   placement_legality:
#     - Blocks keep the original 5x5 size and 1-cell inter-block gaps.
#     - Motifs keep 3x3 slots with 1-cell slot gaps, preventing component merges even when many
#       same-colored motifs appear nearby.
#     - Destination and motif regions are placed by the original split-by-axis strategy with
#       explicit outer margins and a 1-cell region gap, so they never overlap or touch.
#     - The canvas is enlarged to 36..48 by 36..58, which comfortably fits the largest legal
#       combination: a 3x3 block region (17x17) plus a 45-slot motif region (19x35 or 35x19).
#   visual_observability:
#     - The new rule is visible: within each color pool, one exact motif shape appears three
#       times while two other same-colored shapes appear once each.
#     - No hidden ordering by color id or slot order is used. The decisive signal is the visible
#       repetition count of same-colored shapes.
#   visual_cleanliness:
#     - Even with 40-45 motifs, the slot lattice keeps the region legible.
#     - The dense interference lives in one separated motif region, so the block region remains
#       easy to parse and the output carving remains easy to verify.
#   color_inference:
#     - Colors never transform. They only partition candidate motif pools.
#     - Because almost all non-background colors may be consumed by the 8-9 blocks, same-color
#       decoys are the only scalable distractor type; this also makes the fusion behavior more
#       visible than adding one leftover distractor color.
#   behavior_metric:
#     - Per sample, distractor_to_logical_match_ratio = 2.0 because each color contributes two
#       decoy motifs for one logical stencil choice.
#     - Per color, most_frequent_mask_count = 3 and both decoy counts are 1, with winner area
#       strictly larger than each decoy area, so "most frequent" and "smallest area" disagree.
#   solver_generator_contract:
#     - The generator guarantees for every block color:
#         1) one winner mask repeated exactly 3 times,
#         2) two distinct decoy masks repeated once each,
#         3) winner area > each decoy area,
#         4) all masks fit in a full 3x3 bbox and are pairwise distinct.
#     - Therefore the new solver's frequency logic is necessary, and the old smallest-area solver
#       would produce a different carving for every generated color pool.


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


def _is_solid_block_component(grid: np.ndarray, comp: dict[str, object]) -> bool:
    """Return whether a component is one of the solid 5x5 destination blocks."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    return bool((crop_bbox(grid, b) == int(col)).all())


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Hashable key for exact motif-shape frequency counting."""
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve the repeated same-colored motifs into many solid blocks and repack the results.

    Input structure:
    - bg is the majority color.
    - There are 8 or 9 solid 5x5 monochrome blocks arranged as a rectangular lattice, each with
      a distinct non-bg color.
    - Elsewhere, every block color appears on a large pool of small same-colored 3x3-bounded
      motifs: one exact shape repeats several times while other same-colored shapes are decoys.

    Rule:
    - Detect the 5x5 blocks, then for each block color gather all non-block components of that
      same color and count exact motif-shape frequencies.
    - Use the most frequent same-colored motif as the stencil for that block color, center it
      inside the 5x5 square, and recolor the stencil cells to bg to carve holes.
    - Pack all carved blocks into a compact output grid that preserves the source row/column
      order, with 1-cell gaps between blocks and a 1-cell bg border around the whole result.

    Key insight:
    - Color only identifies the candidate pool; the actual carving stencil is the repeated motif
      shape inside that pool, so the solver must count same-colored shapes instead of choosing by
      size or by first occurrence.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        if not _is_solid_block_component(g, comp):
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, identify the stencil by same-color frequency.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        groups: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(bcol)],
            return_format="dict",
        )
        for comp in comps_col:
            if _is_solid_block_component(g, comp):
                continue

            r0, c0, r1, c1 = map(int, comp["bbox"])
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            key = _mask_key(mask)
            area = int(comp["area"])
            bbox_area = int(h * w)

            bucket = groups.setdefault(
                key,
                {
                    "count": 0,
                    "area": area,
                    "bbox_area": bbox_area,
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                },
            )
            bucket["count"] = int(bucket["count"]) + 1
            if (int(r0), int(c0)) < (int(bucket["r0"]), int(bucket["c0"])):
                bucket["r0"] = int(r0)
                bucket["c0"] = int(c0)

        assert groups
        best = max(
            groups.values(),
            key=lambda item: (
                int(item["count"]),
                -int(item["area"]),
                -int(item["bbox_area"]),
                -int(item["r0"]),
                -int(item["c0"]),
            ),
        )
        motifs[int(bcol)] = np.array(best["mask"], dtype=bool)

    # STEP 3) Carve each 5x5 tile by centering the chosen motif and painting it to bg.
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

    # STEP 4) Pack carved tiles into a compact grid preserving the original lattice order.
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
        rr = int(1 + r_index[int(r0)] * step)
        cc = int(1 + c_index[int(c0)] * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x-combo) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (36, 48)
CANVAS_WIDTH_RANGE = (36, 58)
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
    40: [(5, 8), (8, 5)],
    45: [(5, 9), (9, 5)],
}
WINNER_REPEAT_COUNT = 3
DECOY_REPEAT_COUNT = 1
DECOY_SHAPES_PER_COLOR = 2
WINNER_AREA_RANGE = (6, 8)
LOSER_AREA_RANGE = (4, 5)
MIN_MASK_XOR = 2

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

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


def _configs(H: int, W: int) -> list[tuple[int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        motif_count = count * (WINNER_REPEAT_COUNT + DECOY_SHAPES_PER_COLOR * DECOY_REPEAT_COUNT)
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for Mr, Mc in MOTIF_LAYOUTS[motif_count]:
                mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                if _can_split(H, W, dh, dw, mh, mw):
                    out.append((R, C, Mr, Mc))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int] | None:
    """Sample a feasible layout while keeping 8- and 9-block cases both reachable."""
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask(area_range: tuple[int, int]) -> np.ndarray:
    """Sample one motif mask from the precomputed masks within the given area range."""
    lo, hi = area_range
    area = random.randint(int(lo), int(hi))
    return random.choice(MASKS_BY_AREA[area]).copy()


def _mask_difference(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing cells between two equal-size boolean masks."""
    return int(np.count_nonzero(a.astype(bool) ^ b.astype(bool)))


def _choose_winner_and_decoys() -> tuple[np.ndarray, list[np.ndarray]]:
    """Pick one winning mask and two smaller decoy masks for a color pool."""
    for _ in range(400):
        winner = _random_mask(WINNER_AREA_RANGE)
        decoys: list[np.ndarray] = []
        ok = True
        for _k in range(DECOY_SHAPES_PER_COLOR):
            for _inner in range(200):
                decoy = _random_mask(LOSER_AREA_RANGE)
                if int(decoy.sum()) >= int(winner.sum()):
                    continue
                if _mask_difference(winner, decoy) < MIN_MASK_XOR:
                    continue
                if any(_mask_difference(decoy, prev) < MIN_MASK_XOR for prev in decoys):
                    continue
                if any(np.array_equal(decoy, prev) for prev in decoys):
                    continue
                decoys.append(decoy)
                break
            else:
                ok = False
                break
        if ok and len(decoys) == DECOY_SHAPES_PER_COLOR:
            return winner, decoys
    raise RuntimeError("failed to sample distinct winner/decoy motifs")


def _choose_palette(bg: int, count: int) -> list[int]:
    """Pick destination colors from the non-background ARC palette."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    assert count <= len(palette)
    return palette[:count]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], winner_masks: dict[int, np.ndarray]
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
            tile[motif_slice, motif_slice][winner_masks[color]] = bg
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
    winner_masks: dict[int, np.ndarray],
    decoy_masks: dict[int, list[np.ndarray]],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for the densely occupied motif region."""
    items: list[tuple[int, np.ndarray]] = []
    for color in block_colors:
        items.extend(
            (int(color), winner_masks[int(color)].copy())
            for _ in range(WINNER_REPEAT_COUNT)
        )
        for decoy in decoy_masks[int(color)]:
            items.extend((int(color), decoy.copy()) for _ in range(DECOY_REPEAT_COUNT))
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    winner_masks: dict[int, np.ndarray],
    decoy_masks: dict[int, list[np.ndarray]],
) -> None:
    """Randomly assign repeated winners and singleton decoys to the motif slot lattice."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, winner_masks, decoy_masks)
    assert len(slot_items) == len(slots)

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
    winner_masks: dict[int, np.ndarray],
    decoy_masks: dict[int, list[np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, winner_masks, decoy_masks
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
    R, C, Mr, Mc = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors = _choose_palette(bg, count)
    winner_masks: dict[int, np.ndarray] = {}
    decoy_masks: dict[int, list[np.ndarray]] = {}
    for color in block_colors:
        winner, decoys = _choose_winner_and_decoys()
        winner_masks[int(color)] = winner
        decoy_masks[int(color)] = decoys

    expected = _render_expected(bg, R, C, block_colors, winner_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, winner_masks, decoy_masks, placement
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

# Combo variant of 58f5dbd5 (index 9)
# combo_spec: D1/0+D5/2
# Name: 形状嵌合匹配+预设图案方块

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste


# Design analysis
# a) Concrete behavior and trace:
#    - Base task: find a solid 5x5 monochrome block, locate the small motif with the same color, then carve that
#      motif into the block center in the packed output.
#    - Fused task: each destination block is now a two-color 5x5 preset pattern, and the input block already has
#      one bg notch cut into a side-centered 3x3 slot. The correct motif is the small sprite whose exact shape
#      would fill that notch, regardless of motif color. Example:
#        input block = checkerboard tile using red/yellow, with a top-edge notch shaped like
#          111
#          010
#          011
#        matching motif elsewhere = blue sprite with that exact occupied mask.
#      The output first restores the full checkerboard tile, then carves the matched motif into the center 3x3
#      area, so the visible hole cuts across whichever pattern colors lie under it.
#    - The two edits are coupled: shape decides which motif belongs to a block, while the block's own stored
#      pattern decides what survives around the carving. If either piece is ignored, the output is wrong.
# b) Complexity calibration:
#    - Block patterns come from a small preset library rather than arbitrary art. This keeps the rule learnable
#      from a few examples, while still making the blocks visibly non-solid.
#    - Real notch masks are unique within a sample, and distractor motifs use other unique masks. Matching is
#      therefore explicit shape comparison instead of an ambiguous search.
#    - Matching motif colors are a derangement of the blocks' primary colors, so the original color-key shortcut
#      is actively misleading rather than merely absent.
# c) Implicit assumption review against the original main():
#    - Original block detection assumed one 5x5 monochrome connected component per block. This breaks completely:
#      patterned blocks split into many monochrome pieces. The solver must instead scan 5x5 windows and verify
#      them against known pattern templates plus one legal edge notch.
#    - Original motif lookup indexed by block color. This no longer holds, so motifs are indexed by normalized
#      binary shape keys after excluding any monochrome components that lie inside detected block bboxes.
#    - Original carving started from a solid-color tile. The fused version must reconstruct the intact preset
#      pattern first and only then overwrite the matched center cells with bg.
# d) Placement legality:
#    - Blocks still sit on the original 6-step lattice (5 body + 1 gap), and motif slots still use the original
#      4-step lattice. The two regions are placed with a separating gap along one axis.
#    - Each notch is confined to one side-centered 3x3 window. This guarantees in-bounds carving, a clear visual
#      "missing piece", and a finite set of candidate locations for the solver.
# e) Visual observability:
#    - Humans can see the missing notch in each patterned block and can compare that silhouette against the motif
#      sprites. The correct match is therefore exposed as a visible shape-fit problem.
#    - The output visibly preserves stripes/checkers/X-like patterns around the hole, making the D5 change clear:
#      the carving removes parts of an existing pattern instead of punching a uniform color square.
# f) Visual cleanliness:
#    - The scene still has only two separated object families: 5x5 notched patterned blocks and isolated 3x3
#      motif sprites. No extra markers or hidden bookkeeping are introduced.
#    - Using one shared accent color across the block region makes the pattern family read coherently, while the
#      distinct primary colors keep neighboring blocks easy to segment by eye.
# g) Rule inferability and color audit:
#    - Colors do not encode the match. They only paint the preset tile patterns and motif sprites.
#    - The useful structure is spatial: side-notch shape identifies the motif; the observed tile pattern tells the
#      solver what to restore before centered carving. This is recoverable from 2-4 ARC examples.
# h) New solver assumptions and generator guarantees:
#    - Solver assumes each valid block window uses exactly two non-bg colors plus one legal notch mask in a legal
#      side-centered slot. Generator constructs every block that way and rejects any sample whose windows become
#      ambiguous under the solver's detector.
#    - Solver assumes each true notch shape appears among motif candidates and no occupied motif shape is reused.
#      Generator samples pairwise-distinct masks for all visible motifs and places one matching sprite per block.
#    - Solver assumes motifs can be isolated by removing components that intersect block bboxes. Generator keeps
#      destination and motif regions spatially separated, so no true motif touches a block window.


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


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given coords."""
    out = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = True
    return out


def _tight_mask_from_coords(coords) -> np.ndarray:
    """Normalize absolute or relative coords into a tight boolean mask."""
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    rel = [(int(r) - r0, int(c) - c0) for r, c in coords]
    return mask_from_coords(rel, (int(r1 - r0 + 1), int(c1 - c0 + 1)))


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    """Hashable key for exact shape matching."""
    h, w = map(int, mask.shape)
    bits = tuple(int(v) for v in mask.astype(int).ravel())
    return h, w, bits


def _build_pattern_templates() -> list[np.ndarray]:
    """Preset 5x5 binary templates used by patterned destination blocks."""
    rr, cc = np.indices((DST_BLOCK_SIZE, DST_BLOCK_SIZE))
    return [
        ((rr + cc) % 2 == 0),
        (cc % 2 == 0),
        (rr % 2 == 0),
        ((rr == cc) | (rr + cc == DST_BLOCK_SIZE - 1)),
    ]


PATTERN_TEMPLATES = [p.astype(bool) for p in _build_pattern_templates()]


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
ALL_MASKS = [m for area in sorted(MASKS_BY_AREA) for m in MASKS_BY_AREA[area]]


def _is_valid_motif_mask(mask: np.ndarray) -> bool:
    """Return whether a boolean mask is one of the legal 3x3 motif/notch shapes."""
    if tuple(mask.shape) != (MOTIF_SIZE, MOTIF_SIZE):
        return False
    area = int(mask.sum())
    if not (MOTIF_AREA_RANGE[0] <= area <= MOTIF_AREA_RANGE[1]):
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    return bool(is_contiguous(mask.astype(int), background=0, connectivity=8))


def _render_pattern_tile(primary: int, accent: int, pattern: np.ndarray) -> np.ndarray:
    """Build one intact 5x5 patterned block from its two colors and template."""
    return np.where(pattern, int(accent), int(primary)).astype(int)


def _identify_block_patch(patch: np.ndarray, bg: int) -> dict[str, object] | None:
    """Recognize one legal 5x5 patterned block window and recover its intact tile plus notch shape."""
    if tuple(patch.shape) != (DST_BLOCK_SIZE, DST_BLOCK_SIZE):
        return None

    colors = sorted(int(c) for c in np.unique(patch) if int(c) != int(bg))
    if len(colors) != 2:
        return None

    bg_mask = (patch == int(bg))
    if not bool(bg_mask.any()):
        return None

    matches: list[dict[str, object]] = []
    for pattern_id, pattern in enumerate(PATTERN_TEMPLATES):
        for primary, accent in itertools.permutations(colors, 2):
            full_tile = _render_pattern_tile(primary, accent, pattern)
            for nr, nc in NOTCH_OFFSETS:
                slot_mask = np.zeros((DST_BLOCK_SIZE, DST_BLOCK_SIZE), dtype=bool)
                slot_mask[nr : nr + MOTIF_SIZE, nc : nc + MOTIF_SIZE] = True
                if bool(bg_mask[~slot_mask].any()):
                    continue
                if not bool(np.array_equal(patch[~slot_mask], full_tile[~slot_mask])):
                    continue

                slot = patch[nr : nr + MOTIF_SIZE, nc : nc + MOTIF_SIZE]
                full_slot = full_tile[nr : nr + MOTIF_SIZE, nc : nc + MOTIF_SIZE]
                notch = (slot == int(bg))
                if not _is_valid_motif_mask(notch):
                    continue
                if not bool(np.array_equal(slot[~notch], full_slot[~notch])):
                    continue

                matches.append(
                    {
                        "pattern_id": int(pattern_id),
                        "primary": int(primary),
                        "accent": int(accent),
                        "tile": full_tile,
                        "notch": notch.copy(),
                    }
                )

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match motifs to patterned block notches by exact shape fit, then carve centered copies into restored pattern tiles.

    Input structure:
    - bg is the most common color.
    - Several 5x5 destination blocks form a separated region; each block uses one preset two-color pattern and also contains one bg notch cut into a side-centered 3x3 slot.
    - Elsewhere are small monochrome motif sprites. The useful cue is shape: the correct motif is the one whose tight mask exactly fills a block's notch, regardless of motif color.

    Rule:
    - Detect each legal 5x5 patterned block window, recover its intact preset tile, and extract the notch mask.
    - Find the motif with the same normalized shape, then carve that matched mask into the center of the restored patterned tile.
    - Pack the carved 5x5 tiles into a compact output grid preserving the original row/column order, using 1-cell gaps and a 1-cell outer border.

    Key insight: The edge notch is only a matching socket. Once the fitting motif shape is identified, the solver discards the notch position, rebuilds the original pattern, and reuses the shape as a centered carving stencil.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    # STEP 1) Detect destination blocks by scanning for legal 5x5 patterned windows with one side-notch.
    blocks: list[dict[str, object]] = []
    block_occupancy = np.zeros((H, W), dtype=bool)
    for r0 in range(H - DST_BLOCK_SIZE + 1):
        for c0 in range(W - DST_BLOCK_SIZE + 1):
            patch = g[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE]
            info = _identify_block_patch(patch, bg)
            if info is None:
                continue
            info["r0"] = int(r0)
            info["c0"] = int(c0)
            blocks.append(info)
            block_occupancy[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE] = True

    assert blocks

    # STEP 2) Index motif candidates by tight shape, ignoring any monochrome fragments that belong to blocks.
    motifs: dict[tuple[int, int, tuple[int, ...]], np.ndarray | None] = {}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        coords = comp["coords"]
        if bool(block_occupancy[coords[:, 0], coords[:, 1]].any()):
            continue
        mask = _tight_mask_from_coords(coords)
        if not _is_valid_motif_mask(mask):
            continue
        key = _mask_key(mask)
        if key in motifs:
            motifs[key] = None
        else:
            motifs[key] = mask

    # STEP 3) Restore each block's full pattern and carve the matched motif into the center.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        tile = np.array(block["tile"], dtype=int).copy()
        notch = np.array(block["notch"], dtype=bool)
        matched = motifs.get(_mask_key(notch))
        assert matched is not None
        mh, mw = map(int, matched.shape)
        off_r = (DST_BLOCK_SIZE - mh) // 2
        off_c = (DST_BLOCK_SIZE - mw) // 2
        for rr, cc in np.argwhere(matched):
            tile[int(off_r + rr), int(off_c + cc)] = int(bg)
        carved[(int(block["r0"]), int(block["c0"]))] = tile

    # STEP 4) Pack carved tiles into the compact output grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * DST_BLOCK_SIZE + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * DST_BLOCK_SIZE + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for block in blocks:
        i = int(r_index[int(block["r0"])])
        j = int(c_index[int(block["c0"])])
        rr = int(1 + i * DST_BLOCK_STEP)
        cc = int(1 + j * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[
            (int(block["r0"]), int(block["c0"]))
        ]

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
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for filled in range(min_filled, max_filled + 1):
                distractors = filled - count
                if not (0 <= distractors <= count):
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


def _choose_unique_masks(count: int) -> list[np.ndarray]:
    """Sample pairwise-distinct motif masks."""
    return [mask.copy() for mask in random.sample(ALL_MASKS, count)]


def _choose_diverse_patterns(count: int) -> list[np.ndarray]:
    """Choose block patterns with guaranteed within-sample variety."""
    template_ids = list(range(len(PATTERN_TEMPLATES)))
    random.shuffle(template_ids)
    chosen = template_ids[: min(count, len(template_ids))]
    while len(chosen) < count:
        chosen.append(random.choice(template_ids))
    random.shuffle(chosen)
    return [PATTERN_TEMPLATES[idx].copy() for idx in chosen]


def _choose_diverse_notch_offsets(count: int) -> list[tuple[int, int]]:
    """Choose notch sides with guaranteed within-sample variety."""
    offsets = list(NOTCH_OFFSETS)
    random.shuffle(offsets)
    chosen = offsets[: min(count, len(offsets))]
    while len(chosen) < count:
        chosen.append(random.choice(offsets))
    random.shuffle(chosen)
    return [tuple(offset) for offset in chosen]


def _choose_block_colors(bg: int, count: int) -> tuple[int, list[int]]:
    """Pick one shared accent plus distinct primary colors for the destination blocks."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    accent = int(palette[0])
    primary = [int(c) for c in palette[1 : 1 + count]]
    return accent, primary


def _deranged_match_colors(block_colors: list[int]) -> list[int]:
    """Give true matching motifs colors that never equal their own block primary color."""
    shift = random.randint(1, len(block_colors) - 1)
    return [int(block_colors[(i + shift) % len(block_colors)]) for i in range(len(block_colors))]


def _choose_distractor_colors(bg: int, accent: int, block_colors: list[int], count: int) -> list[int]:
    """Pick distractor motif colors, reusing the non-bg palette when necessary."""
    if count <= 0:
        return []
    preferred = [
        c
        for c in range(COLOR_COUNT)
        if c != bg and c != accent and c not in set(block_colors)
    ]
    fallback = [c for c in range(COLOR_COUNT) if c != bg]
    pool = preferred if preferred else fallback
    return [int(random.choice(pool)) for _ in range(count)]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif through its mask into the grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    accent: int,
    block_colors: list[int],
    block_patterns: list[np.ndarray],
    block_masks: list[np.ndarray],
) -> np.ndarray:
    """Render the oracle output from restored patterned tiles plus centered carvings."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            tile = _render_pattern_tile(block_colors[idx], accent, block_patterns[idx])
            tile[
                MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
                MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
            ][block_masks[idx]] = int(bg)
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
    accent: int,
    block_colors: list[int],
    block_patterns: list[np.ndarray],
    block_masks: list[np.ndarray],
    notch_offsets: list[tuple[int, int]],
) -> None:
    """Paint the notched patterned 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            block = _render_pattern_tile(block_colors[idx], accent, block_patterns[idx])
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
    """Build the shuffled motif payloads for the occupied motif slots."""
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
    """Randomly assign matching and distractor motifs to visible slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    items = _slot_items(match_colors, block_masks, distractor_colors, distractor_masks)
    for (sr, sc), (color, mask) in zip(slots, items):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    accent: int,
    block_colors: list[int],
    match_colors: list[int],
    distractor_colors: list[int],
    block_patterns: list[np.ndarray],
    block_masks: list[np.ndarray],
    distractor_masks: list[np.ndarray],
    notch_offsets: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled patterned blocks and motif sprites into one input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(
        inp,
        bg,
        dst_r,
        dst_c,
        R,
        C,
        accent,
        block_colors,
        block_patterns,
        block_masks,
        notch_offsets,
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


def _tile_uses_both_colors(tile: np.ndarray, bg: int) -> bool:
    """Return whether the tile still visibly contains both non-bg pattern colors."""
    colors = {int(c) for c in np.unique(tile) if int(c) != int(bg)}
    return len(colors) == 2


def _carve_covers_both_pattern_colors(tile: np.ndarray, mask: np.ndarray) -> bool:
    """Return whether the centered carving mask crosses both colors of the intact pattern tile."""
    center = tile[
        MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
        MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
    ]
    covered = {int(center[r, c]) for r, c in np.argwhere(mask)}
    return len(covered) == 2


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle or fusion checks disagree."""
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
    accent, block_colors = _choose_block_colors(bg, count)
    match_colors = _deranged_match_colors(block_colors)
    distractor_colors = _choose_distractor_colors(bg, accent, block_colors, filled - count)
    sampled_masks = _choose_unique_masks(filled)
    block_masks = sampled_masks[:count]
    distractor_masks = sampled_masks[count:]
    block_patterns = _choose_diverse_patterns(count)
    notch_offsets = _choose_diverse_notch_offsets(count)

    expected = _render_expected(bg, R, C, accent, block_colors, block_patterns, block_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        accent,
        block_colors,
        match_colors,
        distractor_colors,
        block_patterns,
        block_masks,
        distractor_masks,
        notch_offsets,
        placement,
    )

    # Reject any sample that visually degenerates back toward solid-block behavior.
    for idx in range(count):
        base_tile = _render_pattern_tile(block_colors[idx], accent, block_patterns[idx])
        notched_tile = base_tile.copy()
        nr, nc = notch_offsets[idx]
        notched_tile[nr : nr + MOTIF_SIZE, nc : nc + MOTIF_SIZE][block_masks[idx]] = int(bg)
        carved_tile = base_tile.copy()
        carved_tile[
            MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
            MOTIF_OFFSET_IN_BLOCK : MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE,
        ][block_masks[idx]] = int(bg)
        if not (
            _tile_uses_both_colors(notched_tile, bg)
            and _tile_uses_both_colors(carved_tile, bg)
            and _carve_covers_both_pattern_colors(base_tile, block_masks[idx])
        ):
            return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

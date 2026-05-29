# Combo variant of 58f5dbd5 (index 33)
# combo_spec: D3/2+D11/2
# Name: 5x5大motif+3x3方块

from __future__ import annotations

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
    """Build a boolean mask with True at the given in-bounds coordinates."""
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


# Design analysis for the fused D3/2 + D11/2 variant
# ---------------------------------------------------
# a) Concrete behavior trace
# - Original 58f5dbd5:
#   a same-colored 3x3-bounded connected motif was centered into a solid 5x5 block,
#   so the whole motif footprint fit inside the block and carved holes directly.
# - Fused variant here:
#   the source motif now has an exact 5x5 bbox, while the destination block shrinks
#   to a solid 3x3 square. Centering a 5x5 mask onto a 3x3 tile means the mask is
#   not rescaled or clipped manually; it is placed at offset (-1, -1), so only the
#   motif's central 3x3 window overlaps the tile and can carve holes.
# - Concrete trace:
#   suppose a red 5x5 motif has a plus-shaped 3x3 core and extra red border arms on
#   all four sides. The old task could never show that source geometry. Under the new
#   rule, the visible 5x5 sprite is still matched by color to the red 3x3 block, but
#   the output only reflects the overlapping central 3x3 core. The outer ring stays
#   important as input evidence even though it lies outside the carved tile.
#
# b) Complexity calibration
# - If motifs were allowed to be loose <=5x5 bboxes, many samples would visually look
#   like ordinary 3x3 motifs floating inside larger empty slots. To keep D3/2 active,
#   every usable motif is forced to occupy an exact 5x5 bbox with colored pixels on
#   all four sides.
# - If the central 3x3 overlap were too unconstrained, examples could become noisy or
#   ambiguous. The generator therefore reuses the original task's 3x3 connected-core
#   vocabulary (areas 4..8) and expands that core into 5x5 by attaching border arms.
# - Object counts remain in the original 3/4/6 family, with 6..8 visible motif slots,
#   so the scene still has repeated support instead of one-off geometry.
#
# c) Implicit-assumption audit of the original main()
# - Old STEP 1 assumed "destination = solid 5x5 square". Broken: it must now detect
#   solid 3x3 squares.
# - Old STEP 2 treated any same-colored non-5x5 component as the motif. Broken: with
#   this combo the intended motif is specifically a connected same-colored component
#   whose bbox is exactly 5x5.
# - Old STEP 3 implicitly assumed the motif fully fits after centering. Broken: here
#   the centered offset is negative, so carving must rely on ordinary bounds checks.
#   That clipping behavior is the whole point of the fusion.
# - Old STEP 4 hard-coded 5x5 tiles and pack step 6. Broken: packed tiles are now 3x3
#   with step 4.
#
# d) Placement legality
# - Destination blocks are placed on a regular lattice with top-left step 4
#   (3 body + 1 gap), so 3x3 solids never touch.
# - Motifs are placed on a separate 5x5 slot lattice with top-left step 6
#   (5 body + 1 gap), so even large motifs stay disconnected.
# - The two regions are separated on at least one axis with 1-cell outer margins and a
#   1-cell gap, preventing overlap, same-color merging, and out-of-bounds painting.
#
# e) Visual observability
# - The input remains human-readable: one region shows tiny solid 3x3 blocks, another
#   region shows visibly larger 5x5 same-colored motifs. The size contrast itself is
#   the new clue.
# - The output is also visibly different from the original task because packed tiles are
#   now 3x3 rather than 5x5, and the carved shape can only come from the motif's center.
# - No invisible property such as color id ordering is used. Matching stays purely by
#   visible same-color correspondence.
#
# f) Visual cleanliness
# - Only two structured regions appear in each input: the block grid and the motif-slot
#   grid. There is no overlap and no extra markup.
# - Distinct block colors and a small number of distractor motifs keep the scene tidy.
#
# g) Fusion metric used after implementation
# - For each generated sample we can measure:
#   1) every destination block bbox is exactly 3x3,
#   2) every matching motif bbox is exactly 5x5,
#   3) each matching motif has strictly positive active pixels outside its central 3x3.
#   Metric (3) proves the source motif is genuinely larger than the acting overlap and
#   that the 5x5 witness is not collapsing into a plain 3x3 sprite.
#
# h) New solver assumptions and generator guarantees
# - `main()` now assumes each true destination object is a solid 3x3 monochrome square.
# - It assumes each block color has exactly one same-colored non-block component with a
#   5x5 bbox; distractor colors are disjoint from block colors so matching is unique.
# - It assumes row/column order can be read from block top-left coordinates because the
#   generator still uses a clean rectangular lattice for destination placement.


COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 3
DST_BLOCK_STEP = 4
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif geometry.
MOTIF_SIZE = 5
MOTIF_STEP = 6
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
MOTIF_AREA_RANGE = (4, 8)  # area of the active central 3x3 core
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2
CORE_SLICE = slice(1, 4)

# Canvas and packing.
CANVAS_HEIGHT_RANGE = (25, 30)
CANVAS_WIDTH_RANGE = (25, 30)
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _is_solid_block(grid: np.ndarray, comp: dict[str, object]) -> bool:
    """Return whether a component is a solid 3x3 monochrome destination block."""
    bbox = comp["bbox"]
    if bbox_h(bbox) != DST_BLOCK_SIZE or bbox_w(bbox) != DST_BLOCK_SIZE:
        return False
    color = int(comp["color"])
    return bool((crop_bbox(grid, bbox) == color).all())


def _extract_mask(comp: dict[str, object]) -> np.ndarray:
    """Return the component mask normalized to its bbox."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return mask_from_coords(rel_coords, (h, w))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve the central overlap of large same-colored motifs into tiny solid blocks and repack them.

    Input structure:
    - The background is the most common color.
    - Several solid 3x3 monochrome blocks appear in a rectangular grid, each with a distinct non-background color.
    - Elsewhere, each block color has one connected same-colored motif whose bbox is exactly 5x5; optional distractor motifs use other colors.

    Rule:
    - Match every 3x3 block with the same-colored 5x5 motif.
    - Center the 5x5 motif on the 3x3 block without rescaling; because the motif is larger, only its overlapping central 3x3 window can affect the block, and those covered cells become background.
    - Pack the carved 3x3 blocks into a compact output that preserves the input row/column order, with 1-cell gaps and a 1-cell background border.

    Key insight:
    - The source motif is deliberately larger than the carrier tile. The solver should not shrink the motif; it simply centers the 5x5 mask and lets ordinary clipping keep only the central 3x3 overlap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 3x3 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        if not _is_solid_block(g, comp):
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the same-colored non-block component whose bbox is exactly 5x5.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(block["col"]) for block in blocks}
    for color in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        for comp in comps_col:
            if _is_solid_block(g, comp):
                continue
            bbox = comp["bbox"]
            h = int(bbox_h(bbox))
            w = int(bbox_w(bbox))
            if h != MOTIF_SIZE or w != MOTIF_SIZE:
                continue
            r0, c0, _r1, _c1 = map(int, bbox)
            mask = _extract_mask(comp)
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            motifs[int(color)] = candidates[0][4]

    assert len(motifs) == len(block_colors)

    # STEP 3) Carve each 3x3 tile by centering the 5x5 motif and letting clipping
    # keep only the overlapped central 3x3 footprint.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        color = int(block["col"])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
        motif = motifs[int(color)]
        mh, mw = map(int, motif.shape)
        off_r = (DST_BLOCK_SIZE - mh) // 2
        off_c = (DST_BLOCK_SIZE - mw) // 2
        for rr, cc in np.argwhere(motif):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                tile[tr, tc] = int(bg)
        carved[(int(block["r0"]), int(block["c0"]))] = tile

    # STEP 4) Pack carved 3x3 tiles into a compact grid.
    rs = sorted({int(block["r0"]) for block in blocks})
    cs = sorted({int(block["c0"]) for block in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * DST_BLOCK_SIZE + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * DST_BLOCK_SIZE + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for block in blocks:
        rr = int(OUTPUT_BORDER + r_index[int(block["r0"])] * DST_BLOCK_STEP)
        cc = int(OUTPUT_BORDER + c_index[int(block["c0"])] * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[
            (int(block["r0"]), int(block["c0"]))
        ]

    return out


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _make_core_masks() -> list[np.ndarray]:
    """Enumerate valid connected 3x3 core masks with exact 3x3 bbox."""
    min_area, max_area = MOTIF_AREA_RANGE
    out: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in itertools.product([False, True], repeat=9):
        core = np.array(bits, dtype=bool).reshape(3, 3)
        area = int(core.sum())
        if not (min_area <= area <= max_area):
            continue
        if not (core[0].any() and core[-1].any() and core[:, 0].any() and core[:, -1].any()):
            continue
        if not is_contiguous(core.astype(int), background=0, connectivity=8):
            continue
        out[_canonical_mask(core)] = core
    return list(out.values())


CORE_MASKS = _make_core_masks()


def _expand_core_to_large_motif(core: np.ndarray) -> np.ndarray:
    """Embed a 3x3 core into a connected 5x5 motif by attaching visible border arms."""
    motif = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    motif[CORE_SLICE, CORE_SLICE] = core

    top_cols = [int(c + 1) for c in np.where(core[0])[0]]
    bottom_cols = [int(c + 1) for c in np.where(core[2])[0]]
    left_rows = [int(r + 1) for r in np.where(core[:, 0])[0]]
    right_rows = [int(r + 1) for r in np.where(core[:, 2])[0]]

    top_c = random.choice(top_cols)
    bottom_c = random.choice(bottom_cols)
    left_r = random.choice(left_rows)
    right_r = random.choice(right_rows)

    motif[0, top_c] = True
    motif[4, bottom_c] = True
    motif[left_r, 0] = True
    motif[right_r, 4] = True

    for dc in (-1, 1):
        if random.random() < 0.5 and 1 <= top_c + dc <= 3:
            motif[0, top_c + dc] = True
        if random.random() < 0.5 and 1 <= bottom_c + dc <= 3:
            motif[4, bottom_c + dc] = True
    for dr in (-1, 1):
        if random.random() < 0.5 and 1 <= left_r + dr <= 3:
            motif[left_r + dr, 0] = True
        if random.random() < 0.5 and 1 <= right_r + dr <= 3:
            motif[right_r + dr, 4] = True

    assert motif[0].any() and motif[-1].any() and motif[:, 0].any() and motif[:, -1].any()
    assert is_contiguous(motif.astype(int), background=0, connectivity=8)
    return motif


def _random_large_motif() -> np.ndarray:
    """Sample a random 5x5 motif whose central 3x3 overlap is a valid original-style core."""
    return _expand_core_to_large_motif(random.choice(CORE_MASKS).copy())


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


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the solid 3x3 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 5x5 motif through its mask into the input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[int(color)]) for color in block_colors
    ]
    items.extend((int(color), _random_large_motif()) for color in distractor_colors)
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


def _carved_tile(bg: int, color: int, motif_mask: np.ndarray) -> np.ndarray:
    """Render one carved 3x3 tile from a centered 5x5 motif."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    core = motif_mask[CORE_SLICE, CORE_SLICE]
    tile[core] = int(bg)
    return tile


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carved_tile(bg, color, motif_masks[int(color)])
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


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
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
    )
    return inp


def _matching_metrics(
    motif_masks: dict[int, np.ndarray]
) -> tuple[int, int]:
    """Return minimum and maximum active pixels that lie outside the central 3x3 overlap."""
    extras = []
    for motif in motif_masks.values():
        border_active = int(motif.sum()) - int(motif[CORE_SLICE, CORE_SLICE].sum())
        extras.append(border_active)
    return min(extras), max(extras)


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
    motif_masks = {int(color): _random_large_motif() for color in block_colors}

    min_extra, _max_extra = _matching_metrics(motif_masks)
    if min_extra <= 0:
        return None

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

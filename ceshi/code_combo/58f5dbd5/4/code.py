# Combo variant of 58f5dbd5 (index 4)
# combo_spec: D7/2+D9/2
# Name: 反向掩码保留+原位替换

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


# Design analysis for combo D7/2 + D9/2
# --------------------------------------
# a) Concrete behavior and trace:
#    - The input keeps the seed task structure: one region contains 3-6 solid 5x5
#      monochrome blocks, another region contains 6-8 small 3x3-slot motifs, and one
#      motif color matches each block color.
#    - Original seed trace: find a red block, find the red motif, center it in the 5x5,
#      turn motif-covered cells into bg, then move the carved tile into a compact packed
#      output canvas.
#    - Combo trace: find the same red pair, center the motif in the 5x5, keep only the
#      motif-covered cells red while every other cell in that 5x5 becomes bg, then write
#      that transformed tile back at the block's original coordinates on a full-size bg
#      canvas. All source motifs disappear.
#    - Net difference: D7 flips the local block transform from "remove motif shape" to
#      "preserve motif shape", while D9 flips the global output from packed extraction to
#      in-place replacement on the original canvas.
# b) Complexity calibration:
#    - We keep distinct block colors, one matching motif per block color, and extra
#      distractor motifs in other colors. This preserves nontrivial color-based matching.
#    - Motifs remain connected 3x3 masks with area 4-8 and forced edge-touch on all four
#      sides of the 3x3 bbox, making the preserved silhouettes visible and asymmetric.
#    - Keeping 3-6 blocks and 6-8 motif slots gives repeated evidence that blocks stay in
#      place and that only motif silhouettes survive inside them.
# c) Implicit assumptions audit:
#    - Step 1 assumes solid 5x5 monochrome components are exactly the destination blocks.
#      The generator still creates only those as 5x5 solids.
#    - Step 2 assumes each block color has at least one smaller same-colored component and
#      distractor colors never reuse a block color. The generator enforces that.
#    - The seed Step 3 assumed the tile default fill was the block color; D7 breaks that,
#      so the combo must default each output tile to bg and only restore motif cells.
#    - The seed Step 4 assumed output shape depended only on the block lattice; D9 breaks
#      that, so the combo must instead rebuild a full HxW bg canvas and redraw blocks in
#      place.
#    - A new combo assumption is that the matched motif bbox fits comfortably when centered
#      inside 5x5. The generator guarantees this by restricting motifs to 3x3-slot masks.
# d) Placement legality:
#    - Destination and motif regions are placed by the original axis-separated scheme with
#      outer margin >= 1 and inter-region gap >= 1, so they do not overlap or go out of
#      bounds.
#    - Within each region, fixed lattice steps (6 for blocks, 4 for motif slots) prevent
#      collisions between same-region objects.
# e) Visual observability:
#    - The combo creates two strong cues visible across a few examples: the output keeps the
#      input canvas size, and each former solid block becomes only a centered colored motif
#      silhouette on bg.
#    - This is visually distinct from both the seed task (packed output plus holes) and from
#      a trivial "erase motif region" rule, because colored remnants remain only inside the
#      original block footprints.
# f) Visual cleanliness:
#    - Output contains only background and transformed block footprints; the source motif
#      region is entirely cleared, so no layered clutter remains.
#    - The input stays clean because blocks and motifs live in separate regions.
# g) Color-rule inferability:
#    - No hidden numeric ordering is used. Color only acts as the visible join key between
#      a block and its motif, and each block color appears exactly twice in the input: once
#      as a 5x5 block and once as its motif.
# h) Quantifiable anti-degeneration target:
#    - Every generated sample should satisfy `output.shape == input.shape`.
#    - For every transformed 5x5 block, the number of non-bg output cells should equal the
#      matched motif area, not `25 - motif_area`. This guarantees the combo is not silently
#      collapsing back to the in-place hole-carving behavior.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes there is at least one destination block; generation always creates
#      3-6 of them.
#    - `main()` chooses the smallest non-5x5 same-colored component as the motif; generator
#      ensures the matching motif is the only such component for each block color.
#    - `main()` clears the whole canvas before redrawing transformed blocks, so no other
#      non-background objects may need to survive. The generator only paints blocks and
#      motifs, satisfying that requirement.


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
    Preserve color-matched motif silhouettes inside solid blocks while keeping the original canvas.

    Input structure:
    - The background is the dominant grid color.
    - Several non-background 5x5 monochrome squares serve as destination blocks.
    - Elsewhere, small monochrome motif sprites appear; one sprite shares the color of each block.

    Rule:
    - Match each 5x5 block with the smallest non-block component of the same color.
    - Center that motif within the block footprint and keep only the motif-covered cells in the
      block color; every other cell of the 5x5 becomes background.
    - Return a grid with the same shape as the input: redraw the transformed blocks in their
      original positions and erase the entire motif region back to background.

    Key insight:
    - The motif is a protection mask, not a cutting stencil, and the output stays in place
      rather than being repacked into a new compact layout.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": r0, "c0": c0})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is NOT one of the 5x5 blocks.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []

        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[bcol],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == bcol).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[bcol] = candidates[0][4]

    # STEP 3) Transform each 5x5 tile:
    # center the motif mask within the 5x5 and keep only those mask cells in block color.
    preserved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), bg, dtype=int)
        motif = motifs.get(col)
        if motif is not None:
            mh, mw = map(int, motif.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(motif):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = col
        preserved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Keep the original canvas size: clear motifs and redraw only transformed blocks.
    out = np.full((H, W), bg, dtype=int)
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        out[r0 : r0 + 5, c0 : c0 + 5] = preserved[(r0, c0)]

    return out


# --------------- generator (o5x) ---------------

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

# Region placement.
OUTER_MARGIN = 1
REGION_GAP = 1
MOTIF_OFFSET_IN_BLOCK = 1

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


def _configs_for_count(
    H: int, W: int, target_count: int
) -> list[tuple[int, int, int, int, int]]:
    """Filter feasible configurations to one destination-block count."""
    return [cfg for cfg in _configs(H, W) if cfg[0] * cfg[1] == target_count]


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
    H: int,
    W: int,
    bg: int,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the oracle output for the in-place reverse-mask variant."""
    expected = np.full((H, W), bg, dtype=int)
    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), bg, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[color]] = color
            paste(
                expected,
                tile,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
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
    target_count = random.choice(sorted(DST_LAYOUTS))
    H = W = bg = None
    config = None
    for _ in range(24):
        cand_H = random.randint(*CANVAS_HEIGHT_RANGE)
        cand_W = random.randint(*CANVAS_WIDTH_RANGE)
        configs = _configs_for_count(cand_H, cand_W, target_count)
        if not configs:
            continue
        H = cand_H
        W = cand_W
        bg = random.randrange(COLOR_COUNT)
        config = random.choice(configs)
        break
    if config is None or H is None or W is None or bg is None:
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

    dst_r, dst_c, _motif_r, _motif_c = placement
    expected = _render_expected(H, W, bg, dst_r, dst_c, R, C, block_colors, motif_masks)
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

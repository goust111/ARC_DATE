# Combo variant of 58f5dbd5 (index 14)
# combo_spec: D9/2+D10/0
# Name: 原位替换+保留源motif

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# ---------------------------------------------------------------------------
# Design analysis for the D9/2 + D10/0 combo
# ---------------------------------------------------------------------------
# design_analysis:
#   a_concretization:
#     input:
#       - Keep the original scene grammar from 58f5dbd5: a region of solid 5x5
#         monochrome blocks and a separate region of small 3x3-slot motif sprites.
#       - Each block color is unique and has exactly one matching same-colored
#         motif; extra distractor motifs use other colors.
#     output:
#       - Apply the carving in place on the original HxW canvas instead of
#         repacking blocks into a new compact board.
#       - Unlike D9/2 alone, do not erase the motif region. Every source motif,
#         including distractors, remains exactly where it was in the input.
#     trace_example:
#       - Original task: a red motif is matched to a red 5x5 block, carved into a
#         5x5 tile, and that tile is moved into a separate packed output canvas.
#       - This combo: the same red motif still determines which red cells inside
#         the 5x5 become background, but the red block stays at its original
#         coordinates and the motif sprite itself also stays visible in its source
#         slot. The output is therefore "input with carved blocks" rather than a
#         packed summary.
#   b_complexity_calibration:
#     - The matching rule stays one-to-one through color identity, so preserving
#       motifs does not create ambiguity between multiple candidate tools.
#     - The visual change is strong: compared with the original task, the output
#       keeps the full canvas; compared with D9/2 alone, the motif region is no
#       longer wiped away. Both differences are visible in every sample.
#     - Distractor motifs are retained, so the task still requires identifying
#       which motifs are operative rather than assuming all motifs carve blocks.
#   c_implicit_assumption_audit:
#     - Block detection still relies on finding solid 5x5 monochrome components.
#       The generator continues to create exactly such components.
#     - Motif extraction still relies on "smallest same-colored non-block
#       component" being the matched stencil. The generator preserves this by
#       never reusing a block color for distractors.
#     - The original step 4 assumed the output was a newly packed canvas. That
#       assumption is invalid here; `main()` must preserve all non-block pixels
#       and overwrite only the block footprints.
#     - D9/2's clearing step assumed no non-block object should survive. That is
#       also invalid here, because D10/0 explicitly keeps the source motifs.
#   d_placement_legality:
#     - The generator keeps the original axis-separated placement between the
#       block region and the motif-slot region, so motifs never overlap blocks.
#     - Because output only rewrites block footprints in place, any legal input
#       placement automatically yields a legal output with no new collisions.
#   e_visual_observability:
#     - A solver can infer the rule from few examples because the same motif shape
#       remains visible in the source region while the corresponding centered hole
#       pattern appears inside the same-colored 5x5 block.
#     - No ordering or rule branch depends on hidden numeric color ids.
#   f_visual_tidiness:
#     - There is no new overlay layer: motifs remain in their own region and
#       blocks remain in theirs, so the output stays visually separated and clean.
#   g_color_inference:
#     - Colors are preserved. The human-visible cue is simple: same-colored motif
#       survives in place and also explains the carved hole shape in the block.
#   h_anti_degeneration_target:
#     - Every generated sample must satisfy `output.shape == input.shape`.
#     - Outside all destination-block cells, output must equal input exactly.
#     - Each sample must differ from the D9/2 "clear motif region" output, so the
#       keep-source behavior is active rather than accidental.
#   i_new_main_assumptions_and_generator_guarantees:
#     - `main()` assumes motifs fit when centered in 5x5; the generator only uses
#       connected masks within a 3x3 bbox, so this always holds.
#     - `main()` preserves every non-block object. The generator therefore keeps
#       the scene restricted to destination blocks plus motif sprites so that
#       preserving non-block pixels matches the task definition exactly.


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


def _find_blocks(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Return all solid 5x5 monochrome destination blocks."""
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
    return blocks


def _find_motifs(g: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, np.ndarray]:
    """Return the matched motif mask for each block color."""
    motifs: dict[int, np.ndarray] = {}
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
    return motifs


def _carved_tile(color: int, bg: int, motif: np.ndarray | None) -> np.ndarray:
    """Render one carved 5x5 block from its color and optional motif mask."""
    tile = np.full((5, 5), int(color), dtype=int)
    if motif is None:
        return tile

    mh, mw = map(int, motif.shape)
    off_r = (5 - mh) // 2
    off_c = (5 - mw) // 2
    for rr, cc in np.argwhere(motif):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < 5 and 0 <= tc < 5:
            tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks while keeping the full scene in place.

    Input structure:
    - The dominant color is the background.
    - One region contains several solid 5x5 monochrome destination blocks, each with a distinct non-background color.
    - A separate region contains small monochrome motif sprites; one sprite shares the color of each block, and extra distractor motifs may also appear.

    Rule:
    - For each block color, find the smallest same-colored non-block component and treat it as that block's carving stencil.
    - Center the stencil within the corresponding 5x5 block and repaint the covered cells to background, creating holes.
    - Return a grid with the same shape as the input: overwrite each block in place with its carved version, but keep every source motif sprite exactly where it originally appeared.

    Key insight:
    - The motifs still function as tools for carving, but unlike the original task or the pure in-place variant, the tool objects are not removed; the output is simply the original canvas with the blocks transformed.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks = _find_blocks(g, bg)
    assert blocks

    block_colors = {int(b["col"]) for b in blocks}
    motifs = _find_motifs(g, bg, block_colors)

    out = np.array(g, copy=True)
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        col = int(b["col"])
        out[r0 : r0 + 5, c0 : c0 + 5] = _carved_tile(col, bg, motifs.get(col))

    return out


# --------------- generator (o5x + in-place keep-source output) ---------------

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


def _block_mask(H: int, W: int, dst_r: int, dst_c: int, R: int, C: int) -> np.ndarray:
    """Return a boolean mask for all destination-block cells."""
    mask = np.zeros((H, W), dtype=bool)
    for i in range(R):
        for j in range(C):
            r0 = int(dst_r + i * DST_BLOCK_STEP)
            c0 = int(dst_c + j * DST_BLOCK_STEP)
            mask[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE] = True
    return mask


def _render_expected(
    inp: np.ndarray,
    bg: int,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the oracle output for the in-place keep-source combo."""
    expected = np.array(inp, copy=True)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carved_tile(color, bg, motif_masks[color])
            paste(
                expected,
                tile,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )
    return expected


def _render_d9_only_expected(
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
    """Reference output for the pure D9/2 variant that clears the motif region."""
    expected = np.full((H, W), bg, dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carved_tile(color, bg, motif_masks[color])
            paste(
                expected,
                tile,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )
    return expected


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

    dst_r, dst_c, _motif_r, _motif_c = placement
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, placement
    )
    expected = _render_expected(inp, bg, dst_r, dst_c, R, C, block_colors, motif_masks)
    d9_only = _render_d9_only_expected(H, W, bg, dst_r, dst_c, R, C, block_colors, motif_masks)

    block_mask = _block_mask(H, W, dst_r, dst_c, R, C)
    if not np.any(inp[~block_mask] != bg):
        return None
    if not np.array_equal(expected[~block_mask], inp[~block_mask]):
        return None
    if not np.any(expected[block_mask] != inp[block_mask]):
        return None
    if np.array_equal(expected, d9_only):
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

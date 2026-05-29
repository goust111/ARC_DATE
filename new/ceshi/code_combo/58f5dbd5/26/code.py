# Combo variant of 58f5dbd5 (index 26)
# combo_spec: D9/0+D18/1
# Name: 紧凑无边框+背景色由区域决定

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, is_contiguous, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The input still has two separated regions: a destination side containing several solid 5x5
#      blocks, and a motif side containing 3x3 same-colored sprites plus distractors.
#    - Relative to the original task, two rules change together:
#      1. The destination side sits inside a rectangular panel whose background color `block_bg`
#         differs from the rest-of-canvas motif-side background `motif_bg`.
#      2. The output no longer keeps a 1-cell border or 1-cell gaps; carved 5x5 blocks are packed
#         edge-to-edge into an exact (R*5) x (C*5) array.
#    - Small trace:
#      original: if a red block matches a red 3x3 motif, the motif pixels become the single global
#      background color, and the carved tiles are separated by visible background gutters.
#      combo: the same red motif still matches the red block, but the holes must become the panel
#      color seen around the 5x5 blocks, while the packed output removes every gutter and border.
#      So both the carving color and the output geometry differ at once.
# b) Complexity calibration:
#    - The change is visible but controlled: only two backgrounds exist, and only one of them is
#      eligible for carving. This adds a single structured color decision rather than many mappings.
#    - Tight packing changes the global output silhouette in an obvious repeated way across samples.
#    - Because block colors remain unique and motifs remain same-colored, matching is still stable.
# c) Broken assumptions from the original solver:
#    - `get_bg(grid)` no longer gives the hole color, because the motif side can dominate globally.
#    - Excluding only one background during component search is unsafe; the other regional background
#      would appear as a normal component. The fix is to inspect all monochrome components and select
#      only solid 5x5 squares as true blocks.
#    - The original packer assumed border+gap geometry in the output. In the combo, row/col order is
#      preserved but tile origins advance by exactly 5 rather than 6, and output size loses both
#      outer padding cells.
# d) Placement legality:
#    - The generator must place a full rectangular destination panel, not just bare blocks, because
#      the solver recovers `block_bg` from 1-cell rings around blocks.
#    - The panel keeps a 1-cell margin around the outer block lattice, and 1-cell internal gaps are
#      preserved between 5x5 blocks. This guarantees that every true block has a complete ring of
#      `block_bg` and remains a separate 25-cell component.
#    - Motifs stay outside the panel and are axis-separated from it exactly as in the base task.
# e) Visual observability:
#    - A human can infer the color rule because the destination panel repeatedly shows which color
#      surrounds the 5x5 blocks, while the motif region sits on a different background.
#    - A human can infer the packing rule because the output rectangle shrinks from bordered/gapped
#      layout to an exact mosaic of touching 5x5 tiles.
# f) Visual cleanliness:
#    - There are still only two clean regions in the input: one rectangular panel plus one motif
#      region on the other background. No overlays or mixed markers are introduced.
#    - In the output, touching tiles are still distinguishable because each tile keeps its own block
#      color and only the carved holes use the shared `block_bg`.
# g) Color-change inference:
#    - The hole color is not arbitrary; it is structurally tied to the region that already contains
#      the blocks. The repeated panel cue keeps this learnable from 2-3 examples.
# h) Solver assumptions introduced here and generator guarantees:
#    - Assumption: every real destination object is an isolated 5x5 monochrome component of area 25.
#      Guarantee: blocks use unique non-background colors and are placed on a 6-step lattice.
#    - Assumption: the 1-cell ring around each block is dominated entirely by `block_bg`.
#      Guarantee: the destination panel extends one cell beyond the outermost blocks and fills every
#      inter-block gap with `block_bg`.
#    - Assumption: each block color has one smaller same-colored motif component elsewhere.
#      Guarantee: exactly one matching motif is generated per block color, while distractors use
#      different colors from all block colors and both backgrounds.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the listed in-bounds coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _all_color_components(grid: np.ndarray) -> list[dict]:
    """Return monochrome connected components across every color present in the grid."""
    return find_components(
        grid,
        bg=-1,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )


def _ring_values(grid: np.ndarray, r0: int, c0: int, size: int) -> np.ndarray:
    """Return the 1-cell-thick outer ring around a size×size square."""
    patch = grid[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1]
    mask = np.ones((size + 2, size + 2), dtype=bool)
    mask[1:-1, 1:-1] = False
    return patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then pack the tiles flush together.

    Input structure:
    - The canvas has two regional backgrounds: the motif side uses one color, while the 5x5 block
      side sits inside a separate rectangular panel of another color.
    - Several solid 5x5 monochrome blocks appear inside that panel, each in a distinct color.
    - Elsewhere, each block color has a smaller same-colored motif sprite, with optional distractors.

    Rule:
    - Detect the true 5x5 blocks without assuming a single global background, then recover the
      carving color from the one-cell ring around those blocks.
    - Match each motif to the block of the same color and center it inside the 5x5 tile.
    - Recolor the motif cells to the block-region background, then pack the carved tiles in their
      original row/column order with no gaps and no outer border.

    Key insight:
    - The motif still acts as a stencil, but both the hole color and the output geometry come from
      the destination-block region: use the panel background for carving, and output a tight mosaic.
    """
    g = np.array(grid, dtype=int)

    # STEP 1) Identify destination blocks without assuming a single global background.
    blocks: list[dict[str, int]] = []
    for comp in _all_color_components(g):
        b = comp["bbox"]
        if int(comp["area"]) != 25 or bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": r0, "c0": c0})

    assert blocks

    # STEP 2) Recover the destination-panel background from the 1-cell ring around the blocks.
    ring_counter: Counter[int] = Counter()
    for b in blocks:
        vals = _ring_values(g, int(b["r0"]), int(b["c0"]), 5)
        ring_counter.update(map(int, vals.tolist()))
    block_bg = int(ring_counter.most_common(1)[0][0])

    # STEP 3) For each block color, find the smaller same-colored component that acts as its motif.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=-1,
            connectivity=8,
            monochromatic=True,
            colors=[bcol],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if int(comp["area"]) == 25 and bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == bcol).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[bcol] = candidates[0][4]

    # STEP 4) Carve each tile using the block-side background, then pack them flush together.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = block_bg
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(len(rs) * 5)
    out_w = int(len(cs) * 5)
    out = np.full((out_h, out_w), block_bg, dtype=int)

    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[r0])
        j = int(c_index[c0])
        rr = int(i * 5)
        cc = int(j * 5)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


# --------------- generator (o5x, combo variant) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 24)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry in the input.
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

# Region placement and output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
MOTIF_OFFSET_IN_BLOCK = 1
BLOCK_PANEL_MARGIN = 1
OUTPUT_BORDER = 0
OUTPUT_STEP = DST_BLOCK_SIZE

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
    """Choose a valid separated placement for destination panel and motif region."""
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
            dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            panel_h = dst_h + 2 * BLOCK_PANEL_MARGIN
            panel_w = dst_w + 2 * BLOCK_PANEL_MARGIN
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
                        if _can_split(H, W, panel_h, panel_w, mh, mw):
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


def _configs_for_count(H: int, W: int, count: int) -> list[tuple[int, int, int, int, int]]:
    """Return only the layouts whose block count matches the requested branch."""
    return [cfg for cfg in _configs(H, W) if cfg[0] * cfg[1] == count]


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


FEASIBLE_SIZES_BY_COUNT = {
    count: [
        (H, W)
        for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1)
        for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1)
        if _configs_for_count(H, W, count)
    ]
    for count in sorted(DST_LAYOUTS)
}


def _choose_palette(
    block_bg: int, motif_bg: int, count: int, filled: int
) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c not in {block_bg, motif_bg}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    block_bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, OUTPUT_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, OUTPUT_STEP)
    expected = np.full((out_h, out_w), block_bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[color]] = block_bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * OUTPUT_STEP,
                OUTPUT_BORDER + j * OUTPUT_STEP,
            )
    return expected


def _paint_destination_panel(
    grid: np.ndarray, dst_r: int, dst_c: int, dst_h: int, dst_w: int, block_bg: int
) -> None:
    """Paint the rectangular block-side panel whose background defines the output."""
    panel = np.full((dst_h, dst_w), int(block_bg), dtype=int)
    paste(grid, panel, dst_r, dst_c)


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    base_r = int(dst_r + BLOCK_PANEL_MARGIN)
    base_c = int(dst_c + BLOCK_PANEL_MARGIN)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(
                grid,
                block,
                base_r + i * DST_BLOCK_STEP,
                base_c + j * DST_BLOCK_STEP,
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
    motif_bg: int,
    block_bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination panel/blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * BLOCK_PANEL_MARGIN
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * BLOCK_PANEL_MARGIN

    inp = np.full((H, W), motif_bg, dtype=int)
    _paint_destination_panel(inp, dst_r, dst_c, dst_h, dst_w, block_bg)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    motif_bg, block_bg = random.sample(range(COLOR_COUNT), 2)

    feasible_counts = [count for count, sizes in FEASIBLE_SIZES_BY_COUNT.items() if sizes]
    if not feasible_counts:
        return None
    count = random.choice(feasible_counts)
    H, W = random.choice(FEASIBLE_SIZES_BY_COUNT[count])

    configs = _configs_for_count(H, W, count)
    if not configs:
        return None
    R, C, Mr, Mc, filled = random.choice(configs)

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * BLOCK_PANEL_MARGIN
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP) + 2 * BLOCK_PANEL_MARGIN
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(block_bg, motif_bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(block_bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        motif_bg,
        block_bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_masks,
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

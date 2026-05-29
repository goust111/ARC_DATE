# Variant of 58f5dbd5
# Dimension: D5 — 目标对象：方块结构 (dst)
# Change: 5x5纯色单色实心方块，每个方块颜色不同... → 尺寸各异的方块
#
# Design analysis
# ---
# variant_spec:
#   core_change:
#     from: "all destination blocks are solid 5x5 squares"
#     to: "destination blocks are solid squares with side lengths in {3,5,7}"
#   input_appearance:
#     dst_region:
#       structure: "a single horizontal strip or a single vertical strip of solid squares"
#       sizes: "each square side is 3, 5, or 7; every sample uses at least two sizes"
#       colors: "all block colors are distinct and non-background"
#       spacing: "adjacent block bboxes are separated by exactly 1 background cell"
#     motif_region:
#       structure: "a separated grid of 3x3 monochrome motifs"
#       matching_rule: "each destination color has one same-colored motif; optional distractor motifs use other colors"
#       shape_rule: "motifs are connected, non-solid, and touch all four sides of their 3x3 bbox"
#   output_appearance:
#     rule: "for each block, scale its same-colored 3x3 motif to the block side length and carve those cells to background"
#     packing: "pack the carved blocks in the same strip orientation with a 1-cell outer border and 1-cell gaps"
#   behavior_trace:
#     original_local_step:
#       - "detect only solid 5x5 squares"
#       - "center an unscaled motif inside the 5x5 tile"
#       - "pack tiles on a fixed 5+1 lattice"
#     variant_local_step:
#       - "detect solid squares of side 3, 5, or 7"
#       - "resize the motif mask to the square side length before carving"
#       - "pack tiles using cumulative true sizes instead of a fixed lattice"
#     concrete_example:
#       - "a plus-like 3x3 motif matched to a 7x7 block becomes a 7x7 enlarged hole pattern"
#       - "the same motif matched to a 3x3 block stays 1:1"
#       - "therefore block size now changes the carved geometry, not just the packed canvas size"
#   complexity_calibration:
#     - "At least two destination sizes appear in every sample, so the new rule is observable."
#     - "Block colors remain unique, which keeps motif matching unambiguous."
#     - "Allowed sizes are restricted to three odd values, keeping the scaling hypothesis small enough for humans."
#   implicit_assumptions_review:
#     - "Original code assumed bbox_h == bbox_w == 5 when identifying destination objects; this must become a set membership test."
#     - "Original carving offsets (5-mh)//2 and fixed 5x5 tiles relied on one invariant size; variable squares need per-block tiles."
#     - "Original output dimensions used len(rows)*5 and step=6; variable sizes require cumulative sums of actual side lengths."
#     - "Sorting by unique row and column coordinates only works cleanly if the variable-size blocks form one strip; generator enforces that."
#   placement_legality:
#     - "The generator keeps destination blocks in a 1D strip because a true 2D compact square grid with independent square sizes would misalign top-left anchors."
#     - "Destination and motif regions are still separated along one axis with margins, so components never overlap."
#     - "The solver expects the destination blocks to share one row or one column; generator guarantees this."
#   visual_observability:
#     - "Humans can compare differently sized same-role blocks across 2-4 examples and see that larger squares contain enlarged versions of the motif holes."
#     - "No ordering or transformation depends on hidden color ids; color is used only for visible matching."
#   cleanliness:
#     - "Destination strip and motif grid remain spatially separated, so block bodies and motif sprites are visually distinct."
#     - "Output keeps a single strip and border, avoiding clutter from ragged multi-row packing."

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


DST_SIZES = (3, 5, 7)
DST_GAP = 1
MOTIF_SIZE = 3


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


def _solid_square_info(grid: np.ndarray, comp: dict) -> tuple[bool, int]:
    """Return whether a component is a solid allowed-size square and its side length."""
    b = comp["bbox"]
    side_h = int(bbox_h(b))
    side_w = int(bbox_w(b))
    if side_h != side_w or side_h not in DST_SIZES:
        return False, side_h
    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    return bool((patch == col).all()), side_h


def _resize_mask(mask: np.ndarray, target_side: int) -> np.ndarray:
    """Resize a boolean motif mask to target_side × target_side by nearest-neighbor sampling."""
    target = int(target_side)
    src_h, src_w = map(int, mask.shape)
    if src_h == target and src_w == target:
        return mask.copy()
    row_idx = np.rint(np.linspace(0, src_h - 1, target)).astype(int)
    col_idx = np.rint(np.linspace(0, src_w - 1, target)).astype(int)
    return mask[row_idx][:, col_idx]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into variable-size solid squares and pack them into a compact strip.

    Input structure:
    - bg = most common color.
    - Several solid monochrome squares of side 3, 5, or 7, each a distinct non-bg color.
    - For each block color, a same-colored 3x3 motif sprite appears elsewhere on the grid.

    Rule:
    - Match each motif to the solid square with the same color.
    - Scale the motif mask to that square's side length, then repaint the scaled mask cells to bg.
    - Pack all carved squares in their original strip order using 1-cell gaps and a 1-cell outer border.

    Key insight: The motif is still a stencil, but now the hole pattern grows or shrinks with the square size rather than being inserted at one fixed 5x5 scale.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 3x3 / 5x5 / 7x7 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        is_solid, side = _solid_square_info(g, comp)
        if not is_solid:
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0), "side": int(side)})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the solid destination squares.
    motifs: dict[int, np.ndarray] = {}
    block_bboxes_by_color = {(int(b["col"]), int(b["r0"]), int(b["c0"]), int(b["side"])) for b in blocks}
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
            is_solid, side = _solid_square_info(g, comp)
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if is_solid and (int(bcol), int(r0), int(c0), int(side)) in block_bboxes_by_color:
                continue
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each square by resizing its motif mask to the square size.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        side = int(b["side"])
        tile = np.full((side, side), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            scaled = _resize_mask(mm, side)
            tile[scaled] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved squares into a compact strip with 1-cell gaps and border.
    same_row = len({int(b["r0"]) for b in blocks}) == 1
    same_col = len({int(b["c0"]) for b in blocks}) == 1
    assert same_row or same_col

    if same_row:
        ordered = sorted(blocks, key=lambda b: int(b["c0"]))
        max_side = max(int(b["side"]) for b in ordered)
        total_w = sum(int(b["side"]) for b in ordered) + (len(ordered) - 1) * DST_GAP
        out = np.full((1 + max_side + 1, 1 + total_w + 1), int(bg), dtype=int)
        cc = 1
        for b in ordered:
            side = int(b["side"])
            out[1 : 1 + side, cc : cc + side] = carved[(int(b["r0"]), int(b["c0"]))]
            cc += side + DST_GAP
        return out

    ordered = sorted(blocks, key=lambda b: int(b["r0"]))
    max_side = max(int(b["side"]) for b in ordered)
    total_h = sum(int(b["side"]) for b in ordered) + (len(ordered) - 1) * DST_GAP
    out = np.full((1 + total_h + 1, 1 + max_side + 1), int(bg), dtype=int)
    rr = 1
    for b in ordered:
        side = int(b["side"])
        out[rr : rr + side, 1 : 1 + side] = carved[(int(b["r0"]), int(b["c0"]))]
        rr += side + DST_GAP
    return out


# --------------- generator ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

BLOCK_COUNT_OPTIONS = (3, 4)
ORIENTATIONS = ("h", "v")

MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = ((2, 2), (2, 3), (3, 2))
MAX_DISTRACTORS = 2
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
GENERATION_RETRIES = 400


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks or area == MOTIF_SIZE * MOTIF_SIZE:
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


MASKS_BY_AREA = _make_masks_by_area()


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _sample_sizes(count: int) -> list[int]:
    """Sample destination square sizes with visible variation in every instance."""
    unique_count = random.choice([2, 2, 3])
    chosen = random.sample(list(DST_SIZES), unique_count)
    sizes = [chosen[i] for i in range(min(count, unique_count))]
    while len(sizes) < count:
        sizes.append(random.choice(chosen))
    random.shuffle(sizes)
    if len(set(sizes)) < 2 or all(size == 5 for size in sizes):
        return _sample_sizes(count)
    return [int(size) for size in sizes]


def _dst_region_size(sizes: list[int], orientation: str) -> tuple[int, int]:
    """Return the destination strip bbox size."""
    if orientation == "h":
        return max(sizes), sum(sizes) + (len(sizes) - 1) * DST_GAP
    return sum(sizes) + (len(sizes) - 1) * DST_GAP, max(sizes)


def _motif_region_size(Mr: int, Mc: int) -> tuple[int, int]:
    """Return the motif grid bbox size."""
    return Mr * MOTIF_SIZE + (Mr - 1) * (MOTIF_STEP - MOTIF_SIZE), Mc * MOTIF_SIZE + (Mc - 1) * (MOTIF_STEP - MOTIF_SIZE)


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


def _place_regions(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> tuple[int, int, int, int] | None:
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    sizes: list[int],
    block_colors: list[int],
    orientation: str,
) -> None:
    """Paint the solid variable-size destination squares into the input grid."""
    rr = int(dst_r)
    cc = int(dst_c)
    for size, color in zip(sizes, block_colors):
        block = np.full((size, size), int(color), dtype=int)
        paste(grid, block, rr, cc)
        if orientation == "h":
            cc += int(size) + DST_GAP
        else:
            rr += int(size) + DST_GAP


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


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


def _render_expected(
    bg: int, sizes: list[int], block_colors: list[int], motif_masks: dict[int, np.ndarray], orientation: str
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled strip of destination squares."""
    if orientation == "h":
        out_h = 2 * OUTPUT_BORDER + max(sizes)
        out_w = 2 * OUTPUT_BORDER + sum(sizes) + (len(sizes) - 1) * DST_GAP
        expected = np.full((out_h, out_w), bg, dtype=int)
        cc = OUTPUT_BORDER
        for size, color in zip(sizes, block_colors):
            tile = np.full((size, size), int(color), dtype=int)
            tile[_resize_mask(motif_masks[color], size)] = int(bg)
            paste(expected, tile, OUTPUT_BORDER, cc)
            cc += int(size) + DST_GAP
        return expected

    out_h = 2 * OUTPUT_BORDER + sum(sizes) + (len(sizes) - 1) * DST_GAP
    out_w = 2 * OUTPUT_BORDER + max(sizes)
    expected = np.full((out_h, out_w), bg, dtype=int)
    rr = OUTPUT_BORDER
    for size, color in zip(sizes, block_colors):
        tile = np.full((size, size), int(color), dtype=int)
        tile[_resize_mask(motif_masks[color], size)] = int(bg)
        paste(expected, tile, rr, OUTPUT_BORDER)
        rr += int(size) + DST_GAP
    return expected


def _render_input(
    H: int,
    W: int,
    bg: int,
    sizes: list[int],
    orientation: str,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination strip and motif grid into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, sizes, block_colors, orientation)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks)
    return inp


def _sample_motif_config(count: int) -> tuple[int, int, int] | None:
    """Choose a motif grid and filled-slot count compatible with the block count."""
    choices: list[tuple[int, int, int]] = []
    for Mr, Mc in MOTIF_LAYOUTS:
        capacity = Mr * Mc
        min_filled = count
        max_filled = min(capacity, count + MAX_DISTRACTORS)
        for filled in range(min_filled, max_filled + 1):
            empties = capacity - filled
            if empties <= MAX_EMPTY_SLOTS:
                choices.append((Mr, Mc, filled))
    if not choices:
        return None
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    count = random.choice(BLOCK_COUNT_OPTIONS)
    orientation = random.choice(ORIENTATIONS)
    sizes = _sample_sizes(count)
    dst_h, dst_w = _dst_region_size(sizes, orientation)

    motif_cfg = _sample_motif_config(count)
    if motif_cfg is None:
        return None
    Mr, Mc, filled = motif_cfg
    motif_h, motif_w = _motif_region_size(Mr, Mc)
    if not _can_split(H, W, dst_h, dst_w, motif_h, motif_w):
        return None

    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, sizes, block_colors, motif_masks, orientation)
    inp = _render_input(
        H,
        W,
        bg,
        sizes,
        orientation,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D5 input")

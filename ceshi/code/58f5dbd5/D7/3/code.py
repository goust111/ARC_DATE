# Variant of 58f5dbd5
# Dimension: D7 — 变换操作：雕刻方式 (transform)
# Change: 将motif覆盖区域的方块像素替换为背景色（减法雕刻/打洞）... → 计数编码变换

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste

# Design analysis:
# spec:
#   variant_core:
#     keep the original color matching and compact repacking, but replace
#     shape carving with count encoding: the matched motif contributes only
#     its filled-cell area, and the block shows that area as a fixed number
#     of bg dots on a centered 3x3 tally lattice.
#   behavior_trace:
#     original:
#       a 3x3 motif of area 5 is centered into the 5x5 block and its exact
#       occupied cells become holes.
#     variant:
#       the same motif is still matched by color, but only area=5 is used;
#       the block stays otherwise solid and the first 5 sites of the fixed
#       interior 3x3 lattice are turned to bg.
#     difference:
#       output no longer preserves motif geometry, only motif cardinality.
#   complexity_calibration:
#     - blocks stay solid 5x5 squares so destination detection is unchanged.
#     - motifs still live in 3x3 slots with area 4..8, which fits naturally
#       into a 9-site tally lattice.
#     - the generator deliberately repeats some motif areas across different
#       colors while sampling different shapes for those equal-area motifs,
#       so the "count not shape" rule is visible in a few examples.
#   implicit_assumption_audit:
#     - block detection depends on 5x5 monochrome squares; preserved.
#     - motif lookup depends on exactly one non-block component per block
#       color; generator preserves color uniqueness across visible motifs.
#     - the old centered-mask write loop depended on motif geometry; this is
#       the line-level assumption that changes, so it is replaced by a fixed
#       dot-lattice renderer keyed only by motif area.
#     - packing by block top-left row/col is still valid and unchanged.
#   placement_legality:
#     destination and motif regions are still placed as separated rectangles
#     with margins and 1-cell inter-region gap, so nothing overlaps and every
#     motif remains an isolated connected component for the solver.
#   observability_and_cleanliness:
#     tally dots always live on the same centered 3x3 lattice, making the
#     count easy to compare across tiles; using bg holes preserves contrast
#     against the solid colored blocks without introducing new colors.
#   new_solver_assumptions:
#     main() now assumes the motif area is at most the tally capacity (9),
#     and generate_input() enforces area 4..8, well within that bound.


COUNT_DOT_COORDS = [
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
]


def _render_count_tile(color: int, bg: int, count: int) -> np.ndarray:
    """Render one 5x5 block with `count` bg tally dots on the inner 3x3 lattice."""
    tile = np.full((5, 5), int(color), dtype=int)
    for rr, cc in COUNT_DOT_COORDS[: max(0, min(int(count), len(COUNT_DOT_COORDS)))]:
        tile[int(rr), int(cc)] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode the area of each color-matched motif as tally dots inside solid blocks.

    Input structure:
    - bg = most common color.
    - Several solid 5×5 monochrome blocks, each a distinct non-bg color.
    - For each block color, a small same-colored motif sprite elsewhere on the grid.

    Rule:
    - Match each motif to its same-colored 5×5 block.
    - Ignore the motif's geometry and count only its filled cells.
    - Repaint that many cells inside the block to bg using a fixed centered 3×3 tally lattice, then pack all encoded blocks into a compact grid with 1-cell gaps and a 1-cell outer border.

    Key insight: The motif is no longer a stencil; it is a numeric clue, and the output records motif area rather than motif shape.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, find the smallest same-colored component
    # that is not itself a solid 5x5 block, and keep only its area.
    motif_areas: dict[int, int] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int]] = []  # (area, bbox_area, r0, c0)
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
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == int(bcol)).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0)))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motif_areas[int(bcol)] = int(candidates[0][0])

    # STEP 3) Encode each 5x5 tile with tally dots for the motif area.
    encoded: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        area = int(motif_areas.get(int(col), 0))
        encoded[(int(b["r0"]), int(b["c0"]))] = _render_count_tile(col, bg, area)

    # STEP 4) Pack the encoded 5x5 tiles into a compact grid.
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
        out[rr : rr + 5, cc : cc + 5] = encoded[(int(r0), int(c0))]

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

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

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


def _random_mask(area: int | None = None) -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    if area is None:
        area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[int(area)]).copy()


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


def _sample_block_areas(count: int) -> list[int]:
    """Choose block motif areas with deliberate repetition so count classes recur."""
    values = list(range(MOTIF_AREA_RANGE[0], MOTIF_AREA_RANGE[1] + 1))
    if count == 3:
        a, b = random.sample(values, 2)
        areas = [a, a, b]
        random.shuffle(areas)
        return areas

    group_count = 2 if count == 4 else random.choice([2, 3])
    chosen = random.sample(values, group_count)
    repeats = [1] * group_count
    repeats[0] += 1
    remaining = count - sum(repeats)
    while remaining > 0:
        repeats[random.randrange(group_count)] += 1
        remaining -= 1

    areas: list[int] = []
    for area, freq in zip(chosen, repeats):
        areas.extend([int(area)] * int(freq))
    random.shuffle(areas)
    return areas


def _choose_matching_motifs(
    block_colors: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Assign each block color a motif mask and a repeated-support area value."""
    areas = _sample_block_areas(len(block_colors))
    area_to_masks: dict[int, list[np.ndarray]] = {}
    for area in sorted(set(areas)):
        need = areas.count(area)
        bank = MASKS_BY_AREA[int(area)]
        if need <= len(bank):
            chosen = [m.copy() for m in random.sample(bank, need)]
        else:
            chosen = [_random_mask(area) for _ in range(need)]
        random.shuffle(chosen)
        area_to_masks[int(area)] = chosen

    motif_masks: dict[int, np.ndarray] = {}
    motif_areas: dict[int, int] = {}
    for color, area in zip(block_colors, areas):
        motif_masks[int(color)] = area_to_masks[int(area)].pop()
        motif_areas[int(color)] = int(area)
    return motif_masks, motif_areas


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_areas: dict[int, int]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _render_count_tile(color, bg, int(motif_areas[color]))
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
    motif_masks, motif_areas = _choose_matching_motifs(block_colors)

    expected = _render_expected(bg, R, C, block_colors, motif_areas)
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 input")

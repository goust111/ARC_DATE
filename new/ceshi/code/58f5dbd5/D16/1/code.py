# Variant of 58f5dbd5
# Dimension: D16 — 符号编码 (symbol_instructions)
# Change: 无符号编码，所有信息直接通过颜色和空间关系表达 → 颜色映射图例

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D16 variant
# ===================================
# a) Concrete behavior and flow impact
# - The original task used only direct color identity: motif color A matched 5x5 block color A,
#   and the centered motif cells were repainted to the background.
# - This variant keeps the color-based block↔motif match, but adds a symbolic legend in the
#   top-left corner. The legend is a vertical stack of horizontal 1x2 color pairs [A, B] placed
#   at rows 0, 2, 4, ... with blank separator rows. Each pair means:
#     block color A -> recolor the centered motif footprint to color B.
# - The replacement colors B are a derangement of the block colors, so every block uses a
#   non-background replacement that differs from its own body color. This makes the new behavior
#   visually obvious: the packed output contains colored inlays instead of background holes.
# - Small traced example:
#   original: red block + red motif mask -> centered motif cells become bg.
#   variant: red block + red motif mask + legend [red blue] -> centered motif cells become blue.
#   The match stage is unchanged; only the carve/recolor stage now consults the legend.
#
# b) Complexity calibration
# - The legend contains one entry per real block color, not per distractor color. This keeps the
#   mapping task compact and avoids ambiguous "unused instructions".
# - Replacement colors come from the set of block colors rather than introducing many new colors.
#   That preserves the original 3..6 block count without exceeding the ARC 10-color palette.
# - The mapping is a bijection with no fixed points, so the legend matters for every block and
#   cannot silently collapse back to the original bg-carving behavior.
#
# c) Implicit-assumption audit against the original main()
# - Block detection via "solid 5x5 monochrome component" still holds.
# - The original motif search assumed that the smallest non-5x5 same-colored component was the
#   motif. That becomes false once the legend adds 1x1 same-colored instruction cells. The new
#   solver therefore parses the legend first and excludes legend cells from motif candidates.
# - The original carve step unconditionally wrote bg. That is no longer harmless, because the
#   replacement color now varies by block; it must read legend_map[block_color].
# - The original pack step relied only on block positions, so it stays valid unchanged.
#
# d) Placement legality
# - The generator reserves a parse-guard rectangle for the corner legend: rows 0..2*count and
#   cols 0..2. This ensures the solver can scan even rows for legend entries and reliably stop at
#   the first empty pair row before any other object enters those columns.
# - Destination and motif regions are still axis-separated, non-overlapping, and kept inside the
#   canvas exactly as before, with an extra rejection check against the legend guard rectangle.
#
# e) Visual observability
# - Humans can infer the new rule from 2-4 examples because the same motif shape still appears,
#   but the carved area is no longer background-colored; it takes the right-hand color from the
#   corner pair whose left-hand color matches the block.
# - The change is visible in every block because the replacement map is a derangement.
#
# f) Visual cleanliness
# - The legend uses only a narrow 2-column corner strip with blank rows between pairs, so it is
#   visually distinct from both the 5x5 blocks and the 3x3 motif slot grid.
# - No object overlaps the legend guard strip, preventing mixed signals near the symbol area.
#
# g) Color-change inferability
# - The mapping is explicitly shown by local A->B pairs; the solver does not rely on color ids,
#   numeric ordering, or invisible conventions.
# - Because B is always another visible block color, the output repeats those colors in structured
#   motif footprints, giving strong support for the mapping rule.
#
# 9) New solver assumptions and generator guarantees
# - Assumption: the legend is always the top-left even-row stack of 1x2 pairs. Guaranteed by
#   _paint_legend() and the legend guard rectangle in placement.
# - Assumption: every block color appears exactly once as a legend key and has one same-colored
#   non-legend motif component. Guaranteed by the derangement map and motif painting logic.
# - Assumption: legend cells are isolated from motif/block components. Guaranteed by the reserved
#   guard rectangle and region separation checks.


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


def _read_corner_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    """Read the top-left [A,B] legend pairs placed on even rows."""
    H, W = grid.shape
    if W < 2:
        return {}, set()

    legend_map: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    row = 0
    while row < H:
        left = int(grid[row, 0])
        right = int(grid[row, 1])
        if left == bg or right == bg:
            break
        legend_map[left] = right
        legend_cells.add((row, 0))
        legend_cells.add((row, 1))
        row += 2
    return legend_map, legend_cells


def _is_legend_component(comp: dict, legend_cells: set[tuple[int, int]]) -> bool:
    coords = comp["coords"]
    return all((int(r), int(c)) in legend_cells for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor color-matched motifs into solid blocks according to a corner legend, then pack the blocks.

    Input structure:
    - bg = most common color.
    - Several solid 5x5 monochrome blocks, each a distinct non-bg color.
    - For each block color, a small same-colored motif sprite elsewhere on the grid.
    - In the top-left corner, a vertical list of 1x2 color pairs [A, B] gives the replacement color for block color A.

    Rule:
    - Match each motif to its same-colored 5x5 block.
    - Center the motif within the block and recolor those cells to the legend's mapped color B, not to bg.
    - Pack all recolored blocks into an output grid preserving their original row/column order, separated by 1-cell gaps and a 1-cell outer border.

    Key insight: The motif still acts like a stencil, but the corner legend changes what color gets written into the stencil footprint for each block.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legend_map, legend_cells = _read_corner_legend(g, bg)

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

    # STEP 2) Parse matching motifs, excluding the corner legend's 1x1 instruction cells.
    motifs: dict[int, np.ndarray] = {}
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
            if _is_legend_component(comp, legend_cells):
                continue

            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == int(bcol)).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        assert candidates
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs[int(bcol)] = candidates[0][4]

    assert all(int(col) in legend_map for col in block_colors)

    # STEP 3) Recolor each 5x5 tile using its centered motif mask and legend-mapped fill color.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        fill = int(legend_map[int(col)])
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
                    tile[tr, tc] = fill
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack recolored 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

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
MOTIF_OFFSET_IN_BLOCK = 1

# Generation retry budget.
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 40


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


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_regions(
    H: int,
    W: int,
    dh: int,
    dw: int,
    mh: int,
    mw: int,
    blocked: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None

    for _ in range(PLACEMENT_RETRIES):
        scheme = random.choice(schemes)
        if scheme in {"dst_above", "motif_above"}:
            placement = _place_vertical(H, W, dh, dw, mh, mw, scheme)
        else:
            placement = _place_horizontal(H, W, dh, dw, mh, mw, scheme)

        dst_r, dst_c, motif_r, motif_c = placement
        dst_rect = (dst_r, dst_c, dst_r + dh - 1, dst_c + dw - 1)
        motif_rect = (motif_r, motif_c, motif_r + mh - 1, motif_c + mw - 1)
        if blocked is not None and (
            _rects_intersect(dst_rect, blocked) or _rects_intersect(motif_rect, blocked)
        ):
            continue
        return placement
    return None


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


def _random_derangement(values: list[int]) -> dict[int, int]:
    """Map each value to a different value, using a permutation with no fixed points."""
    shuffled = values[:]
    while True:
        random.shuffle(shuffled)
        if all(int(a) != int(b) for a, b in zip(values, shuffled)):
            return {int(a): int(b) for a, b in zip(values, shuffled)}


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _paint_legend(grid: np.ndarray, block_colors: list[int], replacement_map: dict[int, int]) -> None:
    """Paint the top-left vertical stack of [A, B] legend pairs."""
    for idx, src in enumerate(block_colors):
        row = 2 * idx
        grid[row, 0] = int(src)
        grid[row, 1] = int(replacement_map[int(src)])


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    replacement_map: dict[int, int],
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
            tile[motif_slice, motif_slice][motif_masks[color]] = int(replacement_map[color])
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
    replacement_map: dict[int, int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks, motifs, and legend into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_legend(inp, block_colors, replacement_map)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
    )
    return inp


def _legend_guard_rect(count: int) -> tuple[int, int, int, int]:
    """Reserve the legend rows plus the first blank even row and a 1-cell right moat."""
    return (0, 0, 2 * count, 2)


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

    count = R * C
    blocked = _legend_guard_rect(count)
    if blocked[2] >= H or blocked[3] >= W:
        return None

    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w, blocked=blocked)
    if placement is None:
        return None

    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    replacement_map = _random_derangement(block_colors)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks, replacement_map)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_masks,
        replacement_map,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D16 input")

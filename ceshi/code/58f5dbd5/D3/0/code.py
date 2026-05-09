# Variant of 58f5dbd5
# Dimension: D3 — 源对象：motif结构 (src)
# Change: 单色、8连通、3x3外接矩形内的小型motif精灵，面积4-8像素... → 多色motif精灵

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete pixel behavior
#    - The 5x5 destination blocks stay unchanged: several solid monochrome squares with distinct colors.
#    - The source motifs remain small 3x3-bounded 8-connected sprites in a separate region, but each motif now
#      uses exactly two visible colors:
#      1. the block-matching color;
#      2. one shared decoration color that appears in every motif.
#    - When centered onto the matching 5x5 block, the motif no longer means "all filled cells become bg".
#      Instead, the two colors encode two actions:
#      1. cells of the block-matching color carve holes to bg;
#      2. cells of the decoration color repaint the block to that decoration color.
#    - Small trace:
#      original: motif cells {X X / . X / X .} on a red block => those 4 cells become bg.
#      variant: motif cells {red teal / . red / teal .} on a red block => red cells become bg, teal cells become teal.
#      Difference: one input sprite now carries both the matching cue and the output recolor cue.
# b) Complexity calibration
#    - A single shared decoration color keeps the extra reasoning visible but not noisy. If every motif used a
#      different second color, the rule would be harder to infer from 2-4 ARC examples.
#    - Matching stays unique because only one motif color can belong to the destination-block palette; distractor
#      motifs use non-block primary colors.
#    - Decoration cells vary per motif (1-3 pixels), so the output is visibly different from the original carve-only
#      task and does not collapse into a trivial fixed marker.
# c) Original-code assumptions that break under the variant
#    - Original STEP 2 searched monochromatic components per block color. That fails now because each useful motif is
#      multicolor, so the matching color alone is only one monochrome fragment instead of the whole motif.
#    - Original STEP 3 converted every motif cell to bg. That is no longer harmless: decoration-colored cells must
#      survive as a third color inside the block.
#    - Original generator painted each motif as one color and rendered expected output with a single boolean mask.
#      The variant needs role-aware patterns: primary-mask -> bg, decoration-mask -> decoration color.
# d) Placement legality
#    - Destination blocks keep the original 5x5 grid layout and 1-cell gaps.
#    - Motifs keep the original 3x3 slot grid with 1-cell gaps between slots.
#    - Regions are still separated along one axis, so no motif can touch a block and accidentally merge into the same
#      multicolor component.
#    - The shared decoration color repeats across motifs, but slot spacing prevents accidental component merging.
# e) Visual observability
#    - In the input, every motif obviously has two colors, and one of them always matches exactly one big block.
#    - In the output, every transformed block contains both holes and decoration-colored pixels. This repeated pattern
#      makes the new color-role rule inferable by eye.
#    - No invisible ordering or color-id arithmetic is used; all matching is based on visible color equality.
# f) Visual neatness
#    - The variant avoids extra overlaid markers or labels. The only new visual element is the second motif color,
#      reused consistently.
#    - The packed output remains the same clean bordered grid as in the original task.
# g) Color-change inferability
#    - The decoration color is shared globally and appears both in the source motifs and in the packed output blocks.
#      This gives a direct structural clue that those pixels are copied as color rather than erased.
# h) Variant-validity metric used after implementation
#    - Compare the new output with the old carve-only output on the same generated input.
#      A nonzero count of interior decoration-colored pixels means the sample truly exercises the new rule.
# i) New solver assumptions and generator guarantees
#    - Solver assumption: every useful motif is an 8-connected non-bg component with bbox <= 3x3 and exactly two colors,
#      of which exactly one is a block color. Generator guarantee: all placed motifs satisfy this.
#    - Solver assumption: the decoration color is not itself a block color. Generator guarantee: decoration color is
#      drawn from the palette before block and distractor colors are chosen.
#    - Solver assumption: each block color appears in exactly one motif. Generator guarantee: one matching motif per
#      block color is always placed, distractor primary colors are distinct and disjoint from block colors.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given local coords."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    """Extract one component into its tight bbox while preserving its colors."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use two-color motifs as dual-action stencils for solid color blocks, then pack the transformed blocks.

    Input structure:
    - The background is the most common color.
    - Several destination objects are solid 5x5 monochrome squares with distinct non-background colors.
    - Elsewhere are small 8-connected motifs in 3x3-or-smaller bboxes; each useful motif has exactly two colors:
      one matches a destination block, and the other is a decoration color. Extra distractor motifs may also appear.

    Rule:
    - Match each block to the unique small motif that contains the block's color.
    - Center that motif on the 5x5 block. Cells of the matching motif color turn into background holes, while cells
      of the motif's other color repaint those block cells to that decoration color.
    - Pack all transformed 5x5 blocks into a compact output grid, preserving the original row/column order with
      1-cell gaps and a 1-cell background border.

    Key insight:
    - The motif is not copied literally. Its two colors encode two different actions inside the block: erase the
      matching-color cells, but keep the decoration-color cells as a visible inlaid color.
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

    # STEP 2) Identify multicolor motifs. A useful motif is a small 8-connected component whose colors are
    # exactly {block_color, decoration_color}, with the block color unique among destination blocks.
    block_colors = {int(b["col"]) for b in blocks}
    motif_candidates: dict[int, list[tuple[int, int, int, int, np.ndarray, np.ndarray, int]]] = {}

    comps_multi = find_components(g, bg=bg, connectivity=8, monochromatic=False, min_area=4, return_format="dict")
    for comp in comps_multi:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue
        if not (4 <= int(comp["area"]) <= 8):
            continue

        patch = component_patch(g, comp, bg)
        colors = sorted({int(v) for v in patch.flatten() if int(v) != int(bg)})
        if len(colors) != 2:
            continue

        matching = [int(c) for c in colors if int(c) in block_colors]
        if len(matching) != 1:
            continue

        match_color = int(matching[0])
        deco_color = int(colors[0] if colors[1] == match_color else colors[1])
        primary_mask = patch == int(match_color)
        deco_mask = patch == int(deco_color)
        if not primary_mask.any() or not deco_mask.any():
            continue

        r0, c0, _r1, _c1 = map(int, b)
        motif_candidates.setdefault(match_color, []).append(
            (
                int(comp["area"]),
                int(patch.shape[0] * patch.shape[1]),
                int(r0),
                int(c0),
                primary_mask,
                deco_mask,
                int(deco_color),
            )
        )

    motifs: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for block_color, candidates in motif_candidates.items():
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        _area, _bbox_area, _r0, _c0, primary_mask, deco_mask, deco_color = candidates[0]
        motifs[int(block_color)] = (primary_mask, deco_mask, int(deco_color))

    # STEP 3) Transform each 5x5 tile by centering the matched motif and applying its two color roles.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motif = motifs.get(int(col))
        if motif is not None:
            primary_mask, deco_mask, deco_color = motif
            mh, mw = map(int, primary_mask.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(primary_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)
            for rr, cc in np.argwhere(deco_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(deco_color)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack transformed 5x5 tiles into a compact grid.
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
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x, D3 variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
DECOR_AREA_RANGE = (1, 3)
MIN_PRIMARY_AREA = 3
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_patterns_by_area() -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Precompute valid 3x3 two-color motif patterns keyed by total occupied area."""
    min_area, max_area = MOTIF_AREA_RANGE
    patterns: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        shape = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(shape.sum())
        if area not in patterns:
            continue
        if not (shape[0].any() and shape[-1].any() and shape[:, 0].any() and shape[:, -1].any()):
            continue
        if not is_contiguous(shape.astype(int), background=0, connectivity=8):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
        max_decor = min(int(DECOR_AREA_RANGE[1]), area - 1)
        for decor_area in range(int(DECOR_AREA_RANGE[0]), max_decor + 1):
            for idxs in itertools.combinations(range(len(coords)), decor_area):
                deco = np.zeros_like(shape, dtype=bool)
                for idx in idxs:
                    r, c = coords[idx]
                    deco[r, c] = True
                primary = shape & ~deco
                if int(primary.sum()) < MIN_PRIMARY_AREA:
                    continue
                patterns[area].append((primary.copy(), deco.copy()))
    return patterns


PATTERNS_BY_AREA = _make_patterns_by_area()


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


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
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
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, filled))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_pattern() -> tuple[np.ndarray, np.ndarray]:
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(PATTERNS_BY_AREA[area])


def _choose_palette(bg: int, count: int, filled: int) -> tuple[int, list[int], list[int]]:
    """Pick one shared decoration color, then block colors, then distractor primary colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    decor_color = palette[0]
    block_colors = palette[1 : 1 + count]
    distractor_colors = palette[1 + count : 1 + count + (filled - count)]
    return int(decor_color), [int(c) for c in block_colors], [int(c) for c in distractor_colors]


def _paint_motif(
    grid: np.ndarray,
    r0: int,
    c0: int,
    primary_color: int,
    decor_color: int,
    bg: int,
    pattern: tuple[np.ndarray, np.ndarray],
) -> None:
    primary_mask, decor_mask = pattern
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(bg), dtype=int)
    patch[primary_mask] = int(primary_color)
    patch[decor_mask] = int(decor_color)
    paste(grid, patch, r0, c0, mask=(primary_mask | decor_mask))


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            primary_mask, decor_mask = motif_patterns[color]
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            sub = tile[motif_slice, motif_slice]
            sub[primary_mask] = int(bg)
            sub[decor_mask] = int(decor_color)
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
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int],
    distractor_colors: list[int],
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    items: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = [
        (int(color), motif_patterns[color]) for color in block_colors
    ]
    items.extend((int(color), _random_pattern()) for color in distractor_colors)
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
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
    bg: int,
) -> None:
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, motif_patterns)

    for (sr, sc), (primary_color, pattern) in zip(slots, slot_items):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            int(primary_color),
            int(decor_color),
            int(bg),
            pattern,
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
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_patterns,
        decor_color,
        bg,
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
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
    decor_color, block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_patterns = {color: _random_pattern() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_patterns, decor_color)
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
        motif_patterns,
        decor_color,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D3 input")

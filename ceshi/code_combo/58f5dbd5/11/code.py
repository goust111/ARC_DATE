# Combo variant of 58f5dbd5 (index 11)
# combo_spec: D2/0+D4/2
# Name: 颜色频次消歧+重复颜色干扰

from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D2/0 + D4/2 combo variant
# ==================================================
# a) Concrete behavior and trace
#    - Original task: each block color had one meaningful same-colored motif, so color match
#      alone recovered the stencil. The original solver happened to sort by area, but that was
#      effectively a no-op because there was no same-color competition.
#    - Combo task: every block color now appears on three 3x3 motifs:
#      two copies of the true stencil shape plus one different same-colored decoy.
#      Example color pool for a red block:
#      red motifs = [plus(area 7), plus(area 7), bent-L(area 4)].
#      Original / D4-only smallest-area behavior would pick bent-L because it is the smallest
#      same-colored component. The combo behavior must instead count exact shapes inside the red
#      pool, notice that plus occurs twice, and carve the plus-shaped hole.
#    - Visible consequence: repeated-color interference is real in the input, and the output hole
#      matches the repeated same-colored shape rather than the smallest same-colored one.
# b) Complexity calibration
#    - The sample is intentionally restricted to 3 destination blocks. That allows a full 3x3 motif
#      board with 9 occupied slots = 3 colors * (2 winners + 1 loser) while still fitting inside
#      the original 19..22 x 19..24 canvas budget with separated regions.
#    - Every color pool uses the same 2-vs-1 support pattern. This avoids under-exercising the
#      combo rule and gives a human solver repeated evidence that "frequency beats size".
#    - The losing same-colored motif is always strictly smaller than the winner, so the combo does
#      not collapse back to D4/2's smallest-area rule.
# c) Hidden assumptions review
#    - STEP 1 still assumes destination objects are detectable as solid 5x5 monochrome squares.
#      The generator preserves that exactly.
#    - The original STEP 2 assumed one meaningful same-colored motif per block color and chose the
#      smallest one. That assumption is false now. The solver must instead group same-colored
#      non-block components by exact mask shape and compare counts.
#    - Exact mask grouping is only safe if identical shapes have identical cropped bboxes. The
#      generator enforces that every motif touches all four edges of its 3x3 slot, so identical
#      shapes always crop to the same 3x3 mask rather than shifted sub-bboxes.
#    - STEP 3 still assumes the chosen motif bbox fits inside the 5x5 block when centered. Since
#      all motifs remain 3x3-slot sprites, that remains valid.
#    - STEP 4 still assumes destination blocks form a rectangular row/column layout recoverable
#      from sorted unique top-left rows and columns. The generator keeps the original 1x3 or 3x1
#      layouts, so that assumption still holds.
# d) Placement legality
#    - Destination blocks keep the original 5x5 bodies with 1-cell gaps between adjacent blocks.
#    - Motifs keep 3x3 slots with 1-cell gaps, so even identical-color repeated motifs never merge
#      into one component under 8-connectivity.
#    - The destination region and motif region are still placed in separate axis-split bands with
#      at least one background row/column between them, so there is no overlap or accidental merge.
# e) Visual observability
#    - Humans can see the rule from examples because each block color visibly appears on three motifs:
#      one shape repeats twice and one different same-colored shape appears once.
#    - The singleton same-colored decoy is also smaller, so examples consistently demonstrate that
#      the chosen output hole is not "the smallest same-colored motif"; it is "the repeated one".
#    - No decision depends on color ids, coordinates, or any hidden ordering. Frequency and shape
#      are directly observable.
# f) Visual cleanliness
#    - All repeated-color competition is confined to the motif panel, which remains a tidy 3x3 slot
#      lattice. The block panel stays unchanged and regular.
#    - Because every slot is isolated by background and every block color is distinct, the repeated
#      same-colored motifs read as separate candidates rather than clutter.
# g) New solver assumptions and generator guarantees
#    - Solver assumption: for each block color there is at least one same-colored non-block motif.
#      Generator guarantee: exactly three such motifs exist per block color.
#    - Solver assumption: the highest-frequency same-colored mask is unique. Generator guarantee:
#      frequency split is always 2 vs 1, never tied.
#    - Solver assumption: the combo meaningfully differs from the old smallest-area heuristic.
#      Generator guarantee: each loser has strictly smaller area than its winner, so the baseline
#      smallest-area solver would disagree on every block color.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given relative coordinates."""
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
    """Return a hashable signature for an exact motif shape."""
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve repeated same-colored motifs into solid blocks and repack the carved blocks.

    Input structure:
    - The background is the most common color.
    - One region contains several solid 5x5 monochrome blocks, each in a distinct non-background color.
    - A separate motif region contains small monochrome sprites in 3x3 slots; for each block color, one
      same-colored motif shape appears multiple times while a different same-colored motif is present as a decoy.

    Rule:
    - For each block color, collect all same-colored non-block motifs and group them by exact shape.
    - Choose the most frequent same-colored shape, center it inside the matching 5x5 block, and turn those cells
      into background to carve the hole.
    - Pack all carved blocks into an output grid that preserves the original row/column order, with 1-cell gaps
      between blocks and a 1-cell background border.

    Key insight:
    - Color only tells you which motif pool belongs to a block; the true stencil is the same-colored shape that
      repeats most often, not the smallest or first one you see.
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

    # STEP 2) For each block color, group same-colored non-block motifs by exact shape and
    # choose the most frequent group.
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
        mask = motifs[int(col)]
        mh, mw = map(int, mask.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mask):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        rr = int(1 + r_index[int(r0)] * step)
        cc = int(1 + c_index[int(c0)] * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x-frequency-repeat) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (19, 22)
CANVAS_WIDTH_RANGE = (19, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUT = (3, 3)
WINNER_REPEAT_COUNT = 2
LOSER_REPEAT_COUNT = 1
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
    """Precompute valid 3x3 connected motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(mask.sum())
        if area not in masks:
            continue
        if not (
            mask[0].any()
            and mask[-1].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        ):
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


def _configs(H: int, W: int) -> list[tuple[int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int]] = []
    Mr, Mc = MOTIF_LAYOUT
    mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    for count, dst_layouts in DST_LAYOUTS.items():
        assert count == 3
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            if _can_split(H, W, dh, dw, mh, mw):
                out.append((R, C, Mr, Mc))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int] | None:
    """Sample one feasible layout."""
    configs = _configs(H, W)
    if not configs:
        return None
    return random.choice(configs)


def _random_mask(area_range: tuple[int, int] = MOTIF_AREA_RANGE) -> np.ndarray:
    """Sample one motif mask from the precomputed masks within the given area range."""
    lo, hi = area_range
    area = random.randint(int(lo), int(hi))
    return random.choice(MASKS_BY_AREA[area]).copy()


def _mask_difference(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing cells between two equal-size boolean masks."""
    return int(np.count_nonzero(a.astype(bool) ^ b.astype(bool)))


def _choose_frequency_pair() -> tuple[np.ndarray, np.ndarray]:
    """Pick a repeated winner and a smaller same-colored loser."""
    for _ in range(200):
        winner = _random_mask(WINNER_AREA_RANGE)
        loser = _random_mask(LOSER_AREA_RANGE)
        if int(loser.sum()) >= int(winner.sum()):
            continue
        if _mask_difference(winner, loser) < MIN_MASK_XOR:
            continue
        return winner, loser
    raise RuntimeError("failed to sample a distinct winner/loser motif pair")


def _choose_palette(bg: int, count: int) -> list[int]:
    """Pick distinct destination colors from the non-background palette."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
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
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            tile[motif_slice, motif_slice][winner_masks[int(color)]] = int(bg)
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
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _slot_items(
    block_colors: list[int],
    winner_masks: dict[int, np.ndarray],
    loser_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled 9-slot payload: 2 winners + 1 loser for every block color."""
    items: list[tuple[int, np.ndarray]] = []
    for color in block_colors:
        items.extend(
            (int(color), winner_masks[int(color)].copy())
            for _ in range(WINNER_REPEAT_COUNT)
        )
        items.extend(
            (int(color), loser_masks[int(color)].copy())
            for _ in range(LOSER_REPEAT_COUNT)
        )
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
    loser_masks: dict[int, np.ndarray],
) -> None:
    """Assign repeated winners and singleton same-colored losers to the motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    items = _slot_items(block_colors, winner_masks, loser_masks)
    assert len(slots) == len(items)

    for (sr, sc), (color, mask) in zip(slots, items):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            color,
            mask,
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
    winner_masks: dict[int, np.ndarray],
    loser_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, winner_masks, loser_masks
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
    loser_masks: dict[int, np.ndarray] = {}
    for color in block_colors:
        winner, loser = _choose_frequency_pair()
        winner_masks[int(color)] = winner
        loser_masks[int(color)] = loser

    expected = _render_expected(bg, R, C, block_colors, winner_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, winner_masks, loser_masks, placement
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

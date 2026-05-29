# Combo variant of 58f5dbd5 (index 35)
# combo_spec: D6/2+D17/2
# Name: 嵌套布局+四象限分离

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Original task: detect solid 5x5 color blocks, find the smaller same-colored
#      motif for each, center-carve that motif into the block as background holes,
#      then repack the carved blocks as a compact RxC rectangle.
#    - Fused behavior:
#      1. The carving semantics stay the same.
#      2. The source blocks no longer encode a rectangular grid. They encode a
#         two-level nested layout with exactly five blocks:
#         four outer-corner blocks plus one inner center block.
#      3. The source scene is also split by quadrants. The four outer-corner blocks
#         live together as a 2x2 mini-grid inside one quadrant, while the single
#         center-level block lives in the diagonally opposite quadrant. Motifs occupy
#         the other diagonal pair as two 2x2 motif-slot chunks.
#      Small trace:
#      - Original 2x2 case: blocks might be at rows [1,1,7,7], cols [1,7,1,7], so
#        sorting unique rows/cols already gives the packed output order.
#      - Combo case: suppose the outer 2x2 block chunk is in TL and the singleton
#        center block is in BR. The old unique-row/col packer would place them at
#        output slots {(0,0),(0,1),(1,0),(1,1),(2,2)}. The intended combo output is
#        different: the same carved tiles must occupy the nested 3x3 pattern
#        {(0,0),(0,2),(2,0),(2,2),(1,1)}.
# b) Complexity calibration
#    - The nested cue must be visible in every sample, so the generator fixes exactly
#      five destination blocks. Allowing 3-4 blocks would often collapse back to a
#      non-nested pattern; allowing many more would overfill the original 18-24 sized
#      canvas once the four-quadrant split is also enforced.
#    - The motif side keeps the original small connected 3x3 stencils and 6-8 total
#      occupied slots. This preserves the color-matching reasoning while making the
#      new global layout visually distinct.
# c) Implicit assumptions audited against `main()`
#    - Step 1 still assumes destination objects are solid 5x5 monochrome squares.
#      Generator preserves that exactly.
#    - Step 2 still assumes each block color has one smaller same-colored component
#      that should be used as the carving stencil. Generator keeps block colors unique
#      and never reuses those colors for distractors.
#    - Original step 4 assumed the block arrangement was recoverable from sorted
#      unique source rows/cols. That assumption is false here because the intended
#      output is a nested pattern, not a full Cartesian product of those coordinates.
#    - The D17/2 row-split/col-split solver would also be invalid here: the diagonal
#      split is by nested level, not by contiguous logical rows or columns.
# d) Placement legality
#    - The four quadrants contain non-overlapping chunks with fixed maximum sizes:
#      11x11 for the outer 2x2 block chunk, 5x5 for the singleton center block, and
#      7x7 for each 2x2 motif chunk. Restricting the canvas to 22-24 by 22-24 lets
#      every chunk stay fully inside its visible midpoint-defined quadrant while
#      preserving a 1-cell outer margin.
#    - All motifs are placed in 3x3 slots on a 4-step lattice, so same-colored block
#      and motif components remain 8-disconnected and never merge.
# e) Visual observability
#    - Humans can still see the stable local cue: one large 5x5 solid square and one
#      smaller same-colored motif belong together.
#    - The fused behavior is globally visible because the blocks appear as a 4+1 split
#      across diagonal quadrants, and the output always restores the corner-plus-center
#      nested layout rather than any rectangular packing.
# f) Visual cleanliness
#    - Each occupied quadrant contains one clean regular chunk: either a 2x2 block
#      grid, one singleton 5x5 block, or a 2x2 motif-slot patch. No quadrant mixes
#      blocks and motifs, and no objects overlap.
# g) New solver assumptions and generator guarantees
#    - Solver assumption: exactly two diagonally opposite quadrants contain blocks,
#      with counts 4 and 1, and the 4-block quadrant is one of the two upper
#      quadrants so midpoint-based quadrant decoding is stable. Generator guarantees
#      this explicitly rather than relying on rejection.
#    - Solver assumption: the 4-block quadrant forms a proper 2x2 local lattice with
#      two distinct source rows and two distinct source columns. Generator guarantees
#      that as well.
#    - Solver assumption: the remaining two quadrants contain all motifs. Generator
#      always fills both motif quadrants so the four-way split is visible in every
#      sample.


# --------------- shared constants ---------------

CANVAS_HEIGHT_RANGE = (22, 24)
CANVAS_WIDTH_RANGE = (22, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)

OUTER_MARGIN = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
GENERATION_RETRIES = 500

OUTER_BLOCK_GRID = (2, 2)
OUTER_BLOCK_SPAN = (11, 11)
INNER_BLOCK_SPAN = (5, 5)
MOTIF_CHUNK_GRID = (2, 2)
MOTIF_CHUNK_SPAN = (7, 7)

BLOCK_COUNT = 5
FILLED_SLOT_RANGE = (6, 8)

QUADRANTS = ("TL", "TR", "BL", "BR")
OPPOSITE = {"TL": "BR", "BR": "TL", "TR": "BL", "BL": "TR"}
OUTER_QUAD_BRANCHES = ("TL", "TR")

OUTER_OFFSETS = [(0, 0), (0, DST_BLOCK_STEP), (DST_BLOCK_STEP, 0), (DST_BLOCK_STEP, DST_BLOCK_STEP)]
NESTED_OUTPUT_SLOTS = {
    "TL": (0, 0),
    "TR": (0, 2),
    "BL": (2, 0),
    "BR": (2, 2),
    "C": (1, 1),
}


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


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
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
GENERATE_CALL_COUNT = 0


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _quadrant_bounds(H: int, W: int, quadrant: str, h: int, w: int) -> tuple[int, int, int, int] | None:
    """Return inclusive origin bounds that keep an object fully inside one visible quadrant."""
    mid_r = H // 2
    mid_c = W // 2

    if quadrant == "TL":
        r_min, r_max = OUTER_MARGIN, mid_r - h
        c_min, c_max = OUTER_MARGIN, mid_c - w
    elif quadrant == "TR":
        r_min, r_max = OUTER_MARGIN, mid_r - h
        c_min, c_max = mid_c, W - OUTER_MARGIN - w
    elif quadrant == "BL":
        r_min, r_max = mid_r, H - OUTER_MARGIN - h
        c_min, c_max = OUTER_MARGIN, mid_c - w
    else:
        r_min, r_max = mid_r, H - OUTER_MARGIN - h
        c_min, c_max = mid_c, W - OUTER_MARGIN - w

    if r_min > r_max or c_min > c_max:
        return None
    return int(r_min), int(r_max), int(c_min), int(c_max)


def _place_quadrants(H: int, W: int, reqs: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]] | None:
    """Choose one origin per quadrant while keeping each chunk fully inside its visible quadrant."""
    origins: dict[str, tuple[int, int]] = {}
    for quadrant, (h, w) in reqs.items():
        bounds = _quadrant_bounds(H, W, quadrant, int(h), int(w))
        if bounds is None:
            return None
        r_min, r_max, c_min, c_max = bounds
        origins[quadrant] = (
            random.randint(int(r_min), int(r_max)),
            random.randint(int(c_min), int(c_max)),
        )
    return origins


def _choose_placement(H: int, W: int, outer_quad: str) -> dict[str, object] | None:
    """Sample one diagonal four-quadrant arrangement for blocks and motifs."""
    inner_quad = OPPOSITE[outer_quad]
    motif_quads = [q for q in QUADRANTS if q not in {outer_quad, inner_quad}]
    reqs = {
        outer_quad: OUTER_BLOCK_SPAN,
        inner_quad: INNER_BLOCK_SPAN,
        motif_quads[0]: MOTIF_CHUNK_SPAN,
        motif_quads[1]: MOTIF_CHUNK_SPAN,
    }
    origins = _place_quadrants(H, W, reqs)
    if origins is None:
        return None
    return {
        "outer_quad": outer_quad,
        "inner_quad": inner_quad,
        "motif_quads": tuple(motif_quads),
        "origins": origins,
    }


def _quadrant_of_point(r0: int, c0: int, H: int, W: int) -> str:
    """Classify a top-left point into one of the four visible quadrants."""
    top = int(r0) < H // 2
    left = int(c0) < W // 2
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Extract all solid 5x5 destination blocks."""
    blocks: list[dict[str, int]] = []
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != DST_BLOCK_SIZE or bbox_w(b) != DST_BLOCK_SIZE:
            continue
        col = int(comp["color"])
        patch = crop_bbox(grid, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})
    return blocks


def _find_motifs(grid: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, np.ndarray]:
    """For each block color, recover the smallest same-colored non-block component."""
    motifs: dict[int, np.ndarray] = {}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(bcol)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == DST_BLOCK_SIZE and bbox_w(b) == DST_BLOCK_SIZE:
                patch = crop_bbox(grid, b)
                if bool((patch == int(bcol)).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]
    return motifs


def _carve_tiles(blocks: list[dict[str, int]], motifs: dict[int, np.ndarray], bg: int) -> dict[tuple[int, int], np.ndarray]:
    """Center-carve each matched motif mask into its 5x5 solid block."""
    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        col = int(block["col"])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(col), dtype=int)
        motif_mask = motifs.get(int(col))
        if motif_mask is not None:
            mh, mw = map(int, motif_mask.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(motif_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                    tile[tr, tc] = int(bg)
        carved[(int(block["r0"]), int(block["c0"]))] = tile
    return carved


def _decode_nested_layout(blocks: list[dict[str, int]], H: int, W: int) -> dict[str, dict[str, int]]:
    """Recover which source block fills each nested output slot."""
    assert len(blocks) == BLOCK_COUNT

    groups: dict[str, list[dict[str, int]]] = {}
    for block in blocks:
        q = _quadrant_of_point(int(block["r0"]), int(block["c0"]), H, W)
        groups.setdefault(q, []).append(block)

    assert len(groups) == 2
    counts = sorted(len(group) for group in groups.values())
    assert counts == [1, 4]

    outer_quad = next(q for q, group in groups.items() if len(group) == 4)
    inner_quad = next(q for q, group in groups.items() if len(group) == 1)
    assert OPPOSITE[outer_quad] == inner_quad

    outer_group = groups[outer_quad]
    outer_rows = sorted({int(block["r0"]) for block in outer_group})
    outer_cols = sorted({int(block["c0"]) for block in outer_group})
    assert len(outer_rows) == 2 and len(outer_cols) == 2

    decoded: dict[str, dict[str, int]] = {}
    for block in outer_group:
        i = outer_rows.index(int(block["r0"]))
        j = outer_cols.index(int(block["c0"]))
        if i == 0 and j == 0:
            decoded["TL"] = block
        elif i == 0 and j == 1:
            decoded["TR"] = block
        elif i == 1 and j == 0:
            decoded["BL"] = block
        else:
            decoded["BR"] = block
    decoded["C"] = groups[inner_quad][0]
    return decoded


def _render_nested_output(decoded: dict[str, dict[str, int]], carved: dict[tuple[int, int], np.ndarray], bg: int) -> np.ndarray:
    """Render the nested corner-plus-center packed output."""
    out_h = int(2 * OUTPUT_BORDER + 3 * DST_BLOCK_SIZE + 2 * 1)
    out_w = int(2 * OUTPUT_BORDER + 3 * DST_BLOCK_SIZE + 2 * 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for key, (i, j) in NESTED_OUTPUT_SLOTS.items():
        block = decoded[key]
        rr = int(OUTPUT_BORDER + i * DST_BLOCK_STEP)
        cc = int(OUTPUT_BORDER + j * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[(int(block["r0"]), int(block["c0"]))]
    return out


def _legacy_slot_signature(blocks: list[dict[str, int]]) -> set[tuple[int, int]]:
    """Return the slot coordinates the original unique-row/col packer would use."""
    rs = sorted({int(block["r0"]) for block in blocks})
    cs = sorted({int(block["c0"]) for block in blocks})
    return {
        (rs.index(int(block["r0"])), cs.index(int(block["c0"])))
        for block in blocks
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into nested-level blocks and rebuild the nested output.

    Input structure:
    - The background is the majority color.
    - Exactly five solid 5x5 monochrome blocks are split across two diagonally opposite quadrants.
    - One occupied block quadrant contains a 2x2 chunk of outer-corner blocks, and the opposite occupied block quadrant contains the singleton inner-center block.
    - Same-colored motif sprites, plus optional distractor motifs of other colors, occupy the remaining diagonal pair in two separate 2x2 motif chunks.

    Rule:
    - Match each small motif to the 5x5 solid block of the same color.
    - Center that motif inside the block and repaint the motif cells to background, carving holes into the block.
    - Ignore the source quadrants when arranging the result: reconstruct the hidden nested layout by sending the 4-block chunk to the four output corners and the singleton block to the output center, all with the usual 1-cell gaps and 1-cell border.

    Key insight:
    - Color still tells you which motif carves which block, but position is no longer a rectangular pack order. The four-quadrant source scene hides a corner-plus-center nested structure that must be restored in the output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks = _find_blocks(g, bg)
    assert len(blocks) == BLOCK_COUNT

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 5x5 blocks.
    block_colors = {int(block["col"]) for block in blocks}
    motifs = _find_motifs(g, bg, block_colors)

    # STEP 3) Carve each 5x5 tile by centering the matched motif and repainting
    # those cells to background.
    carved = _carve_tiles(blocks, motifs, bg)

    # STEP 4) Recover the hidden nested 4-corners-plus-center layout from the
    # diagonal quadrant split, then render the nested packed output.
    decoded = _decode_nested_layout(blocks, H, W)
    return _render_nested_output(decoded, carved, bg)


# --------------- generator ---------------


def _choose_palette(bg: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick block colors first, then distractor colors, from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:BLOCK_COUNT]
    distractor_colors = palette[BLOCK_COUNT : BLOCK_COUNT + (filled - BLOCK_COUNT)]
    return block_colors, distractor_colors


def _render_expected(bg: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]) -> np.ndarray:
    """Render the oracle nested output corresponding to the sampled color assignment."""
    out_h = int(2 * OUTPUT_BORDER + 3 * DST_BLOCK_SIZE + 2 * 1)
    out_w = int(2 * OUTPUT_BORDER + 3 * DST_BLOCK_SIZE + 2 * 1)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    slot_to_color = {
        "TL": int(block_colors[0]),
        "TR": int(block_colors[1]),
        "BL": int(block_colors[2]),
        "BR": int(block_colors[3]),
        "C": int(block_colors[4]),
    }
    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for key, (i, j) in NESTED_OUTPUT_SLOTS.items():
        color = int(slot_to_color[key])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        tile[motif_slice, motif_slice][motif_masks[color]] = int(bg)
        paste(
            expected,
            tile,
            OUTPUT_BORDER + i * DST_BLOCK_STEP,
            OUTPUT_BORDER + j * DST_BLOCK_STEP,
        )
    return expected


def _paint_block(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    """Paint one solid 5x5 block."""
    paste(grid, np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int), int(r0), int(c0))


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its boolean mask."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, int(r0), int(c0), mask=mask)


def _sample_motif_quadrant_fill(filled: int) -> tuple[int, int]:
    """Split the occupied motif slots across the two motif quadrants."""
    for _ in range(50):
        a = random.randint(2, 4)
        b = int(filled - a)
        if 2 <= b <= 4:
            return int(a), int(b)
    raise AssertionError("failed to split motif occupancy across quadrants")


def _sample_matching_partition(quad_fill: tuple[int, int]) -> tuple[int, int]:
    """Split the five matching motifs across the two visible motif quadrants."""
    qa, qb = quad_fill
    options = []
    for a in (2, 3):
        b = BLOCK_COUNT - a
        if a <= qa and b <= qb:
            options.append((a, b))
    assert options
    return random.choice(options)


def _render_input(
    H: int,
    W: int,
    bg: int,
    placement: dict[str, object],
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    filled: int,
) -> np.ndarray:
    """Render the four-quadrant fused input scene."""
    inp = np.full((H, W), int(bg), dtype=int)

    origins = dict(placement["origins"])
    outer_quad = str(placement["outer_quad"])
    inner_quad = str(placement["inner_quad"])
    motif_quad_a, motif_quad_b = placement["motif_quads"]

    outer_origin = origins[outer_quad]
    for (dr, dc), color in zip(OUTER_OFFSETS, block_colors[:4]):
        _paint_block(inp, int(outer_origin[0] + dr), int(outer_origin[1] + dc), int(color))

    inner_origin = origins[inner_quad]
    _paint_block(inp, int(inner_origin[0]), int(inner_origin[1]), int(block_colors[4]))

    quad_fill = _sample_motif_quadrant_fill(int(filled))
    match_fill = _sample_matching_partition(quad_fill)

    matching_items = [(int(color), motif_masks[int(color)]) for color in block_colors]
    random.shuffle(matching_items)
    distractor_items = [(int(color), _random_mask()) for color in distractor_colors]
    random.shuffle(distractor_items)

    quad_specs = [
        (str(motif_quad_a), int(quad_fill[0]), int(match_fill[0])),
        (str(motif_quad_b), int(quad_fill[1]), int(match_fill[1])),
    ]
    for quadrant, occ, match_count in quad_specs:
        slots = [(0, 0), (0, MOTIF_STEP), (MOTIF_STEP, 0), (MOTIF_STEP, MOTIF_STEP)]
        random.shuffle(slots)
        origin = origins[quadrant]
        items = [matching_items.pop() for _ in range(match_count)]
        need_distractors = int(occ - match_count)
        items.extend(distractor_items.pop() for _ in range(need_distractors))
        random.shuffle(items)
        for (dr, dc), (color, mask) in zip(slots[:occ], items):
            _paint_motif(inp, int(origin[0] + dr), int(origin[1] + dc), int(color), mask)

    assert not matching_items
    assert not distractor_items
    return inp


def _gen_attempt(outer_quad: str) -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle or behavior checks fail."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)
    filled = random.randint(*FILLED_SLOT_RANGE)

    placement = _choose_placement(H, W, outer_quad)
    if placement is None:
        return None

    block_colors, distractor_colors = _choose_palette(bg, filled)
    motif_masks = {int(color): _random_mask() for color in block_colors}

    expected = _render_expected(bg, block_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        placement,
        block_colors,
        distractor_colors,
        motif_masks,
        filled,
    )

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    metrics = combo_behavior_metrics(inp)
    if metrics["diagonal_block_split"] != 1:
        return None
    if metrics["motif_quadrant_split"] != 1:
        return None
    if metrics["legacy_slot_pattern_differs"] != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    global GENERATE_CALL_COUNT
    target_outer_quad = OUTER_QUAD_BRANCHES[GENERATE_CALL_COUNT % len(OUTER_QUAD_BRANCHES)]
    GENERATE_CALL_COUNT += 1
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(target_outer_quad)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a generated sample actually expresses the fused behavior.

    Returns:
        diagonal_block_split:
            1 iff the five solid blocks occupy exactly two opposite quadrants with
            counts 4 and 1.
        motif_quadrant_split:
            1 iff non-block components occupy exactly the complementary diagonal pair.
        legacy_slot_pattern_differs:
            1 iff the intended nested output slots differ from the original unique
            row/column packing slots, so the sample does not collapse back to the
            base-task layout rule.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    blocks = _find_blocks(g, bg)
    block_groups: dict[str, int] = {}
    for block in blocks:
        q = _quadrant_of_point(int(block["r0"]), int(block["c0"]), H, W)
        block_groups[q] = block_groups.get(q, 0) + 1
    nonzero_blocks = {q: n for q, n in block_groups.items() if n > 0}
    diagonal_block_split = int(
        len(nonzero_blocks) == 2
        and sorted(nonzero_blocks.values()) == [1, 4]
        and any(OPPOSITE[q] in nonzero_blocks for q in nonzero_blocks)
    )

    motif_groups: dict[str, int] = {}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        color = int(comp["color"])
        if bbox_h(b) == DST_BLOCK_SIZE and bbox_w(b) == DST_BLOCK_SIZE:
            patch = crop_bbox(g, b)
            if bool((patch == color).all()):
                continue
        r0, c0, _r1, _c1 = map(int, b)
        q = _quadrant_of_point(int(r0), int(c0), H, W)
        motif_groups[q] = motif_groups.get(q, 0) + 1
    nonzero_motifs = {q: n for q, n in motif_groups.items() if n > 0}

    motif_quadrant_split = 0
    if diagonal_block_split and len(nonzero_motifs) == 2:
        block_quads = set(nonzero_blocks)
        motif_quads = set(nonzero_motifs)
        motif_quadrant_split = int(motif_quads == {q for q in QUADRANTS if q not in block_quads})

    legacy_slots = _legacy_slot_signature(blocks)
    nested_slots = set(NESTED_OUTPUT_SLOTS.values())
    legacy_slot_pattern_differs = int(legacy_slots != nested_slots)

    return {
        "diagonal_block_split": int(diagonal_block_split),
        "motif_quadrant_split": int(motif_quadrant_split),
        "legacy_slot_pattern_differs": int(legacy_slot_pattern_differs),
    }

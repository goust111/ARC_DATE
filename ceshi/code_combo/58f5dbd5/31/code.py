# Combo variant of 58f5dbd5 (index 31)
# combo_spec: D4/0+D13/0
# Name: 无干扰项+仅2个方块

# Design analysis for the combo variant
# - concretization:
#   The base task may show 3-6 solid 5x5 blocks and a larger motif region with matching motifs,
#   distractor motifs, and occasional empty slots. This fused variant collapses that scene to the
#   smallest clean case: exactly two solid 5x5 blocks in a 1x2 or 2x1 grid, plus exactly two small
#   motifs elsewhere, one per block color and no extra objects. A concrete trace:
#   original logic might inspect a red block among several blocks, ignore unrelated motif colors,
#   choose the smallest red non-block component, carve that centered stencil, then repeat for the
#   rest. Here there are only two colors to pair, but the operative relation is still color match:
#   each same-colored motif carves centered background holes into its 5x5 block, and the two carved
#   tiles are then packed with the usual 1-cell border and 1-cell gap. The difference is not a new
#   output transform; it is a deliberately narrower, cleaner input family.
# - complexity calibration:
#   Reducing to two pairs can become too trivial if position alone identifies the mapping, so the
#   generator preserves two important degrees of freedom: block layout is random between horizontal
#   and vertical, motif layout is sampled independently, and when both layouts match the motif order
#   is forced to be reversed. That keeps same-color matching necessary without reintroducing noise.
#   Motifs remain connected 3x3-bounded shapes of area 4-8, so the carved holes stay visually clear.
# - implicit assumption audit:
#   The original solver line "pick the smallest same-colored non-block component" relied on possible
#   distractors of other colors, but not on multiple same-colored candidates. In this combo, a stricter
#   assumption is valid: each block color appears exactly twice in the input, once as a 5x5 block and
#   once as a motif. The code now makes that explicit by asserting exactly two blocks, exactly two block
#   colors, and exactly one non-block component per block color. The packing code also now assumes the
#   blocks form a proper 1x2 or 2x1 lattice, which the generator guarantees via fixed step-6 placement.
# - placement legality:
#   The generator still uses two separated regions, one for destination blocks and one for motifs, with
#   at least a 1-cell outer margin and a 1-cell gap between regions. Blocks are painted on a 6-step
#   lattice (5 body + 1 gap); motifs are painted on a 4-step lattice (3 body + 1 gap). This prevents
#   overlap, out-of-bounds painting, and same-color merging between a block and its motif.
# - visual observability / tidiness / inference:
#   The scene is simpler but still inferable from a few examples because the two colors each appear as
#   one large solid square and one small sprite, and the output shows those two squares with centered
#   cavities shaped like the matching sprite. There are no hidden numeric-color rules: the only cue is
#   visible color equality. Removing distractors also makes the variant visually cleaner rather than
#   cluttered, which is important given the very small object count.
# - anti-degeneration target:
#   Successful samples must satisfy three measurable conditions: exactly two destination blocks, exactly
#   two motif components, and zero extra non-background components beyond those four objects. That proves
#   the generator is actually instantiating the D4/0 + D13/0 restriction instead of drifting back toward
#   the broader original distribution.
# - new_main_assumptions_and_generator_guarantees:
#   `main()` assumes motifs fit when centered in 5x5, that each block color owns one motif, and that the
#   blocks occupy a 1x2 or 2x1 regular grid. `generate_input()` enforces all three by sampling only
#   connected 3x3-bounded masks, using exactly two distinct non-background colors, and placing the blocks
#   on one of the two legal layouts with fixed step-6 spacing.

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


BLOCK_SIZE = 5
BLOCK_STEP = 6
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_OFFSET_IN_BLOCK = 1
OUTPUT_BORDER = 1
OUTER_MARGIN = 1
REGION_GAP = 1

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10
MOTIF_AREA_RANGE = (4, 8)
BLOCK_LAYOUTS = [(1, 2), (2, 1)]
MOTIF_LAYOUTS = [(1, 2), (2, 1)]
GENERATION_RETRIES = 300


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given local coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for `count` equally spaced items."""
    return (count - 1) * step + item_size


def _is_solid_block(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    """Return whether the bbox is a solid monochrome 5x5 destination block."""
    if bbox_h(bbox) != BLOCK_SIZE or bbox_w(bbox) != BLOCK_SIZE:
        return False
    patch = crop_bbox(grid, bbox)
    return bool((patch == int(color)).all())


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Return the two destination blocks."""
    blocks: list[dict[str, int]] = []
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        bbox = comp["bbox"]
        color = int(comp["color"])
        if not _is_solid_block(grid, bbox, color):
            continue
        r0, c0, _r1, _c1 = map(int, bbox)
        blocks.append({"col": color, "r0": r0, "c0": c0})
    blocks.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return blocks


def _find_motifs(grid: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, np.ndarray]:
    """Return the unique non-block motif mask for each block color."""
    motifs: dict[int, np.ndarray] = {}
    for color in sorted(block_colors):
        comps = find_components(
            grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        candidates: list[np.ndarray] = []
        for comp in comps:
            bbox = comp["bbox"]
            if _is_solid_block(grid, bbox, color):
                continue
            r0, c0, r1, c1 = map(int, bbox)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            candidates.append(mask_from_coords(rel_coords, (h, w)))
        assert len(candidates) == 1
        motifs[int(color)] = candidates[0]
    return motifs


def _carved_tile(color: int, bg: int, motif: np.ndarray) -> np.ndarray:
    """Render one carved 5x5 block."""
    tile = np.full((BLOCK_SIZE, BLOCK_SIZE), int(color), dtype=int)
    mh, mw = map(int, motif.shape)
    off_r = (BLOCK_SIZE - mh) // 2
    off_c = (BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(motif):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match two clean motifs to two same-colored blocks, carve the blocks, and repack them.

    Input structure:
    - The dominant color is the background.
    - Exactly two non-background solid 5x5 blocks appear in a 1x2 or 2x1 arrangement, using two distinct colors.
    - Elsewhere, exactly two small monochrome motif sprites appear, one in each block color and with no distractor motifs.

    Rule:
    - For each block color, locate the unique same-colored non-block component and treat it as that block's stencil.
    - Center the stencil inside the corresponding 5x5 block and repaint those covered cells to background.
    - Pack the two carved 5x5 tiles in their original row/column order using a 1-cell border and a 1-cell gap.

    Key insight:
    - The simplification removes filtering noise, but the motif is still a carving tool rather than an object to move: color equality identifies which small sprite hollows out which larger square.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks = _find_blocks(g, bg)
    assert len(blocks) == 2

    block_colors = {int(block["col"]) for block in blocks}
    assert len(block_colors) == 2
    motifs = _find_motifs(g, bg, block_colors)
    assert len(motifs) == 2

    rs = sorted({int(block["r0"]) for block in blocks})
    cs = sorted({int(block["c0"]) for block in blocks})
    assert len(rs) * len(cs) == 2

    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        color = int(block["col"])
        carved[(r0, c0)] = _carved_tile(color, bg, motifs[color])

    r_index = {int(r): idx for idx, r in enumerate(rs)}
    c_index = {int(c): idx for idx, c in enumerate(cs)}
    out_h = int(OUTPUT_BORDER + len(rs) * BLOCK_SIZE + (len(rs) - 1) + OUTPUT_BORDER)
    out_w = int(OUTPUT_BORDER + len(cs) * BLOCK_SIZE + (len(cs) - 1) + OUTPUT_BORDER)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        rr = int(OUTPUT_BORDER + r_index[r0] * BLOCK_STEP)
        cc = int(OUTPUT_BORDER + c_index[c0] * BLOCK_STEP)
        out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = carved[(r0, c0)]

    return out


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute connected 3x3 motif masks that touch all four bbox sides."""
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


def _random_mask() -> np.ndarray:
    """Sample one valid motif mask."""
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_two_distinct_masks() -> list[np.ndarray]:
    """Sample two different motif masks so both carvings are visibly supported."""
    first = _random_mask()
    for _ in range(50):
        second = _random_mask()
        if not np.array_equal(first, second):
            return [first, second]
    return [first, second]


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible separations for the block region and motif region."""
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["blocks_above", "motifs_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["blocks_left", "motifs_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place vertically separated regions and return block then motif origins."""
    if scheme == "blocks_above":
        top_h, top_w, bot_h, bot_w = dh, dw, mh, mw
    else:
        top_h, top_w, bot_h, bot_w = mh, mw, dh, dw

    top_r = random.randint(OUTER_MARGIN, H - top_h - bot_h - OUTER_MARGIN - REGION_GAP)
    bot_r = random.randint(top_r + top_h + REGION_GAP, H - bot_h - OUTER_MARGIN)
    top_c = random.randint(OUTER_MARGIN, W - top_w - OUTER_MARGIN)
    bot_c = random.randint(OUTER_MARGIN, W - bot_w - OUTER_MARGIN)
    if scheme == "blocks_above":
        return top_r, top_c, bot_r, bot_c
    return bot_r, bot_c, top_r, top_c


def _place_horizontal(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place horizontally separated regions and return block then motif origins."""
    if scheme == "blocks_left":
        left_h, left_w, right_h, right_w = dh, dw, mh, mw
    else:
        left_h, left_w, right_h, right_w = mh, mw, dh, dw

    left_c = random.randint(OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP)
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w - OUTER_MARGIN)
    left_r = random.randint(OUTER_MARGIN, H - left_h - OUTER_MARGIN)
    right_r = random.randint(OUTER_MARGIN, H - right_h - OUTER_MARGIN)
    if scheme == "blocks_left":
        return left_r, left_c, right_r, right_c
    return right_r, right_c, left_r, left_c


def _place_regions(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> tuple[int, int, int, int] | None:
    """Choose a legal separated placement for the two regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"blocks_above", "motifs_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _choose_config(H: int, W: int) -> tuple[int, int, int, int] | None:
    """Pick one feasible pair of 2-object layouts."""
    configs: list[tuple[int, int, int, int]] = []
    for R, C in BLOCK_LAYOUTS:
        dh = _spanned_size(R, BLOCK_SIZE, BLOCK_STEP)
        dw = _spanned_size(C, BLOCK_SIZE, BLOCK_STEP)
        for Mr, Mc in MOTIF_LAYOUTS:
            mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
            mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
            if _place_regions(H, W, dh, dw, mh, mw) is not None:
                configs.append((R, C, Mr, Mc))
    if not configs:
        return None
    return random.choice(configs)


def _choose_palette(bg: int) -> list[int]:
    """Pick the two distinct block colors."""
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    random.shuffle(palette)
    return palette[:2]


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the two solid destination blocks."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * BLOCK_STEP, dst_c + j * BLOCK_STEP)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif through its mask."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the packed oracle output for the sampled block layout."""
    out_h = int(2 * OUTPUT_BORDER + _spanned_size(R, BLOCK_SIZE, BLOCK_STEP))
    out_w = int(2 * OUTPUT_BORDER + _spanned_size(C, BLOCK_SIZE, BLOCK_STEP))
    expected = np.full((out_h, out_w), int(bg), dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carved_tile(color, bg, motif_masks[color])
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * BLOCK_STEP,
                OUTPUT_BORDER + j * BLOCK_STEP,
            )
    return expected


def _motif_color_order(
    block_colors: list[int], block_layout: tuple[int, int], motif_layout: tuple[int, int]
) -> list[int]:
    """Choose motif slot order while preventing a pure position shortcut."""
    if block_layout == motif_layout:
        return list(reversed(block_colors))
    colors = list(block_colors)
    if random.random() < 0.5:
        colors.reverse()
    return colors


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    block_layout: tuple[int, int],
) -> None:
    """Paint exactly two motif slots with no distractors and no empties."""
    slot_colors = _motif_color_order(block_colors, block_layout, (Mr, Mc))
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    assert len(slots) == 2
    for (sr, sc), color in zip(slots, slot_colors):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            int(color),
            motif_masks[int(color)],
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
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the two-block, two-motif fused input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_masks, (R, C))
    return inp


def _matches_combo_contract(grid: np.ndarray, bg: int) -> bool:
    """Validate the fused variant contract: 2 blocks, 2 motifs, no extra components."""
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    blocks = []
    motifs = []
    for comp in comps:
        color = int(comp["color"])
        bbox = comp["bbox"]
        if _is_solid_block(grid, bbox, color):
            blocks.append(comp)
        else:
            motifs.append(comp)
    block_colors = {int(comp["color"]) for comp in blocks}
    motif_colors = {int(comp["color"]) for comp in motifs}
    return (
        len(comps) == 4
        and len(blocks) == 2
        and len(motifs) == 2
        and len(block_colors) == 2
        and motif_colors == block_colors
    )


def _gen_attempt() -> np.ndarray | None:
    """Sample one fused input and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc = config

    dst_h = _spanned_size(R, BLOCK_SIZE, BLOCK_STEP)
    dst_w = _spanned_size(C, BLOCK_SIZE, BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    block_colors = _choose_palette(bg)
    masks = _choose_two_distinct_masks()
    motif_masks = {int(color): mask for color, mask in zip(block_colors, masks)}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(H, W, bg, R, C, Mr, Mc, block_colors, motif_masks, placement)
    if not _matches_combo_contract(inp, bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate one valid input for the fused two-block, no-distractor variant."""
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

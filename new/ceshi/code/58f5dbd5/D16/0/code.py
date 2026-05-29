# Variant of 58f5dbd5
# Dimension: D16 — 符号编码 (symbol_instructions)
# Change: 无符号编码，所有信息直接通过颜色和空间关系表达 → 图例指示雕刻方向

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D16 variant
# a) Concrete behavior and trace:
#    The original task centers each 3x3 motif inside its matched 5x5 block, so the carve offset is always (1,1).
#    This variant inserts a 2x2 legend in one corner of the input grid. The legend uses one extra non-bg color on
#    background cells and chooses one of three boolean patterns:
#      upper_left  = [[1,0],[0,0]]
#      center      = [[0,1],[1,0]]
#      lower_right = [[0,0],[0,1]]
#    Example: with a red 3x3 motif matched to a red 5x5 block, the original solver would carve that motif at rows
#    1..3 and cols 1..3 of the block. In this variant, the corner legend may instead force rows 0..2 / cols 0..2
#    or rows 2..4 / cols 2..4. The only changed stage is "how to place the stencil inside the 5x5 tile"; matching
#    by color and output packing stay the same.
# b) Complexity calibration:
#    Three alignments are enough to make the instruction visually meaningful without creating too many branches.
#    The motif masks remain asymmetric enough in practice, and even symmetric masks produce visibly shifted holes
#    because the carve window moves inside the 5x5 block.
# c) Implicit-assumption audit against the old main():
#    - Block detection assumed the only 5x5 monochrome components are targets. That still holds because the legend is 2x2.
#    - Motif lookup assumed the smallest same-colored non-5x5 component is the stencil. That still holds because the
#      legend color is kept distinct from every block/motif color.
#    - The old carve step hard-coded centering via (5-mh)//2 and (5-mw)//2. That assumption is exactly what D16 changes.
#      The new solver must decode the legend first and then choose one of three offsets.
#    - Output packing depends only on the block top-left coordinates, so it is unaffected.
# d) Placement legality:
#    The generator keeps the original separated destination and motif regions, then adds the 2x2 legend in exactly one
#    corner. Placements are rejected if either large region intersects that corner patch, so there is no overwrite,
#    no overlap, and the solver can rely on "only the legend touches the boundary".
# e) Visual observability:
#    A human can compare 2-4 examples and see that the same motif shape sometimes appears carved flush to the upper-left,
#    sometimes centered, and sometimes flush to the lower-right, while a corner mini-symbol changes accordingly.
#    The legend meaning is shape-based, not color-id-based.
# f) Visual cleanliness:
#    The legend sits alone in a corner; destination blocks and motif slots remain in their original separated regions.
#    This avoids mixing symbolic cells into the dense motif area, which would make the instruction harder to isolate.
# g) Color-rule observability:
#    No color remapping is introduced. The extra legend color only marks the symbol itself and does not participate in
#    color matching. This keeps the symbolic change orthogonal to the original same-color block↔motif relation.
# New solver assumptions introduced by D16 and guaranteed by the generator:
#    - Exactly one corner 2x2 patch contains the legend; the other three corner 2x2 patches stay fully background.
#    - That patch matches one of the three known boolean code masks.
#    - The legend color is distinct from all block and motif colors, so motif lookup by color remains unambiguous.


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


ALIGNMENT_MASKS: dict[str, np.ndarray] = {
    "upper_left": np.array([[1, 0], [0, 0]], dtype=bool),
    "center": np.array([[0, 1], [1, 0]], dtype=bool),
    "lower_right": np.array([[0, 0], [0, 1]], dtype=bool),
}

CORNER_SLICES = (
    (slice(0, 2), slice(0, 2)),
    (slice(0, 2), slice(-2, None)),
    (slice(-2, None), slice(0, 2)),
    (slice(-2, None), slice(-2, None)),
)


def _decode_alignment(grid: np.ndarray, bg: int) -> str:
    """Read the 2x2 legend from whichever corner contains the only non-background corner patch."""
    for rs, cs in CORNER_SLICES:
        patch = np.array(grid[rs, cs], dtype=int)
        legend_mask = patch != int(bg)
        if not np.any(legend_mask):
            continue
        for name, code in ALIGNMENT_MASKS.items():
            if np.array_equal(legend_mask, code):
                return name
    raise AssertionError("missing corner legend")


def _alignment_offsets(tile_h: int, tile_w: int, motif_h: int, motif_w: int, mode: str) -> tuple[int, int]:
    """Convert a decoded alignment mode into the row/col offset for placing the motif inside a tile."""
    if mode == "upper_left":
        return 0, 0
    if mode == "center":
        return (tile_h - motif_h) // 2, (tile_w - motif_w) // 2
    if mode == "lower_right":
        return tile_h - motif_h, tile_w - motif_w
    raise ValueError(f"unknown alignment mode: {mode}")


def _solve_with_mode(grid: np.ndarray, forced_mode: str | None = None) -> np.ndarray:
    """Core solver shared by main() and validation helpers."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    alignment = forced_mode or _decode_alignment(g, bg)

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
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each 5x5 tile using the legend-decoded alignment instead of always centering the stencil.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = motifs.get(int(col))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r, off_c = _alignment_offsets(5, 5, mh, mw, alignment)
            for rr, cc in np.argwhere(mm):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, but let a corner legend choose the carve alignment.

    Input structure:
    - bg = most common color.
    - Several solid 5×5 monochrome blocks, each a distinct non-bg color.
    - For each block color, a small same-colored motif sprite elsewhere on the grid.
    - One 2×2 legend sits in a corner on the grid boundary and encodes whether carving is upper-left, centered, or lower-right.

    Rule:
    - Decode the legend first to obtain the shared alignment mode.
    - Match each motif to its same-colored 5×5 block and repaint motif-shaped cells to bg at that decoded offset.
    - Pack all carved blocks into an output grid preserving their original row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight:
    - The small corner symbol is a global instruction: colors still tell you which motif belongs to which block, but the legend tells you where inside each 5×5 block that stencil must be applied.
    """
    return _solve_with_mode(grid)


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
GENERATION_RETRIES = 300

LEGEND_SIZE = 2
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
ALIGNMENT_MODES = tuple(ALIGNMENT_MASKS.keys())


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
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
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible orderings for separated destination and motif regions."""
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str) -> tuple[int, int, int, int]:
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


def _place_horizontal(H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str) -> tuple[int, int, int, int]:
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
                if filled + 1 > COLOR_COUNT - 1:
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


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int], int]:
    """Pick destination colors, distractor colors, and one extra legend color."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count:filled]
    legend_color = int(palette[filled])
    return block_colors, distractor_colors, legend_color


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    alignment: str,
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    off_r, off_c = _alignment_offsets(DST_BLOCK_SIZE, DST_BLOCK_SIZE, MOTIF_SIZE, MOTIF_SIZE, alignment)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            for rr, cc in np.argwhere(motif_masks[color]):
                tile[off_r + int(rr), off_c + int(cc)] = int(bg)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


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


def _corner_origin(H: int, W: int, corner: str) -> tuple[int, int]:
    """Return the top-left coordinate of the legend patch for the chosen corner."""
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, W - LEGEND_SIZE
    if corner == "bottom_left":
        return H - LEGEND_SIZE, 0
    if corner == "bottom_right":
        return H - LEGEND_SIZE, W - LEGEND_SIZE
    raise ValueError(f"unknown legend corner: {corner}")


def _paint_legend(grid: np.ndarray, bg: int, legend_color: int, corner: str, alignment: str) -> None:
    """Paint the 2x2 symbolic legend into one corner of the input."""
    r0, c0 = _corner_origin(*grid.shape, corner)
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(bg), dtype=int)
    patch[ALIGNMENT_MASKS[alignment]] = int(legend_color)
    grid[r0:r0 + LEGEND_SIZE, c0:c0 + LEGEND_SIZE] = patch


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two inclusive rectangles overlap."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend_bbox(H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    """Inclusive bbox occupied by the 2x2 legend."""
    r0, c0 = _corner_origin(H, W, corner)
    return r0, c0, r0 + LEGEND_SIZE - 1, c0 + LEGEND_SIZE - 1


def _placement_avoids_corners(
    H: int,
    W: int,
    placement: tuple[int, int, int, int],
    dst_h: int,
    dst_w: int,
    motif_h: int,
    motif_w: int,
) -> bool:
    """Reject placements where either major region spills into any corner 2x2 patch."""
    dst_r, dst_c, motif_r, motif_c = placement
    dst_box = (dst_r, dst_c, dst_r + dst_h - 1, dst_c + dst_w - 1)
    motif_box = (motif_r, motif_c, motif_r + motif_h - 1, motif_c + motif_w - 1)
    corner_boxes = [_legend_bbox(H, W, corner) for corner in LEGEND_CORNERS]
    return all(
        not _rects_overlap(dst_box, corner_box) and not _rects_overlap(motif_box, corner_box)
        for corner_box in corner_boxes
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
    legend_color: int,
    legend_corner: str,
    alignment: str,
) -> np.ndarray:
    """Render the sampled destination blocks, motif slots, and corner legend into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks)
    _paint_legend(inp, bg, legend_color, legend_corner, alignment)
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

    legend_corner = random.choice(LEGEND_CORNERS)
    alignment = random.choice(ALIGNMENT_MODES)
    placement = None
    for _ in range(30):
        candidate = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
        if candidate is None:
            continue
        if _placement_avoids_corners(H, W, candidate, dst_h, dst_w, motif_h, motif_w):
            placement = candidate
            break
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors, legend_color = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks, alignment)
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
        placement,
        legend_color,
        legend_corner,
        alignment,
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

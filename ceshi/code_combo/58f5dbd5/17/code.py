# Combo variant of 58f5dbd5 (index 17)
# combo_spec: D5/1+D11/1
# Name: 尺寸各异方块+由motif决定方块尺寸
#
# Design analysis
# ---
# a) Concretization and behavior trace
#    - Original task:
#      every destination object is a solid 5x5 square, every matching motif fits in a 3x3 bbox,
#      carving always uses the same centered 3x3 stencil, and the packed output lives on a fixed 5+1 lattice.
#    - Fused variant used here:
#      destination objects are still solid monochrome squares, but their side lengths now vary because the
#      matching motif itself can have a tight square bbox of side 1, 3, or 5. The block side is defined as
#      motif_bbox_side + 2, so visible block sizes are 3, 5, and 7.
#    - Concrete trace:
#      a 1x1 same-colored motif now pairs with a 3x3 block and produces one centered hole;
#      a 3x3 motif pairs with a 5x5 block and reproduces the original-looking centered carving;
#      a 5x5 motif pairs with a 7x7 block and carves a much larger hole pattern.
#      The old solver would have treated all three cases as if the destination were 5x5; the fused solver
#      instead reads the motif bbox size first, then uses that size relation when carving and packing.
#    - Integration choice:
#      D5 contributes "block sizes are not uniform"; D11 constrains where those sizes come from.
#      Rather than combining two independent size rules, the fusion makes variable square size a visible
#      consequence of motif geometry itself.
# b) Complexity calibration
#    - Size freedom is restricted to three observable tiers {3,5,7}, derived from motif bbox sides {1,3,5}.
#      This is enough to make the variant visibly different without creating an open-ended size search.
#    - Every sample is forced to contain at least two destination sizes and at least one 7x7 block, so the
#      task cannot collapse back to the original all-5x5 regime.
#    - Colors remain one-to-one between destination blocks and their true motifs, which keeps matching
#      unambiguous even with distractor motifs.
# c) Hidden-assumption audit of the original solver
#    - Original block detection assumed "solid 5x5 square". That fails here; we replace it with
#      "solid square whose side is one of {3,5,7}".
#    - Original motif skipping assumed the same-colored destination object was specifically 5x5.
#      Now the solver must skip whichever same-colored component is the detected solid square for that color.
#    - Original carving hard-coded a 5x5 tile and original packing hard-coded step=6.
#      Both assumptions break once blocks become 3/5/7, so packing must be based on per-row and per-column
#      maxima derived from the detected block sizes.
#    - D5's single-variant strip-only simplification is intentionally not reused. Instead, the generator keeps
#      the original row/column grid structure and aligns variable-size squares on row/column starts so that
#      compact repacking is still well-defined.
# d) Placement legality
#    - Destination blocks are painted on an RxC layout with 1-cell gaps between row bands / column bands,
#      using row heights and column widths equal to the maximum block size in that row/column. This prevents
#      overlap for mixed 3/5/7 squares while preserving row-major order.
#    - Motifs are painted into a separate virtual slot grid of 5x5 slots stepped by 6, centered inside each
#      slot. Because the largest motif bbox is 5x5, components never touch across slots.
#    - The two regions are separated along at least one axis with outer margins, so same-colored components
#      cannot accidentally merge.
# e) Visual observability
#    - A human can compare motif bbox sizes directly in the source region and see that larger motif bboxes
#      correspond to larger same-colored solid squares and to larger carved outputs.
#    - No ordering depends on hidden numeric color ids; color is only used for visible same-color matching.
#    - The rule is recoverable from 2-4 examples because the generator repeats the size relation multiple
#      times inside each sample rather than hiding it in a one-off exception.
# f) Visual cleanliness
#    - The input still has two clean regions: solid squares in one region, motif sprites in another.
#      There is no overprinting, no legend, and no mixed semantics inside one component.
#    - The output keeps the original 1-cell border and 1-cell inter-tile gaps, so mixed block sizes read as a
#      compact grid rather than a cluttered collage.
# g) Change inferability
#    - The only "mapping" is geometric: motif tight square bbox side -> block side via +2.
#      This is structural and repeatedly visible; there is no arbitrary color remapping to infer.
# h) New solver assumptions introduced here
#    - Matching motifs use tight square bboxes of side 1, 3, or 5, and destination blocks use the paired
#      square sizes 3, 5, or 7. The generator guarantees this.
#    - Within the destination region, all blocks in the same logical row share the same top row start and all
#      blocks in the same logical column share the same left column start. The generator paints them that way
#      so row/column order is recoverable from top-left coordinates alone.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


COLOR_COUNT = 10

MOTIF_BBOX_SIDES = (1, 3, 5)
BLOCK_PADDING = 1
BLOCK_SIDES = tuple(side + 2 * BLOCK_PADDING for side in MOTIF_BBOX_SIDES)

PACK_GAP = 1
OUTPUT_BORDER = 1

CANVAS_HEIGHT_RANGE = (28, 36)
CANVAS_WIDTH_RANGE = (28, 36)
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SLOT_SIZE = 5
MOTIF_STEP = 6
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
GENERATION_RETRIES = 500


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


def _is_solid_square(grid: np.ndarray, comp: dict) -> tuple[bool, int]:
    """Return whether a component is a solid allowed-size square and its side."""
    b = comp["bbox"]
    side_h = int(bbox_h(b))
    side_w = int(bbox_w(b))
    if side_h != side_w or side_h not in BLOCK_SIDES:
        return False, side_h
    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    return bool((patch == col).all()), side_h


def _layout_from_sizes(
    rows: list[int], cols: list[int], sizes: dict[tuple[int, int], int], gap: int
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """Return packed row/column offsets for variable-size square tiles."""
    row_sizes = {int(r): max(int(sizes[(r, c)]) for c in cols if (r, c) in sizes) for r in rows}
    col_sizes = {int(c): max(int(sizes[(r, c)]) for r in rows if (r, c) in sizes) for c in cols}

    row_starts: dict[int, int] = {}
    cursor = 0
    for r in rows:
        row_starts[int(r)] = int(cursor)
        cursor += int(row_sizes[int(r)]) + int(gap)
    total_h = int(cursor - gap if rows else 0)

    col_starts: dict[int, int] = {}
    cursor = 0
    for c in cols:
        col_starts[int(c)] = int(cursor)
        cursor += int(col_sizes[int(c)]) + int(gap)
    total_w = int(cursor - gap if cols else 0)

    return row_starts, col_starts, total_h, total_w


def _tile_side_from_mask(mask: np.ndarray) -> int:
    """Block side rule for the fused variant: motif bbox side plus one cell per side."""
    h, w = map(int, mask.shape)
    assert h == w and h in MOTIF_BBOX_SIDES
    return int(h + 2 * BLOCK_PADDING)


def _matched_tile_side(mask: np.ndarray) -> int | None:
    """Return the fused-rule tile side implied by a motif bbox, or None if unsupported."""
    h, w = map(int, mask.shape)
    if h != w or h not in MOTIF_BBOX_SIDES:
        return None
    return int(h + 2 * BLOCK_PADDING)


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _make_3x3_masks() -> list[np.ndarray]:
    """Enumerate connected non-solid 3x3 masks whose tight bbox is exactly 3x3."""
    out: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in itertools.product([False, True], repeat=9):
        m = np.array(bits, dtype=bool).reshape(3, 3)
        if not m.any() or m.all():
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if not is_contiguous(m.astype(int), background=0, connectivity=8):
            continue
        out[_canonical_mask(m)] = m
    return list(out.values())


def _make_5x5_masks() -> list[np.ndarray]:
    """Return a small curated library of connected 5x5 masks with full 5x5 bbox."""
    raws = [
        [
            "00100",
            "00100",
            "11111",
            "00100",
            "00100",
        ],
        [
            "10001",
            "01010",
            "00100",
            "01010",
            "10001",
        ],
        [
            "00100",
            "01110",
            "11111",
            "01110",
            "00100",
        ],
        [
            "11111",
            "10001",
            "10001",
            "10001",
            "11111",
        ],
        [
            "11111",
            "00001",
            "11111",
            "10000",
            "11111",
        ],
        [
            "11111",
            "01110",
            "00100",
            "01110",
            "11111",
        ],
        [
            "11111",
            "10000",
            "10111",
            "10001",
            "11111",
        ],
        [
            "00111",
            "00100",
            "11111",
            "00100",
            "11100",
        ],
    ]

    out: list[np.ndarray] = []
    for raw in raws:
        mask = np.array([[ch == "1" for ch in row] for row in raw], dtype=bool)
        assert mask.shape == (5, 5)
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        assert mask.any() and not mask.all()
        assert is_contiguous(mask.astype(int), background=0, connectivity=8)
        out.append(mask)
    return out


MASKS_BY_SIDE = {
    1: [np.array([[True]], dtype=bool)],
    3: _make_3x3_masks(),
    5: _make_5x5_masks(),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into variable-size solid squares and repack the results.

    Input structure:
    - The background is the most common color.
    - One region contains several solid monochrome squares of side 3, 5, or 7, each a distinct non-bg color.
    - Another separated region contains small same-colored motif sprites whose tight square bboxes are 1x1, 3x3,
      or 5x5, plus optional distractor motifs in other colors.

    Rule:
    - Match each motif to the solid square with the same color.
    - Center the motif's tight mask inside that square and repaint the motif cells to the background color.
    - Pack the carved squares back into a compact grid that preserves the original row/column order, using
      1-cell gaps and a 1-cell outer border.

    Key insight:
    - The motif is both the carving stencil and the size cue: its tight square bbox determines which of the
      three block sizes appears, because every block is exactly one cell larger on each side than its motif.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 3x3 / 5x5 / 7x7 squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        is_solid, side = _is_solid_square(g, comp)
        if not is_solid:
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0), "side": int(side)})

    assert blocks

    # STEP 2) For each block color, find the smallest same-colored component that is not the solid square.
    motifs: dict[int, np.ndarray] = {}
    block_by_color = {int(b["col"]): b for b in blocks}
    for bcol, block in block_by_color.items():
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
            is_solid, side = _is_solid_square(g, comp)
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if (
                is_solid
                and r0 == int(block["r0"])
                and c0 == int(block["c0"])
                and side == int(block["side"])
            ):
                continue
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            matched_side = _matched_tile_side(mask)
            if matched_side != int(block["side"]):
                continue
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        assert candidates
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each square by centering its matched motif mask inside it.
    carved: dict[tuple[int, int], np.ndarray] = {}
    sizes: dict[tuple[int, int], int] = {}
    for b in blocks:
        col = int(b["col"])
        mm = motifs[int(col)]
        side = _tile_side_from_mask(mm)
        assert side == int(b["side"])
        tile = np.full((side, side), int(col), dtype=int)
        mh, mw = map(int, mm.shape)
        off_r = (side - mh) // 2
        off_c = (side - mw) // 2
        for rr, cc in np.argwhere(mm):
            tile[int(off_r + rr), int(off_c + cc)] = int(bg)
        key = (int(b["r0"]), int(b["c0"]))
        carved[key] = tile
        sizes[key] = int(side)

    # STEP 4) Pack the carved squares using their recovered row/column grid and true sizes.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    row_starts, col_starts, inner_h, inner_w = _layout_from_sizes(rs, cs, sizes, gap=PACK_GAP)

    out = np.full((inner_h + 2 * OUTPUT_BORDER, inner_w + 2 * OUTPUT_BORDER), int(bg), dtype=int)
    for b in blocks:
        key = (int(b["r0"]), int(b["c0"]))
        rr = int(OUTPUT_BORDER + row_starts[int(b["r0"])])
        cc = int(OUTPUT_BORDER + col_starts[int(b["c0"])])
        paste(out, carved[key], rr, cc)
    return out


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed slots placed at a constant step."""
    return int((count - 1) * step + item_size)


def _layout_tile_positions(
    R: int, C: int, tile_sides: list[int]
) -> tuple[list[tuple[int, int]], int, int]:
    """Return row-major tile positions plus total region span for mixed square sizes."""
    row_heights = [max(int(tile_sides[i * C + j]) for j in range(C)) for i in range(R)]
    col_widths = [max(int(tile_sides[i * C + j]) for i in range(R)) for j in range(C)]

    row_offsets: list[int] = []
    cursor = 0
    for h in row_heights:
        row_offsets.append(int(cursor))
        cursor += int(h) + PACK_GAP
    total_h = int(cursor - PACK_GAP)

    col_offsets: list[int] = []
    cursor = 0
    for w in col_widths:
        col_offsets.append(int(cursor))
        cursor += int(w) + PACK_GAP
    total_w = int(cursor - PACK_GAP)

    positions = [(int(row_offsets[i]), int(col_offsets[j])) for i in range(R) for j in range(C)]
    return positions, total_h, total_w


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


def _choose_config() -> tuple[int, int, int, int, int]:
    """Sample compatible destination and motif-grid layouts."""
    count = random.choice(sorted(DST_LAYOUTS))
    R, C = random.choice(DST_LAYOUTS[count])

    motif_choices: list[tuple[int, int, int]] = []
    for capacity, layouts in MOTIF_LAYOUTS.items():
        min_filled = max(FILLED_SLOT_RANGE[0], count, capacity - MAX_EMPTY_SLOTS)
        max_filled = min(FILLED_SLOT_RANGE[1], capacity)
        if min_filled > max_filled:
            continue
        filled = random.randint(min_filled, max_filled)
        Mr, Mc = random.choice(layouts)
        motif_choices.append((Mr, Mc, filled))

    Mr, Mc, filled = random.choice(motif_choices)
    return R, C, Mr, Mc, filled


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _random_mask_for_side(side: int) -> np.ndarray:
    return random.choice(MASKS_BY_SIDE[int(side)]).copy()


def _choose_block_masks(count: int) -> list[np.ndarray]:
    """Sample matching motifs while forcing visible size variation."""
    while True:
        sides = [random.choice(MOTIF_BBOX_SIDES) for _ in range(count)]
        if len(set(sides)) < 2:
            continue
        if 5 not in sides:
            continue
        if all(side == 3 for side in sides):
            continue
        return [_random_mask_for_side(side) for side in sides]


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Render the oracle output and expose the per-block square sides."""
    tile_sides = [_tile_side_from_mask(motif_masks[int(color)]) for color in block_colors]
    positions, inner_h, inner_w = _layout_tile_positions(R, C, tile_sides)
    expected = np.full((inner_h + 2 * OUTPUT_BORDER, inner_w + 2 * OUTPUT_BORDER), bg, dtype=int)

    for idx, color in enumerate(block_colors):
        mask = motif_masks[int(color)]
        side = int(tile_sides[idx])
        tile = np.full((side, side), int(color), dtype=int)
        off = (side - mask.shape[0]) // 2
        tile[off : off + mask.shape[0], off : off + mask.shape[1]][mask] = int(bg)
        pos_r, pos_c = positions[idx]
        paste(expected, tile, OUTPUT_BORDER + pos_r, OUTPUT_BORDER + pos_c)
    return expected, tile_sides


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    tile_sides: list[int],
) -> None:
    """Paint the solid variable-size destination squares into the input grid."""
    positions, _h, _w = _layout_tile_positions(R, C, tile_sides)
    for idx, color in enumerate(block_colors):
        side = int(tile_sides[idx])
        block = np.full((side, side), int(color), dtype=int)
        pos_r, pos_c = positions[idx]
        paste(grid, block, dst_r + pos_r, dst_c + pos_c)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif component into the input grid."""
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[int(color)]) for color in block_colors
    ]
    for color in distractor_colors:
        items.append((int(color), _random_mask_for_side(random.choice(MOTIF_BBOX_SIDES))))
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
        off = (MOTIF_SLOT_SIZE - mask.shape[0]) // 2
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP + off,
            motif_c + sc * MOTIF_STEP + off,
            int(color),
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
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    tile_sides: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination and motif regions into one input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, tile_sides)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    R, C, Mr, Mc, filled = _choose_config()
    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)

    block_masks = _choose_block_masks(count)
    motif_masks = {int(color): block_masks[idx] for idx, color in enumerate(block_colors)}
    expected, tile_sides = _render_expected(bg, R, C, block_colors, motif_masks)

    _positions, dst_h, dst_w = _layout_tile_positions(R, C, tile_sides)
    motif_h = _spanned_size(Mr, MOTIF_SLOT_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SLOT_SIZE, MOTIF_STEP)
    if not _can_split(H, W, dst_h, dst_w, motif_h, motif_w):
        return None

    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, tile_sides, placement
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

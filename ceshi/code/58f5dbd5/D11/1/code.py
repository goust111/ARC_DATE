# Variant of 58f5dbd5
# Dimension: D11 — 方块尺寸 (meta)
# Change: 所有方块固定为5x5... → 方块尺寸由motif决定

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D11 variant
# -----------------------------------
# a) Concretization and behavior trace
#    - The original task had one rigid tile size: every destination block was a solid 5x5 square,
#      every matching motif fit a 3x3 bbox, and carving always happened with a fixed (1,1) offset.
#    - In this variant the motifs still live in 3x3 slots, but their *tight* bbox may now be one of
#      2x2, 2x3, 3x2, or 3x3. The matching solid destination block is resized to exactly
#      (motif_bbox_h + 2) x (motif_bbox_w + 2), so the motif still lands with a 1-cell margin on
#      each side after centering. This yields visible tile sizes 4x4, 4x5, 5x4, and 5x5.
#    - Small trace:
#      motif bbox 2x3 -> old logic: always carve into a 5x5 square at offset (1,1);
#      new logic: build a 4x5 block, center the 2x3 stencil inside it, then pack that 4x5 carved tile.
#      The difference is visible in both the input destination region and the packed output geometry.
# b) Complexity calibration
#    - The size rule only has four observable categories, so it is not too unconstrained.
#    - The generator forces at least two motif-bbox categories per sample and at least one non-3x3
#      motif, so the task does not visually collapse back to the original all-5x5 regime.
# c) Hidden-assumption audit of the original solver
#    - Block detection originally assumed bbox_h == 5 and bbox_w == 5. That breaks immediately here.
#      We replace it with “solid monochrome rectangle with both sides >= 4”, because motifs remain
#      bounded by 3x3 while resized destination blocks are always at least 4x4.
#    - Motif skipping originally used “ignore the same-colored 5x5 component”. That now needs to skip
#      the unique same-colored detected block, whatever its rectangle size is.
#    - Carving originally hard-coded a 5x5 tile and a fixed output step of 6. Both are replaced by
#      per-block sizes and by row/column spans derived from the detected blocks.
# d) Placement legality
#    - Destination blocks are laid out on an RxC grid with 1-cell gaps, but row heights and column
#      widths are the maxima of the tiles in that row/column. This prevents overlap while preserving
#      row/column order for variable-sized tiles.
#    - Motifs remain in separated 3x3 slots stepped by 4, so they never touch each other.
# e) Visual observability
#    - The size rule is entirely visible: the motif's tight bbox and the block's outer size differ by
#      exactly one cell on every side. Output rows/columns also become uneven whenever tile sizes vary.
# f) Visual cleanliness
#    - The input still uses two separated regions: solid destination blocks in one region and motif
#      sprites in another. The output keeps a 1-cell border and 1-cell inter-slot gaps, so nothing
#      overlaps or becomes ambiguous.
# g) Color inference
#    - Colors are still only used for same-color matching. No sorting or transformation depends on the
#      numeric color ids.
# h) New solver assumptions introduced here
#    - Each destination color appears exactly twice in the input: once as one solid rectangle block and
#      once as its motif component. The generator guarantees that.
#    - Destination blocks share aligned row starts inside a row and aligned column starts inside a
#      column. The generator uses the same slot geometry when painting them.


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


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    """Return whether a component fills its bbox completely with its own color."""
    b = comp["bbox"]
    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    return bool((patch == col).all())


def _layout_from_sizes(
    rows: list[int], cols: list[int], sizes: dict[tuple[int, int], tuple[int, int]], gap: int
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """Return start offsets plus total packed size for variable tiles."""
    row_sizes = {int(r): max(int(sizes[(r, c)][0]) for c in cols if (r, c) in sizes) for r in rows}
    col_sizes = {int(c): max(int(sizes[(r, c)][1]) for r in rows if (r, c) in sizes) for c in cols}

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each motif to its same-colored solid block, carve the motif out, and repack the results.

    Input structure:
    - bg is the dominant color.
    - One region contains several solid monochrome rectangles, each a distinct non-bg color.
    - Another separated region contains small same-colored motif sprites, plus optional distractor motifs.
    - A block's size is not fixed: it is chosen so the motif's tight bbox fits inside it with a 1-cell margin.

    Rule:
    - For every destination color, find the same-colored motif component.
    - Center that motif's tight mask inside the solid rectangle of the same color and repaint the motif cells to bg.
    - Pack the carved rectangles back into a compact grid that preserves their original row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight:
    - The motif is a stencil, and its tight bbox also explains the destination tile size: the block is the motif bbox with one extra cell of padding on each side.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid monochrome rectangles larger than any motif bbox.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        if h < 4 or w < 4:
            continue
        if not _is_solid_rect(g, comp):
            continue
        r0, c0, _r1, _c1 = map(int, b)
        blocks.append(
            {
                "col": int(comp["color"]),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(h),
                "w": int(w),
            }
        )

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not the solid destination block.
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
            b = comp["bbox"]
            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if (
                r0 == int(block["r0"])
                and c0 == int(block["c0"])
                and h == int(block["h"])
                and w == int(block["w"])
                and _is_solid_rect(g, comp)
            ):
                continue

            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each solid block by centering the matched motif within that block's own size.
    carved: dict[tuple[int, int], np.ndarray] = {}
    sizes: dict[tuple[int, int], tuple[int, int]] = {}
    for b in blocks:
        col = int(b["col"])
        bh = int(b["h"])
        bw = int(b["w"])
        tile = np.full((bh, bw), int(col), dtype=int)
        mm = motifs.get(int(col))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (bh - mh) // 2
            off_c = (bw - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < bh and 0 <= tc < bw:
                    tile[tr, tc] = int(bg)
        key = (int(b["r0"]), int(b["c0"]))
        carved[key] = tile
        sizes[key] = (bh, bw)

    # STEP 4) Pack the carved tiles with per-row heights and per-column widths.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    row_starts, col_starts, inner_h, inner_w = _layout_from_sizes(rs, cs, sizes, gap=1)

    out = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    for b in blocks:
        key = (int(b["r0"]), int(b["c0"]))
        rr = int(1 + row_starts[int(b["r0"])])
        cc = int(1 + col_starts[int(b["c0"])])
        paste(out, carved[key], rr, cc)

    return out


# --------------- generator (o5x / D11) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination layouts still use 3-6 blocks, but block sizes are now per-motif.
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif slots remain 3x3 slots separated by one cell.
MOTIF_SLOT_SIZE = 3
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
PACK_GAP = 1
BLOCK_PADDING = 1

# Generation retry budget.
GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _make_masks_by_bbox() -> dict[tuple[int, int], list[np.ndarray]]:
    """Precompute valid cropped motif masks grouped by their tight bbox size."""
    masks: dict[tuple[int, int], dict[tuple[tuple[int, ...], ...], np.ndarray]] = {
        (2, 2): {},
        (2, 3): {},
        (3, 2): {},
        (3, 3): {},
    }
    min_area, max_area = MOTIF_AREA_RANGE
    for bits in itertools.product([False, True], repeat=MOTIF_SLOT_SIZE * MOTIF_SLOT_SIZE):
        raw = np.array(bits, dtype=bool).reshape(MOTIF_SLOT_SIZE, MOTIF_SLOT_SIZE)
        area = int(raw.sum())
        if not (min_area <= area <= max_area):
            continue
        if not is_contiguous(raw.astype(int), background=0, connectivity=8):
            continue
        coords = np.argwhere(raw)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = raw[r0 : r1 + 1, c0 : c1 + 1].copy()
        key = tuple(map(int, cropped.shape))
        if key not in masks:
            continue
        masks[key][_canonical_mask(cropped)] = cropped
    return {key: list(entries.values()) for key, entries in masks.items()}


MASKS_BY_BBOX = _make_masks_by_bbox()
BBOX_CHOICES = sorted(MASKS_BY_BBOX)


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
    """Sample destination and motif grid layouts."""
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


def _random_mask_for_bbox(key: tuple[int, int]) -> np.ndarray:
    return random.choice(MASKS_BY_BBOX[key]).copy()


def _choose_block_masks(count: int) -> list[np.ndarray]:
    """Sample block motifs while forcing visible size variation."""
    while True:
        bbox_keys = [random.choice(BBOX_CHOICES) for _ in range(count)]
        if len(set(bbox_keys)) < 2:
            continue
        if not any(key != (3, 3) for key in bbox_keys):
            continue
        return [_random_mask_for_bbox(key) for key in bbox_keys]


def _block_size(mask: np.ndarray) -> tuple[int, int]:
    """Tile size rule for the variant: motif bbox plus one cell of padding per side."""
    h, w = map(int, mask.shape)
    return int(h + 2 * BLOCK_PADDING), int(w + 2 * BLOCK_PADDING)


def _layout_tile_positions(
    R: int, C: int, tile_sizes: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    """Return row-major tile positions plus region span using row/column maxima."""
    row_heights = [
        max(int(tile_sizes[i * C + j][0]) for j in range(C))
        for i in range(R)
    ]
    col_widths = [
        max(int(tile_sizes[i * C + j][1]) for i in range(R))
        for j in range(C)
    ]

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

    positions = [
        (int(row_offsets[i]), int(col_offsets[j]))
        for i in range(R)
        for j in range(C)
    ]
    return positions, total_h, total_w


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one cropped motif into an input grid."""
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render the oracle output and also expose the underlying per-block sizes."""
    tile_sizes = [_block_size(motif_masks[int(color)]) for color in block_colors]
    positions, inner_h, inner_w = _layout_tile_positions(R, C, tile_sizes)
    expected = np.full((inner_h + 2 * OUTPUT_BORDER, inner_w + 2 * OUTPUT_BORDER), bg, dtype=int)

    for idx, color in enumerate(block_colors):
        mask = motif_masks[int(color)]
        tile_h, tile_w = tile_sizes[idx]
        tile = np.full((tile_h, tile_w), int(color), dtype=int)
        off_r = (tile_h - mask.shape[0]) // 2
        off_c = (tile_w - mask.shape[1]) // 2
        for rr, cc in np.argwhere(mask):
            tile[int(off_r + rr), int(off_c + cc)] = int(bg)
        pos_r, pos_c = positions[idx]
        paste(expected, tile, OUTPUT_BORDER + pos_r, OUTPUT_BORDER + pos_c)
    return expected, tile_sizes


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    tile_sizes: list[tuple[int, int]],
) -> None:
    """Paint the solid destination blocks into the input grid."""
    positions, _h, _w = _layout_tile_positions(R, C, tile_sizes)
    for idx, color in enumerate(block_colors):
        tile_h, tile_w = tile_sizes[idx]
        block = np.full((tile_h, tile_w), int(color), dtype=int)
        pos_r, pos_c = positions[idx]
        paste(grid, block, dst_r + pos_r, dst_c + pos_c)


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[int(color)]) for color in block_colors
    ]
    for color in distractor_colors:
        items.append((int(color), _random_mask_for_bbox(random.choice(BBOX_CHOICES))))
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
    tile_sizes: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, tile_sizes)
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
    expected, tile_sizes = _render_expected(bg, R, C, block_colors, motif_masks)

    _positions, dst_h, dst_w = _layout_tile_positions(R, C, tile_sizes)
    motif_h = _spanned_size(Mr, MOTIF_SLOT_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SLOT_SIZE, MOTIF_STEP)
    if not _can_split(H, W, dst_h, dst_w, motif_h, motif_w):
        return None
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, tile_sizes, placement
    )
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D11 input")

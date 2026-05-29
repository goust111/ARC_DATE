# Variant of 58f5dbd5
# Dimension: D2 — 匹配条件：干扰项过滤 (match_cond)
# Change: 输入中包含干扰motif（颜色与任何方块不匹配），solver通过颜色匹配自然过滤... → 连接器颜色匹配

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D2 connector-match variant
# a) Concrete behavior and trace:
#    - Each 5x5 destination block keeps its original solid fill color, but now has a
#      single marker pixel centered one cell above it.
#    - Each 3x3 motif body is monochrome in a shared neutral color, and also has a
#      single marker pixel centered one cell above it.
#    - Matching is done by marker color, not by the motif body's color.
#    - Small trace: in the original task, a red motif body carved the red block.
#      Here all motif bodies can be color 8, so a color-only solver has no usable cue.
#      Instead, if a motif has a blue marker and the green 5x5 block also has a blue
#      marker above it, that motif carves the green block.
# b) Complexity calibration:
#    - Blocks stay at 3 or 4 so we can afford distinct fill colors, a shared motif
#      body color, and several distractor marker colors within the 10-color palette.
#    - Matching marker colors repeat exactly twice in the input: once above one block
#      and once above one true motif. Distractor marker colors appear only above motifs.
# c) Original-code assumptions that break:
#    - The old STEP 2 searched same-colored components per block fill color. That fails
#      because motif bodies no longer share block colors.
#    - The old generator could rely on motif color uniqueness. That is invalid here
#      because motif bodies deliberately collapse to one neutral color.
#    - The carving and packing logic remain valid because the stencil mask geometry
#      inside each 3x3 motif body is unchanged.
# d) Placement legality:
#    - Connectors use the built-in 1-cell gap row above each object slot, so no extra
#      overlap risk is introduced inside a region.
#    - Destination and motif regions are still separated by axis-aligned splitting,
#      but their region heights now include that extra connector row.
# e) Visual observability:
#    - A human sees neutral motif bodies plus colored singleton pixels above both
#      motifs and blocks. Because block fill colors are deranged against their own
#      marker colors, the new rule is visible and cannot be mistaken for body-color
#      matching.
# f) Visual cleanliness:
#    - All connectors sit in consistent centered-above positions, so the input keeps
#      the same tidy two-region layout as the original task.
# g) Color-rule inferability:
#    - Matching colors live only on connectors. Motif bodies share one color, so the
#      only stable correspondence signal across examples is connector color equality.
# h) Variant-specific generator guard:
#    - Each block's connector color is a cyclic permutation of block fill colors, so
#      no block matches its own fill color marker. This prevents accidental collapse
#      back to the original same-color rule.
# i) New solver assumptions checked against generation:
#    - Every true motif spans a full 3x3 bbox and has a marker at (top-center - 1 row).
#    - Every block has a marker one cell above its center column.
#    - Every block-marker color appears on exactly one true motif marker.
#    - Distractor motif markers use colors absent from the set of block markers.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve connector-matched motifs into solid blocks and pack the carved blocks tightly.

    Input structure:
    - `bg` is the majority color.
    - Several solid 5×5 monochrome blocks appear in one region, each with a 1-pixel marker directly above its center.
    - Elsewhere, 3×3 motif sprites appear in a separate region; their bodies share a neutral color, and each motif also has a 1-pixel marker directly above its center. Extra motif+marker pairs may be distractors.

    Rule:
    - For each 5×5 block, read the color of its marker pixel and find the motif whose marker has the same color.
    - Center that motif mask inside the block and turn those covered cells into background, creating holes.
    - Pack all carved 5×5 blocks into a compact output that preserves the original row/column order, with 1-cell gaps and a 1-cell border.

    Key insight:
    - The motif body color is a decoy; the true correspondence is carried only by the small connector pixels above motifs and blocks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares with a marker above.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != DST_BLOCK_SIZE or bbox_w(b) != DST_BLOCK_SIZE:
            continue
        fill = int(comp["color"])
        patch = crop_bbox(g, b)
        if not bool((patch == int(fill)).all()):
            continue
        r0, c0, _r1, _c1 = map(int, b)
        conn_r = int(r0 - 1)
        conn_c = int(c0 + BLOCK_CONNECTOR_COL)
        if conn_r < 0 or int(g[conn_r, conn_c]) == int(bg):
            continue
        blocks.append(
            {
                "fill": int(fill),
                "r0": int(r0),
                "c0": int(c0),
                "conn": int(g[conn_r, conn_c]),
            }
        )

    assert blocks

    # STEP 2) Identify 3x3 motif bodies and index them by their marker color.
    motifs: dict[int, np.ndarray] = {}
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != MOTIF_SIZE or bbox_w(b) != MOTIF_SIZE:
            continue
        area = int(comp["area"])
        if not (MOTIF_AREA_RANGE[0] <= area <= MOTIF_AREA_RANGE[1]):
            continue
        r0, c0, r1, c1 = map(int, b)
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = mask_from_coords(rel_coords, (int(r1 - r0 + 1), int(c1 - c0 + 1)))
        conn_r = int(r0 - 1)
        conn_c = int(c0 + MOTIF_CONNECTOR_COL)
        if conn_r < 0 or int(g[conn_r, conn_c]) == int(bg):
            continue
        motifs[int(g[conn_r, conn_c])] = mask

    block_keys = {int(b["conn"]) for b in blocks}
    assert block_keys.issubset(set(motifs))

    # STEP 3) Carve each 5x5 tile using the motif selected by connector color.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(b["fill"]), dtype=int)
        mm = motifs[int(b["conn"])]
        mh, mw = map(int, mm.shape)
        off_r = (DST_BLOCK_SIZE - mh) // 2
        off_c = (DST_BLOCK_SIZE - mw) // 2
        for rr, cc in np.argwhere(mm):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(OUTPUT_BORDER + len(rs) * DST_BLOCK_SIZE + (len(rs) - 1) * 1 + OUTPUT_BORDER)
    out_w = int(OUTPUT_BORDER + len(cs) * DST_BLOCK_SIZE + (len(cs) - 1) * 1 + OUTPUT_BORDER)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for b in blocks:
        i = int(r_index[int(b["r0"])])
        j = int(c_index[int(b["c0"])])
        rr = int(OUTPUT_BORDER + i * DST_BLOCK_STEP)
        cc = int(OUTPUT_BORDER + j * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[(int(b["r0"]), int(b["c0"]))]

    return out


# --------------- generator (o5x, connector-match variant) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
BLOCK_CONNECTOR_COL = 2
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_CONNECTOR_COL = 1
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
            dh = 1 + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
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
                        mh = 1 + _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
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


def _choose_palette(
    bg: int, count: int, filled: int
) -> tuple[list[int], list[int], int, list[int]]:
    """Pick block fills, a deranged connector-color cycle, one motif-body color, and distractor connector colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    motif_body_color = int(palette[count])
    connector_colors = block_colors[1:] + block_colors[:1]
    remaining = [c for c in palette[count + 1 :] if c not in block_colors]
    distractor_connector_colors = remaining[: filled - count]
    return block_colors, connector_colors, motif_body_color, distractor_connector_colors


def _paint_motif(
    grid: np.ndarray, r0: int, c0: int, body_color: int, connector_color: int, mask: np.ndarray
) -> None:
    """Paint one motif body and its connector pixel into the input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(body_color), dtype=int)
    paste(grid, patch, r0 + 1, c0, mask=mask)
    grid[int(r0), int(c0 + MOTIF_CONNECTOR_COL)] = int(connector_color)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            fill = int(block_colors[idx])
            conn = int(connector_colors[idx])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), fill, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[conn]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    connector_colors: list[int],
) -> None:
    """Paint the solid 5x5 destination blocks plus their top markers into the input grid."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            fill = int(block_colors[idx])
            conn = int(connector_colors[idx])
            body_r = int(dst_r + 1 + i * DST_BLOCK_STEP)
            body_c = int(dst_c + j * DST_BLOCK_STEP)
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), fill, dtype=int)
            paste(grid, block, body_r, body_c)
            grid[int(body_r - 1), int(body_c + BLOCK_CONNECTOR_COL)] = int(conn)


def _slot_items(
    connector_colors: list[int],
    distractor_connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(conn), motif_masks[int(conn)]) for conn in connector_colors
    ]
    items.extend((int(conn), _random_mask()) for conn in distractor_connector_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    body_color: int,
    connector_colors: list[int],
    distractor_connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(connector_colors, distractor_connector_colors, motif_masks)

    for (sr, sc), (conn, mask) in zip(slots, slot_items):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            body_color,
            conn,
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
    connector_colors: list[int],
    body_color: int,
    distractor_connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, connector_colors)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        body_color,
        connector_colors,
        distractor_connector_colors,
        motif_masks,
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

    dst_h = 1 + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = 1 + _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, connector_colors, body_color, distractor_connector_colors = _choose_palette(bg, count, filled)
    motif_masks = {int(conn): _random_mask() for conn in connector_colors}

    expected = _render_expected(bg, R, C, block_colors, connector_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        connector_colors,
        body_color,
        distractor_connector_colors,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D2 connector-match input")

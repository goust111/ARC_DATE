# Combo variant of 58f5dbd5 (index 16)
# combo_spec: D2/2+D16/2
# Name: 连接器颜色匹配+标记像素指示排列顺序

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the fused D2/2 + D16/2 variant
# ---------------------------------------------------
# a) Concrete behavior and trace:
#    - Original task: a red 3x3 motif matches the red 5x5 block by body color, carves
#      holes, and the carved blocks are packed in source row/column order.
#    - Fused task: motif bodies are all the same neutral color, so body color no longer
#      carries correspondence. Instead, each motif has a single colored connector pixel
#      on its left, and each block has a matching colored connector pixel on its left edge.
#    - In parallel, each block also has a 2x3 dot window above it. The number of dots
#      gives that block's output rank. Example: a green block with a blue left connector
#      and four dots above it must be carved by the motif whose left connector is blue,
#      then placed into output slot 4 rather than by source position.
#    - The key difference is therefore twofold: connector color chooses "which stencil",
#      and dot count chooses "where the carved tile goes".
# b) Complexity calibration:
#    - We keep only 3, 4, or 6 blocks so the rank code stays within a visible 2x3 window
#      and the 10-color palette still supports distinct block fills, connector colors,
#      a shared neutral motif/rank color, and a few distractor connector colors.
#    - Connector colors are a cyclic permutation of block fill colors, so they are unique
#      and observable but never collapse back to "body color equals block color".
#    - Dot counts are a permutation of 1..N and forced non-identity, so the reordering
#      is visible in the output rather than silently matching the original packing rule.
# c) Hidden assumptions review:
#    - Original STEP 2 assumed motifs can be found by block fill color. That breaks here
#      because all motif bodies share one neutral color, so the solver must detect motif
#      bodies geometrically and index them by their side connector colors instead.
#    - Original STEP 4 assumed sorting blocks by (r0, c0) yields the output order. That
#      is explicitly invalid now; the solver must read the local dot-count window above
#      each block and map rank -> packed row-major slot.
#    - Adding both signals at once creates a possible conflict only if connector pixels
#      and rank dots overlap. The layout prevents that: connectors live on the left side
#      of blocks/motifs, while rank dots live in a dedicated window above blocks.
# d) Placement legality:
#    - Each source block slot is 6 cells wide (1 connector column + 5 block columns) and
#      8 cells tall (2 marker rows + 1 gap + 5 block rows). Steps add one empty row/col
#      between neighboring block slots, so marker windows and connectors never collide.
#    - Each motif slot is 4 cells wide (1 connector column + 3 motif columns) and 3 cells
#      tall, with one-cell slot gaps preserved by 5-column / 4-row stepping.
#    - The generator enumerates only block/motif layout pairs that fit inside a canvas
#      split on one axis, then verifies them with the solver before accepting.
# e) Visual observability:
#    - Humans can see that motif bodies all have the same color, so carving cannot be
#      explained by the original body-color matching rule.
#    - Humans can also see that output order no longer follows source geometry whenever
#      the dot-count permutation differs from identity, which the generator enforces.
#    - Both new cues are local and countable: repeated connector colors across regions
#      establish matching, and repeated dot windows above all blocks establish ranking.
# f) Visual cleanliness:
#    - The input stays split into a block region and a motif region.
#    - Inside the block region, the top header and left connector serve different roles
#      and do not overlap the solid 5x5 bodies, so object boundaries remain legible.
#    - Inside the motif region, every visible motif is still a tidy 3x3 sprite with one
#      side connector, so distractors look like the same object type rather than noise.
# g) Change inferability:
#    - No rule depends on hidden color ids or arithmetic. Colors only express connector
#      equality, and the order code is purely the visible number of dots.
#    - Because connectors are deranged against block fills and ranks are non-identity,
#      the fused behavior differs from both the original task and from either single cue
#      being ignored.
# solver_assumptions_checked_against_generator:
#   - Every true block is a solid 5x5 monochrome component with a non-bg connector pixel
#     immediately left of its center row.
#   - Every true block has a non-empty 2x3 rank window above it using only bg + one
#     neutral color, so counting non-bg pixels recovers a unique rank.
#   - Every true motif is a 3x3 monochrome component with a non-bg connector pixel
#     immediately left of its center row.
#   - Each block connector color appears on exactly one true motif connector.
#   - Distractor motif connector colors are absent from the set of block connector colors.
#   - The source blocks occupy a full R x C rectangle, so the packed output shape is still
#     inferable from the distinct source block rows and columns.


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


def _marker_mask(rank: int) -> np.ndarray:
    """Connected 2x3 prefix mask with exactly `rank` lit cells."""
    m = np.zeros((MARKER_H, MARKER_W), dtype=bool)
    for idx in range(int(rank)):
        m[idx // MARKER_W, idx % MARKER_W] = True
    return m


def _solid_blocks(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Find 5x5 solid blocks that carry the fused side-connector and top-rank signals."""
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != DST_BLOCK_SIZE or bbox_w(b) != DST_BLOCK_SIZE:
            continue
        fill = int(comp["color"])
        patch = crop_bbox(g, b)
        if not bool((patch == fill).all()):
            continue
        r0, c0, _r1, _c1 = map(int, b)
        conn_r = int(r0 + BLOCK_CONNECTOR_ROW)
        conn_c = int(c0 - BLOCK_CONNECTOR_GAP)
        if conn_c < 0 or int(g[conn_r, conn_c]) == int(bg):
            continue
        blocks.append(
            {
                "fill": fill,
                "r0": r0,
                "c0": c0,
                "conn": int(g[conn_r, conn_c]),
            }
        )
    return blocks


def _extract_motifs(g: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    """Recover 3x3 motif bodies and index them by their left connector colors."""
    motifs: dict[int, np.ndarray] = {}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != MOTIF_SIZE or bbox_w(b) != MOTIF_SIZE:
            continue
        area = int(comp["area"])
        if not (MOTIF_AREA_RANGE[0] <= area <= MOTIF_AREA_RANGE[1]):
            continue
        r0, c0, r1, c1 = map(int, b)
        conn_r = int(r0 + MOTIF_CONNECTOR_ROW)
        conn_c = int(c0 - MOTIF_CONNECTOR_GAP)
        if conn_c < 0 or int(g[conn_r, conn_c]) == int(bg):
            continue
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = mask_from_coords(rel_coords, (int(r1 - r0 + 1), int(c1 - c0 + 1)))
        motifs[int(g[conn_r, conn_c])] = mask
    return motifs


def _rank_from_marker_window(g: np.ndarray, bg: int, block_r0: int, block_c0: int) -> int:
    """Count non-bg pixels in the fixed 2x3 window above a block."""
    rr0 = int(block_r0 - MARKER_H - MARKER_GAP)
    rr1 = int(block_r0 - MARKER_GAP)
    cc0 = int(block_c0 + MARKER_INSET_C)
    cc1 = int(cc0 + MARKER_W)
    window = g[rr0:rr1, cc0:cc1]
    vals = sorted({int(v) for v in window.flatten() if int(v) != bg})
    assert len(vals) == 1, "each rank window must contain exactly one non-bg color"
    rank = int((window != bg).sum())
    assert 1 <= rank <= MARKER_H * MARKER_W, "rank out of supported range"
    return rank


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve connector-matched motifs into solid blocks, then reorder the carved blocks by local dot-count ranks.

    Input structure:
    - `bg` is the majority color.
    - Several solid 5x5 monochrome blocks appear in one region; each block has a single colored connector pixel on its left edge and a 2x3 dot window above it.
    - Elsewhere, 3x3 motif sprites appear in a separate region; their bodies all share one neutral color, and each motif has its own colored connector pixel on the left. Some motif+connector pairs may be distractors.

    Rule:
    - For each block, match by connector color: find the motif whose side connector has the same color as the block's side connector.
    - Center that motif mask inside the 5x5 block and recolor those covered cells to `bg`, creating holes.
    - Read the number of dots in the block's 2x3 window as its destination rank, then place carved blocks into the bordered packed output in row-major rank order.

    Key insight:
    - The motif body color is a decoy and the block's source position is also a decoy; connector color chooses the stencil, while dot count chooses the final slot.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks = _solid_blocks(g, bg)
    assert blocks

    motifs = _extract_motifs(g, bg)
    block_connector_colors = {int(b["conn"]) for b in blocks}
    assert block_connector_colors.issubset(set(motifs)), "missing connector-matched motif"

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    R = len(rs)
    C = len(cs)
    assert len(blocks) == R * C, "source blocks must form a full rectangle"

    carved_by_rank: dict[int, np.ndarray] = {}
    seen_ranks: set[int] = set()
    for b in blocks:
        rank = _rank_from_marker_window(g, bg, int(b["r0"]), int(b["c0"]))
        assert rank not in seen_ranks, "rank markers must be unique"
        seen_ranks.add(rank)

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
        carved_by_rank[rank] = tile

    assert seen_ranks == set(range(1, len(blocks) + 1)), "ranks must be exactly 1..N"

    out_h = int(2 * OUTPUT_BORDER + (R - 1) * PACK_STEP + DST_BLOCK_SIZE)
    out_w = int(2 * OUTPUT_BORDER + (C - 1) * PACK_STEP + DST_BLOCK_SIZE)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for rank, tile in carved_by_rank.items():
        pos = int(rank - 1)
        i = int(pos // C)
        j = int(pos % C)
        rr = int(OUTPUT_BORDER + i * PACK_STEP)
        cc = int(OUTPUT_BORDER + j * PACK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = tile

    return out


# --------------- generator (o5x, fused D2 + D16 variant) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (28, 32)
CANVAS_WIDTH_RANGE = (28, 32)
COLOR_COUNT = 10

# Destination block geometry in the packed output.
DST_BLOCK_SIZE = 5
PACK_GAP = 1
PACK_STEP = DST_BLOCK_SIZE + PACK_GAP
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

# Source block geometry: top rank window + left connector + 5x5 body.
MARKER_H = 2
MARKER_W = 3
MARKER_GAP = 1
MARKER_INSET_C = 1
BLOCK_CONNECTOR_ROW = 2
BLOCK_CONNECTOR_GAP = 1
SRC_BLOCK_REGION_H = MARKER_H + MARKER_GAP + DST_BLOCK_SIZE
SRC_BLOCK_REGION_W = BLOCK_CONNECTOR_GAP + DST_BLOCK_SIZE
SRC_BLOCK_ROW_STEP = SRC_BLOCK_REGION_H + 1
SRC_BLOCK_COL_STEP = SRC_BLOCK_REGION_W + 1

# Source layouts.
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_CONNECTOR_ROW = 1
MOTIF_CONNECTOR_GAP = 1
MOTIF_SLOT_H = MOTIF_SIZE
MOTIF_SLOT_W = MOTIF_CONNECTOR_GAP + MOTIF_SIZE
MOTIF_ROW_STEP = MOTIF_SLOT_H + 1
MOTIF_COL_STEP = MOTIF_SLOT_W + 1
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

# Region placement.
OUTER_MARGIN = 1
REGION_GAP = 1

# Generation retry budget.
GENERATION_RETRIES = 500


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


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int, int]] = []
    min_filled, max_filled = FILLED_SLOT_RANGE
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, SRC_BLOCK_REGION_H, SRC_BLOCK_ROW_STEP)
            dw = _spanned_size(C, SRC_BLOCK_REGION_W, SRC_BLOCK_COL_STEP)
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
                        mh = _spanned_size(Mr, MOTIF_SLOT_H, MOTIF_ROW_STEP)
                        mw = _spanned_size(Mc, MOTIF_SLOT_W, MOTIF_COL_STEP)
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


def _choose_unique_true_masks(connector_colors: list[int]) -> dict[int, np.ndarray]:
    """Assign pairwise-distinct motif masks to true connector colors."""
    pool: list[np.ndarray] = []
    seen: set[bytes] = set()
    for area_masks in MASKS_BY_AREA.values():
        for mask in area_masks:
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            pool.append(mask)

    assert len(pool) >= len(connector_colors), "insufficient unique motif masks"
    chosen = random.sample(pool, len(connector_colors))
    return {int(conn): mask.copy() for conn, mask in zip(connector_colors, chosen)}


def _choose_palette(
    bg: int, count: int, filled: int
) -> tuple[list[int], list[int], int, list[int]]:
    """Pick block fills, deranged connector colors, one neutral motif/rank color, and distractor connector colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    neutral_color = int(palette[count])
    connector_colors = block_colors[1:] + block_colors[:1]
    distractor_count = filled - count
    remaining = [c for c in palette[count + 1 :] if c not in block_colors]
    distractor_connector_colors = remaining[:distractor_count]
    assert len(distractor_connector_colors) == distractor_count
    return block_colors, connector_colors, neutral_color, distractor_connector_colors


def _choose_ranks(count: int) -> list[int]:
    """Return a non-identity permutation of 1..count so the reorder signal is active."""
    ranks = list(range(1, count + 1))
    while True:
        random.shuffle(ranks)
        if any(rank != idx + 1 for idx, rank in enumerate(ranks)):
            return ranks


def _paint_motif(
    grid: np.ndarray,
    slot_r: int,
    slot_c: int,
    body_color: int,
    connector_color: int,
    mask: np.ndarray,
) -> None:
    """Paint one neutral motif body with its left connector pixel."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(body_color), dtype=int)
    paste(grid, patch, slot_r, slot_c + MOTIF_CONNECTOR_GAP, mask=mask)
    grid[int(slot_r + MOTIF_CONNECTOR_ROW), int(slot_c)] = int(connector_color)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    ranks: list[int],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = int(2 * OUTPUT_BORDER + (R - 1) * PACK_STEP + DST_BLOCK_SIZE)
    out_w = int(2 * OUTPUT_BORDER + (C - 1) * PACK_STEP + DST_BLOCK_SIZE)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for src_idx, fill in enumerate(block_colors):
        conn = int(connector_colors[src_idx])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(fill), dtype=int)
        tile[motif_slice, motif_slice][motif_masks[conn]] = bg

        pos = int(ranks[src_idx] - 1)
        i = int(pos // C)
        j = int(pos % C)
        paste(
            expected,
            tile,
            OUTPUT_BORDER + i * PACK_STEP,
            OUTPUT_BORDER + j * PACK_STEP,
        )
    return expected


def _source_block_body_origin(src_r: int, src_c: int, i: int, j: int) -> tuple[int, int]:
    """Map a source layout slot to the 5x5 block body top-left."""
    return (
        int(src_r + i * SRC_BLOCK_ROW_STEP + MARKER_H + MARKER_GAP),
        int(src_c + j * SRC_BLOCK_COL_STEP + BLOCK_CONNECTOR_GAP),
    )


def _paint_destination_blocks(
    grid: np.ndarray,
    src_r: int,
    src_c: int,
    R: int,
    C: int,
    block_colors: list[int],
    connector_colors: list[int],
    ranks: list[int],
    neutral_color: int,
) -> None:
    """Paint solid source blocks together with their side connectors and top rank dots."""
    marker_patch = np.full((MARKER_H, MARKER_W), int(neutral_color), dtype=int)
    idx = 0
    for i in range(R):
        for j in range(C):
            fill = int(block_colors[idx])
            conn = int(connector_colors[idx])
            block_r, block_c = _source_block_body_origin(src_r, src_c, i, j)

            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), fill, dtype=int)
            paste(grid, block, block_r, block_c)

            marker_r = int(block_r - MARKER_H - MARKER_GAP)
            marker_c = int(block_c + MARKER_INSET_C)
            paste(grid, marker_patch, marker_r, marker_c, mask=_marker_mask(int(ranks[idx])))

            grid[int(block_r + BLOCK_CONNECTOR_ROW), int(block_c - BLOCK_CONNECTOR_GAP)] = conn
            idx += 1


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
            motif_r + sr * MOTIF_ROW_STEP,
            motif_c + sc * MOTIF_COL_STEP,
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
    neutral_color: int,
    distractor_connector_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    ranks: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled block region and motif region into the input grid."""
    src_r, src_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(
        inp,
        src_r,
        src_c,
        R,
        C,
        block_colors,
        connector_colors,
        ranks,
        neutral_color,
    )
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        neutral_color,
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

    src_h = _spanned_size(R, SRC_BLOCK_REGION_H, SRC_BLOCK_ROW_STEP)
    src_w = _spanned_size(C, SRC_BLOCK_REGION_W, SRC_BLOCK_COL_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SLOT_H, MOTIF_ROW_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SLOT_W, MOTIF_COL_STEP)
    placement = _place_regions(H, W, src_h, src_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, connector_colors, neutral_color, distractor_connector_colors = _choose_palette(
        bg, count, filled
    )
    motif_masks = _choose_unique_true_masks(connector_colors)
    ranks = _choose_ranks(count)

    expected = _render_expected(bg, R, C, block_colors, connector_colors, motif_masks, ranks)
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
        neutral_color,
        distractor_connector_colors,
        motif_masks,
        ranks,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

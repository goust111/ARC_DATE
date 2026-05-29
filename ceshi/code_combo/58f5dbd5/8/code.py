# Combo variant of 58f5dbd5 (index 8)
# combo_spec: D6/0+D9/0
# Name: 不规则散布+紧凑裁剪无边框

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


# Design analysis for the D6/0 + D9/0 combo variant
# ==================================================
# a) Concrete behavior
#    - The stencil logic is unchanged: a small same-colored motif is centered inside its
#      matching solid 5x5 block and those covered cells become background.
#    - The first change is on input placement. The destination blocks still encode a full
#      logical RxC rectangle, but their top-lefts are mildly scattered instead of sitting on
#      one exact 6-step lattice. Same-row blocks have overlapping row intervals [r0, r0+4];
#      same-column blocks have overlapping column intervals [c0, c0+4].
#    - The second change is on output packing. In the original task, carved tiles were placed
#      with a 1-cell outer border and 1-cell gaps. Here they are packed edge-to-edge.
#    - Small trace:
#      original 2x2 input dst tops might be rows [1,1,7,7], cols [1,7,1,7], and output size
#      is 13x13 because each 5x5 tile is separated by border/gap cells.
#      combo input might instead use rows [1,2,8,9], cols [1,8,2,9]. Sorting unique r0/c0
#      would falsely suggest 4 rows and 4 cols, so the solver must group overlapping 5-cell
#      intervals into two row bands and two column bands. The combo output is then exactly
#      10x10: four carved 5x5 tiles packed flush, with no separating background lines.
# b) Complexity calibration
#    - The scatter freedom stays narrow: inter-band gaps are only 6 or 7, and within-band
#      jitter is only 0 or 1. That is enough to visibly break the old fixed lattice without
#      making row/column structure ambiguous for humans.
#    - Tight packing removes visual separators, so keeping all block colors distinct remains
#      important; otherwise adjacent tiles could visually merge. The generator preserves that.
# c) Original assumptions that break
#    - `sorted({r0})` and `sorted({c0})` no longer recover logical rows/cols because one
#      logical band may contain multiple close top-left values.
#    - `step = 6` and the formulas with +1 border / +1 gap are no longer valid for output.
#    - The old solver also assumed that output dimensions indirectly encode the gap/border;
#      now the only output geometry is R*5 by C*5, so row/col order must be inferred first.
# d) Placement legality
#    - Generator guarantee: neighboring logical row bases differ by 6 or 7 and each block's
#      row jitter is at most 1, so blocks from adjacent rows never overlap vertically.
#      The same holds horizontally for columns. This keeps all 5x5 blocks disjoint.
#    - The motif region remains axis-separated from the destination region, so components
#      for blocks and motifs stay easy to extract.
# e) Visual observability
#    - Every accepted sample is forced to be genuinely irregular: at least one logical row
#      contains multiple visible r0 values or one logical column contains multiple visible c0
#      values. That means the D6 behavior is visible in the input, not just latent.
#    - Every output is borderless and gapless, so the D9 behavior is always visible too.
#      Since block colors are distinct, humans can still see the tile boundaries.
# f) Visual cleanliness
#    - Motifs stay in their own clean region rather than being mixed among scattered blocks.
#    - The scattering is mild banded jitter rather than arbitrary noise, so the scene still
#      reads as "several blocks plus a motif palette" rather than clutter.
# g) New solver assumptions and generator guarantees
#    - Solver assumption: logical rows/cols are exactly the connected components of the 1D
#      overlap relation on 5-cell projections. The generator enforces that.
#    - Solver assumption: each row-band/column-band pair contains exactly one block. The
#      generator always paints a full RxC Cartesian occupancy, so packing by band indices is safe.
#    - Solver assumption: adjacent packed tiles remain visually separable without gap cells.
#      The generator enforces distinct block colors, which makes the borderless packing legible.


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


def _group_overlapping_bands(blocks: list[dict[str, int]], key: str) -> dict[int, int]:
    """Assign each block to a logical row-band or column-band using 5-cell interval overlap."""
    indexed = sorted(
        ((int(block[key]), idx) for idx, block in enumerate(blocks)),
        key=lambda t: (t[0], t[1]),
    )
    band_of: dict[int, int] = {}
    current_band = -1
    current_end = -1
    for start, idx in indexed:
        end = int(start + 4)
        if current_band < 0 or start > current_end:
            current_band += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        band_of[int(idx)] = int(current_band)
    return band_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into scattered 5x5 blocks and pack them flush by inferred band order.

    Input structure:
    - The background is the most frequent color.
    - Several solid 5x5 monochrome blocks appear in irregular scattered positions rather than on a strict lattice.
    - Those blocks still encode a full logical RxC arrangement: same-row blocks overlap vertically, and same-column
      blocks overlap horizontally.
    - For each block color, a small same-colored motif sprite appears elsewhere in a separate region.

    Rule:
    - Match each motif to the solid 5x5 block of the same color.
    - Center the motif within that block and repaint the covered cells to background, producing a carved tile.
    - Recover the original logical row/column ordering from the scattered block positions by grouping overlapping
      5-cell projections, then pack the carved tiles directly adjacent to each other with no gaps and no border.

    Key insight:
    - The motif is still only a stencil for removing cells, but now the solver must infer hidden row/column bands
      from geometry and then emit a fully tight-packed mosaic.
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
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": r0, "c0": c0})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 5x5 blocks.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[bcol],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == bcol).all()):
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

    # STEP 3) Carve each 5x5 tile by centering its motif mask and turning those cells into bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        mask = motifs.get(col)
        if mask is not None:
            mh, mw = map(int, mask.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = bg
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Infer logical row/column bands from scattered placement and pack tiles flush.
    row_of = _group_overlapping_bands(blocks, "r0")
    col_of = _group_overlapping_bands(blocks, "c0")
    R = 1 + max(row_of.values())
    C = 1 + max(col_of.values())
    assert R * C == len(blocks)

    slots: dict[tuple[int, int], np.ndarray] = {}
    for idx, b in enumerate(blocks):
        pos = (int(row_of[idx]), int(col_of[idx]))
        assert pos not in slots
        slots[pos] = carved[(int(b["r0"]), int(b["c0"]))]

    for i in range(R):
        for j in range(C):
            assert (i, j) in slots

    out = np.full((int(R * 5), int(C * 5)), int(bg), dtype=int)
    for i in range(R):
        for j in range(C):
            rr = int(i * 5)
            cc = int(j * 5)
            out[rr : rr + 5, cc : cc + 5] = slots[(i, j)]

    return out


# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 23)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}
DST_GAP_RANGE = (6, 7)
DST_JITTER = 1

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

# Region placement and carving geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
MOTIF_OFFSET_IN_BLOCK = 1

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


def _dst_span_bounds(count: int) -> tuple[int, int]:
    """Return min/max possible scattered span for one axis with `count` logical bands."""
    if count <= 0:
        raise ValueError("count must be positive")
    min_gap, max_gap = DST_GAP_RANGE
    min_span = DST_BLOCK_SIZE + (count - 1) * min_gap
    max_span = DST_BLOCK_SIZE + (count - 1) * max_gap + DST_JITTER
    return int(min_span), int(max_span)


def _config_templates() -> list[tuple[int, int, int, int, int]]:
    """Enumerate logical layout configurations before choosing a canvas size."""
    out: list[tuple[int, int, int, int, int]] = []
    min_filled, max_filled = FILLED_SLOT_RANGE
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
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
                        out.append((R, C, Mr, Mc, filled))
    return out


CONFIG_TEMPLATES = _config_templates()


def _feasible_canvas_sizes(R: int, C: int, Mr: int, Mc: int) -> list[tuple[int, int]]:
    """List canvas sizes that can fit the worst-case scattered span for a layout config."""
    _dh_min, dh_max = _dst_span_bounds(R)
    _dw_min, dw_max = _dst_span_bounds(C)
    mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

    feasible: list[tuple[int, int]] = []
    for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1):
        for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1):
            if _can_split(H, W, dh_max, dw_max, mh, mw):
                feasible.append((H, W))
    return feasible


def _choose_config_and_canvas() -> tuple[int, int, int, int, int, int, int] | None:
    """Choose a layout branch first, then sample a canvas that can actually host it."""
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in CONFIG_TEMPLATES})
    random.shuffle(counts)

    chosen_count = counts[0]
    count_candidates: list[tuple[tuple[int, int, int, int, int], list[tuple[int, int]]]] = []
    for R, C, Mr, Mc, filled in CONFIG_TEMPLATES:
        if R * C != chosen_count:
            continue
        sizes = _feasible_canvas_sizes(R, C, Mr, Mc)
        if sizes:
            count_candidates.append(((R, C, Mr, Mc, filled), sizes))

    if not count_candidates:
        all_candidates: list[tuple[tuple[int, int, int, int, int], list[tuple[int, int]]]] = []
        for R, C, Mr, Mc, filled in CONFIG_TEMPLATES:
            sizes = _feasible_canvas_sizes(R, C, Mr, Mc)
            if sizes:
                all_candidates.append(((R, C, Mr, Mc, filled), sizes))
        if not all_candidates:
            return None
        count_candidates = all_candidates

    (R, C, Mr, Mc, filled), sizes = random.choice(count_candidates)
    H, W = random.choice(sizes)
    return int(H), int(W), int(R), int(C), int(Mr), int(Mc), int(filled)


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


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output for the tight-packed carved layout."""
    expected = np.full((int(R * DST_BLOCK_SIZE), int(C * DST_BLOCK_SIZE)), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[color]] = bg
            paste(
                expected,
                tile,
                i * DST_BLOCK_SIZE,
                j * DST_BLOCK_SIZE,
            )
    return expected


def _sample_scattered_positions(R: int, C: int) -> list[tuple[int, int]] | None:
    """Sample a full RxC scattered placement that preserves visible row/column bands."""
    min_gap, max_gap = DST_GAP_RANGE

    for _ in range(80):
        row_bases = [0]
        col_bases = [0]
        for _i in range(1, R):
            row_bases.append(int(row_bases[-1] + random.randint(min_gap, max_gap)))
        for _j in range(1, C):
            col_bases.append(int(col_bases[-1] + random.randint(min_gap, max_gap)))

        positions: list[tuple[int, int]] = []
        for i in range(R):
            for j in range(C):
                r = int(row_bases[i] + random.randint(0, DST_JITTER))
                c = int(col_bases[j] + random.randint(0, DST_JITTER))
                positions.append((r, c))

        min_r = min(r for r, _c in positions)
        min_c = min(c for _r, c in positions)
        positions = [(int(r - min_r), int(c - min_c)) for r, c in positions]

        if _placement_is_irregular(positions, R, C):
            return positions
    return None


def _placement_is_irregular(
    positions: list[tuple[int, int]],
    R: int,
    C: int,
) -> bool:
    """Reject placements that collapse back to the original exact row/column keys."""
    for i in range(R):
        row_vals = {int(positions[i * C + j][0]) for j in range(C)}
        if len(row_vals) > 1:
            return True
    for j in range(C):
        col_vals = {int(positions[i * C + j][1]) for i in range(R)}
        if len(col_vals) > 1:
            return True
    return False


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    positions: list[tuple[int, int]],
    block_colors: list[int],
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for (local_r, local_c), color in zip(positions, block_colors):
        block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        paste(grid, block, dst_r + int(local_r), dst_c + int(local_c))


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
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    positions: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled scattered destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, positions, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    chosen = _choose_config_and_canvas()
    if chosen is None:
        return None
    H, W, R, C, Mr, Mc, filled = chosen
    bg = random.randrange(COLOR_COUNT)

    positions = _sample_scattered_positions(R, C)
    if positions is None:
        return None

    dst_h = int(max(r for r, _c in positions) + DST_BLOCK_SIZE)
    dst_w = int(max(c for _r, c in positions) + DST_BLOCK_SIZE)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_masks,
        positions,
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

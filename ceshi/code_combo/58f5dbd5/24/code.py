# Combo variant of 58f5dbd5 (index 24)
# combo_spec: D6/0+D17/0
# Name: 不规则散布+交错排列

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Core carving semantics stay unchanged: every solid 5x5 monochrome block is
#      paired with one smaller same-colored motif; the motif is centered inside the
#      block and converted to background holes in the output.
#    - D6 changes how block order is read. Blocks still represent a full logical
#      RxC rectangle, but one axis is no longer recoverable from `sorted(unique r0)`
#      or `sorted(unique c0)`. Instead, same-row or same-column blocks can differ by
#      0-1 cells on that axis while their 5-cell projections still overlap.
#    - D17 removes the old "block region vs motif region" split. Blocks and motifs now
#      live in one mixed frame. To keep the frame feasible inside the original
#      18-22 by 18-24 canvas, the combo interleaves along a single axis at a time:
#      horizontal motif bands with irregular block columns, or vertical motif bands
#      with irregular block rows.
#    - Small trace:
#      original 2x3: block top-lefts might be [(1,1),(1,7),(1,13),(7,1),(7,7),(7,13)]
#      and motifs live in a separate area. The old solver can sort unique rows/cols.
#      combo-horizontal 2x3: block rows may stay at [3,8], but columns become
#      row 0 -> [0,7,13], row 1 -> [1,6,14], while motifs occupy top/middle/bottom
#      horizontal bands in the same overall frame. `sorted(unique c0)` now invents
#      too many columns, so the solver must group overlapping [c0,c0+4] intervals.
# b) Complexity calibration
#    - The new freedom is deliberately narrow. Irregularity is only 0/1 jitter on one
#      axis with base gaps in {6,7}, which is enough to visibly break the old lattice
#      without making row/column recovery ambiguous.
#    - Interleaving is also constrained: motifs occupy repeated 3-cell bands and 3x3
#      slots on a 4-step lattice. This makes the mixed layout obvious to humans and
#      prevents chaotic overlap while still destroying the original two-region cue.
#    - Full two-axis weaving plus two-axis block jitter would exceed the original
#      canvas for 3x2 / 2x3 cases. The chosen one-axis mix is the smallest change
#      that visibly preserves both requested behaviors on all layouts.
# c) Implicit assumptions audited against `main()`
#    - Step 1 still only assumes solid 5x5 monochrome destination blocks exist.
#      Generator guarantees that.
#    - Step 2 still assumes the matching motif is another smaller component of the
#      same color. The mixed-frame generator guarantees one such component per block
#      color and keeps it 8-disconnected from the block.
#    - Original step 4 assumed every logical row shared one exact `r0` and every
#      logical column shared one exact `c0`. That is false now on the jittered axis,
#      so the combo solver must infer logical rows/cols from overlap bands instead.
# d) Placement legality
#    - Objects never overlap: blocks live in 5x5 slots; motifs live in separate 3x3
#      bands/slots on the mixed axis. The irregular axis only changes top-left
#      offsets within a 0/1 range, so neighboring block intervals remain disjoint
#      across logical bands while same-band intervals still overlap.
#    - Same-colored block/motif pairs are matched only to slots whose 3x3 bbox is
#      separated from the 5x5 block bbox by at least one blank cell under 8-neighbor
#      connectivity, preventing accidental component merging.
# e) Visual observability
#    - Humans can still use the visible same-color cue: one large 5x5 square and one
#      small monochrome sprite of that color belong together.
#    - The combo behavior is visible in every sample because generator rejects layouts
#      where the old exact-lattice solver would still work, and motifs are embedded
#      into the same frame rather than placed in a detached side region.
# f) Visual cleanliness
#    - Mixed layouts use repeated rectangular bands and evenly stepped motif slots.
#      There is no multi-layer overlap and no hidden metadata. Blocks remain dominant
#      5x5 solids; motifs remain smaller 3x3-connected sprites.
# g) New solver assumptions and generator guarantees
#    - Solver assumption: logical rows and columns are recoverable as connected
#      components of the 1D overlap relation on 5-cell intervals. Generator enforces
#      exactly that on both axes.
#    - Solver assumption: each block color has exactly one smaller same-colored
#      component worth using as stencil. Generator guarantees one matching motif and
#      makes all distractor motif colors distinct from block colors.


# --------------- solver ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
PACK_STEP = 6
IRREGULAR_GAP_RANGE = (6, 7)
IRREGULAR_JITTER = 1

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
FILLED_SLOT_RANGE = (6, 8)

OUTER_MARGIN = 1
INPUT_FRAME_MARGIN = 0
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
GENERATION_RETRIES = 500

DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
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


def _group_overlapping_bands(blocks: list[dict[str, int]], key: str) -> dict[int, int]:
    """Assign each block to a row-band or column-band using 5-cell interval overlap."""
    indexed = sorted(
        ((int(block[key]), idx) for idx, block in enumerate(blocks)),
        key=lambda t: (t[0], t[1]),
    )
    band_of: dict[int, int] = {}
    current_band = -1
    current_end = -1
    for start, idx in indexed:
        end = int(start + DST_BLOCK_SIZE - 1)
        if current_band < 0 or start > current_end:
            current_band += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        band_of[int(idx)] = int(current_band)
    return band_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks and repack them by inferred band order.

    Input structure:
    - The background is the most frequent color.
    - Several solid 5x5 monochrome blocks appear together with smaller monochrome
      motif sprites inside one mixed frame rather than in two separated regions.
    - The block arrangement is not on a perfect lattice: on at least one axis,
      same-row or same-column blocks can differ by 0-1 cells while still forming
      coherent logical bands.
    - For each block color, there is one smaller same-colored motif somewhere in
      the mixed frame, plus optional distractor motifs of other colors.

    Rule:
    - Detect every solid 5x5 block and find the smaller component of the same color.
    - Center that motif mask inside the block and repaint the covered cells to
      background, turning the motif into a carving stencil.
    - Recover the original logical row/column order from overlapping 5-cell block
      projections, then pack the carved 5x5 tiles tightly with 1-cell gaps and a
      1-cell background border.

    Key insight:
    - Two original shortcuts are both gone: motifs are no longer isolated in a
      separate region, and the block lattice is no longer exact. The stable cue is
      still color matching plus 5-cell overlap bands, not absolute coordinates.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != DST_BLOCK_SIZE or bbox_w(b) != DST_BLOCK_SIZE:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == col).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": col, "r0": r0, "c0": c0})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the solid 5x5 blocks.
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
            if bbox_h(b) == DST_BLOCK_SIZE and bbox_w(b) == DST_BLOCK_SIZE:
                patch = crop_bbox(g, b)
                if bool((patch == bcol).all()):
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

    # STEP 3) Carve each 5x5 tile by centering the matched motif and repainting
    # those cells to background.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                    tile[tr, tc] = bg
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Infer logical row/column bands from overlapping 5-cell projections.
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

    out_h = int(1 + R * DST_BLOCK_SIZE + (R - 1) + 1)
    out_w = int(1 + C * DST_BLOCK_SIZE + (C - 1) + 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            rr = int(1 + i * PACK_STEP)
            cc = int(1 + j * PACK_STEP)
            out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = slots[(i, j)]

    return out


# --------------- generator ---------------


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


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick block colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled logical block layout."""
    out_h = 2 * OUTPUT_BORDER + (R - 1) * PACK_STEP + DST_BLOCK_SIZE
    out_w = 2 * OUTPUT_BORDER + (C - 1) * PACK_STEP + DST_BLOCK_SIZE
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[color]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * PACK_STEP,
                OUTPUT_BORDER + j * PACK_STEP,
            )
    return expected


def _interleaved_patterns(block_count: int) -> list[tuple[int, ...]]:
    """Enumerate motif-band placements that visibly mix with the block bands."""
    patterns: list[tuple[int, ...]] = []
    positions = list(range(block_count + 1))
    for bits in itertools.product([0, 1], repeat=block_count + 1):
        chosen = tuple(pos for pos, bit in zip(positions, bits) if bit)
        if not chosen:
            continue
        has_internal = any(0 < pos < block_count for pos in chosen)
        has_both_outers = 0 in chosen and block_count in chosen
        if has_internal or has_both_outers:
            patterns.append(chosen)
    return patterns


def _axis_bands(block_count: int, motif_positions: tuple[int, ...]) -> tuple[list[int], list[int], int]:
    """Return block starts, motif-band starts, and total span along the mixed axis."""
    block_starts: list[int] = []
    motif_starts: list[int] = []
    cur = 0
    motif_set = set(int(pos) for pos in motif_positions)
    for pos in range(block_count + 1):
        if pos in motif_set:
            motif_starts.append(int(cur))
            cur += MOTIF_SIZE
        if pos < block_count:
            block_starts.append(int(cur))
            cur += DST_BLOCK_SIZE
    return block_starts, motif_starts, int(cur)


def _sample_irregular_axis_positions(axis_count: int, group_count: int) -> list[list[int]] | None:
    """
    Sample one jittered axis layout.

    `axis_count` is the number of logical bands on the irregular axis.
    `group_count` is the number of rows/columns across which those band positions vary.
    """
    if axis_count <= 0 or group_count <= 1:
        return None

    min_gap, max_gap = IRREGULAR_GAP_RANGE
    for _ in range(80):
        bases = [0]
        for _idx in range(1, axis_count):
            bases.append(int(bases[-1] + random.randint(min_gap, max_gap)))

        jitters = [
            [random.randint(0, IRREGULAR_JITTER) for _ in range(axis_count)]
            for _ in range(group_count)
        ]
        if not any(
            len({int(jitters[g][k]) for g in range(group_count)}) > 1
            for k in range(axis_count)
        ):
            continue

        positions = [
            [int(bases[k] + jitters[g][k]) for k in range(axis_count)]
            for g in range(group_count)
        ]
        return positions
    return None


def _slot_positions(span: int) -> list[int]:
    """Return all 3x3 slot starts on a 4-step lattice within the given span."""
    return [int(x) for x in range(0, span - MOTIF_SIZE + 1, MOTIF_STEP)]


def _expanded_span_for_slots(base_span: int, band_count: int, min_total_slots: int) -> int | None:
    """
    Expand the orthogonal motif span until the mixed frame can host enough 3x3 slots.

    This is needed for skinny logical block layouts such as 1x3 or 3x1, where a
    one-axis interleave can otherwise expose too few motif slots to cover the
    required 6-8 total motifs.
    """
    if band_count <= 0 or min_total_slots <= 0:
        return None

    span = int(base_span)
    while len(_slot_positions(span)) * band_count < min_total_slots:
        span += MOTIF_STEP
        if span > 64:
            return None
    return int(span)


def _safe_pair(block: dict[str, int], slot: dict[str, int]) -> bool:
    """Return whether a same-colored motif slot is safely 8-disconnected from a block."""
    br0 = int(block["r0"])
    bc0 = int(block["c0"])
    br1 = int(br0 + DST_BLOCK_SIZE - 1)
    bc1 = int(bc0 + DST_BLOCK_SIZE - 1)
    sr0 = int(slot["r0"])
    sc0 = int(slot["c0"])
    sr1 = int(sr0 + MOTIF_SIZE - 1)
    sc1 = int(sc0 + MOTIF_SIZE - 1)
    return sr1 < br0 - 1 or sr0 > br1 + 1 or sc1 < bc0 - 1 or sc0 > bc1 + 1


def _assign_safe_slots(
    blocks: list[dict[str, int]], slots: list[dict[str, int]]
) -> list[int] | None:
    """Assign one safely separated motif slot to each block using backtracking."""
    options: list[list[int]] = []
    for block in blocks:
        cand = [idx for idx, slot in enumerate(slots) if _safe_pair(block, slot)]
        if not cand:
            return None
        random.shuffle(cand)
        options.append(cand)

    order = sorted(range(len(blocks)), key=lambda idx: (len(options[idx]), idx))
    used: set[int] = set()
    chosen: dict[int, int] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        bi = order[k]
        for si in options[bi]:
            if si in used:
                continue
            used.add(si)
            chosen[bi] = si
            if dfs(k + 1):
                return True
            used.remove(si)
            chosen.pop(bi, None)
        return False

    if not dfs(0):
        return None
    return [int(chosen[i]) for i in range(len(blocks))]


def _horizontal_layout(H: int, W: int, R: int, C: int) -> dict[str, object] | None:
    """Build a mixed frame with horizontal motif bands and irregular block columns."""
    col_positions = _sample_irregular_axis_positions(C, R)
    if col_positions is None:
        return None

    base_region_w = int(max(max(row) for row in col_positions) + DST_BLOCK_SIZE)
    if base_region_w + 2 * INPUT_FRAME_MARGIN > W:
        return None

    candidates: list[dict[str, object]] = []
    patterns = _interleaved_patterns(R)
    random.shuffle(patterns)
    for motif_positions in patterns:
        row_starts, motif_rows, region_h = _axis_bands(R, motif_positions)
        if region_h + 2 * INPUT_FRAME_MARGIN > H:
            continue

        region_w = _expanded_span_for_slots(base_region_w, len(motif_rows), FILLED_SLOT_RANGE[0])
        if region_w is None or region_w + 2 * INPUT_FRAME_MARGIN > W:
            continue
        slot_xs = _slot_positions(region_w)
        if not slot_xs:
            continue

        blocks: list[dict[str, int]] = []
        for i in range(R):
            for j in range(C):
                blocks.append(
                    {
                        "row": int(i),
                        "col": int(j),
                        "r0": int(row_starts[i]),
                        "c0": int(col_positions[i][j]),
                    }
                )

        slots = [
            {"r0": int(mr), "c0": int(cx)}
            for mr in motif_rows
            for cx in slot_xs
        ]
        if len(slots) < FILLED_SLOT_RANGE[0]:
            continue

        matching = _assign_safe_slots(blocks, slots)
        if matching is None:
            continue

        candidates.append(
            {
                "axis": "horizontal",
                "region_h": int(region_h),
                "region_w": int(region_w),
                "blocks": blocks,
                "slots": slots,
                "matching": matching,
            }
        )

    if not candidates:
        return None
    return random.choice(candidates)


def _vertical_layout(H: int, W: int, R: int, C: int) -> dict[str, object] | None:
    """Build a mixed frame with vertical motif bands and irregular block rows."""
    row_positions_by_col = _sample_irregular_axis_positions(R, C)
    if row_positions_by_col is None:
        return None

    base_region_h = int(max(max(col) for col in row_positions_by_col) + DST_BLOCK_SIZE)
    if base_region_h + 2 * INPUT_FRAME_MARGIN > H:
        return None

    candidates: list[dict[str, object]] = []
    patterns = _interleaved_patterns(C)
    random.shuffle(patterns)
    for motif_positions in patterns:
        col_starts, motif_cols, region_w = _axis_bands(C, motif_positions)
        if region_w + 2 * INPUT_FRAME_MARGIN > W:
            continue

        region_h = _expanded_span_for_slots(base_region_h, len(motif_cols), FILLED_SLOT_RANGE[0])
        if region_h is None or region_h + 2 * INPUT_FRAME_MARGIN > H:
            continue
        slot_ys = _slot_positions(region_h)
        if not slot_ys:
            continue

        blocks: list[dict[str, int]] = []
        for i in range(R):
            for j in range(C):
                blocks.append(
                    {
                        "row": int(i),
                        "col": int(j),
                        "r0": int(row_positions_by_col[j][i]),
                        "c0": int(col_starts[j]),
                    }
                )

        slots = [
            {"r0": int(ry), "c0": int(mc)}
            for mc in motif_cols
            for ry in slot_ys
        ]
        if len(slots) < FILLED_SLOT_RANGE[0]:
            continue

        matching = _assign_safe_slots(blocks, slots)
        if matching is None:
            continue

        candidates.append(
            {
                "axis": "vertical",
                "region_h": int(region_h),
                "region_w": int(region_w),
                "blocks": blocks,
                "slots": slots,
                "matching": matching,
            }
        )

    if not candidates:
        return None
    return random.choice(candidates)


def _sample_layout(H: int, W: int, R: int, C: int) -> dict[str, object] | None:
    """Sample one feasible mixed layout for the chosen logical block rectangle."""
    builders = []
    if R > 1:
        builders.append(_horizontal_layout)
    if C > 1:
        builders.append(_vertical_layout)
    random.shuffle(builders)
    for build in builders:
        layout = build(H, W, R, C)
        if layout is not None:
            return layout
    return None


def _place_origin(H: int, W: int, region_h: int, region_w: int) -> tuple[int, int]:
    """Choose a legal origin for the mixed frame."""
    origin_r = random.randint(INPUT_FRAME_MARGIN, H - region_h - INPUT_FRAME_MARGIN)
    origin_c = random.randint(INPUT_FRAME_MARGIN, W - region_w - INPUT_FRAME_MARGIN)
    return int(origin_r), int(origin_c)


def _paint_block(grid: np.ndarray, block: dict[str, int], origin: tuple[int, int], color: int) -> None:
    """Paint one solid 5x5 block."""
    r_off, c_off = origin
    patch = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    paste(grid, patch, int(r_off + block["r0"]), int(c_off + block["c0"]))


def _paint_motif_slot(
    grid: np.ndarray,
    slot: dict[str, int],
    origin: tuple[int, int],
    color: int,
    mask: np.ndarray,
) -> None:
    """Paint one 3x3 motif in its assigned slot."""
    r_off, c_off = origin
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, int(r_off + slot["r0"]), int(c_off + slot["c0"]), mask=mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    layout: dict[str, object],
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    filled: int,
) -> np.ndarray:
    """Render the mixed-frame input grid."""
    inp = np.full((H, W), bg, dtype=int)
    origin = _place_origin(H, W, int(layout["region_h"]), int(layout["region_w"]))

    blocks = list(layout["blocks"])
    slots = list(layout["slots"])
    matching = list(layout["matching"])
    used_slot_indices = set(int(idx) for idx in matching)

    for block, color in zip(blocks, block_colors):
        _paint_block(inp, block, origin, int(color))

    for block_idx, color in enumerate(block_colors):
        slot = slots[int(matching[block_idx])]
        _paint_motif_slot(inp, slot, origin, int(color), motif_masks[int(color)])

    free_slot_indices = [idx for idx in range(len(slots)) if idx not in used_slot_indices]
    random.shuffle(free_slot_indices)
    distractor_items = [(int(color), _random_mask()) for color in distractor_colors]
    for slot_idx, (color, mask) in zip(free_slot_indices, distractor_items):
        _paint_motif_slot(inp, slots[int(slot_idx)], origin, int(color), mask)

    assert len(block_colors) + len(distractor_colors) == filled
    return inp


def _solver_breaks_on_unique_coords(layout: dict[str, object]) -> bool:
    """Quantify whether the old exact-coordinate packing assumption would fail."""
    blocks = list(layout["blocks"])
    rs = sorted({int(block["r0"]) for block in blocks})
    cs = sorted({int(block["c0"]) for block in blocks})
    logical_rows = 1 + max(int(block["row"]) for block in blocks)
    logical_cols = 1 + max(int(block["col"]) for block in blocks)
    return len(rs) != logical_rows or len(cs) != logical_cols


def _axis_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two bboxes are disjoint on rows or on columns."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_disjoint = ar1 < br0 or br1 < ar0
    col_disjoint = ac1 < bc0 or bc1 < ac0
    return bool(row_disjoint or col_disjoint)


def combo_behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Measure whether a generated sample actually expresses the combo behavior.

    Returns:
        irregular_breaks_unique: 1 iff old unique-coordinate packing would fail.
        interleaved_not_separated: 1 iff motif/block union bboxes are not axis-separated.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    blocks: list[dict[str, int]] = []
    motifs: list[tuple[int, int, int, int]] = []
    for comp in comps:
        b = comp["bbox"]
        color = int(comp["color"])
        if bbox_h(b) == DST_BLOCK_SIZE and bbox_w(b) == DST_BLOCK_SIZE:
            patch = crop_bbox(g, b)
            if bool((patch == color).all()):
                r0, c0, _r1, _c1 = map(int, b)
                blocks.append({"r0": int(r0), "c0": int(c0)})
                continue
        motifs.append(tuple(map(int, b)))

    row_of = _group_overlapping_bands(blocks, "r0")
    col_of = _group_overlapping_bands(blocks, "c0")
    annotated = [
        {
            "r0": int(block["r0"]),
            "c0": int(block["c0"]),
            "row": int(row_of[idx]),
            "col": int(col_of[idx]),
        }
        for idx, block in enumerate(blocks)
    ]
    irregular = int(_solver_breaks_on_unique_coords({"blocks": annotated}))

    block_bbox = (
        min(int(block["r0"]) for block in blocks),
        min(int(block["c0"]) for block in blocks),
        max(int(block["r0"]) + DST_BLOCK_SIZE - 1 for block in blocks),
        max(int(block["c0"]) + DST_BLOCK_SIZE - 1 for block in blocks),
    )
    motif_bbox = (
        min(int(b[0]) for b in motifs),
        min(int(b[1]) for b in motifs),
        max(int(b[2]) for b in motifs),
        max(int(b[3]) for b in motifs),
    )
    interleaved = int(not _axis_separated(block_bbox, motif_bbox))
    return {
        "irregular_breaks_unique": int(irregular),
        "interleaved_not_separated": int(interleaved),
    }


def _gen_attempt() -> np.ndarray | None:
    """
    Sample one input grid and reject it unless the solver and combo metrics agree.

    Shape is chosen first so the five logical layout branches have comparable
    exposure; otherwise the easier 2x2 frame dominates because it fits more
    random canvas sizes than 1x3 or 3x1.
    """
    shape_options = [
        (int(count), int(R), int(C))
        for count, layouts in DST_LAYOUTS.items()
        for R, C in layouts
    ]
    count, R, C = random.choice(shape_options)

    for _ in range(40):
        H = random.randint(*CANVAS_HEIGHT_RANGE)
        W = random.randint(*CANVAS_WIDTH_RANGE)
        bg = random.randrange(COLOR_COUNT)

        layout = _sample_layout(H, W, R, C)
        if layout is None:
            continue

        min_filled = max(int(count), FILLED_SLOT_RANGE[0])
        max_filled = min(FILLED_SLOT_RANGE[1], len(list(layout["slots"])))
        if min_filled > max_filled:
            continue

        filled = random.randint(min_filled, max_filled)
        if filled - count > COLOR_COUNT - 1 - count:
            continue

        block_colors, distractor_colors = _choose_palette(bg, count, filled)
        motif_masks = {int(color): _random_mask() for color in block_colors}
        expected = _render_expected(bg, R, C, block_colors, motif_masks)
        inp = _render_input(
            H,
            W,
            bg,
            layout,
            block_colors,
            distractor_colors,
            motif_masks,
            filled,
        )

        metrics = combo_behavior_metrics(inp)
        if metrics["irregular_breaks_unique"] != 1:
            continue
        if metrics["interleaved_not_separated"] != 1:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

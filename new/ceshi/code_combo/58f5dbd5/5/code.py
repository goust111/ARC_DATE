# Combo variant of 58f5dbd5 (index 5)
# combo_spec: D2/1+D17/0
# Name: 空间对齐匹配+交错排列

from __future__ import annotations

import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)


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


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization / behavior trace
# - Original 58f5dbd5 had two clean regions: one region of solid 5x5 blocks and one
#   region of small same-colored motifs. The solver could match by color and ignore
#   distractors automatically.
# - This combo removes both conveniences at once:
#   * D2/1: motif colors no longer encode correspondence. Matching is recovered only
#     from shared row centers or shared column centers.
#   * D17/0: motifs are no longer parked in a separate region. They are threaded into
#     the same overall frame as the blocks.
# - Concrete example:
#   * Suppose the block layout is 2x3 in column mode. The woven frame inserts three
#     motif rows among/around the two block rows, while the block columns stay on the
#     original 5+1+5+1+5 spacing. Each block column therefore has exactly two motifs
#     aligned on its center column, one above/between and one below/between depending
#     on the chosen motif rows. Those two motifs pair top-to-bottom with the two blocks
#     in that column. Extra distractors appear only in motif-row x motif-col cells, so
#     they are visibly interleaved but share neither block-row centers nor block-column
#     centers.
# - Relative to the original task, the output rule is unchanged: once the matched masks
#   are recovered, each motif is centered inside its block and carved out as background.
#
# b) Complexity calibration
# - Pure alignment becomes ambiguous unless every aligned band contains the same number
#   of usable motifs as blocks. The generator therefore designates exactly one carrier
#   motif slot per block along the active alignment axis and leaves all extra same-axis
#   cells empty.
# - Distractors are restricted to motif x motif cells. This preserves the D17-style mixed
#   layout without corrupting the counting logic that distinguishes row mode from column
#   mode.
# - The carrier-side band count is kept small: at most one extra outer motif band beyond
#   the minimum needed for matching. This keeps the visual rule legible from a few ARC
#   examples instead of exploding the number of unused cells.
#
# c) Implicit-assumption audit against the old main()
# - Old STEP 2 assumed "same color => same motif". That is invalid here because motif
#   colors are sampled from a palette disjoint from block colors and may repeat.
# - Old per-color component search also assumed each block color owns exactly one small
#   component. Repeated motif colors would collapse unrelated motifs together, so the
#   new solver must detect all motifs first and pair them geometrically.
# - Old STEP 4 still holds: once block top-lefts are recovered, the packed output depends
#   only on row/column ordering, not on how the motif was matched.
#
# d) Placement legality
# - The generator uses explicit row/column band sequences with band sizes:
#   block=5, motif=3, gap=1. Every object is painted inside one precomputed cell, so
#   overlaps and out-of-bounds placements are structurally prevented.
# - Matching motifs live only in block x motif or motif x block cells; distractors live
#   only in motif x motif cells. This guarantees that no distractor accidentally satisfies
#   the solver's alignment predicate.
#
# e) Visual observability
# - Because motif colors never overlap the block palette, color matching is visibly ruled
#   out. A human must look for the repeated row/column-center alignment instead.
# - Because motifs are inserted into the same frame as the blocks rather than exiled to a
#   second region, the D17 change is also directly visible from the pixels.
#
# f) Visual cleanliness
# - The mixed frame remains rectilinear. Large 5x5 solids and small 3x3 sprites occupy
#   separate cells and never overlap.
# - On the non-matching axis, block bands keep simple 1-cell background gaps. This avoids
#   the fully woven layout from becoming too large or too noisy while still making the
#   interleaving cue obvious on the matching axis.
#
# g) Color inference
# - Colors are intentionally non-informative for matching. Block colors are all distinct.
#   Motif colors come from a disjoint palette and may repeat across different motifs, so
#   there is no hidden color-id ordering to exploit.
#
# h) Fusion metric used after implementation
# - We record whether the union bbox of all motifs is axis-separated from the union bbox of
#   all blocks and whether the motif palette overlaps the block palette. A non-degenerate
#   combo sample should have `axis_separated == False` and `motif_palette_overlap == 0`.
#
# i) New solver assumptions audited
# - `_pair_by_alignment()` assumes exactly one mode is valid:
#   row mode => every block row has as many motifs on its center row as blocks in that row;
#   column mode => every block column has as many motifs on its center column as blocks in
#   that column. The generator enforces this by keeping distractors out of any block-aligned
#   row or column and by leaving all non-carrier same-axis cells empty.


def _solid_5x5_block(comp: dict, grid: np.ndarray) -> bool:
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    return bool((crop_bbox(grid, b) == int(col)).all())


def _extract_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return mask_from_coords(rel_coords, (int(r1 - r0 + 1), int(c1 - c0 + 1)))


def _pair_by_alignment(
    blocks: list[dict[str, int]], motifs: list[dict[str, object]]
) -> dict[tuple[int, int], np.ndarray]:
    """Recover one motif mask per block using shared row-center or column-center alignment."""
    block_rows: dict[int, list[dict[str, int]]] = defaultdict(list)
    block_cols: dict[int, list[dict[str, int]]] = defaultdict(list)
    motif_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    motif_cols: dict[int, list[dict[str, object]]] = defaultdict(list)

    for b in blocks:
        block_rows[int(b["center_r"])].append(b)
        block_cols[int(b["center_c"])].append(b)
    for m in motifs:
        motif_rows[int(m["center_r"])].append(m)
        motif_cols[int(m["center_c"])].append(m)

    row_ok = all(
        len(motif_rows.get(int(rc), [])) == len(row_blocks)
        for rc, row_blocks in block_rows.items()
    )
    col_ok = all(
        len(motif_cols.get(int(cc), [])) == len(col_blocks)
        for cc, col_blocks in block_cols.items()
    )

    if row_ok and not col_ok:
        pairing: dict[tuple[int, int], np.ndarray] = {}
        for rc, row_blocks in block_rows.items():
            bs = sorted(row_blocks, key=lambda b: int(b["c0"]))
            ms = sorted(motif_rows[int(rc)], key=lambda m: int(m["c0"]))
            for b, m in zip(bs, ms):
                pairing[(int(b["r0"]), int(b["c0"]))] = np.array(m["mask"], dtype=bool)
        return pairing

    if col_ok and not row_ok:
        pairing = {}
        for cc, col_blocks in block_cols.items():
            bs = sorted(col_blocks, key=lambda b: int(b["r0"]))
            ms = sorted(motif_cols[int(cc)], key=lambda m: int(m["r0"]))
            for b, m in zip(bs, ms):
                pairing[(int(b["r0"]), int(b["c0"]))] = np.array(m["mask"], dtype=bool)
        return pairing

    raise AssertionError("could not infer a unique row/column alignment mode")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve spatially aligned motifs out of solid blocks and repack the carved blocks.

    Input structure:
    - bg is the dominant color.
    - Several solid 5x5 monochrome blocks form a rectangular block lattice.
    - Smaller 3x3 monochrome motifs are woven into the same overall frame as the blocks.
      Their colors are decoys: a motif matches a block only by sharing that block row's
      center line or that block column's center line.

    Rule:
    - Detect all 5x5 solid blocks and all small motif components.
    - Infer whether the active cue is row alignment or column alignment, then pair motifs
      to blocks in row-major order within each aligned band.
    - Center each matched motif inside its 5x5 block and repaint those motif cells to the
      background color, then pack the carved blocks with 1-cell gaps and a 1-cell border.

    Key insight:
    The task still uses the motif as a stencil to remove from a block, but both the matching
    cue and the layout changed: correspondence is geometric rather than chromatic, and the
    motifs are threaded through the block arrangement instead of living in a separate region.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks: list[dict[str, int]] = []
    motifs: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if _solid_5x5_block(comp, g):
            blocks.append(
                {
                    "col": int(comp["color"]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": int(r0 + 2),
                    "center_c": int(c0 + 2),
                }
            )
        elif bbox_h(comp["bbox"]) <= 3 and bbox_w(comp["bbox"]) <= 3:
            motifs.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "center_r": int((r0 + r1) // 2),
                    "center_c": int((c0 + c1) // 2),
                    "mask": _extract_mask(comp),
                }
            )

    assert blocks
    pairing = _pair_by_alignment(blocks, motifs)

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = pairing[(int(b["r0"]), int(b["c0"]))]
        mh, mw = map(int, mm.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mm):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        i = int(r_index[int(b["r0"])])
        j = int(c_index[int(b["c0"])])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(b["r0"]), int(b["c0"]))]

    return out


# --------------- generator (combo: alignment + interleaving) ---------------

CANVAS_HEIGHT_RANGE = (18, 24)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
MOTIF_SIZE = 3
GAP_SIZE = 1
PACK_STEP = 6
MOTIF_AREA_RANGE = (4, 8)
FILLED_SLOT_RANGE = (6, 8)
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
GENERATION_RETRIES = 600

DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}


def _band_size(kind: str) -> int:
    if kind == "block":
        return DST_BLOCK_SIZE
    if kind == "motif":
        return MOTIF_SIZE
    if kind == "gap":
        return GAP_SIZE
    raise ValueError(f"unknown band kind: {kind}")


def _span(seq: list[str]) -> int:
    return int(sum(_band_size(kind) for kind in seq))


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
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_block_colors(bg: int, count: int) -> list[int]:
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    return [int(c) for c in palette[:count]]


def _choose_motif_colors(bg: int, block_colors: list[int], total: int) -> list[int]:
    motif_palette = [c for c in range(COLOR_COUNT) if c != bg and c not in block_colors]
    assert motif_palette
    random.shuffle(motif_palette)
    use_k = random.randint(1, len(motif_palette))
    pool = motif_palette[:use_k]
    return [int(random.choice(pool)) for _ in range(total)]


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: list[np.ndarray]
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + (R - 1) * PACK_STEP + DST_BLOCK_SIZE
    out_w = 2 * OUTPUT_BORDER + (C - 1) * PACK_STEP + DST_BLOCK_SIZE
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[idx]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * PACK_STEP,
                OUTPUT_BORDER + j * PACK_STEP,
            )
    return expected


def _make_row_mode_row_seq(R: int, top: int, bottom: int) -> list[str]:
    seq: list[str] = []
    if top:
        seq.append("motif")
    for i in range(R):
        seq.append("block")
        if i < R - 1:
            seq.append("gap")
    if bottom:
        seq.append("motif")
    return seq


def _make_row_mode_col_seq(C: int, left: int, right: int) -> list[str]:
    seq: list[str] = []
    if left:
        seq.append("motif")
    for j in range(C):
        seq.append("block")
        if j < C - 1:
            seq.append("motif")
    if right:
        seq.append("motif")
    return seq


def _make_col_mode_row_seq(R: int, top: int, bottom: int) -> list[str]:
    seq: list[str] = []
    if top:
        seq.append("motif")
    for i in range(R):
        seq.append("block")
        if i < R - 1:
            seq.append("motif")
    if bottom:
        seq.append("motif")
    return seq


def _make_col_mode_col_seq(C: int, left: int, right: int) -> list[str]:
    seq: list[str] = []
    if left:
        seq.append("motif")
    for j in range(C):
        seq.append("block")
        if j < C - 1:
            seq.append("gap")
    if right:
        seq.append("motif")
    return seq


def _choose_filled_range(count: int, distractor_capacity: int) -> tuple[int, int] | None:
    filled_low = max(FILLED_SLOT_RANGE[0], count + 1)
    filled_high = min(FILLED_SLOT_RANGE[1], count + distractor_capacity)
    if filled_low > filled_high:
        return None
    return int(filled_low), int(filled_high)


def _layout_options(H: int, W: int) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for count, layouts in DST_LAYOUTS.items():
        for R, C in layouts:
            for top, bottom in itertools.product((0, 1), repeat=2):
                if top + bottom == 0:
                    continue
                for left, right in itertools.product((0, 1), repeat=2):
                    if left + right == 0:
                        continue

                    # Row mode: interleave along columns, keep simple gaps between block rows.
                    motif_cols = (C - 1) + left + right
                    motif_rows = top + bottom
                    if motif_cols >= C:
                        row_seq = _make_row_mode_row_seq(R, top, bottom)
                        col_seq = _make_row_mode_col_seq(C, left, right)
                        region_h = _span(row_seq)
                        region_w = _span(col_seq)
                        if region_h <= H and region_w <= W:
                            filled = _choose_filled_range(int(count), motif_rows * motif_cols)
                            if filled is not None:
                                options.append(
                                    {
                                        "count": int(count),
                                        "R": int(R),
                                        "C": int(C),
                                        "mode": "row",
                                        "row_seq": row_seq,
                                        "col_seq": col_seq,
                                        "filled_range": filled,
                                    }
                                )

                    # Column mode: interleave along rows, keep simple gaps between block cols.
                    motif_rows = (R - 1) + top + bottom
                    motif_cols = left + right
                    if motif_rows >= R and motif_cols >= 1:
                        row_seq = _make_col_mode_row_seq(R, top, bottom)
                        col_seq = _make_col_mode_col_seq(C, left, right)
                        region_h = _span(row_seq)
                        region_w = _span(col_seq)
                        if region_h <= H and region_w <= W:
                            filled = _choose_filled_range(int(count), motif_rows * motif_cols)
                            if filled is not None:
                                options.append(
                                    {
                                        "count": int(count),
                                        "R": int(R),
                                        "C": int(C),
                                        "mode": "col",
                                        "row_seq": row_seq,
                                        "col_seq": col_seq,
                                        "filled_range": filled,
                                    }
                                )
    return options


def _choose_config(H: int, W: int) -> dict[str, object] | None:
    configs = _layout_options(H, W)
    if not configs:
        return None
    counts = sorted({int(cfg["count"]) for cfg in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if int(cfg["count"]) == count])


def _band_infos(origin: int, seq: list[str], axis: str) -> list[dict[str, int | str]]:
    infos: list[dict[str, int | str]] = []
    cur = int(origin)
    for idx, kind in enumerate(seq):
        size = _band_size(kind)
        infos.append(
            {
                "axis_index": int(idx),
                "kind": kind,
                f"{axis}0": int(cur),
                f"{axis}1": int(cur + size - 1),
                "size": int(size),
            }
        )
        cur += size
    return infos


def _block_cells(
    row_bands: list[dict[str, int | str]], col_bands: list[dict[str, int | str]]
) -> list[dict[str, int]]:
    cells: list[dict[str, int]] = []
    block_rows = [rb for rb in row_bands if rb["kind"] == "block"]
    block_cols = [cb for cb in col_bands if cb["kind"] == "block"]
    for ri, rb in enumerate(block_rows):
        for ci, cb in enumerate(block_cols):
            cells.append(
                {
                    "r0": int(rb["r0"]),
                    "c0": int(cb["c0"]),
                    "ri": int(ri),
                    "ci": int(ci),
                    "h": int(rb["size"]),
                    "w": int(cb["size"]),
                }
            )
    return cells


def _choose_carriers(motif_bands: list[dict[str, int | str]], needed: int) -> list[dict[str, int | str]]:
    assert len(motif_bands) >= needed
    if len(motif_bands) == needed:
        return motif_bands
    if len(motif_bands) == needed + 1:
        if random.choice([True, False]):
            return motif_bands[:needed]
        return motif_bands[1:]
    raise AssertionError("unexpected carrier-band surplus")


def _row_mode_cells(
    row_bands: list[dict[str, int | str]],
    col_bands: list[dict[str, int | str]],
    C: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    block_rows = [rb for rb in row_bands if rb["kind"] == "block"]
    motif_rows = [rb for rb in row_bands if rb["kind"] == "motif"]
    motif_cols = [cb for cb in col_bands if cb["kind"] == "motif"]
    carrier_cols = _choose_carriers(motif_cols, C)

    matching: list[dict[str, int]] = []
    for ri, rb in enumerate(block_rows):
        for ci, cb in enumerate(carrier_cols):
            matching.append(
                {
                    "r0": int(rb["r0"]),
                    "c0": int(cb["c0"]),
                    "ri": int(ri),
                    "ci": int(ci),
                    "h": int(rb["size"]),
                    "w": int(cb["size"]),
                }
            )

    distractors: list[dict[str, int]] = []
    for rb in motif_rows:
        for cb in motif_cols:
            distractors.append(
                {
                    "r0": int(rb["r0"]),
                    "c0": int(cb["c0"]),
                    "ri": int(rb["axis_index"]),
                    "ci": int(cb["axis_index"]),
                    "h": int(rb["size"]),
                    "w": int(cb["size"]),
                }
            )
    return matching, distractors


def _col_mode_cells(
    row_bands: list[dict[str, int | str]],
    col_bands: list[dict[str, int | str]],
    R: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    motif_rows = [rb for rb in row_bands if rb["kind"] == "motif"]
    block_cols = [cb for cb in col_bands if cb["kind"] == "block"]
    motif_cols = [cb for cb in col_bands if cb["kind"] == "motif"]
    carrier_rows = _choose_carriers(motif_rows, R)

    matching: list[dict[str, int]] = []
    for ri, rb in enumerate(carrier_rows):
        for ci, cb in enumerate(block_cols):
            matching.append(
                {
                    "r0": int(rb["r0"]),
                    "c0": int(cb["c0"]),
                    "ri": int(ri),
                    "ci": int(ci),
                    "h": int(rb["size"]),
                    "w": int(cb["size"]),
                }
            )

    distractors: list[dict[str, int]] = []
    for rb in motif_rows:
        for cb in motif_cols:
            distractors.append(
                {
                    "r0": int(rb["r0"]),
                    "c0": int(cb["c0"]),
                    "ri": int(rb["axis_index"]),
                    "ci": int(cb["axis_index"]),
                    "h": int(rb["size"]),
                    "w": int(cb["size"]),
                }
            )
    return matching, distractors


def _paint_destination_blocks(
    grid: np.ndarray, block_cells: list[dict[str, int]], block_colors: list[int]
) -> None:
    for cell, color in zip(block_cells, block_colors):
        patch = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        paste(grid, patch, int(cell["r0"]), int(cell["c0"]))


def _paint_motif_in_cell(
    grid: np.ndarray, cell: dict[str, int], color: int, mask: np.ndarray
) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    off_r = (int(cell["h"]) - MOTIF_SIZE) // 2
    off_c = (int(cell["w"]) - MOTIF_SIZE) // 2
    paste(grid, patch, int(cell["r0"]) + off_r, int(cell["c0"]) + off_c, mask=mask)


def _bbox_union(cells: list[dict[str, int]]) -> tuple[int, int, int, int]:
    r0 = min(int(cell["r0"]) for cell in cells)
    c0 = min(int(cell["c0"]) for cell in cells)
    r1 = max(int(cell["r0"]) + int(cell["h"]) - 1 for cell in cells)
    c1 = max(int(cell["c0"]) + int(cell["w"]) - 1 for cell in cells)
    return int(r0), int(c0), int(r1), int(c1)


def _axis_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _render_input(
    H: int,
    W: int,
    bg: int,
    config: dict[str, object],
    block_colors: list[int],
    match_masks: list[np.ndarray],
    motif_colors: list[int],
    distractor_masks: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    R = int(config["R"])
    C = int(config["C"])
    row_seq = list(config["row_seq"])
    col_seq = list(config["col_seq"])
    region_h = _span(row_seq)
    region_w = _span(col_seq)
    origin_r = random.randint(0, H - region_h)
    origin_c = random.randint(0, W - region_w)

    row_bands = _band_infos(origin_r, row_seq, "r")
    col_bands = _band_infos(origin_c, col_seq, "c")
    block_cells = _block_cells(row_bands, col_bands)
    assert len(block_cells) == R * C

    mode = str(config["mode"])
    if mode == "row":
        matching_cells, distractor_cells = _row_mode_cells(row_bands, col_bands, C)
    else:
        matching_cells, distractor_cells = _col_mode_cells(row_bands, col_bands, R)
    assert len(matching_cells) == R * C
    assert len(distractor_cells) >= len(distractor_masks)

    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, block_cells, block_colors)

    for cell, color, mask in zip(matching_cells, motif_colors[: R * C], match_masks):
        _paint_motif_in_cell(inp, cell, int(color), mask)

    free_cells = distractor_cells.copy()
    random.shuffle(free_cells)
    for cell, color, mask in zip(free_cells, motif_colors[R * C :], distractor_masks):
        _paint_motif_in_cell(inp, cell, int(color), mask)

    meta = {
        "mode": mode,
        "origin_r": int(origin_r),
        "origin_c": int(origin_c),
        "region_h": int(region_h),
        "region_w": int(region_w),
        "block_bbox": _bbox_union(block_cells),
        "motif_bbox": _bbox_union(matching_cells + free_cells[: len(distractor_masks)]),
        "matching_cell_count": int(len(matching_cells)),
        "distractor_cell_capacity": int(len(distractor_cells)),
    }
    return inp, meta


def _gen_attempt(return_meta: bool = False):
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None

    R = int(config["R"])
    C = int(config["C"])
    count = int(config["count"])
    filled_low, filled_high = tuple(config["filled_range"])
    filled = random.randint(int(filled_low), int(filled_high))
    distractor_count = int(filled - count)

    block_colors = _choose_block_colors(bg, count)
    match_masks = [_random_mask() for _ in range(count)]
    distractor_masks = [_random_mask() for _ in range(distractor_count)]
    motif_colors = _choose_motif_colors(bg, block_colors, filled)

    expected = _render_expected(bg, R, C, block_colors, match_masks)
    inp, meta = _render_input(
        H,
        W,
        bg,
        config,
        block_colors,
        match_masks,
        motif_colors,
        distractor_masks,
    )

    if not np.array_equal(main(inp), expected):
        return None

    if not return_meta:
        return inp

    block_bbox = tuple(meta["block_bbox"])
    motif_bbox = tuple(meta["motif_bbox"])
    meta.update(
        {
            "bg": int(bg),
            "R": int(R),
            "C": int(C),
            "filled": int(filled),
            "distractor_count": int(distractor_count),
            "block_colors": tuple(int(c) for c in block_colors),
            "motif_colors": tuple(int(c) for c in motif_colors),
            "motif_palette_overlap": int(len(set(block_colors) & set(motif_colors))),
            "axis_separated": int(_axis_separated(block_bbox, motif_bbox)),
        }
    )
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")


def sample_variant_metrics(n: int = 32) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        sample = _gen_attempt(return_meta=True)
        if sample is None:
            continue
        _inp, meta = sample
        out.append(meta)
    return out

# Variant of 58f5dbd5
# Dimension: D17 — 方块与motif的空间分离方式 (meta)
# Change: 方块区域与motif区域沿一个轴（行或列方向）分离，互不重叠... → 交错排列

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D17 variant
# ----------------------------------
# a) Concrete behavior:
#    The carving rule stays the same, but the input is no longer split into a
#    "block region" and a separate "motif region". Instead, both object types
#    live inside one woven frame. The frame alternates 5-cell block bands and
#    3-cell motif bands with no extra region gap; blocks occupy 5x5 block/block
#    intersections, while motifs occupy the remaining 5x3 / 3x5 / 3x3 cells.
#    Example trace:
#    - Original: solver can conceptually read the top/left area as all blocks
#      and the other area as all motifs.
#    - Variant: a 2x3 block grid may appear with one internal motif row and two
#      internal motif columns, so motif sprites are inserted between and around
#      the blocks. The solver must scan the whole mixed frame, identify all 5x5
#      solids as destinations, and treat the smaller same-colored components as
#      stencils.
#    - Difference: matching now depends on object recognition inside one mixed
#      layout rather than on coarse spatial segregation.
# b) Complexity calibration:
#    The object vocabulary is unchanged: 3-6 blocks, 6-8 occupied motif slots,
#    one matching motif per block color, distractor motifs in distinct colors,
#    and connected 3x3 masks. This keeps the task recognizable while making the
#    spatial parsing meaningfully harder. Thin block layouts (1x3, 3x1) and the
#    2x2 layout receive one outer motif band so the mixed frame still has enough
#    motif capacity without changing the core task.
# c) Old implicit assumptions checked:
#    - `main()` step 1 only assumes solid 5x5 monochrome components exist.
#      That still holds.
#    - `main()` step 2 only assumes the matching motif is another same-colored
#      component smaller than the solid 5x5 block. That still holds, but the
#      generator must prevent a motif from touching its same-colored block under
#      8-connectivity, because the new woven frame is denser.
#    - Packing logic only depends on the discovered block top-left coordinates;
#      it never relied on a separate motif region.
# d) Placement legality:
#    The generator explicitly builds band rectangles, chooses a single origin for
#    the whole mixed frame, and places every object inside a precomputed cell.
#    Matching motifs are assigned with bipartite backtracking to cells whose band
#    coordinates are not edge- or corner-adjacent to their own block cell, which
#    prevents accidental same-color component merging.
# e) Visual observability:
#    Human solvers can still see that large solid 5x5 squares and small connected
#    monochrome sprites share colors. The new behavior is visible because motif
#    sprites now appear threaded between block rows/columns instead of in a clean
#    second region. The mapping is still by visible color, not by hidden ids.
# f) Visual cleanliness:
#    The woven frame uses aligned rectangular cells, so the input remains tidy.
#    Mixed cell types never overlap, and motifs are centered within their host
#    cells, keeping the composition legible even without region separation.
# g) Color inference:
#    Colors are unchanged structurally: each block color appears exactly twice in
#    the input (one 5x5 block, one matching motif), and distractor motif colors
#    are all distinct. No ordering or transformation depends on color ids.
# h) Variant metric used after implementation:
#    For generated samples we measure whether the union bbox of all motifs is
#    axis-separated from the union bbox of all blocks. The original generator
#    always made that true; this variant should make it false essentially always.
# i) New solver assumptions audited:
#    `main()` now implicitly relies on the generator enforcing non-touching
#    same-color block/motif pairs. The generator guarantees this by only using
#    motif cells with cell-grid Chebyshev distance > 1 from the paired block.


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
    Carve color-matched motifs into solid blocks and repack the carved blocks.

    Input structure:
    - bg is the dominant color.
    - Several solid 5x5 monochrome blocks appear in a mixed frame together with
      smaller monochrome motif sprites; blocks and motifs are interleaved rather
      than split into two separate regions.
    - For each block color, there is one smaller same-colored motif somewhere in
      that mixed layout, plus optional distractor motifs in other colors.

    Rule:
    - Detect every solid 5x5 block and match it with the smaller component of
      the same color.
    - Center that motif mask inside the 5x5 block and turn the covered cells
      into background, effectively carving holes out of the block.
    - Output all carved 5x5 blocks packed tightly in their original row/column
      order with 1-cell gaps and a 1-cell background border.

    Key insight:
    The spatial arrangement of motifs changed, but the operative cue did not:
    a motif is a same-colored stencil to remove from its block, not a shape to
    paste somewhere else.
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

    # STEP 3) Carve each 5x5 tile by centering the matched motif and repainting
    # those cells to background.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact output grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[r0])
        j = int(c_index[c0])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


# --------------- generator (D17 mixed layout) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
MOTIF_SIZE = 3
MOTIF_AREA_RANGE = (4, 8)
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2
OUTER_MARGIN = 1
OUTPUT_BORDER = 1
PACK_STEP = 6
MOTIF_OFFSET_IN_BLOCK = 1
GENERATION_RETRIES = 400

DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}


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
    """Render the oracle output corresponding to the sampled block grid."""
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


def _band_sequence(block_count: int, add_before: int, add_after: int) -> list[str]:
    """Return a row/column band type sequence using internal motif bands plus optional outers."""
    seq: list[str] = []
    if add_before:
        seq.append("motif")
    for i in range(block_count):
        seq.append("block")
        if i < block_count - 1:
            seq.append("motif")
    if add_after:
        seq.append("motif")
    return seq


def _band_spans(origin: int, seq: list[str]) -> list[tuple[int, int]]:
    """Return inclusive [start, end] spans for each band type in the sequence."""
    spans: list[tuple[int, int]] = []
    cur = int(origin)
    for band in seq:
        size = DST_BLOCK_SIZE if band == "block" else MOTIF_SIZE
        spans.append((cur, cur + size - 1))
        cur += size
    return spans


def _motif_cell_capacity(row_seq: list[str], col_seq: list[str]) -> int:
    """Count cells that may host a 3x3 motif."""
    total = 0
    for rb in row_seq:
        for cb in col_seq:
            if rb == "block" and cb == "block":
                continue
            total += 1
    return total


def _layout_options(H: int, W: int) -> list[tuple[int, int, int, int, list[str], list[str]]]:
    """Enumerate feasible woven layouts for the sampled canvas."""
    options: list[tuple[int, int, int, int, list[str], list[str]]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            for top in (0, 1):
                for bottom in (0, 1):
                    for left in (0, 1):
                        for right in (0, 1):
                            row_seq = _band_sequence(R, top, bottom)
                            col_seq = _band_sequence(C, left, right)
                            region_h = sum(DST_BLOCK_SIZE if b == "block" else MOTIF_SIZE for b in row_seq)
                            region_w = sum(DST_BLOCK_SIZE if b == "block" else MOTIF_SIZE for b in col_seq)
                            if region_h + 2 * OUTER_MARGIN > H or region_w + 2 * OUTER_MARGIN > W:
                                continue
                            capacity = _motif_cell_capacity(row_seq, col_seq)
                            for filled in range(FILLED_SLOT_RANGE[0], FILLED_SLOT_RANGE[1] + 1):
                                if filled < count:
                                    continue
                                if capacity < filled:
                                    continue
                                if capacity - filled > MAX_EMPTY_SLOTS:
                                    continue
                                options.append((R, C, filled, capacity, row_seq, col_seq))
    return options


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, list[str], list[str]] | None:
    """Sample a feasible mixed-layout configuration."""
    configs = _layout_options(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _filled, _cap, _rs, _cs in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _cell_rects(
    row_seq: list[str], col_seq: list[str], origin_r: int, origin_c: int
) -> list[dict[str, int | str]]:
    """Return cell rectangles in band-grid coordinates."""
    row_spans = _band_spans(origin_r, row_seq)
    col_spans = _band_spans(origin_c, col_seq)
    cells: list[dict[str, int | str]] = []
    for ri, ((r0, r1), rkind) in enumerate(zip(row_spans, row_seq)):
        for ci, ((c0, c1), ckind) in enumerate(zip(col_spans, col_seq)):
            cells.append(
                {
                    "ri": int(ri),
                    "ci": int(ci),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "row_kind": rkind,
                    "col_kind": ckind,
                }
            )
    return cells


def _split_cells(
    cells: list[dict[str, int | str]]
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    """Split all cells into block slots and motif-capable slots."""
    block_cells: list[dict[str, int | str]] = []
    motif_cells: list[dict[str, int | str]] = []
    for cell in cells:
        is_block = cell["row_kind"] == "block" and cell["col_kind"] == "block"
        if is_block:
            block_cells.append(cell)
        else:
            motif_cells.append(cell)
    return block_cells, motif_cells


def _safe_for_pair(block_cell: dict[str, int | str], motif_cell: dict[str, int | str]) -> bool:
    """Return whether a motif cell is far enough from its same-colored block cell."""
    dr = abs(int(block_cell["ri"]) - int(motif_cell["ri"]))
    dc = abs(int(block_cell["ci"]) - int(motif_cell["ci"]))
    return max(dr, dc) > 1


def _assign_matching_slots(
    block_cells: list[dict[str, int | str]], motif_cells: list[dict[str, int | str]]
) -> list[dict[str, int | str]] | None:
    """Assign one non-touching motif cell to each block using backtracking."""
    options: list[list[int]] = []
    for block_cell in block_cells:
        candidates = [
            idx
            for idx, motif_cell in enumerate(motif_cells)
            if _safe_for_pair(block_cell, motif_cell)
        ]
        if not candidates:
            return None
        options.append(candidates)

    order = sorted(range(len(block_cells)), key=lambda i: (len(options[i]), i))
    used: set[int] = set()
    chosen: dict[int, int] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        bi = order[k]
        for mi in options[bi]:
            if mi in used:
                continue
            used.add(mi)
            chosen[bi] = mi
            if dfs(k + 1):
                return True
            used.remove(mi)
            chosen.pop(bi, None)
        return False

    if not dfs(0):
        return None
    return [motif_cells[chosen[i]] for i in range(len(block_cells))]


def _paint_destination_blocks(
    grid: np.ndarray, block_cells: list[dict[str, int | str]], block_colors: list[int]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for cell, color in zip(block_cells, block_colors):
        block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        paste(grid, block, int(cell["r0"]), int(cell["c0"]))


def _paint_motif_in_cell(
    grid: np.ndarray, cell: dict[str, int | str], color: int, mask: np.ndarray
) -> None:
    """Paint one 3x3 motif centered inside its assigned cell."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    off_r = (int(cell["h"]) - MOTIF_SIZE) // 2
    off_c = (int(cell["w"]) - MOTIF_SIZE) // 2
    paste(grid, patch, int(cell["r0"]) + off_r, int(cell["c0"]) + off_c, mask=mask)


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif cells."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[color]) for color in block_colors
    ]
    items.extend((int(color), _random_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _render_input(
    H: int,
    W: int,
    bg: int,
    row_seq: list[str],
    col_seq: list[str],
    block_cells: list[dict[str, int | str]],
    motif_cells: list[dict[str, int | str]],
    matching_cells: list[dict[str, int | str]],
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the mixed woven frame into the input grid."""
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, block_cells, block_colors)

    matching_lookup = {
        (int(cell["ri"]), int(cell["ci"])): idx for idx, cell in enumerate(matching_cells)
    }
    used_keys = set(matching_lookup)

    for cell in matching_cells:
        idx = matching_lookup[(int(cell["ri"]), int(cell["ci"]))]
        color = int(block_colors[idx])
        _paint_motif_in_cell(inp, cell, color, motif_masks[color])

    distractor_items = [(int(color), _random_mask()) for color in distractor_colors]
    free_cells = [cell for cell in motif_cells if (int(cell["ri"]), int(cell["ci"])) not in used_keys]
    random.shuffle(free_cells)
    for cell, (color, mask) in zip(free_cells, distractor_items):
        _paint_motif_in_cell(inp, cell, color, mask)

    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, filled, _capacity, row_seq, col_seq = config

    region_h = sum(DST_BLOCK_SIZE if b == "block" else MOTIF_SIZE for b in row_seq)
    region_w = sum(DST_BLOCK_SIZE if b == "block" else MOTIF_SIZE for b in col_seq)
    origin_r = random.randint(OUTER_MARGIN, H - region_h - OUTER_MARGIN)
    origin_c = random.randint(OUTER_MARGIN, W - region_w - OUTER_MARGIN)

    cells = _cell_rects(row_seq, col_seq, origin_r, origin_c)
    block_cells, motif_cells = _split_cells(cells)
    if len(block_cells) != R * C:
        return None

    matching_cells = _assign_matching_slots(block_cells, motif_cells)
    if matching_cells is None:
        return None

    count = R * C
    if filled < count:
        return None
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        row_seq,
        col_seq,
        block_cells,
        motif_cells,
        matching_cells,
        block_colors,
        distractor_colors,
        motif_masks,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D17 input")

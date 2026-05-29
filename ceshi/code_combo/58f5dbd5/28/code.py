# Combo variant of 58f5dbd5 (index 28)
# combo_spec: D10/2+D18/2
# Name: motif位置标记已使用+透明背景

from __future__ import annotations

import itertools
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import crop_bbox, find_components, get_bg, is_contiguous, paste


"""
Design analysis for combo D10/2 + D18/2
=======================================

a) Concrete behavior and trace
------------------------------
- Original task:
  each same-colored 3x3 motif is only a stencil. The output keeps only the carved 5x5
  blocks, packed tightly, and every non-block pixel in that packed canvas is just `bg`.
- Combo task:
  the output has two packed bands.
  1. Top band: the original carved 5x5 block grid, but only carved holes use `bg`.
  2. Bottom band: the motif-slot grid is packed too; every matching motif footprint is
     recolored to a fixed `USED_COLOR`, while distractor motifs disappear.
  All outer border cells, all inter-block gaps, all inter-slot gaps, and all empty motif
  slots use a fixed `EMPTY_COLOR`.
- Mini-trace:
  suppose block colors are red/green/blue and the visible motif grid has six slots:
  red, cyan, green / yellow, blue, orange.
  Original output: only three carved 5x5 tiles, with `bg` in the carved centers and `bg`
  also everywhere around them.
  Combo output: the same three carved tiles appear in the top band; below them, the red,
  green, and blue motif shapes reappear at their original slot positions but recolored to
  `USED_COLOR`; cyan/yellow/orange slots become `EMPTY_COLOR`.
- Difference from original:
  the source motifs no longer vanish without trace, and `bg` no longer means generic blank
  output space. It means only "carved hole inside a block".

b) Complexity calibration
-------------------------
- The combo introduces only two extra visible semantics, both with fixed colors:
  `USED_COLOR` for "this motif was consumed" and `EMPTY_COLOR` for all packed-canvas
  padding and gaps.
- To keep those semantics visually stable, the generator reserves both colors globally and
  caps the number of visible motif colors at 7. That prevents palette collisions with the
  10-color ARC palette.
- The block grid is restricted to the wider layouts `(1x3)`, `(2x2)`, `(2x3)`, and motif
  grids to `(2x3)` or `(2x4)`. This keeps the two-band output compact and easy to read.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- "Output can be initialized with `bg`":
  broken by D18/2. Now `bg` must be reserved for carved holes only, so the packed output
  canvas must start as `EMPTY_COLOR`.
- "Only the block grid needs to be reconstructed":
  broken by D10/2. The solver must also recover the motif-slot lattice from visible motif
  bboxes so it can render the used-marker band.
- "Motif source disappears entirely":
  broken by D10/2. Matching motifs now survive as footprints, but only in a second packed
  panel and only in the special marker color.
- "There is no semantic distinction among non-block pixels":
  broken by the fusion. The updated solver must separate:
  1. block color,
  2. carved-hole `bg`,
  3. packed-space `EMPTY_COLOR`,
  4. consumed-motif `USED_COLOR`.

d) Placement legality
---------------------
- Destination blocks still occupy a rectangular lattice with 5x5 bodies and step 6.
- Visible motifs still occupy a rectangular slot lattice with 3x3 bodies and step 4.
- When capacity is 8, exactly one slot may be empty; because each row/column contains at
  least two slots, removing one slot never erases a whole row or column, so the solver can
  still recover the full motif lattice from the remaining visible motifs.
- Generator placement keeps destination and motif regions axis-separated with margins, so
  no components overlap or merge.

e) Visual observability
-----------------------
- The fusion is easy to spot in 2-4 examples:
  the output always has a lower used-marker band;
  those lower shapes all share one fixed color;
  the top band uses the sample's input `bg` only inside carved holes;
  all surrounding blank space is the other fixed color.
- No rule depends on numeric color ids or hidden ordering. All cues are geometric or based
  on fixed recurring colors.

f) Visual cleanliness
---------------------
- The two-band output prevents stacking marks on top of carved blocks, so the three output
  semantics stay visually disjoint.
- Wide layouts avoid very tall skinny outputs.
- Empty slots and distractor slots are both rendered as plain `EMPTY_COLOR`, which keeps
  the lower band readable instead of noisy.

g) Change inferability
----------------------
- Color behavior is structured:
  block colors are copied directly from input;
  carved holes always use the input `bg`;
  used motifs always use the fixed `USED_COLOR`;
  all packing/background gaps always use the fixed `EMPTY_COLOR`.
- Because `USED_COLOR` and `EMPTY_COLOR` recur across samples while `bg` varies by sample,
  a human can infer that `bg` is not the global background anymore.

h) Post-implementation combo metric
-----------------------------------
- Core metric 1: every generated output must contain `USED_COLOR` cells.
- Core metric 2: every generated output must contain both `EMPTY_COLOR` cells and `bg`
  cells, and those two counts should both be positive.
- Core metric 3: the number of used-marker motifs in the lower band must equal the number
  of destination blocks in the upper band.
- Supporting metric: many samples should also include distractor or empty motif slots, so
  the lower band is not trivially full.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: reserved colors never appear as input object colors.
  Guarantee: generator draws `bg`, block colors, and distractor colors from the palette
  excluding `EMPTY_COLOR` and `USED_COLOR`.
- Assumption: every visible motif touches all four sides of its 3x3 bbox, so bbox top-left
  is the slot origin.
  Guarantee: generator samples only such masks.
- Assumption: blocks and motifs each lie on a shared row/column lattice.
  Guarantee: generator places them on fixed-step grids.
- Assumption: each block color appears in exactly one visible motif.
  Guarantee: generator constructs one matching motif per block color and uses distinct
  distractor colors for all non-matching motifs.
"""


EMPTY_COLOR = 0
USED_COLOR = 5
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
MOTIF_SIZE = 3
MOTIF_STEP = 4
OUTER_BORDER = 1
PANEL_GAP = 1
CANVAS_HEIGHT_RANGE = (21, 22)
CANVAS_WIDTH_RANGE = (21, 24)

DST_LAYOUTS = {
    3: [(1, 3)],
    4: [(2, 2)],
    6: [(2, 3)],
}
MOTIF_LAYOUTS = {
    6: [(2, 3)],
    8: [(2, 4)],
}
FILLED_OPTIONS = {
    3: [6, 7],
    4: [6, 7],
    6: [7],
}

MOTIF_AREA_RANGE = (4, 8)
OUTER_MARGIN = 1
REGION_GAP = 1
GENERATION_RETRIES = 400


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the provided relative coords."""
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = True
    return out


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return int((count - 1) * step + item_size)


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    masks = {area: [] for area in range(MOTIF_AREA_RANGE[0], MOTIF_AREA_RANGE[1] + 1)}
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
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _collect_blocks_and_motifs(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    blocks: list[dict] = []
    motifs: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        b = comp["bbox"]
        r0, c0, r1, c1 = map(int, b)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch = crop_bbox(g, b)
        if h == DST_BLOCK_SIZE and w == DST_BLOCK_SIZE and bool((patch == color).all()):
            blocks.append({"color": color, "r0": r0, "c0": c0})
            continue

        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = mask_from_coords(rel_coords, (h, w))
        motifs.append({"color": color, "r0": r0, "c0": c0, "mask": mask, "shape": (h, w)})

    blocks.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    motifs.sort(key=lambda d: (int(d["r0"]), int(d["c0"]), int(d["color"])))
    return bg, blocks, motifs


def _carved_tile(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    if motif_mask is None:
        return tile

    mh, mw = map(int, motif_mask.shape)
    off_r = (DST_BLOCK_SIZE - mh) // 2
    off_c = (DST_BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(motif_mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
            tile[tr, tc] = int(bg)
    return tile


def _render_output(
    bg: int,
    blocks: list[dict],
    motifs: list[dict],
) -> np.ndarray:
    assert blocks
    assert motifs

    block_rows = sorted({int(b["r0"]) for b in blocks})
    block_cols = sorted({int(b["c0"]) for b in blocks})
    motif_rows = sorted({int(m["r0"]) for m in motifs})
    motif_cols = sorted({int(m["c0"]) for m in motifs})

    motif_by_color = {int(m["color"]): m for m in motifs}
    top_h = _spanned_size(len(block_rows), DST_BLOCK_SIZE, DST_BLOCK_STEP)
    top_w = _spanned_size(len(block_cols), DST_BLOCK_SIZE, DST_BLOCK_STEP)
    bot_h = _spanned_size(len(motif_rows), MOTIF_SIZE, MOTIF_STEP)
    bot_w = _spanned_size(len(motif_cols), MOTIF_SIZE, MOTIF_STEP)

    inner_w = max(top_w, bot_w)
    out_h = int(2 * OUTER_BORDER + top_h + PANEL_GAP + bot_h)
    out_w = int(2 * OUTER_BORDER + inner_w)
    out = np.full((out_h, out_w), EMPTY_COLOR, dtype=int)

    top_r = int(OUTER_BORDER)
    bot_r = int(OUTER_BORDER + top_h + PANEL_GAP)
    top_c = int(OUTER_BORDER + (inner_w - top_w) // 2)
    bot_c = int(OUTER_BORDER + (inner_w - bot_w) // 2)

    block_row_index = {int(r): idx for idx, r in enumerate(block_rows)}
    block_col_index = {int(c): idx for idx, c in enumerate(block_cols)}
    for b in blocks:
        color = int(b["color"])
        motif = motif_by_color.get(color)
        tile = _carved_tile(color, bg, None if motif is None else motif["mask"])
        rr = int(top_r + block_row_index[int(b["r0"])] * DST_BLOCK_STEP)
        cc = int(top_c + block_col_index[int(b["c0"])] * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = tile

    motif_row_index = {int(r): idx for idx, r in enumerate(motif_rows)}
    motif_col_index = {int(c): idx for idx, c in enumerate(motif_cols)}
    block_colors = {int(b["color"]) for b in blocks}
    for m in motifs:
        if int(m["color"]) not in block_colors:
            continue
        rr = int(bot_r + motif_row_index[int(m["r0"])] * MOTIF_STEP)
        cc = int(bot_c + motif_col_index[int(m["c0"])] * MOTIF_STEP)
        patch = np.full((MOTIF_SIZE, MOTIF_SIZE), USED_COLOR, dtype=int)
        paste(out, patch, rr, cc, mask=m["mask"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve matching motifs into 5x5 blocks, then also mark where the consumed motifs came from.

    Input structure:
    - `bg` is still the most common color in the input canvas.
    - Several solid 5x5 monochrome blocks appear in one separated region.
    - A second separated region contains 3x3 monochrome motif sprites on a slot lattice; some
      motifs match block colors and the rest are distractors.

    Rule:
    - Match each block to the motif with the same color.
    - In the top output band, center that motif inside the 5x5 block and recolor the motif cells
      to `bg`, producing carved holes.
    - In the bottom output band, keep only the footprints of the matching motifs at their original
      slot positions, recolored to `USED_COLOR`.
    - Fill all border cells, panel gaps, block gaps, slot gaps, distractor slots, and empty slots
      with the fixed `EMPTY_COLOR`.

    Key insight:
    - The fused task distinguishes three non-block meanings in the output: `bg` means carved hole
      inside a block, `USED_COLOR` means a motif was consumed there, and `EMPTY_COLOR` means packed
      canvas space that is not part of any object.
    """
    bg, blocks, motifs = _collect_blocks_and_motifs(np.array(grid, dtype=int))
    assert blocks
    assert motifs
    return _render_output(bg, blocks, motifs)


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    palette = [c for c in range(COLOR_COUNT) if c not in {bg, EMPTY_COLOR, USED_COLOR}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count:filled]
    return block_colors, distractor_colors


def _paint_destination_blocks(
    grid: np.ndarray,
    dst_r: int,
    dst_c: int,
    rows: int,
    cols: int,
    block_colors: list[int],
) -> None:
    for i in range(rows):
        for j in range(cols):
            color = int(block_colors[i * cols + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int,
    W: int,
    dh: int,
    dw: int,
    mh: int,
    mw: int,
    scheme: str,
) -> tuple[int, int, int, int]:
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
    H: int,
    W: int,
    dh: int,
    dw: int,
    mh: int,
    mw: int,
    scheme: str,
) -> tuple[int, int, int, int]:
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
    H: int,
    W: int,
    dh: int,
    dw: int,
    mh: int,
    mw: int,
) -> tuple[int, int, int, int] | None:
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int, int]] = []
    for count, layouts in DST_LAYOUTS.items():
        for rows, cols in layouts:
            dst_h = _spanned_size(rows, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dst_w = _spanned_size(cols, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for filled in FILLED_OPTIONS[count]:
                capacity = 6 if filled == 6 else 8
                for mr, mc in MOTIF_LAYOUTS[capacity]:
                    motif_h = _spanned_size(mr, MOTIF_SIZE, MOTIF_STEP)
                    motif_w = _spanned_size(mc, MOTIF_SIZE, MOTIF_STEP)
                    if _placement_schemes(H, W, dst_h, dst_w, motif_h, motif_w):
                        out.append((rows, cols, mr, mc, filled, capacity))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int, int] | None:
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({rows * cols for rows, cols, _mr, _mc, _filled, _cap in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _slot_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(rows) for j in range(cols)]


def _render_expected(
    bg: int,
    block_rows: int,
    block_cols: int,
    motif_rows: int,
    motif_cols: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    slot_payloads: dict[tuple[int, int], tuple[int, np.ndarray]],
) -> np.ndarray:
    blocks: list[dict] = []
    for i in range(block_rows):
        for j in range(block_cols):
            blocks.append({"color": int(block_colors[i * block_cols + j]), "r0": i, "c0": j})

    motifs: list[dict] = []
    for (sr, sc), (color, mask) in slot_payloads.items():
        motifs.append({"color": int(color), "r0": int(sr), "c0": int(sc), "mask": mask, "shape": mask.shape})

    # Convert logical row/col indices into lattice coordinates so `_render_output` can reuse the same code path.
    for b in blocks:
        b["r0"] = int(b["r0"] * DST_BLOCK_STEP)
        b["c0"] = int(b["c0"] * DST_BLOCK_STEP)
    for m in motifs:
        m["r0"] = int(m["r0"] * MOTIF_STEP)
        m["c0"] = int(m["c0"] * MOTIF_STEP)

    return _render_output(bg, blocks, motifs)


def _render_input(
    H: int,
    W: int,
    bg: int,
    block_rows: int,
    block_cols: int,
    motif_rows: int,
    motif_cols: int,
    block_colors: list[int],
    slot_payloads: dict[tuple[int, int], tuple[int, np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    grid = np.full((H, W), bg, dtype=int)

    _paint_destination_blocks(grid, dst_r, dst_c, block_rows, block_cols, block_colors)
    for (sr, sc), (color, mask) in slot_payloads.items():
        rr = int(motif_r + sr * MOTIF_STEP)
        cc = int(motif_c + sc * MOTIF_STEP)
        _paint_motif(grid, rr, cc, int(color), mask)
    return grid


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.choice([c for c in range(COLOR_COUNT) if c not in {EMPTY_COLOR, USED_COLOR}])

    config = _choose_config(H, W)
    if config is None:
        return None
    block_rows, block_cols, motif_rows, motif_cols, filled, capacity = config

    dst_h = _spanned_size(block_rows, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(block_cols, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(motif_rows, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(motif_cols, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = int(block_rows * block_cols)
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {int(color): _random_mask() for color in block_colors}

    slot_positions = _slot_positions(motif_rows, motif_cols)
    random.shuffle(slot_positions)
    slot_payloads: dict[tuple[int, int], tuple[int, np.ndarray]] = {}

    items: list[tuple[int, np.ndarray]] = [(int(color), motif_masks[int(color)]) for color in block_colors]
    items.extend((int(color), _random_mask()) for color in distractor_colors)
    random.shuffle(items)
    for slot, item in zip(slot_positions, items):
        slot_payloads[slot] = item

    # Capacity 8 with filled 7 leaves one empty slot, which is intentional and recoverable.
    assert len(slot_payloads) == filled
    assert capacity - filled in {0, 1}

    expected = _render_expected(
        bg,
        block_rows,
        block_cols,
        motif_rows,
        motif_cols,
        block_colors,
        motif_masks,
        slot_payloads,
    )
    inp = _render_input(
        H,
        W,
        bg,
        block_rows,
        block_cols,
        motif_rows,
        motif_cols,
        block_colors,
        slot_payloads,
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

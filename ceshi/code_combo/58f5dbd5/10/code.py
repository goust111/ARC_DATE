# Combo variant of 58f5dbd5 (index 10)
# combo_spec: D8/1+D16/0
# Name: 标记锚点对齐+图例指示对齐方向

from __future__ import annotations

"""
Design analysis for the D8/1 + D16/0 fusion.

a) Concrete behavior and trace:
   - Original task:
     each matched motif is carved into its same-colored 5x5 block at the fixed centered offset (1,1).
   - Fused task:
     each block now contains one cue-colored marker pixel, and one corner 2x2 legend says how that marker
     should be read: it is the motif's upper-left anchor, its center anchor, or its lower-right anchor.
   - Example trace:
     suppose a motif occupies mask cells {(0,0),(0,1),(1,1),(2,2)} in a 3x3 bbox and a block marker sits at
     local tile position (1,2). If the legend means "center", the motif anchor is (1,1), so its top-left becomes
     (0,1). If the legend means "upper_left", the anchor is (0,0), so the top-left becomes (1,2). The same block
     marker therefore yields different carved holes depending on the corner symbol.

b) Complexity calibration:
   - The legend has only three reusable states: upper-left, center, lower-right.
   - Marker pixels share one cue color across all blocks, so their role is visually stable.
   - Marker positions vary across blocks inside the same sample, preventing collapse to a simple global offset rule.
   - Matching still uses color equality between motif and block body, so the new difficulty is only alignment logic.

c) Implicit-assumption audit against the original main():
   - Original block detection expected a solid 5x5 monochrome square. That breaks because each block now has
     one foreign cue pixel, so the solver must detect a 24-cell same-colored component whose 5x5 patch contains
     exactly one non-bg, non-body pixel.
   - Original carving hard-coded centering. That breaks because alignment now depends on both the legend and the
     per-block marker.
   - Original motif lookup skipped only solid 5x5 blocks. That breaks because destination blocks are now 5x5 /
     area-24 components, so the solver must skip those instead.
   - Original output packing uses only block top-left coordinates; that still holds unchanged.

d) Placement legality:
   - Matching motifs still have exact 3x3 bounding boxes, so a marker interpreted as upper-left, center, or
     lower-right can be kept in a legal 3x3 set of tile positions without clipping.
   - The corner legend occupies one 2x2 corner patch, and generator rejects placements where the destination
     region or motif region touches any corner legend patch.
   - Cue color is reserved for marker pixels and the legend only, never for motifs or block bodies.

e) Visual observability:
   - A solver can first spot the repeated cue-colored singleton inside every block, then notice the same cue color
     forming a tiny corner symbol.
   - Because marker positions differ across blocks, outputs cannot be explained by a single whole-tile offset.
   - Because the legend changes across samples, the marker cannot be interpreted with one fixed anchor semantics.

f) Visual cleanliness:
   - The legend is isolated at a corner on background cells.
   - Each block carries only one extra pixel, so the destination region stays readable.
   - Motif slots remain in their original separate region, avoiding symbol clutter around the blocks.

g) Change inferability:
   - No hidden color ordering is used. The only new color rule is structural: one reserved cue color marks
     "instructional" pixels, while body colors still drive motif↔block matching.
   - Generator rejects samples where all markers collapse to the neutral position for the chosen legend, because
     those would imitate the older fixed-placement behavior too closely.

New solver assumptions introduced here and guaranteed by the generator:
   - Exactly one corner 2x2 patch contains the legend mask.
   - Every destination block has exactly one cue-colored marker pixel.
   - Matching motifs have 3x3 bounding boxes and include the legend-selected anchor cell, so the cue pixel maps
     to a visible carved hole.
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path("/home/lixinyu/snap/new")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


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

ALIGNMENT_MODES = tuple(ALIGNMENT_MASKS.keys())
LEGEND_SIZE = 2
CORNER_SLICES = (
    (slice(0, 2), slice(0, 2)),
    (slice(0, 2), slice(-2, None)),
    (slice(-2, None), slice(0, 2)),
    (slice(-2, None), slice(-2, None)),
)
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
NEUTRAL_MARKER_BY_MODE = {
    "upper_left": (1, 1),
    "center": (2, 2),
    "lower_right": (3, 3),
}
MARKER_CHOICES_BY_MODE = {
    "upper_left": [(r, c) for r in range(0, 3) for c in range(0, 3)],
    "center": [(r, c) for r in range(1, 4) for c in range(1, 4)],
    "lower_right": [(r, c) for r in range(2, 5) for c in range(2, 5)],
}


def _decode_anchor_mode(grid: np.ndarray, bg: int) -> str:
    """Read the 2x2 corner legend and return the marker-interpretation mode it encodes."""
    for rs, cs in CORNER_SLICES:
        patch = np.array(grid[rs, cs], dtype=int)
        legend_mask = patch != int(bg)
        if not np.any(legend_mask):
            continue
        for mode, code in ALIGNMENT_MASKS.items():
            if np.array_equal(legend_mask, code):
                return mode
    raise AssertionError("missing or invalid corner legend")


def _motif_anchor(mode: str, mh: int, mw: int) -> tuple[int, int]:
    """Return which motif cell the block marker refers to for the decoded mode."""
    if mode == "upper_left":
        return 0, 0
    if mode == "center":
        return mh // 2, mw // 2
    if mode == "lower_right":
        return mh - 1, mw - 1
    raise ValueError(f"unknown mode: {mode}")


def _extract_block(comp: dict, grid: np.ndarray, bg: int, mode: str) -> dict[str, int] | None:
    """Recognize a 5x5 destination block whose singleton cue pixel is interpreted via the legend."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5 or int(comp["area"]) != 24:
        return None

    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    if int((patch == int(col)).sum()) != 24:
        return None
    if int((patch == int(bg)).sum()) != 0:
        return None

    others = np.argwhere(patch != int(col))
    if others.shape[0] != 1:
        return None

    ar, ac = map(int, others[0])
    marker_col = int(patch[int(ar), int(ac)])
    if int(marker_col) in {int(bg), int(col)}:
        return None
    if (int(ar), int(ac)) not in MARKER_CHOICES_BY_MODE[mode]:
        return None

    r0, c0, _r1, _c1 = map(int, b)
    return {
        "col": int(col),
        "r0": int(r0),
        "c0": int(c0),
        "anchor_r": int(ar),
        "anchor_c": int(ac),
        "marker_col": int(marker_col),
    }


def _collect_motifs(grid: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, np.ndarray]:
    """Pick the smallest same-colored non-block component for each destination color."""
    motifs: dict[int, np.ndarray] = {}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(bcol)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5 and int(comp["area"]) == 24:
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
    return motifs


def _carve_blocks(blocks: list[dict[str, int]], bg: int, motifs: dict[int, np.ndarray], mode: str) -> dict[tuple[int, int], np.ndarray]:
    """Carve every 5x5 block by aligning the legend-selected motif anchor to the block marker."""
    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        col = int(block["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motif = motifs.get(int(col))
        if motif is not None:
            mh, mw = map(int, motif.shape)
            anchor_r, anchor_c = _motif_anchor(mode, mh, mw)
            off_r = int(block["anchor_r"]) - int(anchor_r)
            off_c = int(block["anchor_c"]) - int(anchor_c)
            for rr, cc in np.argwhere(motif):
                tile[int(off_r + int(rr)), int(off_c + int(cc))] = int(bg)
        carved[(int(block["r0"]), int(block["c0"]))] = tile
    return carved


def _pack_blocks(blocks: list[dict[str, int]], bg: int, carved: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Pack carved 5x5 results using the original destination row/column order."""
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        rr = int(1 + r_index[int(r0)] * step)
        cc = int(1 + c_index[int(c0)] * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]
    return out


def _solve_with_mode(grid: np.ndarray, forced_mode: str | None = None) -> np.ndarray:
    """Core solver shared by main() and generator validation."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mode = forced_mode or _decode_anchor_mode(g, bg)

    # STEP 1) Identify destination blocks: 5x5 almost-solid squares with one cue marker pixel.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        block = _extract_block(comp, g, bg, mode)
        if block is not None:
            blocks.append(block)

    assert blocks

    # STEP 2) For each block color, identify its motif by same-color component lookup.
    block_colors = {int(block["col"]) for block in blocks}
    motifs = _collect_motifs(g, bg, block_colors)

    # STEP 3) Carve each 5x5 tile by aligning the legend-selected motif anchor to the block's marker.
    carved = _carve_blocks(blocks, bg, motifs, mode)

    # STEP 4) Pack the carved tiles into the compact output layout.
    return _pack_blocks(blocks, bg, carved)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into marked 5x5 blocks, with a corner legend deciding how each marker is read.

    Input structure:
    - bg is the most common color.
    - Several destination blocks are 5x5 almost-solid squares: 24 body cells of one color plus 1 cue-colored marker pixel.
    - Elsewhere, each destination color has a small same-colored motif sprite; distractor motifs may also exist.
    - A 2x2 legend sits on one grid corner and encodes whether the marker stands for the motif's upper-left, center, or lower-right anchor.

    Rule:
    - Decode the corner legend first.
    - Match each motif to the block with the same body color, then place that motif so the legend-selected motif anchor lands on the block's cue marker.
    - Recolor the covered motif cells to bg and finally pack all carved 5x5 blocks in original row/column order with 1-cell gaps and a 1-cell border.

    Key insight:
    - The cue pixel inside a block is only half the instruction: it gives a location, while the corner legend tells you which motif point should be attached to that location.
    """
    return _solve_with_mode(grid)


# --------------- generator (o5x / combo D8+D16) ---------------

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
BLOCK_COUNT_OPTIONS = tuple(sorted(DST_LAYOUTS))

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

# Generation retry budget.
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 30
_GENERATE_CALL_COUNT = 0


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks with exact 3x3 bbox."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
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
MASKS_BY_MODE_AND_AREA: dict[str, dict[int, list[np.ndarray]]] = {}
for _mode in ALIGNMENT_MODES:
    ar, ac = _motif_anchor(_mode, MOTIF_SIZE, MOTIF_SIZE)
    MASKS_BY_MODE_AND_AREA[_mode] = {
        area: [mask.copy() for mask in masks if bool(mask[int(ar), int(ac)])]
        for area, masks in MASKS_BY_AREA.items()
    }


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


def _choose_config(H: int, W: int, preferred_count: int | None = None) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout, optionally preferring one destination-block count branch."""
    configs = _configs(H, W)
    if not configs:
        return None
    if preferred_count is not None:
        preferred = [cfg for cfg in configs if cfg[0] * cfg[1] == preferred_count]
        if not preferred:
            return None
        return random.choice(preferred)
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask(mode: str) -> np.ndarray:
    """Sample one motif mask that exposes the legend-selected anchor cell."""
    available = [area for area, masks in MASKS_BY_MODE_AND_AREA[mode].items() if masks]
    area = random.choice(available)
    return random.choice(MASKS_BY_MODE_AND_AREA[mode][area]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int], int]:
    """Pick block colors, distractor colors, and one shared cue color."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count:filled]
    cue_color = int(palette[filled])
    return block_colors, distractor_colors, cue_color


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into the grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _choose_anchor_positions(count: int, mode: str) -> list[tuple[int, int]]:
    """Sample legal marker positions while preventing collapse to the neutral fixed-offset case."""
    choices = MARKER_CHOICES_BY_MODE[mode]
    neutral = NEUTRAL_MARKER_BY_MODE[mode]
    while True:
        anchors = [random.choice(choices) for _ in range(count)]
        if len(set(anchors)) < 2:
            continue
        if not any(anchor != neutral for anchor in anchors):
            continue
        return anchors


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
    mode: str,
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            motif = motif_masks[color]
            anchor_r, anchor_c = anchors[idx]
            motif_ar, motif_ac = _motif_anchor(mode, *motif.shape)
            off_r = int(anchor_r) - int(motif_ar)
            off_c = int(anchor_c) - int(motif_ac)
            for rr, cc in np.argwhere(motif):
                tile[int(off_r + int(rr)), int(off_c + int(cc))] = int(bg)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _render_global_alignment_baseline(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Render the old D16-style output where the legend controls one global tile offset only."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            motif = motif_masks[color]
            if mode == "upper_left":
                off_r, off_c = 0, 0
            elif mode == "center":
                off_r = (DST_BLOCK_SIZE - motif.shape[0]) // 2
                off_c = (DST_BLOCK_SIZE - motif.shape[1]) // 2
            else:
                off_r = DST_BLOCK_SIZE - motif.shape[0]
                off_c = DST_BLOCK_SIZE - motif.shape[1]
            for rr, cc in np.argwhere(motif):
                tile[int(off_r + int(rr)), int(off_c + int(cc))] = int(bg)
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
    anchors: list[tuple[int, int]],
    cue_color: int,
) -> None:
    """Paint 5x5 destination blocks with their internal cue markers."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            ar, ac = anchors[idx]
            block[int(ar), int(ac)] = int(cue_color)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [(int(color), motif_masks[color]) for color in block_colors]
    items.extend((int(color), _random_mask(random.choice(ALIGNMENT_MODES))) for color in distractor_colors)
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
    raise ValueError(f"unknown corner: {corner}")


def _paint_legend(grid: np.ndarray, bg: int, cue_color: int, corner: str, mode: str) -> None:
    """Paint the cue-colored 2x2 legend into one corner of the input."""
    r0, c0 = _corner_origin(*grid.shape, corner)
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(bg), dtype=int)
    patch[ALIGNMENT_MASKS[mode]] = int(cue_color)
    grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = patch


def _next_coverage_targets() -> tuple[str, str, int]:
    """Cycle through generator branches so repeated sampling covers modes, corners, and block counts evenly."""
    global _GENERATE_CALL_COUNT
    idx = _GENERATE_CALL_COUNT
    _GENERATE_CALL_COUNT += 1
    mode = ALIGNMENT_MODES[idx % len(ALIGNMENT_MODES)]
    corner = LEGEND_CORNERS[idx % len(LEGEND_CORNERS)]
    preferred_count = BLOCK_COUNT_OPTIONS[(idx // 2) % len(BLOCK_COUNT_OPTIONS)]
    return mode, corner, preferred_count


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two inclusive rectangles overlap."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend_bbox(H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    """Inclusive bbox occupied by the corner legend."""
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
    """Reject placements where either major region spills into a corner legend patch."""
    dst_r, dst_c, motif_r, motif_c = placement
    dst_box = (dst_r, dst_c, dst_r + dst_h - 1, dst_c + dst_w - 1)
    motif_box = (motif_r, motif_c, motif_r + motif_h - 1, motif_c + motif_w - 1)
    for corner in LEGEND_CORNERS:
        corner_box = _legend_bbox(H, W, corner)
        if _rects_overlap(dst_box, corner_box) or _rects_overlap(motif_box, corner_box):
            return False
    return True


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
    cue_color: int,
    motif_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
    legend_corner: str,
    mode: str,
) -> np.ndarray:
    """Render destination blocks, motif slots, and the corner legend into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, anchors, cue_color)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks)
    _paint_legend(inp, bg, cue_color, legend_corner, mode)
    return inp


def _gen_attempt(requested_mode: str, requested_corner: str, preferred_count: int) -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees or the fusion degenerates."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W, preferred_count=preferred_count)
    if config is None:
        return None
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

    legend_corner = requested_corner
    mode = requested_mode

    placement = None
    for _ in range(PLACEMENT_RETRIES):
        candidate = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
        if candidate is None:
            continue
        if _placement_avoids_corners(H, W, candidate, dst_h, dst_w, motif_h, motif_w):
            placement = candidate
            break
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors, cue_color = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask(mode) for color in block_colors}
    anchors = _choose_anchor_positions(count, mode)

    expected = _render_expected(bg, R, C, block_colors, motif_masks, anchors, mode)
    if np.array_equal(expected, _render_global_alignment_baseline(bg, R, C, block_colors, motif_masks, mode)):
        return None

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
        cue_color,
        motif_masks,
        anchors,
        placement,
        legend_corner,
        mode,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    mode, legend_corner, preferred_count = _next_coverage_targets()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(mode, legend_corner, preferred_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

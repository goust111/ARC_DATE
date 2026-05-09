# Combo variant of 58f5dbd5 (index 25)
# combo_spec: D7/1+D8/1+D16/0
# Name: 颜色替换+标记锚点+图例指示方向

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


"""
Design analysis for the combo variant D7/1 + D8/1 + D16/0.

combo_spec:
  concrete_behavior_trace:
    - Original task:
      1. Find a same-colored 3x3 motif for each solid 5x5 block.
      2. Always center the motif inside the block.
      3. Paint the motif footprint to background, then pack all 5x5 tiles.
    - Combo task:
      1. Decode a 2x2 corner legend. It no longer means a fixed global offset; it tells which motif reference point
         is the anchor: upper-left cell, center cell, or lower-right cell of the motif bbox.
      2. Each 5x5 block has a single cue-colored marker pixel. That pixel is the destination anchor location for that
         block, so different blocks can shift the same global legend rule to different local positions.
      3. The matched motif is bicolored. Its full occupied footprint is projected onto the block, and every occupied
         cell is repainted to the motif's second color instead of being erased to background.
    - Example trace:
      If the legend says "lower_right", a block marker sits at local (3,4), and the motif occupies cells
      {(0,0),(1,1),(2,2),(2,1)}, then motif cell (2,2) must land on (3,4). The placed footprint becomes
      {(1,2),(2,3),(3,4),(3,3)}, and those four cells become the accent color. In the original task the same motif
      would have been centered at rows/cols 1..3 and painted to background, so both the placement and the result color
      are different.
  complexity_calibration:
    - The legend uses exactly three reusable modes, so examples can establish the symbolic vocabulary quickly.
    - The same cue color is reused for both the corner legend and all block markers, which visually groups
      "instructional pixels" without relying on hidden color ids.
    - Motifs are still small 3x3 connected sprites, but the generator forces them to contain the three candidate anchor
      cells (upper-left / center / lower-right). This guarantees every legend mode can visibly matter.
    - Distractors remain present, but the color budget is tightened to at most 7 visible motif colors so the puzzle
      stays readable with the added cue and accent colors.
  implicit_assumption_audit:
    - Old block detection assumed a solid 5x5 monochrome square. That fails because each block now has one foreign
      marker pixel, so the solver must recognize a 24-cell body plus one cue pixel.
    - Old motif lookup assumed motifs were monochrome. That fails because motifs now carry a second accent color.
      The solver therefore scans multicolor components and extracts both the occupied mask and the accent color.
    - Old D8-only logic assumed the motif center was always the anchored point. That fails once the legend can switch
      the anchored motif point to upper-left or lower-right. The solver must decode the legend before using markers.
    - Old D16-only logic assumed the legend directly picked a fixed offset inside each block. In the combo it instead
      picks which motif point attaches to the per-block marker, so legend and marker are not independent changes.
  placement_legality:
    - Large destination and motif regions still use the original separated placement scheme.
    - The 2x2 legend is added only in a corner that neither large region touches.
    - For a chosen legend mode, block markers are sampled only from positions that keep a 3x3 motif fully inside the
      5x5 tile when the decoded motif anchor point is aligned there.
  visual_observability:
    - Across examples, the same corner symbol changes which motif point is treated as the anchor, while block markers
      change where that anchored footprint lands in each tile.
    - The input motifs visibly contain a second color, and the output blocks are recolored to that accent rather than
      opened into background holes, so the D7 change is easy to see.
    - The generator rejects all samples where every block would land on the original centered placement, preventing
      degenerate examples that look like the base task.
  cleanliness:
    - Instructional pixels stay in two clean places only: one 2x2 corner legend and one singleton marker inside each
      block. Motifs remain in their own region, so layers do not overlap.
    - Accent color is reserved for motif interiors and output recoloring only; it is never reused for the legend or
      block markers.
  new_solver_assumptions_and_generator_guarantees:
    - Exactly one corner 2x2 patch encodes a legend mask from the three known codes.
    - Every valid motif component has exactly two non-background colors, one of which is the fixed accent color and
      the other of which matches exactly one destination block color.
    - Every block marker position is legal for the decoded anchor mode, and every motif bbox is exactly 3x3 with the
      three candidate anchor cells occupied.
"""


ACCENT_COLOR = 2
ALIGNMENT_MASKS: dict[str, np.ndarray] = {
    "upper_left": np.array([[1, 0], [0, 0]], dtype=bool),
    "center": np.array([[0, 1], [1, 0]], dtype=bool),
    "lower_right": np.array([[0, 0], [0, 1]], dtype=bool),
}
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
CORNER_SLICES = (
    (slice(0, 2), slice(0, 2)),
    (slice(0, 2), slice(-2, None)),
    (slice(-2, None), slice(0, 2)),
    (slice(-2, None), slice(-2, None)),
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


def _decode_alignment(grid: np.ndarray, bg: int) -> str:
    """Decode the legend mask from the unique non-background 2x2 corner patch."""
    for rs, cs in CORNER_SLICES:
        patch = np.array(grid[rs, cs], dtype=int)
        mask = patch != int(bg)
        if not np.any(mask):
            continue
        for mode, code in ALIGNMENT_MASKS.items():
            if np.array_equal(mask, code):
                return mode
    raise AssertionError("missing corner legend")


def _motif_anchor(mode: str, h: int, w: int) -> tuple[int, int]:
    """Return which motif cell is treated as the anchor for the decoded mode."""
    if mode == "upper_left":
        return 0, 0
    if mode == "center":
        return h // 2, w // 2
    if mode == "lower_right":
        return h - 1, w - 1
    raise ValueError(f"unknown mode: {mode}")


def _valid_marker_positions(mode: str) -> set[tuple[int, int]]:
    """Allowed marker positions that keep a 3x3 motif inside a 5x5 block for the given mode."""
    anchor_r, anchor_c = _motif_anchor(mode, 3, 3)
    rows = range(anchor_r, anchor_r + 3)
    cols = range(anchor_c, anchor_c + 3)
    return {(int(r), int(c)) for r in rows for c in cols}


def _extract_block(comp: dict, grid: np.ndarray, bg: int, mode: str) -> dict[str, int] | None:
    """Recognize a 5x5 destination block with one cue-colored anchor marker."""
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
    if int(marker_col) in {int(bg), int(col), int(ACCENT_COLOR)}:
        return None
    if (int(ar), int(ac)) not in _valid_marker_positions(mode):
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


def _solve(grid: np.ndarray) -> np.ndarray:
    """Shared combo solver logic."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    alignment = _decode_alignment(g, bg)

    # STEP 1) Identify destination blocks: 5x5 almost-monochrome squares with one legal anchor marker.
    blocks: list[dict[str, int]] = []
    mono_comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in mono_comps:
        block = _extract_block(comp, g, bg, alignment)
        if block is not None:
            blocks.append(block)

    assert blocks

    # STEP 2) Collect bicolored motifs:
    # each valid motif is a small multicolor component with one block color and the shared accent color.
    motifs: dict[int, tuple[np.ndarray, int]] = {}
    block_colors = {int(b["col"]) for b in blocks}
    candidates_by_color: dict[int, list[tuple[int, int, int, int, np.ndarray, int]]] = {
        color: [] for color in block_colors
    }
    multi_comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in multi_comps:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue

        r0, c0, r1, c1 = map(int, b)
        patch = crop_bbox(g, b)
        rel_mask = patch != int(bg)
        if not rel_mask.any():
            continue

        colors = {int(v) for v in patch[rel_mask]}
        matching = colors & block_colors
        if len(colors) != 2 or len(matching) != 1 or int(ACCENT_COLOR) not in colors:
            continue

        primary = int(next(iter(matching)))
        accent = int(next(iter(colors - {primary})))
        if accent != int(ACCENT_COLOR):
            continue

        mh, mw = rel_mask.shape
        anchor_r, anchor_c = _motif_anchor(alignment, mh, mw)
        if not bool(rel_mask[int(anchor_r), int(anchor_c)]):
            continue
        candidates_by_color[primary].append(
            (int(comp["area"]), int(mh * mw), int(r0), int(c0), rel_mask.copy(), int(accent))
        )

    for color, items in candidates_by_color.items():
        if not items:
            continue
        items.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        _area, _bbox_area, _r0, _c0, mask, accent = items[0]
        motifs[int(color)] = (mask, int(accent))

    # STEP 3) Recolor each 5x5 tile:
    # align the legend-chosen motif anchor point to the block's marker pixel, then repaint the whole footprint.
    recolored: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motif = motifs.get(int(col))
        if motif is not None:
            mm, accent = motif
            mh, mw = map(int, mm.shape)
            anchor_r, anchor_c = _motif_anchor(alignment, mh, mw)
            off_r = int(b["anchor_r"]) - int(anchor_r)
            off_c = int(b["anchor_c"]) - int(anchor_c)
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(accent)
        recolored[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack the recolored 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = recolored[(int(r0), int(c0))]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor color-matched motifs onto marked 5x5 blocks, with a corner legend choosing which motif point anchors to each block marker.

    Input structure:
    - The background is the most common color.
    - Several destination blocks are 5x5 almost-monochrome squares: 24 cells of one color plus 1 cue-colored marker pixel.
    - Elsewhere, each useful motif is a small connected 3x3-bbox sprite containing the block color and one shared accent color.
    - A 2x2 legend sits in one corner and encodes whether the motif's upper-left cell, center cell, or lower-right cell should be treated as the anchor.

    Rule:
    - Decode the corner legend first.
    - Match each bicolored motif to the block with the same main color, then align the legend-selected motif anchor point to that block's marker pixel.
    - Repaint every occupied motif cell to the motif's accent color on the 5x5 block, and finally pack all recolored blocks in original row/column order with 1-cell gaps and a 1-cell border.

    Key insight:
    - The legend and the in-block marker work together: the legend says which motif point matters, the marker says where that point goes, and the motif's second color says what color the whole stamped footprint becomes.
    """
    return _solve(grid)


# --------------- generator (combo o5x) ---------------

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

# Motif slot geometry and shape constraints.
MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 7)
MAX_EMPTY_SLOTS = 2
ACCENT_CELL_RANGE = (1, 2)
REQUIRED_ANCHOR_CELLS = ((0, 0), (1, 1), (2, 2))

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
LEGEND_SIZE = 2

# Generation retry budget.
GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not all(bool(m[int(r), int(c)]) for r, c in REQUIRED_ANCHOR_CELLS):
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
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for filled in range(min_filled, max_filled + 1):
                distractors = filled - count
                if not (0 <= distractors <= count):
                    continue
                if filled + 1 > COLOR_COUNT - 2:
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


def _random_accent_mask(mask: np.ndarray) -> np.ndarray:
    """Pick 1-2 occupied cells to visibly carry the accent color in the input motif."""
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    max_cells = min(ACCENT_CELL_RANGE[1], len(coords) - 1)
    min_cells = min(ACCENT_CELL_RANGE[0], max_cells)
    count = random.randint(min_cells, max_cells)
    chosen = set(random.sample(coords, count))
    accent = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        accent[int(r), int(c)] = True
    return accent


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int], int]:
    """Pick block colors, distractor colors, and one cue color, reserving the accent color."""
    palette = [c for c in range(COLOR_COUNT) if c not in {bg, ACCENT_COLOR}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count:filled]
    cue_color = int(palette[filled])
    return block_colors, distractor_colors, cue_color


def _paint_motif(
    grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray, accent_mask: np.ndarray
) -> None:
    """Paint one bicolored 3x3 motif into the input grid."""
    primary_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    accent_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(ACCENT_COLOR), dtype=int)
    paste(grid, primary_patch, r0, c0, mask=mask)
    paste(grid, accent_patch, r0, c0, mask=accent_mask)


def _anchor_choices_for_mode(mode: str) -> list[tuple[int, int]]:
    """All legal marker positions for the decoded legend mode."""
    return sorted(_valid_marker_positions(mode))


def _original_equivalent_anchor(mode: str) -> tuple[int, int]:
    """Marker position that reproduces the base task's centered top-left offset (1,1)."""
    anchor_r, anchor_c = _motif_anchor(mode, 3, 3)
    return 1 + int(anchor_r), 1 + int(anchor_c)


def _choose_anchor_positions(count: int, mode: str) -> list[tuple[int, int]]:
    """Sample diverse per-block markers so the anchor rule is visibly nontrivial."""
    choices = _anchor_choices_for_mode(mode)
    baseline = _original_equivalent_anchor(mode)
    min_unique = min(3, count)
    min_shifted = min(2, count)
    while True:
        anchors = [random.choice(choices) for _ in range(count)]
        if sum(anchor != baseline for anchor in anchors) < min_shifted:
            continue
        if len(set(anchors)) < min_unique:
            continue
        return anchors


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
    alignment: str,
) -> np.ndarray:
    """Render the oracle output for the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    anchor_r, anchor_c = _motif_anchor(alignment, MOTIF_SIZE, MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            block_anchor_r, block_anchor_c = anchors[idx]
            off_r = int(block_anchor_r) - int(anchor_r)
            off_c = int(block_anchor_c) - int(anchor_c)
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            for rr, cc in np.argwhere(motif_masks[color]):
                tile[int(off_r + int(rr)), int(off_c + int(cc))] = int(ACCENT_COLOR)
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
    """Paint 5x5 destination blocks with one cue-colored anchor marker each."""
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
    accent_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Build shuffled motif payloads for the occupied motif slots."""
    items: list[tuple[int, np.ndarray, np.ndarray]] = [
        (int(color), motif_masks[color], accent_masks[color]) for color in block_colors
    ]
    for color in distractor_colors:
        mask = _random_mask()
        items.append((int(color), mask, _random_accent_mask(mask)))
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
    accent_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, motif_masks, accent_masks)
    for (sr, sc), (color, mask, accent_mask) in zip(slots, slot_items):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask, accent_mask)


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
    raise ValueError(f"unknown legend corner: {corner}")


def _paint_legend(grid: np.ndarray, bg: int, cue_color: int, corner: str, alignment: str) -> None:
    """Paint the 2x2 legend into one corner of the input grid."""
    r0, c0 = _corner_origin(*grid.shape, corner)
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(bg), dtype=int)
    patch[ALIGNMENT_MASKS[alignment]] = int(cue_color)
    grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = patch


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two inclusive rectangles overlap."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend_bbox(H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    """Inclusive bbox occupied by the 2x2 legend."""
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
    """Reject placements where either large region touches any corner legend patch."""
    dst_r, dst_c, motif_r, motif_c = placement
    dst_box = (dst_r, dst_c, dst_r + dst_h - 1, dst_c + dst_w - 1)
    motif_box = (motif_r, motif_c, motif_r + motif_h - 1, motif_c + motif_w - 1)
    corner_boxes = [_legend_bbox(H, W, corner) for corner in LEGEND_CORNERS]
    return all(
        not _rects_overlap(dst_box, corner_box) and not _rects_overlap(motif_box, corner_box)
        for corner_box in corner_boxes
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
    cue_color: int,
    motif_masks: dict[int, np.ndarray],
    accent_masks: dict[int, np.ndarray],
    anchors: list[tuple[int, int]],
    placement: tuple[int, int, int, int],
    legend_corner: str,
    alignment: str,
) -> np.ndarray:
    """Render the sampled destination blocks, motif slots, and corner legend into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors, anchors, cue_color)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_masks,
        accent_masks,
    )
    _paint_legend(inp, bg, cue_color, legend_corner, alignment)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg_choices = [c for c in range(COLOR_COUNT) if c != ACCENT_COLOR]
    bg = random.choice(bg_choices)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

    legend_corner = random.choice(LEGEND_CORNERS)
    alignment = random.choice(tuple(ALIGNMENT_MASKS))
    placement = None
    for _ in range(30):
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
    motif_masks = {color: _random_mask() for color in block_colors}
    accent_masks = {color: _random_accent_mask(motif_masks[color]) for color in block_colors}
    anchors = _choose_anchor_positions(count, alignment)

    expected = _render_expected(bg, R, C, block_colors, motif_masks, anchors, alignment)
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
        accent_masks,
        anchors,
        placement,
        legend_corner,
        alignment,
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

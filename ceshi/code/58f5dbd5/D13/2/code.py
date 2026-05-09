# Variant of 58f5dbd5
# Dimension: D13 — 方块数量 (meta)
# Change: 3-6个方块，布局为规则的RxC网格... → 方块数量由符号编码决定

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)

# Design analysis for the D13 variant.
#
# a) Concrete behavior and trace:
# - The input still has solid 5x5 destination blocks and same-colored 3x3 motifs.
# - The new visible element is a small border legend: a row or column of separated 2x2
#   solid markers placed flush against one outer edge. The marker count is 3, 4, or 6.
# - All destination blocks now exist inside a larger full rectangular grid, and every
#   block still has a matching motif. Some of those blocks are distractors.
# - The legend count says how many blocks survive. To remove ambiguity, the legend
#   orientation also matters:
#   * horizontal legend (top/bottom edge): keep all block rows but only the leftmost
#     needed columns, so the kept set is the top-left sub-rectangle spanning full height.
#   * vertical legend (left/right edge): keep all block columns but only the topmost
#     needed rows, so the kept set is the top-left sub-rectangle spanning full width.
# - Example trace: suppose the block grid is 2x4, the legend is horizontal, and it has
#   6 markers. Old logic would carve and pack all 8 blocks. New logic decodes 6, notes
#   the horizontal legend, keeps the leftmost 3 columns across both rows (a 2x3 prefix),
#   ignores the last column entirely, then performs the original carving/pacing rule on
#   only those 6 blocks.
#
# b) Complexity calibration:
# - The legend only uses three counts (3, 4, 6), matching the original observed output
#   cardinalities, so the branching factor stays low.
# - Distractor blocks are real blocks with real motifs, so the legend is necessary.
# - The kept subset is always a clean top-left rectangle; this avoids hard-to-see
#   scattered selection patterns while still making the new step observable.
#
# c) Old solver assumptions that break:
# - "all detected 5x5 blocks are outputs" is no longer valid.
# - "the set of unique block rows/cols directly defines output size" is only true after
#   trimming by the legend.
# - "the smallest same-colored non-5x5 component is the motif" remains valid, but the
#   solver must detect the legend first so border markers are not mistaken for semantics.
#
# d) Placement legality:
# - The generator reserves a 2-cell legend band on one edge and places the block region
#   and motif region inside the remaining rectangle with no overlap and at least one-axis
#   separation, preserving the original spatial organization.
# - The block grid is fully populated, regular, and uses the same 6-cell top-left step.
# - Motif slots remain regular 3x3 slots with 4-cell step; 0-2 empty motif slots are
#   still allowed so the motif area does not become too rigid.
#
# e) Visual observability:
# - From a few I/O pairs, a human can see that the output contains only a top-left chunk
#   of the larger block array, and the number of legend markers matches the number of
#   packed tiles in that chunk.
# - The legend does not rely on color IDs for ordering. It is counted visually.
#
# f) Visual cleanliness:
# - The legend is isolated in its own edge band and uses repeated identical 2x2 markers.
# - Blocks, motifs, and the legend occupy different scales (5x5, 3x3, 2x2), so they are
#   easy to distinguish.
#
# g) Color inference:
# - Colors keep their original role: same color means block↔motif match. The legend does
#   not encode a color mapping; it only encodes quantity/orientation.
#
# New solver-side assumptions introduced here, and how generation guarantees them:
# - There is exactly one legend color whose border-touching solid 2x2 components form the
#   instruction strip. The generator reserves that strip explicitly.
# - The decoded count is compatible with the full block grid and the legend orientation:
#   horizontal legends always satisfy count % total_rows == 0, vertical legends satisfy
#   count % total_cols == 0. Config sampling enforces this.
# - The kept subset is always the top-left rectangle implied by that rule, so packing its
#   unique row/column coordinates reproduces the intended compact output.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given coords (in-bounds only)."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _solid_bbox_component(grid: np.ndarray, comp: dict, h: int, w: int) -> bool:
    """Return whether a component is exactly a solid h x w rectangle in its own color."""
    b = comp["bbox"]
    if bbox_h(b) != int(h) or bbox_w(b) != int(w):
        return False
    patch = crop_bbox(grid, b)
    return bool((patch == int(comp["color"])).all())


def _decode_legend(grid: np.ndarray, bg: int) -> tuple[int, str, set[tuple[int, int]]]:
    """Decode the border legend as (marker_count, orientation, occupied_coords)."""
    H, W = grid.shape
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    by_color: dict[int, list[tuple[dict, set[str]]]] = {}
    for comp in comps:
        if not _solid_bbox_component(grid, comp, 2, 2):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        edges: set[str] = set()
        if r0 == 0:
            edges.add("top")
        if r1 == H - 1:
            edges.add("bottom")
        if c0 == 0:
            edges.add("left")
        if c1 == W - 1:
            edges.add("right")
        if not edges:
            continue
        by_color.setdefault(int(comp["color"]), []).append((comp, edges))

    candidates: list[tuple[int, str, list[dict]]] = []
    for color, items in by_color.items():
        del color
        comps_only = [item[0] for item in items]
        count = len(comps_only)
        if count not in {3, 4, 6}:
            continue

        shared_vertical = None
        if all("left" in item[1] for item in items):
            shared_vertical = "left"
        elif all("right" in item[1] for item in items):
            shared_vertical = "right"

        shared_horizontal = None
        if all("top" in item[1] for item in items):
            shared_horizontal = "top"
        elif all("bottom" in item[1] for item in items):
            shared_horizontal = "bottom"

        if shared_horizontal is not None and shared_vertical is None:
            candidates.append((count, "horizontal", comps_only))
        elif shared_vertical is not None and shared_horizontal is None:
            candidates.append((count, "vertical", comps_only))

    assert candidates
    candidates.sort(key=lambda t: (-t[0], 0 if t[1] == "horizontal" else 1))
    count, orientation, legend_comps = candidates[0]

    legend_cells: set[tuple[int, int]] = set()
    for comp in legend_comps:
        for r, c in comp["coords"]:
            legend_cells.add((int(r), int(c)))
    return int(count), orientation, legend_cells


def _select_active_blocks(
    blocks: list[dict[str, int]], active_count: int, legend_orientation: str
) -> list[dict[str, int]]:
    """Trim the full block grid to the top-left active sub-rectangle implied by the legend."""
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    assert len(blocks) == len(rs) * len(cs)

    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    if legend_orientation == "horizontal":
        assert active_count % len(rs) == 0
        active_cols = active_count // len(rs)
        assert 1 <= active_cols <= len(cs)
        active = [
            b for b in blocks if int(c_index[int(b["c0"])]) < int(active_cols)
        ]
    else:
        assert legend_orientation == "vertical"
        assert active_count % len(cs) == 0
        active_rows = active_count // len(cs)
        assert 1 <= active_rows <= len(rs)
        active = [
            b for b in blocks if int(r_index[int(b["r0"])]) < int(active_rows)
        ]

    active.sort(key=lambda b: (int(b["r0"]), int(b["c0"])))
    assert len(active) == int(active_count)
    return active


def _motif_for_color(
    grid: np.ndarray, bg: int, color: int, legend_cells: set[tuple[int, int]]
) -> np.ndarray | None:
    """Return the smallest same-colored non-block component as a local boolean mask."""
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    comps_col = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    for comp in comps_col:
        if all((int(r), int(c)) in legend_cells for r, c in comp["coords"]):
            continue
        if _solid_bbox_component(grid, comp, 5, 5):
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = mask_from_coords(rel_coords, (h, w))
        candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return candidates[0][4]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a border legend, keep only that many blocks, then carve them with color-matched motifs.

    Input structure:
    - The background is still the most common color.
    - One region contains a full regular grid of solid 5x5 monochrome blocks, each with a unique color.
    - A separate region contains one same-colored motif for every block, and an edge legend contains 3/4/6 identical 2x2 markers.

    Rule:
    - Count the legend markers to determine how many blocks survive.
    - Use the legend orientation to choose a top-left sub-rectangle of the full block grid: a horizontal legend keeps all rows and trims columns; a vertical legend keeps all columns and trims rows.
    - For only those surviving blocks, find the same-colored motif, center it in the 5x5 tile, carve those cells to background, and pack the carved tiles with 1-cell gaps and a 1-cell border.

    Key insight:
    - The motifs still define the hole pattern, but the legend now decides which subset of the larger block array is even eligible to be carved and emitted.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    active_count, legend_orientation, legend_cells = _decode_legend(g, bg)

    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        if _solid_bbox_component(g, comp, 5, 5):
            r0, c0, _r1, _c1 = map(int, comp["bbox"])
            blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert blocks
    blocks.sort(key=lambda b: (int(b["r0"]), int(b["c0"])))
    active_blocks = _select_active_blocks(blocks, active_count, legend_orientation)

    motifs: dict[int, np.ndarray] = {}
    for color in {int(b["col"]) for b in active_blocks}:
        mm = _motif_for_color(g, bg, color, legend_cells)
        if mm is not None:
            motifs[int(color)] = mm

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in active_blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        mm = motifs.get(int(col))
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

    rs = sorted({int(b["r0"]) for b in active_blocks})
    cs = sorted({int(b["c0"]) for b in active_blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in active_blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        rr = int(1 + int(r_index[int(r0)]) * step)
        cc = int(1 + int(c_index[int(c0)]) * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
MAX_EMPTY_SLOTS = 2

OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1
LEGEND_MARKER_SIZE = 2
LEGEND_MARKER_STEP = 3
LEGEND_BAND = 2
REGION_GAP = 1

GENERATION_RETRIES = 400

BLOCK_CONFIGS = [
    {"total": (1, 4), "count": 3, "legend": "horizontal"},
    {"total": (2, 3), "count": 3, "legend": "vertical"},
    {"total": (3, 2), "count": 3, "legend": "horizontal"},
    {"total": (2, 3), "count": 4, "legend": "horizontal"},
    {"total": (3, 2), "count": 4, "legend": "vertical"},
    {"total": (2, 4), "count": 4, "legend": "horizontal"},
    {"total": (2, 4), "count": 6, "legend": "horizontal"},
]


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


def _choose_active_shape(total_R: int, total_C: int, count: int, legend: str) -> tuple[int, int]:
    """Return the selected top-left active rectangle implied by a config."""
    if legend == "horizontal":
        assert count % total_R == 0
        return int(total_R), int(count // total_R)
    assert legend == "vertical"
    assert count % total_C == 0
    return int(count // total_C), int(total_C)


def _choose_palette(bg: int, total_count: int) -> tuple[list[int], int]:
    """Pick unique block colors and a distinct legend color."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:total_count]
    legend_color = palette[total_count]
    return block_colors, int(legend_color)


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    """Return whether the two regions can be separated by one axis."""
    return (
        dh + mh + REGION_GAP <= H and max(dw, mw) <= W
    ) or (
        dw + mw + REGION_GAP <= W and max(dh, mh) <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible orderings for separated destination and motif regions."""
    schemes: list[str] = []
    if dh + mh + REGION_GAP <= H and max(dw, mw) <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + REGION_GAP <= W and max(dh, mh) <= H:
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

    top_r = random.randint(0, H - top_h - bot_h - REGION_GAP)
    bot_r = random.randint(top_r + top_h + REGION_GAP, H - bot_h)
    top_c = random.randint(0, W - top_w)
    bot_c = random.randint(0, W - bot_w)
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

    left_c = random.randint(0, W - left_w - right_w - REGION_GAP)
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w)
    left_r = random.randint(0, H - left_h)
    right_r = random.randint(0, H - right_h)
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


def _config_candidates(H: int, W: int) -> list[tuple[dict, int, int]]:
    """Return feasible (block_config, motif_rows, motif_cols) tuples for this canvas."""
    out: list[tuple[dict, int, int]] = []
    for cfg in BLOCK_CONFIGS:
        total_R, total_C = cfg["total"]
        legend = cfg["legend"]
        total_count = int(total_R * total_C)
        dh = _spanned_size(total_R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
        dw = _spanned_size(total_C, DST_BLOCK_SIZE, DST_BLOCK_STEP)

        inner_H = H - LEGEND_BAND if legend == "horizontal" else H
        inner_W = W if legend == "horizontal" else W - LEGEND_BAND
        if inner_H <= 0 or inner_W <= 0:
            continue

        for capacity, layouts in MOTIF_LAYOUTS.items():
            if not (total_count <= capacity <= total_count + MAX_EMPTY_SLOTS):
                continue
            for Mr, Mc in layouts:
                mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                if _can_split(inner_H, inner_W, dh, dw, mh, mw):
                    out.append((cfg, int(Mr), int(Mc)))
    return out


def _select_active_colors(
    block_colors: list[int], total_R: int, total_C: int, active_R: int, active_C: int
) -> list[int]:
    """Take the top-left active rectangle from the full row-major block color list."""
    out: list[int] = []
    for i in range(active_R):
        row_start = i * total_C
        out.extend(int(c) for c in block_colors[row_start : row_start + active_C])
    assert len(out) == int(active_R * active_C)
    return out


def _render_expected(
    bg: int, active_R: int, active_C: int, active_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output for the selected active rectangle."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(active_R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(active_C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(active_R):
        for j in range(active_C):
            color = int(active_colors[i * active_C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            tile[motif_slice, motif_slice][motif_masks[color]] = bg
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, total_R: int, total_C: int, block_colors: list[int]
) -> None:
    """Paint the full destination block grid."""
    for i in range(total_R):
        for j in range(total_C):
            color = int(block_colors[i * total_C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into the grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign all block motifs to the visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    for (sr, sc), color in zip(slots, block_colors):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            int(color),
            motif_masks[int(color)],
        )


def _legend_span(count: int) -> int:
    """Return the horizontal/vertical span occupied by count legend markers."""
    return _spanned_size(count, LEGEND_MARKER_SIZE, LEGEND_MARKER_STEP)


def _paint_legend(
    grid: np.ndarray, legend_color: int, count: int, legend: str
) -> None:
    """Paint the border legend as separated 2x2 markers."""
    H, W = grid.shape
    patch = np.full((LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE), int(legend_color), dtype=int)
    span = _legend_span(count)

    if legend == "horizontal":
        side = random.choice(["top", "bottom"])
        r0 = 0 if side == "top" else H - LEGEND_BAND
        c0 = random.randint(0, W - span)
        for k in range(count):
            paste(grid, patch, r0, c0 + k * LEGEND_MARKER_STEP)
        return

    side = random.choice(["left", "right"])
    c0 = 0 if side == "left" else W - LEGEND_BAND
    r0 = random.randint(0, H - span)
    for k in range(count):
        paste(grid, patch, r0 + k * LEGEND_MARKER_STEP, c0)


def _render_input(
    H: int,
    W: int,
    bg: int,
    cfg: dict,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    legend_color: int,
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render legend, destination blocks, and all motifs into the sampled canvas."""
    total_R, total_C = cfg["total"]
    legend = cfg["legend"]

    dst_r, dst_c, motif_r, motif_c = placement
    if legend == "horizontal":
        inner_row_offset = LEGEND_BAND if random.choice([False, True]) else 0
        if inner_row_offset == LEGEND_BAND:
            legend_side = "top"
        else:
            legend_side = "bottom"
    else:
        inner_col_offset = LEGEND_BAND if random.choice([False, True]) else 0
        if inner_col_offset == LEGEND_BAND:
            legend_side = "left"
        else:
            legend_side = "right"

    inp = np.full((H, W), bg, dtype=int)

    if legend == "horizontal":
        if legend_side == "top":
            dst_r += LEGEND_BAND
            motif_r += LEGEND_BAND
            _paint_destination_blocks(inp, dst_r, dst_c, total_R, total_C, block_colors)
            _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_masks)
            patch = np.full((LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE), int(legend_color), dtype=int)
            span = _legend_span(cfg["count"])
            c0 = random.randint(0, W - span)
            for k in range(cfg["count"]):
                paste(inp, patch, 0, c0 + k * LEGEND_MARKER_STEP)
        else:
            _paint_destination_blocks(inp, dst_r, dst_c, total_R, total_C, block_colors)
            _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_masks)
            patch = np.full((LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE), int(legend_color), dtype=int)
            span = _legend_span(cfg["count"])
            c0 = random.randint(0, W - span)
            for k in range(cfg["count"]):
                paste(inp, patch, H - LEGEND_BAND, c0 + k * LEGEND_MARKER_STEP)
    else:
        if legend_side == "left":
            dst_c += LEGEND_BAND
            motif_c += LEGEND_BAND
            _paint_destination_blocks(inp, dst_r, dst_c, total_R, total_C, block_colors)
            _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_masks)
            patch = np.full((LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE), int(legend_color), dtype=int)
            span = _legend_span(cfg["count"])
            r0 = random.randint(0, H - span)
            for k in range(cfg["count"]):
                paste(inp, patch, r0 + k * LEGEND_MARKER_STEP, 0)
        else:
            _paint_destination_blocks(inp, dst_r, dst_c, total_R, total_C, block_colors)
            _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_masks)
            patch = np.full((LEGEND_MARKER_SIZE, LEGEND_MARKER_SIZE), int(legend_color), dtype=int)
            span = _legend_span(cfg["count"])
            r0 = random.randint(0, H - span)
            for k in range(cfg["count"]):
                paste(inp, patch, r0 + k * LEGEND_MARKER_STEP, W - LEGEND_BAND)

    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    target_count = random.choice([3, 4, 6])
    for _ in range(20):
        H = random.randint(*CANVAS_HEIGHT_RANGE)
        W = random.randint(*CANVAS_WIDTH_RANGE)
        bg = random.randrange(COLOR_COUNT)

        configs = [
            item for item in _config_candidates(H, W) if int(item[0]["count"]) == int(target_count)
        ]
        if not configs:
            continue

        cfg, Mr, Mc = random.choice(configs)
        total_R, total_C = cfg["total"]
        total_count = int(total_R * total_C)
        active_R, active_C = _choose_active_shape(total_R, total_C, int(cfg["count"]), str(cfg["legend"]))

        dh = _spanned_size(total_R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
        dw = _spanned_size(total_C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

        inner_H = H - LEGEND_BAND if cfg["legend"] == "horizontal" else H
        inner_W = W if cfg["legend"] == "horizontal" else W - LEGEND_BAND
        placement = _place_regions(inner_H, inner_W, dh, dw, mh, mw)
        if placement is None:
            continue

        block_colors, legend_color = _choose_palette(bg, total_count)
        motif_masks = {int(color): _random_mask() for color in block_colors}
        active_colors = _select_active_colors(block_colors, total_R, total_C, active_R, active_C)

        expected = _render_expected(bg, active_R, active_C, active_colors, motif_masks)
        inp = _render_input(
            H,
            W,
            bg,
            cfg,
            Mr,
            Mc,
            block_colors,
            legend_color,
            motif_masks,
            placement,
        )

        if np.array_equal(main(inp), expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D13 input")

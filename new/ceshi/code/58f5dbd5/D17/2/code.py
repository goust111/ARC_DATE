# Variant of 58f5dbd5
# Dimension: D17 — 方块与motif的空间分离方式 (meta)
# Change: 方块区域与motif区域沿一个轴（行或列方向）分离，互不重叠... → 四象限分离

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D17 variant:
# a) Concrete behavior:
#    The carved-block rule itself is unchanged, but the spatial cue is harder.
#    Blocks no longer occupy one contiguous source region and motifs no longer occupy
#    one opposing region. Instead, the logical block grid is cut into two chunks and
#    those chunks are placed in diagonal quadrants; the motif grid is cut the same way
#    and placed in the opposite diagonal pair. A row-split example is:
#    - original: one 2x2 block region on the left, one motif region on the right
#    - variant: top logical block row in TL, bottom logical block row in BR,
#      top motif rows in TR, bottom motif rows in BL.
#    The carving step is identical, but the packing step must now reconstruct the
#    hidden row/column order from disjoint chunks before compacting the output.
# b) Complexity calibration:
#    Only the placement topology changes. Colors still give one-to-one matching, motif
#    masks stay small and connected, and distractors remain unique-colored. This keeps
#    the core task legible while making the quadrant split visually obvious.
# c) Implicit assumptions review:
#    - Block detection by "solid 5x5 component" still holds; diagonal placement does
#      not change object shape.
#    - Motif lookup by same color still holds because matching colors remain unique.
#    - The old packer assumed a full Cartesian product of global row/col coordinates.
#      That assumption is false here because a row-split yields extra absolute columns
#      and a col-split yields extra absolute rows. The new solver therefore detects
#      whether rows or columns form the stable logical grouping.
# d) Placement legality:
#    Each quadrant receives exactly one non-overlapping chunk. Split indices are chosen
#    so both chunks are non-empty, and a sampled configuration is accepted only if all
#    four quadrant rectangles can contain their assigned chunk sizes inside the original
#    18-22 by 18-24 canvas bounds.
# e) Visual observability:
#    The diagonal occupancy is directly visible: blocks appear in one diagonal pair and
#    motifs in the other. The solver never sorts by color id; only visible position and
#    same-color matching matter.
# f) Visual cleanliness:
#    Objects stay monochrome, use the original fixed spacing, and never overlap across
#    quadrants. Each quadrant contains one clean mini-region, so the input remains easy
#    to parse despite the more global arrangement.
# g) Color inference:
#    The task still uses a small palette with one block color per matching motif and a
#    few distractors. No hidden color remapping is introduced.


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


def _pack_by_rows_or_cols(
    blocks: list[dict[str, int]],
    carved: dict[tuple[int, int], np.ndarray],
    bg: int,
) -> np.ndarray:
    """Recover the hidden logical grid from a row-split or col-split arrangement."""
    row_map: dict[int, list[dict[str, int]]] = {}
    col_map: dict[int, list[dict[str, int]]] = {}
    for block in blocks:
        row_map.setdefault(int(block["r0"]), []).append(block)
        col_map.setdefault(int(block["c0"]), []).append(block)

    row_groups = [sorted(group, key=lambda b: int(b["c0"])) for _r, group in sorted(row_map.items())]
    col_groups = [sorted(group, key=lambda b: int(b["r0"])) for _c, group in sorted(col_map.items())]

    row_lengths = {len(group) for group in row_groups}
    col_heights = {len(group) for group in col_groups}

    if len(row_lengths) == 1:
        R = len(row_groups)
        C = next(iter(row_lengths))
        out_h = int(1 + R * 5 + (R - 1) + 1)
        out_w = int(1 + C * 5 + (C - 1) + 1)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        for i, group in enumerate(row_groups):
            for j, block in enumerate(group):
                rr = int(1 + i * 6)
                cc = int(1 + j * 6)
                out[rr : rr + 5, cc : cc + 5] = carved[(int(block["r0"]), int(block["c0"]))]
        return out

    if len(col_heights) == 1:
        R = next(iter(col_heights))
        C = len(col_groups)
        out_h = int(1 + R * 5 + (R - 1) + 1)
        out_w = int(1 + C * 5 + (C - 1) + 1)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        for j, group in enumerate(col_groups):
            for i, block in enumerate(group):
                rr = int(1 + i * 6)
                cc = int(1 + j * 6)
                out[rr : rr + 5, cc : cc + 5] = carved[(int(block["r0"]), int(block["c0"]))]
        return out

    raise AssertionError("blocks do not form a recoverable row-split or col-split grid")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then repack the decoded block grid.

    Input structure:
    - The background is the majority color.
    - Solid 5x5 monochrome blocks occupy two diagonally opposite quadrants.
    - Same-colored motif sprites, plus optional distractor motifs, occupy the other diagonal pair.

    Rule:
    - Match each motif to the solid 5x5 block of the same color.
    - Center that motif inside the block and turn the motif cells into background holes.
    - Recover the hidden logical row/column order of the split block grid, then pack the carved blocks tightly with 1-cell gaps and a 1-cell border.

    Key insight:
    - The motif is still a carving stencil, but the real challenge is reconstructing the original block ordering from a diagonal four-quadrant layout.
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
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is NOT one of the 5x5 blocks.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(block["col"]) for block in blocks}
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
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each 5x5 tile:
    # center the motif mask within the 5x5 and repaint those mask cells to bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        col = int(block["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motif_mask = motifs.get(int(col))
        if motif_mask is not None:
            mh, mw = map(int, motif_mask.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(motif_mask):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(bg)
        carved[(int(block["r0"]), int(block["c0"]))] = tile

    # STEP 4) Pack carved blocks by recovering the logical grid from split rows or columns.
    return _pack_by_rows_or_cols(blocks, carved, bg)


# --------------- generator (o5x) ---------------

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
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

# Region placement and packed output geometry.
OUTER_MARGIN = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

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
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled logical block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
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
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _axis_choices(R: int, C: int, Mr: int, Mc: int) -> list[str]:
    """Return split axes that leave both block and motif chunks non-empty."""
    axes: list[str] = []
    if R >= 2 and Mr >= 2:
        axes.append("row")
    if C >= 2 and Mc >= 2:
        axes.append("col")
    return axes


def _quadrant_requirements(
    axis: str,
    diag: str,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    dst_split: int,
    motif_split: int,
) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Describe each quadrant's chunk size and which quadrant each logical chunk uses."""
    if axis == "row":
        dst_first = (_spanned_size(dst_split, DST_BLOCK_SIZE, DST_BLOCK_STEP), _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP))
        dst_second = (_spanned_size(R - dst_split, DST_BLOCK_SIZE, DST_BLOCK_STEP), _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP))
        motif_first = (_spanned_size(motif_split, MOTIF_SIZE, MOTIF_STEP), _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP))
        motif_second = (_spanned_size(Mr - motif_split, MOTIF_SIZE, MOTIF_STEP), _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP))
        if diag == "main":
            reqs = {"TL": dst_first, "TR": motif_first, "BL": motif_second, "BR": dst_second}
            mapping = {"dst_first": "TL", "dst_second": "BR", "motif_first": "TR", "motif_second": "BL"}
        else:
            reqs = {"TL": motif_first, "TR": dst_first, "BL": dst_second, "BR": motif_second}
            mapping = {"dst_first": "TR", "dst_second": "BL", "motif_first": "TL", "motif_second": "BR"}
        return reqs, mapping

    dst_first = (_spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP), _spanned_size(dst_split, DST_BLOCK_SIZE, DST_BLOCK_STEP))
    dst_second = (_spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP), _spanned_size(C - dst_split, DST_BLOCK_SIZE, DST_BLOCK_STEP))
    motif_first = (_spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP), _spanned_size(motif_split, MOTIF_SIZE, MOTIF_STEP))
    motif_second = (_spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP), _spanned_size(Mc - motif_split, MOTIF_SIZE, MOTIF_STEP))
    if diag == "main":
        reqs = {"TL": dst_first, "TR": motif_second, "BL": motif_first, "BR": dst_second}
        mapping = {"dst_first": "TL", "dst_second": "BR", "motif_first": "BL", "motif_second": "TR"}
    else:
        reqs = {"TL": motif_first, "TR": dst_second, "BL": dst_first, "BR": motif_second}
        mapping = {"dst_first": "BL", "dst_second": "TR", "motif_first": "TL", "motif_second": "BR"}
    return reqs, mapping


def _quadrants_fit(H: int, W: int, reqs: dict[str, tuple[int, int]]) -> bool:
    """Check whether all four quadrants can host their assigned chunks."""
    need_top = max(reqs["TL"][0], reqs["TR"][0])
    need_bottom = max(reqs["BL"][0], reqs["BR"][0])
    need_left = max(reqs["TL"][1], reqs["BL"][1])
    need_right = max(reqs["TR"][1], reqs["BR"][1])
    return (
        need_top + need_bottom + 2 * OUTER_MARGIN <= H
        and need_left + need_right + 2 * OUTER_MARGIN <= W
    )


def _place_quadrants(
    H: int,
    W: int,
    reqs: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]] | None:
    """Choose concrete origins for one chunk inside each quadrant."""
    if not _quadrants_fit(H, W, reqs):
        return None

    need_top = max(reqs["TL"][0], reqs["TR"][0])
    need_bottom = max(reqs["BL"][0], reqs["BR"][0])
    need_left = max(reqs["TL"][1], reqs["BL"][1])
    need_right = max(reqs["TR"][1], reqs["BR"][1])

    top_cap = random.randint(need_top, H - 2 * OUTER_MARGIN - need_bottom)
    bottom_cap = H - 2 * OUTER_MARGIN - top_cap
    left_cap = random.randint(need_left, W - 2 * OUTER_MARGIN - need_right)
    right_cap = W - 2 * OUTER_MARGIN - left_cap

    bounds = {
        "TL": (OUTER_MARGIN, OUTER_MARGIN, OUTER_MARGIN + top_cap - 1, OUTER_MARGIN + left_cap - 1),
        "TR": (OUTER_MARGIN, W - OUTER_MARGIN - right_cap, OUTER_MARGIN + top_cap - 1, W - OUTER_MARGIN - 1),
        "BL": (H - OUTER_MARGIN - bottom_cap, OUTER_MARGIN, H - OUTER_MARGIN - 1, OUTER_MARGIN + left_cap - 1),
        "BR": (H - OUTER_MARGIN - bottom_cap, W - OUTER_MARGIN - right_cap, H - OUTER_MARGIN - 1, W - OUTER_MARGIN - 1),
    }

    origins: dict[str, tuple[int, int]] = {}
    for quadrant, (h, w) in reqs.items():
        r0, c0, r1, c1 = bounds[quadrant]
        origins[quadrant] = (
            random.randint(int(r0), int(r1 - h + 1)),
            random.randint(int(c0), int(c1 - w + 1)),
        )
    return origins


def _placement_options(H: int, W: int, R: int, C: int, Mr: int, Mc: int) -> list[dict[str, object]]:
    """Enumerate feasible quadrant-split placements for blocks and motifs."""
    options: list[dict[str, object]] = []
    for axis in _axis_choices(R, C, Mr, Mc):
        dst_splits = range(1, R) if axis == "row" else range(1, C)
        motif_splits = range(1, Mr) if axis == "row" else range(1, Mc)
        for dst_split in dst_splits:
            for motif_split in motif_splits:
                for diag in ("main", "anti"):
                    reqs, mapping = _quadrant_requirements(
                        axis, diag, R, C, Mr, Mc, int(dst_split), int(motif_split)
                    )
                    if _quadrants_fit(H, W, reqs):
                        options.append(
                            {
                                "axis": axis,
                                "diag": diag,
                                "dst_split": int(dst_split),
                                "motif_split": int(motif_split),
                                "reqs": reqs,
                                "mapping": mapping,
                            }
                        )
    return options


def _place_regions(H: int, W: int, R: int, C: int, Mr: int, Mc: int) -> dict[str, object] | None:
    """Choose one feasible four-quadrant arrangement and materialize chunk origins."""
    options = _placement_options(H, W, R, C, Mr, Mc)
    if not options:
        return None

    choice = random.choice(options)
    origins = _place_quadrants(H, W, choice["reqs"])
    if origins is None:
        return None

    mapping = choice["mapping"]
    return {
        "axis": choice["axis"],
        "diag": choice["diag"],
        "dst_split": choice["dst_split"],
        "motif_split": choice["motif_split"],
        "dst_first": origins[mapping["dst_first"]],
        "dst_second": origins[mapping["dst_second"]],
        "motif_first": origins[mapping["motif_first"]],
        "motif_second": origins[mapping["motif_second"]],
    }


_FEASIBLE_CANVASES_CACHE: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}


def _feasible_canvas_sizes(R: int, C: int, Mr: int, Mc: int) -> list[tuple[int, int]]:
    """List canvas sizes that can host this logical block/motif configuration."""
    key = (int(R), int(C), int(Mr), int(Mc))
    cached = _FEASIBLE_CANVASES_CACHE.get(key)
    if cached is not None:
        return cached

    sizes: list[tuple[int, int]] = []
    for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1):
        for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1):
            if _placement_options(H, W, R, C, Mr, Mc):
                sizes.append((H, W))
    _FEASIBLE_CANVASES_CACHE[key] = sizes
    return sizes


def _configs() -> list[tuple[int, int, int, int, int]]:
    """Enumerate logical layouts that fit at least one allowed canvas size."""
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
                        if _feasible_canvas_sizes(R, C, Mr, Mc):
                            out.append((R, C, Mr, Mc, filled))
    return out


def _choose_config() -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving the original two-stage count weighting."""
    configs = _configs()
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _paint_destination_blocks(
    grid: np.ndarray,
    R: int,
    C: int,
    block_colors: list[int],
    placement: dict[str, object],
) -> None:
    """Paint the logical block grid into its two diagonal quadrant chunks."""
    axis = str(placement["axis"])
    split = int(placement["dst_split"])
    first_r, first_c = placement["dst_first"]
    second_r, second_c = placement["dst_second"]

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            if axis == "row":
                if i < split:
                    rr = int(first_r + i * DST_BLOCK_STEP)
                    cc = int(first_c + j * DST_BLOCK_STEP)
                else:
                    rr = int(second_r + (i - split) * DST_BLOCK_STEP)
                    cc = int(second_c + j * DST_BLOCK_STEP)
            else:
                if j < split:
                    rr = int(first_r + i * DST_BLOCK_STEP)
                    cc = int(first_c + j * DST_BLOCK_STEP)
                else:
                    rr = int(second_r + i * DST_BLOCK_STEP)
                    cc = int(second_c + (j - split) * DST_BLOCK_STEP)
            paste(grid, block, rr, cc)


def _slot_items(
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [(int(color), motif_masks[color]) for color in block_colors]
    items.extend((int(color), _random_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _motif_slot_positions(Mr: int, Mc: int, placement: dict[str, object]) -> list[tuple[int, int]]:
    """Return concrete top-left coordinates for each logical motif slot."""
    axis = str(placement["axis"])
    split = int(placement["motif_split"])
    first_r, first_c = placement["motif_first"]
    second_r, second_c = placement["motif_second"]

    slots: list[tuple[int, int]] = []
    for i in range(Mr):
        for j in range(Mc):
            if axis == "row":
                if i < split:
                    rr = int(first_r + i * MOTIF_STEP)
                    cc = int(first_c + j * MOTIF_STEP)
                else:
                    rr = int(second_r + (i - split) * MOTIF_STEP)
                    cc = int(second_c + j * MOTIF_STEP)
            else:
                if j < split:
                    rr = int(first_r + i * MOTIF_STEP)
                    cc = int(first_c + j * MOTIF_STEP)
                else:
                    rr = int(second_r + i * MOTIF_STEP)
                    cc = int(second_c + (j - split) * MOTIF_STEP)
            slots.append((rr, cc))
    return slots


def _paint_motif_slots(
    grid: np.ndarray,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: dict[str, object],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = _motif_slot_positions(Mr, Mc, placement)
    random.shuffle(slots)
    for (r0, c0), (color, mask) in zip(slots, _slot_items(block_colors, distractor_colors, motif_masks)):
        _paint_motif(grid, int(r0), int(c0), int(color), mask)


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
    motif_masks: dict[int, np.ndarray],
    placement: dict[str, object],
) -> np.ndarray:
    """Render the sampled block and motif chunks into the input grid."""
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, R, C, block_colors, placement)
    _paint_motif_slots(inp, Mr, Mc, block_colors, distractor_colors, motif_masks, placement)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    config = _choose_config()
    if config is None:
        return None
    R, C, Mr, Mc, filled = config
    H, W = random.choice(_feasible_canvas_sizes(R, C, Mr, Mc))
    bg = random.randrange(COLOR_COUNT)

    placement = _place_regions(H, W, R, C, Mr, Mc)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, placement
    )

    try:
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D17 input")

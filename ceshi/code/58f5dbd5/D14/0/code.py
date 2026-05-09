# Variant of 58f5dbd5
# Dimension: D14 — 输出排列顺序 (grid_shape)
# Change: 按方块在输入网格中的行列位置排序（保持原始空间顺序）... → 按motif面积排序
#
# Design analysis
# - Concrete behavior: the input still contains a rectangular grid of solid 5x5 blocks and
#   a separate region of same-colored 3x3 motifs. Each block is carved by its own motif, but
#   the packed output no longer preserves the blocks' spatial order. Instead, the carved tiles
#   are sorted by motif filled-cell count and then written back into the original packed grid
#   shape in row-major slot order. Example for a 2x2 block grid: spatial areas [7, 4, 8, 5]
#   used to stay in that order; the variant outputs them as [4, 5, 7, 8].
# - Complexity calibration: sorting by an invisible tie-break would be a bad ARC signal, so
#   the generator makes all matching motif areas distinct. Because valid areas are only 4..8,
#   the original 6-block layouts are dropped; otherwise ties would be unavoidable and the new
#   rule would often collapse back into the old one.
# - Old assumptions audited: block detection still relies on solid 5x5 monochrome components,
#   which remains valid. Color-based motif matching also stays valid because each matching motif
#   keeps a unique color. The part that depended on the old rule was the final packing step:
#   it used input row/column indices as output slot indices. That step is replaced with an
#   area sort. A positional tie-break remains only as a deterministic fallback for off-spec
#   inputs; the generator avoids such ties.
# - Placement legality: destination blocks and motif slots keep the original spacing and
#   separated-region construction, so components do not overlap, do not merge under 8-connectivity,
#   and stay easy for the solver to isolate.
# - Visual observability: generated samples are rejected when the input's spatial block order is
#   already area-sorted. This guarantees that the variant's ordering change is visible in the
#   input/output pair instead of silently matching the original task.
# - New solver assumptions and generator guarantees: every block color has exactly one matching
#   motif; matching motifs have distinct areas; block layouts remain rectangular (1x3, 3x1, 2x2),
#   so sorted tiles can still be repacked into the same compact grid shape.

from __future__ import annotations

import itertools
import random

import numpy as np

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


def _is_solid_block(grid: np.ndarray, comp: dict[str, object]) -> bool:
    """Return whether the component is one solid 5x5 monochrome destination block."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    return bool((crop_bbox(grid, b) == int(col)).all())


def _carve_tile(color: int, bg: int, mask: np.ndarray | None) -> np.ndarray:
    """Create one 5x5 carved tile from a block color and its motif mask."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    if mask is None:
        return tile

    mh, mw = map(int, mask.shape)
    off_r = (DST_BLOCK_SIZE - mh) // 2
    off_c = (DST_BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
            tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then reorder the carved tiles by motif area.

    Input structure:
    - The background is the most common color.
    - Several distinct non-background colors each form one solid 5x5 destination block.
    - Elsewhere, each destination color has a same-colored connected motif sprite that fits in a small bbox.

    Rule:
    - Pair every block with the smallest non-5x5 component of the same color and treat that component as its motif.
    - Center the motif inside the 5x5 block and turn those covered cells into background holes.
    - Sort the carved blocks by motif area from smallest to largest, then pack them into the original compact output grid shape with 1-cell gaps and a 1-cell border.

    Key insight:
    - Color still tells you which motif carves which block, but the final slot order is controlled by motif size rather than by where the blocks sat in the input.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        if not _is_solid_block(g, comp):
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert not (not blocks)

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 5x5 blocks.
    motifs: dict[int, dict[str, object]] = {}
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
            if _is_solid_block(g, comp):
                continue

            r0, c0, r1, c1 = map(int, comp["bbox"])
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            area, _bbox_area, _r0, _c0, mask = candidates[0]
            motifs[int(bcol)] = {"mask": mask, "area": int(area)}

    # STEP 3) Carve each 5x5 tile and remember the matched motif area for sorting.
    carved_blocks: list[dict[str, object]] = []
    for b in blocks:
        col = int(b["col"])
        motif = motifs.get(int(col))
        mask = None if motif is None else motif["mask"]
        area = 999 if motif is None else int(motif["area"])
        carved_blocks.append(
            {
                "r0": int(b["r0"]),
                "c0": int(b["c0"]),
                "area": int(area),
                "tile": _carve_tile(int(col), int(bg), mask),
            }
        )

    # STEP 4) Pack carved tiles into the original rectangular slot grid,
    # but fill the slots in ascending motif-area order.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    R = len(rs)
    C = len(cs)

    out_h = int(1 + R * DST_BLOCK_SIZE + (R - 1) * 1 + 1)
    out_w = int(1 + C * DST_BLOCK_SIZE + (C - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    ordered = sorted(
        carved_blocks,
        key=lambda item: (int(item["area"]), int(item["r0"]), int(item["c0"])),
    )
    for idx, item in enumerate(ordered):
        i = idx // C
        j = idx % C
        rr = int(OUTPUT_BORDER + i * DST_BLOCK_STEP)
        cc = int(OUTPUT_BORDER + j * DST_BLOCK_STEP)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = item["tile"]

    return out


# --------------- generator (D14 area ordering) ---------------

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
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

# Generation retry budget.
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
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
                if distractors > COLOR_COUNT - 1 - count:
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


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _build_block_specs(block_colors: list[int]) -> list[dict[str, object]]:
    """Assign each block a unique motif area and mask in its input-space order."""
    areas = random.sample(
        list(range(MOTIF_AREA_RANGE[0], MOTIF_AREA_RANGE[1] + 1)),
        len(block_colors),
    )
    return [
        {
            "color": int(color),
            "area": int(area),
            "mask": random.choice(MASKS_BY_AREA[int(area)]).copy(),
        }
        for color, area in zip(block_colors, areas)
    ]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _ordered_specs(block_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return block specs sorted by motif area, with input order as a fallback tie-break."""
    indexed = list(enumerate(block_specs))
    indexed.sort(key=lambda pair: (int(pair[1]["area"]), int(pair[0])))
    return [spec for _idx, spec in indexed]


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_specs: list[dict[str, object]],
    *,
    sort_by_area: bool,
) -> np.ndarray:
    """Render the oracle output for either spatial order or area-sorted order."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    ordered_specs = _ordered_specs(block_specs) if sort_by_area else list(block_specs)
    for idx, spec in enumerate(ordered_specs):
        i = idx // C
        j = idx % C
        tile = _carve_tile(int(spec["color"]), int(bg), spec["mask"])
        paste(
            expected,
            tile,
            OUTPUT_BORDER + i * DST_BLOCK_STEP,
            OUTPUT_BORDER + j * DST_BLOCK_STEP,
        )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_specs: list[dict[str, object]]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for idx, spec in enumerate(block_specs):
        i = idx // C
        j = idx % C
        block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(spec["color"]), dtype=int)
        paste(
            grid,
            block,
            dst_r + i * DST_BLOCK_STEP,
            dst_c + j * DST_BLOCK_STEP,
        )


def _slot_items(
    block_specs: list[dict[str, object]], distractor_colors: list[int]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(spec["color"]), spec["mask"]) for spec in block_specs
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
    block_specs: list[dict[str, object]],
    distractor_colors: list[int],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_specs, distractor_colors)

    for (sr, sc), (color, mask) in zip(slots, slot_items):
        _paint_motif(
            grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_specs: list[dict[str, object]],
    distractor_colors: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_specs)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_specs, distractor_colors)
    return inp


def _inversion_count(values: list[int]) -> int:
    """Count out-of-order pairs relative to ascending order."""
    return sum(
        1
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if int(values[i]) > int(values[j])
    )


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    block_specs = _build_block_specs(block_colors)
    spatial_areas = [int(spec["area"]) for spec in block_specs]
    if _inversion_count(spatial_areas) == 0:
        return None

    expected = _render_expected(bg, R, C, block_specs, sort_by_area=True)
    spatial_expected = _render_expected(bg, R, C, block_specs, sort_by_area=False)
    if np.array_equal(expected, spatial_expected):
        return None

    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_specs, distractor_colors, placement
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D14 input")

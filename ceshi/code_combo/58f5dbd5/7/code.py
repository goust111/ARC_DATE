# Combo variant of 58f5dbd5 (index 7)
# combo_spec: D3/0+D7/1+D16/1
# Name: 多色motif+颜色替换+映射图例

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Base task: each 5x5 solid block matched a same-colored small motif; centered motif cells became background.
#    - This combo changes two stages at once.
#      1. The source motif is now a 2-color connected sprite inside a 3x3 bbox:
#         - primary color A = the block-matching color;
#         - decoration color D = a shared accent that also occupies some motif cells.
#      2. A top-left corner legend contains 1x2 pairs [A, B] on even rows, meaning:
#         block color A -> recolor the motif's primary cells to B.
#    - When the motif is centered on the 5x5 block:
#      - cells that were color A in the motif become legend color B;
#      - cells that were color D in the motif become D.
#    - Example trace:
#      original: red block + red mask -> mask cells become bg.
#      combo: red/decor motif + legend [red blue] -> red motif cells become blue, decor motif cells stay decor.
#      So the output block now contains three visible roles: untouched body color, legend-driven recolor, and
#      motif-decoration recolor.
# b) Complexity calibration
#    - The decoration color is shared across all motifs. This keeps the second motif role easy to spot instead of
#      creating many one-off color meanings.
#    - Legend replacement colors come from a derangement of the block colors. Every real block therefore needs the
#      legend, and no block can ignore it by mapping to itself.
#    - Motif patterns always contain both primary and decoration cells, so both color roles appear in every example.
# c) Implicit assumption audit
#    - Original same-color motif search breaks: useful motifs are multicolor, so matching must scan non-background
#      multicolor components and infer which of the two colors is the block color.
#    - Original "write bg into motif footprint" breaks twice:
#      primary cells now use legend_map[block_color], and decoration cells keep the shared decoration color.
#    - Adding the legend introduces small multicolor [A,B] components in the corner. They must not be mistaken for
#      motifs, so the solver parses the legend first and motif candidates are restricted to area 4..8, bbox <= 3x3.
#    - New solver assumption: each useful motif contains exactly one block color and one non-block decoration color.
#      Generator guarantee: every matching motif is built that way, and distractors never use a block color.
# d) Placement legality
#    - The legend reserves a top-left guard rectangle covering the used pair rows, the first empty even row, and one
#      extra moat column. Neither the 5x5 block region nor the motif-slot region may intersect it.
#    - Block layout, motif-slot spacing, and axis-separated region placement remain the original geometry, so motifs,
#      blocks, and legend stay disjoint and within bounds.
# e) Visual observability
#    - Across a few examples, humans can match block color A to a motif containing A, then see that the legend pair
#      [A,B] controls one subset of the output footprint while the repeated decoration color controls the other.
#    - Because every output block shows both mapped-primary pixels and decoration pixels, the fused rule is visible
#      without relying on color ids or hidden ordering.
# f) Visual cleanliness
#    - The legend is confined to a narrow corner strip with blank separator rows.
#    - Motifs remain in a separate slot grid, and the packed output keeps the original clean 1-cell gaps and border.
# g) Color-change inferability
#    - The legend gives explicit local evidence for A -> B.
#    - The shared decoration color reappears both in the motifs and inside every output block, making its role as the
#      second recolor channel easy to infer.
# h) Variant-validity metric used after implementation
#    - On generated samples, count output pixels equal to legend-mapped colors inside packed blocks and count output
#      pixels equal to the shared decoration color inside packed blocks. Both counts must be positive; otherwise the
#      sample would collapse toward a simpler single-rule task.
# i) Solver assumptions checked against generator
#    - Legend format assumption: top-left even-row [A,B] stack. Guaranteed by _paint_legend and _legend_guard_rect().
#    - Motif assumption: each real motif is an 8-connected 2-color component with area 4..8 and bbox <= 3x3.
#      Guaranteed by _make_patterns_by_area() and _paint_motif().
#    - Match uniqueness assumption: each block color appears in exactly one useful motif and exactly one legend entry.
#      Guaranteed by palette construction, shuffled slot assignment, and one motif pattern per block color.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given local coords."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def component_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    """Extract one component into its tight bbox while preserving its colors."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _read_corner_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    """Read the top-left vertical stack of [src, dst] legend pairs placed on even rows."""
    H, W = grid.shape
    if W < 2:
        return {}, set()

    legend_map: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    row = 0
    while row < H:
        left = int(grid[row, 0])
        right = int(grid[row, 1])
        if left == bg or right == bg:
            break
        legend_map[int(left)] = int(right)
        legend_cells.add((int(row), 0))
        legend_cells.add((int(row), 1))
        row += 2
    return legend_map, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply two-color motifs to solid blocks using a corner legend, then pack the transformed blocks.

    Input structure:
    - The background is the most common color.
    - Several destination objects are solid 5x5 monochrome squares with distinct non-background colors.
    - Elsewhere are small 8-connected motifs in 3x3-or-smaller bboxes; each useful motif has exactly two colors:
      one matches a destination block, and the other is a decoration color.
    - In the top-left corner, a vertical list of 1x2 color pairs [A, B] specifies how the motif's primary color A
      should be recolored inside the matching block.

    Rule:
    - Match each block to the unique small motif that contains the block's color.
    - Center that motif on the 5x5 block. Cells of the motif's block-matching color turn into the legend-mapped
      color B, while cells of the motif's decoration color turn into that decoration color.
    - Pack all transformed 5x5 blocks into a compact output grid, preserving the original row/column order with
      1-cell gaps and a 1-cell background border.

    Key insight:
    - The motif is a role-coded stencil. One motif color only says which block to use and gets remapped through the
      corner legend; the other motif color stays as a visible accent inside the packed block.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legend_map, _legend_cells = _read_corner_legend(g, bg)

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    mono_comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in mono_comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(g, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) Identify useful multicolor motifs: small 2-color components containing exactly one block color.
    block_colors = {int(b["col"]) for b in blocks}
    assert all(int(col) in legend_map for col in block_colors)

    motif_candidates: dict[int, list[tuple[int, int, int, int, np.ndarray, np.ndarray, int]]] = {}
    multi_comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, min_area=4, return_format="dict"
    )
    for comp in multi_comps:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue
        if not (4 <= int(comp["area"]) <= 8):
            continue

        patch = component_patch(g, comp, bg)
        colors = sorted({int(v) for v in patch.flatten() if int(v) != int(bg)})
        if len(colors) != 2:
            continue

        matching = [int(c) for c in colors if int(c) in block_colors]
        if len(matching) != 1:
            continue

        primary_color = int(matching[0])
        decor_color = int(colors[0] if colors[1] == primary_color else colors[1])
        if decor_color in block_colors:
            continue
        if primary_color not in legend_map:
            continue

        primary_mask = patch == int(primary_color)
        decor_mask = patch == int(decor_color)
        if not primary_mask.any() or not decor_mask.any():
            continue

        r0, c0, _r1, _c1 = map(int, b)
        motif_candidates.setdefault(primary_color, []).append(
            (
                int(comp["area"]),
                int(patch.shape[0] * patch.shape[1]),
                int(r0),
                int(c0),
                primary_mask.copy(),
                decor_mask.copy(),
                int(decor_color),
            )
        )

    motifs: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for block_color in block_colors:
        candidates = motif_candidates.get(int(block_color), [])
        assert candidates
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        _area, _bbox_area, _r0, _c0, primary_mask, decor_mask, decor_color = candidates[0]
        motifs[int(block_color)] = (primary_mask, decor_mask, int(decor_color))

    # STEP 3) Transform each 5x5 tile by centering the matched motif and applying both color roles.
    transformed: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        fill = int(legend_map[int(col)])
        tile = np.full((5, 5), int(col), dtype=int)
        primary_mask, decor_mask, decor_color = motifs[int(col)]
        mh, mw = map(int, primary_mask.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2

        for rr, cc in np.argwhere(primary_mask):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(fill)

        for rr, cc in np.argwhere(decor_mask):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(decor_color)

        transformed[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack transformed 5x5 tiles into a compact grid.
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
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = transformed[(int(r0), int(c0))]

    return out


# --------------- generator (combo variant) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
DECOR_AREA_RANGE = (1, 3)
MIN_PRIMARY_AREA = 3
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1
MOTIF_OFFSET_IN_BLOCK = 1

GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 40
TARGET_BLOCK_COUNTS = tuple(sorted(DST_LAYOUTS))
_count_cycle: list[int] = []


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_patterns_by_area() -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Precompute valid 3x3 two-color motif patterns keyed by total occupied area."""
    min_area, max_area = MOTIF_AREA_RANGE
    patterns: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        shape = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(shape.sum())
        if area not in patterns:
            continue
        if not (shape[0].any() and shape[-1].any() and shape[:, 0].any() and shape[:, -1].any()):
            continue
        if not is_contiguous(shape.astype(int), background=0, connectivity=8):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
        max_decor = min(int(DECOR_AREA_RANGE[1]), area - 1)
        for decor_area in range(int(DECOR_AREA_RANGE[0]), max_decor + 1):
            for idxs in itertools.combinations(range(len(coords)), decor_area):
                decor = np.zeros_like(shape, dtype=bool)
                for idx in idxs:
                    r, c = coords[idx]
                    decor[int(r), int(c)] = True
                primary = shape & ~decor
                if int(primary.sum()) < MIN_PRIMARY_AREA:
                    continue
                patterns[area].append((primary.copy(), decor.copy()))
    return patterns


PATTERNS_BY_AREA = _make_patterns_by_area()


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
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
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
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


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_regions(
    H: int,
    W: int,
    dh: int,
    dw: int,
    mh: int,
    mw: int,
    blocked: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None

    for _ in range(PLACEMENT_RETRIES):
        scheme = random.choice(schemes)
        if scheme in {"dst_above", "motif_above"}:
            placement = _place_vertical(H, W, dh, dw, mh, mw, scheme)
        else:
            placement = _place_horizontal(H, W, dh, dw, mh, mw, scheme)

        dst_r, dst_c, motif_r, motif_c = placement
        dst_rect = (dst_r, dst_c, dst_r + dh - 1, dst_c + dw - 1)
        motif_rect = (motif_r, motif_c, motif_r + mh - 1, motif_c + mw - 1)
        if blocked is not None and (
            _rects_intersect(dst_rect, blocked) or _rects_intersect(motif_rect, blocked)
        ):
            continue
        return placement
    return None


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
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
                if distractors > COLOR_COUNT - 2 - count:
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


def _choose_config(H: int, W: int, target_count: int | None = None) -> tuple[int, int, int, int, int] | None:
    configs = _configs(H, W)
    if not configs:
        return None
    if target_count is not None:
        configs = [cfg for cfg in configs if cfg[0] * cfg[1] == int(target_count)]
        if not configs:
            return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_pattern() -> tuple[np.ndarray, np.ndarray]:
    area = random.randint(*MOTIF_AREA_RANGE)
    primary, decor = random.choice(PATTERNS_BY_AREA[area])
    return primary.copy(), decor.copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[int, list[int], list[int]]:
    """Pick one shared decoration color, then block colors, then distractor primary colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    decor_color = int(palette[0])
    block_colors = [int(c) for c in palette[1 : 1 + count]]
    distractor_colors = [int(c) for c in palette[1 + count : 1 + count + (filled - count)]]
    return int(decor_color), block_colors, distractor_colors


def _random_derangement(values: list[int]) -> dict[int, int]:
    shuffled = values[:]
    while True:
        random.shuffle(shuffled)
        if all(int(a) != int(b) for a, b in zip(values, shuffled)):
            return {int(a): int(b) for a, b in zip(values, shuffled)}


def _paint_motif(
    grid: np.ndarray,
    r0: int,
    c0: int,
    primary_color: int,
    decor_color: int,
    bg: int,
    pattern: tuple[np.ndarray, np.ndarray],
) -> None:
    primary_mask, decor_mask = pattern
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(bg), dtype=int)
    patch[primary_mask] = int(primary_color)
    patch[decor_mask] = int(decor_color)
    paste(grid, patch, r0, c0, mask=(primary_mask | decor_mask))


def _paint_legend(grid: np.ndarray, block_colors: list[int], replacement_map: dict[int, int]) -> None:
    for idx, src in enumerate(block_colors):
        row = 2 * idx
        grid[row, 0] = int(src)
        grid[row, 1] = int(replacement_map[int(src)])


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
    replacement_map: dict[int, int],
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), int(bg), dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            primary_mask, decor_mask = motif_patterns[color]
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            sub = tile[motif_slice, motif_slice]
            sub[primary_mask] = int(replacement_map[color])
            sub[decor_mask] = int(decor_color)
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    block_colors: list[int],
    distractor_colors: list[int],
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    items: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = [
        (int(color), motif_patterns[color]) for color in block_colors
    ]
    items.extend((int(color), _random_pattern()) for color in distractor_colors)
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
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
    bg: int,
) -> None:
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, motif_patterns)

    for (sr, sc), (primary_color, pattern) in zip(slots, slot_items):
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            int(primary_color),
            int(decor_color),
            int(bg),
            pattern,
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
    motif_patterns: dict[int, tuple[np.ndarray, np.ndarray]],
    decor_color: int,
    replacement_map: dict[int, int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_legend(inp, block_colors, replacement_map)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_patterns,
        decor_color,
        bg,
    )
    return inp


def _legend_guard_rect(count: int) -> tuple[int, int, int, int]:
    """Reserve legend rows, the first terminating blank even row, and a 1-cell right moat."""
    return (0, 0, 2 * count, 2)


def _next_target_count() -> int:
    global _count_cycle
    if not _count_cycle:
        _count_cycle = list(TARGET_BLOCK_COUNTS)
        random.shuffle(_count_cycle)
    return int(_count_cycle.pop())


def _gen_attempt(target_count: int | None = None) -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W, target_count=target_count)
    if config is None:
        return None
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)

    count = R * C
    blocked = _legend_guard_rect(count)
    if blocked[2] >= H or blocked[3] >= W:
        return None

    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w, blocked=blocked)
    if placement is None:
        return None

    decor_color, block_colors, distractor_colors = _choose_palette(bg, count, filled)
    replacement_map = _random_derangement(block_colors)
    motif_patterns = {color: _random_pattern() for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_patterns, decor_color, replacement_map)
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
        motif_patterns,
        decor_color,
        replacement_map,
        placement,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    target_count = _next_target_count()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(target_count=target_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

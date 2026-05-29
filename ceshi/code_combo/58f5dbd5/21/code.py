# Combo variant of 58f5dbd5 (index 21)
# combo_spec: D8/2+D14/1
# Name: 右下角对齐+反转行列顺序

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis
# combo_variant:
#   preserved_structure:
#     - 输入仍由两个分离区域组成：若干个5x5纯色目标块，以及若干个3x3槽位中的小型同色/干扰motif。
#     - 匹配关系仍只靠颜色：每个目标块颜色唯一，matching motif 与其同色，干扰 motif 用其他非背景色。
#     - 输出仍是把所有5x5块雕刻后打包成紧凑网格，块之间保留1格间隔，外层保留1格背景边框。
#   fused_changes:
#     - D8/2: motif 不再在5x5块内居中，而是按其“裁剪后 bbox”贴到右下角，偏移量改为 (5-h, 5-w)。
#     - D14/1: 输出打包时，块的行序和列序同时反转；输入左上角的块会被放到输出右下角。
#   concrete_behavior_trace:
#     - 例：输入目标块按 2x2 排列，颜色顺序是
#         A B
#         C D
#       某个颜色 A 的 matching motif 若在3x3槽位里只有一个 2x3 的裁剪后 bbox，
#       原题会把这块洞放在 A 块中心附近，且 A 仍出现在输出左上。
#     - 本组合里，同一个 2x3 bbox 会被贴到 A 块右下，因此洞靠近块的右边界/下边界；
#       同时打包顺序在两个轴都翻转，所以上述 2x2 会变成
#         D C
#         B A
#       因而 A 的雕刻块最终落在输出右下。
#     - 两个改动协同后，解题者既要看出“洞的位置靠右下”，又要看出“颜色块整体次序反向映射”。
#   complexity_calibration:
#     - 如果 matching motif 的裁剪后 bbox 总是 3x3，那么居中和右下角对齐都会得到同一偏移 (1,1)，D8/2 将退化不可见。
#     - 因此 generator 必须强制 matching motif 来自 anchor-sensitive 池：其裁剪后 bbox 至少有一个维度小于3。
#     - D14/1 依赖块的空间次序可见，所以仍保持 3-6 个不同颜色块，避免重复颜色让“反转顺序”失去辨识度。
#   implicit_assumption_review:
#     - main 中识别目标块的逻辑依赖“完整纯色 5x5 方块”；组合后该前提不变，generator 继续严格生成纯色块。
#     - main 中按颜色寻找最小同色非5x5组件，依赖“每个块颜色唯一，matching motif 同色且比块小，也没有同色碎片”；
#       generator 继续保证 matching motif 与干扰 motif 颜色互不重复，且每种颜色至多出现一个小组件。
#     - 右下角对齐依赖的是组件 bbox，而不是槽位尺寸；因此 generator 的 oracle 必须先裁掉 motif 槽位空边，再贴到5x5右下。
#     - 反转行列顺序依赖输入块能形成规则的行集/列集；原题本就把目标块放成 RxC 矩形网格，这个隐式前提保持成立。
#   placement_legality:
#     - motif 仍绘制在互相间隔1格的3x3槽位中，即使裁剪后只有2x2或2x3，也不会越界或碰触相邻槽位。
#     - 目标块区与 motif 区仍用原始分区逻辑在行轴或列轴上分离，至少隔1格，避免组件混连。
#   visual_observability:
#     - 右下锚点通过“洞总是贴近块的下边/右边”可观察；顺序反转通过“输入左上块出现在输出右下”等位置对照可观察。
#     - 两个规律都只依赖几何位置和颜色对应，不依赖颜色编号大小等人类不可视信息。
#   cleanliness:
#     - 输入仍是两个清晰分区，没有叠加标记；输出仍只显示雕刻后的5x5块阵列，整体视觉干净。
#   generator_validation_plan:
#     - 除常规 `main(generate_input())` 成功率测试外，还要统计：
#       1) matching motif 中有多少比例的裁剪 bbox 小于3x3，确保右下锚点不是退化情况；
#       2) 新规则输出与“原题居中+不反转顺序”输出的差异比例，确认融合行为在样本中真实发生。


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


def crop_true_bbox(mask: np.ndarray) -> np.ndarray:
    """Crop a boolean mask to the bbox of its True cells."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then repack the carved blocks in reversed order.

    Input structure:
    - The most frequent color is the background.
    - Several non-background 5x5 solid squares form a rectangular block layout.
    - Elsewhere on the grid, each block color has one smaller same-colored motif component, with optional distractor motifs in other colors.

    Rule:
    - Match each small motif to the 5x5 square of the same color.
    - Use the motif's occupied bbox as a carving stencil, align that stencil to the lower-right corner of the 5x5 block, and turn the covered cells into background.
    - Pack the carved 5x5 blocks into a compact output grid with 1-cell gaps and a 1-cell border, but reverse both the row order and the column order relative to the input block layout.

    Key insight:
    - The sprite is removed from, not pasted onto, its same-colored block, and this variant combines two geometric cues: the holes hug the block's lower-right corner, while the whole block array is mirrored in placement order across both axes.
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
            blocks.append({"col": col, "r0": int(r0), "c0": int(c0)})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is not one of the 5x5 blocks.
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

    # STEP 3) Carve each 5x5 tile:
    # align the motif bbox to the bottom-right corner and repaint those cells to bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        mm = motifs.get(col)
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = 5 - mh
            off_c = 5 - mw
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = bg
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid with reversed row/column order.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    step = 6
    last_i = len(rs) - 1
    last_j = len(cs) - 1
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = last_i - int(r_index[r0])
        j = last_j - int(c_index[c0])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


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
REGION_GAP = 1
OUTPUT_BORDER = 1

# Generation retry budget.
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif-slot masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


def _make_sensitive_masks_by_area(
    masks_by_area: dict[int, list[np.ndarray]]
) -> dict[int, list[np.ndarray]]:
    """Keep only masks whose cropped bbox is smaller than 3x3 on at least one axis."""
    out: dict[int, list[np.ndarray]] = {area: [] for area in masks_by_area}
    for area, masks in masks_by_area.items():
        for mask in masks:
            cropped = crop_true_bbox(mask)
            if cropped.shape != (MOTIF_SIZE, MOTIF_SIZE):
                out[area].append(mask)
    return out


MASKS_BY_AREA = _make_masks_by_area()
SENSITIVE_MASKS_BY_AREA = _make_sensitive_masks_by_area(MASKS_BY_AREA)


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
    """Sample a valid layout, biasing toward 2D block grids when available."""
    configs = _configs(H, W)
    if not configs:
        return None

    # Prefer examples that reveal reversal on both axes in a single grid.
    two_dimensional = [cfg for cfg in configs if cfg[0] > 1 and cfg[1] > 1]
    if two_dimensional and random.random() < 0.7:
        configs = two_dimensional

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask(*, sensitive: bool) -> np.ndarray:
    """Sample one 3x3 slot mask, optionally forcing anchor-sensitive cropped bboxes."""
    pools = SENSITIVE_MASKS_BY_AREA if sensitive else MASKS_BY_AREA
    min_area, max_area = MOTIF_AREA_RANGE
    valid_areas = [area for area in range(min_area, max_area + 1) if pools[area]]
    area = random.choice(valid_areas)
    return random.choice(pools[area]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif-slot mask into the input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _carved_tile_from_mask(bg: int, color: int, slot_mask: np.ndarray) -> np.ndarray:
    """Carve a 5x5 block using the cropped motif bbox aligned to the lower-right corner."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    cropped = crop_true_bbox(slot_mask)
    mh, mw = map(int, cropped.shape)
    off_r = DST_BLOCK_SIZE - mh
    off_c = DST_BLOCK_SIZE - mw
    for rr, cc in np.argwhere(cropped):
        tile[int(off_r + rr), int(off_c + cc)] = int(bg)
    return tile


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    last_i = R - 1
    last_j = C - 1
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carved_tile_from_mask(bg, color, motif_masks[color])
            paste(
                expected,
                tile,
                OUTPUT_BORDER + (last_i - i) * DST_BLOCK_STEP,
                OUTPUT_BORDER + (last_j - j) * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _slot_items(
    block_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[color]) for color in block_colors
    ]
    items.extend((int(color), _random_mask(sensitive=False)) for color in distractor_colors)
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
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks
    )
    return inp


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
    motif_masks = {color: _random_mask(sensitive=True) for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, distractor_colors, motif_masks, placement
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

# Variant of 58f5dbd5
# Dimension: D4 — 源对象：motif数量与干扰 (src)
# Change: 总共6-8个motif插槽，其中与方块颜色匹配的motif数量等于方块数，其余为干扰项（0到方块数个干扰），可有0-2个... → 重复颜色干扰

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis
# ---------------
# spec_variant:
#   core_change:
#     original: 每个 5x5 方块颜色在 motif 区只出现一次，因此“按颜色匹配”就足够。
#     variant: 部分干扰 motif 会复用某些方块颜色，但形状不同。每个方块仍只雕刻一个 motif，
#       规则改成“同色非方块组件中，面积最小的那个才是正确 stencil；更大的同色组件只是干扰”。
#   concretization:
#     - 输入仍由两个分离区域组成：一个是 5x5 纯色方块阵列，一个是 3x3 motif 插槽阵列。
#     - 方块颜色仍互不相同。
#     - 每个方块颜色至少有一个对应 motif；其中部分颜色会额外再出现一个同色 motif。
#     - 这两个同色 motif 的颜色相同，但形状不同，并且正确 motif 的填充面积严格更小。
#     - 输出仍是把每个方块按原行列顺序紧凑打包，并在方块中心用正确 motif 挖空。
#   behavior_trace:
#     - 旧流程：红色方块 -> 找到唯一红色 motif -> 居中雕刻。
#     - 新流程：红色方块 -> 看到两个红色 motif（例如面积 4 与面积 7）-> 选面积 4 的那个雕刻。
#     - 差异不在输出排版，而在输入里出现“颜色足以匹配但仍需消歧”的来源对象。
#   complexity_calibration:
#     - 每个样例至少包含 1 个重复颜色干扰，避免退化成原题。
#     - 每个颜色至多增加 1 个同色干扰，避免同一区域重复颜色过多而显得杂乱。
#     - 正确 motif 面积限制在 4..6，同色干扰面积必须更大，便于肉眼比较大小并归纳规则。
#     - 仍保留 0..2 空槽与少量异色干扰，维持原题“motif 面板里有噪声”的整体观感。
#   implicit_assumptions_review:
#     - 原 main 的 STEP 1 假设 5x5 纯色块可单独识别；该假设仍成立，保持不变。
#     - 原 main 的 STEP 2 用“同色最小组件”选 motif。原题里这是多余但无害；变体里它变成关键规则。
#     - 该规则要求：每个方块颜色都至少有一个非 5x5 同色组件，且正确组件面积唯一最小。
#       生成器必须保证这一点，否则按面积排序会产生隐藏歧义。
#     - STEP 3 居中雕刻依赖 motif bbox 不超过 3x3；生成器继续只产生 3x3 槽位内连通图形。
#     - STEP 4 仅依赖方块原始网格布局，与 D4 无关，保持不变。
#   placement_legality:
#     - 方块区与 motif 区继续沿某一轴分离放置，保证不重叠且背景可分辨。
#     - motif 槽步长仍为 4，因此不同槽位的同色 motif 不会因 8 邻接意外连成一个组件。
#     - 空槽保留背景色，不与任何 motif 叠加。
#   visual_observability:
#     - 解题者会在输入里直接看到某些颜色出现两次，但输出里的孔洞只对应较小的那一个。
#     - 只用“面积更小”这一肉眼可见属性做消歧，不依赖颜色 id、坐标排序或其他不可见元信息。
#   cleanliness:
#     - 重复颜色只发生在 motif 区；方块区仍是规整矩阵，输出也仍是规整打包，画面不会变脏。
#   color_reasoning:
#     - 没有新的颜色映射。颜色仍只是把 motif 候选集限定到对应方块颜色；
#       真正新增的是同色候选之间的面积比较。


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given relative coordinates."""
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
    Carve each solid block using the smallest same-colored motif and repack the blocks.

    Input structure:
    - The background is the most common color.
    - One region contains several solid 5x5 monochrome blocks, each with a distinct non-background color.
    - A separate region contains small monochrome motifs in 3x3 slots; some colors appear once, while others appear twice because a larger same-colored distractor is present.

    Rule:
    - For each block color, gather all non-block motifs of that same color and choose the one with the smallest filled area.
    - Center that chosen motif inside the matching 5x5 block and turn the motif cells into background, leaving the rest of the block unchanged.
    - Pack the carved blocks in original row/column order with 1-cell gaps and a 1-cell background border.

    Key insight:
    - Color still tells you which motifs belong to which block, but it is no longer enough on its own: repeated-color decoys exist, so the true stencil is the smallest same-colored sprite.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination blocks: solid 5x5 monochrome squares.
    blocks: list[dict[str, int]] = []
    motif_candidates_by_color: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        col = int(comp["color"])
        if bbox_h(b) == 5 and bbox_w(b) == 5:
            patch = crop_bbox(g, b)
            if bool((patch == col).all()):
                r0, c0, _r1, _c1 = map(int, b)
                blocks.append({"col": col, "r0": r0, "c0": c0})
                continue

        r0, c0, r1, c1 = map(int, b)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = mask_from_coords(rel_coords, (h, w))
        motif_candidates_by_color.setdefault(col, []).append(
            (int(comp["area"]), int(h * w), int(r0), int(c0), mask)
        )

    assert blocks

    # STEP 2) For each block color, pick the smallest same-colored non-block component.
    motifs: dict[int, np.ndarray] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates = motif_candidates_by_color.get(int(bcol), [])
        assert candidates
        candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        motifs[int(bcol)] = candidates[0][4]

    # STEP 3) Carve each 5x5 tile by centering the chosen motif and turning it into bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        mm = motifs[col]
        mh, mw = map(int, mm.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mm):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = bg
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        i = int(r_index[int(b["r0"])])
        j = int(c_index[int(b["c0"])])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(b["r0"]), int(b["c0"]))]

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
CORRECT_MOTIF_AREA_RANGE = (4, 6)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2
MIN_REPEATED_COLOR_DISTRACTORS = 1

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
    """Precompute valid 3x3 connected motif masks keyed by filled-cell area."""
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
                if not (MIN_REPEATED_COLOR_DISTRACTORS <= distractors <= count):
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


def _random_mask(area_range: tuple[int, int] = MOTIF_AREA_RANGE) -> np.ndarray:
    """Sample one motif mask from the precomputed masks within the requested area range."""
    lo, hi = area_range
    area = random.randint(lo, hi)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _sample_correct_mask() -> np.ndarray:
    """Pick the true stencil mask from the smaller end of the allowed area range."""
    return _random_mask(CORRECT_MOTIF_AREA_RANGE)


def _sample_larger_same_color_mask(correct_mask: np.ndarray) -> np.ndarray:
    """Pick a larger-area same-color distractor for one block color."""
    correct_area = int(correct_mask.sum())
    areas = list(range(correct_area + 1, MOTIF_AREA_RANGE[1] + 1))
    masks = [m for area in areas for m in MASKS_BY_AREA[area]]
    return random.choice(masks).copy()


def _choose_colors(
    bg: int, count: int, distractor_count: int
) -> tuple[list[int], list[int], list[int]]:
    """Pick block colors, repeated block colors, and foreign distractor colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    foreign_palette = palette[count:]

    min_repeat = max(
        MIN_REPEATED_COLOR_DISTRACTORS,
        distractor_count - len(foreign_palette),
    )
    max_repeat = min(count, distractor_count)
    repeat_count = random.randint(min_repeat, max_repeat)
    repeat_colors = random.sample(block_colors, repeat_count)
    foreign_count = distractor_count - repeat_count
    foreign_colors = foreign_palette[:foreign_count]
    return block_colors, repeat_colors, foreign_colors


def _build_block_motifs(
    block_colors: list[int], repeat_colors: list[int]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Create the true motif for every block color plus larger same-color decoys where needed."""
    correct_masks: dict[int, np.ndarray] = {}
    repeated_masks: dict[int, np.ndarray] = {}
    repeat_set = set(map(int, repeat_colors))
    for color in block_colors:
        correct = _sample_correct_mask()
        correct_masks[int(color)] = correct
        if int(color) in repeat_set:
            repeated_masks[int(color)] = _sample_larger_same_color_mask(correct)
    return correct_masks, repeated_masks


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
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
    block_colors: list[int],
    repeat_colors: list[int],
    foreign_colors: list[int],
    correct_masks: dict[int, np.ndarray],
    repeated_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), correct_masks[int(color)]) for color in block_colors
    ]
    items.extend(
        (int(color), repeated_masks[int(color)]) for color in repeat_colors
    )
    items.extend((int(color), _random_mask()) for color in foreign_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    slot_items: list[tuple[int, np.ndarray]],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)

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
    block_colors: list[int],
    slot_items: list[tuple[int, np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, slot_items)
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
    distractor_count = filled - count
    block_colors, repeat_colors, foreign_colors = _choose_colors(
        bg, count, distractor_count
    )
    correct_masks, repeated_masks = _build_block_motifs(block_colors, repeat_colors)
    slot_items = _slot_items(
        block_colors, repeat_colors, foreign_colors, correct_masks, repeated_masks
    )

    expected = _render_expected(bg, R, C, block_colors, correct_masks)
    inp = _render_input(
        H, W, bg, R, C, Mr, Mc, block_colors, slot_items, placement
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 D4 variant input")

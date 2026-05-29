# Variant of 58f5dbd5
# Dimension: D19 — src-dst基数关系 (meta)
# Change: 1:1 — 每个方块恰好对应一个匹配motif... → 一对多（一个motif雕刻多个方块）

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


# Design analysis for the D19 variant
# -----------------------------------
# a) 具体化与行为追踪
# - 目标区域仍然是若干个 5x5 纯色块，按规则网格摆放，输出仍然把这些块按原始行列顺序紧凑拼回。
# - 变体的唯一核心变化是：目标块颜色不再两两不同。输入里会有 3-6 个块，但只有 2-4 种块色；
#   因而某些颜色会在多个 5x5 块上重复出现。
# - motif 区域不再“每个块一个匹配 motif”，而是“每种块色一个匹配 motif”。例如四个块的颜色顺序可以是
#   [红, 蓝, 红, 黄]，motif 区域里只会出现红/蓝/黄三个匹配 motif。solver 必须推断“红色 motif 要复制
#   到两个红色块上”，于是输出中两个红块被雕刻成同一个洞形。
# - 原逻辑在颜色 [红, 蓝, 绿, 黄] 时，是“每个颜色只服务一个块”；新逻辑在 [红, 蓝, 红, 黄] 时，
#   变成“红色这个颜色类服务两个块”。像素级动作并未改变，改变的是颜色到目标实例的基数。
#
# b) 复杂度校准
# - 若允许所有块颜色都不同，样本会退化回原题；因此生成器强制至少一个颜色重复。
# - 若只让一个颜色重复一次而其余几乎都唯一，变体痕迹仍然偏弱；因此只使用少量离散分组模板，例如
#   3 块 -> [2,1]，4 块 -> [2,2]/[3,1]/[2,1,1]，6 块 -> [2,2,2]/[3,2,1]/[4,2]/[3,3]/[2,2,1,1]。
#   这些模板保证视觉上稳定出现“同色多块共享同一洞形”。
# - motif 仍然限制在 3x3、面积 4-8、8 连通，并保持少量 distractor，避免把任务复杂度改到别的维度。
#
# c) 原 main() 中与旧前提相关的隐式假设审查
# - STEP 1 收集 5x5 方块时没有要求颜色唯一，因此这一段在变体下仍然成立。
# - STEP 2 使用 `block_colors = {b['col'] for b in blocks}`，这其实已经是“按不同颜色找 motif”而不是
#   “按块实例找 motif”；该写法在变体下是正确的，但原注释会误导，所以必须重写说明。
# - STEP 3 按块遍历并用 `motifs[col]` 雕刻，这天然会把同一个 motif 应用到所有同色块；这也是变体核心。
# - STEP 4 依据块的原始行列位置打包，与颜色基数无关，可以保留。
# - 潜在风险在 generator：原生成器默认 `block_colors` 全 distinct，`_slot_items()` 也按块数放匹配 motif。
#   若不改 generator，输入不会体现新规则。变体实现重点因此落在样本构造。
#
# d) 放置合法性
# - 5x5 块仍按步长 6 放入目标区，彼此不重叠。
# - motif 仍按 3x3 槽位、步长 4 放入 motif 区，匹配 motif 仅按“不同块色数”放置一次，其他槽位可放
#   distractor 或留空；目标区和 motif 区仍沿某个轴分离。
# - 同色块之间虽然颜色相同，但由于间隔至少 1 格背景，不会在 8 连通下合并。
#
# e) 视觉可观察性
# - 人类解题者能直接看到：输入里多个大块是同一颜色，但外面只出现一个同色小 motif；输出里这些同色块
#   出现完全相同的洞形。这是 2-3 个训练样例中就能归纳出的稳定关系。
# - 这里没有依赖颜色 id 的排序或数值运算，只依赖肉眼可见的“同色归类”。
#
# f) 可视化整洁度
# - 保留原题的“目标区”和“motif 区”双区域布局，不额外叠加标记。
# - 重复颜色只发生在 5x5 块内部，不引入新的图层，因此视觉上仍然干净。
#
# g) 颜色变化可推断性
# - 没有新颜色变换；颜色只用于建立“共享 motif 的等价类”。
# - 生成器控制不同块色数为 2-4，避免颜色种类过多导致匹配关系不可读。
#
# h) 新 main() 的隐式假设与 generator 保证
# - main() 假设：每一种出现在 5x5 块上的颜色，在别处至多需要找到一个匹配 motif，且 distractor 不会
#   复用这些块色。generator 通过“每种块色 exactly one motif、distractor 颜色全部不同且不属于块色”
#   来保证这一点。
# - main() 还假设：匹配 motif 不是完整 5x5 块。generator 继续把 motif 限制为触边的 3x3 连通图形。


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
    Use one color-matched motif as a shared stencil for every block of that color.

    Input structure:
    - bg is the most common color.
    - Several solid 5x5 monochrome blocks appear in a separated destination region; some block colors may repeat.
    - Elsewhere, there is one small motif for each distinct block color, plus optional distractor motifs in other colors.

    Rule:
    - Group the 5x5 blocks by color and find the single same-colored motif for each color class.
    - Center that motif inside every 5x5 block of the same color and repaint motif cells to bg, carving holes.
    - Pack all carved 5x5 blocks into a compact output grid that preserves their original row/column order, with 1-cell gaps and a 1-cell border.

    Key insight:
    - The motif is a shared stencil for a whole color class: once one motif is matched to a color, it must be copied onto every block of that color rather than paired with just one block.
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

    # STEP 2) For each distinct block color, identify the shared motif:
    # the smallest same-colored component that is NOT one of the 5x5 blocks.
    motifs: dict[int, np.ndarray] = {}
    distinct_block_colors = {int(b["col"]) for b in blocks}
    for bcol in distinct_block_colors:
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

    # STEP 3) Carve each 5x5 tile by reusing the motif of its color class.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
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
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
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
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

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

# Repetition templates: one motif color must fan out to multiple blocks.
COLOR_GROUPS = {
    3: [(2, 1)],
    4: [(2, 2), (3, 1), (2, 1, 1)],
    6: [(2, 2, 2), (3, 2, 1), (4, 2), (3, 3), (2, 2, 1, 1)],
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
                if filled > COLOR_COUNT - 1:
                    continue
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
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


def _sample_group_sizes(count: int) -> list[int]:
    """Choose a repeated-color partition for the destination blocks."""
    return list(random.choice(COLOR_GROUPS[int(count)]))


def _choose_palette(bg: int, group_sizes: list[int], filled: int) -> tuple[list[int], list[int], list[int]]:
    """Pick repeated block colors, their per-block assignment, and distractor colors."""
    matched_count = len(group_sizes)
    distractor_count = filled - matched_count
    assert distractor_count >= 0

    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    matched_colors = palette[:matched_count]
    distractor_colors = palette[matched_count : matched_count + distractor_count]

    block_colors: list[int] = []
    for color, size in zip(matched_colors, group_sizes):
        block_colors.extend([int(color)] * int(size))
    random.shuffle(block_colors)
    return block_colors, matched_colors, distractor_colors


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
    matched_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[color]) for color in matched_colors
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
    matched_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(matched_colors, distractor_colors, motif_masks)

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
    matched_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, matched_colors, distractor_colors, motif_masks
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
    group_sizes = _sample_group_sizes(count)
    block_colors, matched_colors, distractor_colors = _choose_palette(bg, group_sizes, filled)
    motif_masks = {color: _random_mask() for color in matched_colors}

    if len(set(block_colors)) >= count:
        return None

    expected = _render_expected(bg, R, C, block_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        matched_colors,
        distractor_colors,
        motif_masks,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D19 variant input")

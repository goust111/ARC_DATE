# Combo variant of 58f5dbd5 (index 1)
# combo_spec: D1/2+D14/0
# Name: 大小排序匹配+面积排序输出
#
# Design analysis
# combo_variant_spec:
#   core_change:
#     - 原题先用“同色”匹配 motif 和 5x5 block，再按 block 原空间顺序输出。
#     - 本融合版同时改两处：先把 motif 按面积从小到大排序，再把这些 motif
#       依次匹配给按“左到右”排序的 block；输出时又按匹配到的 motif 面积从小到大
#       重排 carved block，而不是保留输入里的 block 空间顺序。
#   concretization:
#     - “左到右”若直接套在多行 block 上会有歧义，所以这里具体化为列优先顺序：
#       先按 c0 从小到大，再在同一列内按 r0 从小到大。
#       这仍然忠于“先看更靠左的 block”，同时给多行布局一个可见且稳定的 tie-break。
#     - 输出的紧凑网格仍保持原题的矩形行列形状，但填槽顺序是常规行优先（row-major）。
#       因此融合后的真实效果是：match 阶段按列优先给 block 分配面积排名，
#       output 阶段再按面积升序写回行优先槽位。
#   behavior_trace:
#     - 以 2x3 block 网格举例，输入中的空间位置若记为
#         [A B C
#          D E F]
#       列优先左到右顺序则是 [A, D, B, E, C, F]。
#     - 若六个有效 motif 的面积升序为 [3, 4, 5, 6, 7, 8]，
#       则匹配后 A,D,B,E,C,F 分别得到这些洞型。
#     - 输出阶段再按面积升序把 carved block 写入 2x3 的行优先槽位，
#       所以输出中的颜色块顺序就是 [A, D, B, E, C, F]，而不再是输入里的 [A, B, C, D, E, F]。
#     - 相比原题，颜色不再提供配对线索；相比单独的 D1 变体，最终 block 的排布也不再跟输入空间顺序一致。
#   complexity_calibration:
#     - 为了支持最多 6 个 block 且避免面积 tie，匹配用 motif 面积改为 3..8 的 6 个档位。
#       允许 motif 的真实 bbox 小于 3x3，这样 area=3 也合法且视觉上明显更小。
#     - distractor motif 统一使用 area=9，比所有有效 motif 都大；solver 取面积最小的 N 个
#       就能稳定剔除干扰项。
#     - 为避免融合退化，generator 只保留同时具有多行和多列的 block 布局：
#       2x2、2x3、3x2。若用了 1xN 或 Nx1，则“列优先匹配”和“行优先输出”会重合，
#       D14 的改动将不可见。
#   implicit_assumptions_review:
#     - 原 main 逐色寻找同色 motif，这依赖“每个 block 都有唯一同色候选”。融合后该前提失效，
#       因为 motif 颜色被刻意做成与 block 颜色无关且允许复用。
#     - 原 main 的输出打包依赖“carved block 继续按输入 row/col 索引放回”。融合后必须先按
#       匹配到的 motif 面积重新排序，再填入紧凑网格。
#     - 原 generator 默认 3x3 motif 全都占满四边并可用颜色唯一识别；这里改为 trimmed mask，
#       允许 bbox 小于 3x3，并使用一个很小的 motif 颜色池重复着色，主动打破颜色线索。
#   placement_legality:
#     - 5x5 block 仍放在规则步长 6 的矩形网格上；motif 仍放在步长 4 的 3x3 slot 网格上。
#     - 两个区域继续沿某一轴彻底分离，且都有 1 格外边距；对象之间不重叠，也不会在 8 邻接下误并。
#     - motif mask 先在 3x3 bitmask 中采样，再裁成真实 bbox；绘制时重新居中到 slot 中，
#       因此不会越界。
#   visual_observability:
#     - motif 颜色与 block 颜色脱钩，而且 motif 颜色重复，观察者无法用“同色对应”解释输出。
#     - 有效 motif 的面积严格递增，洞型大小差异直接可见；多余 distractor 统一更大，也容易识别为候选之外。
#     - 输入中 block 的空间颜色顺序与输出中的颜色顺序系统性不同：输入按矩形位置排，
#       输出则变成“列优先匹配得到的面积序列再按行优先写回”，2-3 组样例足以暴露该规律。
#   cleanliness:
#     - 仍然保持“block 区 + motif 区”两大清晰分区，不增加额外标记层。
#     - 所有对象都是单色连通体，输出仍是规整的 5x5 打孔块阵列，可读性高。
#   solver_assumptions_checked_against_generator:
#     - solver 假设至少存在一个 solid 5x5 block：generator 保证 count 只能是 4 或 6。
#     - solver 假设非 block 组件数 >= block 数：generator 总会放置 count 个有效 motif。
#     - solver 假设有效 motif 是面积最小的 count 个组件：generator 令有效面积属于 3..8，
#       distractor 固定为 9。
#     - solver 假设 block 网格既有多行又有多列，使“列优先匹配”和“行优先输出”不同：
#       generator 只采样 2x2、2x3、3x2。

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask with True at the given coords (in-bounds only)."""
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    """Crop empty rows/cols so the mask bbox matches the visible motif bbox."""
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_solid_block(grid: np.ndarray, comp: dict[str, object]) -> bool:
    """Return whether the component is one solid 5x5 monochrome destination block."""
    bbox = comp["bbox"]
    if bbox_h(bbox) != 5 or bbox_w(bbox) != 5:
        return False
    color = int(comp["color"])
    return bool((crop_bbox(grid, bbox) == int(color)).all())


def _carve_tile(color: int, bg: int, mask: np.ndarray) -> np.ndarray:
    """Center one motif mask inside a 5x5 block and carve its cells to background."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
    mh, mw = map(int, mask.shape)
    off_r = (DST_BLOCK_SIZE - mh) // 2
    off_c = (DST_BLOCK_SIZE - mw) // 2
    for rr, cc in np.argwhere(mask):
        tile[int(off_r + rr), int(off_c + cc)] = int(bg)
    return tile


def _column_major_block_order(blocks: list[dict[str, int]]) -> list[dict[str, int]]:
    """Concrete 'left-to-right' order: columns first, then top-to-bottom within a column."""
    return sorted(blocks, key=lambda b: (int(b["c0"]), int(b["r0"])))


def _row_major_indices(R: int, C: int) -> list[int]:
    """Row-major indices for an RxC rectangular block layout."""
    return [int(r * C + c) for r in range(R) for c in range(C)]


def _column_major_indices(R: int, C: int) -> list[int]:
    """Column-major indices for an RxC rectangular block layout."""
    return [int(r * C + c) for c in range(C) for r in range(R)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve area-ranked motifs into left-to-right-ranked blocks, then repack by motif area.

    Input structure:
    - The background is the most common color.
    - One region contains several solid 5x5 monochrome blocks arranged on a rectangular grid.
    - A separate region contains several small connected motif sprites in 3x3-like slots; their colors need not match any block.

    Rule:
    - Sort all non-block motif candidates by filled area from smallest to largest.
    - Sort the 5x5 blocks by horizontal position left-to-right, using top-to-bottom inside a shared column, and pair the i-th block with the i-th smallest motif.
    - Carve each paired motif into the center of its block, then sort the carved blocks by their matched motif area and place them into the compact output grid in row-major slot order with 1-cell gaps and a 1-cell border.

    Key insight: Size controls both stages, but not in the same coordinate order: motif size chooses which input block gets which hole, and then those same sizes decide how the carved blocks are rearranged in the packed output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks: list[dict[str, int]] = []
    motifs: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = map(int, bbox)
        if _is_solid_block(g, comp):
            blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})
            continue

        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        motifs.append(
            {
                "area": int(comp["area"]),
                "bbox_area": int(h * w),
                "r0": int(r0),
                "c0": int(c0),
                "mask": mask_from_coords(rel_coords, (h, w)),
            }
        )

    assert not (not blocks)
    assert len(motifs) >= len(blocks)

    blocks_for_match = _column_major_block_order(blocks)
    motifs_sorted = sorted(
        motifs,
        key=lambda m: (int(m["area"]), int(m["bbox_area"]), int(m["r0"]), int(m["c0"])),
    )

    carved_blocks: list[dict[str, object]] = []
    for block, motif in zip(blocks_for_match, motifs_sorted):
        carved_blocks.append(
            {
                "r0": int(block["r0"]),
                "c0": int(block["c0"]),
                "area": int(motif["area"]),
                "tile": _carve_tile(
                    int(block["col"]),
                    int(bg),
                    np.array(motif["mask"], dtype=bool),
                ),
            }
        )

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    R = len(rs)
    C = len(cs)
    out_h = int(2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP))
    out_w = int(2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP))
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
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = np.array(item["tile"], dtype=int)

    return out


# --------------- generator (combo: size-order match + size-order output) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif slot geometry and shape constraints.
MOTIF_SLOT_SIZE = 3
MOTIF_STEP = 4
MATCHED_AREAS = (3, 4, 5, 6, 7, 8)
DISTRACTOR_AREA = 9
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
CANVAS_RETRIES_PER_LAYOUT = 40


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute connected motif masks, stored in their trimmed visible bbox."""
    areas = list(range(min(MATCHED_AREAS), DISTRACTOR_AREA + 1))
    masks: dict[int, list[np.ndarray]] = {area: [] for area in areas}
    seen: dict[int, set[tuple[tuple[int, ...], ...]]] = {area: set() for area in areas}
    for bits in itertools.product([False, True], repeat=MOTIF_SLOT_SIZE * MOTIF_SLOT_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MOTIF_SLOT_SIZE, MOTIF_SLOT_SIZE)
        area = int(mask.sum())
        if area not in masks:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        trimmed = _trim_mask(mask)
        key = tuple(tuple(int(v) for v in row) for row in trimmed.astype(int))
        if key in seen[area]:
            continue
        seen[area].add(key)
        masks[area].append(trimmed)
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
                distractors = int(filled - count)
                if distractors < 0:
                    continue
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (count <= capacity <= 2 * count):
                        continue
                    if not (filled <= capacity and capacity - filled <= MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SLOT_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SLOT_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, filled))
    return out


def _sample_canvas_and_config_for_layout(
    target_R: int, target_C: int
) -> tuple[int, int, tuple[int, int, int, int, int]] | None:
    """Sample a canvas that supports the requested block layout."""
    for _ in range(CANVAS_RETRIES_PER_LAYOUT):
        H = random.randint(*CANVAS_HEIGHT_RANGE)
        W = random.randint(*CANVAS_WIDTH_RANGE)
        configs = [
            cfg for cfg in _configs(H, W) if (int(cfg[0]), int(cfg[1])) == (target_R, target_C)
        ]
        if configs:
            return H, W, random.choice(configs)
    return None


def _sample_mask_for_area(area: int) -> np.ndarray:
    """Sample one precomputed motif mask of the requested area."""
    return random.choice(MASKS_BY_AREA[int(area)]).copy()


def _choose_colors(bg: int, count: int) -> tuple[list[int], list[int]]:
    """Pick distinct block colors, then a small disjoint motif-color pool with reuse."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    remaining = palette[count:]
    pool_size = random.randint(2, min(3, len(remaining)))
    motif_pool = remaining[:pool_size]
    return block_colors, motif_pool


def _sample_matched_specs(count: int) -> list[dict[str, object]]:
    """Pick one unique-area motif for each block, ordered by increasing area."""
    areas = sorted(random.sample(list(MATCHED_AREAS), int(count)))
    return [{"area": int(area), "mask": _sample_mask_for_area(int(area))} for area in areas]


def _build_block_specs(
    R: int,
    C: int,
    block_colors: list[int],
    matched_specs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Assign ascending motif areas to blocks in column-major input order."""
    specs: list[dict[str, object]] = [{"color": int(color)} for color in block_colors]
    for idx, matched in zip(_column_major_indices(R, C), matched_specs):
        specs[int(idx)]["area"] = int(matched["area"])
        specs[int(idx)]["mask"] = np.array(matched["mask"], dtype=bool)
    return specs


def _paint_motif(grid: np.ndarray, slot_r: int, slot_c: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif centered inside its 3x3 slot."""
    h, w = map(int, mask.shape)
    off_r = (MOTIF_SLOT_SIZE - h) // 2
    off_c = (MOTIF_SLOT_SIZE - w) // 2
    patch = np.full((h, w), int(color), dtype=int)
    paste(grid, patch, int(slot_r + off_r), int(slot_c + off_c), mask=mask)


def _ordered_specs(block_specs: list[dict[str, object]], *, sort_by_area: bool) -> list[dict[str, object]]:
    """Return block specs either in spatial order or area-sorted output order."""
    if not sort_by_area:
        return list(block_specs)

    indexed = list(enumerate(block_specs))
    indexed.sort(
        key=lambda pair: (
            int(pair[1]["area"]),
            int(pair[0]),
        )
    )
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

    for idx, spec in enumerate(_ordered_specs(block_specs, sort_by_area=sort_by_area)):
        i = idx // C
        j = idx % C
        tile = _carve_tile(int(spec["color"]), int(bg), np.array(spec["mask"], dtype=bool))
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
    block_specs: list[dict[str, object]],
) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for idx, spec in enumerate(block_specs):
        i = idx // C
        j = idx % C
        block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(spec["color"]), dtype=int)
        paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    matched_specs: list[dict[str, object]],
    distractor_count: int,
    motif_pool: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Build shuffled visible motifs with repeated but non-degenerate motif colors."""
    masks: list[np.ndarray] = []
    for spec in matched_specs:
        masks.append(np.array(spec["mask"], dtype=bool))
    for _ in range(int(distractor_count)):
        masks.append(_sample_mask_for_area(DISTRACTOR_AREA))

    if len(motif_pool) == 1:
        colors = [int(motif_pool[0])] * len(masks)
    else:
        colors = [int(motif_pool[idx % len(motif_pool)]) for idx in range(len(masks))]
        random.shuffle(colors)

    items = list(zip(colors, masks))
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    matched_specs: list[dict[str, object]],
    distractor_count: int,
    motif_pool: list[int],
) -> None:
    """Randomly assign matched and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    for (sr, sc), (color, mask) in zip(slots, _slot_items(matched_specs, distractor_count, motif_pool)):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_specs: list[dict[str, object]],
    matched_specs: list[dict[str, object]],
    distractor_count: int,
    motif_pool: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_specs)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, matched_specs, distractor_count, motif_pool)
    return inp


def _row_major_area_sequence(block_specs: list[dict[str, object]]) -> list[int]:
    """Return matched motif areas in the spatial row-major order of the input block grid."""
    return [int(spec["area"]) for spec in block_specs]


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
    count = random.choice(sorted(DST_LAYOUTS))
    target_R, target_C = random.choice(DST_LAYOUTS[int(count)])
    canvas_and_config = _sample_canvas_and_config_for_layout(int(target_R), int(target_C))
    if canvas_and_config is None:
        return None

    H, W, config = canvas_and_config
    bg = random.randrange(COLOR_COUNT)
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SLOT_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SLOT_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    distractor_count = int(filled - count)
    block_colors, motif_pool = _choose_colors(bg, count)
    matched_specs = _sample_matched_specs(count)
    block_specs = _build_block_specs(R, C, block_colors, matched_specs)

    row_major_areas = _row_major_area_sequence(block_specs)
    if _inversion_count(row_major_areas) == 0:
        return None

    expected = _render_expected(bg, R, C, block_specs, sort_by_area=True)
    spatial_expected = _render_expected(bg, R, C, block_specs, sort_by_area=False)
    if np.array_equal(expected, spatial_expected):
        return None

    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_specs,
        matched_specs,
        distractor_count,
        motif_pool,
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

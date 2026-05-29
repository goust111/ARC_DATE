# Variant of 58f5dbd5
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: motif精灵与5x5方块通过相同颜色匹配（s.color == d.color）... → 大小排序匹配
#
# Design analysis
# variant_spec:
#   core_change:
#     - 原题用“同色”把 motif 对到 5x5 block；这里改成“按面积排序”的一对一映射。
#     - block 不再从颜色得到配对线索；motif 颜色被故意做成与 block 颜色无关。
#   concretization:
#     - 输入仍然有一个 destination 区：若干 5x5 纯色方块，按规则网格摆放。
#     - 输入仍然有一个 motif 区：若干小连通精灵，放在 3x3 slot 中，slot 之间留 1 格。
#     - 每个有效 motif 的“大小”定义为其前景像素数 area；solver 只看 area，不看颜色。
#     - block 的排序具体化为“阅读顺序”：先上排后下排、每排从左到右。
#       这是对“从左到右”的必要落地：原题允许多行 block，若只说 x 排序会出现同列并列，
#       需要一个可见且稳定的次级规则。阅读顺序比列优先排序更容易从 2-4 个样例中看出来。
#   behavior_trace:
#     - 例：4 个 block 颜色依次为 [红, 黄 / 蓝, 绿]，motif area 为 [3, 5, 7, 8]，motif 颜色全都不是这些 block 颜色。
#     - 原题逻辑：会尝试找“红 motif 对红 block、黄 motif 对黄 block ...”。
#     - 变体逻辑：先把 4 个 motif 按 area 升序排成 [3, 5, 7, 8]，再按 block 阅读顺序配到
#       [左上, 右上, 左下, 右下]；输出中四个 block 的洞型只由 area 排名决定。
#     - 差异在于：颜色从“决定配对”降为“纯视觉噪声”，决定输出的是 motif 的面积顺序。
#   complexity_calibration:
#     - 若继续使用原题 area 4..8，则最多只有 5 个档位，无法无歧义覆盖 6 个 block。
#     - 因此把 motif area 扩成 3..9，并允许有效 motif 的实际 bbox 小于 3x3。
#       这样 3,4,5,6,7,8 六个面积值足以覆盖 count=6 的情况，避免 tie。
#     - distractor 保留，但统一做成更大的 area=9；solver 只取最小的 N 个 motif，
#       所以多余 motif 不会混入排序前缀。
#   implicit_assumptions_review:
#     - 原 main 的 Step 2 逐个 block color 找“最小同色组件”。这依赖“每个 block 都有唯一同色 motif”。
#       在本变体中该前提失效，若保留这段代码会完全找不到 motif 或错误忽略同 area 异色 motif。
#     - 原 generator 的 `_choose_palette` / `_slot_items` 也依赖 motif 颜色唯一且与 block 同色；
#       这里改成 block 颜色独立、motif 颜色从剩余颜色池重复采样，主动打破颜色线索。
#     - 原 `_render_expected` 假设所有 motif 都是完整 3x3 mask，固定刻在 block 的中心 3x3 子区。
#       现在 motif bbox 可小于 3x3，所以期望输出必须按真实 bbox 居中雕刻，不能再硬编码切片。
#   placement_legality:
#     - block 区和 motif 区仍按原题分区放置，保证至少沿一个轴完全分离。
#     - motif 仍占用离散 slot；即使颜色复用，slot 间距 4 也保证组件不会因 8 邻接意外连成一片。
#     - 有效 motif area 取 3..8 且互不重复；distractor 统一 area=9，所以 solver 的前 N 个面积严格对应有效 motif。
#   visual_observability:
#     - block 颜色与 motif 颜色彻底脱钩，而且 motif 颜色会重复，解题者无法用“颜色相同”解释输出。
#     - 输出中的洞型沿 block 阅读顺序从小到大变化，输入里 motif 也能直接肉眼比较大小，规则可见。
#     - 去掉 3x1 的纯竖排 block 布局，因为那会弱化“从左到右”的信号。
#   cleanliness:
#     - 仍保持两大区域、固定间距、单层对象，不引入额外标记层。
#     - distractor 只比有效 motif 更大，不与 block 叠加，也不参与输出，视觉上容易当作“多余候选”。
#   new_solver_assumptions_checked_against_generator:
#     - solver 假设至少有一个 solid 5x5 block：generator 保证。
#     - solver 假设非 block 组件数 >= block 数：generator 总是放入 count 个有效 motif。
#     - solver 假设有效 motif 是面积最小的 count 个组件：generator 令有效 motif area∈{3..8} 且 distractor area=9。
#     - solver 假设 block 阅读顺序稳定：generator 仍把 block 放在规则网格上，因此按 (r0,c0) 排序稳定。

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


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    """Crop away empty rows/cols so the mask bbox matches the visible component bbox."""
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_solid_block(comp: dict[str, object], grid: np.ndarray) -> bool:
    """Return whether a component is a solid 5x5 monochrome destination block."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    patch = crop_bbox(grid, b)
    return bool((patch == int(col)).all())


def _carve_tile(bg: int, color: int, mask: np.ndarray) -> np.ndarray:
    """Center a motif mask inside a 5x5 block and carve its cells to background."""
    tile = np.full((5, 5), int(color), dtype=int)
    mh, mw = map(int, mask.shape)
    off_r = (5 - mh) // 2
    off_c = (5 - mw) // 2
    for rr, cc in np.argwhere(mask):
        tile[int(off_r + rr), int(off_c + cc)] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve area-ranked motifs into solid blocks and pack them into a compact grid.

    Input structure:
    - bg is the majority color.
    - Several solid 5x5 monochrome blocks occupy one separated region.
    - Elsewhere, several small monochrome motif sprites occupy slot-like positions; their colors need not match any block.

    Rule:
    - Read the 5x5 blocks in grid reading order (top-to-bottom, left-to-right).
    - Collect every non-block component as a candidate motif, sort those motifs by filled area from smallest to largest, and pair the first N motifs with the N blocks.
    - Center each paired motif inside its block, carve those motif cells to bg, then pack the carved 5x5 blocks with 1-cell gaps and a 1-cell outer border.

    Key insight: The output still comes from carving stencil-shaped holes, but the stencil-to-block correspondence is determined by size rank rather than color identity.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    blocks: list[dict[str, int]] = []
    motifs: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        r0, c0, r1, c1 = map(int, b)
        color = int(comp["color"])
        if _is_solid_block(comp, g):
            blocks.append({"col": int(color), "r0": int(r0), "c0": int(c0)})
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

    blocks_for_match = sorted(blocks, key=lambda b: (int(b["r0"]), int(b["c0"])))
    motifs_sorted = sorted(
        motifs,
        key=lambda m: (int(m["area"]), int(m["bbox_area"]), int(m["r0"]), int(m["c0"])),
    )

    carved: dict[tuple[int, int], np.ndarray] = {}
    for block, motif in zip(blocks_for_match, motifs_sorted):
        carved[(int(block["r0"]), int(block["c0"]))] = _carve_tile(
            bg, int(block["col"]), np.array(motif["mask"], dtype=bool)
        )

    # Pack carved 5x5 tiles into a compact grid preserving the original row/column layout.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        rr = int(1 + r_index[int(r0)] * step)
        cc = int(1 + c_index[int(c0)] * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x, size-order variant) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3)],
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


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute connected motif masks, stored in their trimmed visible bbox."""
    areas = list(range(min(MATCHED_AREAS), DISTRACTOR_AREA + 1))
    masks: dict[int, list[np.ndarray]] = {area: [] for area in areas}
    seen: dict[int, set[tuple[tuple[int, ...], ...]]] = {area: set() for area in areas}
    for bits in itertools.product([False, True], repeat=MOTIF_SLOT_SIZE * MOTIF_SLOT_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SLOT_SIZE, MOTIF_SLOT_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not is_contiguous(m.astype(int), background=0, connectivity=8):
            continue
        trimmed = _trim_mask(m)
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


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving the original two-stage count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None

    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


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


def _paint_motif(grid: np.ndarray, slot_r: int, slot_c: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif centered inside its 3x3 slot."""
    h, w = map(int, mask.shape)
    off_r = (MOTIF_SLOT_SIZE - h) // 2
    off_c = (MOTIF_SLOT_SIZE - w) // 2
    patch = np.full((h, w), int(color), dtype=int)
    paste(grid, patch, int(slot_r + off_r), int(slot_c + off_c), mask=mask)


def _render_expected(
    bg: int, R: int, C: int, block_colors: list[int], matched_specs: list[dict[str, object]]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            color = int(block_colors[idx])
            mask = np.array(matched_specs[idx]["mask"], dtype=bool)
            tile = _carve_tile(bg, color, mask)
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
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    matched_specs: list[dict[str, object]],
    distractor_count: int,
    motif_pool: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Build shuffled visible motifs: matched small ones plus oversized distractors."""
    items: list[tuple[int, np.ndarray]] = []
    for spec in matched_specs:
        items.append((int(random.choice(motif_pool)), np.array(spec["mask"], dtype=bool)))
    for _ in range(int(distractor_count)):
        items.append((int(random.choice(motif_pool)), _sample_mask_for_area(DISTRACTOR_AREA)))
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
    slot_items = _slot_items(matched_specs, distractor_count, motif_pool)

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
    matched_specs: list[dict[str, object]],
    distractor_count: int,
    motif_pool: list[int],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, matched_specs, distractor_count, motif_pool)
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
    motif_h = _spanned_size(Mr, MOTIF_SLOT_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SLOT_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    distractor_count = int(filled - count)
    block_colors, motif_pool = _choose_colors(bg, count)
    matched_specs = _sample_matched_specs(count)

    expected = _render_expected(bg, R, C, block_colors, matched_specs)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D1 size-order input")

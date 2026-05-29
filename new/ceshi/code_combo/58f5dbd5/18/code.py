# Combo variant of 58f5dbd5 (index 18)
# combo_spec: D1/1+D3/3
# Name: 边缘颜色匹配+对角标记对motif

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# yaml:
#   combo_intent:
#     summary: >
#       保留原题“把 motif 居中雕刻进 5x5 块并按原顺序紧凑打包”的主链条，
#       同时把“匹配键”改成边框色、把“motif 形状”改成非连通的对角双点。
#   concrete_pixelization:
#     dst_block:
#       shape: 5x5 framed block
#       border: 16 个边框像素统一为 edge_color
#       interior: 中央 3x3 统一为 fill_color
#       guardrail: edge_color != fill_color
#     src_motif:
#       shape: >
#         每个 motif 只有 2 个同色像素，位于其最小外接框的两个对角。
#         允许的 bbox 为 2x3、3x2、3x3；排除 2x2 以避免 8 连通退化。
#       color_rule: motif 颜色匹配 block 的 edge_color，而不是 fill_color
#   behavior_trace:
#     small_example:
#       block_A: edge=2, fill=4
#       block_B: edge=4, fill=7
#       block_C: edge=7, fill=2
#       motif_color_2: 两个像素位于 3x3 bbox 的左上/右下角
#       original_step1: 旧逻辑找“整体同色的 5x5 块”，这里会直接失效
#       original_step2: 旧逻辑把 motif 当 8 连通小组件；对角双点不是单个组件，也会失效
#       new_step1: 扫描 5x5 patch，识别 uniform border + uniform interior 的 framed block
#       new_step2: 先整体排除所有块占用的 5x5 区域，再对每个 edge_color 收集块外剩余同色像素
#       new_step3: 这两个剩余像素恢复成最小 bbox 内的对角 mask，居中雕刻进对应 framed block
#       difference: >
#         匹配锚点从“整块颜色”切到“边框颜色”，
#         motif 表示方式从“连通精灵”切到“非连通对角双点”。
#   complexity_calibration:
#     preserved:
#       - 目标块仍是规则 5x5 网格，输出仍保持原始行列顺序
#       - motif 仍通过颜色建立一一对应，并被居中作为 stencil 使用
#       - 仍保留 distractor motifs、空槽、区域分离，复杂度与原题同量级
#     strengthened:
#       - fill_color 取 edge_color palette 的非零循环位移，迫使解题者忽略内部填充色
#       - matching motifs 至少出现两种对角形态，并要求“按 fill 错配”时输出不同，防止退化
#   implicit_assumption_audit:
#     broken_by_D1:
#       - “5x5 块整体同色”不成立，必须改为 framed-block 识别
#       - “块色 == motif 色”不成立，必须读 edge_color
#     broken_by_D3:
#       - “motif 是一个 8 连通组件”不成立，不能再从 connected components 直接取 mask
#       - “motif 面积 4..8”不成立，新的 motif 面积固定为 2
#     combo_conflict_and_fix:
#       issue: >
#         fill_color 会复用别的块的 edge_color，因此如果只按颜色收集块外像素而不先排除整块区域，
#         会把别的块的 3x3 interior 误当成 motif。
#       fix: >
#         先识别全部 framed block，再把所有块的 5x5 占位整体从 motif 搜索空间中扣除。
#   placement_legality:
#     - 块区域仍按原题放在一个 RxC 规则网格里，top-left 步长 6，彼此不接触
#     - motif 仍放在 3x3 槽位网格里，槽位步长 4，不重叠、不越界
#     - 块区域与 motif 区域至少在一个轴上分离，避免块像素与 motif 像素混杂
#   visual_observability:
#     - framed block 明确展示“边框提示色”和“内部干扰色”的分层
#     - motif 只有两个对角点，视觉上与原题的小连通精灵显著不同
#     - fill_color 使用 edge palette 的循环位移，所以“motif 颜色也出现在别的块内部”会反复出现
#     - 不依赖颜色编号、面积排序以外的人类不可见因素；关键线索都在可见边框和像素位置上
#   cleanliness:
#     - 输入只含两个主要区域：framed blocks 区和 diagonal motifs 区
#     - 输出只展示被雕刻后的 blocks，不叠加额外标记
#   post_edit_assumptions_checked_against_generator:
#     - 每个 edge_color 在块外恰好对应 2 个 motif 像素
#     - 这 2 个像素位于同一最小 bbox 的两个对角，bbox 属于 {2x3, 3x2, 3x3}
#     - distractor 颜色不与任何 edge_color 重合
#     - 至少存在一种块的 motif 在“按 fill 错配”时会改变输出


def mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given in-bounds coords."""
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        if 0 <= int(r) < H and 0 <= int(c) < W:
            mask[int(r), int(c)] = True
    return mask


# 5x5 destination block structure for the edge-color + diagonal-pair combo.
DST_BLOCK_SIZE = 5
INNER_SIZE = 3
MOTIF_OFFSET_IN_BLOCK = 1
INNER_SLICE = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + INNER_SIZE)
RING_MASK = np.ones((DST_BLOCK_SIZE, DST_BLOCK_SIZE), dtype=bool)
RING_MASK[1:-1, 1:-1] = False

# Fixed diagonal-pair motif vocabulary.
MOTIF_SLOT_SIZE = 3
DIAGONAL_MASKS = [
    mask_from_coords([(0, 0), (1, 2)], (2, 3)),
    mask_from_coords([(0, 2), (1, 0)], (2, 3)),
    mask_from_coords([(0, 0), (2, 1)], (3, 2)),
    mask_from_coords([(0, 1), (2, 0)], (3, 2)),
    mask_from_coords([(0, 0), (2, 2)], (3, 3)),
    mask_from_coords([(0, 2), (2, 0)], (3, 3)),
]


def build_block(edge_color: int, fill_color: int) -> np.ndarray:
    """Construct one 5x5 framed destination block."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(edge_color), dtype=int)
    tile[INNER_SLICE, INNER_SLICE] = int(fill_color)
    return tile


def block_signature(patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    """Return (edge_color, fill_color) iff patch is a valid framed block."""
    if patch.shape != (DST_BLOCK_SIZE, DST_BLOCK_SIZE):
        return None

    edge_vals = patch[RING_MASK]
    fill_vals = patch[INNER_SLICE, INNER_SLICE]
    if len(set(map(int, edge_vals.flatten()))) != 1:
        return None
    if len(set(map(int, fill_vals.flatten()))) != 1:
        return None

    edge_color = int(edge_vals.flat[0])
    fill_color = int(fill_vals.flat[0])
    if edge_color == int(bg) or fill_color == int(bg):
        return None
    if edge_color == fill_color:
        return None
    return edge_color, fill_color


def diagonal_pair_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray | None:
    """Recover a diagonal-pair motif mask from two absolute coords."""
    if len(coords) != 2:
        return None

    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    if not (2 <= h <= MOTIF_SLOT_SIZE and 2 <= w <= MOTIF_SLOT_SIZE):
        return None
    if max(h, w) < MOTIF_SLOT_SIZE:
        return None  # reject 2x2, which would be 8-connected diagonally

    rel = {(int(r) - r0, int(c) - c0) for r, c in coords}
    main_diag = {(0, 0), (h - 1, w - 1)}
    anti_diag = {(0, w - 1), (h - 1, 0)}
    if rel != main_diag and rel != anti_diag:
        return None
    return mask_from_coords(sorted(rel), (h, w))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve edge-color-matched diagonal-pair motifs into framed blocks and pack them tightly.

    Input structure:
    - bg is the most common color.
    - Several 5x5 destination blocks have a uniform 1-cell border color and a different uniform 3x3 interior color.
    - Elsewhere, each relevant border color appears again only as a tiny two-pixel diagonal motif, plus optional distractor motifs of unrelated colors.

    Rule:
    - Detect each 5x5 framed block and treat its border color as the matching key; the interior color is only a distractor cue.
    - For every block, find the two same-colored motif pixels outside all 5x5 blocks, rebuild their minimal diagonal bbox, center that mask in the block, and turn those cells into bg.
    - Pack the carved 5x5 blocks into an output grid that preserves the original row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight: The actionable stencil is not a connected sprite anymore; it is a disconnected diagonal pixel pair keyed by the block's edge marker.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    # STEP 1) Identify destination blocks: 5x5 patches with a uniform border color
    # and a different uniform interior color.
    blocks: list[dict[str, int]] = []
    for r0 in range(0, int(H - DST_BLOCK_SIZE + 1)):
        for c0 in range(0, int(W - DST_BLOCK_SIZE + 1)):
            patch = g[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE]
            sig = block_signature(patch, bg)
            if sig is None:
                continue
            edge_color, fill_color = sig
            blocks.append(
                {
                    "edge": int(edge_color),
                    "fill": int(fill_color),
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )

    assert blocks

    block_mask = np.zeros((H, W), dtype=bool)
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        block_mask[r0 : r0 + DST_BLOCK_SIZE, c0 : c0 + DST_BLOCK_SIZE] = True

    # STEP 2) For each border color, identify its motif:
    # exactly two same-colored pixels that remain after removing all block cells.
    motifs: dict[int, np.ndarray] = {}
    for edge_color in {int(b["edge"]) for b in blocks}:
        coords = [
            (int(r), int(c))
            for r, c in np.argwhere((g == int(edge_color)) & (~block_mask))
        ]
        mm = diagonal_pair_mask_from_coords(coords)
        if mm is not None:
            motifs[int(edge_color)] = mm

    # STEP 3) Carve each 5x5 tile:
    # reconstruct the framed block, then center the matching diagonal-pair mask and turn those cells to bg.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        edge_color = int(b["edge"])
        fill_color = int(b["fill"])
        tile = build_block(edge_color, fill_color)
        mm = motifs.get(int(edge_color))
        if mm is not None:
            mh, mw = map(int, mm.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(mm):
                tile[int(off_r + rr), int(off_c + cc)] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * DST_BLOCK_SIZE + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * DST_BLOCK_SIZE + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        rr = int(1 + r_index[r0] * step)
        cc = int(1 + c_index[c0] * step)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[(r0, c0)]

    return out


# --------------- generator (combo o5x) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination block geometry.
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}

# Motif slot geometry.
MOTIF_STEP = 4
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
GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


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


def _random_diagonal_mask() -> np.ndarray:
    """Sample one diagonal-pair motif mask."""
    return random.choice(DIAGONAL_MASKS).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick edge colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    edge_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return edge_colors, distractor_colors


def _choose_fill_colors(edge_colors: list[int]) -> tuple[list[int], int]:
    """Reuse the edge palette as interior fills via a non-zero cyclic shift."""
    count = len(edge_colors)
    shift = random.randint(1, count - 1)
    fill_colors = [int(edge_colors[(i + shift) % count]) for i in range(count)]
    return fill_colors, shift


def _sample_matching_motif_masks(edge_colors: list[int], shift: int) -> dict[int, np.ndarray]:
    """Assign motif masks while guaranteeing that fill-based mispairing changes the output."""
    count = len(edge_colors)
    for _ in range(80):
        masks = {int(color): _random_diagonal_mask() for color in edge_colors}
        ordered = [masks[int(color)] for color in edge_colors]
        shifted = [masks[int(edge_colors[(i + shift) % count])] for i in range(count)]
        if any(not np.array_equal(a, b) for a, b in zip(ordered, shifted)):
            return masks
    raise RuntimeError("failed to sample non-degenerate diagonal motifs")


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one diagonal motif into a 3x3 slot using its own minimal bbox."""
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    edge_colors: list[int],
    fill_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    *,
    use_fill_match: bool = False,
) -> np.ndarray:
    """Render the oracle output for either edge-match or fill-match pairing."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            edge_color = int(edge_colors[idx])
            fill_color = int(fill_colors[idx])
            tile = build_block(edge_color, fill_color)
            key_color = fill_color if use_fill_match else edge_color
            mask = motif_masks[int(key_color)]
            mh, mw = map(int, mask.shape)
            off_r = (DST_BLOCK_SIZE - mh) // 2
            off_c = (DST_BLOCK_SIZE - mw) // 2
            for rr, cc in np.argwhere(mask):
                tile[int(off_r + rr), int(off_c + cc)] = int(bg)
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
    edge_colors: list[int],
    fill_colors: list[int],
) -> None:
    """Paint the framed 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            idx = i * C + j
            block = build_block(int(edge_colors[idx]), int(fill_colors[idx]))
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _slot_items(
    edge_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[int(color)]) for color in edge_colors
    ]
    items.extend((int(color), _random_diagonal_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    edge_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    for (sr, sc), (color, mask) in zip(slots, _slot_items(edge_colors, distractor_colors, motif_masks)):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    edge_colors: list[int],
    fill_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, edge_colors, fill_colors)
    _paint_motif_slots(
        inp,
        motif_r,
        motif_c,
        Mr,
        Mc,
        edge_colors,
        distractor_colors,
        motif_masks,
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the oracle or fusion guards disagree."""
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
    edge_colors, distractor_colors = _choose_palette(bg, count, filled)
    fill_colors, shift = _choose_fill_colors(edge_colors)
    motif_masks = _sample_matching_motif_masks(edge_colors, shift)

    expected = _render_expected(bg, R, C, edge_colors, fill_colors, motif_masks)
    wrong_fill_expected = _render_expected(
        bg,
        R,
        C,
        edge_colors,
        fill_colors,
        motif_masks,
        use_fill_match=True,
    )
    if np.array_equal(expected, wrong_fill_expected):
        return None

    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        edge_colors,
        fill_colors,
        distractor_colors,
        motif_masks,
        placement,
    )

    if Counter(inp.flatten().tolist()).most_common(1)[0][0] != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

# Variant of 58f5dbd5
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: motif精灵与5x5方块通过相同颜色匹配（s.color == d.color）... → 边缘颜色匹配

from __future__ import annotations

import itertools
import random
import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# yaml:
#   reference_task_247ef758:
#     takeaway: >
#       参考任务把“颜色提示”放在对象边缘，而不是放在对象主体内部。这里沿用同一复杂度：
#       颜色仍然是可见、局部、离散的匹配线索，但只改变匹配锚点，不改变整体搬运/打洞流程。
#   concrete_input_change:
#     original_block: 5x5 单色实心块，motif 颜色直接等于块颜色。
#     variant_block: 5x5 双层块，16 个边框像素同色(edge_color)，中心 3x3 同色(fill_color)。
#     motif_color: 必须匹配 edge_color，而不是 fill_color。
#     visual_guardrail: >
#       每个块都满足 edge_color != fill_color；并且 fill_color 取自“别的块的 edge_color”的循环置换，
#       这样解题者不能靠内部填充色做一一对应。
#   behavior_trace:
#     small_example:
#       block_A: edge=2, fill=4
#       block_B: edge=4, fill=7
#       motif_red: color=2
#       old_logic: 会尝试找“颜色 2 的实心 5x5 块”，在变体里不存在，因此旧 solver 的识别前提失效。
#       new_logic: 先识别 5x5 framed block，再读取其 edge_color=2，把 color=2 的 motif 居中雕刻到该块。
#       difference: 匹配键从“整块颜色”切换成“边框标记色”。
#   complexity_calibration:
#     keep_same:
#       - motif 仍是 3x3 bbox 内的单色连通精灵
#       - motif 仍被居中放到 5x5 中央 3x3 并雕刻为背景
#       - 输出仍按原始块的行列顺序紧凑打包
#     changed_only_for_D1:
#       - 目标块从单色改为“边框色 + 填充色”
#       - motif 与块通过边框色关联
#   implicit_assumption_audit:
#     original_step1: 依赖“5x5 块整体同色”。变体后不成立，必须改为扫描 5x5 patch 并验证 ring/uniform interior。
#     original_step2: 依赖“块色 == motif 色”。变体后要改成“edge_color == motif 色”，并跳过 5x5 ring 组件。
#     original_step3: 依赖“tile 初始化为纯色”。变体后要先重建 framed tile，再在中央按 motif mask 打洞。
#     original_step4: 仅依赖块的空间顺序，与 D1 无关，可保持。
#   placement_legality:
#     - 块区域和 motif 区域仍按原题分离放置，避免重叠和误检。
#     - 块之间保持 1 像素空隙，因此 5x5 framed blocks 不会彼此粘连。
#     - motif 槽位保持 3x3 大小和 1 像素间距，因此 motif 组件不会互相连通。
#   visual_observability:
#     - 所有块都满足 edge_color != fill_color，避免退化回原题。
#     - fill_color 采用 edge palette 的循环移位，因此“motif 颜色等于某个块的内部色”会频繁出现，
#       迫使解题者区分边框与填充。
#     - 不使用颜色编号排序或隐藏规则；只有显式可见的边框颜色参与匹配。
#   cleanliness:
#     - 输入中只有两类关键区域：framed block 区和 motif 区，层次与原题一致。
#     - 输出只展示雕刻后的块，不叠加额外标记，避免画面杂乱。


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


# 5x5 destination block structure for the edge-color variant.
DST_BLOCK_SIZE = 5
INNER_SIZE = 3
MOTIF_OFFSET_IN_BLOCK = 1
RING_MASK = np.ones((DST_BLOCK_SIZE, DST_BLOCK_SIZE), dtype=bool)
RING_MASK[1:-1, 1:-1] = False
INNER_SLICE = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + INNER_SIZE)
RING_AREA = int(RING_MASK.sum())


def build_block(edge_color: int, fill_color: int) -> np.ndarray:
    """Construct one 5x5 framed destination block."""
    tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(edge_color), dtype=int)
    tile[INNER_SLICE, INNER_SLICE] = int(fill_color)
    return tile


def block_signature(patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    """Return (edge_color, fill_color) iff patch is a valid 5x5 framed block."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve edge-color-matched motifs into framed blocks and pack them into a compact grid.

    Input structure:
    - bg = most common color.
    - Several 5x5 non-bg blocks whose 1-cell border is a uniform marker color and whose inner 3x3 is a different uniform fill color.
    - Elsewhere, one small motif sprite per border-marker color, plus optional distractor motifs of unrelated colors.

    Rule:
    - Match each motif to the 5x5 block whose border marker shares the motif color; ignore the block's inner fill color.
    - Center the motif within the block and recolor those covered cells to bg, punching holes only inside the 3x3 interior.
    - Pack the carved blocks into an output grid that preserves their original row/column order with 1-cell gaps and a 1-cell outer border.

    Key insight: The small sprite is a stencil keyed by the block's edge marker, while the interior color is only a distractor cue.
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

    assert not (not blocks)

    # STEP 2) For each border color, identify its motif:
    # the smallest same-colored component that is NOT the 5x5 border ring of a block.
    ring_positions = {(int(b["edge"]), int(b["r0"]), int(b["c0"])) for b in blocks}
    motifs: dict[int, np.ndarray] = {}
    edge_colors = {int(b["edge"]) for b in blocks}
    for edge_color in edge_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(edge_color)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))

            if (
                h == DST_BLOCK_SIZE
                and w == DST_BLOCK_SIZE
                and (int(edge_color), int(r0), int(c0)) in ring_positions
                and int(comp["area"]) == RING_AREA
                and np.array_equal(m, RING_MASK)
            ):
                continue

            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(edge_color)] = candidates[0][4]

    # STEP 3) Carve each 5x5 tile:
    # reconstruct the framed block, then center the matching motif mask and turn those cells to bg.
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
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < DST_BLOCK_SIZE and 0 <= tc < DST_BLOCK_SIZE:
                    tile[tr, tc] = int(bg)
        carved[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack carved 5x5 tiles into a compact grid.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * DST_BLOCK_SIZE + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * DST_BLOCK_SIZE + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6  # 5 block + 1 gap
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + DST_BLOCK_SIZE, cc : cc + DST_BLOCK_SIZE] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x) ---------------

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
    """Pick edge colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    edge_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return edge_colors, distractor_colors


def _derive_fill_colors(edge_colors: list[int]) -> list[int]:
    """Reuse the edge-color palette as interior fills via a non-trivial cyclic shift."""
    count = len(edge_colors)
    shift = random.randint(1, count - 1)
    return [int(edge_colors[(i + shift) % count]) for i in range(count)]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    edge_colors: list[int],
    fill_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            idx = i * C + j
            edge_color = int(edge_colors[idx])
            fill_color = int(fill_colors[idx])
            tile = build_block(edge_color, fill_color)
            tile[INNER_SLICE, INNER_SLICE][motif_masks[edge_color]] = bg
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
            paste(
                grid,
                block,
                dst_r + i * DST_BLOCK_STEP,
                dst_c + j * DST_BLOCK_STEP,
            )


def _slot_items(
    edge_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [
        (int(color), motif_masks[color]) for color in edge_colors
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
    edge_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(edge_colors, distractor_colors, motif_masks)

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
        inp, motif_r, motif_c, Mr, Mc, edge_colors, distractor_colors, motif_masks
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
    edge_colors, distractor_colors = _choose_palette(bg, count, filled)
    fill_colors = _derive_fill_colors(edge_colors)
    motif_masks = {color: _random_mask() for color in edge_colors}

    expected = _render_expected(bg, R, C, edge_colors, fill_colors, motif_masks)
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

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 input")

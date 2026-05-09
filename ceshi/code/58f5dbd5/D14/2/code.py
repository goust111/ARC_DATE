# Variant of 58f5dbd5
# Dimension: D14 — 输出排列顺序 (grid_shape)
# Change: 按方块在输入网格中的行列位置排序（保持原始空间顺序）... → 按motif在输入中的位置排序

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis for the D14 variant
# -----------------------------------
# spec:
#   preserved:
#     - 输入仍由一组 5x5 纯色 block 和一组 3x3 单色 motif 组成。
#     - block 与 motif 仍通过“同色唯一匹配”配对。
#     - motif 仍作为雕刻模板，居中挖空对应颜色的 5x5 block。
#     - 输出仍是紧凑的 block 阵列，5x5 方块之间隔 1 格，外围 1 格边框。
#   changed_core:
#     - 原题第 4 步按 block 在输入中的二维位置放入输出。
#     - 本变体保持输出阵列尺寸不变，仍使用 block 原来的 R x C 形状，
#       但写入这些输出槽位的顺序改为“按匹配 motif 在输入中的扫描顺序”。
#     - 这样只改 D14 的“排列顺序”，不额外引入输出空洞、尺寸变化或新对象。
#   concrete_behavior:
#     - 例：输入中 block 行优先颜色序列是 [A, B, C, D]（2x2）。
#     - 若 matching motif 从左到右、从上到下出现的颜色序列是 [C, A, D, B]，
#       那输出左上、右上、左下、右下就依次放入 carved(C), carved(A), carved(D), carved(B)。
#     - 原题会输出 [A, B, C, D]；本变体输出 [C, A, D, B]。
#   complexity_calibration:
#     - 若 motif 顺序恰好和 block 顺序一致，样例会退化回原题。
#     - 因此生成器显式拒绝 identity permutation，保证每个样例里新旧规则至少有一次重排。
#     - 颜色仍是一一对应且肉眼可见，解题者能通过“同色 motif 的位置”追踪输出次序。
#   implicit_assumption_audit:
#     - 原 solver 第 4 步默认 block 的 r0/c0 同时决定输出中的行列索引。
#       该前提在变体中不成立，只能继续用于推断输出阵列的 R、C，而不能用于排序。
#     - 原 solver 第 2 步只保存 motif mask；变体还需要 motif 的位置，因此必须连同 r0/c0 一起保存。
#     - 原 generator 的 expected 直接用 block_colors 的原顺序渲染；变体中 expected 必须改用 motif 顺序。
#   placement_legality:
#     - block 区和 motif 区仍按原规则在至少一个轴上分离，所有对象不重叠、不越界。
#     - motif 仍落在 3x3 slot 网格中；matching motif 只改变“哪些颜色占据哪些 slot”，不会破坏合法摆放。
#   observability:
#     - 排序依据完全来自可见的 motif 空间位置，不依赖颜色 id 的数值大小。
#     - 颜色只用于配对，排序只看 top-left 扫描顺序，2-4 个样例即可看出输出重排来自 motif 区域。
#   cleanliness:
#     - 不增加任何叠加标记，也不改变 block/motif 的几何大小；画面仍保持原题的清爽分区。


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


def _non_identity_permutation(items: list[int]) -> list[int]:
    """Return a shuffled copy that differs from the input order."""
    out = list(items)
    if len(out) <= 1:
        return out
    for _ in range(20):
        random.shuffle(out)
        if out != items:
            return out
    return list(items[1:] + items[:1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into solid blocks, then reorder the carved tiles by motif position.

    Input structure:
    - The background is the most common color.
    - Several separate 5x5 solid squares appear, each in a unique non-background color.
    - Elsewhere, smaller same-colored motif sprites appear; extra distractor motifs may also be present.

    Rule:
    - For each block color, find the matching same-colored motif and center-carve that motif out of the 5x5 square using the background color.
    - Keep the compact output frame size implied by the original block lattice, but fill its slots in the top-to-bottom, left-to-right order of the matching motifs rather than the blocks.
    - Preserve the usual 1-cell gaps between packed tiles and the 1-cell outer border.

    Key insight:
    - Color still tells you which motif carves which block, but motif position now controls where that carved block lands in the packed output.
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
            blocks.append({"col": col, "r0": r0, "c0": c0})

    assert blocks

    # STEP 2) For each block color, identify its motif:
    # the smallest same-colored component that is NOT one of the 5x5 blocks.
    motifs: dict[int, dict[str, object]] = {}
    block_colors = {int(b["col"]) for b in blocks}
    for bcol in block_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[bcol],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == 5 and bbox_w(b) == 5:
                patch = crop_bbox(g, b)
                if bool((patch == bcol).all()):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            mask = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), r0, c0, mask))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            _area, _bbox_area, r0, c0, mask = candidates[0]
            motifs[bcol] = {"r0": r0, "c0": c0, "mask": mask}

    assert all(int(b["col"]) in motifs for b in blocks)

    # STEP 3) Carve each 5x5 tile:
    # center the motif mask within the 5x5 and repaint those mask cells to bg.
    carved: dict[int, np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), col, dtype=int)
        motif = motifs[col]
        mask = motif["mask"]
        assert isinstance(mask, np.ndarray)
        mh, mw = map(int, mask.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mask):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = bg
        carved[col] = tile

    # STEP 4) Pack carved 5x5 tiles into the original output frame size,
    # but order them by the matching motifs' positions.
    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    rows = len(rs)
    cols = len(cs)
    assert rows * cols == len(blocks)

    ordered_colors = [
        int(b["col"])
        for b in sorted(
            blocks,
            key=lambda b: (
                int(motifs[int(b["col"])]["r0"]),
                int(motifs[int(b["col"])]["c0"]),
            ),
        )
    ]

    out_h = int(1 + rows * 5 + (rows - 1) + 1)
    out_w = int(1 + cols * 5 + (cols - 1) + 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    step = 6
    for idx, col in enumerate(ordered_colors):
        i = idx // cols
        j = idx % cols
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = carved[col]

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


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, ordered_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output after reordering tiles by motif scan order."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    motif_slice = slice(MOTIF_OFFSET_IN_BLOCK, MOTIF_OFFSET_IN_BLOCK + MOTIF_SIZE)
    for idx, color in enumerate(ordered_colors):
        i = idx // C
        j = idx % C
        tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
        tile[motif_slice, motif_slice][motif_masks[int(color)]] = bg
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


def _plan_motif_payloads(
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> tuple[list[tuple[int, int, int, np.ndarray]], list[int]]:
    """Assign matching motifs to slots so the motif-driven order differs from block order."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    match_count = len(block_colors)
    matching_slots = sorted(random.sample(slots, match_count))
    ordered_colors = _non_identity_permutation(block_colors)

    payloads: list[tuple[int, int, int, np.ndarray]] = []
    used_slots = set[tuple[int, int]]()
    for (sr, sc), color in zip(matching_slots, ordered_colors):
        payloads.append((sr, sc, int(color), motif_masks[int(color)]))
        used_slots.add((sr, sc))

    remaining_slots = [slot for slot in slots if slot not in used_slots]
    random.shuffle(remaining_slots)
    for (sr, sc), color in zip(remaining_slots, distractor_colors):
        payloads.append((sr, sc, int(color), _random_mask()))

    random.shuffle(payloads)
    return payloads, ordered_colors


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    payloads: list[tuple[int, int, int, np.ndarray]],
) -> None:
    """Paint the chosen motif payloads into the visible motif slot grid."""
    for sr, sc, color, mask in payloads:
        _paint_motif(
            grid,
            motif_r + int(sr) * MOTIF_STEP,
            motif_c + int(sc) * MOTIF_STEP,
            int(color),
            mask,
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    placement: tuple[int, int, int, int],
    block_colors: list[int],
    motif_payloads: list[tuple[int, int, int, np.ndarray]],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, motif_payloads)
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
    motif_masks = {color: _random_mask() for color in block_colors}
    motif_payloads, ordered_colors = _plan_motif_payloads(
        Mr, Mc, block_colors, distractor_colors, motif_masks
    )

    expected = _render_expected(bg, R, C, ordered_colors, motif_masks)
    inp = _render_input(H, W, bg, R, C, placement, block_colors, motif_payloads)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 input")

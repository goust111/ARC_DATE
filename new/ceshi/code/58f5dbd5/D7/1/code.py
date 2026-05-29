# Variant of 58f5dbd5
# Dimension: D7 — 变换操作：雕刻方式 (transform)
# Change: 将motif覆盖区域的方块像素替换为背景色（减法雕刻/打洞）... → 颜色替换而非擦除

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis:
# spec_variant:
#   core_change:
#     - 原题把 motif 覆盖到 5x5 方块中央后，将覆盖像素改成背景色。
#     - 本变体保留“按颜色匹配 motif → 居中投影到 5x5 → 紧凑打包”的主流程，
#       只把最终像素操作改成“整块重染成 motif 的第二颜色”，这里固定为红色 2。
#   concrete_input:
#     - 目标对象仍是多个彼此分离的 5x5 纯色方块，颜色各不相同。
#     - motif 区域仍是 3x3 槽位网格，但每个 motif 从单色改成双色：
#       主色 = 与某个方块同色；第二色 = 固定红色标记；两色共同组成同一个连通精灵。
#     - distractor motif 也使用“自己的主色 + 同一红色标记”，因此视觉上会出现统一的红色提示。
#   behavior_trace:
#     - 原逻辑：块色 6 + 单色 motif mask -> 5x5 中央对应位置改成 bg。
#     - 新逻辑：块色 6 + 双色 motif（6/2） -> 取整个 motif 占用形状，5x5 中央对应位置全部改成 2。
#     - 差异：输出块内不再出现“孔洞/背景”，而是出现“原块色 + 红色”两色并存的图案。
#   complexity_calibration:
#     - 不引入按颜色 id 排序或多档规则；所有 motif 共享一个固定第二色，减少推断分支。
#     - 保留原来的 3-6 个块、6-8 个 motif、3x3 连通形状和紧凑输出，避免任务退化得过于简单。
#   implicit_assumption_audit:
#     - 原 solver 的 `find_components(... monochromatic=True, colors=[block_color])`
#       假设 motif 是单色；变体下失效，因此改为扫描非块组件并读取其双色集合。
#     - 原 solver 在 Step 3 里无条件 `tile[...] = bg`；这依赖“雕刻色恒为背景”。
#       变体下必须改为写入 motif 的第二色，否则会退回原题行为。
#     - 原 generator 假设 motif 只需要一个 mask；变体下还需显式生成第二色子集，
#       且保证该子集与主色一起构成同一 8 连通精灵。
#   placement_legality:
#     - 仍沿用原始分区放置：方块区域与 motif 区域在至少一个轴上分离，槽位间距不变。
#     - motif 仍限制在 3x3 bbox，块仍是 5x5，输出仍按 1 像素间隔与边框紧凑打包。
#   visual_observability:
#     - 红色会同时出现在输入 motif 和输出块内，2-3 个样例即可看出“不是挖空，而是染成红色”。
#     - 输出里每个块只含两种颜色：原块色与红色，结构清晰，不会与背景混淆。
#   new_solver_assumptions:
#     - 每个有效 motif 组件恰有两种非背景色，其中一种出现在某个 5x5 块中，另一种是重染色。
#     - 每个块色只对应一个 motif；generator 通过唯一主色和唯一匹配槽位保证这一点。


MARKER_COLOR = 2


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
    Recolor color-matched motifs onto solid blocks and pack them into a compact grid.

    Input structure:
    - The background is the most common color.
    - Several solid 5×5 monochrome blocks appear, each using a different non-background color.
    - Elsewhere, small 3×3 motif sprites contain that same block color plus one shared accent color.

    Rule:
    - Match each 5×5 block with the motif sprite that contains the block's color.
    - Center the motif footprint inside the block and repaint every occupied motif cell to the motif's accent color instead of erasing it.
    - Pack the recolored blocks in their original row/column order with 1-cell gaps and a 1-cell outer border.

    Key insight: The sprite still behaves like a stencil, but now its occupied cells define a recolor mask whose fill comes from the sprite's secondary accent color, not from the background.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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

    # STEP 2) Collect motif sprites:
    # each motif is a small multicolor component containing exactly one block color plus one accent color.
    motifs: dict[int, tuple[np.ndarray, int]] = {}
    block_colors = {int(b["col"]) for b in blocks}
    multi_comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in multi_comps:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue

        r0, c0, r1, c1 = map(int, b)
        patch = crop_bbox(g, b)
        rel_mask = patch != int(bg)
        if not rel_mask.any():
            continue

        colors = {int(v) for v in patch[rel_mask]}
        matching = colors & block_colors
        if len(colors) != 2 or len(matching) != 1:
            continue

        primary = int(next(iter(matching)))
        accent = int(next(iter(colors - {primary})))
        motifs[primary] = (rel_mask.copy(), accent)

    # STEP 3) Recolor each 5x5 tile:
    # center the motif footprint within the 5x5 and repaint those footprint cells to the accent color.
    recolored: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        col = int(b["col"])
        tile = np.full((5, 5), int(col), dtype=int)
        motif = motifs.get(int(col))
        if motif is not None:
            mm, accent = motif
            mh, mw = map(int, mm.shape)
            off_r = (5 - mh) // 2
            off_c = (5 - mw) // 2
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = int(accent)
        recolored[(int(b["r0"]), int(b["c0"]))] = tile

    # STEP 4) Pack recolored 5x5 tiles into a compact grid.
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
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = recolored[(int(r0), int(c0))]

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
ACCENT_CELL_RANGE = (1, 2)

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


def _random_accent_mask(mask: np.ndarray) -> np.ndarray:
    """Select 1-2 occupied motif cells that visibly carry the accent color in the input."""
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    max_cells = min(ACCENT_CELL_RANGE[1], len(coords) - 1)
    min_cells = min(ACCENT_CELL_RANGE[0], max_cells)
    count = random.randint(min_cells, max_cells)
    chosen = set(random.sample(coords, count))
    accent = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        accent[int(r), int(c)] = True
    return accent


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors, reserving the marker color."""
    palette = [c for c in range(COLOR_COUNT) if c not in {bg, MARKER_COLOR}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_motif(
    grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray, accent_mask: np.ndarray
) -> None:
    """Paint one 3x3 bicolored motif into the input grid."""
    primary_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    accent_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(MARKER_COLOR), dtype=int)
    paste(grid, primary_patch, r0, c0, mask=mask)
    paste(grid, accent_patch, r0, c0, mask=accent_mask)


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
            tile[motif_slice, motif_slice][motif_masks[color]] = int(MARKER_COLOR)
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
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    accent_masks: dict[int, np.ndarray],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray, np.ndarray]] = [
        (int(color), motif_masks[color], accent_masks[color]) for color in block_colors
    ]
    for color in distractor_colors:
        mask = _random_mask()
        items.append((int(color), mask, _random_accent_mask(mask)))
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
    accent_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(block_colors, distractor_colors, motif_masks, accent_masks)

    for (sr, sc), (color, mask, accent_mask) in zip(slots, slot_items):
        _paint_motif(
            grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask, accent_mask
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
    motif_masks: dict[int, np.ndarray],
    accent_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination blocks and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks, accent_masks
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.choice([c for c in range(COLOR_COUNT) if c != MARKER_COLOR])

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
    accent_masks = {color: _random_accent_mask(motif_masks[color]) for color in block_colors}

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
        distractor_colors,
        motif_masks,
        accent_masks,
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

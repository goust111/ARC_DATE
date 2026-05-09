# Combo variant of 58f5dbd5 (index 34)
# combo_spec: D7/1+D16/0+D19/2
# Name: 颜色替换+图例指示对齐+选择性雕刻

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis:
# a) 具体化与行为追踪
#    - 原题流程是：找到若干 5x5 纯色方块；按颜色找到对应单色 motif；把 motif 居中投到方块里，
#      覆盖区域改成背景色；最后按原行列顺序紧凑打包输出。
#    - 本融合版同时改变三个环节：
#      1. motif 从单色变成双色连通精灵：主色 = 某个方块色，第二色 = 固定标记色 2。
#         输出时不再“挖空”，而是把整个 motif 占用区统一重染成 2。
#      2. 输入四角之一出现 2x2 图例，其布尔形状编码 motif 在 5x5 内的公共对齐方式：
#         左上 / 居中 / 右下。它只决定偏移，不参与颜色匹配。
#      3. 不是所有方块都使用 motif。先按 packed 输出里的行列索引看位置，只对
#         (row_index + col_index) % 2 == 0 的块应用重染；其余块保持原纯色。
#    - 一个 2x3 布局的具体例子：若图例表示 lower_right，某个蓝块的匹配 motif footprint 是 3x3，
#      原题会把它居中挖成背景；现在只有棋盘选中的那些块会在右下角 3x3 窗口里被整块染成红色 2，
#      未选中的蓝块仍是完整的 5x5 蓝方块。
#    - 差异由三部分共同构成：图例控制“放哪儿”，双色 motif 提供“染成什么”，棋盘筛选控制“哪些块生效”。
#
# b) 复杂度校准
#    - 对齐模式只保留三档，避免出现过多空间分支。
#    - 选择条件使用 packed 位置棋盘奇偶，而不是颜色 id 或隐藏排序，肉眼可见且样例数少时也能归纳。
#    - 所有 motif 共享固定标记色 2，减少颜色映射负担；图例也复用这同一个强调色，但被隔离在角落，
#      这样既节省调色板预算，又不会和颜色匹配关系混淆。
#    - 仍保留 3-6 个目标块、6-8 个槽位容量、3x3 连通 motif 和若干干扰项，避免任务退化成单一操作。
#
# c) 隐式假设审查
#    - 原 main() 用 `find_components(... monochromatic=True, colors=[block_color])` 找 motif，
#      这依赖 motif 为单色；融合后失效，必须改为扫描非背景的多色连通分量，并筛出
#      “颜色集合恰为 {某个块色, 2}” 的小组件。
#    - 原雕刻步骤把命中像素写成 bg；这里必须改成写成 marker，否则会退回原题。
#    - 原雕刻偏移固定由居中公式得到；这里必须先从角落图例解码共享对齐模式。
#    - 原逻辑默认每个块都执行匹配后的变换；现在要先计算 packed 行列索引，再按棋盘规则决定是否应用。
#    - 新 solver 额外假设：
#      1. 角落恰有一个 2x2 图例，且形状来自三种已知编码之一；
#      2. 每个有效 motif 是一个 bbox 不超过 3x3 的双色连通组件，且只包含一个块色和标记色；
#      3. 目标块仍位于完整的 RxC 晶格上，因此 packed 行列索引稳定可见。
#
# d) 放置合法性
#    - 生成器继续把目标块区域和 motif 区域沿某一轴分离放置。
#    - 另外显式避开四个 2x2 角块，保证图例不会与大区域重叠；只有一个角会被真正写入图例。
#    - motif 仍落在 3x3 槽位中，槽位间距 1；双色只是对同一连通 footprint 局部换色，不改变占位。
#
# e) 视觉可观察性
#    - 选中块会出现“原块色 + 红色 2”的局部图案，未选中块保持纯色，选择规则在输出里直接可见。
#    - 同一批 motif 若图例不同，会整体偏到左上/中间/右下，空间效果明显。
#    - 因为输出不再出现背景孔洞，而是统一红色重染，所以和原题有稳定、强烈的视觉差异。
#
# f) 可视化整洁度
#    - 图例独占角落；块区和 motif 区仍保持大块分离，不会把说明符号混进 motif 槽位网格。
#    - 干扰 motif 也采用同样的双色样式，因此视觉语法一致，但颜色不匹配时不会被用到。
#
# g) 变化可推断性
#    - 唯一新增的颜色行为是“输出中的 motif footprint 全部变成固定色 2”，这个颜色在输入 motif
#      和角落图例里都有重复出现，结构提示充分。
#    - 选择规则不依赖颜色值，图例意义也不依赖具体颜色 id，只依赖角落 2x2 形状。
#    - 为防退化，生成器保证：背景不等于 2；布局一定同时含有选中块和未选中块；图例模式随机变化。


MARKER_COLOR = 2

ALIGNMENT_MASKS: dict[str, np.ndarray] = {
    "upper_left": np.array([[1, 0], [0, 0]], dtype=bool),
    "center": np.array([[0, 1], [1, 0]], dtype=bool),
    "lower_right": np.array([[0, 0], [0, 1]], dtype=bool),
}
ALIGNMENT_MODES = tuple(ALIGNMENT_MASKS.keys())
ALIGNMENT_COVERAGE_ORDER = ("upper_left", "lower_right", "center")
CORNER_SLICES = (
    (slice(0, 2), slice(0, 2)),
    (slice(0, 2), slice(-2, None)),
    (slice(-2, None), slice(0, 2)),
    (slice(-2, None), slice(-2, None)),
)


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


def _decode_alignment(grid: np.ndarray, bg: int) -> str:
    """Read the only non-background 2x2 corner patch and map its boolean shape to an alignment mode."""
    for rs, cs in CORNER_SLICES:
        patch = np.array(grid[rs, cs], dtype=int)
        mask = patch != int(bg)
        if not np.any(mask):
            continue
        for name, code in ALIGNMENT_MASKS.items():
            if np.array_equal(mask, code):
                return name
    raise AssertionError("missing corner alignment legend")


def _alignment_offsets(tile_h: int, tile_w: int, motif_h: int, motif_w: int, mode: str) -> tuple[int, int]:
    """Convert the decoded alignment into a top-left placement offset inside the 5x5 block."""
    if mode == "upper_left":
        return 0, 0
    if mode == "center":
        return (tile_h - motif_h) // 2, (tile_w - motif_w) // 2
    if mode == "lower_right":
        return tile_h - motif_h, tile_w - motif_w
    raise ValueError(f"unknown alignment mode: {mode}")


def _should_recolor(row_index: int, col_index: int) -> bool:
    """Checkerboard selector over packed block coordinates."""
    return (int(row_index) + int(col_index)) % 2 == 0


def _collect_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Find solid 5x5 monochrome destination blocks."""
    blocks: list[dict[str, int]] = []
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 5 or bbox_w(b) != 5:
            continue
        col = int(comp["color"])
        patch = crop_bbox(grid, b)
        if bool((patch == int(col)).all()):
            r0, c0, _r1, _c1 = map(int, b)
            blocks.append({"col": int(col), "r0": int(r0), "c0": int(c0)})
    return blocks


def _collect_motifs(grid: np.ndarray, bg: int, block_colors: set[int]) -> dict[int, tuple[np.ndarray, int]]:
    """Find small bicolored motif sprites keyed by their matching block color."""
    motifs: dict[int, tuple[np.ndarray, int]] = {}
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) > 3 or bbox_w(b) > 3:
            continue

        patch = crop_bbox(grid, b)
        rel_mask = patch != int(bg)
        if not rel_mask.any():
            continue

        colors = {int(v) for v in patch[rel_mask]}
        matching = colors & block_colors
        if len(colors) != 2 or MARKER_COLOR not in colors or len(matching) != 1:
            continue

        primary = int(next(iter(matching)))
        motifs[primary] = (rel_mask.copy(), int(MARKER_COLOR))
    return motifs


def _solve_core(
    grid: np.ndarray,
    *,
    forced_alignment: str | None = None,
    force_all_blocks: bool = False,
    force_center_alignment: bool = False,
    force_bg_carve: bool = False,
) -> np.ndarray:
    """Shared solver used by main() and post-hoc degeneration checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    blocks = _collect_blocks(g, bg)
    assert blocks

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    alignment = forced_alignment or _decode_alignment(g, bg)
    if force_center_alignment:
        alignment = "center"

    block_colors = {int(b["col"]) for b in blocks}
    motifs = _collect_motifs(g, bg, block_colors)

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) * 1 + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) * 1 + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        col = int(b["col"])
        i = int(r_index[int(b["r0"])])
        j = int(c_index[int(b["c0"])])
        tile = np.full((5, 5), int(col), dtype=int)
        motif = motifs.get(int(col))
        use_motif = force_all_blocks or _should_recolor(i, j)
        if motif is not None and use_motif:
            mm, accent = motif
            mh, mw = map(int, mm.shape)
            off_r, off_c = _alignment_offsets(5, 5, mh, mw, alignment)
            fill_color = int(bg) if force_bg_carve else int(accent)
            for rr, cc in np.argwhere(mm):
                tr = int(off_r + int(rr))
                tc = int(off_c + int(cc))
                if 0 <= tr < 5 and 0 <= tc < 5:
                    tile[tr, tc] = fill_color

        rr = int(1 + i * step)
        cc = int(1 + j * step)
        out[rr : rr + 5, cc : cc + 5] = tile

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the legend-aligned, checkerboard-selected blocks and then pack them tightly.

    Input structure:
    - The background is the most common color.
    - Several solid 5×5 monochrome blocks appear, each using a different non-background color.
    - Elsewhere, small 3×3 motif sprites are bicolored: one color matches a block and the other is a shared marker color.
    - A single 2×2 symbol sits in one corner and encodes whether every used motif should align upper-left, center, or lower-right inside its 5×5 block.

    Rule:
    - Decode the corner legend to get the shared alignment, and match each motif to the block whose color it contains.
    - Only blocks whose packed-grid position has even row+column parity use their motif; those selected blocks repaint the entire motif footprint to the marker color at the decoded offset, while the other blocks stay solid.
    - Pack all resulting 5×5 tiles into an output grid preserving their original row/column order, with 1-cell gaps and a 1-cell outer border.

    Key insight:
    - The motif no longer acts like a hole punch. It is a selective recolor stencil: the corner icon decides where the stencil sits, the block color decides which stencil belongs to which block, and the checkerboard position rule decides whether the stencil is applied at all.
    """
    return _solve_core(grid)


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
LEGEND_SIZE = 2
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
LEGEND_COVERAGE_ORDER = ("top_left", "bottom_right", "top_right", "bottom_left")

# Generation retry budget.
GENERATION_RETRIES = 300

_GENERATE_CALL_COUNT = 0


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
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
    """Sample a valid layout while preserving the original count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask() -> np.ndarray:
    """Sample one motif footprint from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _random_accent_mask(mask: np.ndarray) -> np.ndarray:
    """Pick 1-2 occupied cells that visibly carry the marker color in the input motif."""
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
    """Pick block and distractor primary colors, reserving the shared marker color."""
    palette = [c for c in range(COLOR_COUNT) if c not in {bg, MARKER_COLOR}]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return block_colors, distractor_colors


def _paint_bicolored_motif(
    grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray, accent_mask: np.ndarray
) -> None:
    """Paint one connected bicolored motif into the input grid."""
    primary_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    marker_patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(MARKER_COLOR), dtype=int)
    paste(grid, primary_patch, r0, c0, mask=mask)
    paste(grid, marker_patch, r0, c0, mask=accent_mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    alignment: str,
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled block grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    off_r, off_c = _alignment_offsets(DST_BLOCK_SIZE, DST_BLOCK_SIZE, MOTIF_SIZE, MOTIF_SIZE, alignment)
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            if _should_recolor(i, j):
                for rr, cc in np.argwhere(motif_masks[color]):
                    tile[off_r + int(rr), off_c + int(cc)] = int(MARKER_COLOR)
            paste(expected, tile, OUTPUT_BORDER + i * DST_BLOCK_STEP, OUTPUT_BORDER + j * DST_BLOCK_STEP)
    return expected


def _paint_destination_blocks(grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]) -> None:
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


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
        _paint_bicolored_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask, accent_mask)


def _corner_origin(H: int, W: int, corner: str) -> tuple[int, int]:
    """Return the top-left coordinate of the legend patch for the chosen corner."""
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, W - LEGEND_SIZE
    if corner == "bottom_left":
        return H - LEGEND_SIZE, 0
    if corner == "bottom_right":
        return H - LEGEND_SIZE, W - LEGEND_SIZE
    raise ValueError(f"unknown legend corner: {corner}")


def _paint_legend(grid: np.ndarray, bg: int, corner: str, alignment: str) -> None:
    """Paint the 2x2 alignment legend using the shared marker color."""
    r0, c0 = _corner_origin(*grid.shape, corner)
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), int(bg), dtype=int)
    patch[ALIGNMENT_MASKS[alignment]] = int(MARKER_COLOR)
    grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = patch


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two inclusive rectangles overlap."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _legend_bbox(H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    """Inclusive bbox occupied by a corner legend."""
    r0, c0 = _corner_origin(H, W, corner)
    return r0, c0, r0 + LEGEND_SIZE - 1, c0 + LEGEND_SIZE - 1


def _placement_avoids_corners(
    H: int,
    W: int,
    placement: tuple[int, int, int, int],
    dst_h: int,
    dst_w: int,
    motif_h: int,
    motif_w: int,
) -> bool:
    """Reject placements where either large region overlaps any 2x2 corner legend slot."""
    dst_r, dst_c, motif_r, motif_c = placement
    dst_box = (dst_r, dst_c, dst_r + dst_h - 1, dst_c + dst_w - 1)
    motif_box = (motif_r, motif_c, motif_r + motif_h - 1, motif_c + motif_w - 1)
    return all(
        not _rects_overlap(dst_box, corner_box) and not _rects_overlap(motif_box, corner_box)
        for corner_box in (_legend_bbox(H, W, corner) for corner in LEGEND_CORNERS)
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
    legend_corner: str,
    alignment: str,
) -> np.ndarray:
    """Render destination blocks, bicolored motifs, and the corner legend into one input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_masks, accent_masks)
    _paint_legend(inp, bg, legend_corner, alignment)
    return inp


def _gen_attempt(*, alignment: str | None = None, legend_corner: str | None = None) -> np.ndarray | None:
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

    legend_corner = legend_corner or random.choice(LEGEND_CORNERS)
    alignment = alignment or random.choice(ALIGNMENT_MODES)
    placement = None
    for _ in range(30):
        candidate = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
        if candidate is None:
            continue
        if _placement_avoids_corners(H, W, candidate, dst_h, dst_w, motif_h, motif_w):
            placement = candidate
            break
    if placement is None:
        return None

    count = R * C
    block_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in block_colors}
    accent_masks = {color: _random_accent_mask(motif_masks[color]) for color in block_colors}

    expected = _render_expected(bg, R, C, block_colors, motif_masks, alignment)
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
        legend_corner,
        alignment,
    )

    if not np.array_equal(main(inp), expected):
        return None
    # Reject samples that accidentally collapse to a solver missing one combo dimension.
    if np.array_equal(expected, _solve_core(inp, force_all_blocks=True)):
        return None
    if alignment != "center" and np.array_equal(expected, _solve_core(inp, force_center_alignment=True)):
        return None
    if np.array_equal(expected, _solve_core(inp, force_bg_carve=True)):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    call_index = int(_GENERATE_CALL_COUNT)
    _GENERATE_CALL_COUNT += 1
    alignment = ALIGNMENT_COVERAGE_ORDER[call_index % len(ALIGNMENT_COVERAGE_ORDER)]
    legend_corner = LEGEND_COVERAGE_ORDER[call_index % len(LEGEND_COVERAGE_ORDER)]
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(alignment=alignment, legend_corner=legend_corner)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")

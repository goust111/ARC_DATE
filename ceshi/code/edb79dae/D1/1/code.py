# Variant of edb79dae
# Dimension: D1 — 第一步匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架内目标像素颜色（s.color == d.color），通过颜色一致性将外部精灵与内部单色像素配对... → 边缘颜色匹配
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     - 原题第一步依赖“框内占位块颜色 == 外部精灵颜色”。
#     - 本变体取消这种直接配色，改为：外部精灵的颜色只在框边标记中出现。
#     - 框内只保留统一的导向块；真正的颜色-坐标对应关系由“左/右边同色行标 + 上/下边同色列标”共同给出。
#   concrete_layout:
#     - 仍然有一个矩形灰框，输出仍然只取该框区域。
#     - 框内是规则网格中的若干个实心 NxN 导向块，颜色统一且与任何精灵源色都不同。
#     - 框外仍然有两类对象：每种源色一个 sprite 模板；每种源色一个 src->tgt 的 3x4 图例。
#     - 框边多数格子仍是灰色，但部分边框格子被源色标记替换：
#       - 左右边同一行出现同色标记，表示该颜色允许放置的顶部行。
#       - 上下边同一列出现同色标记，表示该颜色允许放置的左侧列。
#       - 同色行集 × 同色列集 的笛卡尔积，就是该颜色所有导向块的位置。
#   behavior_trace:
#     - 小例子：假设红色 sprite 在框外，框边红色出现在行 {3,7}、列 {5}。
#     - 原题会看框内是否有红色占位块；这里只有统一导向色，因此旧逻辑失效。
#     - 新逻辑会推出红色目标左上角是 (3,5) 和 (7,5)，然后把红色 sprite 形状盖到这两个导向块上。
#     - 最后再根据图例把红色 sprite 重着色成图例中的目标色。
#   complexity_calibration:
#     - 每个颜色对应一个轴对齐矩形槽区（若干相邻槽位），所以一个颜色可能出现 1/2/3/4 次。
#     - 这样既保留“同一个 sprite 可重复放置”的结构性，也避免每个块都要单独编码，规则不会碎裂。
#     - 颜色数实际收紧到 2 个；总块数维持在 6-8。
#     - 为了避免边框标记互相覆盖，生成器要求不同颜色使用互不共享的行组和列组；这会把部分样本的有效行带扩到 4 条，但仍保持原题级别的整体尺寸。
#   old_main_assumption_audit:
#     - `_find_frame` 原先要求整条边框都是同一颜色；现在边框带彩色标记，这个前提不成立。
#       处理：改为寻找“最大的、不接触整张图边界的背景连通域”，再外扩 1 格得到框。
#     - `placeholder_colors` 原先来自框内实心块颜色；现在框内导向块颜色不再携带 sprite 身份。
#       处理：框内只提取所有导向块 bbox；源色集合改从四边标记推断。
#     - `color_map` 原先可借助 placeholder 颜色判定 legend 的 src；现在不能依赖框内颜色。
#       处理：保留 legend 几何检测，但统一用“较大面积那一色是 src”解析 7:5 图例。
#     - `sprite_coords` 原先按 placeholder 颜色查同色 sprite；现在改为按边缘标记得到的源色集合查 sprite。
#     - 最后粘贴阶段原先循环 `placeholders`；现在循环“边缘标记推导出的目标左上角集合”。
#   legality_and_generator_guarantees:
#     - 生成器只使用互不重叠的槽位矩形，保证同色行集 × 同色列集 精确对应现有导向块，不会产生虚假交点。
#     - 导向块全部同尺寸，且 sprite bbox 与导向块尺寸一致。
#     - 边缘标记不会落在四角；框外 sprite/legend 仍通过随机空位放置，带 1 格背景隔离。
#     - 框内至少每行每列都有已填槽位，保证最大的封闭背景域稳定地就是框内部背景。
#   observability:
#     - 人类可直接看到：框内所有块颜色相同，但边框彩色刻度不同；因此“颜色决定位置”的信息显然从内部转移到了边缘。
#     - 多个样例中，同一颜色往往在多行或多列重复，能支持“行标 × 列标”的推断，而不是一次性巧合。
#   cleanliness_and_color_reasoning:
#     - 同一层级只放三类信息：框内导向块、框边颜色标记、框外 sprite/legend；不做额外叠加。
#     - 导向色与源色分离，避免旧规则退化；目标色仍由外部 legend 提示，不引入不可见排序。

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _find_marked_frame(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Find the frame via the largest enclosed background component."""
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for pts in components_mask(g == bg, connectivity=4):
        rs = pts[:, 0]
        cs = pts[:, 1]
        if (rs == 0).any() or (rs == g.shape[0] - 1).any() or (cs == 0).any() or (cs == g.shape[1] - 1).any():
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        candidates.append((len(pts), (int(r0), int(c0), int(r1), int(c1))))
    assert candidates, "no enclosed background component found"
    _, (ir0, ic0, ir1, ic1) = max(candidates, key=lambda item: item[0])
    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert 0 <= fr0 < fr1 < g.shape[0] and 0 <= fc0 < fc1 < g.shape[1], "expanded frame is out of bounds"
    return int(fr0), int(fc0), int(fr1), int(fc1)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a marked frame using edge-color coordinates and a color legend.

    Input structure:
    - bg background, a rectangular gray frame whose border carries a few colored edge markers, and uniform guide blocks inside the frame.
    - Outside the frame: sprite shapes (one per source color) and a small legend mapping each source color to an output color.

    Rule:
    - Find the frame, collect the guide-block bboxes, and read same-color markers from opposite frame edges.
    - For each source color, the marked rows and marked columns determine the guide-block positions where that sprite belongs.
    - Stamp the matching sprite at every marker-defined guide location.
    - Recolor every stamped sprite to the output color defined by the legend.

    Key insight: The sprite identity no longer comes from the guide-block color inside the frame; it comes from the border markers, which convert edge positions into interior coordinates.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    fr0, fc0, fr1, fc1 = _find_marked_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    frame_col = int(Counter(map(int, crop[border_mask].flatten())).most_common(1)[0][0])

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    guides: list[tuple[int, int, int, int]] = []
    inner_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg}
    assert inner_colors, "no guide blocks found inside frame"
    for col in sorted(inner_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            guides.append(tuple(map(int, bbox_coords(pts))))
    assert guides, "no guide components found"

    guide_dims = sorted({(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in guides})
    assert len(guide_dims) == 1, f"ambiguous guide sizes: {guide_dims}"
    guide_h, guide_w = guide_dims[0]
    guide_by_topleft = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in guides}

    row_marks: dict[int, list[int]] = {}
    for r in range(1, H - 1):
        left = int(crop[r, 0])
        right = int(crop[r, W - 1])
        if left == right and left not in (bg, frame_col):
            row_marks.setdefault(left, []).append(int(r))

    col_marks: dict[int, list[int]] = {}
    for c in range(1, W - 1):
        top = int(crop[0, c])
        bottom = int(crop[H - 1, c])
        if top == bottom and top not in (bg, frame_col):
            col_marks.setdefault(top, []).append(int(c))

    source_colors = set(row_marks) & set(col_marks)
    assert source_colors, "no source colors with both row and column markers"

    targets: dict[int, list[tuple[int, int]]] = {}
    for src_col in sorted(source_colors):
        pts = sorted((int(r), int(c)) for r in row_marks[src_col] for c in col_marks[src_col])
        assert pts, f"no targets derived for source color {src_col}"
        for pt in pts:
            assert pt in guide_by_topleft, f"marker-derived target {pt} has no guide block"
        targets[src_col] = pts
    all_targets = {pt for pts in targets.values() for pt in pts}
    assert all_targets == set(guide_by_topleft), "guide blocks do not match marker-derived targets"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(pts_tuples),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
                continue

            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = ci["col"], cj["col"]
            else:
                src, tgt = cj["col"], ci["col"]

            if src not in source_colors:
                continue
            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])
    assert set(color_map) == source_colors, (
        f"legend mappings {sorted(color_map)} do not match source colors {sorted(source_colors)}"
    )

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(source_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for source color {src_col}"
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) == (guide_h, guide_w)
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for source color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])
        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, pts in targets.items():
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (sh, sw) == (guide_h, guide_w), f"guide size mismatch for source color {src_col}"
        for pr0, pc0 in pts:
            pr1 = pr0 + sh - 1
            pc1 = pc0 + sw - 1
            assert guide_by_topleft[(pr0, pc0)] == (pr0, pc0, pr1, pc1), "guide bbox mismatch"
            for dr, dc in coords:
                out[pr0 + dr, pc0 + dc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_SOURCE_COLORS = 2
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300
LAYOUTS = (
    ((2, 2), (1, 2)),  # 6 blocks across 3 row marks and 4 col marks
    ((2, 2), (2, 2)),  # 8 blocks across 4 row marks and 4 col marks
    ((3, 1), (1, 3)),  # 6 blocks across 4 row marks and 4 col marks
    ((3, 2), (1, 1)),  # 7 blocks across 4 row marks and 3 col marks
)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    mask = np.zeros((n, n), dtype=bool)
    c = random.randrange(n)
    for r in range(n):
        c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(n), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True
    return mask


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, tuple[tuple[int, int], ...]] | None:
    options = []
    for n in range(*SPRITE_N_RANGE):
        for layout in LAYOUTS:
            total_rows = sum(rc for rc, _ in layout)
            total_cols = sum(cc for _, cc in layout)
            if total_rows < GRID_ROWS_RANGE[0] or total_rows > 4:
                continue
            if total_cols < GRID_COLS_RANGE[0] or total_cols > 5:
                continue
            ch, cw = _content_extent(n, total_rows, total_cols)
            base_fh = ch + 4
            base_fw = cw + 4
            if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM:
                options.append((n, layout))
    return random.choice(options) if options else None


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, layout = struct
    groups = list(layout)
    random.shuffle(groups)
    k = len(groups)
    assert k == NUM_SOURCE_COLORS
    rows = sum(rc for rc, _ in groups)
    cols = sum(cc for _, cc in groups)

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    if len(usable) < k + 1:
        return None

    src_cols = random.sample(usable, k)
    guide_choices = [c for c in usable if c not in src_cols]
    if not guide_choices:
        return None
    guide_col = random.choice(guide_choices)
    tgt_cols = [random.choice([c for c in usable if c not in (guide_col, src)]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    if y_margin < 2 or x_margin < 2:
        return None
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = input_frame.copy()
    output_frame[1:-1, 1:-1] = bg

    row_cursor = 0
    col_cursor = 0
    for (row_count, col_count), src in zip(groups, src_cols):
        row_marks = [1 + top_margin + rr * (n + 1) for rr in range(row_cursor, row_cursor + row_count)]
        col_marks = [1 + left_margin + cc * (n + 1) for cc in range(col_cursor, col_cursor + col_count)]
        for rr in row_marks:
            input_frame[rr, 0] = src
            input_frame[rr, frame_w - 1] = src
            output_frame[rr, 0] = src
            output_frame[rr, frame_w - 1] = src
        for cc in col_marks:
            input_frame[0, cc] = src
            input_frame[frame_h - 1, cc] = src
            output_frame[0, cc] = src
            output_frame[frame_h - 1, cc] = src
        for rr in range(row_cursor, row_cursor + row_count):
            for cc in range(col_cursor, col_cursor + col_count):
                pr0 = 1 + top_margin + rr * (n + 1)
                pc0 = 1 + left_margin + cc * (n + 1)
                paste(input_frame, np.full((n, n), guide_col, dtype=int), pr0, pc0)
                paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), pr0, pc0, transparent=bg)
        row_cursor += row_count
        col_cursor += col_count

    ext_area = k * (12 + n * n + 10)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D1 input")

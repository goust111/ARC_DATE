# Combo variant of 7666fa5d (index 4)
# combo_spec: D4/1+D5/1
# Name: 交替颜色轮廓填充

import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   summary:
#     original_task: |
#       输入给出 4-5 条位于反对角线 r+c=k 上的短线段。原题对每条主对角线
#       b=r-c 扫描，找到当前同时覆盖该 b 的相邻线段，并把它们之间所有背景格
#       全部涂成红色，得到 3-5 个实心平行四边形带。
#     combo_change: |
#       现在每个区域不再做面积填充，而是先按原题几何规则恢复该区域的实心带，
#       再只保留这个带的 4 邻域外轮廓；不同区域的轮廓按区域顺序红/蓝/红/蓝交替。
#   concrete_trace:
#     small_example: |
#       设 4 条线段的 k 为 [6,10,14,18]，因此有 3 个相邻区域：
#       R0 在 6-10 之间，R1 在 10-14 之间，R2 在 14-18 之间。
#       原题会把 R0/R1/R2 的内部全部涂成红色。
#       本融合版则分别取每个区域的填充 mask，只保留该 mask 的边界：
#       R0 轮廓为红(2)，R1 轮廓为蓝(1)，R2 轮廓再为红(2)。
#       区域内部恢复为背景色，输入中的斜线段本身仍保持原色。
#     key_difference: |
#       改动不是“先做轮廓，再随便换色”，而是“先按原题识别多个独立区域，
#       再对每个区域分别提取封闭边界，并按区域索引着交替色”。
#   complexity_calibration:
#     keep_visible: |
#       为了让“轮廓”与“交替颜色”都在 2-4 个样例里可见，生成器固定使用
#       4-5 个不同的 k 值，从而稳定产生 3-4 个区域；同时要求至少两个颜色都出现，
#       且至少一个区域在 b 方向有足够厚度，使其上下边界清晰可见。
#     keep_constrained: |
#       只使用两种轮廓色 [2,1]，始终从最左侧区域开始红色，避免颜色种类过多。
#       线段颜色排除 1 和 2，避免与新增轮廓色混淆。
#   implicit_assumption_audit_main:
#     - main 仍假设输入只有一种主导非背景色作为线段色；生成器只放一种线段色，成立。
#     - 组件提取仍依赖每个连通块位于单一反对角线 k 上；生成器继续保证这一点。
#     - 原 solver 假设“所有新增像素同色”；融合后不成立，因此必须把“区域识别”和“颜色分配”拆开。
#     - 若继续像 D5 单变体那样按每条 b 扫描时交替着色，会得到逐扫描线切换的颜色，
#       但题面要求是“不同填充区域的轮廓”交替，因此颜色索引必须绑定到区域，而不是绑定到单条 b。
#     - 若保留原题的 same-k split，某个颜色索引会对应多个断裂子区域，视觉上不够整洁；
#       生成器因此去掉 split，并强制所有 k 值互异。main 于是新增了“区域按全局 k 顺序配色”的假设，
#       生成器通过 distinct_k_values 保证它成立。
#   placement_legality:
#     - 所有线段仍通过 _b_bounds_for_k 限制在 16x16 内。
#     - 相邻线段必须在 b 方向至少重叠 OVERLAP_MIN_B，保证区域有真正的封闭轮廓。
#     - 同一格禁止放置两条线段；生成器逐格检查冲突。
#   visual_observability:
#     - 输入只见单色斜线段，输出新增的是多个空心平行四边形轮廓。
#     - 颜色交替依附于从左到右的相邻区域顺序，而不是依附于隐藏数值或扫描时刻；
#       解题者能从多个并列区域直接看出红蓝红蓝模式。
#   visual_cleanliness:
#     - 不做 same-k split，避免一个颜色对应多个零碎轮廓。
#     - 轮廓只写入原背景格，不覆盖线段，边界结构和颜色语义都清晰。
#   anti_degeneracy_metric:
#     - 仅通过 main(inp)==out 还不够，因为若只有一个区域或只有红色轮廓，样例会退化。
#     - 生成器和后续统计都检查：区域数 >= 3、两种轮廓色都出现、轮廓输出与原始实心填充输出不同。


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return line segments as (k, b_min, b_max), sorted by anti-diagonal index."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(mask):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    segs.sort(key=lambda t: (t[0], t[1], t[2]))
    return segs


def _pair_fill_mask(
    shape: tuple[int, int],
    left_seg: tuple[int, int, int],
    right_seg: tuple[int, int, int],
) -> np.ndarray:
    """
    Build the original solid-fill mask for one adjacent pair of anti-diagonal segments.

    The pair contributes only on b-values covered by both segments.
    """
    h, w = shape
    k_left, left_b0, left_b1 = left_seg
    k_right, right_b0, right_b1 = right_seg
    mask = np.zeros((h, w), dtype=bool)

    if k_left >= k_right:
        return mask

    b_start = max(left_b0, right_b0)
    b_end = min(left_b1, right_b1)
    if b_start > b_end:
        return mask

    for b in range(b_start, b_end + 1):
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for r in range(r_min, r_max + 1):
            c = r - b
            k = r + c
            if k_left < k < k_right:
                mask[r, c] = True
    return mask


def _boundary_mask(fill_mask: np.ndarray) -> np.ndarray:
    """Keep only the 4-neighborhood outer boundary of a filled region."""
    h, w = fill_mask.shape
    boundary = np.zeros_like(fill_mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not fill_mask[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w) or not fill_mask[rr, cc]:
                    boundary[r, c] = True
                    break
    return boundary


def _apply_solid_reference(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    fill: int,
    bg: int,
) -> np.ndarray:
    """Reference implementation of the original solid red behavior, used for generator checks."""
    out = grid.copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        for k_left, k_right in zip([t[0] for t in active], [t[0] for t in active][1:]):
            r_min = max(0, b)
            r_max = min(h - 1, (w - 1) + b)
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def _apply_combo_outline(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill_colors: tuple[int, int],
) -> tuple[np.ndarray, dict[int, int], int]:
    """
    Draw alternating-color outlines for each adjacent region.

    Returns:
        output grid, counts per fill color, number of non-empty regions
    """
    out = grid.copy()
    color_counts = {color: 0 for color in fill_colors}
    region_count = 0

    for left_seg, right_seg in zip(segs, segs[1:]):
        fill_mask = _pair_fill_mask(out.shape, left_seg, right_seg)
        if not np.any(fill_mask):
            continue
        boundary = _boundary_mask(fill_mask)
        # Alternate colors by visible region order, not by raw pair index, so
        # skipped / empty pairs cannot desynchronize the pattern.
        color = fill_colors[region_count % len(fill_colors)]
        region_count += 1
        new_cells = boundary & (grid == bg)
        out[new_cells] = color
        color_counts[color] += int(np.sum(new_cells))

    return out, color_counts, region_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw alternating-color outlines for the parallelogram regions induced by the input segments.

    Input structure:
    - The background is the most common color.
    - A single other color forms short diagonal segments, each confined to one anti-diagonal r+c=k.
    - The segments occupy several distinct k-values, so neighboring segments define several potential regions.

    Rule:
    - Sort the detected anti-diagonal segments by k and pair each segment with its next neighbor.
    - For each neighboring pair, recover the same between-segment region that the original task would solid-fill.
    - Discard the interior of that region, keep only its 1-cell-thick outer contour, and color contours by region order:
      first red (2), second blue (1), third red, and so on.

    Key insight:
    - The geometry still comes from overlap between adjacent anti-diagonal segments, but the output no longer marks area.
      One must identify each whole region first, then convert that region into a contour and apply a stable alternating
      color pattern across the sequence of regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g == line)
    out, _counts, _regions = _apply_combo_outline(g, segs, bg=bg, fill_colors=(2, 1))
    return out


# --------------- generator ---------------

"""Generator for the D4/1 + D5/1 combo variant of ARC-AGI-2 task 7666fa5d."""

# -- Grid & color constants --
H = W = 16
BG = 8
FILL_COLORS = (2, 1)
SOLID_REFERENCE_FILL = 2
LINE_COLORS = [3, 4, 5, 6, 7, 9]  # exclude 1/2 so alternating outline colors stay unambiguous

# -- Anti-diagonal (k = r+c) layout --
K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

# -- Segment geometry --
SEG_LEN_MIN, SEG_LEN_MAX = 3, 9
OVERLAP_MIN_B = 4
MIN_OVERLAP_SPAN = 4

# -- Retry budgets --
SEG_RETRY = 60
GEN_RETRY = 500

# -- Visibility / anti-degeneracy checks --
MIN_REGION_COUNT = 3
MIN_OUTLINE_CELLS = 8
MIN_SOLID_ADVANTAGE = 4
MIN_COLOR_CELLS = 2


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that ((k+b)/2, (k-b)/2) lies inside the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Return the cells of one anti-diagonal segment."""
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly spaced anti-diagonals of fixed parity."""
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_gap_triples = [
            (a, b, c)
            for a in GAP_VALUES
            for b in GAP_VALUES
            for c in GAP_VALUES
            if a + b + c in GAP_TARGET_SPANS
        ]
        gaps = list(random.choice(valid_gap_triples))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None

    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_seg(
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    """Sample one segment on anti-diagonal k, optionally forcing overlap with the previous segment."""
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        step = 2 * (length - 1)
        if b_lo + step > b_hi:
            continue

        candidates = list(range(b_lo, b_hi - step + 1, 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                b_min
                for b_min in candidates
                if min(b_min + step, constrain_hi) - max(b_min, constrain_lo) >= MIN_OVERLAP_SPAN
            ]
            if not candidates:
                continue

        b_min = random.choice(candidates)
        b_max = b_min + step
        if constrain_lo is not None and constrain_hi is not None:
            overlap = min(b_max, constrain_hi) - max(b_min, constrain_lo)
            if overlap < OVERLAP_MIN_B:
                continue
        return b_min, b_max
    return None


def _count_pair_regions(segs: list[tuple[int, int, int]]) -> tuple[int, list[int]]:
    """Count adjacent-pair regions and record their b-overlap spans."""
    spans: list[int] = []
    for left_seg, right_seg in zip(segs, segs[1:]):
        span = min(left_seg[2], right_seg[2]) - max(left_seg[1], right_seg[1])
        if span >= 0:
            spans.append(int(span))
    return len(spans), spans


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the combo rule."""
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)

    ks = _choose_k_values(parity)
    if ks is None:
        return None

    segs: list[tuple[int, int, int]] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1]))

    for k in ks[1:]:
        prev_lo, prev_hi = segs[-1][1], segs[-1][2]
        nxt = _random_seg(k, prev_lo, prev_hi)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))

    region_count, overlap_spans = _count_pair_regions(segs)
    if region_count < MIN_REGION_COUNT:
        return None
    if not overlap_spans or max(overlap_spans) < MIN_OVERLAP_SPAN:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    output, color_counts, produced_regions = _apply_combo_outline(
        grid,
        segs,
        bg=BG,
        fill_colors=FILL_COLORS,
    )
    solid = _apply_solid_reference(grid, segs, fill=SOLID_REFERENCE_FILL, bg=BG)

    outline_cells = sum(color_counts.values())
    solid_fill_cells = int(np.sum((solid == SOLID_REFERENCE_FILL) & (grid == BG)))

    if produced_regions < MIN_REGION_COUNT:
        return None
    if outline_cells < MIN_OUTLINE_CELLS:
        return None
    if solid_fill_cells < outline_cells + MIN_SOLID_ADVANTAGE:
        return None
    if np.array_equal(output, solid):
        return None
    if any(color_counts[color] < MIN_COLOR_CELLS for color in FILL_COLORS):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

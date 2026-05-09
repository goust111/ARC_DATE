# Variant of 7666fa5d
# Dimension: D4 — 目标位置：填充区域定义 (dst)
# Change: 沿主对角线方向(b=r-c固定)，在相邻反对角线段之间严格内部的背景色像素位置填充红色... → 仅填充边界（轮廓式填充）

import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     original_dst: |
#       对每条主对角线 b=r-c，找到当前活跃的相邻反对角线段，
#       将它们之间所有严格内部的背景格全部填成红色，得到实心平行四边形区域。
#     new_dst: |
#       不再填满内部面积，而是只保留每个区域的轮廓新增部分：
#       对原本会被填充的每个连续 b-条带，只在该条带的起始 b 和结束 b 上，
#       画出连接两条线段的红色主对角线边界。输入中的线段本身继续保留原色，
#       因而输出中的“轮廓”由两条原色反对角边 + 两条新增红色主对角边组成。
#   concrete_behavior:
#     pixels: |
#       若两条线段 s1=(k1,b1_lo..b1_hi), s2=(k2,b2_lo..b2_hi) 在某段连续的 b 上同时活跃，
#       原任务会在这整段 b 上，对所有 k1<k<k2 的背景格着红；
#       变体只在这段 b 的首尾两条主对角线画红，内部保持背景色。
#     small_trace: |
#       例：k=8 与 k=14 两条线段在 b=2..6 上同时活跃。
#       原规则会在 b=2,3,4,5,6 这五条主对角线上都填红。
#       新规则只在 b=2 与 b=6 两条主对角线上填红，中间 b=3,4,5 保持空白。
#       差异即“面积”变“边框”。
#   complexity_calibration:
#     keep_visible: |
#       若只出现单个区域、且连接边只有 1 个像素，则变体不够显眼。
#       生成器因此要求：至少 2 个区域条带、至少 1 个条带拥有不同的上下两条边界、
#       且新旧规则输出必须不同，避免退化成原题或肉眼难区分的极小案例。
#     keep_constrained: |
#       仍保持 16x16、单一线段色、红色输出、4-5 条反对角线段的原题骨架；
#       不引入额外颜色语义，也不引入旋转/镜像等其它维度变化。
#   implicit_assumption_audit_main:
#     - main 仍假设输入只有一种主导非背景色作为线段色；生成器只放一种线段色，成立。
#     - 组件提取仍假设每个组件应落在单一 r+c=k 上；生成器只生成这种斜线段，成立。
#     - 原 solver 直接逐 b 填满内部；变体若继续这样做就会错误地产生实心区域，因此必须改成“先识别连续活跃条带，再只画条带首尾边界”。
#     - 若活跃相邻关系随 b 改变，简单用两段线的整体 overlap 会漏边界；因此新 solver 按 b 扫描并记录“相邻 pair 的连续运行区间(run)”。
#   placement_legality:
#     - 所有线段仍通过 _b_bounds_for_k 保证在界内。
#     - 生成器继续要求相邻 k 的线段在 b 方向有足够重叠，保证确实形成闭合区域。
#     - 不再启用 same-k split；对轮廓任务而言，断裂同 k 片段更容易制造开放形状或视觉歧义，去掉后轮廓更清晰。
#   visual_observability:
#     - 人类可见规律：输入只有几条平行斜线，输出不是“整块染红”，而是只在两端补出连接边，形成空心平行四边形。
#     - 多个区域重复同一机制，足以从 2-3 个样例中归纳“只画边界”。
#   visual_cleanliness:
#     - 输出只新增红色边，不覆盖原线段颜色，轮廓结构清晰。
#     - 通过限制最小重叠和最小区域数，避免只有零散单点红色的杂乱样本。
#   color_inference:
#     - 颜色规则不变：输入线段色任意单色，输出新增色固定为红色 2。
#     - 红色的结构性线索来自“连接端点的边界线”，而不是颜色值本身。
#   post_change_main_assumptions_checked:
#     - _adjacent_pair_runs 假设按整数 b 扫描可恢复原面积填充条带；这与原 solver 的扫描方式一致。
#     - _apply_outline 假设只有 run 的首尾应新增红边；生成器通过 overlap 约束保证这些 run 对应真实闭合区域而非纯噪声。


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _draw_between_on_b(out: np.ndarray, b: int, k_left: int, k_right: int, bg: int, fill: int) -> None:
    """Draw the red connector on one main diagonal b=r-c between two anti-diagonals."""
    h, w = out.shape
    r_min = max(0, b)
    r_max = min(h - 1, (w - 1) + b)
    if r_min > r_max:
        return
    for r in range(r_min, r_max + 1):
        c = r - b
        k = r + c
        if k_left < k < k_right and out[r, c] == bg:
            out[r, c] = fill


def _adjacent_pair_runs(
    segs: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]]:
    """
    Recover the continuous b-runs that the original solid-fill solver would fill.

    Each returned tuple is (left_seg_id, right_seg_id, b_start, b_end) for one
    contiguous run where the two segments are adjacent among the active segments.
    """
    runs_by_pair: dict[tuple[int, int], list[list[int]]] = {}

    for b in range(-(w - 1), h):
        active = [i for i, (_k, b_lo, b_hi) in enumerate(segs) if b_lo <= b <= b_hi]
        active.sort(key=lambda idx: (segs[idx][0], idx))

        for left_idx, right_idx in zip(active, active[1:]):
            k_left = segs[left_idx][0]
            k_right = segs[right_idx][0]
            if k_left >= k_right:
                continue
            pair = (left_idx, right_idx)
            pair_runs = runs_by_pair.setdefault(pair, [])
            if pair_runs and pair_runs[-1][1] == b - 1:
                pair_runs[-1][1] = b
            else:
                pair_runs.append([b, b])

    runs: list[tuple[int, int, int, int]] = []
    for (left_idx, right_idx), pair_runs in runs_by_pair.items():
        for b_start, b_end in pair_runs:
            runs.append((left_idx, right_idx, int(b_start), int(b_end)))
    return runs


def _apply_outline(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill: int,
) -> np.ndarray:
    """
    Draw only the outer boundary of the original solid-fill region.

    The previous implementation drew the start/end edge of every local run separately,
    which can create an extra red line inside the union of filled regions. Instead, first
    build the full solid fill, then keep only those newly filled cells that touch
    non-filled space in the 4-neighborhood.
    """
    solid = _apply_solid_fill(grid, segs)
    fill_mask = (solid == fill) & (grid == bg)

    out = grid.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if not fill_mask[r, c]:
                continue

            is_boundary = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w) or not fill_mask[rr, cc]:
                    is_boundary = True
                    break

            if is_boundary:
                out[r, c] = fill

    return out


def _apply_solid_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    """Reference implementation of the original solid-area behavior, used for generator checks."""
    out = grid.copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw only the outer outline of the parallelogram regions induced by the input segments.

    Input structure:
    - bg = most common color.
    - One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).

    New D4 rule:
    - First compute exactly the same solid red regions as the original task.
    - Then discard interior red cells and keep only the outer 1-cell-thick boundary
      of the union of those red regions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2

    comps = _cc8(g == line)

    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))

    return _apply_outline(g, segs, bg=bg, fill=fill)


# --------------- generator ---------------

"""Generator for the D4 outline-only variant of ARC-AGI-2 task 7666fa5d."""

# -- Grid & color constants --
H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

# -- Anti-diagonal (k = r+c) layout --
K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

# -- Segment geometry --
SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 4  # larger than the original so the outline has visible top/bottom edges

# -- Retry budgets --
SEG_RETRY = 50
GEN_RETRY = 400

# -- Variant visibility checks --
MIN_OUTLINE_CELLS = 4
MIN_REGION_RUNS = 2
MIN_SEPARATE_BOUNDARY_RUNS = 1
MIN_SOLID_ADVANTAGE = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Return (r,c) cells for a segment on anti-diagonal k spanning b_min..b_max (step 2)."""
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly-spaced anti-diagonal indices with the given parity."""
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_combos = [
            (a, b, c)
            for a in GAP_VALUES
            for b in GAP_VALUES
            for c in GAP_VALUES
            if a + b + c in GAP_TARGET_SPANS
        ]
        gaps = list(random.choice(valid_combos))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None
    k0 = random.choice(k0_candidates)
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
    """Pick a random segment b-range on anti-diagonal k, optionally with minimum overlap."""
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
                if b_min + step >= constrain_lo + OVERLAP_MIN_B
                and b_min <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
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
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        nxt = _random_seg(k, prev_bmin, prev_bmax)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    runs = _adjacent_pair_runs(segs, H, W)
    if len(runs) < MIN_REGION_RUNS:
        return None
    if sum(1 for _li, _ri, b0, b1 in runs if b0 < b1) < MIN_SEPARATE_BOUNDARY_RUNS:
        return None

    outline = _apply_outline(grid, segs, bg=BG, fill=FILL)
    solid = _apply_solid_fill(grid, segs)

    outline_count = int(np.sum(outline == FILL))
    solid_count = int(np.sum(solid == FILL))
    if outline_count < MIN_OUTLINE_CELLS:
        return None
    if solid_count < outline_count + MIN_SOLID_ADVANTAGE:
        return None
    if np.array_equal(outline, solid):
        return None

    if not np.array_equal(main(grid), outline):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

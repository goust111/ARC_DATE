# Variant of 7666fa5d
# Dimension: D11 — 填充规则：扫描方向 (meta)
# Change: 沿主对角线方向(b=r-c固定)扫描，对每条主对角线独立收集活跃反对角线段并填充
#   → 改为逐行正交扫描：对每一行，找出该行上的线段像素，并在同行相邻线段像素之间水平填充。

"""
design_analysis:
  variant_goal:
    preserved_structure:
      - 网格仍为 16x16，背景仍为 8，填充色仍为 2。
      - 输入对象仍是落在反对角线 r+c=k 上的短斜线段，颜色为单一非背景色。
      - 仍然通过多条平行斜线段围出若干待填区域。
    changed_mechanism:
      - 原题按主对角线 b=r-c 扫描：每条 b 线上找“当前活跃”的反对角线段，再沿该斜线方向填。
      - 本变体按水平行扫描：每一行直接读取这一行出现的线段像素列坐标，按列排序后，在相邻线段像素之间水平填红。
      - 因为每条反对角线段在一行中至多贡献一个像素，输入视觉仍像“多条平行斜边”，但输出的推理轴从斜对角改成了肉眼更直接的横向连线。
  concrete_trace:
    example:
      - 两条线段分别位于 k=8 与 k=14，且都覆盖 row=5。
      - 它们在第 5 行上的像素分别是 (5,3) 与 (5,9)。
      - 变体规则会在该行填充 col=4..8 的背景格。
      - 原题不会“按行”看这两个端点，而是会在不同的 b 线上决定哪些格子被填，得到的是沿主对角线展开的斜向区域。
    visible_difference:
      - 同一输入下，变体输出的每一行都会出现连续的水平红条。
      - 原题输出则沿主对角线方向组织，水平切开看时并不是简单的“同行端点之间全填满”。
  complexity_calibration:
    kept_freedoms:
      - 4 或 5 条不同的反对角线 k。
      - 相邻 k 间隔取 4/6/8，保证行内横向距离显著，肉眼能看出“横向填满”。
      - 每条线段长度 2..9。
      - 允许以较低概率把一条线段拆成同 k 的两段，保留少量原题的复杂度。
    anti_degeneration:
      - 生成器要求至少 3 行出现 2 个及以上线段像素，否则不够体现“逐行填充”。
      - 额外比较“新水平扫描输出”和“原主对角扫描输出”，要求两者至少有 8 个格子不同，避免样本退化成看起来仍像原题。
  old_main_assumption_audit:
    old_logic_dependencies:
      - `components_mask(..., connectivity=8)` 假设需要先恢复每条斜线段的连通块。
      - `ks = {r+c}` 假设每个连通块都完全落在单一反对角线上。
      - `bs = [r-c]` 与后续 `s[1] <= b <= s[2]` 假设“线段是否参与填充”由 b 覆盖区间决定。
      - 外层 `for b in ...` 直接把“扫描轴”固定成主对角线。
    variant_effect:
      - D11 的核心变化正是去掉上述 b 轴激活逻辑；若保留，会继续解原题而不是新题。
      - 新 solver 不再依赖连通块和 b 区间，只依赖“同行有哪些线段像素”。
      - 但仍保留“只填背景、不覆盖原线段”的防护，因为这是原题和变体都需要的稳定前提。
  placement_legality:
    representation:
      - 生成器内部把线段记为 `(k, row_lo, row_hi)`，因为新规则的关键是行重叠，而不是 b 重叠。
      - 单条线段的像素为 `(r, k-r)`，其中 `row_lo <= r <= row_hi`，天然形成连续反对角线段。
    bounds_and_overlap:
      - 每个 k 的合法行区间由 `0 <= r <= 15` 且 `0 <= k-r <= 15` 决定。
      - 相邻线段要求行区间至少重叠 2 行，保证存在稳定的水平填充。
      - 拆分时第二段与第一段至少隔 2 行，保证 8 连通被打断。
    solver_compatibility:
      - 输入里只放一种非背景线色，`main()` 可稳定识别。
      - 输出由 `main(input)` 验证回放，防止生成器与 solver 规则漂移。
  observability_and_cleanliness:
    visibility:
      - 每行两个以上端点后，中间整段红色会很直观地呈现“横向连接”。
      - k 间隔固定为 4/6/8，使横向红条至少宽 3 格，不会细到难观察。
    cleanliness:
      - 只有三类视觉元素：背景、斜线边界、红色填充。
      - 不引入额外标记色，也不让多层对象重叠在同一格。
    color_inference:
      - 颜色角色保持原题风格：线色可变但唯一，填充始终为红色 2。
      - 变化规律由空间关系而非颜色数值驱动，人类可从少量样例看出。
  new_main_assumption_audit:
    assumptions:
      - 输入中存在唯一语义上的“线色”；若 2 已经出现在网格里，也优先把非 2 的非背景色视为线色。
      - 同一行中相邻线段像素定义一个待填水平区间。
      - 只填背景格，避免覆盖线段或已有非背景标记。
    generator_guarantees:
      - 只使用一种线色，且不使用 2。
      - 保证有足够多的行出现多个线段像素。
      - 用 `main(inp) == output` 和“与旧规则差异足够大”双重验证。
"""

import numpy as np
import random

from arc_utils import get_bg as _bg_color


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_ROWS = 2

SPLIT_PROB = 0.35
SPLIT_MIN_LEN = 5
SPLIT_ROW_GAP = 2
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 5

SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 300

MIN_FILL_CELLS = 6
MIN_MULTI_ROWS = 3
MIN_DIFFERENCE_FROM_OLD = 8


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect the unique non-background line color.
    - For each row, collect the columns containing line pixels.
    - Sort those columns and fill background cells strictly between each adjacent pair with red.

    The input still consists of anti-diagonal segments, but the fill scan is now orthogonal
    and row-local instead of diagonal and b-axis-based.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    # Prefer the unique non-red foreground as the boundary color; this is robust even if
    # someone accidentally runs the solver on an already-filled output grid.
    line_candidates = [c for c in non_bg if c != FILL]
    line = max(line_candidates or non_bg, key=lambda c: int(np.sum(g == c)))

    out = g.copy()
    for r in range(h):
        cols = [int(c) for c in np.flatnonzero(g[r] == line)]
        if len(cols) < 2:
            continue
        for c_left, c_right in zip(cols, cols[1:]):
            if c_right - c_left <= 1:
                continue
            for c in range(c_left + 1, c_right):
                if out[r, c] == bg:
                    out[r, c] = FILL

    return out


def _row_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive row range for anti-diagonal k=r+c staying inside the 16x16 grid."""
    return max(0, k - (W - 1)), min(H - 1, k)


def _seg_cells(k: int, row_lo: int, row_hi: int) -> list[tuple[int, int]]:
    """Cells of a contiguous anti-diagonal segment represented by a row interval."""
    return [(r, k - r) for r in range(row_lo, row_hi + 1)]


def _apply_fill_rows(grid: np.ndarray) -> np.ndarray:
    """Apply the new row-wise fill rule."""
    return main(grid)


def _apply_fill_diagonal_old(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    """
    Old task logic, kept only as a generator-side regression guard.

    Convert each row interval back into a b=r-c interval and replay the original diagonal scan.
    """
    out = grid.copy()
    diag_segs = [(k, 2 * row_lo - k, 2 * row_hi - k) for k, row_lo, row_hi in segs]
    for b in range(-(W - 1), H):
        active = sorted([s for s in diag_segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _choose_k_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly spaced anti-diagonal indices of one parity, mirroring the original task."""
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
    """
    Pick a random row interval for a segment on anti-diagonal k.

    If constrain_lo/hi are given, the interval must overlap that row range by at least
    OVERLAP_MIN_ROWS rows so that the new horizontal scan definitely sees both segments.
    """
    row_lo, row_hi = _row_bounds_for_k(k)
    if row_lo > row_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        if row_lo + length - 1 > row_hi:
            continue
        candidates = list(range(row_lo, row_hi - length + 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                start
                for start in candidates
                if min(start + length - 1, constrain_hi) - max(start, constrain_lo) + 1 >= OVERLAP_MIN_ROWS
            ]
            if not candidates:
                continue
        start = random.choice(candidates)
        return start, start + length - 1
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    """Try to split one segment into two 8-disconnected pieces on the same anti-diagonal."""
    k, row_lo, row_hi = segs[idx]
    length = row_hi - row_lo + 1
    if length < SPLIT_MIN_LEN:
        return

    _, k_row_hi = _row_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        piece1_len = random.randint(2, length - 2)
        piece1_hi = row_lo + piece1_len - 1
        p2_start_candidates = list(range(piece1_hi + SPLIT_ROW_GAP + 1, k_row_hi))
        if not p2_start_candidates:
            continue
        piece2_lo = random.choice(p2_start_candidates)
        piece2_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        piece2_hi = piece2_lo + piece2_len - 1
        if piece2_hi > k_row_hi:
            continue
        segs[idx] = (k, row_lo, piece1_hi)
        segs.append((k, piece2_lo, piece2_hi))
        return


def _multi_row_count(grid: np.ndarray, line: int) -> int:
    """Number of rows containing at least two line pixels."""
    return sum(int(np.count_nonzero(grid[r] == line) >= 2) for r in range(H))


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the horizontal-scan variant."""
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
        res = _random_seg(k, prev_lo, prev_hi)
        if res is None:
            return None
        segs.append((k, res[0], res[1]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, row_lo, row_hi in segs:
        for r, c in _seg_cells(k, row_lo, row_hi):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    if _multi_row_count(grid, line) < MIN_MULTI_ROWS:
        return None

    output = _apply_fill_rows(grid)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    old_output = _apply_fill_diagonal_old(grid, segs)
    if int(np.sum(output != old_output)) < MIN_DIFFERENCE_FROM_OLD:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

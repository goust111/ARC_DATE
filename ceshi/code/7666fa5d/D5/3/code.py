# Variant of 7666fa5d
# Dimension: D5 — 变换操作：填充颜色规则 (transform)
# Change: 所有填充区域统一使用红色(color 2)，不受线段颜色影响
#   -> 填充颜色由相邻线段对中较短线段的长度编码

"""
design_analysis:
  variant_core:
    original_rule: |
      对每条主对角线 b=r-c，找所有覆盖该 b 的反对角线段，按 k=r+c 排序，
      在相邻线段之间把背景填成统一的红色 2。
    new_rule: |
      几何扫描完全不变，但每一对相邻 active segments 的填充色不再固定。
      先取这两个 segment 的长度（单元格数），使用较短者 short_len 作为编码参数，
      再通过固定查找表 short_len -> fill_color 着色整个对应平行四边形区域。
    lookup_table:
      2: 2
      3: 3
      4: 4
      5: 6

  concrete_behavior_trace:
    example:
      left_segment_length: 5
      right_segment_length: 3
      original: "两线段之间所有背景格都会被写成 2。"
      variant: "较短长度=min(5,3)=3，因此该区域全部改写成 3。"
      difference: "几何边界完全一致，仅填充色随线段长度档位变化。"

  complexity_calibration:
    decision: |
      只使用 4 个长度档位 {2,3,4,5}，对应 4 个固定颜色 {2,3,4,6}。
      这样自由度足够产生多色填充，但不会多到需要从少量样例中猜大表。
    generator_support:
      - 每个样本至少出现 2 种填充颜色。
      - 每个样本至少有一种非红填充，避免退化成原题。

  implicit_assumption_audit:
    original_line: "fill = 2"
    issue: "原 solver 假设所有填充共享同一个常量颜色；D5 下该前提失效。"
    original_line_2: "if ... and out[r, c] == bg: out[r, c] = fill"
    issue_2: |
      无条件写单一 fill 在原题无害；现在必须按当前 segment pair 的 short_len
      动态查表，否则不同区域会被错误合并成同色。
    preserved_assumptions:
      - 输入里仍只有一种线段颜色；generator 保证。
      - 每个连通分量仍是单个反对角线段；generator 保证。
      - 所有 segment 长度都来自查找表键集 {2,3,4,5}；generator 保证。

  placement_legality:
    - 所有 segment 仍放在 16x16 内部的单一 anti-diagonal 上。
    - consecutive k 的 b-range 仍要求重叠，保证存在可填区域。
    - split 仅在同一 k 上制造 8-不连通的第二段，不改变主规则。
    - line_color 避开所有 fill colors，便于从 output 中删除填充恢复 input。

  visual_observability:
    - 几何形状与原题相同，人类容易先锁定“填哪儿”。
    - 新信息只体现在颜色差异上，且颜色档位少、重复出现，有利于从训练样例总结映射。
    - 通过限制 line_color 不与 fill_color 冲突，输出中的彩色平行四边形边界清楚。

  color_inference:
    rationale: |
      颜色变化不能依赖不可见数值，因此显式绑定到可数的 segment length。
      样例中可直接数出边界线段长度，再对照输出区域颜色归纳查表。

  cleanliness:
    - 输入仍只有背景+线段，极简。
    - 输出只有边界线段和 2-4 种填充色，不引入额外标记层。

  solver_generator_contract_after_change:
    main_assumptions:
      - 线段长度必须在 LENGTH_TO_COLOR 中。
      - active segments 的排序仍仅由 k 决定。
      - 填充只覆盖背景格，不覆盖线段。
    generator_guarantees:
      - 只采样允许长度。
      - 至少两个不同 fill colors 真正出现在 output。
      - variant output 与旧版 uniform-red output 不同。
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


H = W = 16
BG = 8
LENGTH_TO_COLOR = {2: 2, 3: 3, 4: 4, 5: 6}
ALLOWED_LENGTHS = sorted(LENGTH_TO_COLOR)
FILL_COLORS = sorted(set(LENGTH_TO_COLOR.values()))
LINE_COLORS = [1, 5, 7, 9]

# -- Anti-diagonal (k = r+c) layout --
K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

# -- Segment geometry --
OVERLAP_MIN_B = 2

# -- Segment splitting --
SPLIT_PROB = 0.35
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4

# -- Retry budgets --
SEG_RETRY = 50
SPLIT_RETRY = 12
GEN_RETRY = 500

# -- Validation --
MIN_FILL_CELLS = 6


def _segment_length(b_min: int, b_max: int) -> int:
    return (int(b_max) - int(b_min)) // 2 + 1


def _extract_segments(g: np.ndarray, line: int) -> list[tuple[int, int, int, int]]:
    segs: list[tuple[int, int, int, int]] = []
    for pts in _cc8(g == int(line)):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        b_min = int(min(bs))
        b_max = int(max(bs))
        segs.append((k, b_min, b_max, _segment_length(b_min, b_max)))
    return segs


def _fill_from_segments(
    grid: np.ndarray,
    segs: list[tuple[int, int, int, int]],
    *,
    bg: int,
    uniform_color: int | None = None,
) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])

        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, k_right = left[0], right[0]
            short_len = min(left[3], right[3])
            fill = int(uniform_color) if uniform_color is not None else LENGTH_TO_COLOR[short_len]
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-shaped region between parallel anti-diagonal line segments.

    Input structure:
    - bg = most common color.
    - One non-bg color forms short diagonal segments, each lying on one anti-diagonal (r+c = const).
    - Segments come in parallel pairs bounding regions to be filled.

    Variant rule:
    - Geometry is unchanged from the original task.
    - For each main diagonal b=r-c, collect active segments covering that b and sort by k=r+c.
    - Between each consecutive pair, fill bg cells strictly inside the strip.
    - The fill color is determined by the shorter of the two segment lengths:
      2->2, 3->3, 4->4, 5->6.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))

    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g, int(line))
    assert segs, "grid must contain at least one valid anti-diagonal segment"

    return _fill_from_segments(g, segs, bg=bg)


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
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
    length: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    step = 2 * (int(length) - 1)
    if b_lo + step > b_hi:
        return None

    candidates = list(range(b_lo, b_hi - step + 1, 2))
    if constrain_lo is not None and constrain_hi is not None:
        candidates = [
            b_min
            for b_min in candidates
            if b_min + step >= constrain_lo + OVERLAP_MIN_B and b_min <= constrain_hi - OVERLAP_MIN_B
        ]
    if not candidates:
        return None

    for _ in range(SEG_RETRY):
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int, int]], idx: int) -> None:
    k, b_min, b_max, length = segs[idx]
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        len1_candidates = [l for l in ALLOWED_LENGTHS if 2 <= l <= length - 2]
        if not len1_candidates:
            return
        len1 = random.choice(len1_candidates)
        len2 = random.choice([l for l in ALLOWED_LENGTHS if l <= length - len1 + 1])

        piece1_max = b_min + 2 * (len1 - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        p2_max = p2_min + 2 * (len2 - 1)
        if p2_max > b_hi:
            continue

        segs[idx] = (k, b_min, piece1_max, len1)
        segs.append((k, p2_min, p2_max, len2))
        return


def _sample_lengths(num_ks: int) -> list[int]:
    while True:
        lengths = [random.choice(ALLOWED_LENGTHS) for _ in range(num_ks)]
        pair_short = {min(a, b) for a, b in zip(lengths, lengths[1:])}
        if len(pair_short) >= 2 and any(v != 2 for v in pair_short):
            return lengths


def _output_fill_colors(output: np.ndarray, line: int) -> set[int]:
    return {int(v) for v in np.unique(output) if int(v) in FILL_COLORS and int(v) != int(line)}


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)

    ks = _choose_k_values(parity)
    if ks is None:
        return None

    lengths = _sample_lengths(len(ks))

    segs: list[tuple[int, int, int, int]] = []
    first = _random_seg(ks[0], lengths[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], lengths[0]))

    for k, length in zip(ks[1:], lengths[1:]):
        prev_b_min, prev_b_max = segs[-1][1], segs[-1][2]
        res = _random_seg(k, length, prev_b_min, prev_b_max)
        if res is None:
            return None
        segs.append((k, res[0], res[1], length))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max, _length in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            grid[r, c] = line

    output = _fill_from_segments(grid, segs, bg=BG)
    uniform_output = _fill_from_segments(grid, segs, bg=BG, uniform_color=2)

    fill_mask = np.isin(output, FILL_COLORS)
    if int(np.sum(fill_mask)) < MIN_FILL_CELLS:
        return None

    used_fill_colors = _output_fill_colors(output, line)
    if len(used_fill_colors) < 2 or used_fill_colors == {2}:
        return None

    if np.array_equal(output, uniform_output):
        return None

    inp = output.copy()
    inp[np.isin(inp, FILL_COLORS)] = BG

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

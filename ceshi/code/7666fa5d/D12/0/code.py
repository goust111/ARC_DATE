# Variant of 7666fa5d
# Dimension: D12 — 背景色与颜色体系 (meta)
# Change: 背景色为azure(8)，线段颜色为单一非背景非红色(可为3/4/9等)，填充色固定为红色(2)... → 多色线段系统

"""
Design analysis for the D12 variant
===================================

source_task:
  reference_task: None
  note: 无需额外阅读其他 solver；只在原题机制上改颜色体系。

variant_spec:
  grid: 16x16
  background: 8
  objects:
    - 4-5 个反对角线索引 k=r+c，上面放置短线段
    - 每个 k 有自己的线段颜色；颜色来自一个 3 色调色板并按 k 递增循环分配
    - 可选地把某个 k 上的一条线段拆成两个 8-不连通片段；两片保持同色
  output_rule:
    - 仍按主对角线 b=r-c 扫描
    - 找出覆盖该 b 的所有线段，按 k 从小到大排序
    - 相邻两条活跃线段之间的背景单元全部填充
    - 填充色不再固定为 2，而是复制较小 k（更靠左上）的边界线段颜色

a) concrete behavior trace:
  - 例：某条 b 线上，k=8 的绿色线段与 k=12 的黄色线段都覆盖该 b。
  - 原题：k 在 (8,12) 之间的背景格一律填成红色 2。
  - 本变体：这些背景格填成绿色，因为它们位于绿色线段与黄色线段之间，
    且规则是“取左上侧边界线段的颜色”。
  - 差异：空间几何完全不变，只有颜色传播来源改变；于是 output 中不同平行四边形带
    会呈现不同颜色，而不是统一红色。

b) complexity calibration:
  - 不使用太多颜色：每题固定选 3 种线段色，在 4-5 条 k 线上循环复用。
  - 这样每个样例里通常会出现 3 种线色、2-3 种填充色，变化肉眼明显但不杂乱。
  - 若只给 2 色或允许所有线段同色，会弱化 D12；生成器因此强制至少 3 条不同颜色线。

c) implicit assumption audit on original main():
  - `non_bg` 后选出现最多的非背景色作为 `line`：旧前提是“所有线段同色”。变体不成立。
  - `fill = 2`：旧前提是“填充色固定红色”。变体不成立。
  - `components_mask(g == line)`：旧前提是“只需要看一个颜色层”。变体不成立；
    现在必须遍历所有非背景颜色并保留每个组件自己的颜色。
  - `segs.append((k,bmin,bmax))`：旧前提是颜色不重要。变体中必须改成保存
    `(k,bmin,bmax,color)`。
  - `out[r,c] = fill`：旧前提是任何填充都同色。变体中必须改成取左边界线段颜色，
    但仍然只能写到背景格，避免覆盖真实线段。

d) placement legality:
  - 仍沿用原题的 k 选择、b 范围、越界检查、相邻线段 b-overlap 约束。
  - 同一 k 若被拆成两段，两段共用一个颜色且 b 范围分离，因此不会在同一 b 上同时激活。
  - 颜色只在已合法放置的线段之间传播，不会引入额外几何冲突。

e) visual observability:
  - 规则不依赖“较小颜色值”之类不可见数值比较，只依赖可见空间关系：
    填充区域继承左上边界颜色。
  - 在 2-4 组样例中，人类可以看到：每个填充带都和其左上边界同色，而不是统一红色。

f) visual cleanliness:
  - 仍是细线段 + 平行四边形填充，没有额外噪声或标记层。
  - 调色板限制为 3 个高区分度颜色，避免整张图颜色过多。

g) color-rule inferability:
  - 同一题内重复使用 3 色循环，输出里会出现多种填充色，且每种都紧贴对应左上边界。
  - 这给出了足够的结构线索，避免“颜色变化存在但无法归纳”的问题。

new solver assumptions to audit after the rewrite:
  - 每个输入非背景 8-连通块都应是单一 k 的线段；生成器保证这一点。
  - 相邻待填区域之间仍以背景分隔；生成器不放置额外噪声对象。
  - 主对角线扫描时，相邻活跃线段按 k 有明确顺序；生成器保证不同 k 递增且 b-overlap 充分。
"""

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


Segment = tuple[int, int, int, int]  # (k, b_min, b_max, color)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(grid: np.ndarray, bg: int) -> list[Segment]:
    segs: list[Segment] = []
    for color in sorted(set(map(int, grid.ravel()))):
        if color == bg:
            continue
        for pts in _cc8(grid == color):
            ks = {r + c for r, c in pts}
            if len(ks) != 1:
                continue
            bs = [r - c for r, c in pts]
            segs.append((int(next(iter(ks))), int(min(bs)), int(max(bs)), int(color)))
    return segs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: fill each gap between consecutive active anti-diagonal segments
    with the color of the lower-k (top-left) boundary segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    segs = _extract_segments(g, bg)
    assert segs, "grid must contain at least one anti-diagonal segment"

    out = g.copy()
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda s: s[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, k_right, fill_color = left[0], right[0], left[3]
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill_color

    return out


# --------------- generator ---------------

H = W = 16
BG = 8
LINE_COLORS = [3, 4, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 2

SPLIT_PROB = 0.4
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 300

MIN_FILL_CELLS = 5
MIN_FILL_COLOR_KINDS = 2
PALETTE_SIZE = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_fill(grid: np.ndarray, segs: list[Segment]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda s: s[0])
        if len(active) < 2:
            continue
        r_min, r_max = max(0, b), min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, k_right, fill_color = left[0], right[0], left[3]
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = fill_color
    return out


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
    k: int, constrain_lo: int | None = None, constrain_hi: int | None = None
) -> tuple[int, int] | None:
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


def _try_split_segment(segs: list[Segment], idx: int) -> None:
    k, bmin, bmax, color = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        first_len = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (first_len - 1)
        p2_candidates = [
            b
            for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2)
            if b % 2 == k % 2
        ]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        second_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (second_len - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max, color)
        segs.append((k, p2_min, p2_max, color))
        return


def _choose_palette() -> list[int]:
    palette = random.sample(LINE_COLORS, PALETTE_SIZE)
    if random.random() < 0.5:
        palette.reverse()
    return palette


def _assign_colors(ks: list[int]) -> dict[int, int]:
    palette = _choose_palette()
    offset = random.randrange(len(palette))
    return {k: palette[(offset + i) % len(palette)] for i, k in enumerate(ks)}


def _fill_colors_in_output(inp: np.ndarray, out: np.ndarray) -> set[int]:
    return {int(out[r, c]) for r, c in np.argwhere(inp == BG) if out[r, c] != BG}


def _gen_attempt() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    color_by_k = _assign_colors(ks)

    segs: list[Segment] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], color_by_k[ks[0]]))

    for k in ks[1:]:
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        res = _random_seg(k, prev_bmin, prev_bmax)
        if res is None:
            return None
        segs.append((k, res[0], res[1], color_by_k[k]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax, color in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = color

    output = _apply_fill(grid, segs)
    fill_mask = (grid == BG) & (output != BG)
    if int(np.sum(fill_mask)) < MIN_FILL_CELLS:
        return None
    if len(_fill_colors_in_output(grid, output)) < MIN_FILL_COLOR_KINDS:
        return None
    if len({color_by_k[k] for k in ks}) < PALETTE_SIZE:
        return None

    inp = output.copy()
    inp[fill_mask] = BG

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

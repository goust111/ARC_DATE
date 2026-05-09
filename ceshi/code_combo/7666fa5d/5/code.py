# Combo variant of 7666fa5d (index 5)
# combo_spec: D5/0+D12/0+D14/0
# Name: 多色线段+同色填充+图例指引

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


"""
Design analysis for the combo variant
=====================================

source_task:
  reference_task: 7666fa5d
  combo_spec: D5/0 + D12/0 + D14/0
  summary:
    - 保留原题几何：主体仍是若干反对角线 k=r+c 上的短线段
    - 引入多色：不同 k 使用 3 色循环
    - 引入图例：左上角固定黑色图例块逐行给出 line -> fill
    - 引入动态填充：每条主对角线 b=r-c 上，相邻活跃线段之间的填充色取左侧较小 k 的线段色

a) concrete behavior trace:
  - 设某个样例里 legend 三行分别是 3->3, 6->6, 9->9。
  - 主体四条线段按 k 递增着色为 [3, 6, 9, 3]。
  - 原题在任意相邻线段之间都填红色 2。
  - 本融合版在同一条 b 线上：
    * 若活跃相邻对是 (k=9,色3) 与 (k=13,色6)，中间背景格填 3。
    * 若活跃相邻对是 (k=13,色6) 与 (k=17,色9)，中间背景格填 6。
    * 若活跃相邻对是 (k=17,色9) 与 (k=21,色3)，中间背景格填 9。
  - 差异有三层：
    * D12: 不再只有一种线段色。
    * D5: 填充色不再固定，而是随左边界线段变化。
    * D14: 角落图例把“哪种线段色会产出哪种填充色”明示出来。

b) complexity calibration:
  - 每题固定使用 3 种主体颜色，并在 4-5 条 k 线上循环复用，避免颜色过多导致噪声。
  - 图例也固定为 3 行，和主体颜色一一对应，给解题者稳定重复的支撑。
  - 保留原题的 4-5 条 k、2-9 长度、相邻 b-overlap 等几何约束，保证形状仍是可辨认的平行四边形带。
  - 不让线段使用红色 2，这样输出不会退化成“看起来像原题的统一红填充”。

c) implicit assumption audit on original main():
  - `line = max(non_bg, key=...)` 依赖“所有主体线段同色”；融合后不成立。
  - `fill = 2` 依赖“填充色全局固定”；融合后不成立。
  - `components_mask(g == line)` 依赖“只看一个颜色层就能取全主体”；融合后不成立。
  - 原题没有额外符号区；融合后若不先抹掉 legend，会把角落色块误识别为主体组件。
  - 原题 `zip(ks, ks[1:])` 只传递几何；融合后必须让每个 segment 同时携带 color，
    并在每个相邻 pair 上读取左侧颜色对应的 fill。
  - D14 单变体里曾可要求左右线段同色才填；本融合版不成立，因为相邻线段通常异色，
    规则改成“无论右侧颜色是什么，都取左侧颜色/图例值”。

d) placement legality:
  - 图例固定占据左上角 5x5 黑色块，最大 k 为 8。
  - 生成器把主体限制到 k>=9，因此线段和填充都不会进入图例区域。
  - 每条线段仍通过 `_b_bounds_for_k` 与 parity 对齐保证不越界。
  - 相邻 k 的线段在 b 轴上至少重叠 2，保证每一对邻接线段至少能形成一段填充。
  - 若对某个 k 做 split，两段保持同色、同 k 且 8-不连通；之后再用输出约束筛掉无效样本。

e) visual observability:
  - 人类能直接看到：不同填充带的颜色随其左上边界变化，而不是固定红色。
  - 图例重复展示 3 个 line->fill 对照行，给出显式颜色线索，不依赖不可见的颜色编号比较。
  - 因为 3 色循环在同一图内多次复现，2-3 个样例就足以支持“左侧线段色决定中间带颜色”的归纳。

f) visual cleanliness:
  - 图例被隔离在固定角落，用黑底包住，不与主体对角线结构混杂。
  - 主体区只保留线段与填充带两层视觉元素，没有额外噪声标记。
  - 调色板限制为 3 色，既能形成明显差异，又不会让 16x16 网格显得过杂。

g) change inferability:
  - 颜色变化不是任意的：只有 legend 出现的 3 种线段色会出现在主体里。
  - 图例右侧 swatch 与左侧 swatch 相同，明确告诉解题者“这种线段会生成同色填充”。
  - 结合原题稳定几何，解题重点自然落在“哪一条边界提供颜色”，而不是猜测复杂的颜色代数。

new main() assumptions review:
  - 假设 1: 左上角 5x5 一定是 legend。
    Generator guarantee: 每个样例都先保留这块区域并最终写入图例。
  - 假设 2: legend 至少给出一行、实际给出三行有效映射。
    Generator guarantee: 固定写入三行 `line -> same color`。
  - 假设 3: 主体只使用 legend keys 中的颜色。
    Generator guarantee: `color_by_k` 仅从 legend palette 取值。
  - 假设 4: 每个被识别的主体连通块都落在单一 k 上。
    Generator guarantee: 所有线段都由 `_seg_cells(k, ...)` 构造。
  - 假设 5: 任意需要填充的位置在输入里都是背景色。
    Generator guarantee: 输入由输出去除主体填充得到，不放置额外对象。
"""


BLACK = 0
H = W = 16
BG = 8

LEGEND_R0 = 0
LEGEND_C0 = 0
LEGEND_H = 5
LEGEND_W = 5
LEGEND_ROWS = (1, 2, 3)
LEGEND_LINE_COL = 1
LEGEND_FILL_COL = 3

LINE_COLORS = [3, 4, 6, 7, 9]
PALETTE_SIZE = 3

K_LO, K_HI = 9, 26
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
GEN_RETRY = 400

MIN_FILL_CELLS = 6
MIN_FILL_COLOR_KINDS = 2

Segment = tuple[int, int, int, int]  # (k, b_min, b_max, color)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W] = True
    return mask


def _read_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    box = grid[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W]
    mapping: dict[int, int] = {}
    for rr in range(box.shape[0]):
        entries = [
            (int(cc), int(box[rr, cc]))
            for cc in range(box.shape[1])
            if int(box[rr, cc]) not in (int(bg), BLACK)
        ]
        if len(entries) != 2:
            continue
        entries.sort(key=lambda t: t[0])
        mapping[entries[0][1]] = entries[1][1]
    assert mapping, "legend mapping missing"
    return mapping


def _draw_legend(grid: np.ndarray, legend: dict[int, int]) -> None:
    grid[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W] = BLACK
    for rr, line in zip(LEGEND_ROWS, sorted(legend)):
        grid[LEGEND_R0 + rr, LEGEND_C0 + LEGEND_LINE_COL] = int(line)
        grid[LEGEND_R0 + rr, LEGEND_C0 + LEGEND_FILL_COL] = int(legend[line])


def _extract_segments(grid: np.ndarray, allowed_colors: set[int]) -> list[Segment]:
    segs: list[Segment] = []
    for color in sorted(allowed_colors):
        for pts in _cc8(grid == int(color)):
            ks = {r + c for r, c in pts}
            if len(ks) != 1:
                continue
            bs = [r - c for r, c in pts]
            segs.append((int(next(iter(ks))), int(min(bs)), int(max(bs)), int(color)))
    return segs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gaps between colored anti-diagonal segments, but choose each fill color
    from the left boundary segment via the corner legend.

    Input structure:
    - bg is the most common color, and a fixed 5x5 black legend sits in the top-left corner.
    - Three legend rows encode line-color -> fill-color pairs; in this combo those pairs are same-color copies.
    - Away from the legend, short 8-connected segments lie on anti-diagonals r+c=k, and different k-values can use different legend colors.

    Rule:
    - Read the legend first, then ignore that corner patch while extracting body segments.
    - For each main diagonal b=r-c, collect all body segments whose b-span covers it and sort them by k.
    - Fill every background cell strictly between consecutive active segments using the legend value of the lower-k segment.

    Key insight:
    The shape logic is still purely geometric, but color now travels locally: each filled band inherits the color of its top-left boundary segment, and the legend confirms that each line color maps to the same fill color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    legend = _read_legend(g, bg)

    body = g.copy()
    body[_legend_mask(g.shape)] = bg
    segs = _extract_segments(body, set(legend))
    assert segs, "grid must contain legend-referenced anti-diagonal segments"

    out = g.copy()
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda s: s[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, k_right = left[0], right[0]
            fill_color = int(legend[left[3]])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill_color

    return out


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _apply_fill(grid: np.ndarray, segs: list[Segment], legend: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda s: s[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, k_right = left[0], right[0]
            fill_color = int(legend[left[3]])
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
            b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2
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


def _assign_colors(ks: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    palette = _choose_palette()
    offset = random.randrange(len(palette))
    ordered = [palette[(offset + i) % len(palette)] for i in range(len(ks))]
    color_by_k = {k: ordered[i] for i, k in enumerate(ks)}
    legend = {color: color for color in sorted(set(palette))}
    return color_by_k, legend


def _body_fill_mask(body_inp: np.ndarray, body_out: np.ndarray) -> np.ndarray:
    return (body_inp == BG) & (body_out != BG)


def _fill_colors_in_output(body_inp: np.ndarray, body_out: np.ndarray) -> set[int]:
    return {int(body_out[r, c]) for r, c in np.argwhere(_body_fill_mask(body_inp, body_out))}


def _gen_attempt() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    color_by_k, legend = _assign_colors(ks)
    if len(legend) != PALETTE_SIZE:
        return None

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

    body = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax, color in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if body[r, c] != BG:
                return None
            body[r, c] = color

    if np.any(body[_legend_mask(body.shape)] != BG):
        return None

    body_out = _apply_fill(body, segs, legend)
    fill_mask = _body_fill_mask(body, body_out)
    if int(np.sum(fill_mask)) < MIN_FILL_CELLS:
        return None

    fill_colors = _fill_colors_in_output(body, body_out)
    if len(fill_colors) < MIN_FILL_COLOR_KINDS:
        return None

    if len(set(color_by_k.values())) != PALETTE_SIZE:
        return None

    inp = body_out.copy()
    inp[fill_mask] = BG
    out = body_out.copy()

    _draw_legend(inp, legend)
    _draw_legend(out, legend)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")

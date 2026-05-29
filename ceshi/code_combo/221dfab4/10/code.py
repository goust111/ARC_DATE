# Combo variant of 221dfab4 (index 10)
# combo_spec: D1/1+D10/0
# Name: 双向推进+宽度递增

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

Y = 4
G = 3
MIN_TRAVEL_MARGIN = 10
MIN_WIDTH_MARGIN = 5
PREVIEW_STEPS = 5

"""
融合设计分析: D1/1 + D10/0

a) 具体化与行为追踪
- 原题: 黄色线段贴边, 只朝一个方向每隔 2 格推进, 条纹宽度始终等于线段跨度.
- 融合后: 黄色线段放在内部, 把该行/列当作 step0 的中心条纹; step1 出现在两侧距离 2 的位置,
  step2 出现在两侧距离 4 的位置, 颜色继续按 Y,Y,G 循环.
- 同时每个 step 的显式条纹宽度从 base_len + step 递增. 设水平线段初始跨度是列 6..9:
  step0: 中心行画 [6..9] 黄;
  step1: 上下两行各画 [6..10] 黄;
  step2: 再外侧两行各画 [5..10] 绿, 且整行已有非背景像素都改成绿;
  gap : 中心与 step1 之间的 gap 清 [6..10], step1 与 step2 之间的 gap 清 [5..10].
- 与原题的关键差异不再是 "一束固定宽度射线", 而是 "从中心向两边张开的对称喇叭形条纹".

b) 复杂度校准
- 若中心线离上下/左右边界太近, 双向传播会退化成单向, 所以生成器强制法向两侧保留至少
  `MIN_TRAVEL_MARGIN` 空间, 确保 step2 和 step5 这两档绿色条纹有机会在两侧同时出现.
- 若线段本身离侧边太近, 递增宽度会被立刻裁切, 看起来像常宽条纹; 因此生成器还要求跨度方向
  保留 `MIN_WIDTH_MARGIN` 空间, 并检查至少多档宽度真正增长未被裁切.
- 若绿色条纹两侧都没有命中前景, 人类只能看到对称黄条; 所以生成器显式放置两枚锚点 blob,
  保证两侧至少各有一条绿色整行/整列重染.

c) 隐式假设审查
- 原 `smask = (g == Y) & border` 假设黄线在边缘; 融合后失效, 必须改成全图寻找最大黄色直线段.
- 原 `dir1` 只编码单向推进; 融合后每个 step 需要同时处理 `start_pos - 2*step` 和
  `start_pos + 2*step`, 且某一侧越界时另一侧仍可继续.
- 原 `span` 是固定区间; 融合后显式绘制和 gap 清除都要改用 `expanded_interval(step)`.
- 原绿色条纹的全行/列重染逻辑仍成立, 但现在要在两个对称位置执行, 且显式条纹覆盖范围更宽.
- 新增假设是 "输入里最大的黄色连通块是一条连续水平或垂直线段"; 生成器通过只放置唯一黄段保证.

d) 放置合法性
- 所有 blob 都通过空位检测放置, 不重叠、不越界, 并保留 1 格 padding.
- 任何 blob 都不得占据中心黄线所在的整行/整列, 这样 yellow seed 唯一且易于抽取.
- 两个锚点 blob 分别被放到中心线两侧, 且命中对称绿色条纹之一, 从而保证融合规则实际发生.

e) 视觉可观察性
- 输入中黄线明显位于中部而不是边缘, 解题者先能观察到 "传播起点" 改了.
- 输出中两侧条纹距离中心相同, 直接暴露 D1 的双向规则.
- 条纹越远越宽, 且 gap 也随之外扩, 人眼能稳定看到 D10 的扇形边界.
- 两侧绿色整行/整列重染为绿色, 说明这不是单纯画几条平行黄线, 而是沿原题机制对称展开.

main() 新增隐式假设与生成保证
- `_extract_segment()` 假设最大黄色块连续且笔直; 生成器只放一个内部黄段.
- `_expanded_interval()` 假设增长宽度由 step 唯一决定; 生成器避免多个黄段造成歧义.
- 双向 step 循环假设中心线两侧有足够样本支持; 生成器强制中心留白并检查两侧绿色命中.
- baseline 对比指标 `fan_extra_pixels` 假设融合输出应区别于仅 D1 的常宽版本; 生成器据此拒绝退化样本.
"""


def _extract_segment(grid: np.ndarray) -> dict[str, int] | None:
    smask = grid == Y
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))
    if pts.size == 0:
        return None

    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        cols = sorted(int(c) for c in cs)
        if cols != list(range(int(cols[0]), int(cols[-1]) + 1)):
            return None
        return {
            "orient": 0,
            "start_pos": int(r0),
            "span_lo": int(cols[0]),
            "span_hi": int(cols[-1]),
            "travel_limit": int(grid.shape[0]),
            "span_limit": int(grid.shape[1]),
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(int(rows[0]), int(rows[-1]) + 1)):
            return None
        return {
            "orient": 1,
            "start_pos": int(c0),
            "span_lo": int(rows[0]),
            "span_hi": int(rows[-1]),
            "travel_limit": int(grid.shape[1]),
            "span_limit": int(grid.shape[0]),
        }

    return None


def _expanded_interval(span_lo: int, span_hi: int, step: int, limit: int) -> tuple[int, int]:
    grow_left = int(step) // 2
    grow_right = int(step) - int(grow_left)
    lo = max(0, int(span_lo) - int(grow_left))
    hi = min(int(limit) - 1, int(span_hi) + int(grow_right))
    return int(lo), int(hi)


def _step_positions(start_pos: int, limit: int, step: int) -> list[int]:
    if int(step) == 0:
        return [int(start_pos)] if 0 <= int(start_pos) < int(limit) else []

    offset = 2 * int(step)
    positions: list[int] = []
    for sign in (-1, 1):
        pos = int(start_pos) + sign * int(offset)
        if 0 <= pos < int(limit) and pos not in positions:
            positions.append(int(pos))
    return positions


def _gap_positions(start_pos: int, limit: int, step: int) -> list[int]:
    offset = 2 * int(step) + 1
    positions: list[int] = []
    for sign in (-1, 1):
        pos = int(start_pos) + sign * int(offset)
        if 0 <= pos < int(limit) and pos not in positions:
            positions.append(int(pos))
    return positions


def _green_positions_by_side(start_pos: int, limit: int) -> tuple[list[int], list[int]]:
    neg: list[int] = []
    pos: list[int] = []
    step = 1
    while True:
        positions = _step_positions(start_pos, limit, step)
        if not positions:
            break
        if int(step) % 3 == 2:
            for value in positions:
                if int(value) < int(start_pos):
                    neg.append(int(value))
                elif int(value) > int(start_pos):
                    pos.append(int(value))
        step += 1
    return neg, pos


def _solve_with_fixed_width(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    seg = _extract_segment(g)
    if seg is None:
        return out

    start_pos = int(seg["start_pos"])
    span_lo = int(seg["span_lo"])
    span_hi = int(seg["span_hi"])
    limit = int(seg["travel_limit"])
    orient = int(seg["orient"])
    cycle = [Y, Y, G]

    step = 0
    while True:
        positions = _step_positions(start_pos, limit, step)
        if not positions:
            break

        color = int(cycle[int(step) % 3])
        for pos in positions:
            if orient == 0:
                if color == G:
                    out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
                out[pos, span_lo : span_hi + 1] = color
            else:
                if color == G:
                    out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
                out[span_lo : span_hi + 1, pos] = color

        for gap in _gap_positions(start_pos, limit, step):
            if orient == 0:
                out[gap, span_lo : span_hi + 1] = bg
            else:
                out[span_lo : span_hi + 1, gap] = bg

        step += 1

    return out


def variant_metrics(grid: np.ndarray, preview_steps: int = PREVIEW_STEPS) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    seg = _extract_segment(g)
    if seg is None:
        return {
            "center_clearance_min": 0,
            "side_clearance_min": 0,
            "growth_steps_unclipped": 0,
            "neg_green_hits": 0,
            "pos_green_hits": 0,
            "fan_extra_pixels": 0,
        }

    orient = int(seg["orient"])
    start_pos = int(seg["start_pos"])
    span_lo = int(seg["span_lo"])
    span_hi = int(seg["span_hi"])
    limit = int(seg["travel_limit"])
    span_limit = int(seg["span_limit"])
    base_width = int(span_hi) - int(span_lo) + 1
    neg_greens, pos_greens = _green_positions_by_side(start_pos, limit)

    if orient == 0:
        neg_hits = sum(int(bool(((g[r, :] != bg) & (g[r, :] != Y)).any())) for r in neg_greens)
        pos_hits = sum(int(bool(((g[r, :] != bg) & (g[r, :] != Y)).any())) for r in pos_greens)
    else:
        neg_hits = sum(int(bool(((g[:, c] != bg) & (g[:, c] != Y)).any())) for c in neg_greens)
        pos_hits = sum(int(bool(((g[:, c] != bg) & (g[:, c] != Y)).any())) for c in pos_greens)

    growth_steps_unclipped = 0
    for step in range(1, int(preview_steps) + 1):
        lo, hi = _expanded_interval(span_lo, span_hi, step, span_limit)
        if int(hi) - int(lo) + 1 == int(base_width) + int(step):
            growth_steps_unclipped += 1

    fan_extra_pixels = int(np.sum(main(g) != _solve_with_fixed_width(g)))

    return {
        "center_clearance_min": min(int(start_pos), int(limit) - 1 - int(start_pos)),
        "side_clearance_min": min(int(span_lo), int(span_limit) - 1 - int(span_hi)),
        "growth_steps_unclipped": int(growth_steps_unclipped),
        "neg_green_hits": int(neg_hits),
        "pos_green_hits": int(pos_hits),
        "fan_extra_pixels": int(fan_extra_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a symmetric fan of stripes from the largest interior yellow segment.

    Fusion rule:
    - Detect the largest straight yellow segment anywhere in the grid.
    - Use that row/column as step0, then propagate to both sides at distances 2, 4, 6, ...
    - Colors cycle Y, Y, G by step index.
    - Stripe width grows by exactly 1 per step, rounded as floor-left / ceil-right.
    - Gap rows/cols are cleared on the next widened interval.
    - Green stripes still recolor all existing non-background pixels on the full row/column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    seg = _extract_segment(g)
    if seg is None:
        return out

    orient = int(seg["orient"])
    start_pos = int(seg["start_pos"])
    span_lo = int(seg["span_lo"])
    span_hi = int(seg["span_hi"])
    limit = int(seg["travel_limit"])
    span_limit = int(seg["span_limit"])
    cycle = [Y, Y, G]

    step = 0
    while True:
        positions = _step_positions(start_pos, limit, step)
        if not positions:
            break

        color = int(cycle[int(step) % 3])
        lo, hi = _expanded_interval(span_lo, span_hi, step, span_limit)

        for pos in positions:
            if orient == 0:
                if color == G:
                    out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
                out[pos, lo : hi + 1] = color
            else:
                if color == G:
                    out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
                out[lo : hi + 1, pos] = color

        gap_lo, gap_hi = _expanded_interval(span_lo, span_hi, step + 1, span_limit)
        for gap in _gap_positions(start_pos, limit, step):
            if orient == 0:
                out[gap, gap_lo : gap_hi + 1] = bg
            else:
                out[gap_lo : gap_hi + 1, gap] = bg

        step += 1

    return out


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    if not bool(rows_any.any()) or not bool(cols_any.any()):
        return None

    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    trimmed = sprite[r0 : r1 + 1, c0 : c1 + 1]
    return np.array(trimmed, dtype=int)


def _sample_sprite(rng: random.Random, bg: int, fg: int) -> np.ndarray | None:
    for _ in range(20):
        bh = rng.randint(2, 7)
        bw = rng.randint(2, 7)
        density = rng.uniform(0.45, 0.82)
        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[fg],
                connectivity=4,
                background=bg,
            )
        except (AssertionError, RecursionError):
            continue

        sprite = _trim_sprite(np.array(sprite, dtype=int), bg)
        if sprite is None:
            continue

        pix = int(np.sum(sprite != bg))
        if 5 <= pix <= 40:
            return sprite
    return None


def _can_place(canvas: np.ndarray, sprite: np.ndarray, pr: int, pc: int, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    sh, sw = sprite.shape
    if pr < 1 or pc < 1 or pr + sh > H - 1 or pc + sw > W - 1:
        return False

    mask = sprite != int(bg)
    if not bool(mask.any()):
        return False

    r0 = max(0, pr - padding)
    r1 = min(H, pr + sh + padding)
    c0 = max(0, pc - padding)
    c1 = min(W, pc + sw + padding)
    neighborhood = canvas[r0:r1, c0:c1]
    if bool((neighborhood != int(bg)).any()):
        return False

    return True


def _place_sprite(canvas: np.ndarray, sprite: np.ndarray, pr: int, pc: int, bg: int) -> None:
    sh, sw = sprite.shape
    for dr in range(sh):
        for dc in range(sw):
            if int(sprite[dr, dc]) != int(bg):
                canvas[pr + dr, pc + dc] = int(sprite[dr, dc])


def _place_anchor_blob(
    rng: random.Random,
    canvas: np.ndarray,
    sprite: np.ndarray,
    orient: int,
    start_pos: int,
    anchor_pos: int,
    bg: int,
) -> bool:
    H, W = canvas.shape
    sh, sw = sprite.shape
    row_candidates: list[int] = []
    col_candidates: list[int] = []

    if orient == 0:
        row_min = max(1, int(anchor_pos) - sh + 1)
        row_max = min(int(anchor_pos), H - sh - 1)
        row_candidates = list(range(row_min, row_max + 1))
        rng.shuffle(row_candidates)

        col_candidates = list(range(1, W - sw))
        rng.shuffle(col_candidates)

        for pr in row_candidates:
            if int(anchor_pos) < int(start_pos) and pr + sh - 1 >= int(start_pos):
                continue
            if int(anchor_pos) > int(start_pos) and pr <= int(start_pos):
                continue
            for pc in col_candidates:
                if _can_place(canvas, sprite, pr, pc, bg):
                    _place_sprite(canvas, sprite, pr, pc, bg)
                    return True
    else:
        col_min = max(1, int(anchor_pos) - sw + 1)
        col_max = min(int(anchor_pos), W - sw - 1)
        col_candidates = list(range(col_min, col_max + 1))
        rng.shuffle(col_candidates)

        row_candidates = list(range(1, H - sh))
        rng.shuffle(row_candidates)

        for pc in col_candidates:
            if int(anchor_pos) < int(start_pos) and pc + sw - 1 >= int(start_pos):
                continue
            if int(anchor_pos) > int(start_pos) and pc <= int(start_pos):
                continue
            for pr in row_candidates:
                if _can_place(canvas, sprite, pr, pc, bg):
                    _place_sprite(canvas, sprite, pr, pc, bg)
                    return True

    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(28, 34)
    W = rng.randint(28, 34)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)
    orient = rng.choice([0, 1])
    seg_len = rng.randint(4, 7)

    grid = np.full((H, W), bg, dtype=int)

    if orient == 0:
        if H <= 2 * MIN_TRAVEL_MARGIN + 1 or W <= seg_len + 2 * MIN_WIDTH_MARGIN:
            return None
        start_pos = rng.randint(MIN_TRAVEL_MARGIN, H - MIN_TRAVEL_MARGIN - 1)
        seg_start = rng.randint(MIN_WIDTH_MARGIN, W - seg_len - MIN_WIDTH_MARGIN)
        span = list(range(seg_start, seg_start + seg_len))
        grid[start_pos, span] = Y
    else:
        if W <= 2 * MIN_TRAVEL_MARGIN + 1 or H <= seg_len + 2 * MIN_WIDTH_MARGIN:
            return None
        start_pos = rng.randint(MIN_TRAVEL_MARGIN, W - MIN_TRAVEL_MARGIN - 1)
        seg_start = rng.randint(MIN_WIDTH_MARGIN, H - seg_len - MIN_WIDTH_MARGIN)
        span = list(range(seg_start, seg_start + seg_len))
        grid[span, start_pos] = Y

    canvas = grid.copy()

    neg_greens, pos_greens = _green_positions_by_side(start_pos, H if orient == 0 else W)
    if not neg_greens or not pos_greens:
        return None

    anchor_targets = [int(neg_greens[0]), int(pos_greens[0])]
    for anchor_pos in anchor_targets:
        sprite = _sample_sprite(rng, bg, fg)
        if sprite is None:
            return None
        if not _place_anchor_blob(rng, canvas, sprite, orient, start_pos, anchor_pos, bg):
            return None

    target_blobs = rng.randint(6, 9)
    placed = 2
    for _ in range(80):
        if placed >= target_blobs:
            break

        sprite = _sample_sprite(rng, bg, fg)
        if sprite is None:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        sprite_mask = sprite != bg
        blob_rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr, :].any())}
        blob_cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}

        if orient == 0 and int(start_pos) in blob_rows:
            continue
        if orient == 1 and int(start_pos) in blob_cols:
            continue

        _place_sprite(canvas, sprite, pr, pc, bg)
        placed += 1

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    inp = canvas.copy()
    out = main(inp)
    if int(get_bg(inp)) != int(bg):
        return None
    if np.array_equal(inp, out):
        return None

    metrics = variant_metrics(inp)
    if metrics["center_clearance_min"] < MIN_TRAVEL_MARGIN:
        return None
    if metrics["side_clearance_min"] < MIN_WIDTH_MARGIN:
        return None
    if metrics["growth_steps_unclipped"] < 4:
        return None
    if metrics["neg_green_hits"] < 1 or metrics["pos_green_hits"] < 1:
        return None
    if metrics["fan_extra_pixels"] < 20:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")

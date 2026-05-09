# Variant of 221dfab4
# Dimension: D1 — 推进方向 (path)
# Change: 从黄色线段所在边缘沿直线向对侧边缘推进（straight_ray）... → 双向推进（从线段向两侧扩展）

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
MIN_SIDE_MARGIN = 5

"""
Design analysis for the D1 variant

a) 具体化与行为追踪
- 原题把黄色线段放在边缘，只沿一个方向推进；本变体把黄色线段放到网格内部，并沿其法向两侧同时推进。
- 例子：若黄色水平线段位于第 10 行、跨度为列 6..11，则输出条纹是：
  step0: 行 10 涂黄；
  step1: 行 8 和 12 涂黄；
  step2: 行 6 和 14 涂绿，且这些整行已有前景都重染为绿；
  gap : 行 9/11、7/13 仅在线段跨度内清成背景。
- 与原题的差异在 solver 的“方向推进”环节：原来只有一个 `dir1`，现在每个 step 要同时处理 `start_pos - 2*step` 与 `start_pos + 2*step` 两个位置。

b) 复杂度校准
- 若线段离某一边太近，就会退化成近似单向行为；因此生成器强制线段到两侧边缘都至少留 `MIN_SIDE_MARGIN`。
- 若两侧绿色条纹没有命中前景，人眼只能看到对称黄线，绿色“整行/整列洪泛”不明显；因此生成器要求两侧至少各有一条绿色条纹命中前景。

c) 隐式假设审查
- 原 `smask = (g == Y) & border` 假设黄色线段一定在边缘；变体中失效，必须改成在全图找最大黄色连通段。
- 原 `dir1 = 1 if start_pos == 0 else -1` 假设推进方向由边缘决定；变体中不存在这个单向方向，要改成每步双侧位置集合。
- 原 `while` 循环只检查一个 `pos`；变体中一侧越界后另一侧仍可能继续，必须改成“只要任一侧仍在界内就继续”。
- 原 gap 清除只处理单侧；变体中必须同时清上下/左右两侧的 gap。
- `out[pos, :] = np.where(...)` / `out[:, pos] = np.where(...)` 仍然成立，因为绿色条纹的规则本身没有变，只是现在会在两个对称位置执行。

d) 放置合法性
- 生成器继续用 `random_free_location(..., border=1, padding=1)` 防止 blob 重叠、越界和贴边。
- 同时禁止 blob 占据黄色线段所在的中心行/列，保证黄色目标是唯一、清晰、可被 solver 稳定提取。
- 生成完成后还检查：前景在中心线两侧都存在，且两侧绿色条纹都至少命中一次前景。

e) 视觉可观察性
- 输入里黄色线段明显位于中部而非边缘；输出里条纹关于该线段对称展开，这是人眼最容易抓到的变化。
- 两侧都出现绿色整行/整列重染，可直接观察到“不是只朝一边传播”，避免样例退化回原题的单射线印象。
"""


def _extract_segment(grid: np.ndarray) -> dict[str, object] | None:
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
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "orient": "h",
            "start_pos": int(r0),
            "span": list(range(cols[0], cols[-1] + 1)),
            "limit": int(grid.shape[0]),
        }

    if c0 == c1:
        rows = sorted(int(r) for r in rs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "orient": "v",
            "start_pos": int(c0),
            "span": list(range(rows[0], rows[-1] + 1)),
            "limit": int(grid.shape[1]),
        }

    return None


def _step_positions(start_pos: int, limit: int, step: int) -> list[int]:
    if int(step) == 0:
        return [int(start_pos)] if 0 <= int(start_pos) < int(limit) else []

    offset = 2 * int(step)
    positions: list[int] = []
    for sign in (-1, 1):
        pos = int(start_pos) + sign * offset
        if 0 <= pos < int(limit) and pos not in positions:
            positions.append(int(pos))
    return positions


def _gap_positions(start_pos: int, limit: int, step: int) -> list[int]:
    offset = 2 * int(step) + 1
    positions: list[int] = []
    for sign in (-1, 1):
        pos = int(start_pos) + sign * offset
        if 0 <= pos < int(limit) and pos not in positions:
            positions.append(int(pos))
    return positions


def _green_positions_by_side(start_pos: int, limit: int) -> tuple[list[int], list[int]]:
    neg: list[int] = []
    pos: list[int] = []
    step = 1
    while True:
        any_valid = False
        if step % 3 == 2:
            left = int(start_pos) - 2 * int(step)
            right = int(start_pos) + 2 * int(step)
            if 0 <= left < int(limit):
                neg.append(int(left))
                any_valid = True
            if 0 <= right < int(limit):
                pos.append(int(right))
                any_valid = True
        else:
            if 0 <= int(start_pos) - 2 * int(step) < int(limit):
                any_valid = True
            if 0 <= int(start_pos) + 2 * int(step) < int(limit):
                any_valid = True
        if not any_valid:
            break
        step += 1
    return neg, pos


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    seg = _extract_segment(g)
    if seg is None:
        return {
            "center_clearance_min": 0,
            "neg_green_count": 0,
            "pos_green_count": 0,
            "neg_green_hits": 0,
            "pos_green_hits": 0,
            "sided_fg_ok": 0,
        }

    orient = str(seg["orient"])
    start_pos = int(seg["start_pos"])
    limit = int(seg["limit"])
    neg_greens, pos_greens = _green_positions_by_side(start_pos, limit)

    if orient == "h":
        neg_hits = sum(int(bool(((g[r, :] != bg) & (g[r, :] != Y)).any())) for r in neg_greens)
        pos_hits = sum(int(bool(((g[r, :] != bg) & (g[r, :] != Y)).any())) for r in pos_greens)
        sided_fg_ok = int(
            bool(((g[:start_pos, :] != bg) & (g[:start_pos, :] != Y)).any())
            and bool(((g[start_pos + 1 :, :] != bg) & (g[start_pos + 1 :, :] != Y)).any())
        )
    else:
        neg_hits = sum(int(bool(((g[:, c] != bg) & (g[:, c] != Y)).any())) for c in neg_greens)
        pos_hits = sum(int(bool(((g[:, c] != bg) & (g[:, c] != Y)).any())) for c in pos_greens)
        sided_fg_ok = int(
            bool(((g[:, :start_pos] != bg) & (g[:, :start_pos] != Y)).any())
            and bool(((g[:, start_pos + 1 :] != bg) & (g[:, start_pos + 1 :] != Y)).any())
        )

    return {
        "center_clearance_min": min(int(start_pos), int(limit) - 1 - int(start_pos)),
        "neg_green_count": len(neg_greens),
        "pos_green_count": len(pos_greens),
        "neg_green_hits": int(neg_hits),
        "pos_green_hits": int(pos_hits),
        "sided_fg_ok": int(sided_fg_ok),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bidirectionally propagate the original Y,Y,G stripe cycle from an interior yellow segment.

    Rule:
    - Detect the largest straight yellow segment anywhere in the grid.
    - Treat that row/column as the center stripe (yellow).
    - For step distances 2, 4, 6, ... paint symmetric stripes on both sides, cycling Y, Y, G.
    - The 1-cell gaps between stripes are reset to background only within the segment's span.
    - Green stripes still recolor every existing non-background pixel on the full row/column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    seg = _extract_segment(g)
    if seg is None:
        return out

    orient = str(seg["orient"])
    start_pos = int(seg["start_pos"])
    span = [int(x) for x in seg["span"]]
    limit = int(seg["limit"])
    cycle = [Y, Y, G]

    step = 0
    while True:
        positions = _step_positions(start_pos, limit, step)
        if not positions:
            break

        color = int(cycle[step % 3])
        for pos in positions:
            if orient == "h":
                if color == G:
                    out[pos, :] = np.where(out[pos, :] != bg, G, out[pos, :])
                out[pos, span] = color
            else:
                if color == G:
                    out[:, pos] = np.where(out[:, pos] != bg, G, out[:, pos])
                out[span, pos] = color

        for gap in _gap_positions(start_pos, limit, step):
            if orient == "h":
                out[gap, span] = bg
            else:
                out[span, gap] = bg

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


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    orient = rng.choice(["h", "v"])
    seg_len = rng.randint(3, 8)

    grid = np.full((H, W), bg, dtype=int)

    if orient == "h":
        if H < 2 * MIN_SIDE_MARGIN + 1:
            return None
        max_start = W - seg_len - 2
        if max_start < 2:
            return None
        start_pos = rng.randint(MIN_SIDE_MARGIN, H - MIN_SIDE_MARGIN - 1)
        seg_start = rng.randint(2, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        grid[start_pos, span] = Y
    else:
        if W < 2 * MIN_SIDE_MARGIN + 1:
            return None
        max_start = H - seg_len - 2
        if max_start < 2:
            return None
        start_pos = rng.randint(MIN_SIDE_MARGIN, W - MIN_SIDE_MARGIN - 1)
        seg_start = rng.randint(2, max_start)
        span = list(range(seg_start, seg_start + seg_len))
        grid[span, start_pos] = Y

    canvas = grid.copy()
    target_blobs = rng.randint(6, 10)
    placed = 0

    for _ in range(target_blobs * 6):
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

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

        sh, sw = sprite.shape
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue

        sprite_mask = sprite != bg
        blob_rows = {int(pr + dr) for dr in range(sh) if bool(sprite_mask[dr, :].any())}
        blob_cols = {int(pc + dc) for dc in range(sw) if bool(sprite_mask[:, dc].any())}

        if orient == "h" and int(start_pos) in blob_rows:
            continue
        if orient == "v" and int(start_pos) in blob_cols:
            continue

        for dr in range(sh):
            for dc in range(sw):
                if int(sprite[dr, dc]) != int(bg):
                    canvas[pr + dr, pc + dc] = int(sprite[dr, dc])

        placed += 1
        if placed >= target_blobs:
            break

    fg_mask = (canvas != bg) & (canvas != Y)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    metrics = variant_metrics(canvas)
    if metrics["center_clearance_min"] < MIN_SIDE_MARGIN:
        return None
    if metrics["neg_green_count"] == 0 or metrics["pos_green_count"] == 0:
        return None
    if metrics["neg_green_hits"] == 0 or metrics["pos_green_hits"] == 0:
        return None
    if metrics["sided_fg_ok"] == 0:
        return None

    inp = canvas.copy()
    out = main(inp)

    if int(get_bg(inp)) != int(bg):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")

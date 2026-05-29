# Variant of e12f9a14
# Dimension: D2 — 碰撞检测与轨迹变化规则 (trajectory)
# Change: collision_detection——90度交角碰撞时两条射线各转45度并在交叉点留1像素间隙、以单像素绘制；45度... → 碰撞时颜色混合

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# a) 具体化与行为追踪:
#    - 这个变体仍然保留原题的对象形态: 4x4 框, 中间 2x2 纯色块, 边框缺口发射射线.
#    - 关键改动只发生在“碰撞后如何显示”:
#      原题在碰撞后改轨迹, 所以输出里的线段转折、留白、粗细是主要线索.
#      现在射线始终沿原方向直走到边界, 碰撞位置只留下“第三种颜色”的标记.
#    - 为了让颜色映射能被肉眼推出, 生成器固定每题正好 3 个不同颜色的内块.
#      若两条不同颜色的射线相撞, 混色结果定义为“这一题 3 个块色中, 没参与本次碰撞的那个颜色”.
#      例: 三个块色是 A/B/C, 那么 A+B -> C, A+C -> B, B+C -> A.
#    - 具体小例子: 左侧块发出水平射线 A, 下方块发出竖直射线 B, 右下块发出左上对角线 C.
#      原题会在 A∩B 或 A∩C 处改向; 本变体里三条线都保持直线, 只把交点染成第三色.
# b) 复杂度校准:
#    - 若块色过多, “颜色混合”会变成任意查表, 2-3 个样例不足以归纳.
#      因此限制为 3 色, 并让 3 组两两碰撞都出现, 保证同一张图里就能看到完整映射闭环.
#    - 若没有碰撞, 新旧任务外观接近, 变体退化. 生成器因此强制产生 3 个可见碰撞点.
# c) 隐式假设审查:
#    - 原 solver 第 252-408 行的碰撞分支都依赖旧前提: 碰撞会截断路径、留空、改 pattern.
#      这些前提在新规则下全部失效, 尤其 `blocked`, `truncate_beam`, `find_partner`,
#      `resolve_*collision` 会错误地把直线射线改写成转弯轨迹.
#    - 新 solver 不再维护“当前拥有该格子的 beam”来驱动后续运动, 而是先独立追踪每条直线,
#      再统一计算共享格/几何交叉并覆写混色标记. 这样不会把“碰撞像素”误当成障碍物.
# d) 放置合法性:
#    - 生成器使用一个固定相对布局模板, 3 个 4x4 框之间保留至少 2 格空隙.
#    - 三条射线的交点彼此分离, 且不会撞到别的框; 再通过平移和 D4 对称变换增加外观变化.
#    - 网格边长限制在 26-30, 保证模板、边距和足够长的射线都能同时成立.
# e) 视觉可观察性:
#    - 解题者在每张图里都能同时看到 3 条原色射线和 3 个混色交点.
#    - 混色不是依赖颜色 id 的数值运算, 而是依赖“第三个现成块色”这个可见集合关系.
# f) 可视化整洁度:
#    - 每块只开 1 个缺口, 避免原题那种多射线叠加造成画面杂乱.
#    - 三个碰撞点分散在不同位置, 没有三重碰撞或大面积重叠.
# g) 颜色变化可推断性:
#    - 混色结果始终取自输入里已经出现的第三个块色, 不引入新颜色.
#    - 同一题里三种颜色都会既作为“源色”出现, 也作为“混色结果”出现, 证据充分.


BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

GRID_SIZE_RANGE = (26, 31)
GEN_RETRIES = 200
PLACEMENT_MARGIN = 2

# Three frames, one ray each:
# - left frame emits east
# - lower-middle frame emits north
# - lower-right frame emits northwest
_TEMPLATE = (
    ((0, 0), ((1, 3),)),
    ((8, 12), ((0, 1),)),
    ((9, 18), ((0, 0),)),
)
_TEMPLATE_H = 13
_TEMPLATE_W = 22


class _Beam:
    __slots__ = ("color", "seed", "direction", "path")

    def __init__(
        self,
        color: int,
        seed: tuple[int, int],
        direction: tuple[int, int],
        path: tuple[tuple[int, int], ...],
    ) -> None:
        self.color = int(color)
        self.seed = seed
        self.direction = direction
        self.path = path


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))

    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))

    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _trace_ray(
    g: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    h, w = g.shape
    r, c = seed
    dr, dc = direction
    path = [seed]
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(g[nr, nc]) != bg:
            break
        path.append((nr, nc))
        r, c = nr, nc
    return tuple(path)


def _extract_beams(g: np.ndarray) -> tuple[int, list[int], list[_Beam]]:
    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    block_colors = sorted({int(col) for col, _, _ in blocks})
    assert len(block_colors) == 3, "variant expects exactly three distinct block colors"

    beams: list[_Beam] = []
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected at least one structural frame gap"
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = (dr, dc)
            assert direction != (0, 0), "expected gap to face outward"
            beams.append(_Beam(int(col), (gr, gc), direction, _trace_ray(g, bg, (gr, gc), direction)))
    assert beams, "expected at least one beam"
    return bg, block_colors, beams


def _mix_color(block_colors: list[int], colors: set[int]) -> int:
    if len(colors) == 1:
        return next(iter(colors))
    assert len(colors) == 2, "variant only supports pairwise color mixing"
    remain = [c for c in block_colors if c not in colors]
    assert len(remain) == 1, "three-color palette must leave a unique third color"
    return remain[0]


def _diagonal_cross_area(
    p0: tuple[int, int],
    p1: tuple[int, int],
    q0: tuple[int, int],
    q1: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    pts = {p0, p1, q0, q1}
    if len(pts) != 4:
        return None
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None
    if rows[1] - rows[0] != 1 or cols[1] - cols[0] != 1:
        return None
    return ((rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1]))


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    _, block_colors, beams = _extract_beams(g)

    out = g.copy()
    visitors: dict[tuple[int, int], set[int]] = {}
    diagonal_segments: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []

    for bid, beam in enumerate(beams):
        for cell in beam.path:
            out[cell] = beam.color
            visitors.setdefault(cell, set()).add(beam.color)
        dr, dc = beam.direction
        if abs(dr) + abs(dc) == 2:
            for a, b in zip(beam.path, beam.path[1:]):
                diagonal_segments.append((bid, beam.color, a, b))

    marks: dict[tuple[int, int], int] = {}
    pair_types: set[tuple[int, int]] = set()
    cell_collisions = 0
    for cell, colors in visitors.items():
        if len(colors) < 2:
            continue
        mix = _mix_color(block_colors, colors)
        marks[cell] = mix
        pair_types.add(tuple(sorted(colors)))
        cell_collisions += 1

    diag_cross_collisions = 0
    for i in range(len(diagonal_segments)):
        bid_a, col_a, p0, p1 = diagonal_segments[i]
        for j in range(i + 1, len(diagonal_segments)):
            bid_b, col_b, q0, q1 = diagonal_segments[j]
            if bid_a == bid_b:
                continue
            area = _diagonal_cross_area(p0, p1, q0, q1)
            if area is None:
                continue
            mix = _mix_color(block_colors, {col_a, col_b})
            for cell in area:
                prev = marks.get(cell)
                assert prev is None or prev == mix, "overlapping collision marks must agree"
                marks[cell] = mix
            pair_types.add(tuple(sorted((col_a, col_b))))
            diag_cross_collisions += 1

    for cell, color in marks.items():
        out[cell] = color

    stats = {
        "beam_count": len(beams),
        "cell_collisions": cell_collisions,
        "diag_cross_collisions": diag_cross_collisions,
        "collision_cells": len(marks),
        "collision_pairs": sorted(pair_types),
        "collision_colors": sorted({int(v) for v in marks.values()}),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight rays from framed 2x2 blocks and recolor only their collision sites.

    Input structure:
    - The background is the most common color; a single shared frame color surrounds each 2x2 monochrome block.
    - Every frame has one or more bg-colored border gaps, and each gap launches one outward ray.

    Rule:
    - A gap pixel is painted with the inner block color, and the ray keeps traveling in that same cardinal/diagonal direction until the grid edge or a non-bg input cell.
    - Rays never turn, merge, thicken, or leave holes; the default output is just the union of these straight colored traces.
    - When two different-colored rays meet at a cell, or two diagonal ray segments cross through a 2x2 square, that collision locus is recolored with a third color.
    - The third color is the one block color present elsewhere in the puzzle that does not belong to the colliding pair.

    Key insight:
    - The hard part is no longer trajectory redirection; all ray geometry is fixed by the frame gaps.
    - The only nontrivial inference is the three-color mixing rule: each pairwise collision becomes the remaining block color from the same input.
    """
    out, _ = _solve_with_stats(grid)
    return out


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=3, replace=False)))
    return bg, frame_color, block_colors


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(4))
    out = np.rot90(grid, k)
    if int(rng.integers(2)) == 1:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    bg, frame_color, block_colors = _pick_colors(rng)

    max_r = n - _TEMPLATE_H - PLACEMENT_MARGIN
    max_c = n - _TEMPLATE_W - PLACEMENT_MARGIN
    if max_r < PLACEMENT_MARGIN or max_c < PLACEMENT_MARGIN:
        return None

    off_r = int(rng.integers(PLACEMENT_MARGIN, max_r + 1))
    off_c = int(rng.integers(PLACEMENT_MARGIN, max_c + 1))
    colors = list(block_colors)
    rng.shuffle(colors)

    grid = np.full((n, n), bg, dtype=int)
    for color, ((r0, c0), gaps) in zip(colors, _TEMPLATE):
        _draw_framed_block(grid, off_r + r0, off_c + c0, bg, frame_color, color, gaps)

    grid = _apply_random_symmetry(grid, rng)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        _, stats = _solve_with_stats(grid)
    except Exception:
        return None

    if stats["beam_count"] != 3:
        return None
    if stats["cell_collisions"] != 3:
        return None
    if stats["diag_cross_collisions"] != 0:
        return None
    if len(stats["collision_pairs"]) != 3:
        return None
    if len(stats["collision_colors"]) != 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D2 variant input")

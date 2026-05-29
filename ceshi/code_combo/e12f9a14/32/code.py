# Combo variant of e12f9a14 (index 32)
# combo_spec: D2/1+D17/0
# Name: 碰撞颜色混合+按颜色区分碰撞

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# a) 具体化与行为追踪:
#    - 保留原题的对象骨架: 4x4 框包住 2x2 单色块, 缺口发射射线.
#    - 原题在碰撞后会改方向, 所以输出的关键信号是“拐弯/留白/粗细变化”.
#      本融合版彻底取消碰撞后的轨迹变化: 每条射线始终按初始方向直走到边界.
#    - 真正的变化只发生在碰撞位点: 若两条射线来自同一色系, 交点改涂 `SAME_FAMILY_MIX`;
#      若来自不同色系, 交点改涂 `DIFFERENT_FAMILY_MIX`.
#    - 一个具体小例子: 蓝色水平线和绿色左上对角线相交时, 二者都不断开也不转向, 交点只变成洋红色;
#      红色竖线与蓝色水平线相交时, 同样保持直线, 但交点改成橙色. 差异不再来自角度, 而来自颜色分组.
# b) 复杂度校准:
#    - 若允许太多块色, “色系分类 + 混色映射”会退化成难以从 2-3 个样例归纳的任意查表.
#      因此固定 4 个源色, 并只用两类色系: 冷色系 {蓝, 绿} 与暖色系 {红, 黄}.
#    - 每张图都强制出现 4 条射线的全部 6 组两两碰撞: 其中 2 组是同系, 4 组是异系.
#      这样解题者在单张图里就能同时看到“同系 -> 一种混色, 异系 -> 另一种混色”的完整证据.
# c) 隐式假设审查:
#    - 原 solver 中 `blocked`、`truncate_beam`、`resolve_*collision`、`pattern/phase`
#      都依赖“碰撞会阻断并改写后续运动”这一旧前提; 融合后这些前提全部失效.
#    - 角度分类也失效: 90 度与 45 度不再决定规则, 所以不能再用交角分支驱动结果.
#    - 新 solver 改为先独立追踪每条固定直线, 再统一扫描“共享格子/对角线几何交叉”并覆写碰撞色.
#      这样碰撞标记不会反过来影响别的射线继续前进.
# d) 放置合法性:
#    - 生成器使用 4 个单缺口框的固定相对模板, 对应东/北/西北/东北四种方向.
#    - 该模板保证 6 个碰撞位点彼此分离, 且任意射线都不会撞到别的框.
#    - 再通过随机平移和 D4 对称得到外观变化, 同时保持所有框都离边界至少 2 格.
# e) 视觉可观察性:
#    - 输出上能直接看到 4 条原色直线, 以及两种稳定出现的碰撞色.
#    - 两次“冷冷/暖暖”碰撞都会变成同一种颜色, 四次“冷热”碰撞都会变成另一种颜色,
#      人类可以从重复样式看出判断依据是色系而不是角度.
# f) 可视化整洁度:
#    - 每块只有 1 条射线, 没有原题那种碰撞后再次分叉造成的复杂纹理.
#    - 6 个碰撞位点间隔至少 3 格, 不会出现三重碰撞或大面积叠压.
# g) 变化的可推断性:
#    - 色系划分使用可见的颜色相似性, 不依赖颜色编号的数值大小.
#    - 混色结果只用两种固定标记色, 映射自由度很低: 先分“同系/异系”, 再映射到两种结果色.


BG_COLOR = 0
FRAME_COLOR = 8
BLOCK_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow
COOL_COLORS = frozenset({1, 3})
WARM_COLORS = frozenset({2, 4})
SAME_FAMILY_MIX = 6
DIFFERENT_FAMILY_MIX = 7

GRID_SIZE_RANGE = (26, 31)
GEN_RETRIES = 200
PLACEMENT_MARGIN = 2

# Four one-gap frames whose rays are:
# - east
# - north
# - northwest
# - northeast
_TEMPLATE = (
    ((0, 0), ((1, 3),)),
    ((10, 7), ((0, 1),)),
    ((9, 13), ((0, 0),)),
    ((12, 1), ((0, 3),)),
)
_TEMPLATE_H = 16
_TEMPLATE_W = 17


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
    assert block_colors == sorted(BLOCK_COLORS), "variant expects blue/red/green/yellow source blocks"

    beams: list[_Beam] = []
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        assert frame_col == FRAME_COLOR, "variant expects a shared cyan frame color"
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert len(gaps) == 1, "variant expects exactly one structural gap per frame"
        gr, gc = gaps[0]
        dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
        dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
        direction = (dr, dc)
        assert direction != (0, 0), "expected gap to face outward"
        beams.append(_Beam(int(col), (gr, gc), direction, _trace_ray(g, bg, (gr, gc), direction)))

    assert len(beams) == 4, "variant expects exactly four beams"
    return bg, block_colors, beams


def _same_family(a: int, b: int) -> bool:
    return (a in COOL_COLORS and b in COOL_COLORS) or (a in WARM_COLORS and b in WARM_COLORS)


def _mix_color(a: int, b: int) -> int:
    return SAME_FAMILY_MIX if _same_family(a, b) else DIFFERENT_FAMILY_MIX


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
    same_family_pairs: set[tuple[int, int]] = set()
    different_family_pairs: set[tuple[int, int]] = set()
    cell_collisions = 0
    same_family_cells = 0
    different_family_cells = 0

    for cell, colors in visitors.items():
        if len(colors) < 2:
            continue
        assert len(colors) == 2, "generator must avoid triple cell collisions"
        a, b = sorted(colors)
        mix = _mix_color(a, b)
        marks[cell] = mix
        pair = (a, b)
        if _same_family(a, b):
            same_family_pairs.add(pair)
            same_family_cells += 1
        else:
            different_family_pairs.add(pair)
            different_family_cells += 1
        cell_collisions += 1

    diag_cross_collisions = 0
    same_family_crosses = 0
    different_family_crosses = 0
    for i in range(len(diagonal_segments)):
        bid_a, col_a, p0, p1 = diagonal_segments[i]
        for j in range(i + 1, len(diagonal_segments)):
            bid_b, col_b, q0, q1 = diagonal_segments[j]
            if bid_a == bid_b:
                continue
            area = _diagonal_cross_area(p0, p1, q0, q1)
            if area is None:
                continue
            mix = _mix_color(col_a, col_b)
            for cell in area:
                prev = marks.get(cell)
                assert prev is None or prev == mix, "overlapping collision marks must agree"
                marks[cell] = mix
            pair = tuple(sorted((col_a, col_b)))
            if _same_family(col_a, col_b):
                same_family_pairs.add(pair)
                same_family_crosses += 1
            else:
                different_family_pairs.add(pair)
                different_family_crosses += 1
            diag_cross_collisions += 1

    for cell, color in marks.items():
        out[cell] = color

    stats = {
        "beam_count": len(beams),
        "block_colors": block_colors,
        "cell_collisions": cell_collisions,
        "diag_cross_collisions": diag_cross_collisions,
        "collision_cells": len(marks),
        "same_family_pairs": sorted(same_family_pairs),
        "different_family_pairs": sorted(different_family_pairs),
        "same_family_cells": same_family_cells,
        "different_family_cells": different_family_cells,
        "same_family_crosses": same_family_crosses,
        "different_family_crosses": different_family_crosses,
        "collision_colors": sorted({int(v) for v in marks.values()}),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight rays from framed 2x2 blocks and recolor collisions by color-family relation.

    Input structure:
    - The background is black and the shared frame color is cyan; inside each 4x4 frame sits one 2x2 monochrome block.
    - The four block colors are blue, red, green, and yellow, with exactly one bg-colored structural gap on each frame border.
    - Each gap launches one outward ray, so the puzzle contains four straight rays in total.

    Rule:
    - Every ray paints its gap cell and then continues forever in its original cardinal or diagonal direction until the grid edge.
    - Rays never turn, merge, thicken, or leave blanks; the default output is simply the union of those four straight traces.
    - Whenever two rays share a cell, that collision cell is recolored instead of redirecting either ray.
    - The recolor depends only on the two ray colors: blue+green or red+yellow (same visible color family) become magenta, while any warm-vs-cool pairing becomes orange.

    Key insight:
    - Geometry only tells you where collisions occur; it no longer chooses the outcome.
    - The real inference is two-layered: first partition the source colors into the two visible families {blue, green} and {red, yellow}, then map same-family vs different-family collisions to two distinct marker colors.
    """
    out, _ = _solve_with_stats(grid)
    return out


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(4))
    out = np.rot90(grid, k)
    if int(rng.integers(2)) == 1:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    max_r = n - _TEMPLATE_H - PLACEMENT_MARGIN
    max_c = n - _TEMPLATE_W - PLACEMENT_MARGIN
    if max_r < PLACEMENT_MARGIN or max_c < PLACEMENT_MARGIN:
        return None

    off_r = int(rng.integers(PLACEMENT_MARGIN, max_r + 1))
    off_c = int(rng.integers(PLACEMENT_MARGIN, max_c + 1))

    colors = list(BLOCK_COLORS)
    rng.shuffle(colors)

    grid = np.full((n, n), BG_COLOR, dtype=int)
    for color, ((r0, c0), gaps) in zip(colors, _TEMPLATE):
        _draw_framed_block(grid, off_r + r0, off_c + c0, BG_COLOR, FRAME_COLOR, int(color), gaps)

    grid = _apply_random_symmetry(grid, rng)

    try:
        _, stats = _solve_with_stats(grid)
    except Exception:
        return None

    if stats["beam_count"] != 4:
        return None
    if stats["cell_collisions"] != 6:
        return None
    if stats["diag_cross_collisions"] != 0:
        return None
    if len(stats["same_family_pairs"]) != 2:
        return None
    if len(stats["different_family_pairs"]) != 4:
        return None
    if stats["same_family_cells"] != 2:
        return None
    if stats["different_family_cells"] != 4:
        return None
    if stats["collision_colors"] != [SAME_FAMILY_MIX, DIFFERENT_FAMILY_MIX]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")

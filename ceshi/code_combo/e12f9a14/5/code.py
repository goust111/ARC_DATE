# Combo variant of e12f9a14 (index 5)
# combo_spec: D2/1+D9/2
# Name: 碰撞颜色混合+交叉点特殊标记

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# a) 具体化与行为追踪:
#    - 保留原题的对象字典: 每个对象仍是 4x4 边框包住 2x2 纯色块, 缺口向外发射直线射线.
#    - 原题在碰撞后会改向、留空或变厚, 所以“碰撞”会改变后续轨迹. 融合后这部分全部取消:
#      射线始终沿初始方向直走到边界或非背景输入像素, 只在碰撞局部改颜色.
#    - 这里把“碰撞局部”拆成两层编码:
#      1) 混色层: 碰撞附近的相邻射线格改成第三种块色, 用来表达是哪两种颜色相撞.
#      2) 标记层: 真正的交点格额外改成固定特殊色 5, 用来显式指出“这是碰撞中心”.
#    - 具体小例子: 三个块色为 A/B/C, 左边一条水平 A, 下方一条竖直 B, 右下有一条左上对角线 C.
#      原题会在 A∩B、A∩C、B∩C 处改轨迹. 本变体里三条线都不转弯:
#      A∩B 的中心格直接变成 5, 交点四周紧邻该中心且属于 A/B 两条线的格子改成 C.
#      因而同一处碰撞同时告诉解题者“这里发生了碰撞”和“碰撞双方是 A 与 B”.
# b) 复杂度校准:
#    - 若块色过多, “两色 -> 第三色”会变成不可归纳的查表; 因此固定每题恰好 3 个不同块色.
#    - 若碰撞太少, D9 的特殊标记和 D2 的混色都可能被误读为偶然噪声; 因此生成器强制 3 条射线两两碰撞.
#    - 若同一碰撞区叠加过多标记, 画面会乱; 因此每块只开 1 个缺口, 并用固定模板把三个碰撞点拉开.
# c) 隐式假设审查:
#    - 原 solver 里 `blocked`、`truncate_beam`、`find_partner`、`resolve_*collision` 都依赖旧前提:
#      “碰撞会阻止原轨迹并创建新轨迹”. 融合后这些前提全部失效, 继续沿用会把直线错画成折线.
#    - 新 solver 改成两段式:
#      1) 先独立追踪每条直线射线, 不让碰撞反过来影响几何路径.
#      2) 再从静态路径里提取共享格/对角几何交叉, 覆写局部混色与特殊标记.
#    - 新规则额外引入两个假设:
#      1) 输入里正好有 3 个不同块色, 这样每对颜色才有唯一“第三色”.
#      2) 碰撞是成对且局部足够分离, 否则不同碰撞的混色环带会互相争夺像素.
#      生成器用固定三射线模板保证这两点.
# d) 放置合法性:
#    - 生成器仍使用 4x4 框 + 2x2 内块, 并把三个框放在一个经过验证的相对模板上.
#    - 模板内框之间天然满足原 spec 的非重叠与可视间距; 整体再随机平移并做 D4 对称变换.
#    - 三条射线都只穿过背景, 不撞到别的框, 且碰撞局部不会互相重叠, 所以 `main()` 的局部覆写始终合法.
# e) 视觉可观察性:
#    - 输出里能直接看到三种源色直线、三种“剩余第三色”的混色局部, 以及统一的 5 号色碰撞中心.
#    - 这让解题者不需要依赖颜色编号做运算: 观察同一题内三组两两碰撞, 就能推出 “两色碰撞 -> 剩下那色”.
#    - 特殊色 5 在输入中完全不存在, 只在输出碰撞中心出现, 其语义非常醒目.
# f) 可视化整洁度:
#    - 交点中心只占 1 格, 混色只覆盖该中心周围紧邻的射线格, 不会把整条线都染乱.
#    - 三个碰撞点在模板中呈 L 形分布, 混色局部之间只有留白或原色线段, 不会糊成一团.
# g) 变化的可推断性:
#    - 混色结果始终取自输入里已经可见的第三个块色, 不引入额外的可变输出色.
#    - 特殊标记色固定为 5, 不参与混色映射, 所以“位置标记”与“颜色关系”这两种信息被明确拆开.
#    - 生成器强制每张图里三对颜色都出现一次, 同一张样例就能形成完整的 A+B->C, A+C->B, B+C->A 闭环证据.


BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
SPECIAL_COLOR = 5

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
    __slots__ = ("bid", "color", "seed", "direction", "path", "path_index")

    def __init__(
        self,
        bid: int,
        color: int,
        seed: tuple[int, int],
        direction: tuple[int, int],
        path: tuple[tuple[int, int], ...],
    ) -> None:
        self.bid = bid
        self.color = int(color)
        self.seed = seed
        self.direction = direction
        self.path = path
        self.path_index = {cell: i for i, cell in enumerate(path)}


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
    assert len(block_colors) == 3, "combo variant expects exactly three distinct block colors"
    assert SPECIAL_COLOR not in block_colors, "special marker color must stay distinct from block colors"

    beams: list[_Beam] = []
    next_bid = 0
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected at least one structural frame gap"
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = (dr, dc)
            assert direction != (0, 0), "expected gap to face outward"
            path = _trace_ray(g, bg, (gr, gc), direction)
            beams.append(_Beam(next_bid, int(col), (gr, gc), direction, path))
            next_bid += 1
    assert beams, "expected at least one beam"
    return bg, block_colors, beams


def _mix_color(block_colors: list[int], colors: set[int]) -> int:
    assert len(colors) == 2, "combo variant only supports pairwise color mixing"
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


def _neighbor_mix_cells(beam: _Beam, cell: tuple[int, int]) -> set[tuple[int, int]]:
    idx = beam.path_index[cell]
    cells: set[tuple[int, int]] = set()
    if idx > 0:
        cells.add(beam.path[idx - 1])
    if idx + 1 < len(beam.path):
        cells.add(beam.path[idx + 1])
    return cells


def _register_mix_cell(
    marks: dict[tuple[int, int], int],
    specials: dict[tuple[int, int], int],
    cell: tuple[int, int],
    color: int,
) -> None:
    assert cell not in specials, "collision halo must not overwrite a special marker cell"
    prev = marks.get(cell)
    if prev is None:
        marks[cell] = color
        return
    assert prev == color, "overlapping collision halos must agree on their mixed color"


def _register_special_cell(
    specials: dict[tuple[int, int], int],
    marks: dict[tuple[int, int], int],
    cell: tuple[int, int],
) -> None:
    assert cell not in marks, "special marker must stay visually distinct from the mixed-color halo"
    prev = specials.get(cell)
    if prev is None:
        specials[cell] = SPECIAL_COLOR
        return
    assert prev == SPECIAL_COLOR, "special markers must agree on their color"


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    _, block_colors, beams = _extract_beams(g)

    out = g.copy()
    visitors: dict[tuple[int, int], set[int]] = {}
    cell_beams: dict[tuple[int, int], set[int]] = {}
    diagonal_segments: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    beam_by_id = {beam.bid: beam for beam in beams}

    for beam in beams:
        for cell in beam.path:
            out[cell] = beam.color
            visitors.setdefault(cell, set()).add(beam.color)
            cell_beams.setdefault(cell, set()).add(beam.bid)
        dr, dc = beam.direction
        if abs(dr) + abs(dc) == 2:
            for a, b in zip(beam.path, beam.path[1:]):
                diagonal_segments.append((beam.bid, beam.color, a, b))

    mix_marks: dict[tuple[int, int], int] = {}
    special_marks: dict[tuple[int, int], int] = {}
    pair_types: set[tuple[int, int]] = set()
    cell_collisions = 0

    for cell, colors in visitors.items():
        if len(colors) < 2:
            continue
        assert len(colors) == 2, "generator must avoid triple cell collisions"
        beam_ids = sorted(cell_beams[cell])
        assert len(beam_ids) == 2, "expected one beam per color at a cell collision"
        mix = _mix_color(block_colors, colors)
        for bid in beam_ids:
            for neighbor in _neighbor_mix_cells(beam_by_id[bid], cell):
                _register_mix_cell(mix_marks, special_marks, neighbor, mix)
        _register_special_cell(special_marks, mix_marks, cell)
        pair_types.add(tuple(sorted(colors)))
        cell_collisions += 1

    diag_cross_collisions = 0
    for i in range(len(diagonal_segments)):
        bid_a, col_a, p0, p1 = diagonal_segments[i]
        for j in range(i + 1, len(diagonal_segments)):
            bid_b, col_b, q0, q1 = diagonal_segments[j]
            if bid_a == bid_b or col_a == col_b:
                continue
            area = _diagonal_cross_area(p0, p1, q0, q1)
            if area is None:
                continue
            mix = _mix_color(block_colors, {col_a, col_b})
            special = min(area)
            for cell in area:
                if cell == special:
                    continue
                _register_mix_cell(mix_marks, special_marks, cell, mix)
            _register_special_cell(special_marks, mix_marks, special)
            pair_types.add(tuple(sorted((col_a, col_b))))
            diag_cross_collisions += 1

    for cell, color in mix_marks.items():
        out[cell] = color
    for cell, color in special_marks.items():
        out[cell] = color

    stats = {
        "beam_count": len(beams),
        "cell_collisions": cell_collisions,
        "diag_cross_collisions": diag_cross_collisions,
        "collision_pairs": sorted(pair_types),
        "collision_colors": sorted({int(v) for v in mix_marks.values()}),
        "special_cells": len(special_marks),
        "mixed_cells": len(mix_marks),
        "special_color": SPECIAL_COLOR,
        "fusion_score": 10 * len(special_marks) + len(mix_marks),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace straight rays from framed 2x2 blocks and encode each collision with two visual layers.

    Input structure:
    - The background is the most common color; one shared frame color surrounds each 2x2 monochrome block.
    - Every frame has one or more bg-colored border gaps, and each structural gap launches one straight outward ray.

    Rule:
    - A gap pixel is painted with the inner block color, and that ray keeps moving in the same cardinal or diagonal direction until the grid edge or a non-bg input cell.
    - Rays never bend, merge, or erase each other; first paint the full set of straight traces implied by the gaps.
    - When two differently colored rays collide at one grid cell, recolor the adjacent ray cells around that center with the pair's mixed color, and recolor the center itself with the fixed special marker color 5.
    - The mixed color is the remaining third block color from the same puzzle. If two diagonal segments cross through a 2x2 square instead of sharing one cell, use that 2x2 square as the collision area: the northwest cell is the special marker and the other three cells take the mixed color.

    Key insight:
    - The geometry is completely determined by the frame gaps because collisions no longer redirect trajectories.
    - Each collision now carries dual information: a universal visible marker for the exact crossing point and a nearby third-color halo that reveals which two source colors interacted.
    """
    out, _ = _solve_with_stats(grid)
    return out


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, SPECIAL_COLOR)]
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
    if stats["special_cells"] != 3:
        return None
    if stats["mixed_cells"] != 12:
        return None
    if stats["fusion_score"] != 42:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")

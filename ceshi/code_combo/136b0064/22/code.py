# Combo variant of 136b0064 (index 22)
# combo_spec: D4/3+D10/1
# Name: 边界反弹螺旋路径

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 输入布局保持原题：左侧 7 列是两列 3x3 glyph，列 7 是整列分隔线，右侧只有一个灰色起点。
   - 原题对每个 glyph 的执行是：先向下走 1 格并着该 glyph 的颜色，再沿 area 指定的固定方向画直线。
   - 融合后仍按列主序读取 glyph，也仍保留 area->(颜色, 名义长度, 初始朝向) 的绑定；变化在于“直线段”改成“局部螺旋步进”：
     先执行那一步下移，再从 area 给出的初始朝向出发逐格前进；每走完 1 格，就把朝向顺时针旋转 90 度。
   - 小例子：若当前位置在 `(r, c) = (3, 5)`，读到 area=6（颜色 3，名义长度 4，初始朝向向左）。
     原题会画成一条左向长度 4 的水平段；D10 单变体只会在撞左右墙时反向。
     本融合版会先下移到 `(4, 5)`，随后依次尝试走“左、上、右”三步；若其中某步越界，则先在该边界反弹，再走这一步。
     因而同一个 glyph 现在不再对应直线，而是对应一个顺时针弯折的小螺旋弧，且弧线还能在边界处镜面折返。

b) 复杂度校准
   - 不引入新符号系统：仍只有 4 种固定 glyph、固定颜色绑定、固定列主序读取。
   - area 仍只落在少数离散档位 `{4,5,6,7}`，长度仍沿用原题的 `{2,3,4,2}`，避免每个 glyph 都变成长轨迹导致画面失控。
   - 为了让“螺旋”肉眼可见，真正产生顺时针转弯的 glyph 主要是 area 5 和 6；生成器会强制它们重复出现，而不是退化成几乎全是 area 4/7 的直线。

c) 隐式假设审查
   - 原 solver 中“每个 glyph 的整段方向恒定”的假设失效，因此不能再用 `end_c` 一次性判断水平段是否出界。
   - 原 solver 中“行坐标单调不减”的隐式性质也失效：area 6 的局部螺旋会出现向上的一步，所以后续路径可能回到更高行。
   - D10/1 叠加后，边界检查粒度必须从“整段是否合法”改为“每一步是否合法”；一步若越界，就在该轴上取反后继续。
   - 由于可能回到上方，路径理论上可能踩回灰色起点。solver 新增了“起点格保留灰色”的处理；generator 同时避免采样到这类样本，减少视觉歧义。

d) 放置合法性
   - 输入对象的放置方式完全沿用原 spec：两列 tile 槽位在 `c0 in {0,4}`，行起点为 `0,4,8,...`，因此 glyph 之间不会重叠。
   - 分隔线始终是满高常量列，灰点仍唯一且位于右半边；solver 对这两个锚点的识别逻辑不需要改变。
   - 输出路径通过逐步反弹 helper 保证始终留在 `rows 0..H-1, cols 0..6`，不会因为局部上移或撞墙而崩溃。

e) 视觉可观察性
   - 仅有“顺时针转弯”还不够，因为 area 4 和 area 7 仍是长度 2，视觉上接近原题。
   - 因此生成器强制每个样本至少包含一个 area 5 和一个 area 6，并要求融合输出与“只有反弹、没有螺旋”的直线版输出存在足够多的像素差异。
   - 同时还要求至少出现一次曲线段上的边界反弹，这样训练样例里能同时看到“转弯”和“撞墙折返”两种核心现象。

f) 可视化整洁度
   - 轨迹仍是一像素宽的单条路径，不新增装饰性标记。
   - 生成器限制不同颜色的重涂次数，避免太多后写覆盖把局部螺旋抹成难以辨认的色块。
   - 同色自重合允许少量存在，因为它更像路径回环，不会像异色覆盖那样破坏可读性。

g) 变化的可推断性
   - 颜色绑定完全保留原题的结构线索：color 仍由 glyph 形状决定，而不是重新采样。
   - 解题者需要新增推断的只是一条结构规则：同样的 area 不再画直线，而是按固定顺时针节奏走；若某一步将越界，则先反弹再继续这个节奏。
   - 这种新增规律来自几何轨迹本身，不依赖颜色 id 的数值含义。

h) main() 的新隐式假设与 generator 对应保证
   - main() 假设：每个 glyph 的第一步仍是“名义向下的一步”，随后才进入 area 对应的局部螺旋。
   - main() 假设：反弹是逐步发生的，且反弹后顺时针旋转应以“实际移动后的朝向”为基准。
   - generator 复用同一个逐步仿真 helper 来采样和验算，因此这些新假设都被构造性满足。
   - generator 还额外保证：存在曲线 glyph、存在曲线反弹、与直线反弹版显著不同、且灰色起点不会被后续路径踩回。
"""


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_INIT_DIR = {4: 2, 5: 1, 6: 3, 7: 3}  # up=0, right=1, down=2, left=3
_AREAS = (4, 5, 6, 7)
_CURVED_AREAS = {5, 6}
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIR_TO_IDX = {d: i for i, d in enumerate(_DIRS)}

# --- generation constraints ---
MAX_ATTEMPTS = 4000
MAX_COLOR_REPAINTS = 2
MAX_SAME_COLOR_OVERLAPS = 4
MIN_DIFF_VS_STRAIGHT = 4
MIN_TOTAL_BOUNCES = 1
MIN_CURVED_BOUNCES = 1
MIN_ROTATIONS = 3


def _collect_glyphs(left: np.ndarray, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        bbox = comp["bbox"]
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"color": color, "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def _step_with_reflection(r: int, c: int, dr: int, dc: int, h: int, w: int) -> tuple[int, int, bool, int, int]:
    nr = int(r + dr)
    nc = int(c + dc)
    bounced = False

    if not (0 <= nr < int(h)):
        dr = int(-dr)
        nr = int(r + dr)
        bounced = True
    if not (0 <= nc < int(w)):
        dc = int(-dc)
        nc = int(c + dc)
        bounced = True

    assert 0 <= nr < int(h) and 0 <= nc < int(w), f"reflected step exits canvas: {(nr, nc)} in {(h, w)}"
    return int(nr), int(nc), bool(bounced), int(dr), int(dc)


def _paint_cell(
    out: np.ndarray | None,
    r: int,
    c: int,
    color: int,
    *,
    protected: set[tuple[int, int]] | None = None,
) -> dict[str, int]:
    if out is None:
        return {"same_color_overlap": 0, "color_repaint": 0, "gray_hit": 0}

    if protected is not None and (int(r), int(c)) in protected:
        return {"same_color_overlap": 0, "color_repaint": 0, "gray_hit": 1}

    old = int(out[int(r), int(c)])
    info = {"same_color_overlap": 0, "color_repaint": 0, "gray_hit": 0}
    if old == int(color):
        info["same_color_overlap"] = 1
    elif old != 0:
        info["color_repaint"] = 1

    out[int(r), int(c)] = int(color)
    return info


def _advance_spiral_segment(
    r_cur: int,
    c_cur: int,
    area: int,
    h: int,
    w: int,
    *,
    color: int | None = None,
    out: np.ndarray | None = None,
    protected: set[tuple[int, int]] | None = None,
) -> tuple[int, int, dict[str, int]]:
    seg_len = int(_AREA_TO_LEN[int(area)])
    dir_idx = int(_AREA_TO_INIT_DIR[int(area)])
    info = {
        "bounces": 0,
        "curved_bounces": 0,
        "rotations": 0,
        "same_color_overlap": 0,
        "color_repaint": 0,
        "gray_hit": 0,
        "boundary_cells": 0,
    }

    r_cur, c_cur, bounced, _dr_eff, _dc_eff = _step_with_reflection(int(r_cur), int(c_cur), 1, 0, int(h), int(w))
    info["bounces"] += int(bounced)
    info["boundary_cells"] += int(r_cur in {0, int(h) - 1} or c_cur in {0, int(w) - 1})
    if out is not None and color is not None:
        paint = _paint_cell(out, r_cur, c_cur, int(color), protected=protected)
        for k, v in paint.items():
            info[k] += int(v)

    for step_idx in range(seg_len - 1):
        dr, dc = _DIRS[int(dir_idx)]
        r_cur, c_cur, bounced, dr_eff, dc_eff = _step_with_reflection(int(r_cur), int(c_cur), int(dr), int(dc), int(h), int(w))
        info["bounces"] += int(bounced)
        if int(area) in _CURVED_AREAS:
            info["curved_bounces"] += int(bounced)
        info["boundary_cells"] += int(r_cur in {0, int(h) - 1} or c_cur in {0, int(w) - 1})

        if out is not None and color is not None:
            paint = _paint_cell(out, r_cur, c_cur, int(color), protected=protected)
            for k, v in paint.items():
                info[k] += int(v)

        dir_idx = int((_DIR_TO_IDX[(int(dr_eff), int(dc_eff))] + 1) % 4)
        if step_idx < seg_len - 2:
            info["rotations"] += 1

    return int(r_cur), int(c_cur), info


def _advance_straight_segment(
    r_cur: int,
    c_cur: int,
    area: int,
    h: int,
    w: int,
    *,
    color: int | None = None,
    out: np.ndarray | None = None,
    protected: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    seg_len = int(_AREA_TO_LEN[int(area)])
    dir_idx = int(_AREA_TO_INIT_DIR[int(area)])

    r_cur, c_cur, _bounced, _dr_eff, _dc_eff = _step_with_reflection(int(r_cur), int(c_cur), 1, 0, int(h), int(w))
    if out is not None and color is not None:
        _paint_cell(out, r_cur, c_cur, int(color), protected=protected)

    for _ in range(seg_len - 1):
        dr, dc = _DIRS[int(dir_idx)]
        r_cur, c_cur, _bounced, dr_eff, dc_eff = _step_with_reflection(int(r_cur), int(c_cur), int(dr), int(dc), int(h), int(w))
        dir_idx = int(_DIR_TO_IDX[(int(dr_eff), int(dc_eff))])
        if out is not None and color is not None:
            _paint_cell(out, r_cur, c_cur, int(color), protected=protected)

    return int(r_cur), int(c_cur)


def _render_trace(areas: list[int], start_col: int, h: int) -> tuple[np.ndarray, dict[str, int]]:
    out = np.zeros((int(h), LEFT_W), dtype=int)
    protected = {(GRAY_START_ROW, int(start_col))}
    out[GRAY_START_ROW, int(start_col)] = GRAY

    metrics = {
        "bounces": 0,
        "curved_bounces": 0,
        "rotations": 0,
        "same_color_overlap": 0,
        "color_repaint": 0,
        "gray_hit": 0,
        "boundary_cells": 0,
        "area5_count": 0,
        "area6_count": 0,
        "curved_glyphs": 0,
    }

    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    for area in areas:
        if int(area) == 5:
            metrics["area5_count"] += 1
        if int(area) == 6:
            metrics["area6_count"] += 1
        if int(area) in _CURVED_AREAS:
            metrics["curved_glyphs"] += 1

        color = int(_AREA_TO_COLOR[int(area)])
        r_cur, c_cur, info = _advance_spiral_segment(
            r_cur,
            c_cur,
            int(area),
            int(h),
            LEFT_W,
            color=color,
            out=out,
            protected=protected,
        )
        for k, v in info.items():
            metrics[k] += int(v)

    metrics["end_r"] = int(r_cur)
    metrics["end_c"] = int(c_cur)
    return out, metrics


def _render_straight_trace(areas: list[int], start_col: int, h: int) -> np.ndarray:
    out = np.zeros((int(h), LEFT_W), dtype=int)
    protected = {(GRAY_START_ROW, int(start_col))}
    out[GRAY_START_ROW, int(start_col)] = GRAY

    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    for area in areas:
        color = int(_AREA_TO_COLOR[int(area)])
        r_cur, c_cur = _advance_straight_segment(
            r_cur,
            c_cur,
            int(area),
            int(h),
            LEFT_W,
            color=color,
            out=out,
            protected=protected,
        )
    return out


def _evaluate_sequence(areas: list[int], start_col: int, h: int) -> dict[str, int]:
    fusion_out, metrics = _render_trace(areas, int(start_col), int(h))
    straight_out = _render_straight_trace(areas, int(start_col), int(h))
    metrics["diff_vs_straight"] = int(np.count_nonzero(fusion_out != straight_out))
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 glyph program and draw a reflected clockwise spiral trace on the left canvas.

    Input structure:
    - Left of the full-height divider is the same 7-column glyph board made of 3x3 monochrome symbols.
    - Right of the divider there is exactly one gray start cell; its shifted column is the drawing start on the left canvas.

    Rule:
    - Remove the divider and move the gray start back into the 7-column drawing canvas.
    - Read glyphs in column-major order: top-to-bottom in the left tile column, then top-to-bottom in the right tile column.
    - Each glyph area still determines color, nominal length, and initial heading: area 4 -> color 6, len 2, start down; area 5 -> color 1, len 3, start right; area 6 -> color 3, len 4, start left; area 7 -> color 2, len 2, start left.
    - For each glyph, first take one downward step, then draw the remaining cells one step at a time; after every step, rotate the heading 90 degrees clockwise, and if a step would leave the canvas, reflect at that boundary before taking it.

    Key insight:
    - The glyph still encodes a short movement program, but that program is no longer a straight segment: it is a fixed clockwise micro-spiral whose visible trace also depends on wall reflections.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((int(h), int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out

    protected = {(int(r_cur), int(c_cur))}
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg, div_color)

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        r_cur, c_cur, _info = _advance_spiral_segment(
            r_cur,
            c_cur,
            area,
            int(h),
            int(div_c),
            color=color,
            out=out,
            protected=protected,
        )

    return out


def _sample_areas(num_glyphs: int) -> list[int]:
    weights = [1, 4, 5, 1]
    return [random.choices(_AREAS, weights=weights, k=1)[0] for _ in range(int(num_glyphs))]


def _sequence_ok(areas: list[int], start_col: int, h: int) -> bool:
    metrics = _evaluate_sequence(areas, int(start_col), int(h))
    if metrics["area5_count"] < 1 or metrics["area6_count"] < 1:
        return False
    if metrics["curved_glyphs"] < 2:
        return False
    if metrics["rotations"] < MIN_ROTATIONS:
        return False
    if metrics["bounces"] < MIN_TOTAL_BOUNCES:
        return False
    if metrics["curved_bounces"] < MIN_CURVED_BOUNCES:
        return False
    if metrics["diff_vs_straight"] < MIN_DIFF_VS_STRAIGHT:
        return False
    if metrics["gray_hit"] > 0:
        return False
    if metrics["color_repaint"] > MAX_COLOR_REPAINTS:
        return False
    if metrics["same_color_overlap"] > MAX_SAME_COLOR_OVERLAPS:
        return False
    if metrics["boundary_cells"] < 2:
        return False
    return True


def _render_grid(n: int, start_col: int, areas: list[int]) -> np.ndarray:
    h = 4 * int(n) - 1
    grid = np.zeros((int(h), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // int(n)]
        r0 = TILE_ROW_STRIDE * (i % int(n))
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    h = 4 * int(n) - 1
    start_col = random.choices(range(LEFT_W), weights=[6, 4, 1, 1, 1, 4, 6], k=1)[0]

    for _ in range(120):
        areas = _sample_areas(2 * int(n))
        if not _sequence_ok(areas, int(start_col), int(h)):
            continue
        return _render_grid(int(n), int(start_col), areas)
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")

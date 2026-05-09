# Variant of 136b0064
# Dimension: D10 — 折线的边界约束 (meta)
# Change: 折线必须始终保持在左画布范围内（列 0..6，行 0..H-1），超出则生成失败... → 折线碰到边界时反弹

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

"""
Design analysis for the D10 variant
===================================

a) 具体化与行为追踪
   - 输入结构不变：左侧 7 列仍是两列 3x3 glyph，列 7 是满高分隔线，右侧只有一个灰色起点。
   - 输出仍是在 7 列左画布上画折线；变化只发生在水平段接近左右边界时。
   - 小例子：若当前列 c=5，读到 area=5（蓝色、名义上向右长 3）。
     原题：先下移一格并上色，再要求终点 c=7，越界后直接失败。
     变体：同样先下移一格；水平逐步前进时先到 c=6，再因下一步会出界而反向，回到 c=5。
     因此同一个 glyph 现在可能在边界处“折返”，输出会出现贴边后回折的痕迹。

b) 复杂度校准
   - 保持 4 种 glyph、固定颜色绑定、固定列主序读取，不引入新的符号系统。
   - 只对左右边界做反弹；竖直 area=4 仍然只向下画，并由生成器保证底部合法。
     这样 D10 仍然是“边界约束变化”，而不是额外引入上下反射的新运动语义。

c) 原 solver 中依赖旧属性的代码审查
   - `end_c = c_cur + dir_h * (seg_len - 1)` + 越界 assert：依赖“整段方向恒定且不能碰墙”，在变体中失效。
   - 水平循环里直接 `c_cur += dir_h`：依赖“每一步都不会要求反向”，在变体中失效。
   - generator 的 `_AREA_DELTA`：原来把水平 glyph 的总位移视为常量；反弹后总 dc 取决于当前列，不能再写死。
   - 竖直 `r_cur < H` / `end_r < H` 的假设仍成立，因为生成器继续保证没有底部越界。

d) 放置合法性
   - glyph 仍放在固定 tile 槽位 `(c0 in {0,4}, r0 = 0,4,8,...)`，彼此不会重叠。
   - 分隔线、灰点位置不变，solver 仍可稳定识别。
   - 输出路径的行坐标仍单调不减；反弹只在同一行内折返，不会与更早的行混淆。

e) 视觉可观察性
   - 仅“碰墙”还不够，因为某些碰墙会得到与原题相同的可见长度，例如 area=7 在最左列反弹后仍只看到 2 格。
   - 生成器因此强制至少出现一次“可见反弹”：即碰墙后该颜色在该行的可见跨度短于其名义长度。
     例如 area=5 从 c=5 向右时只会看到 2 个蓝格；area=6 从 c=1/2 向左时只会看到 3 个绿格。
   - 这样解题者能从少量样例肉眼看出：glyph 所代表的“名义长度”在边界处被反射折返了。

f) 可视化整洁度
   - 不新增标记，不改变布局；输出仍是一条稀疏折线。
   - 反弹只会在当前行内部回折，最多覆盖同色自身轨迹，不会制造多层对象叠加噪声。

g) 颜色变化可推断性
   - 颜色与 glyph 形状的绑定完全保留，不引入新的颜色映射，也不使用颜色 id 做隐藏排序。
   - 颜色仍只承担“这是哪一种指令”的角色；变体的新增信息来自位置与碰墙后的轨迹形态。

h) main() 的新隐式假设与 generator 对应保证
   - main() 现在假设：水平段若下一步将出界，则立即反向后再走这一步。
   - generator 使用同一个逐步仿真 helper，保证所有采样样本都满足该假设。
   - main() 仍假设存在唯一灰点、唯一满高分隔列、以及 2*N 个 3x3 glyph；generator 继续显式构造这些条件。
"""


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
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
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus    -> down 2,  color 6
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond -> right 3, color 1
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross   -> left 4,  color 3
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U       -> left 2,  color 2
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MIN_TOTAL_BOUNCES = 1
MIN_VISIBLE_BOUNCES = 1
MAX_ATTEMPTS = 1000


def _advance_segment(
    r_cur: int,
    c_cur: int,
    area: int,
    H: int,
    left_w: int,
    *,
    color: int | None = None,
    out: np.ndarray | None = None,
) -> tuple[int, int, dict[str, int | bool]]:
    """Advance one decoded instruction, reflecting horizontal steps off side walls."""
    seg_len = int(_AREA_TO_LEN.get(int(area), 2))

    r_cur = int(r_cur + 1)
    assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
    if out is not None and color is not None:
        out[int(r_cur), int(c_cur)] = int(color)

    if int(area) == 4:
        for _ in range(seg_len - 1):
            r_cur = int(r_cur + 1)
            assert r_cur < H, f"vertical segment exits bottom: r_cur={r_cur}, H={H}"
            if out is not None and color is not None:
                out[int(r_cur), int(c_cur)] = int(color)
        return int(r_cur), int(c_cur), {"bounces": 0, "visible_bounce": False, "distinct_cols": 1}

    dir_h = int(_AREA_TO_DIR.get(int(area), 1))
    painted_cols = [int(c_cur)]
    bounces = 0

    for _ in range(seg_len - 1):
        next_c = int(c_cur + dir_h)
        if not (0 <= next_c < int(left_w)):
            dir_h = int(-dir_h)
            bounces += 1
            next_c = int(c_cur + dir_h)
        assert 0 <= next_c < int(left_w), f"reflected step still exits canvas: next_c={next_c}, W={left_w}"
        c_cur = int(next_c)
        painted_cols.append(int(c_cur))
        if out is not None and color is not None:
            out[int(r_cur), int(c_cur)] = int(color)

    distinct_cols = len(set(painted_cols))
    return int(r_cur), int(c_cur), {
        "bounces": int(bounces),
        "visible_bounce": bool(bounces > 0 and distinct_cols < seg_len),
        "distinct_cols": int(distinct_cols),
    }


def _collect_glyphs(left: np.ndarray, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": int(col), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 glyph program and draw the colored polyline on the left canvas, now with wall reflection.

    Input structure:
    - Left of a full-height divider sits a 7-column glyph board containing 3x3 monochrome symbols.
    - Right of the divider there is exactly one gray start cell that indicates the initial column.

    Rule:
    - Remove the divider and shift the gray start back onto the 7-column drawing board.
    - Read glyphs in column-major order (left column top→bottom, then right column top→bottom).
    - Each glyph area still encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2; every glyph first steps down one row.
    - When a horizontal step would leave columns 0..6, reverse direction at the wall and continue drawing the remaining cells of that segment.

    Key insight:
    - The glyph area still fixes color, nominal length, and nominal direction, but the visible path also depends on whether a horizontal run hits a side boundary and reflects.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg, div_color)

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        r_cur, c_cur, _info = _advance_segment(r_cur, c_cur, area, H, int(div_c), color=color, out=out)

    return out


def _area_fits_vertically(area: int, r_cur: int, H: int) -> bool:
    delta_r = 2 if int(area) == 4 else 1
    return int(r_cur) + delta_r < int(H)


def _simulate_areas(areas: list[int], start_col: int, H: int) -> dict[str, int]:
    r, c = GRAY_START_ROW, int(start_col)
    bounces = 0
    visible_bounces = 0
    changes = 0
    prev_vertical: bool | None = None

    for area in areas:
        is_vertical = int(area) == 4
        if prev_vertical is not None and is_vertical != prev_vertical:
            changes += 1
        prev_vertical = is_vertical

        r, c, info = _advance_segment(r, c, int(area), H, LEFT_W)
        bounces += int(info["bounces"])
        visible_bounces += int(bool(info["visible_bounce"]))

    return {
        "end_r": int(r),
        "end_c": int(c),
        "bounces": int(bounces),
        "visible_bounces": int(visible_bounces),
        "direction_changes": int(changes),
    }


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Random-walk a sequence of 2*N instructions using reflective horizontal motion."""
    H = 4 * N - 1
    r, c = GRAY_START_ROW, int(start_col)
    areas: list[int] = []
    total_bounces = 0
    visible_bounces = 0
    prev_vertical: bool | None = None

    for step_idx in range(2 * N):
        opts: list[tuple[int, int, int, dict[str, int | bool], float]] = []
        for area in _AREAS:
            if not _area_fits_vertically(area, r, H):
                continue
            nr, nc, info = _advance_segment(r, c, area, H, LEFT_W)
            weight = 1.0

            if visible_bounces < MIN_VISIBLE_BOUNCES and bool(info["visible_bounce"]):
                weight += 10.0
            elif total_bounces < MIN_TOTAL_BOUNCES and int(info["bounces"]) > 0:
                weight += 4.0

            is_vertical = area == 4
            if prev_vertical is not None and is_vertical != prev_vertical:
                weight += 1.0

            steps_left = 2 * N - step_idx - 1
            if steps_left == 0 and visible_bounces < MIN_VISIBLE_BOUNCES and not bool(info["visible_bounce"]):
                continue

            opts.append((int(area), int(nr), int(nc), info, float(weight)))

        if not opts:
            return None

        choice = random.choices(opts, weights=[opt[4] for opt in opts], k=1)[0]
        area, r, c, info, _weight = choice
        areas.append(int(area))
        total_bounces += int(info["bounces"])
        visible_bounces += int(bool(info["visible_bounce"]))
        prev_vertical = area == 4

    metrics = _simulate_areas(areas, start_col, H)
    if metrics["direction_changes"] < MIN_DIRECTION_CHANGES:
        return None
    if metrics["bounces"] < MIN_TOTAL_BOUNCES:
        return None
    if metrics["visible_bounces"] < MIN_VISIBLE_BOUNCES:
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.choices(range(LEFT_W), weights=[4 if c in (1, 5) else 1 for c in range(LEFT_W)], k=1)[0]
    areas = _sample_path(N, int(start_col))
    if areas is None:
        return None
    return _render_grid(N, int(start_col), areas)


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

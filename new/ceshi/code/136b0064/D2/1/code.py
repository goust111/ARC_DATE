# Variant of 136b0064
# Dimension: D2 — 字形编码方式——面积到指令的映射 (transform)
# Change: 字形像素面积唯一决定线段：area 4→垂直向下2格，area 5→水平向右3格，area 6→水平向左4格，area 7→水平向左2格
#         → 面积到方向的映射反转（小面积对应长线段）
#
# Design analysis
# ===============
# variant_spec:
#   preserved_structure:
#     - 输入仍是 Hx15，H=4*N-1，N in 2..5。
#     - 第 7 列仍是整高分隔线，左侧仍是两列 3x3 字形槽位，右侧仍只有一个灰色起点。
#     - 四种字形模板、字形颜色、读取顺序（列主序）全部保持原题设定。
#   core_change:
#     - 只改 area -> instruction 的对应，不改 shape/color 本体。
#     - 采用一个明确且可见的重排：
#         area 4 -> left 4
#         area 5 -> right 3
#         area 6 -> left 2
#         area 7 -> down 2
#     - 这样最小面积 area 4 变成最长段，最大面积 area 7 变成最短的竖段，D2 的变化直接体现在输出折线。
#   behavior_trace:
#     - 例：起点在左画布列 4，前两个字形面积依次为 [4, 7]。
#     - 原题：
#         area 4: 先下移一格并继续竖直，总位移 (+2, 0)，终点到 (2,4)。
#         area 7: 再下移一格后向左 2，总位移 (+1,-1)，终点到 (3,3)。
#     - 本变体：
#         area 4: 先下移一格后向左画满 4 格，总位移 (+1,-3)，终点到 (1,1)。
#         area 7: 再下移一格并继续竖直，总位移 (+2, 0)，终点到 (3,1)。
#     - 差异：同一组字形在输入端完全不变，但输出折线的方向与长度关系被重新分配。
#   complexity_calibration:
#     - 仍只有 4 个离散 instruction 档位，没有引入额外自由度或隐藏状态。
#     - 为了让人类能在 2-4 个样例中看出变化，生成器要求每个样例都至少出现 area 4 与 area 7，
#       并至少覆盖 3 种不同面积，避免样例长期只落在未变或不明显的子情况。
#   implicit_assumption_review:
#     - find_full_height_constant_col: 仍依赖“只有一条整高非背景列”，变体未改此结构。
#     - gray 起点右移回左画布: 仍成立，因为分隔线与右侧布局不变。
#     - glyph 提取逻辑: 仍依赖“每个 instruction 是单色 3x3、8 连通组件”，变体未改字形本体。
#     - 原 solver 中 `if area == 4` 的分支隐含“只有 area 4 是竖段”；该前提已失效，
#       必须改为查表，不能继续按 area==4 特判。
#     - 原 solver 中 `get(area, ...)` 的默认值在原题里恰好无害；变体里如果出现未知面积会静默走错，
#       因此改为显式断言四种合法面积。
#   placement_legality:
#     - 生成器继续使用 path-first：先在 7 列左画布内模拟折线，再反写成字形。
#     - 每一步都检查总位移后仍在列 0..6、行 < H，保证 solver 的边界断言不会触发。
#     - 字形槽位固定在两列、行间距固定 4，因此 3x3 组件之间天然不重叠。
#   observability_and_cleanliness:
#     - 输出仍是单条单调向下的折线；因为每段开始前都先下移一格，路径不会回到更高的行，
#       视觉上不会和之前路径缠成难以辨认的覆盖团。
#     - 生成器额外过滤“新旧规则输出完全相同”的样本，避免退化回原任务。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


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


# New D2 mapping: smallest area now controls the longest segment.
AREA_TO_STEP = {
    4: {"dr": 0, "dc": -1, "length": 4},  # left 4
    5: {"dr": 0, "dc": 1, "length": 3},   # right 3
    6: {"dr": 0, "dc": -1, "length": 2},  # left 2
    7: {"dr": 1, "dc": 0, "length": 2},   # down 2
}

ORIGINAL_AREA_TO_STEP = {
    4: {"dr": 1, "dc": 0, "length": 2},   # down 2
    5: {"dr": 0, "dc": 1, "length": 3},   # right 3
    6: {"dr": 0, "dc": -1, "length": 4},  # left 4
    7: {"dr": 0, "dc": -1, "length": 2},  # left 2
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the tiled 3x3 glyphs into a colored polyline on the left canvas.

    Input structure:
    - A full-height divider splits the board into a left instruction panel and a right start area.
    - The left panel contains monochrome 3x3 glyph components; the right side contains one gray start cell.

    Rule:
    - Drop the divider, shift the gray start into left-canvas coordinates, and read the glyphs in column-major order.
    - Each glyph's instruction is determined only by its pixel area, but in this variant the area-to-segment map is inverted:
      area 4 → left 4, area 5 → right 3, area 6 → left 2, area 7 → down 2.
    - For every glyph, step down one row, paint that cell with the glyph color, then extend the remaining cells of the segment.

    Key insight:
    - The visible shapes are unchanged from the source task; only the lookup from glyph area to motion has been permuted, so
      solving requires relearning which area now means “long left”, “short left”, “right”, or “vertical”.
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

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        assert area in AREA_TO_STEP, f"unexpected glyph area: {area}"
        step = AREA_TO_STEP[area]

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        dr = int(step["dr"])
        dc = int(step["dc"])
        seg_len = int(step["length"])
        end_r = int(r_cur + dr * (seg_len - 1))
        end_c = int(c_cur + dc * (seg_len - 1))
        assert 0 <= end_r < H, f"segment exits bottom: end_r={end_r}, H={H}"
        assert 0 <= end_c < int(div_c), (
            f"segment exits canvas: c_cur={c_cur}, dc={dc}, seg_len={seg_len}, div_c={div_c}"
        )

        for _ in range(seg_len - 1):
            r_cur = int(r_cur + dr)
            c_cur = int(c_cur + dc)
            out[r_cur, c_cur] = color

    return out


# --------------- generator ---------------

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

# --- fixed glyph templates and colors ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MIN_DISTINCT_AREAS = 3
MAX_ATTEMPTS = 1200


def _total_delta(area: int, mapping: dict[int, dict[str, int]]) -> tuple[int, int]:
    step = mapping[area]
    dr = int(step["dr"])
    dc = int(step["dc"])
    seg_len = int(step["length"])
    return int(1 + dr * (seg_len - 1)), int(dc * (seg_len - 1))


def _trace_signature(areas: list[int], start_col: int, mapping: dict[int, dict[str, int]]) -> tuple[tuple[int, int, int], ...]:
    r, c = GRAY_START_ROW, start_col
    trace: list[tuple[int, int, int]] = [(r, c, GRAY)]
    for area in areas:
        step = mapping[area]
        color = _AREA_TO_COLOR[area]
        r += 1
        trace.append((r, c, color))
        for _ in range(int(step["length"]) - 1):
            r += int(step["dr"])
            c += int(step["dc"])
            trace.append((r, c, color))
    return tuple(trace)


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Random-walk a sequence of 2*N area instructions that stays in the left canvas."""
    H = 4 * N - 1
    r, c = GRAY_START_ROW, start_col
    areas: list[int] = []
    for _ in range(2 * N):
        opts = []
        for area in _AREAS:
            dr_total, dc_total = _total_delta(area, AREA_TO_STEP)
            if r + dr_total < H and 0 <= c + dc_total < LEFT_W:
                opts.append(area)
        if not opts:
            return None
        area = random.choice(opts)
        areas.append(area)
        dr_total, dc_total = _total_delta(area, AREA_TO_STEP)
        r += dr_total
        c += dc_total

    types = [AREA_TO_STEP[a]["dr"] != 0 for a in areas]  # True = vertical
    changes = sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if 4 not in areas or 7 not in areas:
        return None
    if len(set(areas)) < MIN_DISTINCT_AREAS:
        return None
    if _trace_signature(areas, start_col, AREA_TO_STEP) == _trace_signature(areas, start_col, ORIGINAL_AREA_TO_STEP):
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int]) -> np.ndarray:
    """Paint divider, gray start, and all 2*N tiled glyphs into the input grid."""
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[area] * _AREA_TO_COLOR[area]
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(N, start_col)
    if areas is None:
        return None
    return _render_grid(N, start_col, areas)


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

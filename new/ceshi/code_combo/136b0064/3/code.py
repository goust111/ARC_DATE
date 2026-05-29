# Combo variant of 136b0064 (index 3)
# combo_spec: D2/2+D3/1
# Name: 颜色独立编码方向与涂色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_behavior_trace:
#    - 输入仍然保持原题的空间骨架：宽 15，中间第 7 列整列分隔线；左侧 7 列是两列 3x3 字形槽位，
#      右侧只有一个灰色起点。字形仍然是原题那四个固定模板，所以视觉上依然很像原题。
#    - 融合后的变化有两层，而且两层同时作用在同一个 glyph 上：
#        1) 面积只决定长度：area 4/5/6/7 -> len 2/3/4/5。
#        2) 颜色只决定方向，同时也决定输出折线的涂色：蓝(1)=下，红(2)=右，绿(3)=左。
#    - 小例子追踪：
#        * 原题里，一个 area 6 的 cross 必然是绿色，而且必然画“向左 4”。
#        * 现在，一个 area 6 的红色 cross 会画“向右 4 红线”；一个 area 6 的蓝色 cross 会画
#          “向下 4 蓝线”。同一个形状可换方向；同一个颜色跨形状时方向不变但长度随面积变化。
#    - 因此，解题者必须把“颜色->方向/涂色”和“面积->长度”拆开理解，不能再把某个固定 glyph
#      整体当成单条原子指令。
# b) complexity_calibration:
#    - 方向仍限制为 {下, 左, 右} 三类，不引入“上”，避免自由度失控。
#    - 长度采用四档唯一值 2/3/4/5；这样四种面积都保留辨识价值，避免 area 7 继续和 area 4
#      共用长度而削弱“面积编码长度”的可见性。
#    - 颜色只用 1/2/3 三种，既足够支撑三方向映射，也便于人类在少量样例中比较复用模式。
# c) implicit_assumptions_review:
#    - 原 solver 中“area 4 等于竖线、其他 area 等于横线”的判断失效；现在竖/横完全由颜色决定。
#    - 原 generator 把 `_AREA_TO_COLOR` 视为 shape 的固定属性；融合后这条绑定必须删除，否则会退化
#      成原题或只剩 D2 单变体表象不足。
#    - 原 solver 过滤颜色时默认 divider/start 之外的 glyph 颜色都是安全的；这里继续只使用 1/2/3，
#      因而该过滤仍成立，但已不再依赖 shape->color 的旧绑定。
#    - main() 新增的关键前提是：所有有效 glyph 颜色都在 `{1,2,3}` 中。generator 必须显式保证。
# d) placement_legality:
#    - 仍采用 path-first / instruction-first 的合法性检查：先采样 (area, color) 指令序列，
#      再模拟路径，确认不会触底、不会越出 7 列左画布、不会覆盖已有折线单元。
#    - 左侧字形仍只落在固定 tile 槽位 `(c0 in {0,4}, r0 = 4*k)`，因此 3x3 bbox 永不重叠。
# e) visual_observability:
#    - 每个样本至少包含：
#        * 三种方向颜色全部出现；
#        * 至少三种面积；
#        * 一个“同面积不同颜色”的对照；
#        * 一个“同颜色不同面积”的对照。
#      这样单个样本内部就能同时看出“颜色控制方向”和“面积控制长度”。
#    - 另外强制至少两条指令与原题的 area->direction 逻辑不一致，避免样本输出偶然与原题同形。
# f) visual_cleanliness:
#    - 输出路径禁止自交和复写；否则颜色叠压会掩盖“颜色=方向”的证据。
#    - 输入仍然只有左字形区、中间分隔线、右灰起点三个区域，视觉结构保持清晰。
# g) inferability_of_the_change:
#    - 本融合版不依赖任何不可视因素排序；顺序仍由 tile 的几何位置决定（列主序）。
#    - 颜色种类少且会重复复用，面积种类少且会重复复用，恰好为人类提供横向比较线索。
# solver_generator_contract_after_change:
#    - main() 假设左侧只有合法的 3x3 单色 glyph 组件，且 glyph 颜色属于 `{1,2,3}`。
#    - generate_input() 保证：
#        * 至少存在一个 repeated-area/different-colors 配对；
#        * 至少存在一个 repeated-color/different-areas 配对；
#        * 融合规则下的输出不等于原题几何规则下的输出；
#        * 不会触发 main() 中的边界断言。


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a split-grid glyph panel into a colored polyline on the left canvas.

    Input structure:
    - A full-height divider separates a left 7-column glyph area from a right-side start area.
    - The left area contains monochrome 3x3 glyphs in fixed tile slots; the right area contains one gray start pixel.

    Rule:
    - Remove the divider, shift the gray start onto the left output canvas, and read glyphs in column-major order.
    - Each glyph's pixel area determines only the segment length: area 4 -> 2, area 5 -> 3, area 6 -> 4, area 7 -> 5.
    - Each glyph's color determines both the drawing direction and the paint color: blue(1) -> down, red(2) -> right,
      green(3) -> left. For each glyph, first step down one row, then draw the segment in that glyph color.

    Key insight:
    - Shape and color have been fully disentangled: area encodes distance, while color simultaneously encodes direction
      and the visible ink of the segment.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)

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
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 5}
    color_to_dir = {
        1: (1, 0),   # blue -> down
        2: (0, 1),   # red -> right
        3: (0, -1),  # green -> left
    }

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        assert area in area_to_len, f"unsupported glyph area: {area}"
        assert color in color_to_dir, f"unsupported glyph color: {color}"

        seg_len = int(area_to_len[area])
        dir_r, dir_c = color_to_dir[color]

        r_cur = int(r_cur + 1)
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, int(c_cur)] = color

        if dir_r == 1:
            end_r = r_cur + (seg_len - 1)
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            end_c = int(c_cur) + dir_c * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_c}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_c)
                out[r_cur, c_cur] = color

    return out


# --------------- generator ---------------

N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 5}
_COLOR_TO_DIR = {
    1: (1, 0),
    2: (0, 1),
    3: (0, -1),
}
_OLD_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_OLD_AREA_TO_DIR = {
    4: (1, 0),
    5: (0, 1),
    6: (0, -1),
    7: (0, -1),
}
_AREAS = (4, 5, 6, 7)
_GLYPH_COLORS = (1, 2, 3)
_INSTRUCTIONS = tuple((area, color) for area in _AREAS for color in _GLYPH_COLORS)

MIN_DIRECTION_CHANGES = 2
MIN_UNIQUE_AREAS = 3
MIN_OLD_LOGIC_MISMATCHES = 2
MAX_ATTEMPTS = 600
SEQUENCE_ATTEMPTS = 800


def _trace_segment(
    r: int,
    c: int,
    seg_len: int,
    dir_r: int,
    dir_c: int,
    h: int,
    *,
    col_bounds: tuple[int, int] | None,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    r0 = int(r + 1)
    c0 = int(c)
    if r0 >= h:
        return None

    if dir_r == 1:
        end_r = r0 + (int(seg_len) - 1)
        if end_r >= h:
            return None
        if col_bounds is not None:
            col_lo, col_hi = col_bounds
            if not (int(col_lo) <= c0 <= int(col_hi)):
                return None
        cells = [(int(r0 + dr), c0) for dr in range(int(seg_len))]
        return cells, (int(end_r), c0)

    end_c = c0 + int(dir_c) * (int(seg_len) - 1)
    if col_bounds is not None:
        col_lo, col_hi = col_bounds
        lo = min(c0, int(end_c))
        hi = max(c0, int(end_c))
        if lo < int(col_lo) or hi > int(col_hi):
            return None
    cells = [(r0, int(c0 + int(dir_c) * dc)) for dc in range(int(seg_len))]
    return cells, (r0, int(end_c))


def _trace_instruction(
    r: int,
    c: int,
    area: int,
    color: int,
    h: int,
    *,
    fused_rule: bool,
    col_bounds: tuple[int, int] | None,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    if fused_rule:
        seg_len = int(_AREA_TO_LEN[area])
        dir_r, dir_c = _COLOR_TO_DIR[color]
    else:
        seg_len = int(_OLD_AREA_TO_LEN[area])
        dir_r, dir_c = _OLD_AREA_TO_DIR[area]
    return _trace_segment(int(r), int(c), seg_len, int(dir_r), int(dir_c), int(h), col_bounds=col_bounds)


def _direction_changes(instructions: list[tuple[int, int]]) -> int:
    dirs = [tuple(_COLOR_TO_DIR[color]) for _area, color in instructions]
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def _old_logic_mismatches(instructions: list[tuple[int, int]]) -> int:
    return sum(1 for area, color in instructions if tuple(_OLD_AREA_TO_DIR[area]) != tuple(_COLOR_TO_DIR[color]))


def _required_unique_areas(num_glyphs: int) -> int:
    # With 5+ glyphs we can expose all four length classes while still keeping the required contrast pairs.
    return 4 if int(num_glyphs) >= 5 else MIN_UNIQUE_AREAS


def _has_fusion_support(instructions: list[tuple[int, int]], *, min_unique_areas: int = MIN_UNIQUE_AREAS) -> bool:
    colors = {color for _area, color in instructions}
    areas = [area for area, _color in instructions]

    area_to_colors: dict[int, set[int]] = {}
    color_to_areas: dict[int, set[int]] = {}
    for area, color in instructions:
        area_to_colors.setdefault(int(area), set()).add(int(color))
        color_to_areas.setdefault(int(color), set()).add(int(area))

    same_area_diff_colors = any(len(cols) >= 2 for cols in area_to_colors.values())
    same_color_diff_areas = any(len(aset) >= 2 for aset in color_to_areas.values())

    return (
        len(colors) == len(_GLYPH_COLORS)
        and len(set(areas)) >= int(min_unique_areas)
        and same_area_diff_colors
        and same_color_diff_areas
        and _direction_changes(instructions) >= MIN_DIRECTION_CHANGES
        and _old_logic_mismatches(instructions) >= MIN_OLD_LOGIC_MISMATCHES
    )


def _make_support_core() -> list[tuple[int, int]]:
    while True:
        repeated_area, area_b, area_c = random.sample(_AREAS, 3)
        repeated_color, color_b, color_c = random.sample(_GLYPH_COLORS, 3)
        core = [
            (repeated_area, repeated_color),
            (repeated_area, color_b),
            (area_b, repeated_color),
            (area_c, color_c),
        ]
        random.shuffle(core)
        if _has_fusion_support(core):
            return core


def _complete_instruction_list(core: list[tuple[int, int]], num_glyphs: int) -> list[tuple[int, int]]:
    instructions = list(core)
    if int(num_glyphs) >= 5:
        present_areas = {area for area, _color in instructions}
        missing_areas = [area for area in _AREAS if area not in present_areas]
        for area in missing_areas:
            if len(instructions) >= int(num_glyphs):
                break
            instructions.append((int(area), random.choice(_GLYPH_COLORS)))

    while len(instructions) < int(num_glyphs):
        instructions.append(random.choice(_INSTRUCTIONS))
    random.shuffle(instructions)
    return instructions


def _simulate_relative_path(n: int, instructions: list[tuple[int, int]]) -> tuple[int, int] | None:
    h = 4 * n - 1
    r, c = GRAY_START_ROW, 0
    occupied = {(int(r), int(c))}
    min_c = 0
    max_c = 0

    for area, color in instructions:
        traced = _trace_instruction(int(r), int(c), int(area), int(color), int(h), fused_rule=True, col_bounds=None)
        if traced is None:
            return None
        cells, (r, c) = traced
        if any((int(rr), int(cc)) in occupied for rr, cc in cells):
            return None
        for rr, cc in cells:
            occupied.add((int(rr), int(cc)))
            min_c = min(min_c, int(cc))
            max_c = max(max_c, int(cc))

    return int(min_c), int(max_c)


def _simulate_output(
    h: int,
    start_col: int,
    instructions: list[tuple[int, int]],
    *,
    fused_rule: bool,
) -> np.ndarray | None:
    out = np.zeros((h, LEFT_W), dtype=int)
    out[GRAY_START_ROW, int(start_col)] = GRAY

    r, c = GRAY_START_ROW, int(start_col)
    occupied = {(int(r), int(c))}
    for area, color in instructions:
        traced = _trace_instruction(
            int(r),
            int(c),
            int(area),
            int(color),
            int(h),
            fused_rule=fused_rule,
            col_bounds=(0, LEFT_W - 1),
        )
        if traced is None:
            return None
        cells, (r, c) = traced
        if any((int(rr), int(cc)) in occupied for rr, cc in cells):
            return None
        for rr, cc in cells:
            out[int(rr), int(cc)] = int(color)
            occupied.add((int(rr), int(cc)))
    return out


def _sample_instruction_sequence(n: int) -> tuple[list[tuple[int, int]], int, np.ndarray] | None:
    num_glyphs = 2 * n
    h = 4 * n - 1
    min_unique_areas = _required_unique_areas(num_glyphs)

    for _ in range(SEQUENCE_ATTEMPTS):
        instructions = _complete_instruction_list(_make_support_core(), num_glyphs)

        if not _has_fusion_support(instructions, min_unique_areas=min_unique_areas):
            continue

        bounds = _simulate_relative_path(n, instructions)
        if bounds is None:
            continue
        min_c, max_c = bounds

        lo = max(0, -int(min_c))
        hi = min(LEFT_W - 1, LEFT_W - 1 - int(max_c))
        if lo > hi:
            continue

        start_col = random.randint(int(lo), int(hi))
        expected = _simulate_output(h, int(start_col), instructions, fused_rule=True)
        if expected is None:
            continue
        old_output = _simulate_output(h, int(start_col), instructions, fused_rule=False)
        if old_output is not None and np.array_equal(expected, old_output):
            continue

        return instructions, int(start_col), expected
    return None


def _render_grid(n: int, start_col: int, instructions: list[tuple[int, int]]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, (area, color) in enumerate(instructions):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[int(area)] * int(color)
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(N_MIN, N_MAX)
    sampled = _sample_instruction_sequence(n)
    if sampled is None:
        return None

    instructions, start_col, expected = sampled
    return _render_grid(n, start_col, instructions), expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")

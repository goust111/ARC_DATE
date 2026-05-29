# Combo variant of 136b0064 (index 30)
# combo_spec: D2/2+D3/3+D5/2
# Name: 双色编码与长度着色系统

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace:
#    - 布局保留原题：15 列输入，中间第 7 列整列分隔线；左侧 7 列仍是两列 3x3 tile，
#      右侧仍只有一个灰色起点。
#    - 原题里“单色 shape 的面积”一次性决定方向、长度和线段颜色。融合后拆成三层：
#        1) 字形外圈颜色决定方向；
#        2) 字形总面积决定长度；
#        3) 长度再决定输出线段颜色。
#    - 为落实 D3 的“双色字形”，每个 3x3 模板都改成双色：
#        * 一个固定的模板像素固定为长度色；
#        * 其余前景像素全部为方向色。
#      这样同一个 glyph 同时暴露“方向通道”和“长度/输出色通道”，但真正的长度仍由面积给出。
#    - 小例子：一个 area=6 的 cross 若外圈为红(2)、中心为青(8)，则它表示“向右画 4 格，
#      且这段输出用颜色 8”。原题的同面积字形永远是“向左 4 且颜色绑定形状”；
#      差异在于方向与输出色都不再由 shape 单独锁定。
# b) Complexity calibration:
#    - 方向只保留 {下, 右, 左} 三档，避免新增“上”导致组合空间过大。
#    - 长度使用四档 2/3/4/5，对应 area 4/5/6/7，避免原题里两个面积同长导致
#      “长度->颜色”证据被削弱。
#    - 方向色与输出色使用互不重叠的两组小调色板：
#        direction colors = {1,2,3}, output colors = {6,7,8,9}.
#      这样人眼能直接看出“字形外圈色”和“折线段色”是两套不同系统。
#    - 生成器强制出现：
#        * 同面积不同方向色；
#        * 同方向色不同面积；
#        * 至少 3 种面积与 3 种方向。
#      否则解题者可能把规则误读成“某个双色模板直接对应一条固定指令”。
# c) Hidden-assumption review of the original main():
#    - 原 solver 假设“每个字形是单色 8 连通组件”；融合后失效，因为一个 glyph 会被拆成两色。
#      新 solver 改为扫描固定 tile 槽位，而不是按单色组件找字形。
#    - 原 solver 假设 `area == 4` 等价于竖直段；融合后方向改由颜色决定，必须替换为
#      `direction_color -> (dr, dc)`。
#    - 原 solver 直接把 glyph 颜色作为输出色；融合后输出色改为 `length -> color`。
#    - 新 solver 额外依赖一个生成器合同：每个 glyph 的中心像素是长度色，且长度色与
#      `area -> len -> output_color` 一致。solver 会检查这一点，generator 负责保证。
# d) Placement legality:
#    - 仍采用 path-first：先采样合法指令序列，再编码回双色字形。
#    - 路径模拟显式检查：所有中间单元都在左画布 7 列内，底边不越界。
#    - tile 位置仍固定在 c0={0,4}、r0=0,4,8...，因此输入 glyph 永不重叠。
# e) Visual observability:
#    - 同面积不同方向色会在输出里产生“相同长度、不同走向、同一种长度色”的配对。
#    - 同方向色不同面积会产生“同走向、不同长度、不同输出色”的配对。
#    - 固定标记位颜色与输出线段颜色一致，给出长度色映射的直接视觉锚点；其余前景颜色则在
#      多个不同面积上重复，给出方向映射的横向对比。
# f) Visual cleanliness:
#    - 输入仍分成字形区 / 分隔线 / 起点区三块，结构与原题一致。
#    - 输出路径单调向下推进，不会回到更高行；不同颜色段不会在同一像素上叠写，便于观察。
# g) Change inferability:
#    - 颜色总数虽多于原题，但被拆成两种角色明确的局部模式：
#        * 字形外圈色只来自 3 色方向表；
#        * 字形标记位颜色与输出段色只来自 4 色长度表。
#      这种“位置角色 + 重复对比”足以让人从少量样例推断规则。
# Solver/generator contract:
#    - main() 假设左侧非空 tile 都是合法 3x3 glyph，且每个 glyph 恰有两种非背景颜色。
#    - generate_input() 保证每个标记位都使用正确长度色，并拒绝与旧 area->direction
#      逻辑过于接近的序列，避免退化成原题或单一维度变体。


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
    Decode two-color 3x3 glyphs into a length-colored polyline on the left canvas.

    Input structure:
    - A full-height divider splits the grid into a left 7-column glyph panel and a right start-marker panel.
    - Each occupied tile slot on the left contains one 3x3 glyph made of exactly two foreground colors:
      most foreground pixels share one color, and one fixed marker pixel uses a second color.
    - The right side contains one gray (5) start cell.

    Rule:
    - Remove the divider, shift the gray start into the left output canvas, and read glyphs in column-major order.
    - For each glyph, its outer color determines the segment direction, while its total foreground area determines the segment length.
    - The segment is then painted with the color assigned to that length; the glyph's fixed marker pixel is that same length color.
    - As in the base task, each new segment starts one row below the current cursor before extending in its decoded direction.

    Key insight:
    - The instruction is no longer stored in one feature. You must combine a color channel for direction with an area channel
      for distance, then translate the recovered length through a separate length-to-output-color map.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, _div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs: list[dict[str, int]] = []
    num_tile_rows = (h + 1) // TILE_ROW_STRIDE
    for tile_col in TILE_COLS:
        for tile_row in range(int(num_tile_rows)):
            r0 = TILE_ROW_STRIDE * int(tile_row)
            if r0 + 3 > h:
                continue
            tile = left[r0 : r0 + 3, tile_col : tile_col + 3]
            if tile.shape != (3, 3) or not np.any(tile != bg):
                continue

            area = int(np.count_nonzero(tile != bg))
            assert area in AREA_TO_LEN, f"unsupported glyph area: {area}"

            marker_color = int(tile[MARKER_POS])
            assert marker_color != bg, "glyph marker must be occupied"

            fg_colors = sorted(int(v) for v in np.unique(tile) if int(v) != bg)
            assert len(fg_colors) == 2, f"glyph must be bichromatic, got {fg_colors}"

            direction_candidates = [color for color in fg_colors if int(color) != marker_color]
            assert len(direction_candidates) == 1, f"could not isolate direction color from {fg_colors}"
            direction_color = int(direction_candidates[0])
            assert direction_color in DIR_COLOR_TO_DELTA, f"unsupported direction color: {direction_color}"

            expected_paint = int(LEN_TO_OUTPUT_COLOR[AREA_TO_LEN[area]])
            assert marker_color == expected_paint, (
                f"glyph marker color {marker_color} inconsistent with area {area} -> len {AREA_TO_LEN[area]}"
            )

            glyphs.append(
                {
                    "r0": int(r0),
                    "c0": int(tile_col),
                    "area": int(area),
                    "direction_color": int(direction_color),
                    "paint_color": int(marker_color),
                }
            )

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        area = int(glyph["area"])
        seg_len = int(AREA_TO_LEN[area])
        dir_r, dir_c = DIR_COLOR_TO_DELTA[int(glyph["direction_color"])]
        paint_color = int(glyph["paint_color"])

        r_cur = int(r_cur + 1)
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, int(c_cur)] = paint_color

        if dir_r == 1:
            end_r = r_cur + (seg_len - 1)
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = paint_color
        else:
            end_c = int(c_cur) + dir_c * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_c}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_c)
                out[r_cur, c_cur] = paint_color

    return out


# --------------- generator ---------------

N_MIN, N_MAX = 3, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
MARKER_POS = (0, 0)
AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 5}
LEN_TO_OUTPUT_COLOR = {2: 6, 3: 7, 4: 8, 5: 9}
DIR_COLOR_TO_DELTA = {
    1: (1, 0),   # blue -> down
    2: (0, 1),   # red -> right
    3: (0, -1),  # green -> left
}
OLD_AREA_TO_DELTA = {
    4: (1, 0),
    5: (0, 1),
    6: (0, -1),
    7: (0, -1),
}
AREAS = (4, 5, 6, 7)
DIR_COLORS = (1, 2, 3)
INSTRUCTIONS = tuple((area, dir_color) for area in AREAS for dir_color in DIR_COLORS)

MIN_DIRECTION_CHANGES = 2
REQUIRED_UNIQUE_AREAS = len(AREAS)
MIN_OLD_LOGIC_MISMATCHES = 2
MAX_ATTEMPTS = 600
SEQUENCE_ATTEMPTS = 600


def _build_glyph(area: int, direction_color: int) -> np.ndarray:
    glyph = np.zeros((3, 3), dtype=int)
    glyph[TEMPLATES[int(area)] == 1] = int(direction_color)
    glyph[MARKER_POS] = int(LEN_TO_OUTPUT_COLOR[AREA_TO_LEN[int(area)]])
    return glyph


def _instruction_cells(
    r: int, c: int, area: int, direction_color: int, h: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    seg_len = int(AREA_TO_LEN[int(area)])
    dir_r, dir_c = DIR_COLOR_TO_DELTA[int(direction_color)]
    r0 = int(r + 1)
    c0 = int(c)
    if r0 >= h:
        return None

    cells: list[tuple[int, int]] = []
    if dir_r == 1:
        end_r = r0 + (seg_len - 1)
        if end_r >= h:
            return None
        for dr in range(seg_len):
            cells.append((int(r0 + dr), c0))
        return cells, (int(end_r), c0)

    end_c = c0 + dir_c * (seg_len - 1)
    if not (0 <= end_c < LEFT_W):
        return None
    for dc in range(seg_len):
        cells.append((r0, int(c0 + dir_c * dc)))
    return cells, (r0, int(end_c))


def _direction_changes(instructions: list[tuple[int, int]]) -> int:
    deltas = [tuple(DIR_COLOR_TO_DELTA[int(direction_color)]) for _area, direction_color in instructions]
    return sum(1 for i in range(1, len(deltas)) if deltas[i] != deltas[i - 1])


def _old_logic_mismatches(instructions: list[tuple[int, int]]) -> int:
    return sum(
        1
        for area, direction_color in instructions
        if tuple(OLD_AREA_TO_DELTA[int(area)]) != tuple(DIR_COLOR_TO_DELTA[int(direction_color)])
    )


def _has_variant_support(instructions: list[tuple[int, int]]) -> bool:
    areas = [int(area) for area, _direction_color in instructions]
    direction_colors = [int(direction_color) for _area, direction_color in instructions]

    area_to_dirs: dict[int, set[int]] = {}
    dir_to_areas: dict[int, set[int]] = {}
    for area, direction_color in instructions:
        area_to_dirs.setdefault(int(area), set()).add(int(direction_color))
        dir_to_areas.setdefault(int(direction_color), set()).add(int(area))

    same_area_diff_dirs = any(len(dir_set) >= 2 for dir_set in area_to_dirs.values())
    same_dir_diff_areas = any(len(area_set) >= 2 for area_set in dir_to_areas.values())

    return (
        len(set(direction_colors)) == len(DIR_COLORS)
        and len(set(areas)) == REQUIRED_UNIQUE_AREAS
        and same_area_diff_dirs
        and same_dir_diff_areas
        and _direction_changes(instructions) >= MIN_DIRECTION_CHANGES
        and _old_logic_mismatches(instructions) >= MIN_OLD_LOGIC_MISMATCHES
    )


def _make_support_core() -> list[tuple[int, int]]:
    while True:
        repeated_area, area_b, area_c, area_d = random.sample(AREAS, 4)
        repeated_dir, dir_b, dir_c = random.sample(DIR_COLORS, 3)
        core = [
            (repeated_area, repeated_dir),
            (repeated_area, dir_b),
            (area_b, repeated_dir),
            (area_c, dir_c),
            (area_d, repeated_dir),
        ]
        random.shuffle(core)
        if _has_variant_support(core):
            return core


def _simulate_relative_path(n: int, instructions: list[tuple[int, int]]) -> tuple[int, int] | None:
    h = 4 * n - 1
    r, c = GRAY_START_ROW, 0
    occupied = {(int(r), int(c))}
    min_c = 0
    max_c = 0

    for area, direction_color in instructions:
        traced = _instruction_cells(int(r), int(c), int(area), int(direction_color), h)
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


def _sample_instruction_sequence(n: int) -> tuple[list[tuple[int, int]], int] | None:
    num_glyphs = 2 * n
    for _ in range(SEQUENCE_ATTEMPTS):
        instructions = _make_support_core()
        while len(instructions) < num_glyphs:
            instructions.append(random.choice(INSTRUCTIONS))
        random.shuffle(instructions)

        if not _has_variant_support(instructions):
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
        return instructions, int(start_col)
    return None


def _simulate_output(
    h: int,
    start_col: int,
    instructions: list[tuple[int, int]],
    *,
    use_old_motion: bool,
) -> np.ndarray | None:
    out = np.zeros((h, LEFT_W), dtype=int)
    out[GRAY_START_ROW, int(start_col)] = GRAY

    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    for area, direction_color in instructions:
        seg_len = int(AREA_TO_LEN[int(area)])
        paint_color = int(LEN_TO_OUTPUT_COLOR[seg_len])
        if use_old_motion:
            dir_r, dir_c = OLD_AREA_TO_DELTA[int(area)]
        else:
            dir_r, dir_c = DIR_COLOR_TO_DELTA[int(direction_color)]

        r_cur += 1
        if not (0 <= r_cur < h and 0 <= c_cur < LEFT_W):
            return None
        out[r_cur, c_cur] = paint_color
        if dir_r == 1:
            for _ in range(seg_len - 1):
                r_cur += 1
                if not (0 <= r_cur < h):
                    return None
                out[r_cur, c_cur] = paint_color
        else:
            for _ in range(seg_len - 1):
                c_cur += dir_c
                if not (0 <= c_cur < LEFT_W):
                    return None
                out[r_cur, c_cur] = paint_color
    return out


def _render_grid(n: int, start_col: int, instructions: list[tuple[int, int]]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, (area, direction_color) in enumerate(instructions):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _build_glyph(int(area), int(direction_color))
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = random.randint(N_MIN, N_MAX)
    h = 4 * n - 1
    sampled = _sample_instruction_sequence(n)
    if sampled is None:
        return None

    instructions, start_col = sampled
    expected = _simulate_output(h, start_col, instructions, use_old_motion=False)
    assert expected is not None
    old_motion = _simulate_output(h, start_col, instructions, use_old_motion=True)
    if old_motion is not None and np.array_equal(expected, old_motion):
        return None

    return _render_grid(n, start_col, instructions), expected, old_motion


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected, old_motion = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            if old_motion is not None and np.array_equal(out, old_motion):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")

# Variant of 136b0064
# Dimension: D2 — 字形编码方式——面积到指令的映射 (transform)
# Change: 字形像素面积唯一决定线段：area 4→垂直向下2格，area 5→水平向右3格，area 6→水平向左4格，area 7→水平向左2格
#         → 字形颜色决定线段方向，面积决定线段长度

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the D2 variant
# ----------------------------------
# concretization:
#   - 输入布局保持原题：15 列网格，中间第 7 列是整列分隔线；左侧 7 列里放两列 3x3 字形，
#     右侧只保留一个灰色起点。
#   - 变化只发生在“字形如何解码成折线段”这一层。字形的 3x3 形状仍然只通过像素面积被识别，
#     但面积现在只控制长度：area 4/5/6/7 -> len 2/3/4/5。方向改由颜色决定：
#     蓝(1)向下、红(2)向右、绿(3)向左。输出线段仍使用该字形自己的颜色。
#   - 具体追踪一个字形：原题里 area 6 一定画“左 4”；现在一个 area 6 的红色 cross 会画
#     “右 4”，绿色 cross 会画“左 4”，蓝色 cross 会画“下 4”。因此同一形状可在不同样例中
#     复用为不同方向，而相同颜色在不同形状上保持同方向但长度改变。
# complexity_calibration:
#   - 只保留原题已有的三个方向集合 {下, 左, 右}，不引入“上”，避免额外维度。
#   - 长度使用四档 2/3/4/5，让四种面积都保留辨识价值；若继续沿用两个面积同长，会削弱
#     “面积编码长度”的可观察性。
#   - 生成器强制出现“同面积不同颜色”与“同颜色不同面积”，否则解题者可能仍误以为 shape 直接
#     编码完整指令。
# implicit_assumptions_review:
#   - 原 solver 中 `if area == 4` 等价于“竖直段”；这个前提在变体中失效，必须改成由颜色判断。
#   - 原 `area_to_dir = {5:1, 6:-1, 7:-1}` 假设所有非 area4 都是水平段；现在 area 6 也可能向下。
#   - 原 generator 用 `_AREA_TO_COLOR` 把颜色绑定到面积；该绑定必须拆掉，否则输入会退化回原题。
#   - 原 solver 在绘制时直接用 glyph color 作为输出色，这一写法在变体中仍成立，而且正好让
#     “颜色=方向”在 output 里继续可见。
# placement_legality:
#   - 仍按 path-first 生成：先构造一条合法折线，再把每一步编码回字形。
#   - 路径模拟显式检查：不会越出左画布，不会触底，不会重复覆盖既有路径单元。
#   - 左侧 tile 位置和间距沿用原题，所有 3x3 字形 bbox 固定，不会互相重叠。
# visual_observability:
#   - 每个样本至少包含三种方向颜色 1/2/3；否则单个 input->output 无法支撑颜色映射。
#   - 每个样本至少包含三种面积，并包含一个“同面积不同色”的配对和一个“同色不同面积”的配对，
#     让颜色规则与长度规则都能被肉眼比较出来。
#   - 额外要求至少两条指令与原题 area->direction 逻辑不一致，从而避免新样本在输出上偶然与原题相同。
# cleanliness:
#   - 输出路径禁止自交/复写，避免多种颜色叠在同一单元造成难读。
#   - 输入仍只有三类视觉区域：左侧字形区、中间黄分隔线、右侧灰起点区，整体保持整洁。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a colored polyline by decoding 3x3 glyphs on the left half of a split grid.

    Input structure:
    - A full-height divider separates a left 7-column glyph canvas from a right-side area.
    - The left canvas contains monochrome 3x3 glyphs placed in fixed tile slots; the right side contains one gray start pixel.

    Rule:
    - Remove the divider and shift the gray start onto the left output canvas.
    - Read glyphs in column-major order (top-to-bottom in the first tile column, then the second).
    - Each glyph's pixel area determines only the segment length, while the glyph's color determines the direction:
      blue(1) = down, red(2) = right, green(3) = left.
    - For each glyph, step down one row from the current cursor and draw the segment in the glyph's own color.

    Key insight: Shape and color now encode different parts of the instruction, so solving requires combining two independent clues:
    area gives the distance, and color gives the direction.
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

# --- glyph grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 5}
_COLOR_TO_DIR = {
    1: (1, 0),   # blue -> down
    2: (0, 1),   # red -> right
    3: (0, -1),  # green -> left
}
_OLD_AREA_TO_DIR = {
    4: (1, 0),
    5: (0, 1),
    6: (0, -1),
    7: (0, -1),
}
_AREAS = (4, 5, 6, 7)
_GLYPH_COLORS = (1, 2, 3)
_INSTRUCTIONS = tuple((area, color) for area in _AREAS for color in _GLYPH_COLORS)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MIN_UNIQUE_AREAS = 3
MIN_OLD_LOGIC_MISMATCHES = 2
MAX_ATTEMPTS = 500
SEQUENCE_ATTEMPTS = 500


def _instruction_cells(r: int, c: int, area: int, color: int, h: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    seg_len = int(_AREA_TO_LEN[area])
    dir_r, dir_c = _COLOR_TO_DIR[color]
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
    for dc in range(seg_len):
        cells.append((r0, int(c0 + dir_c * dc)))
    return cells, (r0, int(end_c))


def _direction_changes(instructions: list[tuple[int, int]]) -> int:
    dirs = [tuple(_COLOR_TO_DIR[color]) for _area, color in instructions]
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def _old_logic_mismatches(instructions: list[tuple[int, int]]) -> int:
    return sum(1 for area, color in instructions if tuple(_OLD_AREA_TO_DIR[area]) != tuple(_COLOR_TO_DIR[color]))


def _has_variant_support(instructions: list[tuple[int, int]]) -> bool:
    colors = {color for _area, color in instructions}
    areas = [area for area, _color in instructions]
    unique_areas = set(areas)

    area_to_colors: dict[int, set[int]] = {}
    color_to_areas: dict[int, set[int]] = {}
    for area, color in instructions:
        area_to_colors.setdefault(int(area), set()).add(int(color))
        color_to_areas.setdefault(int(color), set()).add(int(area))

    same_area_diff_colors = any(len(cols) >= 2 for cols in area_to_colors.values())
    same_color_diff_areas = any(len(aset) >= 2 for aset in color_to_areas.values())

    return (
        len(colors) == len(_GLYPH_COLORS)
        and len(unique_areas) >= MIN_UNIQUE_AREAS
        and same_area_diff_colors
        and same_color_diff_areas
        and _direction_changes(instructions) >= MIN_DIRECTION_CHANGES
        and _old_logic_mismatches(instructions) >= MIN_OLD_LOGIC_MISMATCHES
    )


def _make_support_quad() -> list[tuple[int, int]]:
    while True:
        repeated_area, area_b, area_c = random.sample(_AREAS, 3)
        repeated_color, color_b, color_c = random.sample(_GLYPH_COLORS, 3)
        quad = [
            (repeated_area, repeated_color),
            (repeated_area, color_b),
            (area_b, repeated_color),
            (area_c, color_c),
        ]
        random.shuffle(quad)
        if _has_variant_support(quad):
            return quad


def _simulate_relative_path(n: int, instructions: list[tuple[int, int]]) -> tuple[int, int] | None:
    h = 4 * n - 1
    r, c = GRAY_START_ROW, 0
    occupied = {(int(r), int(c))}
    min_c = 0
    max_c = 0

    for area, color in instructions:
        traced = _instruction_cells(int(r), int(c), int(area), int(color), h)
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
        instructions = _make_support_quad()
        while len(instructions) < num_glyphs:
            instructions.append(random.choice(_INSTRUCTIONS))
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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    sampled = _sample_instruction_sequence(n)
    if sampled is None:
        return None
    instructions, start_col = sampled
    return _render_grid(n, start_col, instructions)


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

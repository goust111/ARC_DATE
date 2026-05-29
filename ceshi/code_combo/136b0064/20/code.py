# Combo variant of 136b0064 (index 20)
# combo_spec: D2/2+D3/3
# Name: 双色字形的颜色编码

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization:
#   - 保留原题的整体几何：15 列输入、中间第 7 列整列分隔线、左侧 7 列为两列 3x3 tile、右侧
#     只有一个灰色起点；输出仍是去掉分隔线后的左侧 7 列画布。
#   - 融合后的每个字形仍沿用原题 4 种固定 3x3 形状，所以“总占用像素数 area”仍可稳定读出。
#     但字形不再是单色：每个字形恰好有两种颜色。左上角那个固定存在的占用像素改为“方向标记色”，
#     其余占用像素全部是“主色”。
#   - 具体追踪一个例子：原题里 area 6 的 cross 一定画“向左 4”。现在如果该 glyph 是
#     [左上角=红(2), 其余=橙(7)] 的双色 cross，则 area 6 只给出长度 4，红色标记给出
#     “向右”，最终在输出中画一段长度 4 的橙色水平线。和原题相比，shape 不再独自决定整条指令。
# complexity_calibration:
#   - 方向只保留三类 {下, 右, 左}，避免引入“上”导致样例稀疏；映射固定为蓝(1)=下、红(2)=右、
#     绿(3)=左。三种颜色足以让人从 2-4 个样例中对照出规律。
#   - 面积采用四档唯一长度 {4:2, 5:3, 6:4, 7:5}，这样四种形状都保有清晰的长度辨识价值，
#     不会因为重复长度让 area 的作用变弱。
#   - 主色与方向色使用互不重叠的调色板：方向标记只用 {1,2,3}，主色只用 {6,7,8}。这样
#     解题者能肉眼区分“哪个颜色是角标、哪个颜色是要被画出来的主色”。
# implicit_assumptions_review:
#   - 原 solver 用 `monochromatic=True` 找单色组件，这在双色字形下失效；现在必须用
#     `monochromatic=False` 把整个双色 glyph 当作一个 8 连通前景组件处理。
#   - 原 solver 把 `comp['color']` 直接当作输出色；现在组件没有单一颜色，需要从 3x3 bbox 内
#     明确拆出“左上角方向色”和“其余主色”。
#   - 原 solver 假设 area 4 等价于竖直段、其余 area 等价于水平段；融合后方向完全改由
#     方向标记色控制，因此任何 area 都可能对应下/左/右三种方向之一。
#   - 新 solver 额外依赖四个前提：字形 bbox 都是 3x3；左上角单元一定被占用；每个字形恰有两种
#     前景色；方向标记色只出现 1 次。生成器会显式保证这些前提。
# placement_legality:
#   - 仍采用 path-first：先采样合法折线路径，再把每一步编码成 (area, direction-marker, main-color)。
#   - 路径模拟检查不会越出 7 列左画布、不会触底、不会与已有折线复写，从而保证 solver 中的
#     边界 assert 都成立，且输出保持整洁。
#   - tile 布局、bbox 尺寸、列主序读取顺序全部沿用原题，因此不会引入额外的空间歧义。
# visual_observability:
#   - 每个样本强制包含三种方向标记色和三种主色，并至少出现以下四类可比对现象：
#       1. 同面积不同方向色；
#       2. 同方向色不同面积；
#       3. 同方向色不同主色；
#       4. 同主色不同方向色。
#     这样人类可以分别观察“方向由哪个颜色控制”“长度由什么控制”“输出为什么跟主色走”。
#   - 额外要求至少两条指令与原题的 area->direction 逻辑不一致，并统计“融合输出是否和原逻辑
#     输出完全相同”；若仍经常相同，就说明样本退化，需要拒绝重采样。
# cleanliness:
#   - 每个字形内部只有一个角标像素变色，其余像素保持统一主色，视觉上像“主色字形 + 角标”。
#   - 输出折线禁止自交和颜色复写，避免不同语义的线段叠在一起导致读图混乱。


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


def _decode_bicolor_glyph(sprite: np.ndarray, *, bg: int) -> tuple[int, int, int]:
    occupied = sprite[sprite != bg]
    area = int(occupied.size)
    assert int(sprite[0, 0]) != bg, "direction marker must occupy the top-left glyph cell"

    counts = Counter(map(int, occupied.tolist()))
    assert len(counts) == 2, f"glyph must use exactly two foreground colors, got {sorted(counts)}"

    dir_color = int(sprite[0, 0])
    assert counts[dir_color] == 1, "direction marker color must appear exactly once"

    main_colors = [color for color in counts if int(color) != dir_color]
    assert len(main_colors) == 1, f"glyph main color is ambiguous: {counts}"
    main_color = int(main_colors[0])
    return area, dir_color, main_color


def _extract_glyphs(left: np.ndarray, *, bg: int) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=False, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, r1, c1 = map(int, b)
        sprite = left[r0 : r1 + 1, c0 : c1 + 1]
        area, dir_color, main_color = _decode_bicolor_glyph(sprite, bg=bg)
        glyphs.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "area": int(area),
                "dir_color": int(dir_color),
                "main_color": int(main_color),
            }
        )
    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a colored polyline by decoding bicolor 3x3 glyphs on the left side of a split grid.

    Input structure:
    - A full-height divider separates a left 7-column glyph canvas from a right-side start area.
    - Each occupied left-side tile is a fixed 3x3 glyph shape, but every glyph now uses exactly two colors:
      a one-cell corner marker and a dominant main color.
    - The right side still contains one gray start pixel.

    Rule:
    - Remove the divider and shift the gray start into the left output canvas.
    - Read the glyphs in column-major order (top-to-bottom in the first tile column, then the second).
    - The total filled area of a glyph determines the segment length, while the glyph's top-left marker color
      determines the direction: blue(1)=down, red(2)=right, green(3)=left.
    - For each glyph, step down one row from the current cursor and draw the segment in the glyph's main color.

    Key insight: The instruction is split across two visual channels inside the same glyph:
    shape area gives distance, the corner marker gives direction, and the dominant color is only the paint color.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, _div_color = find_full_height_constant_col(g, bg=bg)
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
    glyphs = _extract_glyphs(left, bg=bg)

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 5}
    dir_color_to_delta = {
        1: (1, 0),   # blue -> down
        2: (0, 1),   # red -> right
        3: (0, -1),  # green -> left
    }

    for glyph in glyphs:
        area = int(glyph["area"])
        dir_color = int(glyph["dir_color"])
        main_color = int(glyph["main_color"])

        assert area in area_to_len, f"unsupported glyph area: {area}"
        assert dir_color in dir_color_to_delta, f"unsupported direction marker color: {dir_color}"

        seg_len = int(area_to_len[area])
        dir_r, dir_c = dir_color_to_delta[dir_color]

        r_cur = int(r_cur + 1)
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, int(c_cur)] = main_color

        if dir_r == 1:
            end_r = r_cur + (seg_len - 1)
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = main_color
        else:
            end_c = int(c_cur) + dir_c * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_c}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_c)
                out[r_cur, c_cur] = main_color

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
_DIR_COLOR_TO_DELTA = {
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
_DIR_COLORS = (1, 2, 3)
_MAIN_COLORS = (6, 7, 8)
_INSTRUCTIONS = tuple(
    (area, dir_color, main_color)
    for area in _AREAS
    for dir_color in _DIR_COLORS
    for main_color in _MAIN_COLORS
)

# --- generation constraints ---
MIN_UNIQUE_AREAS = 3
MIN_OLD_LOGIC_MISMATCHES = 2
MAX_ATTEMPTS = 800
SEQUENCE_ATTEMPTS = 800


def _paint_glyph(area: int, dir_color: int, main_color: int) -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    mask = _TEMPLATES[int(area)] == 1
    sprite[mask] = int(main_color)
    sprite[0, 0] = int(dir_color)
    return sprite


def _instruction_cells(
    r: int, c: int, area: int, dir_color: int, h: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    seg_len = int(_AREA_TO_LEN[area])
    dir_r, dir_c = _DIR_COLOR_TO_DELTA[dir_color]
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


def _old_logic_mismatches(instructions: list[tuple[int, int, int]]) -> int:
    return sum(
        1
        for area, dir_color, _main_color in instructions
        if tuple(_OLD_AREA_TO_DIR[area]) != tuple(_DIR_COLOR_TO_DELTA[dir_color])
    )


def _has_variant_support(instructions: list[tuple[int, int, int]]) -> bool:
    areas = [int(area) for area, _dir_color, _main_color in instructions]
    dir_colors = [int(dir_color) for _area, dir_color, _main_color in instructions]
    main_colors = [int(main_color) for _area, _dir_color, main_color in instructions]

    area_to_dirs: dict[int, set[int]] = {}
    dir_to_areas: dict[int, set[int]] = {}
    dir_to_mains: dict[int, set[int]] = {}
    main_to_dirs: dict[int, set[int]] = {}
    for area, dir_color, main_color in instructions:
        area_to_dirs.setdefault(int(area), set()).add(int(dir_color))
        dir_to_areas.setdefault(int(dir_color), set()).add(int(area))
        dir_to_mains.setdefault(int(dir_color), set()).add(int(main_color))
        main_to_dirs.setdefault(int(main_color), set()).add(int(dir_color))

    return (
        len(set(areas)) >= MIN_UNIQUE_AREAS
        and set(dir_colors) == set(_DIR_COLORS)
        and set(main_colors) == set(_MAIN_COLORS)
        and any(len(dirs) >= 2 for dirs in area_to_dirs.values())
        and any(len(aset) >= 2 for aset in dir_to_areas.values())
        and any(len(mset) >= 2 for mset in dir_to_mains.values())
        and any(len(dset) >= 2 for dset in main_to_dirs.values())
        and _old_logic_mismatches(instructions) >= MIN_OLD_LOGIC_MISMATCHES
    )


def _make_support_seed() -> list[tuple[int, int, int]]:
    while True:
        repeated_area = random.choice(_AREAS)
        other_areas = random.sample([area for area in _AREAS if area != repeated_area], 2)
        repeated_dir, dir_b, dir_c = random.sample(_DIR_COLORS, 3)
        repeated_main, main_b, main_c = random.sample(_MAIN_COLORS, 3)

        seed = [
            (repeated_area, repeated_dir, repeated_main),
            (repeated_area, dir_b, main_b),
            (other_areas[0], repeated_dir, main_c),
            (other_areas[1], dir_c, repeated_main),
        ]
        random.shuffle(seed)
        if _has_variant_support(seed):
            return seed


def _simulate_relative_path(n: int, instructions: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    h = 4 * n - 1
    r, c = GRAY_START_ROW, 0
    occupied = {(int(r), int(c))}
    min_c = 0
    max_c = 0

    for area, dir_color, _main_color in instructions:
        traced = _instruction_cells(int(r), int(c), int(area), int(dir_color), h)
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


def _sample_instruction_sequence(n: int) -> tuple[list[tuple[int, int, int]], int] | None:
    num_glyphs = 2 * n
    for _ in range(SEQUENCE_ATTEMPTS):
        instructions = _make_support_seed()
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


def _render_grid(n: int, start_col: int, instructions: list[tuple[int, int, int]]) -> np.ndarray:
    h = 4 * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, (area, dir_color, main_color) in enumerate(instructions):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _paint_glyph(int(area), int(dir_color), int(main_color))
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(N_MIN, N_MAX)
    sampled = _sample_instruction_sequence(n)
    if sampled is None:
        return None
    instructions, start_col = sampled
    return _render_grid(n, start_col, instructions)


def _instance_is_valid(inp: np.ndarray, out: np.ndarray) -> bool:
    if out.shape != (inp.shape[0], LEFT_W):
        return False

    left = inp[:, :DIVIDER_COL]
    glyphs = _extract_glyphs(left, bg=int(get_bg(inp)))
    expected_glyphs = 2 * ((inp.shape[0] + 1) // 4)
    if len(glyphs) != expected_glyphs:
        return False

    instructions = [(g["area"], g["dir_color"], g["main_color"]) for g in glyphs]
    return _has_variant_support(instructions)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if not _instance_is_valid(inp, out):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")

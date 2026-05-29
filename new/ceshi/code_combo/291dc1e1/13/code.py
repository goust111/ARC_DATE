# Combo variant of 291dc1e1 (index 13)
# combo_spec: D5/1+D6/3
# Name: 始终旋转+水平排列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for helper_dir in (ROOT / "arc2_solver_select", ROOT / "solvers"):
    if helper_dir.exists() and str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo D5/1 + D6/3
=====================================

a) 具体化与行为追踪
- 输入仍然是原题结构：一个角点 marker、两条边线、以及散落在两条边线附近的若干实心矩形。
- 原题先按 `(dist_to_blue, dist_to_red, along_blue, along_red)` 排序；如果蓝线竖直才旋转；
  然后把矩形自上而下纵向堆叠。
- 这个融合变体保留排序逻辑，但把“是否旋转”与“如何排布”同时改掉：
  每个矩形都先逆时针旋转 90 度，再从左到右水平排开，并在输出中垂直居中。
- 具体例子：
  若排序后得到 `A=2x4, B=2x2, C=2x6`，且蓝线是水平的。
  原题输出会是三块 `[2x4, 2x2, 2x6]` 纵向堆叠。
  本变体输出会是 `[4x2, 2x2, 6x2]` 横向排列。
  差异不只是方向翻转，还包括原本蓝线水平时不会旋转的物体现在也会旋转。

b) 复杂度校准
- 排序坐标系完全保留，因此题目的核心识别任务仍是“读出两条边线定义的顺序”。
- 改动仅落在两个少量取值的维度：旋转从“条件触发”变为“恒定触发”，排布从“纵向”
  变为“横向”。自由度没有膨胀。
- 为了避免视觉退化，生成器强制至少出现一个宽度大于 2 的矩形，并固定生成
  `blue_orient == "h"` 的样本。这样每个样本都能看见“原本不该旋转的矩形也被旋转了”。

c) 隐式假设审查
- 原 `main()` 里关于 corner 定位、蓝红边识别、边线遮罩、连通块提取、距离排序的前提仍成立；
  这些逻辑只依赖输入几何，不依赖输出布局。
- 被 D5/1 破坏的旧假设：
  `if blue_orient == "v": arr = np.rot90(arr, 1)` 不再成立，必须改为无条件旋转。
- 被 D6/3 破坏的旧假设：
  `outW = max(widths), outH = sum(heights), out_r += h, horizontal centering` 不再成立。
  新布局应改为 `outH = max(heights), outW = sum(widths), out_c += w, vertical centering`。
- 叠加后没有冲突，因为“先旋转再排布”仍然产生一串独立矩形，且画布尺寸可以直接由
  旋转后的高宽集合算出。

d) 放置合法性
- 输入生成仍沿用原题 canonical 模板：两条 2-cell 高的矩形带、矩形之间固定 2 格背景间隔。
- 整体翻转只改变全局朝向，不会引入重叠或越界。
- 输出画布按“旋转后序列的最大高度 + 宽度总和”精确开辟，因此横向粘贴不会重叠。

e) 视觉可观察性
- 对人类而言，最醒目的现象是：输出从“高而窄的竖栈”变成“矮而长的横条”。
- 其次，因为生成器只保留蓝线水平且含非正方形矩形的样本，人能直接看出这些矩形被转向了。
- 这两条线索在 2-4 组样例中都足够稳定，不依赖颜色编号记忆。

f) 可视化整洁度
- 输入区域仍然稀疏，角点、边线、矩形带彼此分离，没有新叠加标记。
- 输出只使用一个简单对齐规则：水平紧贴、垂直居中，因此可读性比额外加间距或分组更强。

g) 颜色变化可推断性
- 本融合不引入新的颜色映射。颜色仍只标识矩形本身。
- 少量 minority 像素保留原题风格，只作为局部不对称，帮助观察旋转，而不承担新语义。

h) 新 solver 的隐式假设与生成器保障
- 新 solver 假设每个提取对象都能当作完整矩形独立旋转；生成器继续只放置无孔矩形。
- 新 solver 假设水平排布至少有一个对象；生成器始终构造 4-9 个矩形。
- 为了保证融合行为真正发生，生成器额外筛掉两类退化样本：
  1. `main(grid)` 与“仅改水平排布、不改旋转规则”的输出相同；
  2. `main(grid)` 与原题输出相同。
  这样每个样本都同时体现 D5/1 与 D6/3。
"""


def _locate_corner(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    height, width = grid.shape
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    for row, col in corners:
        if int(grid[row, col]) == int(bg):
            continue
        neighbors = []
        if row == 0:
            neighbors.append(int(grid[row + 1, col]))
        if row == height - 1:
            neighbors.append(int(grid[row - 1, col]))
        if col == 0:
            neighbors.append(int(grid[row, col + 1]))
        if col == width - 1:
            neighbors.append(int(grid[row, col - 1]))
        if neighbors and all(value != int(grid[row, col]) for value in neighbors):
            return int(row), int(col), int(grid[row, col])
    raise AssertionError("corner marker not found")


def _line_orientations(grid: np.ndarray, corner_row: int, corner_col: int) -> tuple[str, str]:
    height, width = grid.shape
    hstep = 1 if corner_col == 0 else -1
    vstep = 1 if corner_row == 0 else -1
    hcol = int(grid[corner_row, corner_col + hstep])
    vcol = int(grid[corner_row + vstep, corner_col])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
    else:
        hlen = 0
        col = int(corner_col + hstep)
        while 0 <= col < width and int(grid[corner_row, col]) == int(hcol):
            hlen += 1
            col += int(hstep)

        vlen = 0
        row = int(corner_row + vstep)
        while 0 <= row < height and int(grid[row, corner_col]) == int(vcol):
            vlen += 1
            row += int(vstep)

        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return blue_orient, red_orient


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
    row0, col0, row1, col1 = map(int, rect["box"])
    if orient == "v":
        if col0 > edge_idx:
            return int(col0 - edge_idx - 1)
        if col1 < edge_idx:
            return int(edge_idx - col1 - 1)
        return 0
    if row0 > edge_idx:
        return int(row0 - edge_idx - 1)
    if row1 < edge_idx:
        return int(edge_idx - row1 - 1)
    return 0


def _ordered_sequence(grid: np.ndarray, *, always_rotate: bool) -> tuple[int, str, list[np.ndarray]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    corner_row, corner_col, marker = _locate_corner(g, bg)
    blue_orient, red_orient = _line_orientations(g, corner_row, corner_col)

    line_mask = np.zeros(g.shape, dtype=bool)
    line_mask[corner_row, :] = True
    line_mask[:, corner_col] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects

    blue_edge_idx = int(corner_col if blue_orient == "v" else corner_row)
    red_edge_idx = int(corner_col if red_orient == "v" else corner_row)

    def along_from_corner(rect: dict, orient: str) -> int:
        row0, col0, row1, col1 = map(int, rect["box"])
        if orient == "v":
            return int(row0 if corner_row == 0 else -row1)
        return int(col0 if corner_col == 0 else -col1)

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )

    sequence = []
    for rect in rects:
        arr = rect["arr"]
        if always_rotate or blue_orient == "v":
            arr = np.rot90(arr, k=1)
        sequence.append(arr)

    return int(bg), str(blue_orient), sequence


def _assemble(sequence: list[np.ndarray], bg: int, *, horizontal: bool) -> np.ndarray:
    assert sequence
    if horizontal:
        out_height = int(max(int(arr.shape[0]) for arr in sequence))
        out_width = int(sum(int(arr.shape[1]) for arr in sequence))
        out = np.full((out_height, out_width), int(bg), dtype=int)
        out_col = 0
        for arr in sequence:
            height, width = map(int, arr.shape)
            out_row = int((out_height - height) // 2)
            paste(out, arr, int(out_row), int(out_col))
            out_col += int(width)
        return out

    out_width = int(max(int(arr.shape[1]) for arr in sequence))
    out_height = int(sum(int(arr.shape[0]) for arr in sequence))
    out = np.full((out_height, out_width), int(bg), dtype=int)
    out_row = 0
    for arr in sequence:
        height, width = map(int, arr.shape)
        out_col = int((out_width - width) // 2)
        paste(out, arr, int(out_row), int(out_col))
        out_row += int(height)
    return out


def _solve(grid: np.ndarray, *, always_rotate: bool, horizontal: bool) -> np.ndarray:
    bg, _blue_orient, sequence = _ordered_sequence(grid, always_rotate=always_rotate)
    return _assemble(sequence, bg, horizontal=horizontal)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort rectangles by the original blue/red edge ordering, rotate every rectangle
    90 degrees CCW, then place them left-to-right with vertical centering.
    """
    return _solve(grid, always_rotate=True, horizontal=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, always_rotate=False, horizontal=False)


def _layout_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, always_rotate=False, horizontal=True)


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _distribute_band_widths(count: int, used_target: int) -> list[int]:
    extra_units = (used_target - (4 * count - 2)) // 2
    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        low = max(0, remaining - 3 * slots_left)
        high = min(3, remaining)
        take = int(np.random.randint(low, high + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(unit) for unit in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(color) for color in MINORITY_COLORS if int(color) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    while True:
        long_dim = int(np.random.randint(9, 29))
        options = []
        for trailing in (1, 2):
            max_used = long_dim - 2 - trailing
            if max_used < 6 or max_used % 2:
                continue
            max_count = min(5, (max_used + 2) // 4)
            pairs = []
            for count0 in range(2, max_count + 1):
                for count1 in range(2, max_count + 1):
                    if count0 + count1 > 9:
                        continue
                    eligible = []
                    if max_used <= 10 * count0 - 2:
                        eligible.append(0)
                    if max_used <= 10 * count1 - 2:
                        eligible.append(1)
                    if eligible:
                        pairs.append((count0, count1, eligible))
            if pairs:
                options.append((max_used, pairs))

        max_used, pairs = options[int(np.random.randint(len(options)))]
        count0, count1, eligible = pairs[int(np.random.randint(len(pairs)))]
        max_band = int(np.random.choice(eligible))
        max0 = min(max_used, 10 * count0 - 2)
        max1 = min(max_used, 10 * count1 - 2)
        used0 = int(
            max_used if max_band == 0 else np.random.choice(np.arange(4 * count0 - 2, max0 + 1, 2))
        )
        used1 = int(
            max_used if max_band == 1 else np.random.choice(np.arange(4 * count1 - 2, max1 + 1, 2))
        )
        band0 = _distribute_band_widths(count0, used0)
        band1 = _distribute_band_widths(count1, used1)
        if max(band0 + band1) <= 2:
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for row0, widths in ((2, band0), (5, band1)):
            col = 2
            for width in widths:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((2, width), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, row0, col)
                col += int(width) + 2

        return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient="h", corner=corner)


def _combo_metrics(grid: np.ndarray) -> dict[str, object]:
    combo = main(grid)
    legacy = _legacy_main(grid)
    layout_only = _layout_only_main(grid)
    bg, blue_orient, combo_seq = _ordered_sequence(grid, always_rotate=True)
    _bg2, _blue2, legacy_seq = _ordered_sequence(grid, always_rotate=False)
    horizontal = _assemble(combo_seq, bg, horizontal=True)
    vertical = _assemble(combo_seq, bg, horizontal=False)
    return {
        "blue_orient": blue_orient,
        "rect_count": len(combo_seq),
        "combo_vs_legacy": not np.array_equal(combo, legacy),
        "combo_vs_layout_only": not np.array_equal(combo, layout_only),
        "horizontal_vs_vertical": horizontal.shape != vertical.shape
        or not np.array_equal(horizontal, vertical),
        "has_tall_after_rotation": any(int(arr.shape[0]) > int(arr.shape[1]) for arr in combo_seq),
        "layout_only_seq_same_as_combo": all(
            arr0.shape == arr1.shape and np.array_equal(arr0, arr1)
            for arr0, arr1 in zip(combo_seq, legacy_seq)
        ),
    }


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        try:
            metrics = _combo_metrics(grid)
        except Exception:
            continue
        if (
            metrics["blue_orient"] == "h"
            and metrics["rect_count"] >= 4
            and metrics["combo_vs_legacy"]
            and metrics["combo_vs_layout_only"]
            and metrics["horizontal_vs_vertical"]
            and metrics["has_tall_after_rotation"]
        ):
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate input for combo variant 291dc1e1/13")

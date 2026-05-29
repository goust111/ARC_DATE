# Combo variant of 291dc1e1 (index 26)
# combo_spec: D1/3+D9/1
# Name: 距离逆序+十字交叉线

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for this combo variant
# a) 具体化与行为追踪
#    - 原题输入是“角落 marker + 角落横竖边线 + 内部两排矩形”，矩形按 (到蓝线, 到红线) 升序输出。
#    - 本融合版把横竖线整体移到网格中部，交成十字；交点仍保留独立 marker，矩形改为分散在四个象限。
#    - 同时把排序主键反转为降序，所以现在先输出离蓝线最远的矩形；若蓝距相同，再看谁离红线更远。
#    - 例子：若蓝线在中部第 7 行，某上方矩形在第 1-2 行则 db=4，另一个在第 4-5 行则 db=1。
#      原题会先放 db=1 的近矩形；本题会先放 db=4 的远矩形。十字在中部后，db/dr 都是“到中轴线”的间隔。
# b) 复杂度校准
#    - 保留原题的核心对象类型：1 条蓝线、1 条红线、1 个 marker、若干实心矩形。
#    - 为避免退化，生成器强制四个象限都出现矩形，并强制 near/far 两档蓝距都出现。
#    - 每个象限放 1-2 个矩形，总数 4-8，既足够形成排序结构，又不会让画面拥挤难辨。
# c) 隐式假设审查
#    - 原 solver 假设 marker 在角落，并把 corner row/col 整条当线；这里已失效，因此必须改成全局扫描整行/整列找十字。
#    - 原 solver 的 `along_from_corner` 依赖角落方向；这里没有角落，所以 tie-break 改成按轴向坐标的自然顺序。
#    - 原 solver 仍假设矩形 bbox 内无背景孔洞；生成器继续保证这一点。
#    - D1/3 只反转主排序键，不能用 `reverse=True`，否则会把 tie-break 一起反转，造成额外维度变化。
# d) 放置合法性
#    - canonical 布局固定蓝线在中行、红线在中列，矩形都与两条线至少隔 1 格。
#    - 每个象限内部的多个矩形只沿远离红线的方向排开，并保留 2 格背景间隔，不重叠、不越界。
#    - 输出仍是把裁出的矩形按顺序竖直堆叠，宽度取最大值，因此总是合法。
# e) 视觉可观察性
#    - 十字线从角落移到中部，输入外观与原题明显不同；人眼能直接看到四象限结构。
#    - 生成器显式拒绝“降序输出与升序输出相同”的样本，保证距离逆序这个控制变量真的可见。
#    - 通过给矩形加入少量异色点和不同宽度，输出中的顺序交换也容易观察。
# f) 可视化整洁度
#    - 十字线和 marker 都是整齐的单像素线，不和矩形叠压。
#    - 背景色统一，矩形只在四个象限内，且每个象限只做一维排布，因此画面规整。
# g) 颜色变化可推断性
#    - 颜色角色保持简单稳定：蓝线=1，红线=2，marker=0，矩形主体从固定调色板选。
#    - 少量 minority 像素只用于区分矩形身份，不承担任何新规则。
# h) source_task
#    - 该组合直接基于原题和 D1/3 单变体推导，没有额外 source_task 需要适配。
# i) 新 solver 的额外假设审查
#    - 新 solver 假设存在且仅存在 1 条“W-1 同色 + 1 个 marker”的横线，和 1 条“H-1 同色 + 1 个 marker”的竖线。
#      生成器固定构造这一十字并把它放在中部。
#    - 新 solver 假设每个矩形完全位于某个象限，不跨越中轴线；生成器通过至少 1 格留白保证。
#    - 新 solver 的 tie-break 采用轴向自然顺序；生成器不依赖它来制造关键差异，关键差异由 db/dr 逆序保证。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6, 8], dtype=int)


def _find_cross_axes(g: np.ndarray, bg: int) -> dict:
    h, w = map(int, g.shape)
    row_cands = []
    col_cands = []

    for r in range(h):
        counts = Counter(map(int, g[r, :]))
        if len(counts) != 2:
            continue
        line_col, line_count = counts.most_common(1)[0]
        if int(line_count) != w - 1:
            continue
        marker_cols = np.where(g[r, :] != int(line_col))[0]
        if marker_cols.size != 1:
            continue
        mc = int(marker_cols[0])
        marker = int(g[r, mc])
        if marker == int(line_col) or marker == int(bg):
            continue
        row_cands.append({"idx": int(r), "line_col": int(line_col), "marker_col": mc, "marker": marker})

    for c in range(w):
        counts = Counter(map(int, g[:, c]))
        if len(counts) != 2:
            continue
        line_col, line_count = counts.most_common(1)[0]
        if int(line_count) != h - 1:
            continue
        marker_rows = np.where(g[:, c] != int(line_col))[0]
        if marker_rows.size != 1:
            continue
        mr = int(marker_rows[0])
        marker = int(g[mr, c])
        if marker == int(line_col) or marker == int(bg):
            continue
        col_cands.append({"idx": int(c), "line_col": int(line_col), "marker_row": mr, "marker": marker})

    matches = []
    for row in row_cands:
        for col in col_cands:
            if row["marker_col"] != col["idx"]:
                continue
            if col["marker_row"] != row["idx"]:
                continue
            if row["marker"] != col["marker"]:
                continue
            if row["line_col"] == col["line_col"]:
                continue
            matches.append((row, col))

    assert len(matches) == 1
    row, col = matches[0]

    if row["line_col"] == BLUE or col["line_col"] == BLUE:
        blue_col = BLUE
        red_col = int(col["line_col"] if row["line_col"] == BLUE else row["line_col"])
    else:
        blue_col = int(row["line_col"] if w >= h else col["line_col"])
        red_col = int(col["line_col"] if blue_col == row["line_col"] else row["line_col"])

    blue_orient = "h" if int(row["line_col"]) == int(blue_col) else "v"
    red_orient = "v" if blue_orient == "h" else "h"

    return {
        "row_idx": int(row["idx"]),
        "col_idx": int(col["idx"]),
        "marker": int(row["marker"]),
        "h_col": int(row["line_col"]),
        "v_col": int(col["line_col"]),
        "blue_col": int(blue_col),
        "red_col": int(red_col),
        "blue_orient": blue_orient,
        "red_orient": red_orient,
    }


def _parse_task(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = int(get_bg(g))
    axes = _find_cross_axes(g, bg)

    row_idx = int(axes["row_idx"])
    col_idx = int(axes["col_idx"])
    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[row_idx, :] = True
    line_mask[:, col_idx] = True

    mask = (g != int(bg)) & (~line_mask)
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy()})

    assert rects
    return {
        "grid": g,
        "bg": bg,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "blue_orient": str(axes["blue_orient"]),
        "red_orient": str(axes["red_orient"]),
        "rects": rects,
    }


def _dist_to_axis(rect: dict, orient: str, axis_idx: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        col = int(axis_idx)
        if int(c0) > col:
            return int(c0 - col - 1)
        if int(c1) < col:
            return int(col - c1 - 1)
        return 0
    row = int(axis_idx)
    if int(r0) > row:
        return int(r0 - row - 1)
    if int(r1) < row:
        return int(row - r1 - 1)
    return 0


def _axis_anchor(rect: dict, orient: str) -> int:
    r0, c0, _r1, _c1 = map(int, rect["box"])
    return int(c0 if orient == "h" else r0)


def _solve(grid: np.ndarray, *, descending: bool) -> np.ndarray:
    info = _parse_task(grid)
    bg = int(info["bg"])
    blue_orient = str(info["blue_orient"])
    red_orient = str(info["red_orient"])
    rects = [dict(rct) for rct in info["rects"]]

    blue_axis = int(info["col_idx"] if blue_orient == "v" else info["row_idx"])
    red_axis = int(info["col_idx"] if red_orient == "v" else info["row_idx"])

    for rect in rects:
        rect["db"] = int(_dist_to_axis(rect, blue_orient, blue_axis))
        rect["dr"] = int(_dist_to_axis(rect, red_orient, red_axis))

    if descending:
        rects.sort(
            key=lambda rect: (
                -int(rect["db"]),
                -int(rect["dr"]),
                _axis_anchor(rect, blue_orient),
                _axis_anchor(rect, red_orient),
            )
        )
    else:
        rects.sort(
            key=lambda rect: (
                int(rect["db"]),
                int(rect["dr"]),
                _axis_anchor(rect, blue_orient),
                _axis_anchor(rect, red_orient),
            )
        )

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        ah, aw = map(int, arr.shape)
        out_c = int((out_w - aw) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(ah)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles around a centered cross of blue/red lines.

    Combo changes:
    - Detect a centered cross instead of corner-aligned edge lines.
    - Compute distances to the middle blue/red axes.
    - Sort by (distance to blue, distance to red) in descending order.
    - If the blue line is vertical, rotate each rectangle 90 degrees CCW.
    - Stack the rectangles top-to-bottom, centered horizontally.
    """
    return _solve(grid, descending=True)


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.55:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _quadrant_extent(widths: list[int]) -> int:
    return 1 + int(sum(int(w) for w in widths)) + 2 * (len(widths) - 1)


def _choose_widths() -> list[int]:
    count = 1 + int(np.random.rand() < 0.6)
    widths = [int(np.random.choice(WIDTH_CHOICES)) for _ in range(count)]
    return widths


def _make_patch(width: int, color: int, used_signatures: set[tuple[int, ...]]) -> np.ndarray:
    patch = np.full((2, int(width)), int(color), dtype=int)
    patch = _sprinkle_minorities(patch, int(color))
    signature = tuple(int(v) for v in patch.reshape(-1))
    if signature not in used_signatures:
        used_signatures.add(signature)
        return patch

    choices = [int(c) for c in MINORITY_COLORS if int(c) != int(color)]
    for _ in range(8):
        patch = np.full((2, int(width)), int(color), dtype=int)
        patch = _sprinkle_minorities(patch, int(color))
        patch[0, -1] = int(np.random.choice(choices))
        signature = tuple(int(v) for v in patch.reshape(-1))
        if signature not in used_signatures:
            used_signatures.add(signature)
            return patch

    used_signatures.add(signature)
    return patch


def _place_side(grid: np.ndarray, row: int, center_col: int, side: str, widths: list[int], patch_bank: list[np.ndarray]) -> None:
    if side == "left":
        next_right = int(center_col - 2)
        for patch in widths:
            arr = patch_bank.pop(0)
            ph, pw = map(int, arr.shape)
            c0 = int(next_right - pw + 1)
            paste(grid, arr, int(row), int(c0))
            next_right = int(c0 - 3)
        return

    next_left = int(center_col + 2)
    for _ in widths:
        arr = patch_bank.pop(0)
        _ph, pw = map(int, arr.shape)
        paste(grid, arr, int(row), int(next_left))
        next_left += int(pw + 2)


def _build_canonical() -> np.ndarray:
    row_by_quad = {"tl": {4: 1, 1: 4}, "tr": {4: 1, 1: 4}, "bl": {4: 12, 1: 9}, "br": {4: 12, 1: 9}}

    while True:
        far_quads = set(np.random.choice(["tl", "tr", "bl", "br"], size=2, replace=False).tolist())
        quadrants = {}
        total_rects = 0
        for name in ("tl", "tr", "bl", "br"):
            widths = _choose_widths()
            quadrants[name] = {
                "db": 4 if name in far_quads else 1,
                "widths": widths,
                "side": "left" if name.endswith("l") else "right",
                "row": int(row_by_quad[name][4 if name in far_quads else 1]),
            }
            total_rects += len(widths)

        left_used = max(_quadrant_extent(quadrants["tl"]["widths"]), _quadrant_extent(quadrants["bl"]["widths"]))
        right_used = max(_quadrant_extent(quadrants["tr"]["widths"]), _quadrant_extent(quadrants["br"]["widths"]))
        width = 1 + int(left_used) + 1 + int(right_used) + 1
        if 15 <= int(width) <= 29 and total_rects >= 5:
            break

    height = 15
    center_row = 7
    center_col = 1 + int(left_used)
    grid = np.full((height, int(width)), BG, dtype=int)
    grid[center_row, :] = BLUE
    grid[:, center_col] = RED
    grid[center_row, center_col] = MARKER

    used_signatures: set[tuple[int, ...]] = set()
    color_pool = np.random.permutation(DOMINANT_COLORS)
    color_idx = 0

    for name in ("tl", "tr", "bl", "br"):
        widths = quadrants[name]["widths"]
        patches = []
        for width_i in widths:
            if color_idx < len(color_pool):
                color = int(color_pool[color_idx])
                color_idx += 1
            else:
                color = int(np.random.choice(DOMINANT_COLORS))
            patches.append(_make_patch(int(width_i), int(color), used_signatures))
        _place_side(grid, int(quadrants[name]["row"]), int(center_col), str(quadrants[name]["side"]), widths, patches)

    return grid


def _random_transform(grid: np.ndarray, blue_orient: str) -> np.ndarray:
    out = np.array(grid, copy=True)
    if blue_orient == "v":
        out = np.rot90(out, k=1)
    if np.random.rand() < 0.5:
        out = np.flipud(out)
    if np.random.rand() < 0.5:
        out = np.fliplr(out)
    return out


def _quadrant_occupancy(grid: np.ndarray) -> tuple[int, bool]:
    info = _parse_task(grid)
    row_idx = int(info["row_idx"])
    col_idx = int(info["col_idx"])
    seen = set()
    for rect in info["rects"]:
        r0, c0, r1, c1 = map(int, rect["box"])
        rv = "t" if int(r1) < row_idx else "b"
        cv = "l" if int(c1) < col_idx else "r"
        seen.add(rv + cv)
    centered = 1 < row_idx < grid.shape[0] - 2 and 1 < col_idx < grid.shape[1] - 2
    return len(seen), centered


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    grid = _random_transform(canonical, blue_orient)

    try:
        new_out = _solve(grid, descending=True)
        old_out = _solve(grid, descending=False)
        n_quads, centered = _quadrant_occupancy(grid)
    except Exception:
        return None

    if n_quads != 4 or not centered:
        return None
    if np.array_equal(new_out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo D1/3+D9/1")

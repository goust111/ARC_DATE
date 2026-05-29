# Variant of 291dc1e1
# Dimension: D5 — 条件旋转规则 (transform)
# Change: 若蓝线为垂直方向，所有矩形做90度逆时针旋转；蓝线水平时不旋转... → 不旋转（无论蓝线方向如何）
#
# Design analysis
# - Source-task check: this variant has no `source_task`, so there is no extra solver to inspect.
# - a) 具体化与行为追踪:
#   Input 仍然是“角点标记 + 一横一竖两条边线 + 内部若干矩形”。排序规则完全不变，变化只发生在
#   “把排序后的矩形依次堆叠输出”之前。原任务里，若蓝线竖直，矩形 `[[a a a b],[a a a a]]`
#   会先逆时针旋转成 4x2 再堆叠；本变体中它保持 2x4 原样堆叠。也就是说，输入构造、矩形抽取、
#   距离排序都沿用原逻辑，只有每个矩形的最终朝向不再由蓝线方向控制。
# - b) 复杂度校准:
#   若所有矩形都是纯色方块，去掉旋转后肉眼难以观察差异，所以 generator 继续使用大多为 2xw 的
#   非方形矩形，并在部分矩形内部放少量异色像素，让“是否旋转”在 2-4 个样例中明显可见。
#   同时不引入新对象类别或新颜色规则，避免复杂度偏离原题。
# - c) 隐式假设审查:
#   原 `main()` 中真正依赖旧 D5 的只有最终 `if blue_orient == "v": np.rot90(...)` 这一步。
#   角点定位、蓝/红线识别、line mask、solid bbox 检查、距离计算和排序都不依赖矩形会被旋转，
#   所以前提仍然成立。需要避免的唯一错误是保留旧旋转分支，否则会直接违背变体定义。
# - d) 放置合法性:
#   generator 仍从规范姿态（蓝线上、红线左、角点左上）构造，再整体翻转/旋转到其他角落。
#   内部矩形只放在两条离边线至少 1 格的带状区域，矩形之间用 2 格背景隔开，因此不会重叠、
#   不会越界，也不会与边线粘连。整体刚体变换不会破坏这些性质。
# - e) 视觉可观察性:
#   为了让“永不旋转”可见，generator 强化两个条件：
#   1. 偏向生成蓝线竖直的输入，因为只有这类样例和原题不同；
#   2. 拒绝蓝线竖直但新旧输出恰好相同的退化样本。
# - f) 可视化整洁度:
#   背景固定为 8，边线为 1/2，角点为 0，矩形主体使用其余颜色；少量异色像素只出现在矩形内部，
#   不引入额外叠层，视觉上仍能清楚分辨边线、角点和各矩形。
# - g) 颜色变化可推断性:
#   这里没有新的颜色映射规则。颜色仍只用于区分矩形及其内部不对称纹理，不承担额外条件编码，
#   因此不会给解题者增加无结构的颜色负担。
# - h) 变体量化指标:
#   生成后可用两个指标检查是否真的发生 D5 变化：
#   1. `blue_orient == "v"` 的样本比例；
#   2. 这些样本中，新 solver 输出与旧旋转版输出不同的比例。
#   若第二项接近 0，说明生成器退化回“看不出变体”的原题外观，需要拒绝重采样。
# - i) 修改后 `main()` 的新隐式假设:
#   新 solver 仍假设：角点恰在四角之一；两条边线占满对应整行/整列；每个对象是 4 连通且其 bbox
#   内没有背景；至少存在一个矩形。`generate_input()` 通过规范构造和整图刚体变换保证这些前提。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, int, str, list[dict]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == bg:
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == h - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == w - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < w and int(g[cr, c]) == hcol:
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == vcol:
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != bg) & (~line_mask) & (g != marker)
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == bg).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})
    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
            return int(edge_idx - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rect in rects:
        rect["db"] = int(dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            along_from_corner(rect, blue_orient),
            along_from_corner(rect, red_orient),
        )
    )
    return g, bg, blue_orient, rects


def _stack_rectangles(rects: list[dict], bg: int, *, rotate_if_blue_vertical: bool, blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if rotate_if_blue_vertical and blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    _, bg, blue_orient, rects = _parse_scene(grid)
    return _stack_rectangles(rects, bg, rotate_if_blue_vertical=True, blue_orient=blue_orient)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two colored edge lines.

    Variant D5 change:
    - Sorting logic is unchanged.
    - Rectangles are never rotated before stacking, even when the blue edge is vertical.
    """
    _, bg, blue_orient, rects = _parse_scene(grid)
    return _stack_rectangles(rects, bg, rotate_if_blue_vertical=False, blue_orient=blue_orient)


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
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 4:
        return patch

    colors = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    flat = patch.reshape(-1)

    # Place an off-center minority pixel in most non-square rectangles so rotation differences stay visible.
    if patch.shape[1] > 2 and np.random.rand() < 0.85:
        flat[1] = int(np.random.choice(colors))

    if area >= 8 and np.random.rand() < 0.45:
        flat[-2] = int(np.random.choice(colors))

    if area >= 12 and np.random.rand() < 0.25:
        flat[int(patch.shape[1])] = int(np.random.choice(colors))
    return patch


def _build_canonical() -> np.ndarray:
    for _ in range(400):
        long_dim = int(np.random.randint(11, 29))
        options = []
        for trailing in (1, 2):
            max_used = long_dim - 2 - trailing
            if max_used < 8 or max_used % 2:
                continue
            max_count = min(5, (max_used + 2) // 4)
            pairs = []
            for a in range(2, max_count + 1):
                for b in range(2, max_count + 1):
                    if a + b > 9:
                        continue
                    eligible = []
                    if max_used <= 10 * a - 2:
                        eligible.append(0)
                    if max_used <= 10 * b - 2:
                        eligible.append(1)
                    if eligible:
                        pairs.append((a, b, eligible))
            if pairs:
                options.append((max_used, pairs))
        if not options:
            continue

        max_used, pairs = options[int(np.random.randint(len(options)))]
        n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
        max_band = int(np.random.choice(eligible))
        max0 = min(max_used, 10 * n0 - 2)
        max1 = min(max_used, 10 * n1 - 2)
        used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
        used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
        band0 = _distribute_band_widths(n0, used0)
        band1 = _distribute_band_widths(n1, used1)
        widths = band0 + band1
        if sum(int(w > 2) for w in widths) < 2:
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, band in ((2, band0), (5, band1)):
            c = 2
            for w in band:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((2, w), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, r0, c)
                c += int(w) + 2
        return grid
    raise RuntimeError("failed to build canonical input")


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = "v" if np.random.rand() < 0.75 else "h"
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    # Validate solver assumptions and reject vertical cases that visually collapse to the legacy behavior.
    out = main(grid)
    _, _, parsed_blue_orient, rects = _parse_scene(grid)
    if parsed_blue_orient == "v":
        if np.array_equal(out, _legacy_main(grid)):
            return None
        if not any(tuple(map(int, rect["arr"].shape)) != (2, 2) for rect in rects):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D5 variant")

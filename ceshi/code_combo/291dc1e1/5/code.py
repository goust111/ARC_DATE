# Combo variant of 291dc1e1 (index 5)
# combo_spec: D1/2+D11/1
# Name: 色值排序+渐变重着色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) 具体化与行为追踪:
#    原题输入仍然是一个角点 marker + 两条边线 + 内部若干实心矩形的布局；输出仍然是把矩形竖直堆叠，
#    且当蓝线竖直时先对每个矩形做 90 度逆时针旋转。融合后的两处核心变化必须协同发生：
#    1. 排序不再读取“到蓝线/红线的距离”，而是读取每个矩形的主体颜色在固定色序中的名次。
#       本实现采用固定色序 [9, 7, 6, 5, 4, 3]，即 maroon -> orange -> pink -> grey -> yellow -> green。
#    2. 输出不再保留输入颜色，而是按排序后的位置改涂成渐变色 [3, 4, 5, 6, 7, 9]。
#    小例子：若四个矩形宽度分别为 [8, 2, 6, 4]，主体色分别是 [4, 9, 5, 7]，几何上它们可能分散在两条带里。
#    原题会按离边线的距离排序并保留各自颜色；本变体先按主体色序得到 [9, 7, 5, 4] 这四块，再按名次把它们
#    重涂为 [3, 4, 5, 6]，最后堆叠为对应宽度顺序的四段。差异不是“颜色单独变了”或“排序单独变了”，而是
#    输入颜色负责决定次序，输出颜色再编码这个次序。
# b) 复杂度校准:
#    若矩形过多，固定渐变色和形状区分都会变得拥挤；若矩形宽度重复，排序变化可能被同形块抵消。
#    因此 generator 固定生成 4 个矩形，并使用 4 个互不相同的宽度 [2, 4, 6, 8] 的随机排列。
#    这样解题者能用“宽度指纹”追踪每个输入矩形在输出中的位置，同时 4 档渐变色会在每个样例里重复支撑。
# c) 隐式假设审查:
#    - 原 corner/edge 识别逻辑只依赖角点和边线颜色，融合后仍成立；蓝线朝向依旧只负责决定是否旋转。
#    - 原排序代码依赖 `dist_to_edge` 和 `along_from_corner`。这些前提在融合后失效，因为排序不再参考几何距离。
#      新 solver 只依赖每个矩形存在一个明确主体色，并且主体色落在固定色序里。
#    - 原 `sub = crop_bbox(...)` + `if (sub == bg).any(): continue` 假设对象是无洞实心矩形；融合后仍成立。
#    - 新增的重着色步骤要求矩形数不超过渐变色数；generator 固定 4 块，满足此约束。
# d) 放置合法性:
#    仍沿用原题的两条水平带构造：canonical 形态中矩形只放在第 2/5 行带上，每带两个，彼此间隔 2 个背景格，
#    与两条边线至少隔 1 格。随后只做翻转或 90 度旋转到四个角，不会引入重叠或越界。
# e) 视觉可观察性:
#    观察者会看到：输入里矩形的宽度各不相同、主体色各不相同；输出里相同宽度的块被重新排序，同时颜色不再
#    跟随输入而是总按同一梯度变化。两组训练样例就能排除“按空间位置排”和“直接复制颜色”这两种旧解释。
# f) 可视化整洁度:
#    输入仍只有 L 形边线、统一背景和两排稀疏矩形。少量 minority 像素只放在矩形内部，既能强调“主体色”
#    和“输出会覆盖输入颜色”，又不会破坏矩形轮廓。
# g) 颜色变化可推断性:
#    排序色序和输出渐变都限制在 6 色子集内，但每个样例只使用其中 4 个排序色和前 4 个渐变色。颜色类别少，
#    且输出总是 rank1->3, rank2->4, rank3->5, rank4->6；人类可从多组样例归纳“输入颜色决定次序，输出颜色
#    只表示名次”。
# h) 变体量化指标:
#    generator 对每个候选样本同时计算：
#    - `sort_disagreement`: 颜色排序序列与原题几何排序序列在同一输入上有多少个位置不同；
#    - `recolor_mismatch_count`: 颜色排序后的第 i 个矩形，其输入主体色与第 i 个渐变输出色有多少处不同。
#    只有当 `sort_disagreement >= 2` 且 `recolor_mismatch_count == 4` 时样本才被接受。
# i) main() 的新隐式假设:
#    - 每个矩形必须存在唯一主体色；minority 像素数量因此必须严格少于主体色像素数。
#    - 生成样本里 4 个矩形的主体色必须互异，否则“按颜色排序”会出现并列。solver 用 `(色序, r0, c0)` 作为
#      兜底键，但 generator 会保证不触发这个并列。
#    - 若矩形主体色不在固定色序中，solver 无法解释输入；generator 保证只使用允许的排序色。


BG = 8
BLUE = 1
RED = 2
MARKER = 0
SORT_COLOR_ORDER = np.array([9, 7, 6, 5, 4, 3], dtype=int)
RANK_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6, 8], dtype=int)
SORT_INDEX = {int(color): idx for idx, color in enumerate(SORT_COLOR_ORDER.tolist())}


def _locate_layout(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    h, w = map(int, g.shape)
    bg = int(get_bg(g))

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == bg:
            continue
        neighbors = []
        if r == 0:
            neighbors.append(int(g[r + 1, c]))
        if r == h - 1:
            neighbors.append(int(g[r - 1, c]))
        if c == 0:
            neighbors.append(int(g[r, c + 1]))
        if c == w - 1:
            neighbors.append(int(g[r, c - 1]))
        if neighbors and all(int(n) != int(g[r, c]) for n in neighbors):
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
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


def _extract_rects(g: np.ndarray, bg: int, corner: tuple[int, int], marker: int) -> list[dict]:
    h, w = map(int, g.shape)
    cr, cc = map(int, corner)

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": np.array(arr, copy=True)})

    assert rects
    return rects


def _dominant_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.array(arr, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, corner: tuple[int, int]) -> int:
    cr, cc = map(int, corner)
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _legacy_sort_key(rect: dict, blue_orient: str, red_orient: str, corner: tuple[int, int]) -> tuple[int, int, int, int]:
    cr, cc = map(int, corner)
    box = tuple(map(int, rect["box"]))
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    return (
        int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
        int(_dist_to_edge(box, red_orient, red_edge_idx)),
        int(_along_from_corner(box, blue_orient, corner)),
        int(_along_from_corner(box, red_orient, corner)),
    )


def _color_sort_key(rect: dict) -> tuple[int, int, int]:
    dominant = int(_dominant_color(rect["arr"]))
    assert dominant in SORT_INDEX
    r0, c0, _r1, _c1 = map(int, rect["box"])
    return int(SORT_INDEX[dominant]), int(r0), int(c0)


def _rotate_if_needed(arr: np.ndarray, blue_orient: str) -> np.ndarray:
    if blue_orient == "v":
        return np.rot90(np.array(arr, copy=True), k=1)
    return np.array(arr, copy=True)


def _stack_arrays(arrays: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in arrays))
    out_h = int(sum(int(a.shape[0]) for a in arrays))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in arrays:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _sorted_rects_by_color(g: np.ndarray) -> tuple[int, str, list[dict]]:
    bg, corner, marker, blue_orient, _red_orient = _locate_layout(g)
    rects = _extract_rects(g, bg, corner, marker)
    rects = sorted(rects, key=_color_sort_key)
    return int(bg), blue_orient, rects


def _render_rank_recolored(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    assert len(rects) <= len(RANK_COLORS)
    arrays = []
    for idx, rect in enumerate(rects):
        arr = _rotate_if_needed(rect["arr"], blue_orient)
        arrays.append(np.full(arr.shape, int(RANK_COLORS[idx]), dtype=int))
    return _stack_arrays(arrays, bg)


def _render_preserved(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    arrays = [_rotate_if_needed(rect["arr"], blue_orient) for rect in rects]
    return _stack_arrays(arrays, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort rectangles by dominant input color and recolor them by output rank.

    The spatial extraction and optional 90-degree CCW rotation are inherited
    from the source task. The geometric distance sort is replaced by a fixed
    color-order sort, and the final pasted rectangles are overwritten by a
    rank-color gradient.
    """
    g = np.array(grid, dtype=int)
    bg, blue_orient, rects = _sorted_rects_by_color(g)
    return _render_rank_recolored(rects, bg, blue_orient)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return np.array(grid, copy=True)
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


def _sprinkle_minorities(patch: np.ndarray, dominant: int, forbidden: set[int]) -> np.ndarray:
    out = np.array(patch, copy=True)
    area = int(out.size)
    max_minor = min(2, max(0, area // 4))
    if max_minor == 0 or np.random.rand() < 0.35:
        return out

    n_minor = int(np.random.randint(1, max_minor + 1))
    choices = [int(c) for c in MINORITY_COLORS if int(c) not in forbidden and int(c) != int(dominant)]
    if not choices:
        return out

    flat = out.reshape(-1)
    cells = np.random.choice(area, size=n_minor, replace=False)
    colors = np.random.choice(choices, size=n_minor, replace=True)
    for cell, color in zip(cells, colors):
        flat[int(cell)] = int(color)
    return out


def _band_used(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * (len(widths) - 1))


def _choose_color_subset() -> list[int]:
    for _ in range(60):
        colors = [int(c) for c in np.random.choice(SORT_COLOR_ORDER, size=4, replace=False)]
        ordered = sorted(colors, key=lambda color: int(SORT_INDEX[int(color)]))
        mismatch_count = sum(int(color != int(RANK_COLORS[idx])) for idx, color in enumerate(ordered))
        if mismatch_count == 4:
            return colors
    raise RuntimeError("failed to choose non-degenerate color subset")


def _build_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)
    return _sprinkle_minorities(patch, int(dominant), forbidden={BG, MARKER})


def _build_canonical() -> np.ndarray:
    widths = [int(w) for w in np.random.permutation(WIDTH_CHOICES)]
    colors = _choose_color_subset()
    np.random.shuffle(colors)

    top_widths = widths[:2]
    bottom_widths = widths[2:]
    top_colors = colors[:2]
    bottom_colors = colors[2:]

    used0 = _band_used(top_widths)
    used1 = _band_used(bottom_widths)
    trailing = int(np.random.randint(1, 4))
    long_dim = int(2 + max(used0, used1) + trailing)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band_widths, band_colors in ((2, top_widths, top_colors), (5, bottom_widths, bottom_colors)):
        c = 2
        for width, dominant in zip(band_widths, band_colors):
            patch = _build_patch(int(width), int(dominant))
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _legacy_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, corner, marker, blue_orient, red_orient = _locate_layout(g)
    rects = _extract_rects(g, bg, corner, marker)
    rects = sorted(rects, key=lambda rect: _legacy_sort_key(rect, blue_orient, red_orient, corner))
    return _render_preserved(rects, bg, blue_orient)


def _legacy_rank_recolored_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, corner, marker, blue_orient, red_orient = _locate_layout(g)
    rects = _extract_rects(g, bg, corner, marker)
    rects = sorted(rects, key=lambda rect: _legacy_sort_key(rect, blue_orient, red_orient, corner))
    return _render_rank_recolored(rects, bg, blue_orient)


def _color_sorted_preserved_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, blue_orient, rects = _sorted_rects_by_color(g)
    return _render_preserved(rects, bg, blue_orient)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int | bool | tuple[int, ...]]:
    g = np.array(grid, dtype=int)
    bg, corner, marker, blue_orient, red_orient = _locate_layout(g)
    rects = _extract_rects(g, bg, corner, marker)

    legacy_rects = sorted(rects, key=lambda rect: _legacy_sort_key(rect, blue_orient, red_orient, corner))
    color_rects = sorted(rects, key=_color_sort_key)

    legacy_ids = tuple(int(_dominant_color(rect["arr"])) for rect in legacy_rects)
    color_ids = tuple(int(_dominant_color(rect["arr"])) for rect in color_rects)
    sort_disagreement = int(sum(int(a != b) for a, b in zip(legacy_ids, color_ids)))
    recolor_mismatch_count = int(
        sum(int(color != int(RANK_COLORS[idx])) for idx, color in enumerate(color_ids))
    )

    actual = main(g)
    legacy_ranked = _legacy_rank_recolored_output(g)
    preserved_color_sorted = _color_sorted_preserved_output(g)

    return {
        "sort_disagreement": int(sort_disagreement),
        "recolor_mismatch_count": int(recolor_mismatch_count),
        "sort_changed": bool(not np.array_equal(actual, legacy_ranked)),
        "recolor_changed": bool(not np.array_equal(actual, preserved_color_sorted)),
        "legacy_order": legacy_ids,
        "color_order": color_ids,
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _fusion_metrics(grid)
    if not metrics["sort_changed"]:
        return None
    if not metrics["recolor_changed"]:
        return None
    if int(metrics["sort_disagreement"]) < 2:
        return None
    if int(metrics["recolor_mismatch_count"]) != 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate non-degenerate combo input for 291dc1e1")

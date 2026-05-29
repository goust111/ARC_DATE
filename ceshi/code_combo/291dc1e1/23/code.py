# Combo variant of 291dc1e1 (index 23)
# combo_spec: D13/0+D14/2
# Name: 分色多输出+面积去重过滤

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题先找角点与两条边线，再抽出内部矩形；所有矩形统一按
#   `(到蓝线距离, 到红线距离, 沿蓝线离角点先后, 沿红线离角点先后)` 排序，
#   若蓝线竖直则先把每个矩形逆时针转 90 度，最后堆成单列。
# - 本融合版把“谁能进入输出”与“如何组装输出”拆成两步：
#   1. 先按 bbox 面积频次过滤，只保留面积至少出现两次的矩形；
#   2. 再按主色分组，把通过过滤的矩形按原排序的相对顺序分别堆成多列。
# - 小例子：全局排序序列若是
#   `[2x2绿, 2x4黄, 2x6绿(unique), 2x2黄, 2x4绿]`，
#   面积为 4/8/12/4/8。过滤后剩 `[2x2绿, 2x4黄, 2x2黄, 2x4绿]`；
#   再按主色分组，输出变成“绿列: [2x2绿, 2x4绿] + 黄列: [2x4黄, 2x2黄]”。
#   与原题相比，既有对象被删掉，也从单列变成多列。
#
# b) 复杂度校准
# - 若只保证“有重复面积”，可能退化成 D14 单变体；若过滤后只剩一种颜色，
#   又会退化成原题/D14 的单列输出。
# - generator 因此固定制造 4 个会保留的矩形，它们形成 2 个面积组 x 2 个颜色组
#   的交叉结构：两个面积各出现两次，两个颜色各出现两次，且同一面积里混有两种颜色、
#   同一颜色里也混有两种面积。
# - 另外再插入 1-2 个唯一面积矩形作为干扰项。这样“过滤”和“分组”两步都必须生效。
#
# c) 隐式假设审查
# - 原代码默认 `rects` 全部进入单个 `seq`；融合后这一前提失效，必须先做面积计数，
#   再把保留对象按主色拆成多个序列。
# - 原代码可把 bbox 原样当作对象；这里仍成立，因为 generator 只放置实心矩形，
#   少量异色点不会引入背景孔洞。
# - 新增的“按主色分组”不能假设矩形纯色，所以 solver 用 bbox 内出现次数最多的颜色。
# - 新增的“按面积过滤”会让保留集合为空；因此 `main()` 对目标域保持 `assert kept_rects`，
#   而 generator 保证至少两个重复面积组存在。
#
# d) 放置合法性
# - 继续沿用原题的两条水平带：第 1 带放在第 2-3 行，第 2 带放在第 5-6 行。
# - 每带对象之间固定留 2 列背景；左侧与边线之间固定留 1 列背景。
# - 本实现把总矩形数限制在 5-6 个，宽度只取 `{2,4,6,8}`，因此两带的总占宽始终
#   能落在原 spec 的长边范围内，不重叠、不越界。
#
# e) 视觉可观察性
# - 输入里能直接看见：有些矩形宽度重复，有些宽度只出现一次；输出里唯一宽度消失。
# - 过滤后剩下的 4 个矩形又会从单列改成两列，并且每列收集同主色对象。
# - 因为保留对象的颜色顺序在全局排序里是交错的，解题者无法用“本来就相邻”解释输出列。
#
# f) 可视化整洁度
# - 仍然只有边线、角点标记和若干矩形三类元素；无遮挡、无重叠。
# - 输出列之间固定插 1 列背景，避免不同颜色堆叠块粘连。
# - 异色噪点仅少量撒在矩形内部，用来保持旋转可观察，但不会破坏轮廓。
#
# g) 颜色变化可推断性
# - 没有颜色映射；颜色本身仅作为“分组键”。
# - 为避免误读成“只保留某些颜色”，generator 强制至少一个唯一面积干扰矩形复用保留组
#   的颜色，从而展示“同色也可能被过滤，决定因素先是面积频次，再是颜色分组”。


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([BLUE, RED], dtype=int)
WIDTH_POOL = np.array([2, 4, 6, 8], dtype=int)


def _locate_corner(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = map(int, g.shape)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for r, c in corners:
        if int(g[r, c]) == int(bg):
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
            return int(r), int(c)
    raise AssertionError("marked corner not found")


def _dominant_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _infer_edge_orientations(g: np.ndarray, cr: int, cc: int) -> tuple[str, str]:
    h, w = map(int, g.shape)
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < w and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < h and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        blue_col = int(vcol if vlen >= hlen else hcol)

    red_col = int(vcol if hcol == blue_col else hcol)
    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return blue_orient, red_orient


def _extract_rectangles(g: np.ndarray, bg: int, cr: int, cc: int, marker: int) -> list[dict]:
    h, w = map(int, g.shape)
    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, box)),
                "arr": arr.copy(),
                "area": int(arr.shape[0] * arr.shape[1]),
                "group": _dominant_color(arr),
            }
        )
    return rects


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        col = int(edge_idx)
        if c0 > col:
            return int(c0 - col - 1)
        if c1 < col:
            return int(col - c1 - 1)
        return 0
    row = int(edge_idx)
    if r0 > row:
        return int(r0 - row - 1)
    if r1 < row:
        return int(row - r1 - 1)
    return 0


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _analyze_grid(grid: np.ndarray) -> tuple[int, str, list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc = _locate_corner(g, bg)
    marker = int(g[cr, cc])
    blue_orient, red_orient = _infer_edge_orientations(g, cr, cc)
    rects = _extract_rectangles(g, bg, cr, cc, marker)
    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            _along_from_corner(rect, blue_orient, cr, cc),
            _along_from_corner(rect, red_orient, cr, cc),
        )
    )
    return bg, blue_orient, rects


def _stack_column(seq: list[np.ndarray], bg: int) -> np.ndarray:
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


def _stack_blocks(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(max(int(block.shape[0]) for block in blocks))
    out_w = int(sum(int(block.shape[1]) for block in blocks) + max(0, len(blocks) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_c = 0
    for block in blocks:
        _, w = map(int, block.shape)
        paste(out, block, 0, int(out_c))
        out_c += int(w) + 1
    return out


def _solve(grid: np.ndarray, *, min_area_freq: int, group_by_color: bool) -> np.ndarray:
    bg, blue_orient, rects = _analyze_grid(grid)
    area_freq = Counter(int(rect["area"]) for rect in rects)
    kept_rects = [rect for rect in rects if int(area_freq[int(rect["area"])]) >= int(min_area_freq)]
    assert kept_rects

    if not group_by_color:
        seq = []
        for rect in kept_rects:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)
        return _stack_column(seq, bg)

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    for rect in kept_rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        group = int(rect["group"])
        if group not in groups:
            groups[group] = []
            group_order.append(group)
        groups[group].append(arr)

    blocks = [_stack_column(groups[group], bg) for group in group_order]
    return _stack_blocks(blocks, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only repeated-area rectangles, then stack the survivors in color groups.

    Rule chain:
    1. Extract all interior solid rectangles.
    2. Filter out rectangles whose bbox area appears only once.
    3. Preserve the original geometric sort order among the survivors.
    4. If blue is vertical, rotate each survivor 90 degrees CCW.
    5. Group survivors by dominant color and stack each color group independently.
    """
    return _solve(grid, min_area_freq=2, group_by_color=True)


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    max_minor = min(3, max(0, (area - 1) // 3))
    if max_minor == 0 or np.random.rand() < 0.45:
        return patch
    n_minor = int(np.random.randint(1, max_minor + 1))
    cells = np.random.choice(area, size=n_minor, replace=False)
    colors = np.random.choice(MINORITY_COLORS, size=n_minor, replace=True)
    flat = patch.reshape(-1)
    for cell, color in zip(cells, colors):
        if int(color) != int(dominant):
            flat[int(cell)] = int(color)
    return patch


def _count_column_runs(grid: np.ndarray, bg: int) -> int:
    active = np.any(grid != int(bg), axis=0)
    runs = 0
    prev = False
    for flag in active:
        cur = bool(flag)
        if cur and not prev:
            runs += 1
        prev = cur
    return int(runs)


def _build_canonical() -> tuple[np.ndarray, dict]:
    kept_widths = [int(x) for x in np.random.choice(WIDTH_POOL, size=2, replace=False)]
    w0, w1 = kept_widths
    c0, c1 = [int(x) for x in np.random.choice(DOMINANT_COLORS, size=2, replace=False)]

    kept_specs = [
        {"width": int(w0), "dominant": int(c0), "kept": True},
        {"width": int(w1), "dominant": int(c1), "kept": True},
        {"width": int(w0), "dominant": int(c1), "kept": True},
        {"width": int(w1), "dominant": int(c0), "kept": True},
    ]

    remaining_widths = [int(x) for x in WIDTH_POOL if int(x) not in {int(w0), int(w1)}]
    np.random.shuffle(remaining_widths)
    n_unique = int(np.random.choice([1, 2]))
    unique_specs = []
    if n_unique >= 1:
        unique_specs.append({"width": int(remaining_widths[0]), "dominant": int(np.random.choice([c0, c1])), "kept": False})
    if n_unique == 2:
        other_color_choices = [int(x) for x in DOMINANT_COLORS if int(x) not in {int(c0), int(c1)}]
        unique_specs.append({"width": int(remaining_widths[1]), "dominant": int(np.random.choice(other_color_choices)), "kept": False})

    if n_unique == 1:
        seq = [kept_specs[0], kept_specs[1], unique_specs[0], kept_specs[2], kept_specs[3]]
    else:
        if np.random.rand() < 0.5:
            seq = [kept_specs[0], unique_specs[0], kept_specs[1], kept_specs[2], unique_specs[1], kept_specs[3]]
        else:
            seq = [kept_specs[0], kept_specs[1], unique_specs[0], kept_specs[2], unique_specs[1], kept_specs[3]]

    top_specs = seq[:3]
    bottom_specs = seq[3:]
    top_used = int(sum(int(spec["width"]) for spec in top_specs) + 2 * (len(top_specs) - 1))
    bottom_used = int(sum(int(spec["width"]) for spec in bottom_specs) + 2 * (len(bottom_specs) - 1))
    trailing = int(np.random.randint(1, 3))
    long_dim = int(2 + max(top_used, bottom_used) + trailing)
    assert 8 <= long_dim <= 28

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    placements = []
    for row, specs in ((2, top_specs), (5, bottom_specs)):
        col = 2
        for spec in specs:
            patch = np.full((2, int(spec["width"])), int(spec["dominant"]), dtype=int)
            patch = _sprinkle_minorities(patch, int(spec["dominant"]))
            paste(grid, patch, int(row), int(col))
            placements.append(
                {
                    "row": int(row),
                    "col": int(col),
                    "width": int(spec["width"]),
                    "dominant": int(spec["dominant"]),
                    "kept": bool(spec["kept"]),
                }
            )
            col += int(spec["width"]) + 2

    return grid, {"placements": placements, "kept_colors": [int(c0), int(c1)], "kept_widths": [int(w0), int(w1)]}


def _behavior_signature(grid: np.ndarray) -> dict:
    bg, _, rects = _analyze_grid(grid)
    area_freq = Counter(int(rect["area"]) for rect in rects)
    kept = [rect for rect in rects if int(area_freq[int(rect["area"])]) >= 2]

    color_freq = Counter(int(rect["group"]) for rect in kept)
    kept_area_to_colors: dict[int, set[int]] = defaultdict(set)
    kept_color_to_areas: dict[int, set[int]] = defaultdict(set)
    for rect in kept:
        kept_area_to_colors[int(rect["area"])].add(int(rect["group"]))
        kept_color_to_areas[int(rect["group"])].add(int(rect["area"]))

    out_combo = main(grid)
    out_group_only = _solve(grid, min_area_freq=1, group_by_color=True)
    out_filter_only = _solve(grid, min_area_freq=2, group_by_color=False)
    out_original = _solve(grid, min_area_freq=1, group_by_color=False)

    return {
        "bg": int(bg),
        "total_rects": int(len(rects)),
        "kept_rects": int(len(kept)),
        "filtered_rects": int(len(rects) - len(kept)),
        "kept_color_groups": int(len(color_freq)),
        "kept_color_counts": sorted(int(v) for v in color_freq.values()),
        "repeated_areas": sorted(int(area) for area, freq in area_freq.items() if int(freq) >= 2),
        "unique_areas": sorted(int(area) for area, freq in area_freq.items() if int(freq) == 1),
        "cross_cut_areas": int(sum(1 for colors in kept_area_to_colors.values() if len(colors) >= 2)),
        "cross_cut_colors": int(sum(1 for areas in kept_color_to_areas.values() if len(areas) >= 2)),
        "column_runs": int(_count_column_runs(out_combo, int(bg))),
        "diff_vs_group_only": bool(not np.array_equal(out_combo, out_group_only)),
        "diff_vs_filter_only": bool(not np.array_equal(out_combo, out_filter_only)),
        "diff_vs_original": bool(not np.array_equal(out_combo, out_original)),
    }


def _gen_attempt() -> np.ndarray:
    canonical, info = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    sig = _behavior_signature(grid)
    assert sig["total_rects"] == len(info["placements"])
    assert sig["kept_rects"] == 4
    assert sig["filtered_rects"] >= 1
    assert len(sig["repeated_areas"]) == 2
    assert len(sig["unique_areas"]) >= 1
    assert sig["kept_color_groups"] == 2
    assert sig["kept_color_counts"] == [2, 2]
    assert sig["cross_cut_areas"] >= 1
    assert sig["cross_cut_colors"] >= 1
    assert sig["column_runs"] == 2
    assert sig["diff_vs_group_only"]
    assert sig["diff_vs_filter_only"]
    assert sig["diff_vs_original"]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _gen_attempt()
        except (AssertionError, ValueError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 23")

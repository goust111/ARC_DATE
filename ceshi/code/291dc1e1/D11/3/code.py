# Variant of 291dc1e1
# Dimension: D11 — 颜色变换——堆叠时的颜色处理 (transform)
# Change: 矩形颜色在堆叠过程中保持不变，直接复制到输出... → 颜色转移——矩形颜色变为其对应band线的颜色

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for this D11 variant
# a) 具体化与行为追踪:
#    原任务先按矩形到 blue/red 两条边线的距离排序，再把矩形直接堆叠到输出中，颜色原样保留。
#    本变体只改“进入输出序列时的着色”这一步：先按原规则求出每个矩形的 blue-distance(db)。
#    db 最小的一层视为 near band，输出时整块重着色为蓝色(1)；其余更远层视为 far band，输出时重着色为红色(2)。
#    小例子: 两排 2xh 矩形分别位于距 blue 线 1 格和 4 格的位置。原逻辑输出仍是绿/黄/橙等原色矩形；
#    新逻辑输出保留同样的排序、旋转、堆叠形状，但前一排全部改成 1，后一排全部改成 2。
# b) 复杂度校准:
#    只保留两层 band 归属，避免“每个 db 一个颜色”导致规则过细。输入矩形仍允许少量杂色，输出统一改色，
#    这样颜色转移非常明显，同时不会引入额外的对象匹配歧义。
# c) 隐式假设审查:
#    - corner / line 定位逻辑依赖“角点标记 + 一横一竖边线”这一旧属性，D11 不改变它，所以保留。
#    - 矩形提取逻辑依赖“bbox 内无背景色”，变体不需要改这个前提；重着色时仍只改非背景像素，避免未来带孔版本被误填满。
#    - 原始 `seq.append(a)` 默认颜色可直接复用；变体中这一步已不成立，必须在 append 前按 db 映射到 1/2。
# d) 放置合法性:
#    生成器继续使用两条 band、矩形间隔 2 格、整块 paste 的方式，保证不重叠、不越界，也保证 solver 会检测到多个矩形组件。
# e) 视觉可观察性:
#    输入里矩形主色只取 3/4/5/6/7/9，避免与 1/2 混淆；输出统一变成 1/2，因此 2-3 个样例就能看出“颜色由 band 决定”。
# f) 可视化整洁度:
#    仍沿用原题的整齐带状布局，线、标记、矩形彼此分区明确，不增加额外装饰。
# g) 颜色变化可推断性:
#    颜色只压缩到两类，并由“离 blue 线近/远”这个稳定空间线索驱动。每个样例中两个 band 都重复出现，足够支撑归纳。
# h) 参考任务说明:
#    仓内允许参考的文件中未找到 `4c416de3` solver 源码文件，只找到该题 evaluation JSON，因此这里只借鉴其
#    “shape 保留、颜色整体转移”的机制，不引入其他维度变化。
# i) 新增假设与生成器保证:
#    新 solver 假设至少存在一个 near-band 矩形，且生成器最好提供两层不同 db。生成器固定构造两排矩形并用
#    `_variant_is_visible` 自检：必须存在至少两个 db 层，且新旧输出不同、输出同时含 1 和 2。

BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, str, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
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
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})
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

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

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


def _solve(grid: np.ndarray, *, recolor: bool) -> np.ndarray:
    g, bg, blue_orient, rects = _analyze_grid(grid)
    del g

    near_db = min(int(rect["db"]) for rect in rects)
    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        if recolor:
            target = BLUE if int(rect["db"]) == int(near_db) else RED
            arr = np.where(arr != bg, int(target), int(bg))
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


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, recolor=True)


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
    if area < 8 or np.random.rand() < 0.4:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor) + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    options = []
    for trailing in (1, 2):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
            continue
        max_count = min(5, (max_used + 2) // 4)
        pairs = []
        for near_count in range(2, max_count + 1):
            for far_count in range(2, max_count + 1):
                if near_count + far_count > 9:
                    continue
                eligible = []
                if max_used <= 10 * near_count - 2:
                    eligible.append(0)
                if max_used <= 10 * far_count - 2:
                    eligible.append(1)
                if eligible:
                    pairs.append((near_count, far_count, eligible))
        if pairs:
            options.append((max_used, pairs))

    max_used, pairs = options[int(np.random.randint(len(options)))]
    n0, n1, eligible = pairs[int(np.random.randint(len(pairs)))]
    max_band = int(np.random.choice(eligible))
    max0 = min(max_used, 10 * n0 - 2)
    max1 = min(max_used, 10 * n1 - 2)
    used0 = int(max_used if max_band == 0 else np.random.choice(np.arange(4 * n0 - 2, max0 + 1, 2)))
    used1 = int(max_used if max_band == 1 else np.random.choice(np.arange(4 * n1 - 2, max1 + 1, 2)))
    band0 = _distribute_band_widths(n0, used0)
    band1 = _distribute_band_widths(n1, used1)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2
    return grid


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def _variant_is_visible(grid: np.ndarray) -> bool:
    rects = _analyze_grid(grid)[3]
    if len({int(rect["db"]) for rect in rects}) < 2:
        return False
    variant = _solve(grid, recolor=True)
    baseline = _solve(grid, recolor=False)
    if np.array_equal(variant, baseline):
        return False
    colors = set(map(int, variant.flatten()))
    return BLUE in colors and RED in colors


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        try:
            _ = main(grid)
        except Exception:
            continue
        if _variant_is_visible(grid):
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D11 variant")

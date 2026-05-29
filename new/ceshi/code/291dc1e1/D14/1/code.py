# Variant of 291dc1e1
# Dimension: D14 — 干扰项与噪声 (src)
# Change: 网格中除矩形和边缘线外无额外对象... → 增加第三条非功能性装饰线

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the D14 variant

a) 具体化与行为追踪
   - 原题输入里只有三类前景元素: 角点 marker、两条从该角点出发的功能线、若干实心矩形。
   - 本变体保留前两条功能线与矩形的关系，但额外加入一条第三色、1 像素厚、与矩形不接触的装饰线。
   - 这条装饰线放在 canonical 输入中远离 marker 的对侧边缘，并且与上下角都留出空隙，因此在不同翻转/旋转后会变成另一侧边缘上的横线或竖线，但都不会经过 marker。
   - 小例子追踪:
     原始逻辑: corner row/col 被 mask 掉后，其余每个“无背景的连通块”都被当成矩形。
     变体逻辑: 新增装饰线也是“无背景的连通块”，若仍按原逻辑会被错当成一个 1xN 矩形并参与排序。
     修正后逻辑: 仍先找 marker 邻接的两条功能线；提取连通块时忽略所有厚度为 1 的细线组件，只保留至少 2x2 的实心块。

b) 复杂度校准
   - 只增加一条装饰线，不再额外叠加多个噪声源，避免自由度过高。
   - 装饰线长度取 4..W-4，足够显眼，不会退化为单像素噪声。
   - 装饰线颜色从矩形常用色中随机抽取，但该颜色会从该样例的矩形调色板中剔除，避免“装饰线颜色恰好也大量出现在矩形里”导致观察困难。

c) 隐式假设审查（原 main 中与本变体相关的脆弱点）
   - corner 检测依赖: 只有 marker 在角上，且其相邻两格颜色不同于 marker。本变体不改这一点。
   - hcol/vcol 读取依赖: 功能线仍紧邻 marker。本变体不改这一点。
   - line_mask 依赖: 功能线恰好位于 marker 的整行整列。本变体不改这一点，装饰线不会落在该行列上。
   - 连通块提取依赖: “所有无背景连通块都是矩形”。这一点在本变体失效，因为装饰线也是无背景连通块。
   - 修正方式: 在 bbox 内无背景的前提下，再要求组件的高和宽都至少为 2；1 像素厚组件视为干扰线，不参与排序。

d) 放置合法性
   - canonical 网格仍是 8xL，矩形只放在第 2-3 行和第 5-6 行。
   - 装饰线固定放在远离 marker 的对侧边缘列上，只覆盖第 2..6 行，并与上下角都留空隙，因此不与矩形、marker、功能线重叠，也不会和矩形发生 4 邻接粘连。
   - 整体做旋转/翻转后，这个“不重叠”性质保持成立，所以 solver 提取到的矩形组件不会被装饰线切割或粘连。

e) 视觉可观察性
   - 人类可以直接看到: 两条功能线始终从带 marker 的角点出发；第三条线孤立在内部、不接触角点。
   - 仅看 2-4 组样例也能归纳: 输出只由矩形构成，内部第三条线从不出现在输出里，因此它是干扰项。

f) 可视化整洁度
   - 装饰线只占用原本保留出来的对侧边缘空位，不覆盖矩形。
   - 其长度较长、形状单纯，与 2xN / Nx2 矩形在视觉上容易区分，不会造成“哪个对象属于哪个连通块”这种混乱。

g) 颜色变化可推断性
   - 本变体不引入颜色映射规则，颜色只承担“对象类别可分辨”的作用。
   - 为避免把“某个固定颜色总是装饰线”变成捷径，装饰线颜色在多个候选色之间随机变化；同时该颜色不再出现在同一样例的矩形内部，确保结构线索仍然清晰。

h) 变体行为指标
   - 可量化指标 1: 每个输入都必须含有恰好一条内部、单色、1 像素厚、长度 >= 4 的非 marker 细线。
   - 可量化指标 2: 在相同输入上，旧 solver 会把该细线当成额外矩形，因而旧/新输出应高频不同。

i) 新 solver 的隐式假设审查
   - 新增过滤规则假设“真正需要排序的矩形最小厚度为 2”。generator 通过始终生成 2xW 或 Hx2 的矩形来保证。
   - 新增过滤规则还假设装饰线不与矩形粘连。generator 通过把装饰线放在 canonical 空白带中保证这一点。
   - 排序与旋转逻辑仍假设功能线只由 marker 邻接的两条线定义。generator 保持该结构不变。
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
BODY_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_BASE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by proximity to two functional edge lines.

    D14 change:
    - The input may also contain one extra decorative thin line of a third color.
    - Only the two lines adjacent to the corner marker are functional.
    - Thin 1-pixel-thick components away from the marker are ignored as noise.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Locate the marked corner: non-background corner whose neighbors are different from it.
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

    # Functional line colors are the two colors directly adjacent to the marker.
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == 1 or vcol == 1:
        blue_col = 1
        red_col = int(vcol if hcol == 1 else hcol)
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

    # Mask the actual functional axes. The decorative line never passes through the marker row/col.
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    # Extract rectangles: solid, non-background components whose bbox has no holes and thickness >= 2.
    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if bool((sub == int(bg)).any()):
            continue
        if min(map(int, sub.shape)) < 2:
            continue
        rects.append({"box": tuple(map(int, b)), "arr": sub.copy()})

    assert rects

    def dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            col = int(edge_idx)
            if int(c0) > col:
                return int(c0 - col - 1)
            if int(c1) < col:
                return int(col - c1 - 1)
            return 0
        row = int(edge_idx)
        if int(r0) > row:
            return int(r0 - row - 1)
        if int(r1) < row:
            return int(row - r1 - 1)
        return 0

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    def along_from_corner(rect: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, rect["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    for rct in rects:
        rct["db"] = int(dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(dist_to_edge(rct, red_orient, red_edge_idx))

    rects.sort(
        key=lambda r: (
            int(r["db"]),
            int(r["dr"]),
            along_from_corner(r, blue_orient),
            along_from_corner(r, red_orient),
        )
    )

    seq = []
    for rct in rects:
        a = rct["arr"]
        if blue_orient == "v":
            a = np.rot90(a, k=1)
        seq.append(a)

    outW = int(max(int(a.shape[1]) for a in seq))
    outH = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((outW - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)

    return out


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int, deco_color: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
        return patch

    max_minor = min(3, area // 4)
    if max_minor < 2:
        return patch

    minority_choices = [int(c) for c in MINORITY_BASE if int(c) != dominant and int(c) != deco_color]
    if not minority_choices:
        return patch

    n_minor = int(np.random.randint(2, max_minor + 1))
    n_colors = int(np.random.randint(1, min(2, n_minor, len(minority_choices)) + 1))
    colors = np.random.choice(minority_choices, size=n_colors, replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % n_colors])
    return patch


def _add_decorative_line(grid: np.ndarray, deco_color: int) -> None:
    # Put the distractor on the border opposite the marker, but keep gaps to both corners.
    grid[2:7, -1] = int(deco_color)


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(9, 29))
    deco_color = int(np.random.choice(BODY_COLORS))
    body_choices = np.array([int(c) for c in BODY_COLORS if int(c) != deco_color], dtype=int)

    options = []
    for trailing in (2,):
        max_used = long_dim - 2 - trailing
        if max_used < 6 or max_used % 2:
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
            color = int(np.random.choice(body_choices))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color, deco_color)
            paste(grid, patch, r0, c)
            c += w + 2

    _add_decorative_line(grid, deco_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            grid = _gen_attempt()
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D14")

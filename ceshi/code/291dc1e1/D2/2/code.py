# Variant of 291dc1e1
# Dimension: D2 — 排序消歧规则 (match_cond)
# Change: 主键为到蓝线距离，次键为到红线距离，再次键为沿蓝线方向坐标和沿红线方向坐标
# ... → 次键反转（到红线距离降序）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for the D2 variant

Source-task review:
- No source_task was provided, so there is no extra solver to inspect.

a) 具体化与行为追踪
- 原任务的输入是一个“L 形坐标轴”加若干内部矩形。矩形按
  (到蓝线距离, 到红线距离, 两个方向上的位置) 排序，再竖直堆叠输出。
- 本变体只改同一 blue-distance band 内的消歧规则：
  原来按到红线距离 dr 升序，现在改成 dr 降序。
- 在 canonical 输入里，蓝线在上边、红线在左边，两条 band 分别位于第 2 行和第 5 行。
  对于同一 band，dr 实际上就是“离左边红线有多远”，所以新规则会把
  同一 band 中矩形的读取顺序从“左到右”改成“右到左”。
- 小例子：
  左 band 上有三个矩形 A,B,C，它们的 db 都是 1，而 dr 分别是 1,5,9。
  原逻辑排序得到 A,B,C；新逻辑得到 C,B,A。
  这只影响 band 内部次序，不改变 band 之间先后，也不改变 blue 垂直时的旋转规则。

b) 复杂度校准
- 若同一 band 只有 1 个矩形，或者矩形完全同色同宽，则“反转 dr”在视觉上可能不可见。
- 因此生成器固定每个 band 至少 2 个矩形，并让同一 band 内矩形宽度互异、主色互异。
  这样 band 内反转会稳定改变输出的局部顺序，人眼能从少量样例归纳出“第二关键字变号”。
- 仍然保留原任务的两个 distance 维度和随机角落/朝向，避免退化成单一排列小游戏。

c) 隐式假设审查（与变体相关的 main 段落）
- 角点检测、蓝/红线识别、line mask、矩形提取都不依赖 dr 的升降序，因此保持不变。
- `dist_to_edge()` 仍然正确，因为变体没有改变“距离”的定义，只改比较方向。
- 原排序代码默认 `dr` 升序；这是唯一直接依赖旧属性的地方。
  若不改这里，solver 会输出原任务答案而不是变体答案。
- `along_from_corner()` 仍作为更后面的平局消解键保留。它依赖“同一个 dr 可能仍需稳定排序”；
  生成器虽然尽量减少这种平局，但 solver 不应假设它永远不存在。

d) 放置合法性
- 生成器继续使用两个高度为 2 的 band，矩形从第 2 列起以至少 2 格间距排开，不与边线重叠。
- 行高固定为 8，使两个 band 完全分离；列宽由 band 最宽占用量反推，保证不越界。
- 变换到其他 corner / blue orientation 时只做整体翻转或旋转，不会引入重叠。

e) 视觉可观察性
- 每个 band 至少 2 个矩形，且宽度不同、主色不同，所以反转后堆叠顺序明显变化。
- 生成器额外拒绝所有“新旧排序输出完全相同”的样本，直接防止退化回原行为。
- 部分矩形加入少量非背景内部像素，保留原任务里“旋转后内部细节也跟着转”的线索。

f) 可视化整洁度
- 输入仍然是干净的两条边线 + 两排矩形；不同元素之间有固定空隙，避免组件粘连。
- 输出仍为居中竖直堆叠，不增加新的标记层，也不引入覆盖关系。

g) 颜色变化可推断性
- 本变体不引入新的颜色映射。颜色只用于区分矩形身份与内部纹理。
- 为避免颜色过杂，主色仅从 6 个颜色中选；每个 band 内主色不重复。

9) 新 solver 的隐式假设
- solver 仍假设：存在唯一角点标记；内部对象是无背景孔洞的矩形；矩形彼此 4 连通分离。
- generator 通过固定 L 形边线、2 格空隙、2 行高矩形来保证这些假设成立。
- 新增的唯一行为性假设是：若要让 D2 可观察，至少一个 band 里要有可区分的多个矩形。
  generator 通过 band 大小与“新旧输出必须不同”的验收来保证这一点。
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_POOL = np.array([2, 4, 6, 8], dtype=int)


def _locate_structure(g: np.ndarray) -> tuple[int, int, int, str, str]:
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
    return int(cr), int(cc), int(g[cr, cc]), blue_orient, red_orient


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rects(g: np.ndarray) -> tuple[int, list[dict], str]:
    H, W = g.shape
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_structure(g)

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

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect["box"], blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect["box"], red_orient, red_edge_idx))
        rect["ab"] = int(_along_from_corner(rect["box"], blue_orient, cr, cc))
        rect["ar"] = int(_along_from_corner(rect["box"], red_orient, cr, cc))

    return int(bg), rects, blue_orient


def _solve_with_red_order(grid: np.ndarray, red_desc: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, rects, blue_orient = _extract_rects(g)

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            -int(rect["dr"]) if red_desc else int(rect["dr"]),
            int(rect["ab"]),
            int(rect["ar"]),
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
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_red_order(grid, red_desc=True)


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
    if area < 6:
        return patch

    patch = patch.copy()
    n_minor = 1 if area < 10 else int(np.random.randint(1, 3))
    cells = np.random.choice(area, size=n_minor, replace=False)
    choices = [int(c) for c in MINORITY_COLORS if int(c) != int(dominant)]
    colors = np.random.choice(choices, size=n_minor, replace=True)
    flat = patch.reshape(-1)
    for cell, color in zip(cells, colors):
        flat[int(cell)] = int(color)
    return patch


def _sample_band(count: int) -> tuple[list[int], list[int]]:
    widths = [int(w) for w in np.random.choice(WIDTH_POOL, size=count, replace=False)]
    widths = [int(w) for w in np.random.permutation(widths)]
    colors = [int(c) for c in np.random.choice(DOMINANT_COLORS, size=count, replace=False)]
    return widths, colors


def _build_canonical() -> np.ndarray:
    n0 = int(np.random.randint(2, 4))
    n1 = int(np.random.randint(2, 4))
    widths0, colors0 = _sample_band(n0)
    widths1, colors1 = _sample_band(n1)

    used0 = int(sum(widths0) + 2 * (n0 - 1))
    used1 = int(sum(widths1) + 2 * (n1 - 1))
    long_dim = int(max(14, max(used0, used1) + int(np.random.randint(4, 9))))

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, widths, colors in ((2, widths0, colors0), (5, widths1, colors1)):
        c = 2
        for width, color in zip(widths, colors):
            patch = np.full((2, int(width)), int(color), dtype=int)
            patch = _sprinkle_minorities(patch, int(color))
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _variant_is_visible(grid: np.ndarray) -> bool:
    new_out = _solve_with_red_order(grid, red_desc=True)
    old_out = _solve_with_red_order(grid, red_desc=False)
    return not np.array_equal(new_out, old_out)


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    if not _variant_is_visible(grid):
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D2/2")

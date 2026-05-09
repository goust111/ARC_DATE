# Variant of 291dc1e1
# Dimension: D7 — 输出网格尺寸与裁剪 (grid_shape)
# Change: 输出尺寸紧密适配：高度=所有旋转后矩形高度之和，宽度=最大旋转后矩形宽度... → 固定宽度输出（取输入短边长度）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Variant design analysis
=======================

a) 具体化与行为追踪
   - 输入仍然是原题结构：一条水平边线和一条垂直边线在角上相交，角点是独特 marker，
     内部是若干个 2 行高的实心矩形，矩形里允许有少量非背景杂色像素来让旋转可见。
   - 原题输出把排序后的矩形竖向堆叠，并把输出宽度裁到“当前序列里最宽矩形”的宽度。
   - 本变体只改这个裁剪行为：输出宽度固定为输入短边长度 `min(H, W)`，在本题生成器里恒为 8。
     每个矩形仍然居中贴到这个 8 列画布里，两侧多出的列保留背景色。
   - 小例子追踪：
     假设排序/旋转后的三个矩形宽度依次是 4, 2, 6。
     原题：`outW = 6`，三个矩形的左偏移是 1, 2, 0。
     变体：`outW = 8`，左偏移改成 2, 3, 1。
     排序、旋转、堆叠顺序完全不变，只是最终画布更宽，左右留出背景边。

b) 复杂度校准
   - 如果生成器继续经常产出宽度 8 的矩形，那么“固定宽度 8”和“紧配到最大宽度”会在很多样本上完全一致，
     D7 的变化肉眼不可见，所以这里把 canonical 矩形宽度限制在 {2, 4, 6}。
   - 同时保留 2 个矩形带、每带 2-4 个矩形，以及 4 个角/2 种蓝线朝向，这样排序与旋转分支仍然有重复支撑，
     但自由度没有扩张到难以归纳。

c) 隐式假设审查
   - 原 `main()` 的前半段（找角点、识别两条边线、提取矩形、按到边线距离排序）都不依赖输出裁剪方式，
     所以保持不动。
   - 真正依赖旧 D7 的只有：
       `outW = max(a.shape[1] for a in seq)`
       `out_c = (outW - w) // 2`
     变体下第一行必须改为固定 `min(H, W)`；第二行仍然成立，但新增了一个前提：
     所有旋转后的矩形宽度必须不超过输入短边，否则 paste 会越界。
   - 因此 `main()` 里显式断言 `max_width <= min(H, W)`，并由生成器保证这一点。

d) 放置合法性
   - 生成器仍在 canonical 8xN 网格中放两条矩形带，矩形高度固定 2、行带固定在 `r=2` 和 `r=5`，
     矩形之间横向间隔 2，边线与矩形之间也至少隔 1 格，不重叠、不越界。
   - 经过旋转/翻转后，这些对象的拓扑关系保持，solver 仍能按原方式找到线、角点和矩形分量。

e) 视觉可观察性
   - 这个变体唯一要让人看出的就是“输出不再被最宽矩形裁死，而是保持 8 列”。
   - 因此样本必须高频出现 `tight_width < 8`。这里通过构造保证 `tight_width` 只会是 2/4/6，
     所以每个样本都会出现额外背景留白。
   - 当蓝线竖直时，矩形会逆时针旋转，矩形内部少量杂色像素让旋转方向也能直接看出来。

f) 可视化整洁度
   - 输入里只有边线、marker、两条矩形带三层结构，没有额外叠放。
   - 输出里所有矩形只做纵向堆叠并水平居中，留白是纯背景色，不会引入新的视觉噪声。

g) 颜色变化可推断性
   - 本变体不引入新的颜色规则；颜色只沿用原题的“边线颜色定义坐标系、矩形内部颜色随对象一起移动/旋转”。
   - 杂色像素只用于破除对称，数量很少，不会把题目误导成颜色映射题。

New assumptions introduced by the variant
----------------------------------------
- `main()` 现在要求：所有排序并旋转后的矩形宽度 `<= min(H, W)`。
- `generate_input()` 通过只生成宽度 {2,4,6} 的 2xw 矩形来保证：
  * 蓝线水平时，输出 tight width 为 {2,4,6}
  * 蓝线垂直时，矩形旋转后宽度恒为 2
  所以固定宽度 8 一定合法，而且与原题的紧配宽度稳定不同。
"""


def _extract_sorted_sequence(grid: np.ndarray) -> tuple[int, str, list[np.ndarray]]:
    """Return (bg, blue_orient, sorted/rotated rectangles) using the original task logic."""
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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

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

    seq = []
    for rect in rects:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    return int(bg), str(blue_orient), seq


def _tight_output_width(grid: np.ndarray) -> int:
    """Width that the original tight-crop solver would have used on the same input."""
    _bg, _blue_orient, seq = _extract_sorted_sequence(grid)
    return int(max(int(a.shape[1]) for a in seq))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, _blue_orient, seq = _extract_sorted_sequence(g)

    outW = int(min(map(int, g.shape)))
    max_rect_w = int(max(int(a.shape[1]) for a in seq))
    assert max_rect_w <= outW

    outH = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((outH, outW), int(bg), dtype=int)

    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((outW - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)

    return out


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([2, 4, 6], dtype=int)


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


def _sample_band_widths() -> list[int]:
    """Choose 2-4 rectangles whose widths make the D7 widening visible."""
    for _ in range(200):
        count = int(np.random.randint(2, 5))
        widths = [int(x) for x in np.random.choice(WIDTH_CHOICES, size=count, replace=True)]
        used = int(sum(widths) + 2 * (count - 1))
        if used <= 22:
            return widths
    raise RuntimeError("failed to sample band widths")


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.55:
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
    for _ in range(300):
        band0 = _sample_band_widths()
        band1 = _sample_band_widths()
        used0 = int(sum(band0) + 2 * (len(band0) - 1))
        used1 = int(sum(band1) + 2 * (len(band1) - 1))
        min_long = int(max(used0, used1) + 3)
        max_long = int(min(28, min_long + 6))
        if min_long > max_long:
            continue

        long_dim = int(np.random.randint(min_long, max_long + 1))
        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in ((2, band0), (5, band1)):
            c = 2
            for w in widths:
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = np.full((2, int(w)), color, dtype=int)
                patch = _sprinkle_minorities(patch, color)
                paste(grid, patch, int(r0), int(c))
                c += int(w) + 2

        return grid

    raise RuntimeError("failed to build canonical grid")


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    fixed_w = int(min(map(int, grid.shape)))
    tight_w = int(_tight_output_width(grid))
    if tight_w >= fixed_w:
        return None

    out = main(grid)
    if int(out.shape[1]) != fixed_w:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D7 variant")

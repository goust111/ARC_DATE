# Variant of 291dc1e1
# Dimension: D1 — 排序主键——距离度量方式 (match_cond)
# Change: 按 (到蓝线的距离, 到红线的距离) 升序排序，距离为矩形边缘到边线的像素间隔... → 按矩形长边长度排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

"""
Design analysis for the D1 variant

a) 具体化与行为追踪
- 输入外框仍然是原任务那套结构：一个角上有 marker，蓝线与红线沿两条边延伸，内部是若干没有背景孔洞的矩形。
- 变体只改“矩形被取出的顺序”。原任务按矩形与蓝线/红线的间距排序；现在改成按“矩形沿蓝线方向的边长”排序。
- 在 canonical 视角里，蓝线先固定为顶边，所以矩形都是高 2、宽不同的横向条块；若之后把整张图旋转成“蓝线竖直”，这些矩形就会在输入里变成宽 2、高不同的竖条块。solver 因而必须依据 blue_orient 决定测量宽还是高。
- 小例子：蓝线水平时，有三个矩形宽度分别为 6、2、4，空间位置从左到右却是 4、6、2。
  原逻辑：先看到蓝线距离，再看红线距离，结果会按“上排从左到右、下排从左到右”输出。
  变体逻辑：忽略它们离边线的远近，直接按宽度 2 < 4 < 6 输出。
  差异点是：排序依据从“位置”变成“尺寸”。

b) 复杂度校准
- 若允许大量同宽矩形，新的排序规则会退化成依赖 tie-break，肉眼也很难确认规则。所以生成器强制每个样例里的矩形沿蓝线方向长度互不相同。
- 仍保留两排矩形，而不是只放一排。这样输入里继续存在“位置坐标系”这一旧线索，解题者必须主动放弃旧线索，转而比较尺寸，变体才明显。
- 矩形数量保持 4-6 个，足够形成可见排序，但不会拥挤到难以辨认。

c) 隐式假设审查
- 原 main() 中与本变体强相关的代码是 `dist_to_edge(...)`、`db/dr` 赋值、以及按 `(db, dr, ...)` 排序。这一段默认“位置就是主线索”，在变体中已不成立，必须替换成尺寸键。
- 原来的 corner 定位、蓝/红线识别、矩形提取、以及“蓝线竖直则整体旋转输出”都不依赖旧排序属性，因此保留。
- 旧代码中 `crop_bbox` 后只检查是否含背景色，这是在假设对象是实心矩形。生成器继续保证没有背景孔洞，所以这一前提仍成立。

d) 放置合法性
- 生成器仍在整条边线之外留出 1 格背景缓冲：矩形从 canonical 的列 2 开始放，且只占据第 2-3 行与第 5-6 行，避免触碰 marker/边线。
- 每排矩形之间固定留 2 格背景，不重叠、不越界；整张 canonical 网格宽度按本排总占宽动态计算。
- 变换到其他 corner / blue_orient 只使用翻转和 90 度旋转，矩形依然是轴对齐实心矩形，因此 solver 的 bbox 提取仍然合法。

e) 视觉可观察性
- 人类需要从少量样例中看出“不是谁更靠近边线，而是谁更长”。因此生成器把矩形长度打乱到两排里，并拒绝那些“新排序与旧排序刚好相同”的样例。
- 保留少量非对称内部杂色点，使蓝线竖直时的 90 度旋转在输出里可见，避免误以为只是简单重排。

f) 可视化整洁度
- 仍然采用两条整齐横带来承载矩形，元素分层清晰：角标/边线/背景/矩形互不覆盖。
- 不引入额外标记或新颜色机制，避免把 D1 变体污染成别的维度。

g) 颜色变化可推断性
- 颜色不承载规则，只用于区分矩形和维持旋转可见性。dominant color 数量受控，内部 minority pixel 也很少，不会把任务变成颜色映射题。

9) 修改后 solver 的新假设，以及 generator 如何保证
- 新假设 1：每个矩形沿蓝线方向的长度可以从其 bbox 直接读出。
  生成器保证矩形始终轴对齐、无背景孔洞。
- 新假设 2：若多个矩形长度相同，solver 只能退回到位置 tie-break。
  生成器通过不重复长度来避免把规则变成“尺寸 + 隐式位置”的混合题。
- 新假设 3：样例应当真正体现新旧规则的差异。
  生成器显式比较“尺寸排序”和“旧距离排序”的对象顺序；若一致则丢弃重采样。
"""


def _locate_frame(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
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
    return bg, marker, cr, cc, blue_orient, red_orient


def _extract_rectangles(g: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    H, W = g.shape
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
    return rects


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        if int(c0) > int(edge_idx):
            return int(c0 - edge_idx - 1)
        if int(c1) < int(edge_idx):
            return int(edge_idx - c1 - 1)
        return 0
    if int(r0) > int(edge_idx):
        return int(r0 - edge_idx - 1)
    if int(r1) < int(edge_idx):
        return int(edge_idx - r1 - 1)
    return 0


def _thick_dim(box: tuple[int, int, int, int], blue_orient: str) -> int:
    r0, c0, r1, c1 = map(int, box)
    if blue_orient == "h":
        return int(c1 - c0 + 1)
    return int(r1 - r0 + 1)


def _legacy_order_boxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    bg, marker, cr, cc, blue_orient, red_orient = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    rects.sort(
        key=lambda r: (
            int(_dist_to_edge(r["box"], blue_orient, blue_edge_idx)),
            int(_dist_to_edge(r["box"], red_orient, red_edge_idx)),
            int(_along_from_corner(r["box"], blue_orient, cr, cc)),
            int(_along_from_corner(r["box"], red_orient, cr, cc)),
        )
    )
    return [tuple(map(int, r["box"])) for r in rects]


def _size_order_boxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    bg, marker, cr, cc, blue_orient, red_orient = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    rects.sort(
        key=lambda r: (
            int(_thick_dim(r["box"], blue_orient)),
            int(_along_from_corner(r["box"], blue_orient, cr, cc)),
            int(_along_from_corner(r["box"], red_orient, cr, cc)),
        )
    )
    return [tuple(map(int, r["box"])) for r in rects]


def variant_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg, marker, cr, cc, blue_orient, _ = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)
    size_order = _size_order_boxes(g)
    legacy_order = _legacy_order_boxes(g)
    positions_changed = sum(int(a != b) for a, b in zip(size_order, legacy_order))
    lengths = sorted(int(_thick_dim(r["box"], blue_orient)) for r in rects)
    return {
        "blue_orient": blue_orient,
        "rect_count": len(rects),
        "lengths_sorted": lengths,
        "legacy_matches_variant": bool(size_order == legacy_order),
        "positions_changed": int(positions_changed),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder and stack rectangles by size instead of edge distance.

    The corner/edge-line parsing and the "rotate all rectangles when blue is
    vertical" behavior are preserved from the source task. The only semantic
    change is the sort key:
    - old: distance to blue, then distance to red
    - new: rectangle length measured parallel to the blue edge
    """
    g = np.array(grid, dtype=int)
    bg, marker, cr, cc, blue_orient, red_orient = _locate_frame(g)
    rects = _extract_rectangles(g, bg, marker, cr, cc)

    rects.sort(
        key=lambda r: (
            int(_thick_dim(r["box"], blue_orient)),
            int(_along_from_corner(r["box"], blue_orient, cr, cc)),
            int(_along_from_corner(r["box"], red_orient, cr, cc)),
        )
    )

    seq = []
    for rct in rects:
        arr = rct["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)

    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)

    return out


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


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 6:
        return patch
    if np.random.rand() < 0.45:
        return patch

    max_minor = min(3, area // 3)
    n_minor = int(np.random.randint(1, max_minor + 1))
    minority_choices = [int(c) for c in MINORITY_COLORS if int(c) != dominant]
    colors = np.random.choice(minority_choices, size=min(2, n_minor), replace=False)
    cells = np.random.choice(area, size=n_minor, replace=False)
    flat = patch.reshape(-1)
    for idx, cell in enumerate(cells):
        flat[int(cell)] = int(colors[idx % len(colors)])
    return patch


def _build_canonical() -> np.ndarray:
    n_rects = int(np.random.randint(4, 7))
    widths = list(map(int, np.random.choice(np.arange(2, 10), size=n_rects, replace=False)))
    colors = list(map(int, np.random.choice(DOMINANT_COLORS, size=n_rects, replace=False)))

    perm = list(map(int, np.random.permutation(n_rects)))
    specs = [(widths[i], colors[i]) for i in perm]

    top_count = int(np.random.randint(2, n_rects - 1))
    top_specs = specs[:top_count]
    bot_specs = specs[top_count:]

    used_top = sum(w for w, _ in top_specs) + 2 * (len(top_specs) - 1)
    used_bot = sum(w for w, _ in bot_specs) + 2 * (len(bot_specs) - 1)
    trailing = int(np.random.randint(1, 4))
    long_dim = int(max(used_top, used_bot) + 2 + trailing)

    grid = np.full((8, long_dim), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    for r0, band in ((2, top_specs), (5, bot_specs)):
        c = 2
        for width, color in band:
            patch = np.full((2, width), int(color), dtype=int)
            patch = _sprinkle_minorities(patch, int(color))
            paste(grid, patch, int(r0), int(c))
            c += int(width) + 2

    return grid


def _is_non_degenerate(grid: np.ndarray) -> bool:
    metrics = variant_metrics(grid)
    return (not metrics["legacy_matches_variant"]) and len(set(metrics["lengths_sorted"])) == metrics["rect_count"]


def _gen_attempt() -> np.ndarray:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    return _transform(canonical, blue_orient, corner)


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        try:
            if not _is_non_degenerate(grid):
                continue
            _ = main(grid)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate input for 291dc1e1 D1 variant")

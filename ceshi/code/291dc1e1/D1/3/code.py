# Variant of 291dc1e1
# Dimension: D1 — 排序主键——距离度量方式 (match_cond)
# Change: 按 (到蓝线的距离, 到红线的距离) 升序排序，距离为矩形边缘到边线的像素间隔... → 距离降序排序（远→近）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis for this variant
# a) 具体化与行为追踪
#    - 输入仍然是: 一个角点标记、从角点出发的一条横边线和一条竖边线、以及内部若干无背景孔洞的矩形块。
#    - 原任务里，矩形按 (到蓝线距离, 到红线距离) 从近到远堆叠。
#      例如在 canonical 输入中，靠上那排矩形到蓝线距离为 1，靠下那排为 4；
#      同一排内又按到红线的距离从左到右递增，因此原输出会先放上排左侧，再上排右侧，最后下排。
#    - 本变体只把主排序方向反过来: 先看谁离蓝线更远，若相同再看谁离红线更远。
#      同一个例子里，新输出会先放下排右侧，再下排左侧，最后上排右侧、上排左侧。
#    - 变化影响的唯一关键环节是 `rects.sort(...)` 的前两个键；提取矩形、识别角点、旋转、堆叠都不变。
# b) 复杂度校准
#    - 保持原来的“两排矩形 + 两条边线”结构，不引入额外对象类型。
#    - 生成器仍然保证每题至少 4 个矩形，这样排序方向反转会形成明显的重排，不会退化成只交换 1 对对象。
#    - 内部矩形允许少量异色像素，避免不同矩形外形相近时输出顺序肉眼不可见。
# c) 隐式假设审查
#    - 角点定位假设: 角落的 marker 与相邻两条线颜色都不同。生成器继续满足。
#    - 蓝线判定假设: 若颜色 1 存在于某条线，则该线视为蓝线；否则较长线为蓝线。生成器继续直接使用 1/2。
#    - 矩形提取假设: 矩形 bbox 内没有背景色。生成器仍只放满矩形，不挖洞。
#    - 排序逻辑里原先默认“升序”无害地写成正向键。变体下如果直接 `reverse=True` 会把 tie-breaker
#      `along_from_corner(...)` 也反转，等于额外改了非 D1 行为；因此这里只反转 `db/dr` 两个主键。
# d) 放置合法性
#    - canonical 网格固定两条矩形带: 行 2-3 与 5-6；相邻矩形之间始终留 2 列背景，且与边线至少隔 1 格。
#    - `_transform` 只做翻转/旋转，不会破坏不重叠和越界性质。
#    - solver 仍然能稳定地把整条 corner row / corner col 视作边线掩码，矩形不会碰到该掩码。
# e) 视觉可观察性
#    - 反转排序方向要能被人看出来，关键是“新输出不能和旧输出相同”。
#    - 生成器因此显式比较降序 solver 与原升序 solver 的输出，只接受二者不同的样本。
#    - 同时保留少量不对称内部纹理，保证即使矩形尺寸接近，前后顺序交换也能在输出中看见。
# f) 可视化整洁度
#    - 仍使用单一背景色、两条整齐边线、两排规整矩形，没有叠压元素。
#    - 异色点数量上限很低，只用于区分矩形，不会让局部变成噪声块。
# g) 颜色变化可推断性
#    - 颜色不承担新的逻辑角色，仍只是区分边线、marker 和矩形内容。
#    - 不引入新的颜色映射规则，因此不会增加额外归纳负担。
# h) source_task
#    - 该变体没有 source_task，需要跳过参考实现阅读。
# i) 新 solver 的额外假设审查
#    - 新增的唯一假设是“样本中至少存在一个矩形，使得降序与升序堆叠结果不同”；否则变体不可观察。
#      `generate_input()` 通过比较 `_solve(..., descending=True/False)` 的输出显式保证这一点。


def _parse_task(grid: np.ndarray) -> dict:
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
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": arr.copy()})

    assert rects
    return {
        "grid": g,
        "bg": bg,
        "corner": (cr, cc),
        "marker": marker,
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "rects": rects,
    }


def _dist_to_edge(rect: dict, orient: str, edge_idx: int) -> int:
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


def _along_from_corner(rect: dict, orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _solve(grid: np.ndarray, *, descending: bool) -> np.ndarray:
    info = _parse_task(grid)
    bg = int(info["bg"])
    cr, cc = map(int, info["corner"])
    blue_orient = str(info["blue_orient"])
    red_orient = str(info["red_orient"])
    rects = [dict(rct) for rct in info["rects"]]

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rct in rects:
        rct["db"] = int(_dist_to_edge(rct, blue_orient, blue_edge_idx))
        rct["dr"] = int(_dist_to_edge(rct, red_orient, red_edge_idx))

    if descending:
        rects.sort(
            key=lambda r: (
                -int(r["db"]),
                -int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            )
        )
    else:
        rects.sort(
            key=lambda r: (
                int(r["db"]),
                int(r["dr"]),
                _along_from_corner(r, blue_orient, cr, cc),
                _along_from_corner(r, red_orient, cr, cc),
            )
        )

    seq = []
    for rct in rects:
        arr = rct["arr"]
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
    """
    Reorder and stack rectangles by distance to two colored edge lines.

    Variant change:
    - Sort rectangles by (distance to blue edge, distance to red edge) in descending order.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    - Stack top-to-bottom, centered horizontally.
    """
    return _solve(grid, descending=True)


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
        lo = max(0, remaining - 3 * slots_left)
        hi = min(3, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take
    return [2 + 2 * int(u) for u in units]


def _sprinkle_minorities(patch: np.ndarray, dominant: int) -> np.ndarray:
    area = int(patch.size)
    if area < 8 or np.random.rand() < 0.6:
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

    used_signatures: set[tuple[int, ...]] = set()
    color_pool = np.random.permutation(DOMINANT_COLORS)
    color_idx = 0

    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            if color_idx < len(color_pool):
                color = int(color_pool[color_idx])
                color_idx += 1
            else:
                color = int(np.random.choice(DOMINANT_COLORS))

            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)

            signature = tuple(int(v) for v in patch.reshape(-1))
            if signature in used_signatures:
                minority_choices = [int(v) for v in MINORITY_COLORS if int(v) != color]
                patch[0, -1] = int(np.random.choice(minority_choices))
                signature = tuple(int(v) for v in patch.reshape(-1))
            used_signatures.add(signature)

            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    try:
        new_out = _solve(grid, descending=True)
        old_out = _solve(grid, descending=False)
    except Exception:
        return None

    if np.array_equal(new_out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 D1 descending-distance variant")

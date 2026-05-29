# Variant of 291dc1e1
# Dimension: D2 — 排序消歧规则 (match_cond)
# Change: 主键为到蓝线距离，次键为到红线距离，再次键为沿蓝线方向坐标和沿红线方向坐标...
#   → 次键使用沿蓝线方向的坐标（忽略红线距离）

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D2 variant
# a) 具体化与行为追踪:
#    原题输入是“角点标记 + 两条边线 + 内部若干实心矩形”。原 solver 先抽取矩形，再按
#    (到蓝线距离, 到红线距离, 沿蓝线坐标, 沿红线坐标) 排序，最后在输出中纵向堆叠。
#    这个变体只改同一 blue-band 内的消歧方式: 先按到蓝线的层带分组，再按沿蓝线的位置排。
#    但在原几何里，若“沿蓝线位置”仍从角点侧量起，它与“到红线距离”是仿射等价的:
#      例如角点在左上且蓝线水平时，dr = c0 - 1，而沿蓝线位置 = c0。
#    这样 D2 会退化成与原题同序。为了让变体可观察，这里把“沿蓝线坐标”具体化为
#    “从蓝线远端量起的坐标”，即同一 band 内按远离角点的一端优先。例:
#      同一 band 中三个矩形从左到右为 A,B,C。
#      原题: 先 A 再 B 再 C（按到红线距离/靠角点侧）。
#      变体: 先 C 再 B 再 A（按沿蓝线远端坐标）。
#    这样仍然只依赖蓝线方向位置，不再依赖 red distance，而且像素上能直接看出 band 内顺序翻转。
#
# b) 复杂度校准:
#    仍保留原题的两个 blue-band 与 2~5 个矩形/带的规模，不引入额外对象类型。
#    为了让 band 内重排有支撑，生成器保证每个 band 至少有 2 个矩形，并常见 3+ 个。
#    矩形内部允许少量异色点，使它们在输出中更容易区分顺序。
#
# c) 隐式假设审查:
#    - 角点仍然必须在四角之一；corner 检测逻辑继续成立。
#    - 边线仍是角点所在整行/整列；line_mask 继续可用。
#    - 目标对象仍是 bbox 内无背景的实心矩形；components + bbox 过滤继续可用。
#    - 变体相关的关键旧假设是“dr 提供了有效的同 band 次序”；现在不再依赖它。
#      因此 sort key 中删除 dr，改为沿蓝线远端坐标作为次键，并保留沿红线坐标只作最终稳定排序。
#
# d) 放置合法性:
#    生成器在 canonical 形态下只在两条固定 band 行上从左到右放置矩形，矩形间至少隔 2 列，
#    因此不重叠、不越界；再通过翻转/旋转映射到四种角点与两种蓝线朝向。
#    映射不会改变对象之间的不重叠关系，solver 仍能稳定抽取。
#
# e) 视觉可观察性:
#    若仍按原 generator，band 内对象本来就是左到右排列；本变体会把它们在输出中的 band 内相对顺序反转。
#    由于矩形颜色、宽度、内部少数异色点都可能不同，解题者从 2~3 组样例即可看到
#    “同一离蓝线层带里的对象按蓝线方向位置重排，而不是按另一条线的近远重排”。
#
# f) 可视化整洁度:
#    输入仍是规整的两行矩形条带，边线与对象分层清楚；输出仍是居中纵向堆叠。
#    不引入重叠标记或额外杂点区域，视觉结构保持干净。
#
# g) 颜色变化可推断性:
#    本变体不改颜色规则。颜色只用于区分对象，少量 minority 像素仅增强对象可辨识度，
#    不承担额外语义，因此不会制造新的颜色推断负担。


def _find_corner_and_orients(g: np.ndarray, bg: int) -> tuple[tuple[int, int], int, str, str]:
    H, W = g.shape
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
    return corner, marker, blue_orient, red_orient


def _extract_rects(g: np.ndarray, bg: int, corner: tuple[int, int], marker: int) -> list[dict]:
    H, W = g.shape
    cr, cc = map(int, corner)
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
    return rects


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


def _along_from_corner(rect: dict, orient: str, corner: tuple[int, int]) -> int:
    cr, cc = map(int, corner)
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _along_from_far_end(rect: dict, orient: str, corner: tuple[int, int]) -> int:
    # The variant's concrete secondary key:
    # within the same blue-distance band, order by the coordinate along the blue axis
    # measured from the far end, not by distance to the red edge.
    cr, cc = map(int, corner)
    r0, c0, r1, c1 = map(int, rect["box"])
    if orient == "v":
        return int(-r1 if cr == 0 else r0)
    return int(-c1 if cc == 0 else c0)


def _ordered_rects(grid: np.ndarray, *, variant: bool) -> tuple[list[dict], int, str]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    corner, marker, blue_orient, red_orient = _find_corner_and_orients(g, bg)
    cr, cc = map(int, corner)
    rects = _extract_rects(g, bg, corner, marker)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect, red_orient, red_edge_idx))
        rect["ab_corner"] = int(_along_from_corner(rect, blue_orient, corner))
        rect["ab_far"] = int(_along_from_far_end(rect, blue_orient, corner))
        rect["ar_corner"] = int(_along_from_corner(rect, red_orient, corner))

    if variant:
        rects.sort(key=lambda r: (int(r["db"]), int(r["ab_far"]), int(r["ar_corner"])))
    else:
        rects.sort(
            key=lambda r: (
                int(r["db"]),
                int(r["dr"]),
                int(r["ab_corner"]),
                int(r["ar_corner"]),
            )
        )
    return rects, bg, blue_orient


def _stack_rects(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
    seq = []
    for rect in rects:
        arr = rect["arr"]
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


def main(grid: np.ndarray) -> np.ndarray:
    rects, bg, blue_orient = _ordered_rects(grid, variant=True)
    return _stack_rects(rects, bg, blue_orient)


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
    if area < 8 or np.random.rand() < 0.45:
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
    options = []
    while not options:
        long_dim = int(np.random.randint(12, 29))
        options = []
        for trailing in (1, 2):
            max_used = long_dim - 2 - trailing
            if max_used < 10 or max_used % 2:
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
            color = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), color, dtype=int)
            patch = _sprinkle_minorities(patch, color)
            paste(grid, patch, r0, c)
            c += w + 2

    return grid


def _variant_is_visible(grid: np.ndarray) -> bool:
    new_rects, _, _ = _ordered_rects(grid, variant=True)
    old_rects, _, _ = _ordered_rects(grid, variant=False)
    if len(new_rects) != len(old_rects):
        return False

    by_band: dict[int, int] = {}
    for rect in new_rects:
        db = int(rect["db"])
        by_band[db] = by_band.get(db, 0) + 1
    if max(by_band.values()) < 2:
        return False

    new_boxes = [tuple(map(int, rect["box"])) for rect in new_rects]
    old_boxes = [tuple(map(int, rect["box"])) for rect in old_rects]
    return new_boxes != old_boxes


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
    raise RuntimeError("failed to generate input for 291dc1e1 D2 variant")

# Variant of 291dc1e1
# Dimension: D1 — 排序主键——距离度量方式 (match_cond)
# Change: 按 (到蓝线的距离, 到红线的距离) 升序排序，距离为矩形边缘到边线的像素间隔
#   → 按矩形位置行列坐标直接排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the D1 variant
# a) 具体化与行为追踪:
#    输入仍是一个带角点标记的“L”形边线系统，内部有若干无背景色空洞的矩形块；输出仍是把这些矩形
#    纵向堆叠，且当蓝线是垂直方向时做 90 度逆时针旋转。唯一核心变化发生在“矩形排序”阶段：
#    原题把每个矩形映射到“离蓝线多远、离红线多远”，这是一个以角点为原点的局部坐标系；
#    本变体改为直接读取矩形 bbox 左上角 `(r0, c0)`，按绝对网格坐标排序。
#    具体例子：若角点在右上角，蓝线水平在顶边、红线垂直在右边，某一行上从左到右有 A/B/C 三块。
#    原题排序键是 `(距顶边, 距右边)`，因此同一行会按 C,B,A 排；本变体键是 `(r0, c0)`，同一行按
#    A,B,C 排。差异只出现在排序阶段，旋转与堆叠完全沿用原题。
# b) 复杂度校准:
#    若 generator 仍只产生“左上角 + 蓝线水平”的样本，新旧排序会退化一致，看不出变体。为避免过度
#    简化，生成器保留原先多角点/多朝向随机化，并显式拒绝那些“按旧排序渲染后与新排序输出相同”的样本。
#    这样仍是单一规则，但每组样例里都能重复支撑“排序依据变了”。
# c) 隐式假设审查:
#    - 角点识别逻辑只依赖“拐角处有唯一 marker 且相邻两条边线颜色不同”，与 D1 无关，仍成立。
#    - 蓝线/红线判定只服务于“蓝线垂直则旋转”，与旧距离排序解耦，仍成立。
#    - 原 `dist_to_edge` / `along_from_corner` 假设排序坐标必须相对边线测量；在变体中该假设失效，
#      因而整段逻辑必须替换为直接读 bbox 左上角。
#    - 提取矩形时假设每个连通块的 bbox 内没有背景色；生成器继续只放置实心彩色块，因此成立。
# d) 放置合法性:
#    生成器沿用原先两条边线 + 两排矩形带的构造，矩形之间至少隔 2 列，不会互相接触，也不会越界。
#    变换到四个角时仅做翻转/旋转，合法性保持。solver 的组件提取因此稳定。
# e) 视觉可观察性:
#    仅改排序键时，最容易观察的分叉是“右侧角点时同一排的左右顺序反转”或“底部角点时上下排顺序反转”。
#    生成器额外给每个矩形加一个轻微、不对称的内部标记，使得顺序变化和旋转变化都能被肉眼追踪。
# f) 可视化整洁度:
#    仍保持清晰的 L 形边线、统一背景、矩形之间固定空隙；内部标记只占 1-2 个像素，不会让区域混杂。
# g) 颜色变化可推断性:
#    颜色不承载新规则；它只帮助区分对象。边线继续固定为蓝/红，背景固定，矩形颜色来自有限调色板，
#    不引入额外颜色映射任务。
# h) 变体量化指标:
#    生成器用“同一输入上，新排序输出是否与旧排序输出不同”作为硬指标；只有 `new_output != old_output`
#    的样本才会被接受，因此 D1 的控制变量一定实际生效。
# i) main() 的新隐式假设:
#    新 solver 假设至少有一个矩形，并且每个矩形都能用 bbox 左上角唯一地落到一个排序位置。
#    若出现相同 `(r0, c0)` 的两个矩形，它们必然重叠，不可能由生成器产生；生成器通过带间距放置保证此点。


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)


def _find_corner_and_orients(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    H, W = map(int, g.shape)
    bg = int(get_bg(g))

    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == bg:
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

    cr, cc = corner
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
        while 0 <= c < W and int(g[cr, c]) == hcol:
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == vcol:
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
    return bg, corner, marker, blue_orient, red_orient


def _extract_rects(g: np.ndarray, bg: int, corner: tuple[int, int], marker: int) -> list[dict]:
    H, W = map(int, g.shape)
    cr, cc = map(int, corner)
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
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


def _position_sort_key(rect: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = map(int, rect["box"])
    return int(r0), int(c0)


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


def _original_sort_key(rect: dict, blue_orient: str, red_orient: str, corner: tuple[int, int]) -> tuple[int, int, int, int]:
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


def _render_sequence(rects: list[dict], bg: int, blue_orient: str) -> np.ndarray:
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
    """
    Reorder and stack rectangles by their absolute top-left coordinates.

    Input structure and output layout match the original task; only the ranking key changes.
    """
    g = np.array(grid, dtype=int)
    bg, corner, marker, blue_orient, _red_orient = _find_corner_and_orients(g)
    rects = _extract_rects(g, bg, corner, marker)
    rects.sort(key=_position_sort_key)
    return _render_sequence(rects, bg, blue_orient)


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


def _decorate_patch(patch: np.ndarray, dominant: int, tag: int) -> np.ndarray:
    out = patch.copy()
    accent_choices = [int(c) for c in ACCENT_COLORS if int(c) != dominant]
    accent_a = int(accent_choices[tag % len(accent_choices)])
    accent_b = int(accent_choices[(tag + 3) % len(accent_choices)])
    mode = int(tag % 4)

    if mode == 0:
        out[0, 0] = accent_a
        if out.shape[1] >= 4:
            out[-1, -1] = accent_b
    elif mode == 1:
        out[-1, 0] = accent_a
        if out.shape[1] >= 4:
            out[0, -1] = accent_b
    elif mode == 2:
        out[0, -1] = accent_a
        if out.shape[1] >= 4:
            out[-1, 0] = accent_b
    else:
        out[-1, -1] = accent_a
        if out.shape[1] >= 4:
            out[0, 0] = accent_b
    return out


def _build_canonical() -> np.ndarray:
    long_dim = int(np.random.randint(11, 29))
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

    used_tags = 0
    for r0, widths in ((2, band0), (5, band1)):
        c = 2
        for w in widths:
            dominant = int(np.random.choice(DOMINANT_COLORS))
            patch = np.full((2, w), dominant, dtype=int)
            patch = _decorate_patch(patch, dominant, used_tags)
            paste(grid, patch, r0, c)
            used_tags += 1
            c += w + 2

    return grid


def _old_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, corner, marker, blue_orient, red_orient = _find_corner_and_orients(g)
    rects = _extract_rects(g, bg, corner, marker)
    rects = sorted(rects, key=lambda rect: _original_sort_key(rect, blue_orient, red_orient, corner))
    return _render_sequence(rects, bg, blue_orient)


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    choices = [
        ("h", "tl"),
        ("h", "tr"),
        ("h", "bl"),
        ("h", "br"),
        ("v", "tl"),
        ("v", "tr"),
        ("v", "bl"),
        ("v", "br"),
    ]
    blue_orient, corner = choices[int(np.random.randint(len(choices)))]
    grid = _transform(canonical, blue_orient, corner)

    new_out = main(grid)
    old_out = _old_output(grid)
    if np.array_equal(new_out, old_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate non-degenerate D1 variant input for 291dc1e1")

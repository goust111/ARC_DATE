# Combo variant of 20270e3b (index 26)
# combo_spec: D4/0+D9/2+D15/2
# Name: 内部T形连接器嵌入
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题是“两个水平条连接器上下对接”；这里改成“两个同形 T/X 连接器完全重叠，plug 从右侧滑入
#      base 的开口凹槽，除连接器外的 plug 像素全部填入 base 内部的 socket”。
#    - 连接器不再是一行横条，而是固定少量模板中的一个：T4 / T5 / X5。solver 需要比较整个 2D ctrl 形状，
#      不能只看长度。
#    - D15 的“内部连接器”在本融合里采用横向解释：两个精灵的连接器都不贴左右边界，连接器 bbox 左右两侧
#      都能看到 body 像素。这样失去原题“朝上/朝下自由侧”后，观察者会自然转向侧向插接。
#    - 小例子：原题中 plug 会被平移到 base 上方 1 行，输出整体向上/下扩张；本题中 plug 的 ctrl 与 base 的
#      ctrl 在同一位置完全重合，plug 的其它像素进入 base 的右侧凹槽，输出 bbox 与 base bbox 相同，只是
#      凹槽被填满、ctrl 统一改回 body 色。
# b) 复杂度校准
#    - 连接器模板只保留 3 种，方向固定，不做旋转镜像；这样 2-4 个样例里会出现重复支撑，不会把规则自由度
#      放得过大。
#    - plug 不是简单矩形，而是围绕连接器生成一条横向主体并带上下突起；base 的 socket 与之精确匹配且开口
#      通向右边界。这样视觉上明显不是原题的上下拼接，也不会退化成“只看一个中心点就能塞进去”。
# c) 隐式假设审查
#    - 原 `tr = (dr0 - 1) - pr0` 依赖“plug connector 在 base connector 上方 1 行”，这里失效，改为
#      `tr = dr0 - pr0`, `tc = dc0 - pc0`，即两个 2D 连接器完全重合。
#    - 原 overlap 逻辑默认只有拼接缝附近少量 ctrl/obj 接触；这里改为：唯一允许重叠的是 ctrl-vs-ctrl 的
#      连接器区域，其余 plug 像素必须落在 base 的 bg socket 上。
#    - 原题 tight crop 后常比 base 更高或更宽；这里要求 union bbox 与 base bbox 完全一致，说明是嵌入而非
#      外挂。
# d) 放置合法性
#    - generator 采用 output-first：先造 docked 状态中的 plug，再用它反推出带右侧开口凹槽的 base。
#    - 输入画布上 base 固定在原点，plug 以 side/stack 两种方式远离放置，保持 8 邻域分离且 bbox gap 为 1-5。
# e) 视觉可观察性
#    - 人类会看到：base 内部不是封闭洞，而是右边缘开口凹槽；plug 的橙色 T/X 标记埋在形状内部，输出里该
#      标记重染成黄体色且凹槽被填满。和原题相比，拼接方向与 connector 形态都明显变化。
# f) 可视化整洁度
#    - 只有一个 ctrl 层，且最终完全重着色；socket 仅由 bg 构成，不引入额外标记色。
#    - overlap 仅发生在 ctrl 区域，避免 plug/base 多层实体像素叠加导致画面混乱。
# g) 颜色变化可推断性
#    - 仍然只用 bg=1, obj=4, ctrl=7 三色；输出唯一颜色变化仍是 ctrl -> obj。
# h) main() 的新增假设及 generator 对应保证
#    - 两个前景分量恰好各自含一个同形 T/X 连接器；generator 固定模板并保证匹配。
#    - 连接器左右两侧都有 body；generator 的 plug/body 与 base/shell 都显式检查该条件。
#    - plug 在 docked 状态下完全包含于 base bbox 内，且至少有一个 plug body 像素触及 base 右边界，
#      体现“侧面嵌入”；generator 构造时固定 plug 贴 output 右侧。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7

T4 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
T5 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})
X5 = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
CONNECTOR_SHAPES = (T4, T5, X5)


def _component_points(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(cid))}
        if pts:
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    nz = np.argwhere(grid != int(BG))
    if nz.size == 0:
        return np.full((0, 0), int(BG), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _normalize_points(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (r0, c0)


def _shape_hw(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return int(max(r for r, _c in shape) + 1), int(max(c for _r, c in shape) + 1)


def _connector_points(sprite: set[tuple[int, int]], grid: np.ndarray) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(CTRL)}


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        out[rr, cc] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _has_body_on_both_sides(body: set[tuple[int, int]], conn: set[tuple[int, int]]) -> bool:
    _shape, (_r0, c0) = _normalize_points(conn)
    c1 = int(max(c for _r, c in conn))
    return any(int(c) < c0 for _r, c in body) and any(int(c) > c1 for _r, c in body)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return int(_gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1))


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    comps = _component_points(g != int(BG))
    assert len(comps) == 2
    a, b = comps

    conn_a = _connector_points(a, g)
    conn_b = _connector_points(b, g)
    assert conn_a and conn_b

    shape_a, origin_a = _normalize_points(conn_a)
    shape_b, origin_b = _normalize_points(conn_b)
    assert shape_a == shape_b
    assert shape_a in CONNECTOR_SHAPES

    bbox_a = _bbox_of_pts(a)
    bbox_b = _bbox_of_pts(b)
    assert int(bbox_a[0] + bbox_a[1]) == 0 or int(bbox_b[0] + bbox_b[1]) == 0
    if int(bbox_a[0] + bbox_a[1]) <= int(bbox_b[0] + bbox_b[1]):
        base, plug = a, b
        conn_base, conn_plug = conn_a, conn_b
        origin_base, origin_plug = origin_a, origin_b
    else:
        base, plug = b, a
        conn_base, conn_plug = conn_b, conn_a
        origin_base, origin_plug = origin_b, origin_a

    base_body = base - conn_base
    plug_body = plug - conn_plug
    assert _has_body_on_both_sides(base_body, conn_base)
    assert _has_body_on_both_sides(plug_body, conn_plug)

    tr = int(origin_base[0] - origin_plug[0])
    tc = int(origin_base[1] - origin_plug[1])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    plug_conn_shift = {(int(r + tr), int(c + tc)) for r, c in conn_plug}
    assert plug_conn_shift == conn_base

    br0, bc0, br1, bc1 = _bbox_of_pts(base)
    rr0, cc0, rr1, cc1 = _bbox_of_pts(base | plug_shift)
    assert (rr0, cc0, rr1, cc1) == (br0, bc0, br1, bc1)

    plug_body_shift = plug_shift - plug_conn_shift
    assert plug_body_shift
    assert all(br0 <= r <= br1 and bc0 <= c <= bc1 for r, c in plug_body_shift)
    assert any(int(c) == bc1 for _r, c in plug_body_shift)

    for r, c in plug_body_shift:
        assert (r, c) not in base
    for r, c in plug_conn_shift:
        assert (r, c) in base
        assert int(g[int(r), int(c)]) == int(CTRL)

    return {
        "grid": g,
        "base": base,
        "plug": plug,
        "conn_base": conn_base,
        "plug_shift": plug_shift,
        "shift": (tr, tc),
        "base_bbox": (br0, bc0, br1, bc1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlap the matching internal T/X connectors, fill the side socket, recolor ctrl -> obj.
    """
    info = _analyze(np.array(grid, dtype=int))
    g = info["grid"]
    base = info["base"]
    plug_shift = info["plug_shift"]
    tr, tc = info["shift"]
    br0, bc0, br1, bc1 = info["base_bbox"]
    conn_base = info["conn_base"]

    out = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(BG), dtype=int)
    for r, c in base:
        out[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        src_r, src_c = int(r - tr), int(c - tc)
        val = int(g[src_r, src_c])
        cur = int(out[int(r - br0), int(c - bc0)])
        if cur != int(BG) and val != int(BG):
            assert (r, c) in conn_base
            assert cur == int(CTRL) and val == int(CTRL)
        else:
            assert cur == int(BG) or (r, c) in conn_base
        out[int(r - br0), int(c - bc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out)


def _build_plug(shape: frozenset[tuple[int, int]]) -> np.ndarray | None:
    shape_h, shape_w = _shape_hw(shape)
    h = int(shape_h + 2)
    w = 5
    if h > 5:
        return None

    cr0, cc0 = 1, 1
    body = np.zeros((h, w), dtype=bool)
    corridor_row = int(cr0 + min(1, shape_h - 1))

    top_end = _rng.choice([3, 4])
    bottom_start = _rng.choice([0, 1])
    post_top = int(max(0, corridor_row - 1))
    post_bot = int(min(h - 1, corridor_row + 1))

    body[corridor_row, :] = True
    body[0, 0 : top_end + 1] = True
    body[h - 1, bottom_start:w] = True
    body[post_top : post_bot + 1, 0] = True
    body[post_top : post_bot + 1, w - 1] = True

    if _rng.random() < 0.5:
        extra_col = _rng.choice([1, 3])
        body[1 : h - 1, extra_col] = True
    else:
        extra_row = _rng.choice([0, h - 1])
        body[extra_row, :] = True

    for dr, dc in shape:
        body[int(cr0 + dr), int(cc0 + dc)] = False

    grid = np.full((h, w), int(BG), dtype=int)
    grid[body] = int(OBJ)
    for dr, dc in shape:
        grid[int(cr0 + dr), int(cc0 + dc)] = int(CTRL)

    if not is_contiguous(grid != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(grid != int(BG))) < 5 or int(np.sum(grid != int(BG))) > 29:
        return None

    pts = {(int(r), int(c)) for r, c in np.argwhere(grid != int(BG))}
    conn = {(int(r), int(c)) for r, c in np.argwhere(grid == int(CTRL))}
    body_pts = pts - conn
    if not _has_body_on_both_sides(body_pts, conn):
        return None
    if not any(int(c) == w - 1 for _r, c in body_pts):
        return None
    return _crop_non_bg(grid)


def _build_base_and_expected(plug_sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ph, pw = plug_sprite.shape
    h = int(ph + 2)
    w = int(pw + 1)
    if not (5 <= h <= 8 and 3 <= w <= 12):
        return None

    plug_full = np.zeros((h, w), dtype=bool)
    plug_body = np.zeros((h, w), dtype=bool)
    conn_mask = np.zeros((h, w), dtype=bool)
    pr0, pc0 = 1, 1

    for r, c in np.argwhere(plug_sprite != int(BG)):
        rr, cc = int(pr0 + r), int(pc0 + c)
        plug_full[rr, cc] = True
        if int(plug_sprite[int(r), int(c)]) == int(CTRL):
            conn_mask[rr, cc] = True
        else:
            plug_body[rr, cc] = True

    shell = _dilate4(plug_full) & (~plug_body) & (~conn_mask)
    shell[0, 0] = True

    base = np.full((h, w), int(BG), dtype=int)
    base[shell] = int(OBJ)
    base[conn_mask] = int(CTRL)

    if not is_contiguous(base != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(base != int(BG))) < 5 or int(np.sum(base != int(BG))) > 29:
        return None

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base != int(BG))}
    conn_pts = {(int(r), int(c)) for r, c in np.argwhere(base == int(CTRL))}
    base_body = base_pts - conn_pts
    if not _has_body_on_both_sides(base_body, conn_pts):
        return None
    if not bool(np.any(base[:, w - 1] == int(OBJ))):
        return None
    if not bool(np.any((base[:, w - 1] == int(BG)) & plug_body[:, w - 1])):
        return None

    expected = np.array(base, copy=True)
    expected[plug_body] = int(OBJ)
    expected[expected == int(CTRL)] = int(OBJ)
    if not is_contiguous(expected != int(BG), background=0, connectivity=8):
        return None
    return _crop_non_bg(base), _crop_non_bg(expected)


def _place_on_canvas(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    layouts: list[tuple[str, int, int]] = []
    side_min_w = int(bw + 1 + pw)
    if side_min_w <= 13:
        min_h = int(max(7, bh, ph))
        if min_h <= 11:
            layouts.append(("side", min_h, side_min_w))

    stack_min_h = int(bh + 1 + ph)
    if stack_min_h <= 11:
        min_w = int(max(7, bw, pw))
        if min_w <= 13:
            layouts.append(("stack", stack_min_h, min_w))

    if not layouts:
        return None

    mode, min_h, min_w = _rng.choice(layouts)
    if mode == "side":
        canvas_h = _rng.randint(min_h, 11)
        canvas_w = _rng.randint(min_w, 13)
        gap = _rng.randint(1, min(5, canvas_w - (bw + pw)))
        pr = _rng.randint(0, min(canvas_h - ph, bh - 1))
        pc = int(bw + gap)
    else:
        canvas_h = _rng.randint(min_h, 11)
        canvas_w = _rng.randint(min_w, 13)
        gap = _rng.randint(1, min(5, canvas_h - (bh + ph)))
        pr = int(bh + gap)
        pc = _rng.randint(0, min(canvas_w - pw, bw - 1))

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=BG)

    comps = _component_points(canvas != int(BG))
    if len(comps) != 2:
        return None
    b0 = _bbox_of_pts({(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(BG))})
    b1 = _bbox_of_pts({(int(r + pr), int(c + pc)) for r, c in np.argwhere(plug_sprite != int(BG))})
    if not (1 <= _bbox_gap(b0, b1) <= 5):
        return None
    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    shape = _rng.choice(CONNECTOR_SHAPES)
    plug_sprite = _build_plug(shape)
    if plug_sprite is None:
        return None

    built = _build_base_and_expected(plug_sprite)
    if built is None:
        return None
    base_sprite, expected = built

    canvas = _place_on_canvas(base_sprite, plug_sprite)
    if canvas is None:
        return None

    try:
        got = main(canvas)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")

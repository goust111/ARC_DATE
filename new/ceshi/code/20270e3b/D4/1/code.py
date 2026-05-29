# Variant of 20270e3b
# Dimension: D4 — 目标对接方式 (dst)
# Change: plug精灵平移到base精灵上方，连接器行相邻（plug连接器在base连接器正上方1行），直接拼接...
#   → 对接后保持间距；两者共享同一条 ctrl 连接器行作为可见接缝，输出中保留 ctrl 颜色。
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题中，base 的 connector 位于其 body 下边界，plug 的 connector 位于其 body 上边界。
#      对接时把 plug connector 平移到 base connector 正上方 1 行，于是会发生两层重叠：
#      row r-1: plug ctrl 覆盖 base obj；row r: plug obj 覆盖 base ctrl。最后再把所有 ctrl 改回 obj，
#      所以接口完全“消失”。
#    - 本变体把目标改成“保留可见接缝”。具体做法是：把两条 connector 对齐到同一行，
#      该行成为 1 像素厚的 ctrl seam。base body 停在 seam 上方，plug body 从 seam 下方开始，
#      所以最终输出里仍然只有一条水平 ctrl 条带，肉眼可见地把上下两块隔开。
#    - 小例子：若 base connector 在第 4 行、plug connector 在自身第 0 行，
#      原题位移为 tr = 4-1-0 = 3；本题改成 tr = 4-0 = 4。差异正好是多下移 1 行，
#      原先会覆盖掉的那条 seam 现在被保留为 ctrl。
# b) 复杂度校准
#    - 仍然只有 2 个主精灵、1 组匹配连接器、1 种对接方向，推理复杂度与原题同量级。
#    - 为了让变体稳定可见，输出必须保留 ctrl，并且 generator 保证 seam 长度在 1-3 之间，
#      让人能从少量样例中看到“并不是简单并色”，而是“接口被保留了”。
# c) 隐式假设审查
#    - 原 `tr = (dr0 - 1) - pr0` 依赖“plug connector 必须在 base connector 上方 1 行”；
#      变体中这个前提失效，改为 `tr = dr0 - pr0`。
#    - 原 overlap 断言 `cur == ctrl or val == ctrl` 允许 ctrl/obj 交叠，因为原题本来要把 ctrl 抹掉。
#      变体中 seam 要保留，因此只允许 connector 行上的 ctrl/ctrl 重叠，其他任何重叠都视为非法。
#    - 原 `out[out == ctrl] = obj` 依赖“输出中不应出现 ctrl”；变体中必须删除该步。
# d) 放置合法性
#    - 输入阶段仍把 base 放在画布左上角，并把 plug 随机放到别处。
#    - generator 在 assembled 状态下检查：除 seam 行上完全对齐的 ctrl/ctrl 外，不得出现任何重叠；
#      plug 也不得伸到 base 左侧，从而保证 solver 的 `tight crop` 后 base 仍位于输出左上角。
#    - 画布随机放置使用 `padding=1`，避免 8 邻接下对角接触导致两个输入组件意外并成一块。
# e) 视觉可观察性
#    - 原题输出只有 bg+obj；本变体输出固定还会出现 ctrl seam。这是最直接的可见差异。
#    - seam 是规则中心，不依赖颜色编号排序，也不依赖隐藏数值。
# f) 可视化整洁度
#    - 只保留一条 1 像素厚的水平 seam，不引入额外碎标记；其余结构仍保持原题的紧凑裁剪。
# g) 颜色变化可推断性
#    - 颜色集合仍固定为 bg=1, obj=4, ctrl=7。
#    - ctrl 只承担一个功能：标记 docking seam；不会出现多重颜色映射。

from __future__ import annotations

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, pts: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out


def _connector_points(sprite: np.ndarray, ctrl: int, bg: int) -> set[tuple[int, int]]:
    pts = set()
    for r, c in np.argwhere(sprite != bg):
        if int(sprite[int(r), int(c)]) == int(ctrl):
            pts.add((int(r), int(c)))
    return pts


def _normalize_connector(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    assert conn_pts
    rs = sorted({int(r) for r, _c in conn_pts})
    assert len(rs) == 1
    cs = sorted(int(c) for _r, c in conn_pts)
    assert cs == list(range(cs[0], cs[-1] + 1))
    r0, c0 = int(rs[0]), int(cs[0])
    shape = {(0, int(c - c0)) for c in cs}
    return shape, (r0, c0)


def _dock_cropped(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray:
    bg, obj, ctrl = 1, 4, 7

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != bg)}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != bg)}
    assert base_pts and plug_pts

    conn_base = _connector_points(base_sprite, ctrl, bg)
    conn_plug = _connector_points(plug_sprite, ctrl, bg)
    shape_base, (dr0, dc0) = _normalize_connector(conn_base)
    shape_plug, (pr0, pc0) = _normalize_connector(conn_plug)
    assert shape_base == shape_plug

    base_body = base_pts - conn_base
    plug_body = plug_pts - conn_plug
    for r, c in conn_base:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_plug:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    # Variant seam: both connectors align on the same output row and remain ctrl.
    tr = int(dr0 - pr0)
    tc = int(dc0 - pc0)

    shifted_plug_pts = {(int(r + tr), int(c + tc)) for r, c in plug_pts}
    pts_all = base_pts | shifted_plug_pts
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base_pts:
        out[int(r), int(c)] = int(base_sprite[int(r), int(c)])

    for r, c in plug_pts:
        rr = int(r + tr)
        cc = int(c + tc)
        val = int(plug_sprite[int(r), int(c)])
        cur = int(out[int(rr), int(cc)])
        if cur != int(bg):
            assert cur == int(ctrl) and val == int(ctrl)
        out[int(rr), int(cc)] = int(val)

    nz = np.argwhere(out != bg)
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites by matching their horizontal connectors and preserve the
    connector color as a visible seam in the output.
    """
    g = np.array(grid, dtype=int)
    bg, _obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    connA = {(r, c) for (r, c) in A if int(g[int(r), int(c)]) == int(ctrl)}
    connB = {(r, c) for (r, c) in B if int(g[int(r), int(c)]) == int(ctrl)}
    assert connA and connB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_sprite = _crop_component(g, base_pts, bg)
    plug_sprite = _crop_component(g, plug_pts, bg)
    return _dock_cropped(base_sprite, plug_sprite)


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allow: np.ndarray | None = None) -> None:
    """Grow a 4-connected shape on `mask` from `frontier` up to `target` pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc]:
                continue
            if allow is not None and not bool(allow[nr, nc]):
                continue
            nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove removable interior pixels while preserving 8-connectivity."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != bg)
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_base() -> tuple[np.ndarray, int, list[int]] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 9)
    base_target = _rng.randint(5, min(24, base_h * base_w))

    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates = []
    for r in range(1, base_h):
        for c0 in range(0, base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                continue
            candidates.append((int(r), cols))
    if not candidates:
        return None

    conn_row, conn_cols = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = ctrl

    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != bg)) < 5:
        return None
    return base_grid, int(conn_row), [int(c) for c in conn_cols]


def _gen_plug(conn_len: int, max_left: int, max_width_hint: int) -> tuple[np.ndarray, int] | None:
    bg, obj, ctrl = 1, 4, 7
    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, max_width_hint))
    if int(plug_h * plug_w) < 5:
        return None
    max_conn_c0 = min(int(plug_w - conn_len), int(max_left))
    if max_conn_c0 < 0:
        return None
    plug_conn_c0 = _rng.randint(0, max_conn_c0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    allow = np.ones((plug_h, plug_w), dtype=bool)
    allow[0, :] = False
    allow[0, plug_conn_c0 : plug_conn_c0 + conn_len] = True

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True

    plug_target = _rng.randint(5, min(24, int(plug_h * plug_w)))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))), allow=allow)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    keep = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    keep |= {(1, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl
        if plug_grid[1, c] != obj:
            return None

    if np.any(plug_grid[0, :plug_conn_c0] != bg) or np.any(plug_grid[0, plug_conn_c0 + conn_len :] != bg):
        return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None
    if int(np.sum(plug_grid != bg)) < 5:
        return None
    return plug_grid, int(plug_conn_c0)


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = 1, 7

    base_info = _gen_base()
    if base_info is None:
        return None
    base_grid, _conn_row, conn_cols = base_info
    conn_len = len(conn_cols)

    plug_info = _gen_plug(conn_len, max_left=conn_cols[0], max_width_hint=base_grid.shape[1] + 4)
    if plug_info is None:
        return None
    plug_grid, _plug_conn_c0 = plug_info

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    try:
        expected = _dock_cropped(base_sprite, plug_sprite)
    except Exception:
        return None

    if int(np.sum(expected == ctrl)) != conn_len:
        return None
    if expected.shape[0] < 5 or expected.shape[1] < 3 or expected.shape[0] > 9 or expected.shape[1] > 12:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    canvas_h = _rng.randint(max(7, max(bh, ph) + 1), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")

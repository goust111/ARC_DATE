# Combo variant of 20270e3b (index 11)
# combo_spec: D4/1+D7/2
# Name: 间距对接基准裁剪
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题把 plug connector 平移到 base connector 正上方 1 行。这样会出现两层“互相覆盖”：
#      plug 的 ctrl 覆盖到 base 的 obj，plug 的 obj 又覆盖到 base 的 ctrl，最后全部 ctrl
#      还会被改回 obj，所以接口在输出里消失。
#    - D4/1 的关键变化是把两条 connector 对齐到同一行，让这一整行保留为 ctrl seam。
#      在像素上等价于：base body 结束在 seam 上方，plug body 从 seam 下方开始，中间保留
#      一条 1 像素厚的 ctrl 接缝，不再发生 ctrl/obj 互抹。
#    - D7/2 的关键变化是输出窗口不再跟着“合并后的整体外接框”走，而是固定为 base 原始 bbox。
#      所以 plug 即使在对接后向下或向右伸出，也会被裁掉。
#    - 小例子：若 base bbox 为 5x5，base connector 在第 3 行，plug connector 在其自身第 0 行。
#      原题位移是 tr = 3 - 1 - 0 = 2，融合版改成 tr = 3 - 0 = 3，于是 seam 保留在第 3 行。
#      若平移后 plug 还会落到第 5、6 行，则原题/D4 单变体会输出更高的结果，融合版只保留
#      5x5 的 base 窗口，第 5、6 行直接裁掉。
# b) 复杂度校准
#    - 仍然只有 2 个主精灵、1 组水平 connector、1 个 docking 方向，推理链保持单一。
#    - 视觉上必须同时出现两件事：ctrl seam 保留下来，以及 plug 有一部分被 base 窗口裁掉。
#      generator 因此强制：输出中 ctrl 数量等于 connector 长度，且 dock 后至少有 1 个 plug
#      像素落到 base bbox 外，同时仍保留至少少量 plug body 落在 base 窗口内。
# c) 隐式假设审查
#    - 原 `tr = (dr0 - 1) - pr0` 依赖“connector 行相邻”；融合版失效，必须改为 `tr = dr0 - pr0`。
#    - 原 overlap 允许 ctrl/obj 交叠，因为 connector 最终会被抹掉；融合版里 seam 要保留，
#      所以窗口内的重叠只能是 ctrl/ctrl，同一条 seam 上完全对齐，其他重叠一律非法。
#    - 原题先渲染完整合并体再 tight crop；融合版这个假设失效，必须直接在 base bbox 这个
#      固定窗口里渲染，并显式丢弃窗口外的 plug 像素。
#    - 由于输出窗口固定在 base bbox，solver 新增了“dock 后 plug 不会伸到 base 上方或左侧”
#      的假设；generator 用 connector 左对齐约束来保证这一点。
# d) 放置合法性
#    - 输入画布仍然把 base 放在 (0,0)，plug 随机放到其他位置。
#    - 随机放置使用 `padding=1`，避免 8 邻接下对角接触把两个输入组件意外连成一块。
#    - 生成 assembled 状态时，除 seam 上的 ctrl/ctrl 外，不允许任何重叠；plug 允许向下/向右
#      越出 base bbox，但不允许向上/向左越出。
# e) 视觉可观察性
#    - 输出固定保留一条水平 ctrl seam，这让 D4 的变化可以直接肉眼观察。
#    - 输出外框固定等于 base bbox，且 generator 强制存在被裁掉的 plug 像素，这让 D7 的变化
#      也能在 2-4 个样例中稳定显现，不会退化成原题或 D4 单变体。
# f) 可视化整洁度
#    - 只保留一条 1 像素厚 seam，不引入额外标记；裁剪也只发生在 base 的下/右边界，视觉上
#      仍然是清晰的“上半 base + seam + 下半 plug 被截断”的结构。
# g) 变化的可推断性
#    - 颜色仍然只有 bg=1、obj=4、ctrl=7 三种，ctrl 只承担 seam 标记作用。
#      没有新增颜色映射，推理重点集中在“对齐方式”和“裁剪基准”两件事上。
# h) 量化退化检测
#    - 指标 1：`ctrl_count_out`，要求输出中 ctrl 数量恒等于 connector 长度，证明 seam 被保留。
#    - 指标 2：`outside_count`，要求 dock 后至少 1 个 plug 前景像素落在 base bbox 外，证明确实
#      发生了 base-window clipping，而不是和 D4 单变体输出相同。
# i) main() 的新假设与 generator 对应保证
#    - main() 假设恰有 2 个前景组件，且各自恰有一段同长度的水平 ctrl connector。
#    - main() 假设 base 是 bbox 左上角在输入原点的那个组件。
#    - main() 假设 base connector 上方有 body、下方无 body；plug connector 下方有 body、上方无 body。
#    - main() 假设窗口内重叠只会发生在 ctrl seam；generator 通过 assembled 检查保证这一点。
#    - main() 假设 plug dock 后不会伸到 base bbox 上方/左侧；generator 通过 `plug_conn_c0 <= base_conn_left`
#      保证这一点，并且额外强制有一部分 plug 落在窗口外，从而让融合行为稳定可见。

from __future__ import annotations

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


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _connector_points(sprite: np.ndarray, ctrl: int, bg: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(sprite != int(bg))
        if int(sprite[int(r), int(c)]) == int(ctrl)
    }


def _normalize_connector(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    assert conn_pts
    rows = sorted({int(r) for r, _c in conn_pts})
    assert len(rows) == 1
    cols = sorted(int(c) for _r, c in conn_pts)
    assert cols == list(range(cols[0], cols[-1] + 1))
    r0, c0 = int(rows[0]), int(cols[0])
    shape = {(0, int(c - c0)) for c in cols}
    return shape, (r0, c0)


def _dock_base_window(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray:
    bg, ctrl = 1, 7

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}
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

    tr = int(dr0 - pr0)
    tc = int(dc0 - pc0)
    shifted_plug = {(int(r + tr), int(c + tc)) for r, c in plug_pts}

    bh, bw = map(int, base_sprite.shape)
    out = np.full((int(bh), int(bw)), int(bg), dtype=int)
    for r, c in base_pts:
        out[int(r), int(c)] = int(base_sprite[int(r), int(c)])

    for r, c in plug_pts:
        rr = int(r + tr)
        cc = int(c + tc)
        if not (0 <= rr < bh and 0 <= cc < bw):
            continue
        val = int(plug_sprite[int(r), int(c)])
        cur = int(out[int(rr), int(cc)])
        if cur != int(bg):
            assert cur == int(ctrl) and val == int(ctrl)
        out[int(rr), int(cc)] = int(val)

    if shifted_plug:
        assert min(r for r, _c in shifted_plug) >= 0
        assert min(c for _r, c in shifted_plug) >= 0
    return out


def _docked_stats(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> dict[str, int]:
    bg, obj, ctrl = 1, 4, 7

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}
    conn_base = _connector_points(base_sprite, ctrl, bg)
    conn_plug = _connector_points(plug_sprite, ctrl, bg)
    _shape_base, (dr0, dc0) = _normalize_connector(conn_base)
    _shape_plug, (pr0, pc0) = _normalize_connector(conn_plug)
    tr = int(dr0 - pr0)
    tc = int(dc0 - pc0)
    bh, bw = map(int, base_sprite.shape)

    outside = 0
    right = 0
    bottom = 0
    above = 0
    left = 0
    visible_body_inside = 0
    visible_ctrl_inside = 0
    seam_overlap = 0

    for r, c in plug_pts:
        rr = int(r + tr)
        cc = int(c + tc)
        val = int(plug_sprite[int(r), int(c)])
        if rr < 0:
            above += 1
        if cc < 0:
            left += 1
        if rr >= bh or cc >= bw or rr < 0 or cc < 0:
            outside += 1
            if rr >= bh:
                bottom += 1
            if cc >= bw:
                right += 1
            continue
        if val == int(obj):
            visible_body_inside += 1
        if val == int(ctrl):
            visible_ctrl_inside += 1
        if (rr, cc) in base_pts:
            seam_overlap += 1

    return {
        "outside": int(outside),
        "right": int(right),
        "bottom": int(bottom),
        "above": int(above),
        "left": int(left),
        "visible_body_inside": int(visible_body_inside),
        "visible_ctrl_inside": int(visible_ctrl_inside),
        "seam_overlap": int(seam_overlap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the two horizontal connectors, dock them onto one shared ctrl seam,
    and return only the base sprite's original bbox window.
    """
    g = np.array(grid, dtype=int)
    bg, ctrl = 1, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    connA = {(r, c) for (r, c) in A if int(g[int(r), int(c)]) == int(ctrl)}
    connB = {(r, c) for (r, c) in B if int(g[int(r), int(c)]) == int(ctrl)}
    assert connA and connB
    shapeA, _anchorA = _normalize_connector(connA)
    shapeB, _anchorB = _normalize_connector(connB)
    assert shapeA == shapeB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_sprite = _crop_component(g, base_pts, bg)
    plug_sprite = _crop_component(g, plug_pts, bg)
    return _dock_base_window(base_sprite, plug_sprite)


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allow: np.ndarray | None = None,
) -> None:
    """Grow a 4-connected shape from frontier up to `target` pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs: list[tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if bool(mask[nr, nc]):
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
    """Remove interior pixels while preserving 8-connectivity."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _gen_base() -> tuple[np.ndarray, list[int]] | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(4, 7)
    base_w = _rng.randint(3, 9)
    base_target = _rng.randint(6, min(24, int(base_h * base_w)))

    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(bool(base[r, c]) for c in cols):
                continue
            if not all(bool(base[r - 1, c]) for c in cols):
                continue
            if any(r + 1 < base_h and bool(base[r + 1, c]) for c in cols):
                continue
            candidates.append((int(r), [int(c) for c in cols]))
    if not candidates:
        return None

    conn_row, conn_cols = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != int(bg))) < 5:
        return None
    return base_grid, conn_cols


def _gen_plug(conn_len: int, max_left: int, max_width_hint: int) -> tuple[np.ndarray, int] | None:
    bg, obj, ctrl = 1, 4, 7

    plug_h = _rng.randint(3, 7)
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

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(ctrl)
        if int(plug_grid[1, c]) != int(obj):
            return None

    if np.any(plug_grid[0, :plug_conn_c0] != int(bg)):
        return None
    if np.any(plug_grid[0, plug_conn_c0 + conn_len :] != int(bg)):
        return None
    if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(plug_grid != int(bg))) < 5:
        return None
    return plug_grid, int(plug_conn_c0)


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = 1, 7

    base_info = _gen_base()
    if base_info is None:
        return None
    base_grid, conn_cols = base_info
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
        expected = _dock_base_window(base_sprite, plug_sprite)
    except Exception:
        return None

    stats = _docked_stats(base_sprite, plug_sprite)
    if int(stats["above"]) != 0 or int(stats["left"]) != 0:
        return None
    if int(stats["outside"]) == 0:
        return None
    if int(stats["visible_body_inside"]) < 2:
        return None
    if int(stats["seam_overlap"]) != conn_len:
        return None
    if int(np.sum(expected == int(ctrl))) != conn_len:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(4, max(bw, pw) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(bg))
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=int(bg), padding=1)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=int(bg))

    if int(np.sum(canvas == int(bg))) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")

# Combo variant of 20270e3b (index 7)
# combo_spec: D5/1+D9/1
# Name: 矩形连接器焊接扩散
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题是 1xL 水平连接器；本融合把连接器改成 2x2 或 2x3 的实心矩形块。
#      base 的矩形块贴在身体下沿，plug 的矩形块贴在身体上沿，两者形状完全一致。
#    - 对接时不再用“上移 1 行”的一维条逻辑，而是让 plug 的矩形块整体落到
#      base 矩形块正上方：若 base 连接器在 rows 4..5、cols 2..4，plug 连接器在
#      rows 0..1、cols 1..3，则平移量是 tr = 4 - 2 - 0 = 2, tc = 2 - 1 = 1，
#      最终 plug 连接器落在 rows 2..3、cols 2..4。
#    - 盖印后保留可见 ctrl，不再 `ctrl -> obj`。然后以“盖印后的 ctrl 分布”为种子，
#      把其 4 邻接的 obj 一次性染成 ctrl，只扩散一层，形成焊接痕迹。
#    - 具体小例子：2x2 连接器对接后，中间可能出现
#          77
#          77
#          44
#          7?
#      其中第 3 行是 plug body 覆盖 base 上半连接器留下的 obj 支撑层；焊接扩散后，
#      这层 obj 会被整行染成 ctrl，于是 seam 比单纯“保留矩形连接器”更厚更明显。
# b) 复杂度校准
#    - 只允许两种矩形宽度：2 或 3；高度固定 2。这样输入中每题只有一个共享档位，
#      人类能直接看到 base/plug 两个相同矩形并据此匹配。
#    - 扩散规则只有一层、只走 4 邻域，不引入链式蔓延或多颜色映射，变化足够明显但
#      不会让样例失控。
# c) 隐式假设审查
#    - 原 main() 假设连接器是单行条，因此逐像素检查“base 下方空 / plug 上方空”。
#      融合后该假设失效，必须改成“矩形块外侧整行为空、内侧整行有 body 支撑”。
#    - 原位移 `tr = dr0 - 1 - pr0` 把常数 1 当成连接器高度。融合后必须使用
#      `tr = base_top - conn_h - plug_top`，否则 plug 会少上移一行。
#    - 原输出末尾无条件 `ctrl -> obj` 会抹掉 D5 的焊接效果；必须改成“先冻结初始
#      ctrl 掩码，再向相邻 obj 扩散一层”。
#    - 扩散若边染边扩，会意外得到两层焊接，因此需要先记录 `ctrl_mask`，再统一写回。
# d) 放置合法性
#    - 生成器显式构造 base 和 plug 的矩形连接器、支撑行以及 body 生长区域。
#    - 对接合法性检查禁止任何 obj-obj 重叠；只有至少一侧是 ctrl 的重叠才允许。
#    - 输入放置阶段把 base 固定在 (0,0)，plug 只放在右侧或下侧，且 gap 为 1..5 中
#      的可行值，确保两组件既不接触也不越界。
# e) 视觉可观察性
#    - 输入中橙色连接器从细条变成 2D 矩形块，差异肉眼可见。
#    - 输出中 seam 不只是保留矩形块，还会把贴着矩形块的黄色 body 再染一圈橙色；
#      2-3 组样例足以看出“矩形对接 + 一层焊接扩散”。
# f) 可视化整洁度
#    - 焊接仅发生在 seam 邻域，不引入额外标记或全局噪声；颜色仍只有 bg / obj / ctrl。
#    - base 连接器固定在底边、plug 连接器固定在顶边，输入阶段也只保留两个主组件，
#      避免矩形连接器和主体交错得难以辨认。
# g) 变化可推断性
#    - 唯一颜色变化是“与可见 ctrl 4 邻接的 obj -> ctrl”，由局部结构唯一决定，
#      不依赖颜色编号大小等人类不可见因素。
# h) main() 新假设与生成器保证
#    - 假设：每个精灵恰有一个 2x2/2x3 的实心 ctrl 矩形。保证：生成器显式构造并验证。
#    - 假设：base 矩形上方有完整 obj 支撑行，plug 矩形下方有完整 obj 支撑行。
#      保证：生成器把这些行纳入 keep 集合，thin 时绝不删除。
#    - 假设：对接后 plug 不会伸到 base 上方或左侧。保证：base 连接器放在足够靠下位置，
#      plug 连接器列偏移受 base `c0` 限制，且 `_can_dock_cleanly()` 检查 `tr>=0, tc>=0`。
#    - 假设：融合行为不是退化到“仅矩形连接器、不扩散”。保证：生成器要求焊接新增的
#      ctrl 数至少达到连接器宽度 `conn_w`，拒绝扩散不明显的样本。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _connector_info(ctrl_pts: set[tuple[int, int]]) -> dict[str, int | set[tuple[int, int]]]:
    assert ctrl_pts
    r0, c0, r1, c1 = _bbox_of_pts(ctrl_pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == 2 and w in (2, 3)
    shape = {(int(r - r0), int(c - c0)) for r, c in ctrl_pts}
    rect_shape = {(dr, dc) for dr in range(h) for dc in range(w)}
    assert shape == rect_shape
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "shape": shape,
    }


def _apply_weld(out: np.ndarray, obj: int, ctrl: int) -> np.ndarray:
    """Freeze current ctrl cells, then spread ctrl to adjacent body for one 4-neighbor layer."""
    welded = np.array(out, copy=True)
    ctrl_mask = out == int(ctrl)
    weld_mask = np.zeros_like(ctrl_mask, dtype=bool)
    for r, c in np.argwhere(ctrl_mask):
        r = int(r)
        c = int(c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) == int(obj):
                weld_mask[nr, nc] = True
    welded[weld_mask] = int(ctrl)
    return welded


def _render_docked_from_sets(
    source_grid: np.ndarray,
    base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    tr: int,
    tc: int,
    bg: int,
    ctrl: int,
) -> np.ndarray:
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(source_grid[int(r), int(c)])
    for r, c in plug_shift:
        val = int(source_grid[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)
    return out


def _render_docked_local(
    base_grid: np.ndarray,
    plug_grid: np.ndarray,
    base_info: dict[str, int],
    plug_info: dict[str, int],
    bg: int,
    ctrl: int,
) -> np.ndarray:
    tr = int(base_info["r0"] - plug_info["h"] - plug_info["r0"])
    tc = int(base_info["c0"] - plug_info["c0"])
    assert tr >= 0 and tc >= 0

    bh, bw = base_grid.shape
    ph, pw = plug_grid.shape
    out_h = max(bh, tr + ph)
    out_w = max(bw, tc + pw)
    out = np.full((int(out_h), int(out_w)), int(bg), dtype=int)

    for r in range(bh):
        for c in range(bw):
            if int(base_grid[r, c]) != int(bg):
                out[int(r), int(c)] = int(base_grid[r, c])
    for r in range(ph):
        for c in range(pw):
            val = int(plug_grid[r, c])
            if val == int(bg):
                continue
            dr = int(r + tr)
            dc = int(c + tc)
            cur = int(out[dr, dc])
            if cur != int(bg):
                assert cur == int(ctrl) or val == int(ctrl)
            out[dr, dc] = int(val)
    return out


def _weld_growth(seed: np.ndarray, welded: np.ndarray, ctrl: int) -> int:
    return int(np.sum(welded == int(ctrl)) - np.sum(seed == int(ctrl)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching rectangular connectors, then apply a
    one-layer 4-neighbor weld spread from the visible connector pixels.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    infoA = _connector_info(connA)
    infoB = _connector_info(connB)
    assert infoA["shape"] == infoB["shape"]

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD, infoD = A, connA, infoA
        plug, connP, infoP = B, connB, infoB
    else:
        base, connD, infoD = B, connB, infoB
        plug, connP, infoP = A, connA, infoA

    base_body = base - connD
    plug_body = plug - connP
    for c in range(int(infoD["c0"]), int(infoD["c1"]) + 1):
        assert (int(infoD["r0"]) - 1, int(c)) in base_body
        assert (int(infoD["r1"]) + 1, int(c)) not in base_body
    for c in range(int(infoP["c0"]), int(infoP["c1"]) + 1):
        assert (int(infoP["r0"]) - 1, int(c)) not in plug_body
        assert (int(infoP["r1"]) + 1, int(c)) in plug_body

    tr = int(infoD["r0"] - infoP["h"] - infoP["r0"])
    tc = int(infoD["c0"] - infoP["c0"])
    seed = _render_docked_from_sets(g, base, plug, tr, tc, bg, ctrl)
    welded = _apply_weld(seed, obj, ctrl)

    nz = np.argwhere(welded != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return welded[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allowed: np.ndarray) -> None:
    """Grow a connected shape within the allowed region."""
    count = int(mask.sum())
    frontier = list(frontier)
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h
            and 0 <= c + dc < w
            and bool(allowed[r + dr, c + dc])
            and not bool(mask[r + dr, c + dc])
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove interior pixels while preserving 8-connectivity and kept cells."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        rc = (int(r), int(c))
        if rc in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not bool(mask[r + dr, c + dc])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Draw a random Manhattan path between two cells."""
    r, c = map(int, start)
    tr, tc = map(int, end)
    mask[r, c] = True
    while (r, c) != (tr, tc):
        steps = []
        if r != tr:
            steps.append((r + (1 if tr > r else -1), c))
        if c != tc:
            steps.append((r, c + (1 if tc > c else -1)))
        r, c = _rng.choice(steps)
        mask[r, c] = True


def _build_base(conn_h: int, conn_w: int) -> tuple[np.ndarray, dict[str, int]] | None:
    bg, obj, ctrl = 1, 4, 7
    base_h = _rng.randint(4, 6)
    base_w = _rng.randint(max(conn_w + 1, 4), 8)
    conn_r0 = int(base_h - conn_h)
    conn_c0 = _rng.randint(0, base_w - conn_w)

    allowed = np.zeros((base_h, base_w), dtype=bool)
    allowed[:conn_r0, :] = True
    allowed[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = True

    mask = np.zeros((base_h, base_w), dtype=bool)
    mask[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = True
    mask[conn_r0 - 1, conn_c0 : conn_c0 + conn_w] = True
    mask[0, 0] = True
    _draw_path(mask, (0, 0), (conn_r0 - 1, _rng.randint(conn_c0, conn_c0 + conn_w - 1)))

    target_min = max(int(mask.sum()) + 1, 7)
    target_max = min(24, int(allowed.sum()))
    if target_max < target_min:
        return None
    _grow_shape(mask, _rng.randint(target_min, target_max), list(zip(*np.where(mask))), allowed)

    keep = {(0, 0)}
    keep |= {
        (r, c)
        for r in range(conn_r0, conn_r0 + conn_h)
        for c in range(conn_c0, conn_c0 + conn_w)
    }
    keep |= {(conn_r0 - 1, c) for c in range(conn_c0, conn_c0 + conn_w)}
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if not all(mask[conn_r0 - 1, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None

    grid = np.full((base_h, base_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[conn_r0 : conn_r0 + conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)
    return grid, {"r0": conn_r0, "c0": conn_c0, "r1": conn_r0 + conn_h - 1, "c1": conn_c0 + conn_w - 1, "h": conn_h, "w": conn_w}


def _build_plug(conn_h: int, conn_w: int, max_conn_c0: int) -> tuple[np.ndarray, dict[str, int]] | None:
    bg, obj, ctrl = 1, 4, 7
    plug_h = _rng.randint(conn_h + 1, 6)
    plug_w = _rng.randint(conn_w, 8)
    max_plug_conn_c0 = min(int(max_conn_c0), int(plug_w - conn_w))
    if max_plug_conn_c0 < 0:
        return None
    conn_c0 = _rng.randint(0, max_plug_conn_c0)

    allowed = np.zeros((plug_h, plug_w), dtype=bool)
    allowed[conn_h:, :] = True
    allowed[:conn_h, conn_c0 : conn_c0 + conn_w] = True

    mask = np.zeros((plug_h, plug_w), dtype=bool)
    mask[:conn_h, conn_c0 : conn_c0 + conn_w] = True
    mask[conn_h, conn_c0 : conn_c0 + conn_w] = True
    stem_col = _rng.randint(conn_c0, conn_c0 + conn_w - 1)
    _draw_path(mask, (conn_h, stem_col), (plug_h - 1, _rng.randint(0, plug_w - 1)))

    target_min = max(int(mask.sum()) + 1, 7)
    target_max = min(24, int(allowed.sum()))
    if target_max < target_min:
        return None
    _grow_shape(mask, _rng.randint(target_min, target_max), list(zip(*np.where(mask))), allowed)

    keep = {
        (r, c)
        for r in range(0, conn_h)
        for c in range(conn_c0, conn_c0 + conn_w)
    }
    keep |= {(conn_h, c) for c in range(conn_c0, conn_c0 + conn_w)}
    _thin(mask, keep=keep)

    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if not all(mask[conn_h, c] for c in range(conn_c0, conn_c0 + conn_w)):
        return None

    grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[:conn_h, conn_c0 : conn_c0 + conn_w] = int(ctrl)
    return grid, {"r0": 0, "c0": conn_c0, "r1": conn_h - 1, "c1": conn_c0 + conn_w - 1, "h": conn_h, "w": conn_w}


def _can_dock_cleanly(
    base_grid: np.ndarray,
    plug_grid: np.ndarray,
    base_info: dict[str, int],
    plug_info: dict[str, int],
) -> bool:
    bg, ctrl = 1, 7
    tr = int(base_info["r0"] - plug_info["h"] - plug_info["r0"])
    tc = int(base_info["c0"] - plug_info["c0"])
    if tr < 0 or tc < 0:
        return False

    bh, bw = base_grid.shape
    ph, pw = plug_grid.shape
    for r in range(ph):
        for c in range(pw):
            val = int(plug_grid[r, c])
            if val == int(bg):
                continue
            dr = int(r + tr)
            dc = int(c + tc)
            if 0 <= dr < bh and 0 <= dc < bw:
                cur = int(base_grid[dr, dc])
                if cur != int(bg) and cur != int(ctrl) and val != int(ctrl):
                    return False
    return True


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bg = 1
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    placements: list[tuple[str, int]] = []
    for gap in range(1, 6):
        if bh + gap + ph <= 11:
            placements.append(("below", gap))
        if bw + gap + pw <= 13:
            placements.append(("right", gap))
    if not placements:
        return None

    orient, gap = _rng.choice(placements)
    if orient == "below":
        canvas_h_min = bh + gap + ph
        canvas_h = _rng.randint(max(7, canvas_h_min), 11)
        canvas_w = _rng.randint(max(3, bw, pw), 13)
        pr = bh + gap
        pc = _rng.randint(0, canvas_w - pw)
    else:
        canvas_h = _rng.randint(max(7, bh, ph), 11)
        canvas_w_min = bw + gap + pw
        canvas_w = _rng.randint(max(3, canvas_w_min), 13)
        pr = _rng.randint(0, canvas_h - ph)
        pc = bw + gap

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    lab, n = label_mask(canvas != int(bg), connectivity=8)
    if int(n) != 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7
    conn_h = 2
    conn_w = _rng.choice((2, 3))

    built_base = _build_base(conn_h, conn_w)
    if built_base is None:
        return None
    base_grid, base_info = built_base

    built_plug = _build_plug(conn_h, conn_w, max_conn_c0=int(base_info["c0"]))
    if built_plug is None:
        return None
    plug_grid, plug_info = built_plug

    if not _can_dock_cleanly(base_grid, plug_grid, base_info, plug_info):
        return None

    docked_seed = _render_docked_local(base_grid, plug_grid, base_info, plug_info, bg, ctrl)
    docked_welded = _apply_weld(docked_seed, obj, ctrl)
    if _weld_growth(docked_seed, docked_welded, ctrl) < conn_w:
        return None

    base_sprite = _crop(base_grid, bg)
    plug_sprite = _crop(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    canvas = _place_input(base_sprite, plug_sprite)
    if canvas is None:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 3:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != int(bg)):
        return None

    expected = _crop(docked_welded, bg)
    if expected is None or not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")

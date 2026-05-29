# Variant of 20270e3b
# Dimension: D2 — 连接器方向 (match_cond)
# Change: 连接器固定为水平条、按上下方向对接
#   -> 连接器方向由上下文决定：可以是水平条或垂直条；
#      水平条仍按上下方向对接，垂直条改为按左右方向对接。
#
# Design analysis
# ----------------
# a) 具体化与行为追踪
# - 原题只有一种可见结构：两个 sprite 都带水平 ctrl 条。base 的 ctrl 下方留空、
#   上方贴着 body；plug 的 ctrl 上方留空、下方贴着 body。solver 把 plug 平移到
#   “其 ctrl 条正好位于 base ctrl 条上方一行”，于是 plug 的 body 会压到 base 的
#   ctrl 条上，最终所有 ctrl 重涂成 obj。
# - 本变体增加第二种可见结构：两个 sprite 的 ctrl 也可以是垂直条。此时使用严格
#   正交的对接方式：base 的垂直 ctrl 右侧留空、左侧贴 body；plug 的垂直 ctrl 左侧
#   留空、右侧贴 body。solver 把 plug 平移到“其 ctrl 条正好位于 base ctrl 条左边
#   一列”，于是 plug 的 body 会压到 base 的 ctrl 条上，最终同样把 ctrl 重涂成 obj。
# - 小例子：
#   原逻辑：base 的水平 ctrl 在 (3,2)(3,3)，plug 的水平 ctrl 在其本地 row 0。
#   solver 平移后让 plug ctrl 到达全局 row 2, cols 2..3。
#   变体垂直分支：base 的垂直 ctrl 在 rows 1..2, col 4，plug 的垂直 ctrl 在其本地 col 0。
#   solver 平移后让 plug ctrl 到达全局 col 3, rows 1..2。
#   差异点就是：平移公式、合法邻接方向、以及生成器中“body 从 connector 朝哪边生长”。
#
# b) 复杂度校准
# - 连接器方向只保留两档：`horizontal` / `vertical`。这是 D2 的核心自由度，足够明显，
#   但不会引入过多分支。
# - 为了让方向肉眼可见，连接器长度限制为 2-3；不再生成 1 像素连接器，否则方向不可见。
# - 其余颜色、角色判定、两组件结构、tight-crop 输出都保持原任务风格，避免任务漂移。
#
# c) 原 solver 中依赖旧属性的不变性
# - `norm(conn)` + `assert shpA == shpB)`：原题默认 ctrl 都是水平条；变体里仍可保留，
#   但 solver 不能再把“shape 一样”直接当成“水平条”。
# - `for r,c in connD: assert (r+1,c) not in base_body` / `assert (r-1,c) not in plug_body`：
#   这两行把“自由侧一定是上下”写死了。变体必须改成“根据方向分别检查上下或左右”。
# - `tr = (dr0 - 1) - pr0; tc = dc0 - pc0`：这把平移轴固定成竖直方向。垂直连接器
#   分支需要改成 `tr = dr0 - pr0; tc = (dc0 - 1) - pc0`。
# - `assert rr0 == br0 and cc0 == bc0`：这依赖 plug 不会伸到 base 的上方或左方。
#   变体里仍保留该约束，所以垂直分支必须选择“plug 在 base 左边一列对接，但 plug body
#   向右生长”，而不是反过来。
#
# d) 放置合法性
# - 生成器继续把 base 固定在输入画布左上角，plug 单独放在别处。
# - 使用 `random_free_location(..., padding=1)` 保证两个 sprite 在输入里不会以 8 邻接相连。
# - base/plug 的 connector 邻接关系在生成时直接构造，不依赖 oracle 事后碰运气。
# - docked 检查中，plug 与 base 的重叠只允许落在 base 的 ctrl 区；任何压到 base obj 的
#   情况都直接拒绝。
#
# e) 视觉可观察性
# - 在 2-4 个样例里，人类能直接看到：有的例子 ctrl 是横条，输出是上下拼接；有的例子
#   ctrl 是竖条，输出是左右拼接。
# - 方向由几何形状和相邻 body 的一侧共同提示，不依赖颜色编号或隐藏排序。
#
# f) 可视化整洁度
# - 只保留 3 种颜色（bg/obj/ctrl），ctrl 很短，且每个 sprite 恰有一个 connector。
# - 输入中的两个 sprite 有至少 1 格 8-邻域间隔，避免视觉上粘连成一团。
#
# g) 颜色变化可推断性
# - 颜色规则不变：ctrl 只是“接口标记”，输出一律重涂成 obj。变体只改变接口方向和对接轴，
#   不引入新的颜色映射。
#
# h) 新 solver 的隐式假设
# - 仍然假设恰好 2 个 8-connected 前景组件。
# - 仍然假设两个 connector 的归一化形状相同。
# - 新增假设：每个 connector 必须能从“几何条形 + body/free 侧”唯一判定为水平或垂直分支。
#   generator 通过只生成长度 2-3 的直条，并固定 body/free 侧来保证这一点。
#
# i) generator 对 solver 假设的保证
# - 水平分支：base connector 上方全是 body、下方全空；plug connector 下方全是 body、
#   上方天然全空（connector 在本地 top row）。
# - 垂直分支：base connector 左侧全是 body、右侧全空；plug connector 右侧全是 body、
#   左侧天然全空（connector 在本地 left col）。
# - plug connector 固定贴在自身 tight bbox 的上边或左边，保证 solver 的对接后结果不会
#   出现在 base 的上方或左方。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (r0, c0)


def _all_has(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) in body for r, c in pts)


def _all_free(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) not in body for r, c in pts)


def _infer_axis(
    sprite: set[tuple[int, int]],
    conn_pts: set[tuple[int, int]],
    role: str,
) -> str:
    body = sprite - conn_pts
    shape, _anchor = _norm_pts(conn_pts)
    rows = {r for r, _c in shape}
    cols = {c for _r, c in shape}

    geom_axis: str | None = None
    if len(rows) == 1 and len(cols) >= 2:
        geom_axis = "h"
    if len(cols) == 1 and len(rows) >= 2:
        geom_axis = "v"

    options: list[str] = []
    if role == "base":
        if _all_has(body, conn_pts, -1, 0) and _all_free(body, conn_pts, 1, 0):
            options.append("h")
        if _all_has(body, conn_pts, 0, -1) and _all_free(body, conn_pts, 0, 1):
            options.append("v")
    else:
        if _all_has(body, conn_pts, 1, 0) and _all_free(body, conn_pts, -1, 0):
            options.append("h")
        if _all_has(body, conn_pts, 0, 1) and _all_free(body, conn_pts, 0, -1):
            options.append("v")

    if geom_axis is not None:
        options = [axis for axis in options if axis == geom_axis]

    assert len(options) == 1
    axis = options[0]

    if axis == "h":
        assert len({r for r, _c in conn_pts}) == 1
    else:
        assert len({c for _r, c in conn_pts}) == 1
    return axis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant rule:
    - Horizontal connector -> original vertical docking.
    - Vertical connector -> orthogonal horizontal docking.
    """
    g = np.array(grid, dtype=int)

    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, _anchA = _norm_pts(connA)
    shpB, _anchB = _norm_pts(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    axis_base = _infer_axis(base, connD, role="base")
    axis_plug = _infer_axis(plug, connP, role="plug")
    assert axis_base == axis_plug
    axis = axis_base

    _sp, (pr0, pc0) = _norm_pts(connP)
    _sd, (dr0, dc0) = _norm_pts(connD)
    if axis == "h":
        tr = int((dr0 - 1) - pr0)
        tc = int(dc0 - pc0)
    else:
        tr = int(dr0 - pr0)
        tc = int((dc0 - 1) - pc0)
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
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]
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
    """Remove interior pixels to make a thinner connected shape."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _find_base_connector(
    base: np.ndarray,
    axis: str,
    conn_len: int,
) -> tuple[int, list[int]] | tuple[list[int], int] | None:
    h, w = base.shape
    candidates: list[tuple[int, list[int]] | tuple[list[int], int]] = []
    if axis == "h":
        for r in range(1, h):
            for c0 in range(0, w - conn_len + 1):
                cols = list(range(c0, c0 + conn_len))
                if not all(base[r, c] for c in cols):
                    continue
                if not all(base[r - 1, c] for c in cols):
                    continue
                if not all(r + 1 >= h or not base[r + 1, c] for c in cols):
                    continue
                candidates.append((r, cols))
    else:
        for c in range(1, w):
            for r0 in range(0, h - conn_len + 1):
                rows = list(range(r0, r0 + conn_len))
                if not all(base[r, c] for r in rows):
                    continue
                if not all(base[r, c - 1] for r in rows):
                    continue
                if not all(c + 1 >= w or not base[r, c + 1] for r in rows):
                    continue
                candidates.append((rows, c))
    if not candidates:
        return None
    return _rng.choice(candidates)


def _build_plug(
    axis: str,
    conn_len: int,
    base_grid: np.ndarray,
    base_conn: tuple[int, list[int]] | tuple[list[int], int],
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray | None:
    base_h, base_w = base_grid.shape
    if axis == "h":
        conn_row, conn_cols = base_conn
        assert isinstance(conn_row, int)
        assert isinstance(conn_cols, list)
        plug_h = _rng.randint(max(3, 2), 7)
        plug_w = _rng.randint(max(conn_len, 2), min(7, max(conn_len, 2) + 4))
        plug = np.zeros((plug_h, plug_w), dtype=bool)
        keep: set[tuple[int, int]] = set()
        for c in range(conn_len):
            plug[0, c] = True
            plug[1, c] = True
            keep.add((0, c))
            keep.add((1, c))
        plug_target = _rng.randint(max(5, conn_len + 2), min(24, plug_h * plug_w))
        _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
        if not is_contiguous(plug, background=0, connectivity=8):
            return None
        _thin(plug, keep=keep)
        if not all(plug[1, c] for c in range(conn_len)):
            return None

        plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
        plug_grid[plug] = int(obj)
        for c in range(conn_len):
            plug_grid[0, c] = int(ctrl)

        tr = int(conn_row - 1)
        tc = int(conn_cols[0])
        for r in range(plug_h):
            for c in range(plug_w):
                if int(plug_grid[r, c]) == int(bg):
                    continue
                dr = int(tr + r)
                dc = int(tc + c)
                if (
                    0 <= dr < base_h
                    and 0 <= dc < base_w
                    and int(base_grid[dr, dc]) == int(obj)
                    and int(plug_grid[r, c]) == int(obj)
                ):
                    return None
        return plug_grid

    conn_rows, conn_col = base_conn
    assert isinstance(conn_rows, list)
    assert isinstance(conn_col, int)
    plug_h = _rng.randint(max(conn_len, 3), 7)
    plug_w = _rng.randint(2, 7)
    plug = np.zeros((plug_h, plug_w), dtype=bool)
    keep = set()
    for r in range(conn_len):
        plug[r, 0] = True
        plug[r, 1] = True
        keep.add((r, 0))
        keep.add((r, 1))
    plug_target = _rng.randint(max(5, conn_len + 2), min(24, plug_h * plug_w))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep)
    if not all(plug[r, 1] for r in range(conn_len)):
        return None

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for r in range(conn_len):
        plug_grid[r, 0] = int(ctrl)

    tr = int(conn_rows[0])
    tc = int(conn_col - 1)
    for r in range(plug_h):
        for c in range(plug_w):
            if int(plug_grid[r, c]) == int(bg):
                continue
            dr = int(tr + r)
            dc = int(tc + c)
            if (
                0 <= dr < base_h
                and 0 <= dc < base_w
                and int(base_grid[dr, dc]) == int(obj)
                and int(plug_grid[r, c]) == int(obj)
            ):
                return None
    return plug_grid


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7
    axis = _rng.choice(["h", "v"])

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 7)
    base_target = _rng.randint(6, min(24, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(2, min(3, base_w if axis == "h" else base_h))
    base_conn = _find_base_connector(base, axis, conn_len)
    if base_conn is None:
        return None

    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(obj)
    if axis == "h":
        conn_row, conn_cols = base_conn
        assert isinstance(conn_row, int)
        assert isinstance(conn_cols, list)
        for c in conn_cols:
            base_grid[conn_row, c] = int(ctrl)
    else:
        conn_rows, conn_col = base_conn
        assert isinstance(conn_rows, list)
        assert isinstance(conn_col, int)
        for r in conn_rows:
            base_grid[r, conn_col] = int(ctrl)
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_grid = _build_plug(axis, conn_len, base_grid, base_conn, bg, obj, ctrl)
    if plug_grid is None:
        return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    min_h = max(7, bh, ph + 1)
    min_w = max(5, bw, pw + 1)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    if result.shape[0] > 12 or result.shape[1] > 12:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")

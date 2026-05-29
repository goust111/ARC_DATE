# Combo variant of 20270e3b (index 3)
# combo_spec: D2/1+D15/0
# Name: 自适应方向同向翻转
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 原题只有水平 connector，base 自由侧朝下、plug 自由侧朝上，所以可以直接把 plug
#   平移到 base connector 正上方一行。
# - D2/1 把 connector 方向推广成两档：水平条时仍做上下对接；垂直条时改做左右对接。
# - D15/0 把“自由侧相反”改成“自由侧同向”，因此不能直接对接，必须先把 plug 沿对接轴
#   做镜像翻转：
#   * 水平条：base 与 plug 输入态都满足“上方贴 body、下方 free”。solver 先把 plug
#     在自身 bbox 内上下翻转，使其变成“上方 free、下方贴 body”，再把 connector 对齐到
#     base connector 正上方一行。
#   * 垂直条：base 与 plug 输入态都满足“左侧贴 body、右侧 free”。solver 先把 plug
#     在自身 bbox 内左右翻转，使其变成“左侧 free、右侧贴 body”，再把 connector 对齐到
#     base connector 左边一列。
# - 小例子：
#   * 水平分支：base connector 在全局 row 4, cols 2..3；plug 输入 connector 在其 bbox
#     底部一行。翻转后 connector 到 plug 局部 row 0，再平移到全局 row 3, cols 2..3。
#   * 垂直分支：base connector 在全局 rows 1..3, col 5；plug 输入 connector 在其 bbox
#     最右列。翻转后 connector 到 plug 局部 col 0，再平移到全局 col 4, rows 1..3。
#   差异不再只是“平移轴变化”，而是“先判定轴，再按该轴翻转 plug，再正交对接”。
#
# b) 复杂度校准
# - connector 方向只保留 `horizontal` / `vertical` 两档，避免分支爆炸。
# - connector 长度限制为 2..3，保证方向在视觉上可见；1 像素 connector 会让 D2 退化。
# - 翻转仅一档：水平分支上下翻，垂直分支左右翻。不会混入旋转等额外自由度。
#
# c) 对原 main() 隐式假设的审查
# - 原版 `assert shpA == shpB` 仍然可保留，但不能再把“shape 相同”解释成“一定是水平条”。
# - 原版对 `base_body` / `plug_body` 的检查把自由侧写死成上下；融合后必须改成：
#   输入态两者都同向，且方向由 connector 轴决定。
# - 原版直接平移 raw plug；融合后先把 plug 局部网格按轴翻转，再重新计算翻转后
#   connector 的 anchor，最后平移。
# - 原版 `assert rr0 == br0 and cc0 == bc0` 仍然成立，因此 generator 必须保证：
#   水平分支中翻转后的 plug 不会伸到 base 上方；垂直分支中翻转后的 plug 不会伸到
#   base 左方。
#
# d) 放置合法性
# - generator 采用 output-first：先构造 docked 状态下的 base 和“已翻正”的 plug，
#   验证装配合法，再把 plug 按轴翻回输入态。
# - 输入放置时，base 固定在画布左上角，plug 通过 `random_free_location(..., padding=1)`
#   放到别处，避免两个 sprite 在输入里 8 邻接粘连。
# - 装配时只允许 ctrl 与 obj 的重叠；obj 与 obj 的重叠一律拒绝。
#
# e) 视觉可观察性
# - 几组样例里，人类能看到两类现象：
#   * connector 可能是横条，也可能是竖条；
#   * 两个输入 sprite 的 connector 自由侧总是同向，输出里 plug 明显被翻正后再接上。
# - generator 强制 plug 对应翻转轴非对称，避免翻转后外观不变而使 D15 退化。
#
# f) 可视化整洁度
# - 仍只使用 bg=1 / obj=4 / ctrl=7 三种颜色。
# - 每个 sprite 恰有一个短 connector，且输入里两个 sprite 留出至少一格 8 邻域间隔。
# - 最终重叠只发生在对接缝的 ctrl-vs-obj 区域，不会出现多层难以辨认的覆盖。
#
# g) 变化的可推断性
# - 颜色规则未扩张：ctrl 只表示接口，输出统一重涂为 obj。
# - 人类可从少量样例归纳出“先看 connector 是横还是竖，再把 plug 按对应轴翻转并对接”。
#
# h) 新 solver 的隐式假设
# - 恰好有 2 个 8-connected 前景组件。
# - 两个 connector 的归一化形状相同，且是长度 2..3 的直条。
# - 输入态中 base 与 plug 的 connector 自由侧同向：
#   水平时都朝下，垂直时都朝右。
# - plug 在输入态沿相应翻转轴是非对称的，因此翻转行为在视觉上可见。
#
# i) generator 对这些假设的保证
# - 水平分支生成：base/plug 输入都满足 body 在上、free 在下；plug docked 态满足
#   body 在下、free 在上，随后对其做上下翻转得到输入。
# - 垂直分支生成：base/plug 输入都满足 body 在左、free 在右；plug docked 态满足
#   body 在右、free 在左，随后对其做左右翻转得到输入。
# - docked plug 的 connector 固定贴在自身 bbox 的上边或左边，确保装配后不会把输出
#   bbox 的左上角从 base 身上挪走。

from __future__ import annotations

import random as _rng

import numpy as np

try:
    from solvers.arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location
except ImportError:
    from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    r0, c0 = min(rs), min(cs)
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (int(r0), int(c0))


def _all_has(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) in body for r, c in pts)


def _all_free(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) not in body for r, c in pts)


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _sprite_local_grid(
    grid: np.ndarray,
    pts: set[tuple[int, int]],
    bg: int = BG,
) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out, (int(r0), int(c0))


def _grid_points(local_grid: np.ndarray, bg: int = BG) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(local_grid != int(bg))}


def _grid_connector(local_grid: np.ndarray, ctrl: int = CTRL) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(local_grid == int(ctrl))}


def _infer_input_axis(sprite: set[tuple[int, int]], conn_pts: set[tuple[int, int]]) -> str:
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
    if _all_has(body, conn_pts, -1, 0) and _all_free(body, conn_pts, 1, 0):
        options.append("h")
    if _all_has(body, conn_pts, 0, -1) and _all_free(body, conn_pts, 0, 1):
        options.append("v")

    if geom_axis is not None:
        options = [axis for axis in options if axis == geom_axis]

    assert len(options) == 1
    return options[0]


def _is_flipped_plug_orientation(
    axis: str,
    conn_pts: set[tuple[int, int]],
    body_pts: set[tuple[int, int]],
) -> bool:
    if axis == "h":
        return _all_free(body_pts, conn_pts, -1, 0) and _all_has(body_pts, conn_pts, 1, 0)
    return _all_free(body_pts, conn_pts, 0, -1) and _all_has(body_pts, conn_pts, 0, 1)


def _flip_for_axis(local_grid: np.ndarray, axis: str) -> np.ndarray:
    if axis == "h":
        return np.flipud(local_grid)
    return np.fliplr(local_grid)


def _render_assembled(
    base_grid: np.ndarray,
    plug_grid: np.ndarray,
    plug_top: int,
    plug_left: int,
) -> np.ndarray | None:
    base_pts = _grid_points(base_grid)
    plug_pts = _grid_points(plug_grid)
    shifted = {(int(r + plug_top), int(c + plug_left)) for r, c in plug_pts}
    pts_all = base_pts | shifted
    if not pts_all:
        return None

    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)

    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_grid[int(r), int(c)])
    for r, c in plug_pts:
        rr = int(r + plug_top - rr0)
        cc = int(c + plug_left - cc0)
        val = int(plug_grid[int(r), int(c)])
        cur = int(out[rr, cc])
        if cur != int(BG) and val != int(BG):
            assert cur == int(CTRL) or val == int(CTRL)
        out[rr, cc] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, bg=BG)


def _can_overlay(base_grid: np.ndarray, plug_grid: np.ndarray, top: int, left: int) -> bool:
    base_h, base_w = base_grid.shape
    plug_h, plug_w = plug_grid.shape
    for r in range(plug_h):
        for c in range(plug_w):
            val = int(plug_grid[r, c])
            if val == int(BG):
                continue
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < base_h and 0 <= cc < base_w:
                cur = int(base_grid[rr, cc])
                if cur != int(BG) and cur != int(CTRL) and val != int(CTRL):
                    return False
    return True


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
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
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        rc = (int(r), int(c))
        if rc in keep:
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
                candidates.append((int(r), cols))
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
                candidates.append((rows, int(c)))
    if not candidates:
        return None
    return _rng.choice(candidates)


def _build_docked_plug(
    axis: str,
    conn_len: int,
    base_grid: np.ndarray,
    base_conn: tuple[int, list[int]] | tuple[list[int], int],
) -> tuple[np.ndarray, int, int] | None:
    if axis == "h":
        conn_row, conn_cols = base_conn
        assert isinstance(conn_row, int)
        assert isinstance(conn_cols, list)
        plug_h = _rng.randint(3, 7)
        plug_w = _rng.randint(max(conn_len, 2), 7)
        max_conn_c0 = min(int(conn_cols[0]), int(plug_w - conn_len))
        if max_conn_c0 < 0:
            return None
        plug_conn_c0 = _rng.randint(0, max_conn_c0)

        plug = np.zeros((plug_h, plug_w), dtype=bool)
        keep: set[tuple[int, int]] = set()
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug[0, c] = True
            plug[1, c] = True
            keep.add((0, c))
            keep.add((1, c))

        plug_target = _rng.randint(max(5, conn_len + 3), min(24, plug_h * plug_w))
        _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
        if not is_contiguous(plug, background=0, connectivity=8):
            return None
        _thin(plug, keep=keep)
        if not all(plug[1, c] for c in range(plug_conn_c0, plug_conn_c0 + conn_len)):
            return None

        plug_grid = np.full((plug_h, plug_w), int(BG), dtype=int)
        plug_grid[plug] = int(OBJ)
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug_grid[0, c] = int(CTRL)

        dock_top = int(conn_row - 1)
        dock_left = int(conn_cols[0] - plug_conn_c0)
        if not _can_overlay(base_grid, plug_grid, dock_top, dock_left):
            return None
        if np.array_equal(plug_grid, np.flipud(plug_grid)):
            return None
        return plug_grid, dock_top, dock_left

    conn_rows, conn_col = base_conn
    assert isinstance(conn_rows, list)
    assert isinstance(conn_col, int)
    plug_h = _rng.randint(max(conn_len, 3), 7)
    plug_w = _rng.randint(3, 7)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    keep = set()
    for r in range(conn_len):
        plug[r, 0] = True
        plug[r, 1] = True
        keep.add((r, 0))
        keep.add((r, 1))

    plug_target = _rng.randint(max(5, conn_len + 3), min(24, plug_h * plug_w))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep)
    if not all(plug[r, 1] for r in range(conn_len)):
        return None

    plug_grid = np.full((plug_h, plug_w), int(BG), dtype=int)
    plug_grid[plug] = int(OBJ)
    for r in range(conn_len):
        plug_grid[r, 0] = int(CTRL)

    dock_top = int(conn_rows[0])
    dock_left = int(conn_col - 1)
    if not _can_overlay(base_grid, plug_grid, dock_top, dock_left):
        return None
    if np.array_equal(plug_grid, np.fliplr(plug_grid)):
        return None
    return plug_grid, dock_top, dock_left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Adaptive-axis same-side docking with a mandatory plug flip.

    - Horizontal connector: both input connectors face down, so flip plug vertically.
    - Vertical connector: both input connectors face right, so flip plug horizontally.
    - Then dock orthogonally, recolor ctrl->obj, and tight-crop.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != int(BG), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for r, c in sprite if int(g[int(r), int(c)]) == int(CTRL)}

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
        base, conn_base = A, connA
        plug, conn_plug = B, connB
    else:
        base, conn_base = B, connB
        plug, conn_plug = A, connA

    axis_base = _infer_input_axis(base, conn_base)
    axis_plug = _infer_input_axis(plug, conn_plug)
    assert axis_base == axis_plug
    axis = axis_base

    plug_grid, _plug_origin = _sprite_local_grid(g, plug, bg=BG)
    plug_flipped = _flip_for_axis(plug_grid, axis)
    conn_flip = _grid_connector(plug_flipped)
    body_flip = _grid_points(plug_flipped) - conn_flip
    assert conn_flip
    assert _is_flipped_plug_orientation(axis, conn_flip, body_flip)

    _shape_flip, (pr0, pc0) = _norm_pts(conn_flip)
    _shape_base, (dr0, dc0) = _norm_pts(conn_base)
    if axis == "h":
        tr = int((dr0 - 1) - pr0)
        tc = int(dc0 - pc0)
    else:
        tr = int(dr0 - pr0)
        tc = int((dc0 - 1) - pc0)

    plug_shift_vals: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(plug_flipped != int(BG)):
        rr = int(r + tr)
        cc = int(c + tc)
        plug_shift_vals[(rr, cc)] = int(plug_flipped[int(r), int(c)])

    pts_all = base | set(plug_shift_vals)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for (r, c), val in plug_shift_vals.items():
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(BG) and val != int(BG):
            assert cur == int(CTRL) or val == int(CTRL)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    cropped = _crop_non_bg(out, bg=BG)
    if cropped is None:
        return np.full((0, 0), int(BG), dtype=int)
    return cropped


def _gen_attempt() -> np.ndarray | None:
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

    conn_cap = base_w if axis == "h" else base_h
    conn_len = _rng.randint(2, min(3, conn_cap))
    base_conn = _find_base_connector(base, axis, conn_len)
    if base_conn is None:
        return None

    base_grid = np.full((base_h, base_w), int(BG), dtype=int)
    base_grid[base] = int(OBJ)
    if axis == "h":
        conn_row, conn_cols = base_conn
        assert isinstance(conn_row, int)
        assert isinstance(conn_cols, list)
        for c in conn_cols:
            base_grid[conn_row, c] = int(CTRL)
    else:
        conn_rows, conn_col = base_conn
        assert isinstance(conn_rows, list)
        assert isinstance(conn_col, int)
        for r in conn_rows:
            base_grid[r, conn_col] = int(CTRL)
    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None

    docked = _build_docked_plug(axis, conn_len, base_grid, base_conn)
    if docked is None:
        return None
    plug_docked, dock_top, dock_left = docked
    if not is_contiguous(plug_docked != int(BG), background=0, connectivity=8):
        return None

    expected = _render_assembled(base_grid, plug_docked, dock_top, dock_left)
    if expected is None or expected.size == 0:
        return None

    plug_input = _flip_for_axis(plug_docked, axis)
    conn_input = _grid_connector(plug_input)
    body_input = _grid_points(plug_input) - conn_input
    conn_base_pts = _grid_connector(base_grid)
    body_base_pts = _grid_points(base_grid) - conn_base_pts
    if not conn_input or not conn_base_pts:
        return None
    if _infer_input_axis(_grid_points(base_grid), conn_base_pts) != axis:
        return None
    if _infer_input_axis(_grid_points(plug_input), conn_input) != axis:
        return None
    if not _is_flipped_plug_orientation(
        axis,
        _grid_connector(plug_docked),
        _grid_points(plug_docked) - _grid_connector(plug_docked),
    ):
        return None
    if axis == "h":
        if np.array_equal(plug_input, plug_docked):
            return None
        if not (_all_has(body_input, conn_input, -1, 0) and _all_free(body_input, conn_input, 1, 0)):
            return None
        if not (_all_has(body_base_pts, conn_base_pts, -1, 0) and _all_free(body_base_pts, conn_base_pts, 1, 0)):
            return None
    else:
        if np.array_equal(plug_input, plug_docked):
            return None
        if not (_all_has(body_input, conn_input, 0, -1) and _all_free(body_input, conn_input, 0, 1)):
            return None
        if not (_all_has(body_base_pts, conn_base_pts, 0, -1) and _all_free(body_base_pts, conn_base_pts, 0, 1)):
            return None

    base_sprite = _crop_non_bg(base_grid, bg=BG)
    plug_sprite = _crop_non_bg(plug_input, bg=BG)
    if base_sprite is None or plug_sprite is None:
        return None
    if int(np.sum(base_sprite != int(BG))) < 5 or int(np.sum(plug_sprite != int(BG))) < 5:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    min_h = max(7, max(bh, ph) + 2)
    min_w = max(5, max(bw, pw) + 2)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    try:
        pr, pc = random_free_location(
            canvas,
            plug_sprite,
            transparent=int(BG),
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=int(BG))

    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    if result.shape[0] > 12 or result.shape[1] > 12:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 600 attempts")

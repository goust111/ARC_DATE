# Combo variant of 20270e3b (index 2)
# combo_spec: D2/0+D11/0
# Name: 垂直连接器旋转对接

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 颜色保持原题固定值：`bg=1, obj=4, ctrl=7`。
   - canonical 装配态采用 D2 的几何：两个 sprite 都有同长度的竖直 ctrl 条，
     对接方向改为水平；plug 从 base 的左侧或右侧拼上去。
   - 但输入中的 plug 不是 canonical 朝向，而是先做一次四分之一转（90 或 270）。
     因此解题者在输入里通常会看到：
     * base 的 connector 是竖条；
     * plug 的 raw connector 变成横条；
     * 输出里两者装配后重新变成一个紧凑整体，所有 ctrl 都恢复成 obj。
   - 具体 seam（以 `base_side=-1` 为例，表示 base body 在 connector 左侧）：
     * base connector 在列 `c`，其 body-support 列在 `c-1`。
     * canonical plug connector 也为竖条，且其 body 在右侧，因此 dock 后：
       - plug 的 ctrl 覆盖 base 的 `obj` 支撑列 `c-1`
       - plug 的 body 覆盖 base 的 ctrl 列 `c`
     * 最终 ctrl 全部重着色为 obj，视觉上像一条 2 列宽的拼接带。
   - 相比原题：
     * 原题：不旋转，按“上下一格错位”的垂直推理移动。
     * 融合后：先枚举 plug 的 quarter-turn，再按“左右一列错位”的水平推理移动。

b) 复杂度校准
   - 竖条 connector 长度限制为 2 或 3，避免长度 1 时“横/竖不可区分”导致 D2 退化。
   - plug 输入旋转只允许四分之一转，不允许 180 度；这样 raw 输入里 connector 会从竖条
     变成横条，旋转需求肉眼明显。
   - 左接/右接都保留，但都遵守同一条规则：恢复 plug 朝向后再做左右对接。

c) 隐式假设审查
   - 原 `main()` 在比较两个 connector 前就假定它们 shape 相同；融合后 raw plug 可能已被
     旋转成横条，这个假设失效，所以 shape 匹配必须移到旋转枚举之后。
   - 原 `main()` 写死“base free side 在下、plug free side 在上”；融合后需要改为
     检查 connector 两侧的水平 body/free 关系。
   - 原平移公式 `tr=(dr0-1)-pr0, tc=dc0-pc0` 依赖上下对接；融合后必须改成
     `tr=dr0-pr0, tc=(dc0+base_side)-pc0`。
   - 原代码直接从 raw plug 像素盖到输出；融合后必须从“旋转后的候选 plug”进行装配。
   - D2 单变体里 overlap 只做宽松断言；融合后采用更严格的 seam 校验：
     合法重叠只能是 `plug ctrl -> base obj-support` 与 `plug obj-support -> base ctrl`。

d) 放置合法性
   - generator 先构造 canonical 装配态，再把 plug 做一次四分之一转放回输入。
   - base 仍固定贴输入左上角，以保持“哪个是 base”可稳定识别。
   - plug 放置到画布其他空位时使用 `padding=1` 且 8 连通，保证两 sprite 在输入中
     仍是两个独立连通块，不会因对角接触粘连。
   - oracle 校验要求 `main(input) == expected`，不是仅仅“不崩溃”。

e) 视觉可观察性
   - 从少量样例中，人类能看到 base 的竖条 connector 与 plug 的横条 connector 不同向，
     输出却总是左右拼接成功，因此“plug 先旋转再水平对接”的规则可见。
   - 若某样例 raw plug 不需要旋转也能装配，就会退化成纯 D2；generator 会显式过滤掉。

f) 可视化整洁度
   - 仍只有两种前景色（obj/ctrl），没有额外标记。
   - 重叠被限制在一条窄 seam 上，不允许其他区域发生覆盖，避免图像混乱。

g) 变化的可推断性
   - 颜色变化只有一种：输出把 ctrl 全部恢复成 obj。
   - 这条规则与 connector 的结构位置强绑定，没有额外颜色映射需要记忆。

h) 新 solver 假设与 generator 保证
   - solver 假设 base connector 是竖直连续条，plug 在某个非零 quarter-turn 后才会
     成为可对接的竖直连续条。
   - solver 假设合法装配时，plug 旋转候选恰有一个能通过严格 seam 校验。
   - generator 通过构造和候选过滤保证上述假设成立。

i) 融合是否真正发生
   - 关键量化指标是：
     * raw input plug 不能直接用 D2 规则装配；
     * 但存在且仅存在一个 `rot90` 候选能装配；
     * 该候选的校正角度属于 `{1,3}`，即确实是四分之一转而非 180 度。
   - 后续测试脚本会统计这些指标，防止样本退化回单纯的 D2。
"""

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _sprite_from_points(
    grid: np.ndarray,
    pts: set[tuple[int, int]],
    bg: int = BG,
) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out, (int(r0), int(c0))


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr == int(CTRL))}


def _body_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr == int(OBJ))}


def _all_fg_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr != int(BG))}


def _norm_shape(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assert_vertical_bar(shape: set[tuple[int, int]]) -> None:
    if not shape:
        raise AssertionError("empty connector")
    cols = {int(c) for _r, c in shape}
    assert len(cols) == 1
    rows = sorted(int(r) for r, _c in shape)
    assert rows == list(range(rows[0], rows[-1] + 1))


def _connector_body_side(arr: np.ndarray, conn_pts: set[tuple[int, int]]) -> int | None:
    body = _body_coords(arr)
    valid: list[int] = []
    for side in (-1, 1):
        if all((int(r), int(c + side)) in body for r, c in conn_pts) and all(
            (int(r), int(c - side)) not in body for r, c in conn_pts
        ):
            valid.append(int(side))
    if len(valid) != 1:
        return None
    return int(valid[0])


def _assemble_horizontal_docked(base_arr: np.ndarray, plug_arr: np.ndarray) -> np.ndarray | None:
    base_conn = _conn_coords(base_arr)
    plug_conn = _conn_coords(plug_arr)
    if not base_conn or not plug_conn:
        return None

    base_shape, (br0, bc0) = _norm_shape(base_conn)
    plug_shape, (pr0, pc0) = _norm_shape(plug_conn)
    try:
        _assert_vertical_bar(base_shape)
        _assert_vertical_bar(plug_shape)
    except AssertionError:
        return None
    if base_shape != plug_shape:
        return None

    base_side = _connector_body_side(base_arr, base_conn)
    plug_side = _connector_body_side(plug_arr, plug_conn)
    if base_side is None or plug_side is None:
        return None
    if int(plug_side) != int(-base_side):
        return None

    base_support = {(int(r), int(c + base_side)) for r, c in base_conn}
    plug_support_local = {(int(r), int(c + plug_side)) for r, c in plug_conn}
    plug_body = _body_coords(plug_arr)
    if not plug_support_local.issubset(plug_body):
        return None

    tr = int(br0 - pr0)
    tc = int((bc0 + base_side) - pc0)

    shifted_vals: dict[tuple[int, int], int] = {}
    overlap_ctrl: set[tuple[int, int]] = set()
    overlap_body: set[tuple[int, int]] = set()
    for r, c in _all_fg_coords(plug_arr):
        rr, cc = int(r + tr), int(c + tc)
        val = int(plug_arr[int(r), int(c)])
        shifted_vals[(int(rr), int(cc))] = int(val)
        if 0 <= rr < base_arr.shape[0] and 0 <= cc < base_arr.shape[1]:
            cur = int(base_arr[int(rr), int(cc)])
            if cur == int(BG):
                continue
            if cur == int(OBJ) and val == int(CTRL) and (int(rr), int(cc)) in base_support:
                overlap_ctrl.add((int(rr), int(cc)))
                continue
            if cur == int(CTRL) and val == int(OBJ) and (int(rr), int(cc)) in base_conn:
                overlap_body.add((int(rr), int(cc)))
                continue
            return None

    if overlap_ctrl != base_support or overlap_body != base_conn:
        return None

    pts_all = _all_fg_coords(base_arr) | set(shifted_vals)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)

    for r, c in _all_fg_coords(base_arr):
        out[int(r - rr0), int(c - cc0)] = int(base_arr[int(r), int(c)])

    for (rr, cc), val in shifted_vals.items():
        cur = int(out[int(rr - rr0), int(cc - cc0)])
        if cur != int(BG):
            if cur == int(OBJ) and val == int(CTRL) and (int(rr), int(cc)) in base_support:
                out[int(rr - rr0), int(cc - cc0)] = int(val)
                continue
            if cur == int(CTRL) and val == int(OBJ) and (int(rr), int(cc)) in base_conn:
                out[int(rr - rr0), int(cc - cc0)] = int(val)
                continue
            return None
        out[int(rr - rr0), int(cc - cc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, BG)


def _candidate_outputs(base_arr: np.ndarray, plug_input_arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, np.ndarray]] = []
    for rot_k in (1, 2, 3):
        rotated = _crop_non_bg(np.rot90(plug_input_arr, int(rot_k)), BG)
        if rotated is None:
            continue
        assembled = _assemble_horizontal_docked(base_arr, rotated)
        if assembled is not None:
            candidates.append((int(rot_k), assembled))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the plug by a non-zero quarter-turn until its vertical connector can
    dock horizontally to the base's vertical connector, then recolor ctrl -> obj
    and tight-crop.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != int(BG), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_arr, (base_r0, base_c0) = _sprite_from_points(g, base_pts, BG)
    plug_arr, _ = _sprite_from_points(g, plug_pts, BG)
    assert int(base_r0) == 0 and int(base_c0) == 0

    candidates = _candidate_outputs(base_arr, plug_arr)
    assert len(candidates) == 1
    return candidates[0][1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[int(r + dr), int(c + dc)]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = BG, OBJ, CTRL

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    candidates: list[tuple[int, int, int, int]] = []
    for side in (-1, 1):
        for conn_len in range(2, min(3, base_h) + 1):
            for conn_col in range(base_w):
                body_col = int(conn_col + side)
                free_col = int(conn_col - side)
                if not (0 <= body_col < base_w):
                    continue
                for conn_r0 in range(base_h - conn_len + 1):
                    rows = list(range(conn_r0, conn_r0 + conn_len))
                    if not all(base[int(r), int(conn_col)] for r in rows):
                        continue
                    if not all(base[int(r), int(body_col)] for r in rows):
                        continue
                    if any(0 <= free_col < base_w and base[int(r), int(free_col)] for r in rows):
                        continue
                    candidates.append((int(side), int(conn_len), int(conn_r0), int(conn_col)))
    if not candidates:
        return None
    side, conn_len, conn_r0, conn_col = _rng.choice(candidates)
    conn_rows = list(range(conn_r0, conn_r0 + conn_len))

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for r in conn_rows:
        base_grid[int(r), int(conn_col)] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(max(conn_len, 3), 7)
    plug_w = _rng.randint(2, 10)
    plug_body_side = int(-side)
    if plug_body_side == 1:
        if plug_w < 2:
            return None
        plug_conn_col = _rng.randint(0, plug_w - 2)
    else:
        if plug_w < 2:
            return None
        plug_conn_col = _rng.randint(1, plug_w - 1)
    plug_conn_r0 = _rng.randint(0, plug_h - conn_len)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    connector_pts: set[tuple[int, int]] = set()
    plug_support_pts: set[tuple[int, int]] = set()
    keep_plug: set[tuple[int, int]] = set()
    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        connector_pts.add((int(r), int(plug_conn_col)))
        plug_support_pts.add((int(r), int(plug_conn_col + plug_body_side)))
        keep_plug.add((int(r), int(plug_conn_col)))
        keep_plug.add((int(r), int(plug_conn_col + plug_body_side)))
        plug[int(r), int(plug_conn_col)] = True
        plug[int(r), int(plug_conn_col + plug_body_side)] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(max(5, int(plug.sum())), plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep_plug)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    free_col = int(plug_conn_col - plug_body_side)
    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        if not plug[int(r), int(plug_conn_col + plug_body_side)]:
            return None
        if 0 <= free_col < plug_w and plug[int(r), int(free_col)]:
            return None

    dock_r = int(conn_r0 - plug_conn_r0)
    dock_c = int(conn_col + side - plug_conn_col)
    base_support = {(int(r), int(conn_col + side)) for r in conn_rows}
    base_conn = {(int(r), int(conn_col)) for r in conn_rows}

    overlap_ctrl: set[tuple[int, int]] = set()
    overlap_body: set[tuple[int, int]] = set()
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[int(r), int(c)]:
                continue
            dr = int(dock_r + r)
            dc = int(dock_c + c)
            if not (0 <= dr < base_h and 0 <= dc < base_w):
                continue
            cur = int(base_grid[int(dr), int(dc)])
            if cur == int(bg):
                continue
            if cur == int(obj) and (int(r), int(c)) in connector_pts and (int(dr), int(dc)) in base_support:
                overlap_ctrl.add((int(dr), int(dc)))
                continue
            if cur == int(ctrl) and (int(r), int(c)) in plug_support_pts and (int(dr), int(dc)) in base_conn:
                overlap_body.add((int(dr), int(dc)))
                continue
            return None
    if overlap_ctrl != base_support or overlap_body != base_conn:
        return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for r, c in connector_pts:
        plug_grid[int(r), int(c)] = ctrl
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    expected = _assemble_horizontal_docked(base_sprite, plug_sprite)
    if expected is None:
        return None

    rot_choices: list[np.ndarray] = []
    for input_rot in (1, 3):
        plug_input = _crop_non_bg(np.rot90(plug_sprite, int(input_rot)), bg)
        if plug_input is None:
            continue
        found = _candidate_outputs(base_sprite, plug_input)
        need_rot = int((-input_rot) % 4)
        if len(found) == 1 and found[0][0] == int(need_rot) and np.array_equal(found[0][1], expected):
            rot_choices.append(plug_input)
    if not rot_choices:
        return None
    plug_input = _rng.choice(rot_choices)

    bh, bw = base_sprite.shape
    ph, pw = plug_input.shape
    min_canvas_h = max(7, max(bh, ph) + 2)
    min_canvas_w = max(5, max(bw, pw) + 2)
    if min_canvas_h > 11 or min_canvas_w > 13:
        return None
    canvas_h = _rng.randint(min_canvas_h, 11)
    canvas_w = _rng.randint(min_canvas_w, 13)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(
            canvas,
            plug_input,
            transparent=bg,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    if int(pr) + int(pc) == 0:
        return None
    paste(canvas, plug_input, int(pr), int(pc), transparent=bg)

    if int(np.sum(canvas == bg)) <= int(canvas.size // 2):
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

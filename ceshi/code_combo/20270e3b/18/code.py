# Combo variant of 20270e3b (index 18)
# combo_spec: D2/0+D6/1
# Name: 垂直连接器朝向判角色

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
------------------
- 原题里 connector 是横条，plug 从上往下对接；base 由输入位置判定。
- 本融合版把 connector 改成竖条，dock 方向改成左右；并且 base 不再看位置，
  而是看 connector 的自由侧朝向。自由侧朝右者是 base，因此它的 body 一定在
  connector 左边；另一只 sprite 的自由侧朝左，因此它的 body 在 connector 右边。
- 具体小例子：若 base connector 在第 4 列、长度 3，body 紧贴第 3 列；plug 的
  connector 也长度 3，body 紧贴 connector 右侧。对接时 plug connector 被平移到
  base connector 左边一列，于是发生两列宽的接缝：
  * plug 的 ctrl 覆盖 base 的 obj；
  * plug 的 obj 覆盖 base 的 ctrl；
  最后全部 ctrl 统一改回 obj 并紧裁剪。
- 与原题相比，变化不只是“横改竖”，还包括“谁固定谁移动”要靠局部朝向判断，而不
  能再看谁贴左上角。

b) 复杂度校准
-------------
- 为避免 1 像素 connector 的朝向歧义，generator 只生成长度 2 或 3 的竖直条。
- 角色规则只保留两档：自由侧朝右的 base，朝左的 plug。这样样例里重复支撑充分，
  但不会出现太多分支。
- 输入摆放故意让 plug 贴左上角、base 放到别处，稳定制造“位置规则失效”的可见差异。

c) 隐式假设审查
----------------
- 原 `main()` 依赖“connector 是水平条”，这里必须改成检查同一列上的连续 ctrl。
- 原 `main()` 依赖“base 贴输入原点”，这里不再成立，相关 bbox-top-left 判定和
  “输出左上角必须等于 base 左上角”的断言都必须删除。
- 原平移公式是按行错开 1 格；这里改成按列错开 1 格，并且 connector 行范围对齐。
- 新 solver 的额外假设是：每个 sprite 的 ctrl 集合都能被唯一判成“body 在左”或
  “body 在右”；generator 通过构造和校验保证这点。

d) 放置合法性
-------------
- generator 先构造 docked 形态，再裁成两个 sprite 单独摆回输入。
- 输入里 plug 固定放在 `(0,0)`，base 用 `padding=1` 随机放到别处，保证两者既不
  重叠，也不在 8 邻域内接触。
- 组装阶段只允许 connector 接缝发生重叠；任何 ctrl 之外的实体重叠都会被拒绝。

e) 视觉可观察性
----------------
- 竖直 connector 长度固定为 2-3，人眼能稳定看出“这是竖条而不是横条”。
- 样例里经常会看到左上角那只 sprite 最终被移动，说明旧的“左上角是 base”规则不对；
  真正稳定的线索是 connector 哪一侧朝外。

f) 可视化整洁度
----------------
- 仍然只用三种颜色 `bg=1 / obj=4 / ctrl=7`，没有额外标记。
- 接缝只占两列，重叠关系清楚，不会出现大面积覆盖导致难以分辨。

g) 变化的可推断性
------------------
- 颜色变化没有增加复杂性：只有 connector 颜色 `7 -> 4`。
- 颜色变化紧跟在 connector 所在位置上，且 connector 的形状和朝向在输入里可见，
  因而人类能从少量样例归纳出规则。

h) 对称性与退化审查
--------------------
- 这个组合有一条必须明说的几何对称：两只 sprite 的 connector 左右朝向相反时，
  交换“谁是 base、谁是 plug”不会改变 tight-crop 后的最终并合形态。
- 所以 D6 在这里主要体现在“输入中的角色识别线索”，而不是一个会改变最终像素结果的
  独立控制变量。generator 无法通过“新旧输出必须不同”来筛样，否则几乎会把样本全拒绝。
- 可量化的融合行为改为：输入里左上角 sprite 稳定不是 base，同时真正的 base 的
  connector 自由侧稳定朝右。这样位置线索与角色线索明确冲突，融合特征仍然可观察。

i) 新 `main()` 假设与 generator 保证
-------------------------------------
- `main()` 假设：恰有两个 8 连通前景组件；每个组件恰有一个连续竖直 ctrl 条；两条
  connector 等长；其中一条 body 在左、另一条 body 在右。
- `generate_input()` 分别通过候选筛选、保留关键像素 thinning、对接重叠校验和
  “plug 贴左上、base 放到其他位置”的摆放策略来保证这些前提都满足。
"""


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assert_vertical_bar(shape: set[tuple[int, int]]) -> None:
    cols = {int(c) for _r, c in shape}
    assert len(cols) == 1
    rows = sorted(int(r) for r, _c in shape)
    assert rows == list(range(rows[0], rows[-1] + 1))


def _connector_body_side(
    sprite: set[tuple[int, int]], conn_pts: set[tuple[int, int]]
) -> int:
    """
    Return the horizontal side on which the sprite body touches the connector.

    -1 => body on the left, free side on the right
     1 => body on the right, free side on the left
    """
    body = sprite - conn_pts
    valid: list[int] = []
    for side in (-1, 1):
        if all((int(r), int(c + side)) in body for r, c in conn_pts) and all(
            (int(r), int(c - side)) not in body for r, c in conn_pts
        ):
            valid.append(int(side))
    assert len(valid) == 1
    return int(valid[0])


def _extract_parts(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
    int,
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
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

    shpA, _ = _norm_conn(connA)
    shpB, _ = _norm_conn(connB)
    _assert_vertical_bar(shpA)
    _assert_vertical_bar(shpB)
    assert shpA == shpB

    return g, bg, obj, ctrl, A, B, connA, connB


def _render_assembled(
    g: np.ndarray,
    bg: int,
    obj: int,
    ctrl: int,
    base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    connD: set[tuple[int, int]],
    connP: set[tuple[int, int]],
) -> np.ndarray:
    base_side = _connector_body_side(base, connD)
    plug_side = _connector_body_side(plug, connP)
    assert plug_side == -base_side

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)

    tr = int(dr0 - pr0)
    tc = int((dc0 + base_side) - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
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
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _solve(grid: np.ndarray, role_mode: str) -> np.ndarray:
    g, bg, obj, ctrl, A, B, connA, connB = _extract_parts(grid)

    if role_mode == "free_side":
        sideA = _connector_body_side(A, connA)
        sideB = _connector_body_side(B, connB)
        assert {sideA, sideB} == {-1, 1}
        if sideA == -1:
            base, connD = A, connA
            plug, connP = B, connB
        else:
            base, connD = B, connB
            plug, connP = A, connA
        assert _connector_body_side(base, connD) == -1
        assert _connector_body_side(plug, connP) == 1
        return _render_assembled(g, bg, obj, ctrl, base, plug, connD, connP)

    if role_mode == "position":
        r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
        r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
        if int(r0a + c0a) <= int(r0b + c0b):
            base, connD = A, connA
            plug, connP = B, connB
        else:
            base, connD = B, connB
            plug, connP = A, connA
        return _render_assembled(g, bg, obj, ctrl, base, plug, connD, connP)

    raise ValueError(f"unknown role_mode: {role_mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching vertical connectors, then tight-crop.

    Fusion rule:
    - Each sprite has a vertical ctrl-colored connector bar.
    - The sprite whose connector's free side points right is the base.
    - The other sprite has free side left and docks horizontally from the left.
    - After docking, connector color disappears (`ctrl -> obj`) and the result is tight-cropped.
    """
    return _solve(grid, role_mode="free_side")


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
        frontier.append((int(nr), int(nc)))


def _grow_shape_allowed(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray,
) -> bool:
    count = int(mask.sum())
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
            and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((int(nr), int(nc)))
    return count == int(target)


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    h, w = mask.shape
    for r, c in pixels:
        key = (int(r), int(c))
        if key in keep:
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


def _crop_nonbg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    base_grid = None
    conn_len = -1
    conn_r0 = -1
    conn_col = -1
    for _ in range(24):
        base_h = _rng.randint(3, 7)
        base_w = _rng.randint(3, 10)
        base_target = _rng.randint(5, min(29, base_h * base_w))
        base = np.zeros((base_h, base_w), dtype=bool)
        base[0, 0] = True
        _grow_shape(base, base_target, [(0, 0)])
        if not is_contiguous(base, background=0, connectivity=8):
            continue

        candidates: list[tuple[int, int, int]] = []
        for cand_len in range(2, min(3, base_h) + 1):
            for cand_col in range(1, base_w):
                body_col = int(cand_col - 1)
                free_col = int(cand_col + 1)
                for cand_r0 in range(base_h - cand_len + 1):
                    rows = list(range(cand_r0, cand_r0 + cand_len))
                    if not all(base[r, cand_col] for r in rows):
                        continue
                    if not all(base[r, body_col] for r in rows):
                        continue
                    if free_col < base_w and any(base[r, free_col] for r in rows):
                        continue
                    candidates.append((int(cand_len), int(cand_r0), int(cand_col)))
        if not candidates:
            continue

        conn_len, conn_r0, conn_col = _rng.choice(candidates)
        keep_base = {
            (int(r), int(conn_col)) for r in range(conn_r0, conn_r0 + conn_len)
        } | {
            (int(r), int(conn_col - 1)) for r in range(conn_r0, conn_r0 + conn_len)
        }
        _thin(base, keep=keep_base)

        rows = list(range(conn_r0, conn_r0 + conn_len))
        if not all(base[r, conn_col] for r in rows):
            continue
        if not all(base[r, conn_col - 1] for r in rows):
            continue
        if conn_col + 1 < base_w and any(base[r, conn_col + 1] for r in rows):
            continue

        base_grid = np.full((base_h, base_w), bg, dtype=int)
        base_grid[base] = obj
        for r in rows:
            base_grid[r, conn_col] = ctrl
        if not is_contiguous(base_grid != bg, background=0, connectivity=8):
            base_grid = None
            continue
        break
    if base_grid is None:
        return None

    plug_h = _rng.randint(conn_len, 7)
    plug_w = _rng.randint(2, 10)
    plug_conn_col = _rng.randint(0, plug_w - 2)
    plug_conn_r0 = _rng.randint(0, plug_h - conn_len)
    dock_r = int(conn_r0 - plug_conn_r0)
    dock_c = int((conn_col - 1) - plug_conn_col)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    connector_pts: set[tuple[int, int]] = set()
    keep_plug: set[tuple[int, int]] = set()
    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        connector_pts.add((int(r), int(plug_conn_col)))
        keep_plug.add((int(r), int(plug_conn_col)))
        keep_plug.add((int(r), int(plug_conn_col + 1)))
        plug[r, plug_conn_col] = True
        plug[r, plug_conn_col + 1] = True

    allowed = np.ones((plug_h, plug_w), dtype=bool)
    for r in range(plug_h):
        for c in range(plug_w):
            dr = int(dock_r + r)
            dc = int(dock_c + c)
            is_connector_pixel = (int(r), int(c)) in connector_pts
            if (
                0 <= dr < base_grid.shape[0]
                and 0 <= dc < base_grid.shape[1]
                and int(base_grid[dr, dc]) == int(obj)
                and not is_connector_pixel
            ):
                allowed[r, c] = False
    if not all(bool(allowed[r, c]) for r, c in keep_plug):
        return None

    plug_max = min(29, plug_h * plug_w)
    plug_cap = min(plug_max, int(np.sum(allowed)))
    if plug_cap < max(5, int(plug.sum())):
        return None
    plug_target = _rng.randint(max(5, int(plug.sum())), plug_cap)
    if not _grow_shape_allowed(plug, plug_target, list(zip(*np.where(plug))), allowed):
        return None
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep_plug)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        if not bool(plug[r, plug_conn_col + 1]):
            return None
        if plug_conn_col > 0 and bool(plug[r, plug_conn_col - 1]):
            return None

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(dock_r + r)
            dc = int(dock_c + c)
            if (
                0 <= dr < base_grid.shape[0]
                and 0 <= dc < base_grid.shape[1]
                and int(base_grid[dr, dc]) == int(obj)
                and (int(r), int(c)) not in connector_pts
            ):
                return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for r, c in connector_pts:
        plug_grid[r, c] = ctrl
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_nonbg(base_grid, bg)
    plug_sprite = _crop_nonbg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    can_stack_vertical = ph + 1 + bh <= 11
    can_stack_horizontal = pw + 1 + bw <= 13
    if not (can_stack_vertical or can_stack_horizontal):
        return None

    if can_stack_vertical and can_stack_horizontal:
        mode = _rng.choice(["vertical", "horizontal"])
    elif can_stack_vertical:
        mode = "vertical"
    else:
        mode = "horizontal"

    if mode == "vertical":
        canvas_h = _rng.randint(max(7, ph + 1 + bh), 11)
        canvas_w = _rng.randint(max(3, max(pw, bw)), 13)
    else:
        canvas_h = _rng.randint(max(7, max(ph, bh)), 11)
        canvas_w = _rng.randint(max(3, pw + 1 + bw), 13)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, plug_sprite, 0, 0, transparent=bg)
    try:
        br, bc = random_free_location(canvas, base_sprite, transparent=bg, padding=1)
    except ValueError:
        return None
    if int(br + bc) == 0:
        return None
    paste(canvas, base_sprite, br, bc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None

    out_h, out_w = map(int, result.shape)
    if not (5 <= out_h <= 8 and 3 <= out_w <= 12):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")

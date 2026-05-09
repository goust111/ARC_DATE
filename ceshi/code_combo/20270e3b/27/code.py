# Combo variant of 20270e3b (index 27)
# combo_spec: D4/1+D8/2+D13/0
# Name: 一对多间距重叠对接
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题只有 1 个 base 和 1 个 plug。plug 的 ctrl 条被平移到 base ctrl 条正上方 1 行，
#      然后 ctrl 最终被抹成 obj，所以接口“消失”。
#    - 本融合版有 1 个 base 和 2-3 个 plug。每个 plug 仍靠水平 ctrl 条匹配，
#      但改成 D4/1 的“同一行对接”：plug 的 connector 与匹配到的 base connector
#      落在同一输出行，形成一条 1 像素厚的可见 seam。再叠加 D13/0：plug body 允许压进
#      base body，也允许压到更早放置的 plug body 上，后放置者覆盖先放置者。
#    - 一个具体小例子：base 上有三条 connector，行号分别为 2/4/6，长度分别为 3/2/1，
#      且三者共享最右列。最上面的 plug 沿共享列向下长出一根 stem，会先盖住 base 的下方 rib，
#      也会先穿过较低 connector 所在的行；随后较低 connector 对应的 plug 再贴上去，
#      它的 ctrl seam 会把先前那根 stem 局部盖回 ctrl。这样就同时体现了“同一行对接”、
#      “一对多匹配”和“顺序相关覆盖”。
#    - 与原题的差异有三处联动：
#      1. connector 对齐规则从 “base 上方 1 行” 改成 “同一行”；
#      2. base 不再只有 1 个 connector，而是有 2-3 个唯一长度的 connector；
#      3. 渲染不再禁止深层重叠，而是按匹配 connector 的阅读顺序（上到下、左到右）逐个盖上。
# b) 复杂度校准
#    - 若多个 base connector 长度重复，plug 匹配会歧义；因此 generator 固定使用
#      {1,2,3} 中互不相同的长度子集。
#    - 若所有 connector 在同一行，plug-plug 覆盖大多会退化成同色 body/body 重叠，
#      顺序在输出上不明显；因此 generator 把 connector 放在不同高度，并让上层 plug 的 stem
#      穿过下层 seam，使覆盖顺序直接出现在最终像素里。
#    - 若仍把所有 ctrl 最终改回 obj，那么 plug 的覆盖优先级对输出几乎不可见。
#      因此这里采用 D4/1 的具体语义：seam 保留为 ctrl。这样 D13 的“后者覆盖前者”
#      才能在最终图像中留下可观察差异。若改回 obj，则该融合会退化。
# c) 隐式假设审查
#    - 原 `assert len(best_comps) == 2` 失效：现在是 1 个 base + 多个 plug。
#    - 原来每个精灵只有 1 组 ctrl；现在 base 内部要再做一次 4 连通分解，得到多个 connector。
#    - 原 `assert shpA == shpB` 失效：现在变成“每个 plug 的 connector 与某个 base connector
#      单独匹配”，并且依赖 generator 保证长度唯一。
#    - 原 `tr = (dr0 - 1) - pr0` 失效：D4/1 改成 `tr = dr0 - pr0`。
#    - 原 overlap 断言 `cur == ctrl or val == ctrl` 失效：D13 允许 body-body 覆盖，
#      这里直接按顺序覆盖，而不是拒绝重叠。
#    - 新增顺序假设：plug 的覆盖顺序按匹配到的 base connector 的 `(row, col)` 排序。
#      generator 会按同一顺序构造样本，并且要求“正序渲染”与“逆序渲染”输出不同。
# d) 放置合法性
#    - generator 先在 assembled 坐标中构造 base 与所有已对接 plug，再分别裁成独立 sprite，
#      最后把 base 固定放到输入左上角，把各 plug 随机散到别处，使用 `padding=1`
#      和 8 连通检查保证输入里组件互不粘连。
#    - 构造时保证每个 plug 裁剪后 connector 在自身最上行、base 仍是 assembled 图像左上角，
#      因此 solver 的 tight crop 与 role 识别稳定。
# e) 视觉可观察性
#    - 输入中能直接看到“一个带多个端口的 base”和“多个单端口 plug”。
#    - 输出中能看到多个 ctrl seam 留下来的水平条，且较低的 seam 会覆盖较高 plug 的 stem，
#      人类可从 2-4 组样例归纳出：每个 plug 各找同长度端口、同一行贴合、并按端口顺序叠加。
# f) 可视化整洁度
#    - 虽然允许重叠，但重叠集中在共享竖干和少量 rib 交叉处，且 connector 长度只取 1/2/3，
#      整体仍是清晰的“框架 + 竖干 + seam”结构，不会变成杂乱噪声。
# g) 变化的可推断性
#    - 颜色仍固定为 `bg=1, obj=4, ctrl=7`，没有额外颜色映射。
#    - ctrl 只承担“对接 seam”这一结构职责；它被保留是为了让一对多重叠的优先级可见，
#      不是任意换色。
#
# Main() 的新增隐式假设及 generator 保证
# ---------------------------------------
# - 假设每个 plug 恰有 1 个水平 ctrl 条；generator 按此构造。
# - 假设 base 恰有与 plug 数量相同的水平 ctrl 条，且长度互异；generator 按此构造。
# - 假设 base 位于输入左上角；generator 固定把 base paste 到 (0, 0)。
# - 假设 plug 裁剪后 connector 在局部最上方，且 body 在其下方；generator 先 assembled 再裁剪，
#   保证这个局部朝向。
# - 假设按 base connector 的 `(row, col)` 顺序渲染就能得到唯一输出；generator 要求
#   forward-order 与 reverse-order 的输出不同，从而避免“顺序无关”的退化样本。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _ctrl_components(sprite: set[tuple[int, int]], grid: np.ndarray, ctrl: int = CTRL) -> list[set[tuple[int, int]]]:
    mask = np.zeros(grid.shape, dtype=bool)
    for r, c in sprite:
        if int(grid[int(r), int(c)]) == int(ctrl):
            mask[int(r), int(c)] = True
    lab, n = label_mask(mask, connectivity=4)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    return comps


def _normalize_connector(conn_pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    assert conn_pts
    rows = sorted({int(r) for r, _c in conn_pts})
    assert len(rows) == 1
    cols = sorted(int(c) for _r, c in conn_pts)
    assert cols == list(range(cols[0], cols[-1] + 1))
    r0 = int(rows[0])
    c0 = int(cols[0])
    shape = frozenset((0, int(c - c0)) for c in cols)
    return shape, (r0, c0)


def _collect_components(grid: np.ndarray, bg: int = BG) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(grid != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    return comps


def _render_ordered(
    base_grid: np.ndarray,
    plug_grids: list[np.ndarray],
    order: list[int],
    bg: int = BG,
) -> np.ndarray:
    out = np.array(base_grid, copy=True)
    for idx in order:
        plug = plug_grids[int(idx)]
        mask = plug != int(bg)
        out[mask] = plug[mask]
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock multiple plug sprites onto the base by matching horizontal connector bars.

    The matched plug connector is translated onto the same row as the base connector
    (visible seam). Plug sprites are stamped in base-connector reading order, and
    later plugs overwrite earlier ones as well as the base.
    """
    g = np.array(grid, dtype=int)
    bg, ctrl = BG, CTRL

    comps = _collect_components(g, bg)
    assert len(comps) >= 3

    boxes = [_bbox_of_pts(pts) for pts in comps]
    base_idx = min(range(len(comps)), key=lambda i: int(boxes[i][0] + boxes[i][1]))
    base = comps[base_idx]
    assert int(boxes[base_idx][0] + boxes[base_idx][1]) == 0
    plugs = [pts for i, pts in enumerate(comps) if i != base_idx]
    assert 2 <= len(plugs) <= 3

    base_conns = _ctrl_components(base, g, ctrl)
    assert len(base_conns) == len(plugs)
    base_body = base - set().union(*base_conns)

    base_map: dict[frozenset[tuple[int, int]], dict[str, object]] = {}
    ordered_shapes: list[frozenset[tuple[int, int]]] = []
    for conn in sorted(base_conns, key=lambda pts: _normalize_connector(pts)[1]):
        shape, anchor = _normalize_connector(conn)
        assert shape not in base_map
        for r, c in conn:
            assert (int(r - 1), int(c)) in base_body
            assert (int(r + 1), int(c)) not in base_body
        base_map[shape] = {"conn": conn, "anchor": anchor}
        ordered_shapes.append(shape)

    shifted_plugs: list[tuple[int, set[tuple[int, int]], set[tuple[int, int]], int, int]] = []
    used_shapes: set[frozenset[tuple[int, int]]] = set()

    for plug in plugs:
        plug_conns = _ctrl_components(plug, g, ctrl)
        assert len(plug_conns) == 1
        plug_conn = plug_conns[0]
        plug_body = plug - plug_conn
        shape, (pr0, pc0) = _normalize_connector(plug_conn)
        assert shape in base_map
        assert shape not in used_shapes
        used_shapes.add(shape)

        for r, c in plug_conn:
            assert (int(r + 1), int(c)) in plug_body
            assert (int(r - 1), int(c)) not in plug_body

        dr0, dc0 = base_map[shape]["anchor"]  # type: ignore[index]
        tr = int(dr0 - pr0)
        tc = int(dc0 - pc0)
        plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
        order_idx = ordered_shapes.index(shape)
        shifted_plugs.append((order_idx, plug, plug_shift, tr, tc))

    assert len(used_shapes) == len(base_map)

    shifted_plugs.sort(key=lambda item: int(item[0]))
    pts_all = set(base)
    for _order_idx, _plug, plug_shift, _tr, _tc in shifted_plugs:
        pts_all |= plug_shift
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(np.array(list(pts_all), dtype=int)))

    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for _order_idx, plug, _plug_shift, tr, tc in shifted_plugs:
        for r, c in plug:
            rr = int(r + tr - rr0)
            cc = int(c + tc - cc0)
            out[rr, cc] = int(g[int(r), int(c)])

    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


def _build_base() -> tuple[np.ndarray, list[dict[str, object]], int, int] | None:
    n_plugs = _rng.randint(2, 3)
    lengths = sorted(_rng.sample([1, 2, 3], n_plugs), reverse=True)

    top_row = 1 if n_plugs == 3 else _rng.randint(1, 2)
    step = 3
    rows = [int(top_row + step * i) for i in range(n_plugs)]
    bottom = int(rows[-1] + _rng.randint(2, 3))

    anchor_c = _rng.randint(2, 4)
    right_wall = int(anchor_c + _rng.randint(2, 4))
    width = int(right_wall + 1)
    height = int(bottom + 1)
    if height > 11 or width > 11:
        return None

    connectors: list[dict[str, object]] = []
    for row, conn_len in zip(rows, lengths):
        cols = list(range(int(anchor_c - conn_len + 1), int(anchor_c + 1)))
        if cols[0] < 0:
            return None
        connectors.append({"row": int(row), "cols": cols, "len": int(conn_len)})

    base_mask = np.zeros((height, width), dtype=bool)
    base_mask[0, 0:width] = True
    base_mask[:, 0] = True
    base_mask[0 : bottom + 1, right_wall] = True
    base_mask[bottom, 0 : right_wall + 1] = True

    for item in connectors:
        row = int(item["row"])
        cols = list(item["cols"])
        for c in cols:
            base_mask[0:row, int(c)] = True

    rib_rows = sorted({int(row + 2) for row in rows if int(row + 2) < bottom})
    for rr in rib_rows:
        rib_end = _rng.randint(anchor_c, right_wall)
        base_mask[int(rr), 0 : int(rib_end) + 1] = True

    if bottom - 1 > rows[0] + 1 and _rng.random() < 0.7:
        base_mask[int(rows[0] + 1) : bottom + 1, anchor_c] = True

    for item in connectors:
        row = int(item["row"])
        for c in item["cols"]:
            if row + 1 < height:
                base_mask[int(row + 1), int(c)] = False

    base_grid = np.full((height, width), int(BG), dtype=int)
    base_grid[base_mask] = int(OBJ)
    for item in connectors:
        row = int(item["row"])
        for c in item["cols"]:
            base_grid[int(row), int(c)] = int(CTRL)

    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != int(BG))) < 8:
        return None
    return base_grid, connectors, anchor_c, bottom


def _build_plug_grid(
    shape: tuple[int, int],
    row: int,
    cols: list[int],
    anchor_c: int,
    bottom: int,
) -> np.ndarray | None:
    height, width = map(int, shape)
    plug_mask = np.zeros((height, width), dtype=bool)

    for c in cols:
        plug_mask[int(row), int(c)] = True
        if row + 1 >= height:
            return None
        plug_mask[int(row + 1), int(c)] = True

    stem_col = int(anchor_c)
    plug_mask[int(row + 1) : int(bottom) + 1, stem_col] = True

    branch_rows = {int(bottom), min(int(bottom), int(row + 2))}
    for rr in branch_rows:
        left_span = _rng.randint(0, min(3, stem_col))
        right_span = _rng.randint(1, min(3, width - 1 - stem_col))
        plug_mask[int(rr), int(stem_col - left_span) : int(stem_col + right_span) + 1] = True

    if len(cols) >= 2 and row + 2 <= bottom:
        side_row = _rng.randint(int(row + 2), int(bottom))
        plug_mask[int(side_row), int(cols[0]) : int(stem_col) + 1] = True

    plug_grid = np.full((height, width), int(BG), dtype=int)
    plug_grid[plug_mask] = int(OBJ)
    for c in cols:
        plug_grid[int(row), int(c)] = int(CTRL)

    if np.any(plug_grid[:row, :] != int(BG)):
        return None
    if not is_contiguous(plug_grid != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(plug_grid != int(BG))) < 5:
        return None
    return plug_grid


def _place_input_sprites(base_sprite: np.ndarray, plug_sprites: list[np.ndarray]) -> np.ndarray | None:
    sprites = list(plug_sprites)
    _rng.shuffle(sprites)

    max_h = max([base_sprite.shape[0]] + [sprite.shape[0] for sprite in sprites])
    max_w = max([base_sprite.shape[1]] + [sprite.shape[1] for sprite in sprites])
    total_w = int(base_sprite.shape[1] + sum(sprite.shape[1] for sprite in sprites))

    min_h = max(9, int(max_h + 2))
    min_w = max(12, int(max_w + 3), min(17, int(total_w + 2)))
    if min_h > 14 or min_w > 17:
        return None

    for _ in range(50):
        canvas_h = _rng.randint(min_h, 14)
        canvas_w = _rng.randint(min_w, 17)
        canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
        paste(canvas, base_sprite, 0, 0, transparent=BG)

        ok = True
        for sprite in sprites:
            try:
                pr, pc = random_free_location(
                    canvas,
                    sprite,
                    transparent=BG,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            if pr + pc == 0:
                ok = False
                break
            paste(canvas, sprite, pr, pc, transparent=BG)
        if not ok:
            continue

        if int(label_mask(canvas != int(BG), connectivity=8)[1]) != int(1 + len(sprites)):
            continue
        return canvas
    return None


def _behavior_metrics(base_grid: np.ndarray, plug_grids: list[np.ndarray], order: list[int]) -> dict[str, int]:
    plug_union_prev = np.zeros(base_grid.shape, dtype=bool)
    deep_base_overlap = 0
    plug_plug_overlap = 0
    ctrl_over_body = 0

    ordered_rows = [int(np.argwhere(plug_grids[idx] == int(CTRL))[:, 0].min()) for idx in order]
    for pos, idx in enumerate(order):
        plug = plug_grids[int(idx)]
        plug_fg = plug != int(BG)
        plug_plug_overlap += int(np.sum(plug_fg & plug_union_prev))
        ctrl_over_body += int(np.sum((plug == int(CTRL)) & plug_union_prev))
        deep_rows = np.arange(base_grid.shape[0])[:, None] > int(ordered_rows[pos])
        deep_base_overlap += int(np.sum((base_grid == int(OBJ)) & plug_fg & deep_rows))
        plug_union_prev |= plug_fg

    forward = _render_ordered(base_grid, plug_grids, order)
    reverse = _render_ordered(base_grid, plug_grids, list(reversed(order)))
    order_sensitive_pixels = int(np.sum(forward != reverse))
    ctrl_pixels = int(np.sum(forward == int(CTRL)))
    return {
        "deep_base_overlap": int(deep_base_overlap),
        "plug_plug_overlap": int(plug_plug_overlap),
        "ctrl_over_body": int(ctrl_over_body),
        "order_sensitive_pixels": int(order_sensitive_pixels),
        "ctrl_pixels": int(ctrl_pixels),
    }


def _gen_attempt() -> np.ndarray | None:
    built = _build_base()
    if built is None:
        return None
    base_grid, connectors, anchor_c, bottom = built

    plug_grids: list[np.ndarray] = []
    for item in connectors:
        plug_grid = _build_plug_grid(base_grid.shape, int(item["row"]), list(item["cols"]), anchor_c, bottom)
        if plug_grid is None:
            return None
        plug_grids.append(plug_grid)

    order = list(range(len(connectors)))
    metrics = _behavior_metrics(base_grid, plug_grids, order)
    if metrics["deep_base_overlap"] <= 0:
        return None
    if metrics["plug_plug_overlap"] <= 0:
        return None
    if metrics["ctrl_over_body"] <= 0:
        return None
    if metrics["order_sensitive_pixels"] <= 0:
        return None
    if metrics["ctrl_pixels"] <= len(connectors):
        return None

    expected = _render_ordered(base_grid, plug_grids, order)
    if expected.shape[0] < 6 or expected.shape[1] < 5 or expected.shape[0] > 11 or expected.shape[1] > 11:
        return None

    base_sprite = _crop_non_bg(base_grid, BG)
    plug_sprites = [_crop_non_bg(plug_grid, BG) for plug_grid in plug_grids]
    if base_sprite is None or any(sprite is None for sprite in plug_sprites):
        return None

    canvas = _place_input_sprites(base_sprite, [sprite for sprite in plug_sprites if sprite is not None])
    if canvas is None:
        return None
    if np.sum(canvas == int(BG)) <= canvas.size // 2:
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

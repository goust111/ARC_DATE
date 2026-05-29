# Combo variant of 20270e3b (index 30)
# combo_spec: D1/0+D2/0+D8/1+D10/1
# Name: 多色垂直连接器三精灵水平链

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
design_analysis:
  variant_summary:
    kept:
      - body 色仍固定为 obj=4，背景固定为 bg=1，输出仍把所有 connector 颜色统一重涂为 body 色。
      - 输入仍是若干分离的 8 连通精灵，输出仍是把它们按 connector 规则拼接后做 tight crop。
      - 仍然保留原题的“一个锚定在输入左上角的起始精灵 + 其余精灵散落在别处”的可见角色划分。
    changed:
      - 从 2 精灵改为 3 精灵 A-B-C 链式拼接；A 是左端起点，B 是中间件，C 是右端终点。
      - connector 从水平条改为垂直条，推理方向从上下改为左右。
      - B 拥有两个 connector：左侧 connector 与 A 同色，右侧 connector 与 C 同色；两个配对颜色不同。
      - 配对依据从“同形”改为“同色”，并且两段 seam 都强制异长：A_len != B_left_len，B_right_len != C_len。

  a_concretization_and_trace:
    input_look:
      - A 只有 1 个竖直 connector，放在自身最右边界，body 在其左侧，因此它是左端件。
      - B 有 2 个竖直 connector：左边界一个、右边界一个，颜色分别与 A、C 对应。
      - C 只有 1 个竖直 connector，放在自身最左边界，body 在其右侧，因此它是右端件。
      - 例：A 的红色 connector 长 3，B 左边的红色 connector 长 2；B 右边的绿色 connector 长 3，C 的绿色 connector 长 2。
    behavior_trace:
      - 原题：只有 base + plug 两块；取同形水平 connector，把 plug 平移到 base 上方一格。
      - 本融合题：
        1. 先按颜色找到 A 与 B 左 connector 的配对色，并做第一次水平对接；
        2. 再按另一种颜色找到 B 右 connector 与 C 的配对色，并做第二次水平对接；
        3. 两段对接都使用“竖条 top 对齐、左右相邻成桥”的几何，然后把两种 connector 色都改成 obj。
      - 差异点不止一个：
        * 需要先做颜色匹配，而不是看长度 / 形状；
        * 需要判断 connector 的左右朝向，而不是上下朝向；
        * 需要分两步拼成 A-B-C，而不是一次完成。
    small_example:
      - 若 A 右边界是 3 格红竖条，B 左边界是 2 格红竖条，那么第一段桥由两列相邻的红条组成，重涂后变成 body。
      - 若 B 右边界是 3 格绿竖条，C 左边界是 2 格绿竖条，则第二段桥同理。
      - 输出是三者组成的水平长链，且 connector 颜色完全消失。

  b_complexity_calibration:
    - 为了让“竖直而非水平”肉眼稳定可见，connector 长度只取 2 或 3，不生成长度 1。
    - 为了让“颜色匹配而非形状匹配”稳定可见，generator 强制两段配对都异长。
    - 只使用两种 connector 颜色：AB 共享一种，BC 共享另一种；既足够表达链式配对，又不会让颜色体系过杂。
    - B 只有两个 connector，没有额外干扰 connector，避免把观察难度浪费在无关 distractor 上。

  c_hidden_assumption_review_main:
    - 原 `assert len(best_comps) == 2` 失效：现在必须正好有 3 个前景连通块。
    - 原 `ctrl = 7` 失效：现在需要把所有非 {bg,obj} 的颜色都视为 connector 颜色，并且应当恰有 2 种。
    - 原“每块恰有一个 connector”失效：B 现在有 2 个 connector，所以 solver 必须按 4 连通 connector 组件逐个分类。
    - 原“同形即可配对”失效：这里不能按 shape 决定配对，而是先按 shared color 建图。
    - 原“单步平移”失效：需要先求 A->B 的 shift，再用已平移后的 B 去求 B->C 的 shift。
    - 原“上下对接”相关断言失效：需要改成检查 connector 左右哪一边是 body。
    - 本实现新增的隐式假设：
      * A 是唯一的 origin-anchored 单 connector 精灵，并且 connector 在右边界（body side = left / -1）。
      * B 是唯一的双 connector 精灵，其左 connector 与 A 共享颜色且 body side = right / +1；其右 connector 与 C 共享颜色且 body side = left / -1。
      * C 是剩余的单 connector 精灵，connector 在左边界（body side = right / +1）。
      * 两段对接采用“相邻成桥”而不是像原题那样发生像素覆盖；因此 stamping 时不允许重叠。

  d_placement_legality:
    - generator 先在局部坐标里构造 A、B、C 三个 tight-cropped sprite，并验证它们在组装状态下互不重叠。
    - 输入阶段固定把 A 放在 canvas 的 (0,0)，再把 B、C 用 padding=1 的随机空位散开，确保三块在输入中保持 8 邻域分离。
    - 组装时只允许“桥接相邻”，不允许任何两块真正重叠；这使得异长竖条的尾部也保持干净，不会出现多余遮挡。

  e_visual_observability:
    - 人类能从少量样例里同时看到：
      * 输入里总有 3 块，其中只有一块有 2 个 connector，因此它自然是中间件；
      * 左 pair 与右 pair 的 connector 颜色不同，但各自都在两块精灵之间重复出现；
      * 两段 pair 的 connector 长度故意不同，说明不能靠“同长”去推断；
      * 输出总是变成一条水平长链，且两种 connector 颜色都消失。
    - 整个推理只依赖可见结构：颜色重复、connector 在左右边界的位置、以及 A 固定在输入左上角。

  f_visual_cleanliness:
    - connector 都放在各 sprite 的外边界，且 B 的两个 connector 分居左右边界，视觉分工明确。
    - 对接采用相邻拼桥，没有像素叠压，输出比 overlap 版更干净，尤其适合异长竖条。
    - 颜色总数稳定为 4 种：bg、obj、AB-connector-color、BC-connector-color。

  g_color_inference:
    - 颜色变化有明确结构提示：
      * A 与 B 只共享一种颜色；
      * B 与 C 只共享另一种颜色；
      * 输出中这两种颜色都统一变成 obj。
    - 颜色数量不多，且每种颜色都在 exactly 2 个 sprite 上重复出现，足以让少样本归纳成立。

  h_generator_guarantees_for_new_main_assumptions:
    - 保证恰有 3 个前景组件，且只有 A 在输入原点。
    - 保证恰有 2 种 connector 颜色，分别只用于 AB 与 BC 两段。
    - 保证两段配对都异长：A_len != B_left_len 且 B_right_len != C_len。
    - 保证 A/B/C 的 connector 方向满足 solver 的角色判定，并且组装时无重叠、tight crop 后 A 仍位于输出左上角。
"""


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(arr != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allow=None,
) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h
            and 0 <= int(c + dc) < w
            and not bool(mask[int(r + dr), int(c + dc)])
            and (allow is None or bool(allow(int(r + dr), int(c + dc))))
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        frontier.append((int(nr), int(nc)))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r = int(r)
        c = int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _connector_components(
    g: np.ndarray,
    sprite: set[tuple[int, int]],
    connector_colors: set[int],
) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in sprite:
        if int(g[int(r), int(c)]) in connector_colors:
            mask[int(r - r0), int(c - c0)] = True
    lab, n = label_mask(mask, connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {
            (int(r0 + rr), int(c0 + cc))
            for rr, cc in np.argwhere(lab == int(k))
        }
        comps.append(pts)
    return comps


def _classify_connector(
    g: np.ndarray,
    sprite: set[tuple[int, int]],
    conn_pts: set[tuple[int, int]],
    obj: int,
) -> dict[str, object]:
    assert conn_pts
    colors = {int(g[int(r), int(c)]) for r, c in conn_pts}
    assert len(colors) == 1
    color = int(next(iter(colors)))

    cols = sorted({int(c) for _r, c in conn_pts})
    rows = sorted(int(r) for r, _c in conn_pts)
    assert len(cols) == 1
    assert rows == list(range(rows[0], rows[0] + len(rows)))

    body = sprite - conn_pts
    valid_sides = []
    col = int(cols[0])
    for side in (-1, 1):
        if all((int(r), int(col + side)) in body for r in rows) and all(
            (int(r), int(col - side)) not in body for r in rows
        ):
            valid_sides.append(int(side))
    assert len(valid_sides) == 1

    return {
        "pts": set(conn_pts),
        "color": int(color),
        "r0": int(rows[0]),
        "col": int(col),
        "len": int(len(rows)),
        "side": int(valid_sides[0]),
    }


def _sprite_desc(
    g: np.ndarray,
    sprite: set[tuple[int, int]],
    *,
    obj: int,
    connector_colors: set[int],
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    connectors = [
        _classify_connector(g, sprite, comp, obj)
        for comp in _connector_components(g, sprite, connector_colors)
    ]
    return {
        "pts": set(sprite),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "top_left_sum": int(r0 + c0),
        "connectors": connectors,
    }


def _describe_patch(
    patch: np.ndarray,
    *,
    bg: int,
    obj: int,
    connector_colors: set[int],
) -> dict[str, object]:
    pts = {(int(r), int(c)) for r, c in np.argwhere(patch != int(bg))}
    return _sprite_desc(patch, pts, obj=obj, connector_colors=connector_colors)


def _stamp_shifted_no_overlap(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    *,
    bg: int,
) -> None:
    h, w = patch.shape
    for r in range(int(h)):
        for c in range(int(w)):
            val = int(patch[int(r), int(c)])
            if val == int(bg):
                continue
            rr = int(top + r)
            cc = int(left + c)
            assert int(out[int(rr), int(cc)]) == int(bg)
            out[int(rr), int(cc)] = int(val)


def _assemble_ordered_chain(
    A_patch: np.ndarray,
    B_patch: np.ndarray,
    C_patch: np.ndarray,
    *,
    bg: int,
    obj: int,
) -> np.ndarray:
    connector_colors = {int(v) for v in np.unique(np.concatenate([A_patch.flatten(), B_patch.flatten(), C_patch.flatten()]))}
    connector_colors -= {int(bg), int(obj)}
    assert len(connector_colors) == 2

    A_desc = _describe_patch(A_patch, bg=bg, obj=obj, connector_colors=connector_colors)
    B_desc = _describe_patch(B_patch, bg=bg, obj=obj, connector_colors=connector_colors)
    C_desc = _describe_patch(C_patch, bg=bg, obj=obj, connector_colors=connector_colors)
    assert len(A_desc["connectors"]) == 1
    assert len(B_desc["connectors"]) == 2
    assert len(C_desc["connectors"]) == 1

    A_conn = A_desc["connectors"][0]
    C_conn = C_desc["connectors"][0]
    B_by_color = {int(conn["color"]): conn for conn in B_desc["connectors"]}
    assert len(B_by_color) == 2
    assert int(A_conn["color"]) in B_by_color
    assert int(C_conn["color"]) in B_by_color
    assert int(A_conn["color"]) != int(C_conn["color"])

    B_left = B_by_color[int(A_conn["color"])]
    B_right = B_by_color[int(C_conn["color"])]
    assert int(A_conn["side"]) == -1
    assert int(B_left["side"]) == 1
    assert int(B_right["side"]) == -1
    assert int(C_conn["side"]) == 1

    tr_B = int(A_conn["r0"] - B_left["r0"])
    tc_B = int((A_conn["col"] + 1) - B_left["col"])
    tr_C = int((B_right["r0"] + tr_B) - C_conn["r0"])
    tc_C = int((B_right["col"] + tc_B + 1) - C_conn["col"])
    assert tr_B >= 0 and tc_B >= 0 and tr_C >= 0 and tc_C >= 0

    H = max(int(A_patch.shape[0]), int(tr_B + B_patch.shape[0]), int(tr_C + C_patch.shape[0]))
    W = max(int(A_patch.shape[1]), int(tc_B + B_patch.shape[1]), int(tc_C + C_patch.shape[1]))
    out = np.full((int(H), int(W)), int(bg), dtype=int)
    _stamp_shifted_no_overlap(out, A_patch, 0, 0, bg=bg)
    _stamp_shifted_no_overlap(out, B_patch, tr_B, tc_B, bg=bg)
    _stamp_shifted_no_overlap(out, C_patch, tr_C, tc_C, bg=bg)
    out[(out != int(bg)) & (out != int(obj))] = int(obj)
    return _crop_non_bg(out, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    connector_colors = sorted({int(v) for v in g.flatten()} - {int(bg), int(obj)})
    assert len(connector_colors) == 2
    connector_color_set = {int(v) for v in connector_colors}

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 3

    sprites = [
        _sprite_desc(g, pts, obj=obj, connector_colors=connector_color_set)
        for pts in comps
    ]

    origins = [spr for spr in sprites if int(spr["top_left_sum"]) == 0]
    assert len(origins) == 1
    A = origins[0]

    doubles = [spr for spr in sprites if len(spr["connectors"]) == 2]
    singles = [spr for spr in sprites if len(spr["connectors"]) == 1]
    assert len(doubles) == 1 and len(singles) == 2
    B = doubles[0]
    others = [spr for spr in singles if spr is not A]
    assert len(others) == 1
    C = others[0]

    A_conn = A["connectors"][0]
    C_conn = C["connectors"][0]
    B_by_color = {int(conn["color"]): conn for conn in B["connectors"]}
    assert len(B_by_color) == 2

    A_shared = {int(A_conn["color"])} & set(B_by_color)
    C_shared = {int(C_conn["color"])} & set(B_by_color)
    assert len(A_shared) == 1 and len(C_shared) == 1
    ab_color = int(next(iter(A_shared)))
    bc_color = int(next(iter(C_shared)))
    assert ab_color != bc_color
    assert int(A_conn["color"]) == ab_color
    assert int(C_conn["color"]) == bc_color

    B_left = B_by_color[int(ab_color)]
    B_right = B_by_color[int(bc_color)]
    assert int(A_conn["side"]) == -1
    assert int(B_left["side"]) == 1
    assert int(B_right["side"]) == -1
    assert int(C_conn["side"]) == 1

    tr_B = int(A_conn["r0"] - B_left["r0"])
    tc_B = int((A_conn["col"] + 1) - B_left["col"])
    tr_C = int((B_right["r0"] + tr_B) - C_conn["r0"])
    tc_C = int((B_right["col"] + tc_B + 1) - C_conn["col"])

    shifted_B = {(int(r + tr_B), int(c + tc_B)) for r, c in B["pts"]}
    shifted_C = {(int(r + tr_C), int(c + tc_C)) for r, c in C["pts"]}
    pts_all = set(A["pts"]) | shifted_B | shifted_C
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 + 1), int(cc1 + 1)), int(bg), dtype=int)

    Ar0, Ac0, Ar1, Ac1 = map(int, A["bbox"])
    Br0, Bc0, Br1, Bc1 = map(int, B["bbox"])
    Cr0, Cc0, Cr1, Cc1 = map(int, C["bbox"])
    A_patch = g[Ar0 : Ar1 + 1, Ac0 : Ac1 + 1]
    B_patch = g[Br0 : Br1 + 1, Bc0 : Bc1 + 1]
    C_patch = g[Cr0 : Cr1 + 1, Cc0 : Cc1 + 1]

    _stamp_shifted_no_overlap(out, A_patch, Ar0, Ac0, bg=bg)
    _stamp_shifted_no_overlap(out, B_patch, Br0 + tr_B, Bc0 + tc_B, bg=bg)
    _stamp_shifted_no_overlap(out, C_patch, Cr0 + tr_C, Cc0 + tc_C, bg=bg)
    out[(out != int(bg)) & (out != int(obj))] = int(obj)
    return _crop_non_bg(out, int(bg))


def _build_terminal_sprite(
    *,
    conn_len: int,
    side: int,
    color: int,
    bg: int,
    obj: int,
) -> np.ndarray | None:
    h = _rng.randint(int(conn_len), 7)
    w = _rng.randint(3, 7)
    target_low = max(5, int(2 * conn_len + 1))
    target_high = min(24, int(h * w))
    if target_low > target_high:
        return None

    mask = np.zeros((int(h), int(w)), dtype=bool)
    keep: set[tuple[int, int]] = set()
    if int(side) == -1:
        conn_col = int(w - 1)
        body_col = int(w - 2)
    else:
        conn_col = 0
        body_col = 1
    for r in range(int(conn_len)):
        mask[int(r), int(conn_col)] = True
        mask[int(r), int(body_col)] = True
        keep.add((int(r), int(conn_col)))
        keep.add((int(r), int(body_col)))

    frontier = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    target = _rng.randint(target_low, target_high)
    _grow_shape(mask, int(target), frontier)
    if int(mask.sum()) < int(target):
        return None
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None

    grid = np.full(mask.shape, int(bg), dtype=int)
    grid[mask] = int(obj)
    for r in range(int(conn_len)):
        grid[int(r), int(conn_col)] = int(color)

    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(grid != int(bg))) < 5:
        return None
    return _crop_non_bg(grid, int(bg))


def _build_middle_sprite(
    *,
    left_len: int,
    right_len: int,
    left_color: int,
    right_color: int,
    bg: int,
    obj: int,
) -> np.ndarray | None:
    h = _rng.randint(max(int(left_len), int(right_len)), 8)
    w = _rng.randint(5, 8)
    target_low = max(8, int(left_len + right_len + 2))
    target_high = min(29, int(h * w))
    if target_low > target_high:
        return None

    mask = np.zeros((int(h), int(w)), dtype=bool)
    keep: set[tuple[int, int]] = set()
    bridge_r = int(max(left_len, right_len) - 1)

    for r in range(int(left_len)):
        mask[int(r), 0] = True
        mask[int(r), 1] = True
        keep.add((int(r), 0))
        keep.add((int(r), 1))
    for r in range(int(right_len)):
        mask[int(r), int(w - 1)] = True
        mask[int(r), int(w - 2)] = True
        keep.add((int(r), int(w - 1)))
        keep.add((int(r), int(w - 2)))
    for r in range(0, int(bridge_r) + 1):
        mask[int(r), 1] = True
        mask[int(r), int(w - 2)] = True
        keep.add((int(r), 1))
        keep.add((int(r), int(w - 2)))
    for c in range(1, int(w - 1)):
        mask[int(bridge_r), int(c)] = True
        keep.add((int(bridge_r), int(c)))

    frontier = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    target = _rng.randint(target_low, target_high)
    _grow_shape(mask, int(target), frontier)
    if int(mask.sum()) < int(target):
        return None
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None

    grid = np.full(mask.shape, int(bg), dtype=int)
    grid[mask] = int(obj)
    for r in range(int(left_len)):
        grid[int(r), 0] = int(left_color)
    for r in range(int(right_len)):
        grid[int(r), int(w - 1)] = int(right_color)

    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        return None
    if int(np.sum(grid != int(bg))) < 8:
        return None
    return _crop_non_bg(grid, int(bg))


def _gen_attempt() -> np.ndarray | None:
    bg, obj = 1, 4
    ab_color, bc_color = _rng.sample([2, 3, 5, 6, 7, 8, 9], 2)

    A_len = _rng.choice([2, 3])
    B_left_len = 5 - int(A_len)
    B_right_len = _rng.choice([2, 3])
    C_len = 5 - int(B_right_len)

    A = _build_terminal_sprite(conn_len=int(A_len), side=-1, color=int(ab_color), bg=bg, obj=obj)
    if A is None:
        return None
    B = _build_middle_sprite(
        left_len=int(B_left_len),
        right_len=int(B_right_len),
        left_color=int(ab_color),
        right_color=int(bc_color),
        bg=bg,
        obj=obj,
    )
    if B is None:
        return None
    C = _build_terminal_sprite(conn_len=int(C_len), side=1, color=int(bc_color), bg=bg, obj=obj)
    if C is None:
        return None

    try:
        expected = _assemble_ordered_chain(A, B, C, bg=bg, obj=obj)
    except Exception:
        return None

    if expected.size == 0 or not np.any(expected != int(bg)):
        return None
    if int(expected.shape[0]) > 10 or int(expected.shape[1]) > 18:
        return None

    ah, aw = map(int, A.shape)
    bh, bw = map(int, B.shape)
    ch, cw = map(int, C.shape)
    min_h = max(8, max(ah, bh, ch) + 2)
    min_w = max(12, max(aw, bw, cw) + 5)
    max_h = max(min_h, 13)
    max_w = max(min_w, 20)
    canvas_h = _rng.randint(min_h, max_h)
    canvas_w = _rng.randint(min_w, max_w)

    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, A, 0, 0, transparent=bg)
    try:
        br, bc = random_free_location(canvas, B, transparent=bg, padding=1)
        paste(canvas, B, int(br), int(bc), transparent=bg)
        cr, cc = random_free_location(canvas, C, transparent=bg, padding=1)
        paste(canvas, C, int(cr), int(cc), transparent=bg)
    except ValueError:
        return None

    if int(br + bc) == 0 or int(cr + cc) == 0:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 600 attempts")

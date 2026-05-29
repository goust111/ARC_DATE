# Combo variant of 20270e3b (index 14)
# combo_spec: D5/2+D11/1
# Name: 翻转对接颜色交换

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# design_analysis:
#   variant_summary: |
#     这是把两个单变体真正融合在同一条推理链里的版本：
#     1) plug 不能直接平移，必须先做水平翻转；
#     2) 对接完成后，两个精灵的 body 颜色互换，ctrl 也按所属精灵并入互换后的 body 色。
#     因此输出同时依赖“几何镜像后的对象身份”和“对象身份驱动的颜色交换”。
#   a_concretization_and_trace:
#     concrete_example: |
#       设 base 是红色(2)的不对称形状，连接器在其 bbox 右边缘；plug 输入是绿色(3)的不对称形状，
#       但其连接器在 bbox 左边缘。原题会直接把 plug 平移过去，并把所有 7 统一变成单一 obj 色。
#       本融合版不会这么做：
#       - 第一步先把绿色 plug 左右翻转，使其连接器转到右边缘；
#       - 第二步把翻转后的 plug 平移到 base 连接器正上方一行；
#       - 第三步渲染时按对象身份换色：base 全部改成 3，plug 全部改成 2。
#       差异不只是“多了一个翻转”或“多了一个换色”，而是颜色交换发生在翻转后的对接结果上。
#       例如接缝行原本属于 base 的 ctrl 区，在输出里会被上方 plug 的 body 覆盖，且颜色变成 2，
#       解题者必须同时追踪几何来源和颜色来源。
#   b_complexity_calibration:
#     controls:
#       - 颜色只用四种：bg=1、ctrl=7、base_col、plug_col，且强制 base_col != plug_col。
#       - 翻转类型只有一种：水平翻转，不引入旋转或多档位镜像。
#       - 连接器仍限制为长度 1-3 的水平条，并强制位于 bbox 左边缘或右边缘，增强可见性。
#     anti_degeneracy:
#       - 拒绝水平对称 plug，避免“翻了和没翻一样”。
#       - 拒绝 `flip=True` 与 `flip=False` 产生同一输出的样本，确保镜像推理必要。
#       - 拒绝颜色交换前后结果相同的样本；通过强制 body 颜色不同并检查输出中两色都出现实现。
#   c_implicit_assumption_audit:
#     original_assumptions_broken:
#       - 原 solver 写死 body 色是单一 obj=4；融合版必须分别读出 base/plug 的 body 色。
#       - 原 solver 直接用输入 plug 的连接器锚点做平移；融合版必须先对 plug patch `fliplr`，
#         再重新计算翻转后的连接器锚点。
#       - 原 solver `out[out == ctrl] = obj` 假设所有 ctrl 最终并入同一个颜色；融合版不成立，
#         因为 base 侧和 plug 侧应并入不同的交换后 body 色。
#       - 原 solver 用画布当前颜色判断重叠是否合法；融合版下颜色已经交换，合法性必须回到
#         “这个重叠点是否来自 connector 参与的接缝”这一结构条件。
#     combined_policy: |
#       统一做法是：先提取两个组件的 tight patch，再在 patch 层面执行“可选翻转 + 对接 +
#       可选颜色交换”的公共函数。该函数只根据 connector 集合和对象身份决定位置与颜色，
#       不依赖单一 obj 常量。
#   d_placement_legality:
#     generator_strategy: |
#       生成器沿用 output-first 思路，但采用 D11 的更稳健实现：
#       先构造 base，再构造“已经处于 docked 朝向”的 plug；随后把该 plug 水平镜像，
#       得到输入中的 separated plug。这样可直接控制 docked 态的合法性，再把输入态作为其镜像。
#     legality_guards:
#       - base 固定放在输入画布原点。
#       - separated plug 通过候选位置搜索放置，保证不重叠、不越界、无 8 邻接接触，bbox gap 在 1-5。
#       - docked 态只允许接缝处发生覆盖，且覆盖必须是“base connector vs plug body”。
#       - 翻转后的 plug 不得伸到 base 的上方或左方，保证 tight crop 后 base 仍在输出左上角。
#   e_visual_observability:
#     rationale: |
#       人类从 2-4 个例子里应能同时看到两件稳定事情：
#       1) 输入 plug 的连接器总在与输出对接侧相反的一边，说明必须镜像；
#       2) 输出中上下两部分完整互换了输入里的两种 body 色，说明颜色交换跟对象身份绑定。
#       连接器被强制放在边缘、plug 被强制非对称，这让“要翻转”比内嵌细节更容易观察。
#   f_visual_cleanliness:
#     rationale: |
#       画面仍只有两个对象和一种标记色 ctrl，没有额外纹理。叠加层次只有一层：
#       dock 时 plug body 覆盖 base connector。颜色交换发生在最终上色，不会在输入里制造
#       杂乱的多色图案，因此视觉上仍然干净。
#   g_color_change_inferability:
#     rationale: |
#       颜色变化不是任意映射，而是“哪个对象在输入里是 base / plug，输出里就拿对方的 body 色”。
#       因为每个对象保持单色 body，且几何形状在输出里仍可区分，少量样例就足以归纳这个规则。
#   h_main_new_assumptions_to_check:
#     solver_assumptions:
#       - 恰有两个前景 8 连通组件。
#       - 每个组件恰有一条水平 ctrl 条，且两条 connector 规范化形状相同。
#       - 每个组件的非 bg/ctrl 像素只有一种 body 色。
#       - base 组件的 bbox 左上角在输入原点，plug 组件不是。
#       - base connector 的 body 在其上方，plug connector 的 body 在其下方。
#       - 经过水平翻转后再对接，plug 不会伸到 base 的上方或左方。
#     generator_support:
#       - `_build_base` 保证 base 的边缘连接器与朝向约束。
#       - `_build_docked_plug` 先在 docked 朝向构造 plug，再镜像成输入，并排除水平对称。
#       - `_gen_attempt` 用 oracle 检查 `main(input)`，并额外比较 no-flip / no-swap 的反事实输出，
#         防止样本退化回原题或单变体。


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_patch(grid: np.ndarray, pts: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _conn_points(patch: np.ndarray, ctrl: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(patch == int(ctrl))}


def _body_points(patch: np.ndarray, bg: int, ctrl: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere((patch != int(bg)) & (patch != int(ctrl)))
    }


def _single_body_color(patch: np.ndarray, bg: int, ctrl: int) -> int:
    cols = {
        int(patch[int(r), int(c)])
        for r, c in np.argwhere((patch != int(bg)) & (patch != int(ctrl)))
    }
    assert len(cols) == 1
    return int(next(iter(cols)))


def _norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _connector_side(patch: np.ndarray, ctrl: int) -> str:
    conn = _conn_points(patch, ctrl)
    assert conn
    cols = [c for _r, c in conn]
    if int(min(cols)) == 0:
        return "left"
    if int(max(cols)) == int(patch.shape[1] - 1):
        return "right"
    return "interior"


def _dock_from_patches(
    base_patch: np.ndarray,
    plug_patch: np.ndarray,
    *,
    bg: int,
    ctrl: int,
    flip_plug: bool,
    swap_colors: bool,
) -> np.ndarray:
    base = np.array(base_patch, dtype=int)
    plug_in = np.array(plug_patch, dtype=int)
    assert base.size > 0 and plug_in.size > 0

    conn_base = _conn_points(base, ctrl)
    conn_plug_in = _conn_points(plug_in, ctrl)
    assert conn_base and conn_plug_in
    shp_base, (dr0, dc0) = _norm(conn_base)
    shp_plug_in, _ = _norm(conn_plug_in)
    assert shp_base == shp_plug_in

    base_body = _body_points(base, bg, ctrl)
    base_col = _single_body_color(base, bg, ctrl)
    plug_col = _single_body_color(plug_in, bg, ctrl)
    for r, c in conn_base:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body

    work_plug = np.fliplr(plug_in) if flip_plug else plug_in.copy()
    conn_work = _conn_points(work_plug, ctrl)
    shp_work, (pr0, pc0) = _norm(conn_work)
    assert shp_base == shp_work

    plug_body = _body_points(work_plug, bg, ctrl)
    for r, c in conn_work:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)

    plug_pixels: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    for r, c in np.argwhere(work_plug != int(bg)):
        rr, cc = int(r), int(c)
        plug_pixels[(int(rr + tr), int(cc + tc))] = ((rr, cc), int(work_plug[rr, cc]))

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base != int(bg))}
    pts_all = base_pts | set(plug_pixels)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    assert rr0 == 0 and cc0 == 0

    base_out_col = int(plug_col if swap_colors else base_col)
    plug_out_col = int(base_col if swap_colors else plug_col)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(base_out_col)

    for (r, c), (src, src_val) in plug_pixels.items():
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and src_val != int(bg):
            assert (int(r), int(c)) in conn_base or src in conn_work
        out[int(r - rr0), int(c - cc0)] = int(plug_out_col)

    return _crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, ctrl = 1, 7

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a_pts, b_pts = comps

    conn_a = {(r, c) for (r, c) in a_pts if int(g[int(r), int(c)]) == int(ctrl)}
    conn_b = {(r, c) for (r, c) in b_pts if int(g[int(r), int(c)]) == int(ctrl)}
    assert conn_a and conn_b
    shp_a, _ = _norm(conn_a)
    shp_b, _ = _norm(conn_b)
    assert shp_a == shp_b

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(a_pts)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(b_pts)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = a_pts, b_pts
    else:
        base_pts, plug_pts = b_pts, a_pts

    base_patch = _extract_patch(g, base_pts, bg)
    plug_patch = _extract_patch(g, plug_pts, bg)
    return _dock_from_patches(
        base_patch,
        plug_patch,
        bg=bg,
        ctrl=ctrl,
        flip_plug=True,
        swap_colors=True,
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
_rng = random


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w and not mask[int(r + dr), int(c + dc)]
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
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _build_base(base_col: int) -> tuple[np.ndarray, str] | None:
    bg, ctrl = 1, 7
    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, int(base_h * base_w)))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    max_conn_len = min(3, int(base_w - 1))
    if max_conn_len < 1:
        return None
    conn_len = _rng.randint(1, max_conn_len)
    side = _rng.choice(["left", "right"])
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        c0 = 0 if side == "left" else int(base_w - conn_len)
        cols = list(range(int(c0), int(c0 + conn_len)))
        if not all(base[int(r), int(c)] for c in cols):
            continue
        if not all(base[int(r - 1), int(c)] for c in cols):
            continue
        if int(r + 1) < base_h and not all(not base[int(r + 1), int(c)] for c in cols):
            continue
        candidates.append((int(r), cols))
    if not candidates:
        return None

    conn_row, conn_cols = _rng.choice(candidates)
    base_grid = np.full((base_h, base_w), int(bg), dtype=int)
    base_grid[base] = int(base_col)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    base_sprite = _crop_non_bg(base_grid, bg)
    if base_sprite.size == 0:
        return None
    if int(base_sprite.shape[1]) <= int(len(_conn_points(base_sprite, ctrl))):
        return None
    if _connector_side(base_sprite, ctrl) != side:
        return None
    if _rng.random() < 0.5:
        base_sprite = np.fliplr(base_sprite)
        side = "right" if side == "left" else "left"
    if _connector_side(base_sprite, ctrl) != side:
        return None
    return base_sprite, side


def _build_docked_plug(base_sprite: np.ndarray, side: str, plug_col: int) -> np.ndarray | None:
    bg, ctrl = 1, 7
    base_h, base_w = map(int, base_sprite.shape)
    conn_base = _conn_points(base_sprite, ctrl)
    _shape, (base_conn_r0, base_conn_c0) = _norm(conn_base)
    conn_len = int(len(conn_base))

    plug_h = _rng.randint(2, 7)
    if side == "right":
        max_w = base_w
        min_w = max(int(conn_len + 1), 2)
        if max_w < min_w:
            return None
        plug_w = _rng.randint(min_w, max_w)
        dock_conn_c0 = int(plug_w - conn_len)
    else:
        min_w = max(int(conn_len + 1), 2)
        max_w = min(10, int(base_w + 4))
        if max_w < min_w:
            return None
        plug_w = _rng.randint(min_w, max_w)
        dock_conn_c0 = 0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug[0, int(c)] = True
        if plug_h > 1:
            plug[1, int(c)] = True

    plug_max = min(29, int(plug_h * plug_w))
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, [(int(r), int(c)) for r, c in zip(*np.where(plug))])
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    keep = {(0, int(c)) for c in range(dock_conn_c0, dock_conn_c0 + conn_len)}
    _thin(plug, keep=keep)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    if np.array_equal(plug, np.fliplr(plug)):
        return None

    plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
    plug_grid[plug] = int(plug_col)
    for c in range(dock_conn_c0, dock_conn_c0 + conn_len):
        plug_grid[0, int(c)] = int(ctrl)
        if plug_h < 2 or plug_grid[1, int(c)] != int(plug_col):
            return None

    plug_grid = _crop_non_bg(plug_grid, bg)
    if plug_grid.size == 0:
        return None
    if int(plug_grid.shape[1]) <= int(conn_len):
        return None

    if _connector_side(plug_grid, ctrl) != side:
        return None
    if _connector_side(np.fliplr(plug_grid), ctrl) == side:
        return None

    tc = int(base_conn_c0 - dock_conn_c0)
    tr = int(base_conn_r0 - 1)
    if tc < 0 or tr < 0:
        return None

    for r, c in np.argwhere(plug_grid != int(bg)):
        rr, cc = int(r), int(c)
        dr = int(rr + tr)
        dc = int(cc + tc)
        if 0 <= dr < base_h and 0 <= dc < base_w:
            cur = int(base_sprite[dr, dc])
            val = int(plug_grid[rr, cc])
            if cur != int(bg) and val != int(bg) and cur != int(ctrl) and val != int(ctrl):
                return None

    plug_input = np.fliplr(plug_grid)
    try:
        expected = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=True,
            swap_colors=True,
        )
    except AssertionError:
        return None
    try:
        direct = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=False,
            swap_colors=True,
        )
    except AssertionError:
        direct = None
    if direct is not None and np.array_equal(expected, direct):
        return None
    return plug_grid


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


def _placement_candidates(
    canvas_h: int,
    canvas_w: int,
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return []

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    base_dilated = set(base_pts)
    for r, c in list(base_pts):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                base_dilated.add((int(r + dr), int(c + dc)))
    plug_local = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}

    out: list[tuple[int, int]] = []
    base_bbox = (0, 0, int(bh - 1), int(bw - 1))
    for pr in range(0, int(canvas_h - ph + 1)):
        for pc in range(0, int(canvas_w - pw + 1)):
            if pr == 0 and pc == 0:
                continue
            shifted = {(int(r + pr), int(c + pc)) for r, c in plug_local}
            if any(pt in base_dilated for pt in shifted):
                continue
            gap = _bbox_gap(base_bbox, (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1)))
            if 1 <= int(gap) <= 5:
                out.append((int(pr), int(pc)))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = 1, 7
    palette = [2, 3, 4, 5, 6, 8, 9]
    base_col, plug_col = random.sample(palette, 2)

    built = _build_base(base_col)
    if built is None:
        return None
    base_sprite, side = built

    docked_plug = _build_docked_plug(base_sprite, side, plug_col)
    if docked_plug is None:
        return None

    plug_input = np.fliplr(docked_plug)
    if _connector_side(plug_input, ctrl) == side:
        return None

    try:
        expected = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=True,
            swap_colors=True,
        )
    except AssertionError:
        return None
    try:
        direct = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=False,
            swap_colors=True,
        )
    except AssertionError:
        direct = None
    if direct is not None and np.array_equal(expected, direct):
        return None
    try:
        no_swap = _dock_from_patches(
            base_sprite,
            plug_input,
            bg=bg,
            ctrl=ctrl,
            flip_plug=True,
            swap_colors=False,
        )
    except AssertionError:
        return None
    if np.array_equal(expected, no_swap):
        return None
    out_colors = set(map(int, np.unique(expected))) - {int(bg)}
    if out_colors != {int(base_col), int(plug_col)}:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_input.shape)
    min_h = max(7, bh, ph)
    min_w = max(3, bw, pw)
    if min_h > 11 or min_w > 13:
        return None

    for _ in range(30):
        canvas_h = _rng.randint(min_h, 11)
        canvas_w = _rng.randint(min_w, 13)
        placements = _placement_candidates(canvas_h, canvas_w, base_sprite, plug_input, bg)
        if not placements:
            continue
        pr, pc = _rng.choice(placements)
        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        paste(canvas, base_sprite, 0, 0, transparent=bg)
        paste(canvas, plug_input, pr, pc, transparent=bg)
        if np.sum(canvas == int(bg)) <= canvas.size // 2:
            continue
        try:
            result = main(canvas)
        except Exception:
            continue
        if not np.array_equal(result, expected):
            continue
        return canvas
    return None


def generate_input() -> np.ndarray:
    for _ in range(1000):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 1000 attempts")

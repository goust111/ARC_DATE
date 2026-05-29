# Variant of 20270e3b
# Dimension: D5 — 对接后变换 (transform)
# Change: 连接器像素（ctrl色）统一重新着色为body色（obj），连接器在输出中消失... → 交换两个精灵的body颜色

from __future__ import annotations

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
# ---------------------------------------------------------------------------
#
# design_analysis:
#   reference_task:
#     id: a25697e4
#     relevant_mechanism: |
#       参考 solver 的关键不是几何，而是“一个已经选定的空间变换会额外触发颜色交换”。
#       该任务在最终落子阶段，根据对象身份重新给像素上色；因此颜色交换不是靠后处理
#       一个全局常量，而是依赖“这个像素属于哪个对象”。本变体沿用这一思想：
#       对接几何保持 20270e3b 原样，颜色变化在 stamp 阶段按 base / plug 身份执行。
#   a_concretization_and_trace:
#     input_scene: |
#       输入仍然是两个 8 连通精灵，每个精灵各有一条水平 ctrl 连接器。不同点在于：
#       base 精灵 body 全部是 base_col，plug 精灵 body 全部是 plug_col，且通常不同；
#       ctrl 仍固定为 7，背景仍固定为 1。base 仍锚定在输入左上角。
#     output_scene: |
#       先按原题方式把 plug 平移到 base 上方一行完成对接，再交换两精灵的 body 颜色：
#       base 全部变成 plug_col，plug 全部变成 base_col；两条 ctrl 连接器也按所属精灵
#       一并重着色，所以输出只剩 bg + 两个 body 色。
#     trace_example: |
#       例：base 是红色(2)，plug 是绿色(3)，两者 connector 都是橙色(7)。
#       原题：dock 后所有 7 都变成单一 obj=4，因此颜色信息全部消失。
#       本变体：dock 后，base 的原红色区域全部改成 3；plug 的原绿色区域和它的 connector
#       全部改成 2；base connector 被 plug 的第一行 body 覆盖，因此接缝行属于 plug，颜色为 2。
#       差异发生在 render / recolor 环节，而不是匹配或平移环节。
#   b_complexity_calibration:
#     decisions:
#       - 保留“同长度水平连接器 + base 在左上角 + plug 不得延伸到 base 左/上方”这些原题约束，
#         避免把题目改成同时考几何和颜色。
#       - 每个精灵只使用一个 body 色，整题总共四种颜色：bg、ctrl、base_col、plug_col。
#         这样颜色交换足够明显，但不会引入大量无结构颜色。
#       - 强制 base_col != plug_col，使变体在绝大多数样本中必然可见，不会退化成原题的“换色无效”。
#   c_implicit_assumption_audit:
#     original_lines_affected:
#       - 原 solver 写死了 bg=1,obj=4,ctrl=7，其中 obj=4 隐含“两个精灵 body 色完全相同”。
#         变体下这个前提失效，必须从每个组件中动态读取 body 色。
#       - 原 solver `out[out == ctrl] = obj` 依赖“所有 connector 都应变成同一个颜色”。
#         变体下这是错误的：上方 plug connector 应变成 base_col，下方 base body 应变成 plug_col。
#       - 原 solver 用颜色判断 overlap 的合法性：`cur == ctrl or val == ctrl`。
#         这在单一 obj 下无害；在双 body 色下，仍应依据“原像素是否是 connector”判断，
#         不能依据当前画布颜色判断，否则会把合法的颜色覆盖误判为冲突。
#     new_solver_policy: |
#       因此新 solver 先分出 base/plug，再分别提取两个组件的唯一 body 色；render 时直接把
#       base 全部按 plug_col 上色、plug 全部按 base_col 上色。重叠合法性改成按点是否属于
#       base connector 或 plug connector 判定。
#   d_placement_legality:
#     generator_guarantees:
#       - 仍先构造 docked 结构，再把两精灵裁开并分开放置。
#       - base 放在 (0,0)，plug 随机放在其余空位，保证输入正好有两个 8 连通前景组件。
#       - 连接器长度保持一致，base connector 下方无 body，plug connector 上方无 body。
#       - docked 状态下除了“base connector 行被 plug body 覆盖”外，不允许其他重叠。
#       - plug 在 docked 后不能伸到 base 的左边或上边，保证 tight crop 后 base 仍在输出左上角。
#   e_visual_observability:
#     rationale: |
#       人类从少量样例里首先会看到：几何拼接方式和原题一样，但输出中上下两部分交换了输入中的
#       两种 body 颜色。因为两个颜色在输入中分别绑定到两个形状，输出中又完整互换到对方身上，
#       该规律在 2-3 组样例中是可观察和可归纳的。
#   f_visual_cleanliness:
#     rationale: |
#       颜色层次只有四种，叠加层次仍只有“上方 plug body 覆盖下方 base connector”这一层，
#       不引入额外标记或内部纹理，因此图面仍然清晰。连接器统一保持水平短条，视觉角色稳定。
#   g_color_change_inferability:
#     rationale: |
#       颜色变化不是任意映射，而是严格依赖对象身份：base 变成 plug 原色，plug 变成 base 原色。
#       颜色种类固定为两个 body 色 + ctrl，不会出现多对多无提示映射。
#   main_new_assumptions_to_check:
#     - 每个精灵的非-bg、非-ctrl 像素必须只有 1 种颜色。
#     - 两个精灵都必须包含至少 1 个 ctrl 像素，且两个 connector 规范化形状一致。
#     - base 仍由 bbox 左上角最靠近原点的组件决定，并且至少一个组件满足 r0+c0==0。
#     - generator 必须保证 docked 后唯一重叠区满足“base connector vs plug body”这一原题假设。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching horizontal connectors, then swap the two
    sprites' body colors in the docked result and tight-crop.
    """
    g = np.array(grid, dtype=int)
    bg, ctrl = 1, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[r, c]) == int(ctrl)}

    def body_color(sprite: set[tuple[int, int]]) -> int:
        cols = {int(g[r, c]) for (r, c) in sprite if int(g[r, c]) not in (int(bg), int(ctrl))}
        assert len(cols) == 1
        return int(next(iter(cols)))

    def norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    connA = conn(A)
    connB = conn(B)
    assert connA and connB
    shpA, _ = norm(connA)
    shpB, _ = norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_col = body_color(base)
    plug_col = body_color(plug)
    base_body = base - connD
    plug_body = plug - connP

    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = norm(connP)
    _sd, (dr0, dc0) = norm(connD)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    # Stamp the recolored base first: after swapping, base takes plug's original body color.
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(plug_col)

    # Stamp the recolored plug second: after swapping, plug takes base's original body color.
    for r, c in plug_shift:
        src = (int(r - tr), int(c - tc))
        if (r, c) in base:
            assert (r, c) in connD or src in connP
        out[int(r - rr0), int(c - cc0)] = int(base_col)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (anchor-first: build base, grow plug on it, then separate)
# ---------------------------------------------------------------------------


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected 4-neighbor shape on mask from frontier up to target pixels."""
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
    """Remove interior pixels while preserving 8-connectivity."""
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


def _crop(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    bg, ctrl = 1, 7
    palette = [2, 3, 4, 6, 8, 9]
    base_col, plug_col = random.sample(palette, 2)

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if r == 0:
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = int(base_col)
    for c in conn_cols:
        base_grid[conn_row, c] = int(ctrl)
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = int(conn_cols[0] - plug_conn_c0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(conn_row + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == int(base_col):
                if r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    if dc_offset < 0:
        return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = int(plug_col)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(ctrl)

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_h < 2 or plug_grid[1, c] != int(plug_col):
            return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop(base_grid, bg)
    plug_sprite = _crop(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=0)
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

    # Variant guard: the swap must be observable on the assembled output.
    if len({int(x) for x in np.unique(result)} - {int(bg)}) < 2:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")

# Variant of 20270e3b
# Dimension: D5 — 对接后变换 (transform)
# Change: 连接器像素（ctrl色）统一重新着色为body色（obj），连接器在输出中消失... → 连接器颜色扩散到相邻像素
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入结构保持原题：两个 8 连通精灵，各自带一条同长度的水平 ctrl 连接器。
#      base 的连接器下方为空，plug 的连接器上方为空；把 plug 平移到其连接器
#      位于 base 连接器正上方一行，再做紧裁剪。
#    - 原题在合并后执行 `ctrl -> obj`，所以输出只剩 body 色。
#    - 本变体改为：合并后保留仍然可见的 ctrl 条，再把“与这些 ctrl 像素 4 邻接的
#      obj 像素”一次性改成 ctrl。只扩散一层，不允许链式继续扩散。
#    - 小例子：若对接后中间两行为 `777` / `444`（下行是 plug body 覆盖了
#      base connector），原题输出变成 `444` / `444`；本变体输出变成 `777` /
#      `777`。如果连接器端点左右仍有 obj，也会一起被染成 ctrl，形成焊接痕迹。
# b) 复杂度校准
#    - 不引入新的对象类别、排序规则或多档位变体；仍只有 bg/obj/ctrl 三色。
#    - 变化集中在输出局部 seam 附近，但比“仅保留 connector”更明显，因为会额外
#      染到一圈 body。生成器继续允许不同 connector 长度和不同精灵形状，避免样例
#      过于单一。
# c) 隐式假设审查
#    - 组件识别、base/plug 判定、connector 形状匹配、对接位移、可重叠区域约束
#      都与“connector 最终是否消失”无关，因此可原样保留。
#    - 唯一失效的旧前提是 `out[out == ctrl] = obj`。变体中如果仍无条件重着色，
#      就会抹掉所有焊接痕迹，因此必须替换为“保留 ctrl + 一层扩散”。
#    - 扩散时必须基于扩散前的 ctrl 分布计算邻接，不能边染边扩，否则会意外扩成两层。
# d) 放置合法性
#    - 生成器仍沿用原题合法性：两个输入精灵互不重叠；base 固定在左上角；plug 单独
#      放置；对接时除 connector 区外不得压到 base body；plug 不得伸到 base 左侧。
#    - 这些条件保证 solver 的平移、覆盖和裁剪逻辑仍然成立。
# e) 视觉可观察性
#    - 人类能直接看到：输出不再把连接器抹成 body，而是在 seam 位置留下 ctrl 条，
#      且 ctrl 会向相邻 body 扩一圈。2-3 个样例足以归纳。
#    - 为避免“只是保留了一条 connector”的弱变化，生成器额外要求输出中 ctrl 像素
#      数严格大于原始 connector 长度，确保确实发生了扩散。
# f) 可视化整洁度
#    - 扩散只发生在 seam 邻域，颜色层次仍然简单，不增加额外标记，也不引入多对象叠压。
#    - 只用 4 邻域，视觉上是紧贴 connector 的焊接痕迹，不会蔓延成杂乱色块。
# g) 颜色变化可推断性
#    - 颜色变化只有一种：connector 色向相邻 body 色扩散一层。触发条件完全由位置和
#      邻接关系决定，不依赖不可见数值属性。
# h) main() 的新假设与生成器保证
#    - 新 main() 假设合并后的可见 ctrl 至少邻接一些 obj，扩散才有可见意义。
#      生成器通过 plug connector 下方必须紧贴 body 来保证这一点；并在 oracle 中
#      检查输出 ctrl 像素数大于 connector 长度，拒绝退化样本。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Variant rule:
    - Identify the base sprite (closest to grid top-left).
    - Translate the plug so its connector sits one cell above the base connector.
    - Stamp the plug onto the base.
    - Keep the visible connector color, then spread ctrl to 4-neighbor body pixels
      for exactly one layer (no chain reaction).
    - Output is the tight crop around all non-background pixels.
    """
    g = np.array(grid, dtype=int)

    # Fixed colors (consistent across all eval examples)
    bg, obj, ctrl = 1, 4, 7

    # Build foreground components (8-connected: sprites may touch diagonally)
    lab, n = label_mask(g != bg, connectivity=8)
    best_comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        best_comps.append(pts)
    best_comps.sort(key=len, reverse=True)
    assert len(best_comps) == 2  # exactly 2 sprites, no fragments
    A = set(best_comps[0])
    B = set(best_comps[1])

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert not (not connA or not connB)

    def norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    shpA, (_ar0, _ac0) = norm(connA)
    shpB, (_br0, _bc0) = norm(connB)
    assert shpA == shpB

    # base = sprite whose bbox top-left is closest to (0,0)
    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
    # Base must be anchored at grid top-left corner
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    # Connectors dock vertically: base connector's free side faces down,
    # plug connector's free side faces up.
    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body  # base: free side down
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body  # plug: free side up

    _sp, (pr0, pc0) = norm(connP)
    _sd, (dr0, dc0) = norm(connD)

    # Place plug connector directly above base connector.
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    # Render on minimal bbox.
    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    # Base must remain at output top-left after tight crop.
    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0  # plug must not extend above/left of base
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    # Stamp base first.
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    # Stamp plug on top; overlap only allowed at connector docking zone (ctrl meets obj).
    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)  # overlap only at connector
        out[int(r - rr0), int(c - cc0)] = int(val)

    # Variant D5: keep ctrl and spread it to 4-neighbor body pixels for one layer only.
    ctrl_mask = out == int(ctrl)
    weld_mask = np.zeros_like(ctrl_mask, dtype=bool)
    for r, c in np.argwhere(ctrl_mask):
        r = int(r)
        c = int(c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and int(out[nr, nc]) == int(obj):
                weld_mask[nr, nc] = True
    out[weld_mask] = int(ctrl)

    # Tight crop around non-bg.
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (anchor-first: build base, grow plug on it, then separate)
# ---------------------------------------------------------------------------


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < target and frontier:
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
    """Remove interior pixels to make a thin-walled shape."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (r, c) in keep:
            continue
        # Skip if already on boundary (adjacent to False).
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    # Step 1: build base sprite body
    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    # Step 2: place connector on base (can be edge or interior row)
    conn_len = _rng.randint(1, min(3, base_w))
    # Valid connector positions: row where base has obj, with obj above it
    # (connector free side faces down, body side faces up).
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

    # Build base sprite grid with connector.
    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl

    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    # Step 3: grow plug from connector downward.
    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))

    # Align plug connector with base connector in the docked state.
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True  # Guarantees at least one diffusable body layer.

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    # Verify plug doesn't overlap base body at non-connector positions in docked state.
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row + r
            dc = dc_offset + c
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == obj:
                if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                    continue
                return None

    # Plug must not extend left of base, or the output crop anchor would change.
    if dc_offset < 0:
        return None

    # Build plug sprite grid.
    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl

    # Verify plug connector adjacency (body below connector).
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_grid[1, c] != obj:
            return None

    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    def crop(local_grid: np.ndarray) -> np.ndarray | None:
        nz = np.argwhere(local_grid != bg)
        if nz.size == 0:
            return None
        r0, c0, r1, c1 = bbox_coords(nz)
        return local_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    base_sprite = crop(base_grid)
    plug_sprite = crop(plug_grid)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    # Step 4: place on canvas — base at (0,0), plug elsewhere.
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

    # Step 5: oracle verify.
    try:
        result = main(canvas)
    except Exception:
        return None
    if result.size == 0 or not np.any(result != bg):
        return None
    # Variant-specific guard: output must contain the connector plus at least one
    # extra ctrl layer, otherwise the sample visually collapses toward the original rule.
    ctrl_count = int(np.sum(result == ctrl))
    if ctrl_count <= conn_len:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")

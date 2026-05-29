# Variant of 20270e3b
# Dimension: D4 — 目标对接方式 (dst)
# Change: plug精灵平移到base精灵上方，连接器行相邻（plug连接器在base连接器正上方1行），直接拼接... → 嵌入式对接（插入洞中）
#
# Design analysis
# a) 具体化与行为追踪
#    - 仍然只有两个前景精灵，颜色固定为 bg=1, obj=4, ctrl=7。
#    - base 仍然锚定在输入左上角；plug 仍然带一条水平 ctrl 横条，且 body 在横条下方。
#    - 变体的关键改动是：base 不再等待 plug 在外边缘“贴上去”，而是自带一个内部 socket。
#      这个 socket 是一块完全由 bg 构成、形状与 plug 的 obj-body 精确相同的凹槽。
#      base 的 ctrl 横条位于 socket 的正上边界；plug 平移后，ctrl 横条与 base 的 ctrl 横条同一行完全重合，
#      而 plug 的 body 整体落入 socket 内部。输出中 ctrl 依旧消失并改成 obj。
#    - 小例子追踪：
#      原题：base connector 在第 4 行，plug connector 平移到第 3 行，二者上下相邻，输出 bbox 变高。
#      本题：base connector 仍在某一行，但 plug connector 平移到同一行；plug body 的每个像素都必须落到
#      base 内部的 hole 上，输出 bbox 与 base bbox 相同，只是 hole 被填满。
# b) 复杂度校准
#    - 若把 socket 做成简单矩形，规则容易退化为“按 connector 对齐就行”。为避免这一点，plug body 会生成
#      成不规则连通形，socket 也随之成为同形不规则洞；solver 会显式验证“洞形 == plug body 形”。
#    - connector 长度仍限制在 1-3，颜色仍固定三色，避免自由度过大。
# c) 隐式假设审查（相对原 main 的关键变化）
#    - 原代码 `tr = (dr0 - 1) - pr0` 依赖“plug connector 必须在 base connector 上方一行”；变体中该前提失效，
#      必须改成同一行重合。
#    - 原代码允许非 connector 位置几乎没有重叠，只在拼接缝处重叠；变体中 plug 的大部分像素要进入 base bbox
#      内部，因此必须改成“plug body 只能覆盖 socket 的 bg 像素，connector 处才允许 ctrl-overlap”。
#    - 原代码 tight-crop 后通常会比 base 更高；变体中若生成合法，union bbox 应与 base bbox 相同。
# d) 放置合法性
#    - generator 先构造 docked 状态，再把 socket 挖空形成 base、把挖出的 body 做成 plug。
#    - 输入画布上放置 plug 时要求与 base 至少有 1 格 8 邻域间隔，保证 solver 的 8 连通分量始终恰好为 2。
# e) 视觉可观察性
#    - 人类看到 2-4 个样例时，会稳定观察到：输入里 base 内部有一个与右侧/下方 plug 同形的洞，输出里洞被填满，
#      整体外轮廓并未像原题那样向外扩张。这一差异在少量样例中是可见且可归纳的。
# f) 可视化整洁度
#    - socket 只与 plug body 对应；ctrl 仍然只是一条短横线，不叠加多种标记。
#    - base 采用薄壳/边界式结构，避免大面积实心块把洞和 plug 都淹没。
# g) 颜色变化可推断性
#    - 颜色逻辑不变：ctrl 仅表示对接口，输出统一改回 obj。没有新增颜色映射。
# h) 新 main 的额外假设
#    - base socket 是 bbox 内一个 4 连通 bg 分量，且它在每个 connector 列的正下方都有洞口。
#    - plug body 的最上行恰好对应 connector 宽度，这样 base 的 ctrl 横条视觉上明确对应洞口宽度。
#    - generator 会保证这些条件成立。

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7


def _component_points(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(cid))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn(sprite: set[tuple[int, int]], grid: np.ndarray) -> set[tuple[int, int]]:
    return {(r, c) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(CTRL)}


def _norm(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    return {(int(r - r0), int(c - c0)) for r, c in pts}, (int(r0), int(c0))


def _render_sprite(grid: np.ndarray, pts: set[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    comps = _component_points(g != int(BG))
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    connA = _conn(A, g)
    connB = _conn(B, g)
    assert connA and connB

    shpA, (ar0, ac0) = _norm(connA)
    shpB, (br0, bc0) = _norm(connB)
    assert shpA == shpB
    assert len({r for r, _c in connA}) == 1
    assert len({r for r, _c in connB}) == 1

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm(connP)
    _sd, (dr0, dc0) = _norm(connD)
    tr = int(dr0 - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    plug_shift_body = plug_shift - connD

    base_sprite = _render_sprite(g, base, BG)
    br0, bc0, br1, bc1 = _bbox_of_pts(base)
    hole_mask = base_sprite == int(BG)
    hole_lab, hole_n = label_mask(hole_mask, connectivity=4)
    socket_ids = set()
    for r, c in connD:
        rr = int(r - br0 + 1)
        cc = int(c - bc0)
        assert 0 <= rr < hole_mask.shape[0]
        assert hole_mask[rr, cc]
        socket_ids.add(int(hole_lab[rr, cc]))
    assert len(socket_ids) == 1
    socket_id = socket_ids.pop()
    assert socket_id > 0
    socket = {
        (int(br0 + rr), int(bc0 + cc))
        for rr, cc in np.argwhere(hole_lab == int(socket_id))
    }
    assert socket == plug_shift_body
    for r, c in socket:
        assert br0 < r < br1
        assert bc0 <= c <= bc1

    return {
        "grid": g,
        "base": base,
        "plug": plug,
        "conn_base": connD,
        "conn_plug": connP,
        "plug_shift": plug_shift,
        "plug_shift_body": plug_shift_body,
        "socket": socket,
        "base_bbox": (br0, bc0, br1, bc1),
        "shift": (tr, tc),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the plug into the base's internal socket, then tight-crop.

    The connector bars still identify the docking mouth, but unlike the original task the plug is
    translated so its connector overlaps the base connector on the same row, while its body fills a
    matching cavity inside the base.
    """
    info = _analyze_input(grid)
    g = info["grid"]
    base = info["base"]
    plug_shift = info["plug_shift"]
    conn_base = info["conn_base"]
    br0, bc0, br1, bc1 = info["base_bbox"]

    pts_all = set(base) | set(plug_shift)
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    assert (rr0, cc0, rr1, cc1) == (br0, bc0, br1, bc1)
    out = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(BG), dtype=int)

    for r, c in base:
        out[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    tr, tc = info["shift"]
    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - br0), int(c - bc0)])
        if cur != int(BG) and val != int(BG):
            assert (r, c) in conn_base
            assert cur == int(CTRL) and val == int(CTRL)
        out[int(r - br0), int(c - bc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, BG)


# ---------------------------------------------------------------------------
# Generator: build the embedded docked state first, then separate base and plug
# ---------------------------------------------------------------------------
import random as _rng


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]], allowed: np.ndarray | None = None) -> None:
    """Grow a 4-connected shape on `mask` until it reaches `target` pixels."""
    allowed = np.ones_like(mask, dtype=bool) if allowed is None else np.array(allowed, dtype=bool)
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not mask[rr, cc]:
                nbrs.append((rr, cc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        out[r, c] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return int(_gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1))


def _place_with_gap(canvas: np.ndarray, patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    H, W = canvas.shape
    h, w = patch.shape
    occ = canvas != int(bg)
    patch_mask = patch != int(bg)
    dil = _dilate4(occ)
    base_bbox = bbox_coords(np.argwhere(occ))
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            if r0 == 0 and c0 == 0:
                continue
            view = occ[r0 : r0 + h, c0 : c0 + w]
            if bool((view & patch_mask).any()):
                continue
            touch = dil[r0 : r0 + h, c0 : c0 + w]
            if bool((touch & patch_mask).any()):
                continue
            patch_pts = np.argwhere(patch_mask)
            bbox = (
                int(r0 + patch_pts[:, 0].min()),
                int(c0 + patch_pts[:, 1].min()),
                int(r0 + patch_pts[:, 0].max()),
                int(c0 + patch_pts[:, 1].max()),
            )
            gap = _bbox_gap(base_bbox, bbox)
            if 1 <= int(gap) <= 5:
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return _rng.choice(candidates)


def _build_plug() -> tuple[np.ndarray, np.ndarray, int, int]:
    conn_len = _rng.randint(1, 3)
    body_h = _rng.randint(2, 4)
    body_w = _rng.randint(max(2, conn_len), max(3, conn_len + 3))
    body = np.zeros((body_h, body_w), dtype=bool)
    conn_c0 = _rng.randint(0, body_w - conn_len)
    body[0, conn_c0 : conn_c0 + conn_len] = True
    allowed = np.ones_like(body, dtype=bool)
    allowed[0, :] = False
    allowed[0, conn_c0 : conn_c0 + conn_len] = True
    target = _rng.randint(max(conn_len + 2, 4), min(int(body.size), 12))
    frontier = [(0, c) for c in range(conn_c0, conn_c0 + conn_len)]
    _grow_shape(body, target, frontier, allowed=allowed)
    if not body[1:].any():
        return None, None, -1, -1

    plug = np.full((int(body_h + 1), int(body_w)), int(BG), dtype=int)
    plug[1:][body] = int(OBJ)
    plug[0, conn_c0 : conn_c0 + conn_len] = int(CTRL)
    plug = _crop_non_bg(plug, BG)
    conn_pts = {(int(r), int(c)) for r, c in np.argwhere(plug == int(CTRL))}
    assert conn_pts
    conn_cols = [c for _r, c in conn_pts]
    new_c0 = int(min(conn_cols))
    new_len = int(max(conn_cols) - min(conn_cols) + 1)
    body_mask = plug == int(OBJ)
    if not is_contiguous(plug != int(BG), background=0, connectivity=8):
        return None, None, -1, -1
    if int(np.sum(plug != int(BG))) < 5:
        return None, None, -1, -1
    return plug, body_mask[1:].copy(), new_c0, new_len


def _build_base_and_output(body_mask: np.ndarray, conn_c0: int, conn_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    body_h, body_w = body_mask.shape
    top_pad, left_pad, right_pad, bottom_pad = 2, 1, 1, 1
    H = int(top_pad + body_h + bottom_pad)
    W = int(left_pad + body_w + right_pad)
    if not (5 <= H <= 8 and 3 <= W <= 12):
        return None

    socket = np.zeros((H, W), dtype=bool)
    socket[top_pad : top_pad + body_h, left_pad : left_pad + body_w] = body_mask
    shell = _dilate4(socket) & (~socket)
    conn_cols = [int(left_pad + c) for c in range(conn_c0, conn_c0 + conn_len)]
    shell[top_pad - 2, conn_cols] = True

    base = np.full((H, W), int(BG), dtype=int)
    base[shell] = int(OBJ)
    base[top_pad - 1, conn_cols] = int(CTRL)
    if not is_contiguous(base != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(base != int(BG))) < 5 or int(np.sum(base != int(BG))) > 29:
        return None

    out = base.copy()
    out[socket] = int(OBJ)
    out[out == int(CTRL)] = int(OBJ)
    if not is_contiguous(out != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(out != int(BG))) > 29:
        return None
    return _crop_non_bg(base, BG), _crop_non_bg(out, BG)


def _gen_attempt() -> np.ndarray | None:
    plug, body_mask, conn_c0, conn_len = _build_plug()
    if plug is None:
        return None

    built = _build_base_and_output(body_mask, conn_c0, conn_len)
    if built is None:
        return None
    base_sprite, expected = built
    bh, bw = base_sprite.shape
    ph, pw = plug.shape

    min_h = max(7, bh + 1, ph + 1)
    min_w = max(5, bw + pw)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    pos = _place_with_gap(canvas, plug, BG)
    if pos is None:
        return None
    pr, pc = pos
    paste(canvas, plug, pr, pc, transparent=BG)

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
    for _ in range(600):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 600 attempts")

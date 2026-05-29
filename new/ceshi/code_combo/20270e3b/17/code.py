# Combo variant of 20270e3b (index 17)
# combo_spec: D4/0+D11/2
# Name: 嵌入式位置编码旋转
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题里 base 在左上角，plug 的 ctrl 横条已经在“上边”，因此只需把 plug 平移到
#      base connector 上方一行，然后在外侧拼接。
#    - 这里同时叠加两处改动：
#      1) D4/0: 对接改成“嵌入式”。base 内部先挖出一个不规则 socket，plug 的 body 必须精确
#         填入这个洞；connector 不再上下相邻，而是同一行重合。
#      2) D11/2: plug 输入态不再是 canonical 朝向。connector 贴在 plug bbox 的某一条边上，
#         该边本身编码了恢复到 canonical 朝向所需的旋转：
#         top/right/bottom/left -> 旋转 0/90/180/270 度后再嵌入。
#    - 一个小追踪：
#      原题：plug connector 在最上行，直接平移到 base connector 正上方，输出 bbox 变高。
#      融合后：若输入 plug 的 connector 在右边缘，则先逆时针旋转 90 度恢复到“上边 connector、
#      body 向下”的 canonical 形态；随后把 connector 与 base connector 同行重合，再检查
#      plug 的 obj body 是否刚好等于 base 内部从 connector 下方进入的那块 socket。输出 bbox
#      不会向外扩张，而是保持 base 的 bbox，只把洞填满。
# b) 复杂度校准
#    - 旋转只保留四档边缘编码，视觉上可从少量样例归纳。
#    - socket 不是简单矩形，而是由 canonical plug body 直接生成的不规则洞，避免退化成
#      “只看 connector 长度即可对齐”的弱规则。
#    - generator 强制 canonical plug 的 connector 不贴左右角，并拒绝旋转对称 plug，保证
#      “哪条边在编码旋转”与“旋转后形状真的不同”都清楚可见。
# c) 隐式假设审查
#    - 原始 `assert shpA == shpB` 在融合后不能直接对输入做，因为 plug 可能被旋转成竖条；
#      必须先检测 connector 所在边并恢复 canonical 朝向，再比较 connector 形状。
#    - 原始 `tr = (dr0 - 1) - pr0` 假设 plug connector 在 base connector 上方一行；融合后
#      必须改成同一行重合 `tr = dr0 - pr0`。
#    - 原始 overlap 只允许发生在边缘拼接 seam；融合后 plug body 会进入 base bbox 内部，因此
#      非 connector 区域必须覆盖 socket 的 bg，不能碰到 base obj。
#    - 原始输出 bbox 取 union 后通常比 base 更大；融合后合法样例的 union bbox 应与 base bbox
#      完全一致，因为 plug 被嵌入而非外挂。
# d) 放置合法性
#    - generator 采用 output-first：先构造 canonical plug，再用其 body 挖出 base socket，
#      得到 expected output；之后才把 plug 旋转成输入朝向并与 base 分开放回画布。
#    - 输入放置固定保证 8 邻域下仍是两个独立前景分量，不重叠、不越界，且 base 始终锚定在 (0,0)。
# e) 视觉可观察性
#    - 解题者在 2-4 组样例中会同时看到两件事：
#      1) 输出并不是把 plug 接到外侧，而是把 base 内洞填满；
#      2) 不同样例里 plug connector 会出现在不同边，输出前都要先转回再嵌入。
#    - 这两个信号都是结构性的，不依赖颜色数值或不可见排序。
# f) 可视化整洁度
#    - 仍然只有两块精灵、三种输入颜色、两种输出颜色，没有额外标记层。
#    - 重叠只发生在一条短 connector 横条上；socket 与 plug body 一一对应，视觉关系明确。
# g) 变化的可推断性
#    - 颜色规则不新增自由度：ctrl=7 只负责标识接口，输出统一改成 obj=4。
#    - 旋转由“connector 在哪条边”决定，而不是由颜色值或样例索引决定。
# h) 融合核心行为量化思路
#    - 若 generator 退化成总是把 plug 保持 top 边，则“位置编码旋转”其实没起作用。
#    - 若忽略旋转直接按输入 plug 去嵌洞也经常成功，则“先推断朝向再旋转”没有成为必要步骤。
#    - 因此测试后应统计：生成样本的 connector side 分布，以及“非 top 样本中，不旋转解法失败或
#      输出不同”的比例。
# i) main() 的新假设与 generator 保证
#    - main 假设 plug connector 恰好占据一条唯一边缘且不碰角，generator 通过 canonical
#      connector 内缩一列并再旋转来保证。
#    - main 假设 base connector 下方进入的是唯一 socket 4 连通洞，generator 直接从 plug body
#      反向构造这个洞来保证。
#    - main 假设恢复朝向后的 plug body 与 socket 完全同形，generator 通过 output-first 校验
#      `main(input) == expected` 来保证。

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7

SIDE_TO_RESTORE_K = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}

SIDE_TO_INPUT_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _component_records(grid: np.ndarray) -> list[dict[str, object]]:
    lab, n = label_mask(grid != int(BG), connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(cid))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        sprite = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(BG), dtype=int)
        for r, c in coords:
            sprite[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sprite": sprite,
                "coords": {(int(r), int(c)) for r, c in coords},
            }
        )
    return comps


def _connector_points(sprite: np.ndarray) -> np.ndarray:
    return np.argwhere(sprite == int(CTRL))


def _norm_conn(conn_pts: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    assert conn_pts.size > 0
    r0, c0 = conn_pts.min(axis=0)
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _detect_connector_side(sprite: np.ndarray) -> tuple[str, int]:
    conn = _connector_points(sprite)
    assert conn.size > 0
    rows = sorted({int(r) for r, _c in conn})
    cols = sorted({int(c) for _r, c in conn})
    h, w = sprite.shape
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[0] + len(cols)))
        if r == 0:
            assert cols[0] > 0 and cols[-1] < w - 1
            return "top", int(conn.shape[0])
        if r == h - 1:
            assert cols[0] > 0 and cols[-1] < w - 1
            return "bottom", int(conn.shape[0])
    if len(cols) == 1:
        c = cols[0]
        assert rows == list(range(rows[0], rows[0] + len(rows)))
        if c == 0:
            assert rows[0] > 0 and rows[-1] < h - 1
            return "left", int(conn.shape[0])
        if c == w - 1:
            assert rows[0] > 0 and rows[-1] < h - 1
            return "right", int(conn.shape[0])
    raise AssertionError("connector is not a unique edge code")


def _restore_plug(sprite: np.ndarray) -> tuple[np.ndarray, str]:
    side, _conn_len = _detect_connector_side(sprite)
    return np.array(np.rot90(sprite, SIDE_TO_RESTORE_K[side]), dtype=int), side


def _rotationally_distinct(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3))


def _valid_base_connector(sprite: np.ndarray) -> bool:
    conn = _connector_points(sprite)
    if conn.size == 0:
        return False
    rows = sorted({int(r) for r, _c in conn})
    cols = sorted({int(c) for _r, c in conn})
    if len(rows) != 1 or cols != list(range(cols[0], cols[0] + len(cols))):
        return False
    rr = rows[0]
    if rr == 0:
        return False
    for _r, c in conn:
        cc = int(c)
        if int(sprite[rr - 1, cc]) != int(OBJ):
            return False
        if rr + 1 < sprite.shape[0] and int(sprite[rr + 1, cc]) == int(OBJ):
            return False
    return True


def _valid_canonical_plug(sprite: np.ndarray) -> bool:
    try:
        side, _conn_len = _detect_connector_side(sprite)
    except AssertionError:
        return False
    if side != "top":
        return False
    conn = _connector_points(sprite)
    for r, c in conn:
        rr = int(r)
        cc = int(c)
        if rr != 0:
            return False
        if sprite.shape[0] < 2 or int(sprite[1, cc]) != int(OBJ):
            return False
    return True


def _socket_from_base(base_sprite: np.ndarray) -> set[tuple[int, int]]:
    conn = _connector_points(base_sprite)
    hole_mask = base_sprite == int(BG)
    hole_lab, _hole_n = label_mask(hole_mask, connectivity=4)
    socket_ids = set()
    for r, c in conn:
        rr = int(r + 1)
        cc = int(c)
        assert 0 <= rr < hole_mask.shape[0]
        assert bool(hole_mask[rr, cc])
        socket_ids.add(int(hole_lab[rr, cc]))
    assert len(socket_ids) == 1
    socket_id = socket_ids.pop()
    assert socket_id > 0
    return {(int(r), int(c)) for r, c in np.argwhere(hole_lab == int(socket_id))}


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    comps = _component_records(g)
    assert len(comps) == 2
    comps.sort(key=lambda rec: (int(rec["bbox"][0] + rec["bbox"][1]), int(rec["bbox"][0]), int(rec["bbox"][1])))
    base_rec, plug_rec = comps[0], comps[1]
    br0, bc0, br1, bc1 = base_rec["bbox"]
    assert int(br0 + bc0) == 0

    base_sprite = np.array(base_rec["sprite"], dtype=int)
    plug_input = np.array(plug_rec["sprite"], dtype=int)
    assert _valid_base_connector(base_sprite)
    plug_sprite, side = _restore_plug(plug_input)
    assert _valid_canonical_plug(plug_sprite)

    conn_base = _connector_points(base_sprite)
    conn_plug = _connector_points(plug_sprite)
    shp_base, (base_conn_r0, base_conn_c0) = _norm_conn(conn_base)
    shp_plug, (plug_conn_r0, plug_conn_c0) = _norm_conn(conn_plug)
    assert shp_base == shp_plug

    tr = int((br0 + base_conn_r0) - plug_conn_r0)
    tc = int((bc0 + base_conn_c0) - plug_conn_c0)

    socket_local = _socket_from_base(base_sprite)
    socket_abs = {(int(br0 + r), int(bc0 + c)) for r, c in socket_local}
    plug_body_abs = set()
    for r, c in np.argwhere(plug_sprite == int(OBJ)):
        plug_body_abs.add((int(tr + r), int(tc + c)))
    assert plug_body_abs == socket_abs

    plug_abs = {}
    for r in range(plug_sprite.shape[0]):
        for c in range(plug_sprite.shape[1]):
            val = int(plug_sprite[r, c])
            if val != int(BG):
                plug_abs[(int(tr + r), int(tc + c))] = val

    return {
        "grid": g,
        "base_sprite": base_sprite,
        "base_bbox": (int(br0), int(bc0), int(br1), int(bc1)),
        "plug_sprite": plug_sprite,
        "plug_abs": plug_abs,
        "plug_side": side,
        "socket_abs": socket_abs,
        "shift": (tr, tc),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the plug rotation from connector position, restore the canonical orientation,
    then embed the plug into the base socket and recolor the connector away.
    """
    info = _analyze_input(grid)
    base_sprite = np.array(info["base_sprite"], dtype=int)
    br0, bc0, br1, bc1 = info["base_bbox"]
    plug_abs = info["plug_abs"]

    coords_all = np.array(
        [(int(br0 + r), int(bc0 + c)) for r, c in np.argwhere(base_sprite != int(BG))] + list(plug_abs.keys()),
        dtype=int,
    )
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    assert (int(rr0), int(cc0), int(rr1), int(cc1)) == (int(br0), int(bc0), int(br1), int(bc1))

    out = np.array(base_sprite, dtype=int)
    for (ar, ac), val in plug_abs.items():
        rr = int(ar - br0)
        cc = int(ac - bc0)
        cur = int(out[rr, cc])
        if cur != int(BG):
            assert cur == int(CTRL) and int(val) == int(CTRL)
        out[rr, cc] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, BG)


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]
        ]
        if allowed is not None:
            nbrs = [(nr, nc) for nr, nc in nbrs if bool(allowed[nr, nc])]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = random.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    random.shuffle(pixels)
    for r, c in pixels:
        r = int(r)
        c = int(c)
        if (r, c) in keep:
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


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        out[r, c] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _build_canonical_plug() -> tuple[np.ndarray, np.ndarray, int, int] | None:
    conn_len = random.randint(1, 3)
    plug_h = random.randint(3, 5)
    min_w = max(3, conn_len + 2)
    max_w = min(8, conn_len + 5)
    if min_w > max_w:
        return None
    plug_w = random.randint(min_w, max_w)
    conn_c0 = random.randint(1, plug_w - conn_len - 1)

    mask = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(conn_c0, conn_c0 + conn_len):
        mask[0, c] = True
        mask[1, c] = True

    allowed = np.ones_like(mask, dtype=bool)
    allowed[0, :] = False
    allowed[0, conn_c0 : conn_c0 + conn_len] = True
    keep = {(0, c) for c in range(conn_c0, conn_c0 + conn_len)}
    keep.update({(1, c) for c in range(conn_c0, conn_c0 + conn_len)})

    def add_side_seed(side: str) -> bool:
        cols = list(range(conn_c0 - 1, -1, -1)) if side == "left" else list(range(conn_c0 + conn_len, plug_w))
        for c in cols:
            for r in range(1, plug_h):
                if bool(allowed[r, c]):
                    mask[r, c] = True
                    keep.add((int(r), int(c)))
                    return True
        return False

    if not add_side_seed("left") or not add_side_seed("right"):
        return None

    bonus_side = random.choice(["left", "right"])
    bonus_cols = list(range(0, conn_c0)) if bonus_side == "left" else list(range(conn_c0 + conn_len, plug_w))
    random.shuffle(bonus_cols)
    for c in bonus_cols:
        bonus_rows = list(range(2, plug_h))
        random.shuffle(bonus_rows)
        placed = False
        for r in bonus_rows:
            if bool(allowed[r, c]):
                mask[r, c] = True
                keep.add((int(r), int(c)))
                placed = True
                break
        if placed:
            break

    target = random.randint(max(5, conn_len + 4), min(12, int(mask.size)))
    _grow_shape(mask, target, list(zip(*np.where(mask))), allowed=allowed)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    _thin(mask, keep=keep)

    plug = np.full(mask.shape, int(BG), dtype=int)
    plug[mask] = int(OBJ)
    for c in range(conn_c0, conn_c0 + conn_len):
        plug[0, c] = int(CTRL)
        if int(plug[1, c]) != int(OBJ):
            return None
    if not is_contiguous(plug != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(plug != int(BG))) < 5 or int(np.sum(plug != int(BG))) > 29:
        return None

    plug = _crop_non_bg(plug, BG)
    if not _valid_canonical_plug(plug):
        return None
    if not _rotationally_distinct(plug):
        return None

    conn = _connector_points(plug)
    cols = sorted({int(c) for _r, c in conn})
    body_mask = plug == int(OBJ)
    return plug, body_mask[1:].copy(), int(cols[0]), int(len(cols))


def _build_base_and_expected(body_mask: np.ndarray, conn_c0: int, conn_len: int) -> tuple[np.ndarray, np.ndarray] | None:
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
    if not _valid_base_connector(base):
        return None
    if int(np.sum(base != int(BG))) < 5 or int(np.sum(base != int(BG))) > 29:
        return None

    expected = np.array(base, dtype=int)
    expected[socket] = int(OBJ)
    expected[expected == int(CTRL)] = int(OBJ)
    if not is_contiguous(expected != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(expected != int(BG))) > 29:
        return None

    base = _crop_non_bg(base, BG)
    expected = _crop_non_bg(expected, BG)
    return base, expected


def _place_sprites(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    layouts = []
    if max(7, max(bh, ph)) <= 11 and max(5, bw + 1 + pw) <= 13:
        layouts.append("horizontal")
    if max(7, bh + 1 + ph) <= 11 and max(5, max(bw, pw)) <= 13:
        layouts.append("vertical")
    if not layouts:
        return None

    layout = random.choice(layouts)
    if layout == "horizontal":
        canvas_h = random.randint(max(7, max(bh, ph)), 11)
        canvas_w = random.randint(max(5, bw + 1 + pw), 13)
        pr = random.randint(0, canvas_h - ph)
        min_pc = bw + 1
        max_pc = min(canvas_w - pw, bw + 5)
        if min_pc > max_pc:
            return None
        pc = random.randint(min_pc, max_pc)
    else:
        canvas_h = random.randint(max(7, bh + 1 + ph), 11)
        canvas_w = random.randint(max(5, max(bw, pw)), 13)
        min_pr = bh + 1
        max_pr = min(canvas_h - ph, bh + 5)
        if min_pr > max_pr:
            return None
        pr = random.randint(min_pr, max_pr)
        pc = random.randint(0, canvas_w - pw)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    paste(canvas, plug_sprite, pr, pc, transparent=int(BG))

    lab, n = label_mask(canvas != int(BG), connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    built = _build_canonical_plug()
    if built is None:
        return None
    plug_canonical, body_mask, conn_c0, conn_len = built

    base_built = _build_base_and_expected(body_mask, conn_c0, conn_len)
    if base_built is None:
        return None
    base_sprite, expected = base_built

    input_side = random.choice(["top", "right", "bottom", "left"])
    plug_input = np.array(np.rot90(plug_canonical, SIDE_TO_INPUT_K[input_side]), dtype=int)
    side_check, side_len = _detect_connector_side(plug_input)
    if side_check != input_side or side_len != conn_len:
        return None

    canvas = _place_sprites(base_sprite, plug_input)
    if canvas is None:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(800):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 800 attempts")

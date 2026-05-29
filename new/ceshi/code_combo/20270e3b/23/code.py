# Combo variant of 20270e3b (index 23)
# combo_spec: D1/1+D4/0+D11/0
# Name: 旋转嵌入嵌合匹配

from __future__ import annotations

from collections import deque
import random
import random as _rng
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) 具体化与行为追踪
#    - 原题里两个 connector 都是同长度水平条，plug 只需平移到 base 外边缘上方一行。
#    - 本融合版把三件事绑成一条推理链：
#      1. connector 变成 2 x w 的锯齿带。每列恰有一个 ctrl，row-profile 形如 [0,1,0,1]。
#         base 与 plug 的 profile 必须逐列互补。
#      2. plug 的 canonical 朝向仍然是“connector 在上，body 在下”，但输入里会先把 plug
#         做一次四分之一转动，所以原始输入中的 plug connector 常是竖着的。
#      3. base 不再在外边缘等待拼接，而是有一个内部 socket。把 plug 转回正确朝向后，
#         它的 obj-body 必须填进 base 的 bg 凹槽里；只有 connector mouth 允许精确的
#         ctrl-vs-obj 交错覆盖。
#    - 小例子：
#      旧题：base bar 在第 4 行，plug bar 平移到第 3 行，上下贴边，输出 bbox 变高。
#      新题：base 有一个内部 cavity，mouth 是 2 x 4 锯齿带；输入里 plug 先被转成竖直。
#      solver 先尝试把 plug 旋回，再读出锯齿 profile，比如 plug=[0,1,0,1]，则 base 必须是
#      [1,0,1,0]。只有该旋转下，plug 的 body 才会一部分覆盖 base connector、一部分填入
#      cavity，且整体 bbox 不会超出 base。输出依旧把 ctrl 全部改成 obj。
#
# b) 复杂度校准
#    - connector 宽度限制在 3..5，并强制 profile 同时包含 0/1 且至少有一次转折，避免退化回
#      平条或过长导致匹配过于自由。
#    - 旋转只使用非平凡四分之一转动（输入态取 90 或 270 度），这样肉眼能直接看出 plug 方向不对。
#    - 仍然只有两个前景组件、三种固定颜色，推理难点集中在“识别互补锯齿 + 判断旋转 + 验证嵌入”。
#
# c) 隐式假设审查（相对原 main）
#    - 原 `shpA == shpB` 不成立：现在匹配条件是 complementary profile，而不是相同条形。
#    - 原 `tr = (dr0 - 1) - pr0` 也不成立：现在先旋转枚举，再把整个 2-row band 顶部对齐。
#    - 原“plug 可以伸出 base bbox”不成立：嵌入式对接要求 plug 全部落在 base bbox 内。
#    - 原“任意含 ctrl 的 overlap 都行”过宽：现在 overlap 只允许两类精确 mouth 交错：
#      plug ctrl 覆盖指定 base obj，或 plug obj 覆盖指定 base ctrl；其余碰撞全部非法。
#
# d) 放置合法性
#    - generator 先生成 canonical plug，再围绕它构造带 socket 的 base，最后再把 plug 旋转后
#      随机散放到输入画布上。这样 oracle 可直接验证 `main(input) == expected`。
#    - base 始终裁到左上角 `(0,0)`；plug 放置时与 base 保持至少 1 格 8 邻域间隔，避免组件粘连。
#
# e) 视觉可观察性
#    - 从样例上，人类能同时看到：
#      1. base 内部有明显凹槽；
#      2. connector 不是平条而是锯齿带；
#      3. plug 在输入里方向不兼容，输出里则被转回并嵌入。
#    - 三个变化都体现在同一处 mouth/socket 几何，不依赖颜色编号等不可视因素。
#
# f) 可视化整洁度
#    - mouth 的重叠只发生在 connector 带附近；socket 内部仍是干净的 bg 洞。
#    - plug 顶部两行被刻意清空到只剩 connector 区域，减少杂乱像素，让“需要旋转后插入”更清晰。
#
# g) 颜色变化可推断性
#    - 仍然只有 bg=1, obj=4, ctrl=7。ctrl 只作为对接口标记，输出统一 recolor 成 obj。
#    - 没有新增颜色映射；颜色规则与几何规则绑定，解题者只需推断“connector 消失后并入主体”。
#
# h) 新 solver 假设与 generator 保证
#    - 两个组件中恰有一个 base 触及输入原点。
#    - 每个 sprite 恰有一个 2-row zigzag connector；base profile 与 plug profile 互补。
#    - 在正确旋转下，plug ctrl 与 base obj 的重叠列、plug obj 与 base ctrl 的重叠列都由
#      profile 精确决定；generator 明确按此构造并过滤唯一解。


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_points(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(cid))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    return comps


def _sprite_from_points(
    grid: np.ndarray, pts: set[tuple[int, int]], bg: int = BG
) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_of_pts(pts)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out, (int(r0), int(c0))


def _connector_info_from_arr(arr: np.ndarray) -> dict[str, object]:
    conn = {(int(r), int(c)) for r, c in np.argwhere(arr == int(CTRL))}
    assert conn
    r0, c0, r1, c1 = _bbox_of_pts(conn)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in conn}
    assert h == 2 and 3 <= w <= 5
    assert len(rel) == w
    profile = []
    for dc in range(w):
        rows = [dr for dr, cc in rel if cc == dc]
        assert len(rows) == 1 and rows[0] in (0, 1)
        profile.append(int(rows[0]))
    assert set(profile) == {0, 1}
    assert sum(profile[i] != profile[i - 1] for i in range(1, len(profile))) >= 1
    return {
        "pts": conn,
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "profile": tuple(profile),
    }


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


def _coords_of(arr: np.ndarray, val: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr == int(val))}


def _assemble_embedded(base_arr: np.ndarray, plug_arr: np.ndarray) -> np.ndarray | None:
    try:
        base_info = _connector_info_from_arr(base_arr)
        plug_info = _connector_info_from_arr(plug_arr)
    except Exception:
        return None

    if int(base_info["w"]) != int(plug_info["w"]):
        return None
    prof_base = tuple(int(x) for x in base_info["profile"])
    prof_plug = tuple(int(x) for x in plug_info["profile"])
    if tuple(1 - x for x in prof_plug) != prof_base:
        return None

    base_body = _coords_of(base_arr, OBJ)
    plug_body = _coords_of(plug_arr, OBJ)
    base_conn = set(base_info["pts"])
    plug_conn = set(plug_info["pts"])

    for r, c in base_conn:
        if (int(r + 1), int(c)) in base_body:
            return None
    for r, c in plug_conn:
        if (int(r - 1), int(c)) in plug_body:
            return None

    br0 = int(base_info["r0"])
    bc0 = int(base_info["c0"])
    pr0 = int(plug_info["r0"])
    pc0 = int(plug_info["c0"])
    width = int(base_info["w"])

    for dc, qbit in enumerate(prof_plug):
        base_c = int(bc0 + dc)
        plug_c = int(pc0 + dc)
        if int(qbit) == 0:
            if (int(pr0 + 1), int(plug_c)) not in plug_body:
                return None
            if (int(br0), int(base_c)) not in base_body:
                return None
        else:
            if (int(pr0 + 2), int(plug_c)) not in plug_body:
                return None
            if (int(br0 - 1), int(base_c)) not in base_body:
                return None

    tr = int(br0 - pr0)
    tc = int(bc0 - pc0)
    shifted_ctrl = {(int(r + tr), int(c + tc)) for r, c in plug_conn}
    shifted_obj = {(int(r + tr), int(c + tc)) for r, c in plug_body}

    h, w = map(int, base_arr.shape)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted_ctrl | shifted_obj):
        return None

    expected_ctrl_on_obj = {
        (int(br0), int(bc0 + dc)) for dc, qbit in enumerate(prof_plug) if int(qbit) == 0
    }
    expected_obj_on_ctrl = {
        (int(br0 + 1), int(bc0 + dc)) for dc, qbit in enumerate(prof_plug) if int(qbit) == 0
    }

    socket_cells = set()
    for r, c in shifted_ctrl:
        cur = int(base_arr[int(r), int(c)])
        if (int(r), int(c)) in expected_ctrl_on_obj:
            if cur != int(OBJ):
                return None
        else:
            if cur != int(BG):
                return None

    for r, c in shifted_obj:
        cur = int(base_arr[int(r), int(c)])
        if (int(r), int(c)) in expected_obj_on_ctrl:
            if cur != int(CTRL):
                return None
        else:
            if cur != int(BG):
                return None
            socket_cells.add((int(r), int(c)))

    if not socket_cells:
        return None
    if any(r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1 for r, c in socket_cells):
        return None

    out = base_arr.copy()
    for r, c in shifted_ctrl:
        out[int(r), int(c)] = int(CTRL)
    for r, c in shifted_obj:
        out[int(r), int(c)] = int(OBJ)
    out[out == int(CTRL)] = int(OBJ)
    cropped = _crop_non_bg(out, BG)
    if cropped is None:
        return None
    if cropped.shape != base_arr.shape:
        return None
    return cropped


def _candidate_outputs(base_arr: np.ndarray, plug_input_arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, np.ndarray]] = []
    for rot_k in (1, 3):
        rotated = _crop_non_bg(np.rot90(plug_input_arr, int(rot_k)), BG)
        if rotated is None:
            continue
        assembled = _assemble_embedded(base_arr, rotated)
        if assembled is not None:
            candidates.append((int(rot_k), assembled))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the plug back by a quarter-turn, align complementary zigzag connectors,
    verify the plug embeds into the base socket, then recolor ctrl->obj.
    """
    g = np.array(grid, dtype=int)
    comps = _component_points(g != int(BG))
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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _random_profile(width: int) -> tuple[int, ...]:
    while True:
        prof = [random.randint(0, 1) for _ in range(int(width))]
        if set(prof) != {0, 1}:
            continue
        if sum(prof[i] != prof[i - 1] for i in range(1, len(prof))) < 1:
            continue
        return tuple(int(x) for x in prof)


def _paint_path(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    allowed: np.ndarray,
) -> bool:
    h, w = mask.shape
    if not bool(allowed[int(start[0]), int(start[1])]) or not bool(allowed[int(goal[0]), int(goal[1])]):
        return False
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        _rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not bool(allowed[int(nr), int(nc)]):
                continue
            if (int(nr), int(nc)) in prev:
                continue
            prev[(int(nr), int(nc))] = (int(r), int(c))
            q.append((int(nr), int(nc)))
    if goal not in prev:
        return False
    cur: tuple[int, int] | None = goal
    while cur is not None:
        mask[int(cur[0]), int(cur[1])] = True
        cur = prev[cur]
    return True


def _grow_shape(mask: np.ndarray, target: int, allowed: np.ndarray) -> None:
    h, w = mask.shape
    while int(mask.sum()) < int(target):
        frontier = []
        for r, c in zip(*np.where(mask)):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(allowed[int(nr), int(nc)]) and not bool(mask[int(nr), int(nc)]):
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return
        nr, nc = _rng.choice(frontier)
        mask[int(nr), int(nc)] = True


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        if any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _connect_components(mask: np.ndarray, allowed: np.ndarray) -> bool:
    while True:
        lab, n = label_mask(mask, connectivity=8)
        if int(n) <= 1:
            return bool(mask.any())
        reps = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            reps.append((int(pts[0, 0]), int(pts[0, 1])))
        root = reps[0]
        for rep in reps[1:]:
            if not _paint_path(mask, root, rep, allowed):
                return False


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


def _place_with_gap(canvas: np.ndarray, patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = canvas.shape
    occ = canvas != int(bg)
    patch_mask = patch != int(bg)
    dil = _dilate4(occ)
    base_bbox = bbox_coords(np.argwhere(occ))
    candidates: list[tuple[int, int]] = []
    patch_pts = np.argwhere(patch_mask)
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            if int(r0 + c0) == 0:
                continue
            view = occ[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
            if bool((view & patch_mask).any()):
                continue
            touch = dil[int(r0) : int(r0 + h), int(c0) : int(c0 + w)]
            if bool((touch & patch_mask).any()):
                continue
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


def _build_canonical_plug(profile: tuple[int, ...]) -> np.ndarray | None:
    conn_w = int(len(profile))
    plug_h = _rng.randint(4, 7)
    plug_w = _rng.randint(max(3, conn_w), min(8, conn_w + 3))
    if plug_h <= 2:
        return None
    conn_c0 = _rng.randint(0, int(plug_w - conn_w))

    body = np.zeros((int(plug_h), int(plug_w)), dtype=bool)
    allowed = np.ones_like(body, dtype=bool)
    allowed[0, :] = False
    allowed[1, :] = False
    allowed[1, conn_c0 : conn_c0 + conn_w] = True

    conn_pts = []
    required: set[tuple[int, int]] = set()
    for i, bit in enumerate(profile):
        c = int(conn_c0 + i)
        cr = int(bit)
        conn_pts.append((cr, c))
        if int(bit) == 0:
            required.add((1, c))
        else:
            if plug_h < 3:
                return None
            required.add((2, c))

    bottom_anchor = (int(plug_h - 1), _rng.randint(0, int(plug_w - 1)))
    if not bool(allowed[int(bottom_anchor[0]), int(bottom_anchor[1])]):
        return None
    required.add(bottom_anchor)

    req_list = sorted(required)
    body[req_list[0]] = True
    for pt in req_list[1:]:
        if not _paint_path(body, req_list[0], pt, allowed):
            return None

    lo = max(5, int(body.sum()))
    hi = min(int(plug_h * plug_w), max(lo, 16))
    target = _rng.randint(lo, hi)
    _grow_shape(body, target, allowed)

    keep = set(required)
    _thin(body, keep=keep)
    if not is_contiguous(body, background=0, connectivity=8):
        return None

    grid = np.full((int(plug_h), int(plug_w)), int(BG), dtype=int)
    grid[body] = int(OBJ)
    for r, c in conn_pts:
        grid[int(r), int(c)] = int(CTRL)

    cropped = _crop_non_bg(grid, BG)
    if cropped is None:
        return None
    if not is_contiguous(cropped != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(cropped != int(BG))) < 5 or int(np.sum(cropped != int(BG))) > 18:
        return None
    return cropped


def _build_base_from_plug(plug_sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    info = _connector_info_from_arr(plug_sprite)
    prof_plug = tuple(int(x) for x in info["profile"])
    plug_ctrl = _coords_of(plug_sprite, CTRL)
    plug_obj = _coords_of(plug_sprite, OBJ)
    ph, pw = map(int, plug_sprite.shape)
    conn_w = int(info["w"])

    dock_top = _rng.randint(1, 2)
    dock_left = _rng.randint(1, 2)
    bottom_pad = _rng.randint(1, 2)
    right_pad = _rng.randint(1, 2)
    H = int(dock_top + ph + bottom_pad)
    W = int(dock_left + pw + right_pad)
    if H > 9 or W > 10:
        return None

    shifted_ctrl = {(int(r + dock_top), int(c + dock_left)) for r, c in plug_ctrl}
    shifted_obj = {(int(r + dock_top), int(c + dock_left)) for r, c in plug_obj}

    base_ctrl = {
        (int(dock_top + (1 - bit)), int(dock_left + dc))
        for dc, bit in enumerate(prof_plug)
    }
    expected_ctrl_on_obj = {
        (int(dock_top), int(dock_left + dc)) for dc, bit in enumerate(prof_plug) if int(bit) == 0
    }
    expected_obj_on_ctrl = {
        (int(dock_top + 1), int(dock_left + dc)) for dc, bit in enumerate(prof_plug) if int(bit) == 0
    }
    socket = set(shifted_obj) - expected_obj_on_ctrl
    if len(socket) < 3:
        return None
    if any(r <= 0 or c <= 0 or r >= H - 1 or c >= W - 1 for r, c in socket):
        return None

    forbidden_ctrl_bg = set(shifted_ctrl) - expected_ctrl_on_obj
    allowed = np.ones((int(H), int(W)), dtype=bool)
    for r, c in socket | base_ctrl | forbidden_ctrl_bg:
        allowed[int(r), int(c)] = False

    shell = _dilate4(_mask_from_points(socket, (H, W))) & allowed
    if not shell.any():
        return None

    body = np.array(shell, dtype=bool)
    required = {(0, 0)}
    for dc, bit in enumerate(prof_plug):
        c = int(dock_left + dc)
        if int(bit) == 0:
            required.add((int(dock_top), c))
        else:
            required.add((int(dock_top - 1), c))

    for r, c in required:
        if not (0 <= r < H and 0 <= c < W):
            return None
        if not bool(allowed[int(r), int(c)]):
            return None
        body[int(r), int(c)] = True

    if not _connect_components(body, allowed):
        return None

    target = min(24, int(body.sum()) + _rng.randint(0, 3))
    _grow_shape(body, target, allowed)
    _thin(body, keep=required | expected_ctrl_on_obj)

    base = np.full((int(H), int(W)), int(BG), dtype=int)
    base[body] = int(OBJ)
    for r, c in base_ctrl:
        base[int(r), int(c)] = int(CTRL)

    base = _crop_non_bg(base, BG)
    if base is None:
        return None
    if not is_contiguous(base != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(base != int(BG))) < 5 or int(np.sum(base != int(BG))) > 29:
        return None

    expected = _assemble_embedded(base, plug_sprite)
    if expected is None:
        return None
    return base, expected


def _mask_from_points(pts: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in pts:
        out[int(r), int(c)] = True
    return out


def _gen_attempt() -> np.ndarray | None:
    conn_w = _rng.randint(3, 5)
    profile = _random_profile(conn_w)

    plug_sprite = _build_canonical_plug(profile)
    if plug_sprite is None:
        return None

    built = _build_base_from_plug(plug_sprite)
    if built is None:
        return None
    base_sprite, expected = built

    ph0, pw0 = map(int, plug_sprite.shape)
    if ph0 == pw0:
        return None

    rot_choices: list[np.ndarray] = []
    for input_rot in (1, 3):
        plug_input = _crop_non_bg(np.rot90(plug_sprite, int(input_rot)), BG)
        if plug_input is None:
            continue
        found = _candidate_outputs(base_sprite, plug_input)
        need_rot = int((-input_rot) % 4)
        if len(found) == 1 and found[0][0] == need_rot and np.array_equal(found[0][1], expected):
            rot_choices.append(plug_input)
    if not rot_choices:
        return None
    plug_input = _rng.choice(rot_choices)

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_input.shape)
    min_h = max(7, bh, ph)
    min_w = max(5, bw + pw + 1)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)

    canvas = np.full((int(canvas_h), int(canvas_w)), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    pos = _place_with_gap(canvas, plug_input, BG)
    if pos is None:
        return None
    pr, pc = pos
    paste(canvas, plug_input, int(pr), int(pc), transparent=BG)

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
    for _ in range(800):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 800 attempts")

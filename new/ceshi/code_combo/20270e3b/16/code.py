# Combo variant of 20270e3b (index 16)
# combo_spec: D1/3+D9/2
# Name: 边缘模式T形干扰

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG, OBJ, CTRL = 1, 4, 7

T4 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
T5 = frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)})
X5 = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
DECOR_SHAPES = (T4, T5, X5)


# ---------------------------------------------------------------------------
# 设计分析（融合版：D1/3 + D9/2）
# ---------------------------------------------------------------------------
# a) 具体化与行为追踪
# - 原题：两块 sprite 都有可配对的 ctrl 横条；solver 找两条 ctrl，确认形状相同，
#   然后把 plug 平移到 base 上方一行，让 connector 对接。
# - 本融合版：真正的“配对信号”改成边缘行/列模式，不再使用 ctrl 来决定位置。
#   具体说，base 的 bbox 最底行有哪些前景列，必须与 plug 的 bbox 最顶行前景列
#   （各自按 bbox 左边归一化）完全一致；solver 因而把 plug 放到 base 正下方一行。
# - 同时保留 ctrl，但它们只是装饰性干扰：每个 sprite 表面都有一个小型 T/十字形
#   ctrl 图案。它看起来像 connector，却不参与匹配；最终输出仍把所有 ctrl 重着色成 obj。
# - 小例子：
#   原题会用 ctrl 条 `row=3, cols={2,3}` 与 `row=0, cols={2,3}` 对齐，所以 plug 去 base 上面。
#   本题则看 `base.bottom={0,2,3}` 与 `plug.top={0,2,3}`，所以 plug 去 base 下面。
#   即使两块上各自有 T/十字 ctrl，位置仍由边缘模式决定。
#
# b) 复杂度校准
# - seam pattern 强制带 gap，至少是 2 个且不是整段连续列。这样视觉上不会退化成
#   “又是一条横 connector”，而是能看出是稀疏列模式的拼边。
# - 装饰 ctrl 只从三个小模板里取：T4 / T5 / X5，不旋转。模板家族很小，
#   可观察但不会把自由度放得过大。
# - base 与 plug 的装饰形状允许不同，避免解题者把“ctrl 形状相同”误判为真正规则。
#
# c) 原 main() 的隐式假设审查
# - 原 `conn(sprite)` / `norm(conn)` / `assert shpA == shpB)` 假设“ctrl 就是 docking 点”，
#   在融合后失效，必须移除；ctrl 只需要被识别为合法装饰模板，不需要跨 sprite 匹配。
# - 原 `base_body = base - connD` 与“free side faces down/up”假设 connector 决定朝向，
#   在融合后也不成立，删除。
# - 原 `tr = (dr0 - 1) - pr0` 会把 plug 放到 base 上方；融合版应改成
#   `tr = base_bbox_bottom + 1 - plug_bbox_top`，即按边缘模式做上下拼接。
# - 原 overlap 逻辑允许 ctrl 与 obj 在 docking 区共享像素；融合版的两块只边对边相邻，
#   不允许任何重叠。
#
# d) 放置合法性
# - generator 先构造满足 `base.bottom == plug.top` 的 docked 主体，再插入不碰 seam 的
#   T/十字装饰，最后把 plug 散放回输入画布。
# - base 仍固定贴在 `(0,0)`；plug 使用 `padding=1` 随机放置，保证两块在输入中
#   既不重叠，也不 8 邻接粘成一个组件。
# - 装饰模板禁止落在 base 的底边或 plug 的顶边，否则会污染真正的 edge-match 信号。
#
# e) 视觉可观察性
# - 人类看到样例时，应能观察到：
#   1) 接缝来自“上块底边”和“下块顶边”的稀疏列模式；
#   2) T/十字 ctrl 明显存在，但有时两块形状不同，说明它不是匹配依据；
#   3) 输出里 ctrl 全部并入 obj。
# - 因为 seam 一直带 gap，而 ctrl 是二维小图案，两种线索在视觉上可区分。
#
# f) 可视化整洁度
# - 输入只有两块主体，没有额外离散噪点；装饰 ctrl 只有一簇，且不叠到 seam 上。
# - 输出中两块只是上下相邻，不共享像素，轮廓清晰。
#
# g) 变化的可推断性
# - 颜色仍只有 `bg=1, obj=4, ctrl=7` 三种，且唯一颜色变化是 `ctrl -> obj`。
# - 这个颜色变化伴随明确结构线索：ctrl 总是组成一个小 T/十字装饰，所以人类可以从
#   少量样例推断“装饰被吞并为主体色”，而不是任意颜色映射。
#
# main() 的新隐式假设与 generator 的对应保证
# - 假设 1：恰有两个 8 连通前景组件。generator 用 padding=1 的散放保证。
# - 假设 2：base 的底边列模式与 plug 的顶边列模式完全一致且带 gap。generator 强制。
# - 假设 3：每个 sprite 恰有一个 8 连通 ctrl 模板，且模板属于 T4/T5/X5。generator 强制。
# - 假设 4：ctrl 不落在 base.bottom / plug.top seam 行。generator 的装饰放置显式禁止。
# - 假设 5：按 bbox 左边对齐后，plug 在 docked 输出中不会伸到 base 左边。generator 通过
#   先裁剪再统一左对齐构造，保证输出 tight crop 后仍以 base 为左上角。


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_points(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (r0, c0)


def _shape_hw(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return int(max(r for r, _c in shape) + 1), int(max(c for _r, c in shape) + 1)


def _crop_non_bg(grid: np.ndarray) -> np.ndarray | None:
    nz = np.argwhere(grid != BG)
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _edge_cols(pts: set[tuple[int, int]], which: str) -> tuple[int, ...]:
    r0, c0, r1, _c1 = _bbox_of_pts(pts)
    row = r0 if which == "top" else r1
    cols = sorted(int(c - c0) for r, c in pts if int(r) == row)
    assert cols
    return tuple(cols)


def _has_gap(cols: tuple[int, ...]) -> bool:
    return any(int(b - a) > 1 for a, b in zip(cols, cols[1:]))


def _mask_to_pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _sprite_ctrl_shape(sprite: set[tuple[int, int]], grid: np.ndarray) -> tuple[frozenset[tuple[int, int]], set[tuple[int, int]]]:
    ctrl_pts = {(int(r), int(c)) for r, c in sprite if int(grid[int(r), int(c)]) == CTRL}
    assert ctrl_pts
    shape, _origin = _normalize_points(ctrl_pts)
    assert shape in DECOR_SHAPES
    r0, c0, r1, c1 = _bbox_of_pts(ctrl_pts)
    local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in ctrl_pts:
        local[int(r - r0), int(c - c0)] = True
    _lab, n = label_mask(local, connectivity=8)
    assert int(n) == 1
    return shape, ctrl_pts


def _assemble_aligned(base_pts: set[tuple[int, int]], plug_pts: set[tuple[int, int]], grid: np.ndarray) -> np.ndarray:
    br0, bc0, br1, _bc1 = _bbox_of_pts(base_pts)
    pr0, pc0, _pr1, _pc1 = _bbox_of_pts(plug_pts)
    tr = int(br1 + 1 - pr0)
    tc = int(bc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug_pts}
    assert base_pts.isdisjoint(plug_shift)

    coords_all = np.array(list(base_pts | plug_shift), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base_pts:
        out[int(r - rr0), int(c - cc0)] = int(grid[int(r), int(c)])
    for r, c in plug_shift:
        src_r, src_c = int(r - tr), int(c - tc)
        assert int(out[int(r - rr0), int(c - cc0)]) == BG
        out[int(r - rr0), int(c - cc0)] = int(grid[src_r, src_c])

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        if pts:
            comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
        br0, bc0, br1 = r0a, c0a, r1a
    else:
        base, plug = B, A
        br0, bc0, br1 = r0b, c0b, r1b

    _shape_base, ctrl_base = _sprite_ctrl_shape(base, g)
    _shape_plug, ctrl_plug = _sprite_ctrl_shape(plug, g)

    base_bottom = _edge_cols(base, "bottom")
    plug_top = _edge_cols(plug, "top")
    assert base_bottom == plug_top
    assert len(base_bottom) >= 2 and _has_gap(base_bottom)

    for r, _c in ctrl_base:
        assert int(r) != int(br1)
    pr0, pc0, _pr1, _pc1 = _bbox_of_pts(plug)
    for r, _c in ctrl_plug:
        assert int(r) != int(pr0)

    tr = int(br1 + 1 - pr0)
    tc = int(bc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    assert base.isdisjoint(plug_shift)

    return _assemble_aligned(base, plug, g)


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    forbidden: set[tuple[int, int]] | None = None,
) -> None:
    forbidden = forbidden or set()
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc] or (nr, nc) in forbidden:
                continue
            nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
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


def _choose_seam_cols(width: int) -> tuple[int, ...] | None:
    max_k = min(4, int(width) - 1)
    if max_k < 2:
        return None
    for _ in range(60):
        k = _rng.randint(2, max_k)
        cols = tuple(sorted(_rng.sample(range(int(width)), k)))
        if _has_gap(cols):
            return cols
    return None


def _carve_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int], inward_step: int) -> None:
    h, _w = mask.shape
    r, c = map(int, start)
    gr, gc = map(int, goal)
    mask[r, c] = True
    if r == 0 and inward_step > 0 and h >= 2:
        r = 1
        mask[r, c] = True
    elif r == h - 1 and inward_step < 0 and h >= 2:
        r = h - 2
        mask[r, c] = True
    while (r, c) != (gr, gc):
        moves = []
        if r != gr:
            moves.append((int(r + (1 if gr > r else -1)), c))
        if c != gc:
            moves.append((r, int(c + (1 if gc > c else -1))))
        if not moves:
            break
        nr, nc = moves[0] if len(moves) == 1 or _rng.random() < 0.6 else moves[1]
        r, c = nr, nc
        mask[r, c] = True


def _make_base_mask(height: int, width: int, seam_cols: tuple[int, ...]) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, 0] = True
    keep = {(0, 0)}
    for c in seam_cols:
        goal_choices = [(int(r), int(cc)) for r, cc in zip(*np.where(mask)) if int(r) < int(height) - 1]
        if not goal_choices:
            return None
        goal = _rng.choice(goal_choices)
        _carve_path(mask, (int(height) - 1, int(c)), goal, inward_step=-1)
        keep.add((int(height) - 1, int(c)))
    lower = max(5, int(mask.sum()))
    upper = min(24, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    forbidden = {(int(height) - 1, c) for c in range(int(width)) if c not in seam_cols}
    _grow_shape(mask, _rng.randint(lower, upper), list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if tuple(sorted(int(c) for c in np.where(mask[-1])[0])) != seam_cols:
        return None
    return mask


def _make_plug_mask(height: int, width: int, seam_cols: tuple[int, ...]) -> np.ndarray | None:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    anchor_col = int(_rng.choice(seam_cols))
    mask[int(height) - 1, anchor_col] = True
    keep = {(0, int(c)) for c in seam_cols}
    for c in seam_cols:
        goal_choices = [(int(r), int(cc)) for r, cc in zip(*np.where(mask)) if int(r) > 0]
        if not goal_choices:
            return None
        goal = _rng.choice(goal_choices)
        _carve_path(mask, (0, int(c)), goal, inward_step=1)
    lower = max(5, int(mask.sum()))
    upper = min(24, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    forbidden = {(0, c) for c in range(int(width)) if c not in seam_cols}
    _grow_shape(mask, _rng.randint(lower, upper), list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    if tuple(sorted(int(c) for c in np.where(mask[0])[0])) != seam_cols:
        return None
    return mask


def _decorate_with_shape(sprite: np.ndarray, role: str, shape: frozenset[tuple[int, int]]) -> np.ndarray | None:
    mask = sprite != BG
    h, w = mask.shape
    sh, sw = _shape_hw(shape)
    if h < sh or w < sw:
        return None

    seam_name = "bottom" if role == "base" else "top"
    seam_sig = _edge_cols(_mask_to_pts(mask), seam_name)
    forbidden_row = int(h - 1) if role == "base" else 0

    row_values = list(range(0, h - sh + 1))
    col_values = list(range(0, w - sw + 1))
    if role == "base":
        row_values.sort(key=lambda r: (r, _rng.random()))
    else:
        row_values.sort(key=lambda r: (abs((h - sh) - r), _rng.random()))
    _rng.shuffle(col_values)

    for r0 in row_values:
        for c0 in col_values:
            pts = {(int(r0 + r), int(c0 + c)) for r, c in shape}
            if any(int(r) == forbidden_row for r, _c in pts):
                continue
            mask2 = mask.copy()
            for r, c in pts:
                mask2[int(r), int(c)] = True
            if int(mask2.sum()) > 29:
                continue
            if not is_contiguous(mask2, background=0, connectivity=8):
                continue
            if _edge_cols(_mask_to_pts(mask2), seam_name) != seam_sig:
                continue

            exposed = 0
            for r, c in pts:
                if int(r) in (0, h - 1) or int(c) in (0, w - 1):
                    exposed += 1
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if not mask2[nr, nc]:
                        exposed += 1
                        break
            if exposed < 2:
                continue

            out = np.full(sprite.shape, BG, dtype=int)
            out[mask2] = OBJ
            for r, c in pts:
                out[int(r), int(c)] = CTRL
            return out
    return None


def _decorate_sprite(sprite: np.ndarray, role: str, preferred: frozenset[tuple[int, int]]) -> np.ndarray | None:
    order = [preferred] + [shape for shape in DECOR_SHAPES if shape != preferred]
    for shape in order:
        out = _decorate_with_shape(sprite, role, shape)
        if out is not None:
            return out
    return None


def _assemble_expected(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    out = np.full((int(bh + ph), int(max(bw, pw))), BG, dtype=int)
    paste(out, base_sprite, 0, 0, transparent=BG)
    paste(out, plug_sprite, bh, 0, transparent=BG)
    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _ctrl_pref_pair() -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    base_shape = _rng.choice(DECOR_SHAPES)
    if _rng.random() < 0.65:
        others = [shape for shape in DECOR_SHAPES if shape != base_shape]
        plug_shape = _rng.choice(others)
    else:
        plug_shape = _rng.choice(DECOR_SHAPES)
    return base_shape, plug_shape


def _gen_attempt() -> np.ndarray | None:
    width = _rng.randint(4, 9)
    seam_cols = _choose_seam_cols(width)
    if seam_cols is None:
        return None

    base_h = _rng.randint(3, 5)
    plug_h_max = min(5, 8 - int(base_h))
    if plug_h_max < 2:
        return None
    plug_h = _rng.randint(2, plug_h_max)

    base_mask = _make_base_mask(base_h, width, seam_cols)
    plug_mask = _make_plug_mask(plug_h, width, seam_cols)
    if base_mask is None or plug_mask is None:
        return None

    base_grid = np.full(base_mask.shape, BG, dtype=int)
    base_grid[base_mask] = OBJ
    plug_grid = np.full(plug_mask.shape, BG, dtype=int)
    plug_grid[plug_mask] = OBJ

    base_sprite = _crop_non_bg(base_grid)
    plug_sprite = _crop_non_bg(plug_grid)
    if base_sprite is None or plug_sprite is None:
        return None

    base_sig = _edge_cols(_mask_to_pts(base_sprite != BG), "bottom")
    plug_sig = _edge_cols(_mask_to_pts(plug_sprite != BG), "top")
    if base_sig != plug_sig or not _has_gap(base_sig):
        return None

    pref_base, pref_plug = _ctrl_pref_pair()
    base_decor = _decorate_sprite(base_sprite, "base", pref_base)
    plug_decor = _decorate_sprite(plug_sprite, "plug", pref_plug)
    if base_decor is None or plug_decor is None:
        return None

    if int(np.sum(base_decor != BG)) < 5 or int(np.sum(plug_decor != BG)) < 5:
        return None
    if int(np.sum(base_decor != BG)) > 29 or int(np.sum(plug_decor != BG)) > 29:
        return None

    expected = _assemble_expected(base_decor, plug_decor)
    if expected.shape[0] < 5 or expected.shape[0] > 8 or expected.shape[1] < 3 or expected.shape[1] > 12:
        return None

    for sprite, role in ((base_decor, "base"), (plug_decor, "plug")):
        pts = {(int(r), int(c)) for r, c in np.argwhere(sprite != BG)}
        _shape, ctrl_pts = _sprite_ctrl_shape(pts, sprite)
        if role == "base" and any(int(r) == sprite.shape[0] - 1 for r, _c in ctrl_pts):
            return None
        if role == "plug" and any(int(r) == 0 for r, _c in ctrl_pts):
            return None

    bh, bw = map(int, base_decor.shape)
    ph, pw = map(int, plug_decor.shape)
    min_h = max(7, max(bh, ph) + 2)
    min_w = max(5, max(bw, pw) + 1)
    if min_h > 11 or min_w > 13:
        return None
    canvas_h = _rng.randint(min_h, 11)
    canvas_w = _rng.randint(min_w, 13)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    paste(canvas, base_decor, 0, 0, transparent=BG)
    try:
        pr, pc = random_free_location(canvas, plug_decor, transparent=BG, padding=1)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_decor, pr, pc, transparent=BG)

    lab, n = label_mask(canvas != BG, connectivity=8)
    if int(n) != 2:
        return None
    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        got = main(canvas)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")

# Variant of 20270e3b
# Dimension: D2 — 连接器方向 (match_cond)
# Change: 连接器为水平条，对接方向为垂直（plug在base上方对接）... → L形连接器角落对接
#
# Design analysis
# ===============
# variant_spec:
#   concrete_form:
#     - 仍然只有两块前景 sprite，颜色保持固定：bg=1, obj=4, ctrl=7。
#     - 原题的 connector 是一条水平 ctrl 横条；本变体改成 3 像素 L 形角连接器。
#     - 每个连接器都占据一个 2x2 框中的 3 个格子，也就是“缺一个角”的小 L。
#     - base 使用“缺左上角”的 L：{TR, BL, BR}；plug 使用互补的“缺右下角”的 L：{TL, TR, BL}。
#     - 对接时不再让 plug 整体移到 base 正上方一行，而是把两个 2x2 connector 框完全对齐；
#       两个 L 的并集恰好填满这个 2x2 框，形成一个小矩形。
#   behavior_trace:
#     - 原逻辑：
#       1. 找到两个完全同形的水平 ctrl 条。
#       2. 令 plug connector 的行 = base connector 行 - 1，列完全对齐。
#       3. 叠加后 seam 是一条水平接缝，ctrl 最后统一改成 obj。
#     - 变体逻辑：
#       1. 找到两个 3 像素 L 形 ctrl 角，且都落在 2x2 框中。
#       2. 识别各自缺失的角；两者必须互补，平移后 connector 框重合。
#       3. 叠加后 seam 不再是一条线，而是一个 2x2 小矩形。
#       4. 再把 ctrl 改回 obj，并紧裁剪。
#     - 小例子：
#       base connector 相对框 = {(0,1),(1,0),(1,1)}，plug connector 相对框 = {(0,0),(0,1),(1,0)}。
#       原题会尝试“上移一行”并要求形状相同，这里显然不成立；
#       新题则让两个 2x2 框重合，得到 {(0,0),(0,1),(1,0),(1,1)} 的实心矩形。
#   complexity_calibration:
#     - 为了让 L 形变化稳定可见，这里固定使用 3 像素 L，而不是引入 2/3 像素混合子类型。
#       否则 2 像素情况会退化成普通折线/骨牌，视觉上不够明确，也会引入识别歧义。
#     - 复杂度主要来自“识别 L 的朝向 + 理解两个不同朝向会拼成矩形”，
#       而不是来自额外颜色、额外对象或多重匹配规则。
#   implicit_assumption_review:
#     - 原 `norm(conn)` + `assert shpA == shpB)` 依赖“两个连接器归一化后完全相同”。
#       变体中这一前提失效；现在要允许不同朝向，只要求它们叠加后充满同一个 2x2 框。
#     - 原 `tr = (dr0 - 1) - pr0, tc = dc0 - pc0` 写死了“plug 在 base 上方一行”。
#       变体中改成 connector 框完全对齐：`tr = dr0 - pr0`, `tc = dc0 - pc0`。
#     - 原来的 overlap 检查默认 seam 是 ctrl 与 obj 的纵向接触带；
#       现在 overlap 只允许出现在两个 connector 自身的重叠像素上，因此需要按“shift 后的 ctrl 集合”精确校验。
#   placement_legality:
#     - 生成器把 base connector 固定在 sprite 右下角的 2x2 框，把 plug connector 固定在 sprite 左上角的 2x2 框。
#     - 这样在装配态中，plug 只需平移到 `(base_h - 2, base_w - 2)`，
#       并且除 connector 框外不会与 base 重叠，也不会跑到 base 的上方/左方。
#     - 输入画布上使用 `padding=1, connectivity=8` 随机放置 plug，保证两个 sprite 在 solver 的 8 连通分量下仍然分离。
#   visual_observability:
#     - 从几组 input/output 可以肉眼看出：橙色 ctrl 不再是一条横线，而是两个互补的 L 角；
#       输出里这两个角会拼成一个 2x2 小方块并被染回 obj 色。
#     - 规则只涉及一种 connector 机制，没有隐藏排序，也不依赖颜色数值大小。
#   cleanliness:
#     - 每个 sprite 仍然是单色主体 + 少量 ctrl 标记，不引入额外颜色层。
#     - 连接区域被限制在一个清晰的 2x2 小框内，避免大片覆盖导致难以分辨。
#   color_inference:
#     - 颜色逻辑与原题一致：只有 ctrl(7) 在输出中变回 obj(4)。
#     - 由于变化只发生在结构上，人类可以把注意力放在 connector 形状和拼接方式，而不是新的颜色映射。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7

CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNER_TO_CELLS = {
    "TL": {(0, 1), (1, 0), (1, 1)},
    "TR": {(0, 0), (1, 0), (1, 1)},
    "BL": {(0, 0), (0, 1), (1, 1)},
    "BR": {(0, 0), (0, 1), (1, 0)},
}
BODY_DIRS = {
    "TL": [(-1, 0), (0, -1)],
    "TR": [(-1, 0), (0, 1)],
    "BL": [(1, 0), (0, -1)],
    "BR": [(1, 0), (0, 1)],
}
FREE_DIRS = {
    "TL": [(1, 0), (0, 1)],
    "TR": [(1, 0), (0, -1)],
    "BL": [(-1, 0), (0, 1)],
    "BR": [(-1, 0), (0, -1)],
}
OPPOSITE_CORNER = {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _connector_info(
    g: np.ndarray, sprite: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], str, tuple[int, int]]:
    conn = {(r, c) for (r, c) in sprite if int(g[r, c]) == CTRL}
    assert len(conn) == 3
    r0, c0, r1, c1 = _bbox_of_pts(conn)
    assert r1 - r0 == 1 and c1 - c0 == 1
    rel = {(int(r - r0), int(c - c0)) for r, c in conn}
    corner = None
    for name, shape in CORNER_TO_CELLS.items():
        if rel == shape:
            corner = name
            break
    assert corner is not None
    return conn, corner, (int(r0), int(c0))


def _validate_connector_body(
    conn: set[tuple[int, int]], body: set[tuple[int, int]], missing_corner: str
) -> None:
    allowed_dirs = BODY_DIRS[missing_corner]
    forbidden_dirs = FREE_DIRS[missing_corner]
    touches_body = False
    for r, c in conn:
        for dr, dc in allowed_dirs:
            if (int(r + dr), int(c + dc)) in body:
                touches_body = True
        for dr, dc in forbidden_dirs:
            assert (int(r + dr), int(c + dc)) not in body
    assert touches_body


def _dock(base_grid: np.ndarray, plug_grid: np.ndarray) -> np.ndarray:
    base_h, base_w = base_grid.shape
    plug_h, plug_w = plug_grid.shape
    out = np.full((base_h + plug_h - 2, base_w + plug_w - 2), BG, dtype=int)
    paste(out, base_grid, 0, 0, transparent=BG)

    for r in range(plug_h):
        for c in range(plug_w):
            val = int(plug_grid[r, c])
            if val == BG:
                continue
            rr = base_h - 2 + r
            cc = base_w - 2 + c
            cur = int(out[rr, cc])
            if cur != BG:
                assert cur == CTRL and val == CTRL
            out[rr, cc] = val

    out[out == CTRL] = OBJ
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites via complementary L-shaped connectors, then tight-crop.

    The base sprite is the component anchored closest to the input origin.
    Its connector and the plug connector each occupy 3 cells in a shared 2x2 box
    after docking; together they fill the full 2x2 rectangle.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
    else:
        base, plug = B, A

    conn_base, base_corner, (br0, bc0) = _connector_info(g, base)
    conn_plug, plug_corner, (pr0, pc0) = _connector_info(g, plug)
    assert OPPOSITE_CORNER[base_corner] == plug_corner

    base_body = base - conn_base
    plug_body = plug - conn_plug
    _validate_connector_body(conn_base, base_body, base_corner)
    _validate_connector_body(conn_plug, plug_body, plug_corner)

    tr = int(br0 - pr0)
    tc = int(bc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}
    shifted_conn = {(int(r + tr), int(c + tc)) for r, c in conn_plug}

    conn_union = conn_base | shifted_conn
    ur0, uc0, ur1, uc1 = _bbox_of_pts(conn_union)
    assert ur1 - ur0 == 1 and uc1 - uc0 == 1
    assert len(conn_union) == 4

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_coords = np.array(list(base), dtype=int)
    assert rr0 == int(base_coords[:, 0].min())
    assert cc0 == int(base_coords[:, 1].min())

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[r, c])

    for r, c in plug_shift:
        src_r = int(r - tr)
        src_c = int(c - tc)
        val = int(g[src_r, src_c])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != BG:
            assert (r, c) in conn_base and (src_r, src_c) in conn_plug
            assert cur == CTRL and val == CTRL
        out[int(r - rr0), int(c - cc0)] = val

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    if nz.size == 0:
        return np.full((0, 0), BG, dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


import random as _rng


def _grow_shape_allowed(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> None:
    count = int(mask.sum())
    while count < target and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = int(r + dr)
            nc = int(c + dc)
            if (nr, nc) in allowed and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.append((nr, nc))
        count += 1


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


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    er, ec = map(int, end)
    path = [(r, c)]
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        elif r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        elif c > ec:
            options.append((r, c - 1))
        r, c = _rng.choice(options)
        path.append((int(r), int(c)))
    return path


def _build_base_sprite(height: int, width: int) -> np.ndarray:
    h = int(height)
    w = int(width)
    conn_cells = {(h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)}
    conn_square = {(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)}
    allowed = {(r, c) for r in range(h) for c in range(w)} - conn_square

    support = {(h - 3, w - 1), (h - 3, w - 2), (h - 2, w - 3), (h - 1, w - 3)}
    path = set(_monotone_path((h - 3, w - 2), (0, 0)))
    keep = {tuple(map(int, cell)) for cell in (support | path)}

    mask = np.zeros((h, w), dtype=bool)
    for r, c in keep:
        mask[r, c] = True

    max_body = min(26, len(allowed))
    target_body = _rng.randint(len(keep), max_body)
    _grow_shape_allowed(mask, target_body, list(keep), allowed)
    _thin(mask, keep=keep)

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    for r, c in conn_cells:
        grid[r, c] = CTRL

    assert is_contiguous(grid != BG, background=0, connectivity=8)
    assert int(np.sum(grid != BG)) <= 29
    return grid


def _build_plug_sprite(height: int, width: int) -> np.ndarray:
    h = int(height)
    w = int(width)
    conn_cells = {(0, 0), (0, 1), (1, 0)}
    conn_square = {(0, 0), (0, 1), (1, 0), (1, 1)}
    allowed = {(r, c) for r in range(h) for c in range(w)} - conn_square

    support = {(0, 2), (1, 2), (2, 1), (2, 0)}
    path = set(_monotone_path((1, 2), (h - 1, w - 1)))
    keep = {tuple(map(int, cell)) for cell in (support | path)}

    mask = np.zeros((h, w), dtype=bool)
    for r, c in keep:
        mask[r, c] = True

    max_body = min(26, len(allowed))
    target_body = _rng.randint(len(keep), max_body)
    _grow_shape_allowed(mask, target_body, list(keep), allowed)
    _thin(mask, keep=keep)

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    for r, c in conn_cells:
        grid[r, c] = CTRL

    assert is_contiguous(grid != BG, background=0, connectivity=8)
    assert int(np.sum(grid != BG)) <= 29
    return grid


def _gen_attempt() -> np.ndarray | None:
    for _ in range(20):
        base_h = _rng.randint(3, 6)
        base_w = _rng.randint(3, 7)
        plug_h = _rng.randint(3, 6)
        plug_w = _rng.randint(3, 7)
        if base_h + plug_h - 2 <= 10 and base_w + plug_w - 2 <= 12:
            break
    else:
        return None

    base_sprite = _build_base_sprite(base_h, base_w)
    plug_sprite = _build_plug_sprite(plug_h, plug_w)

    expected = _dock(base_sprite, plug_sprite)
    if expected.size == 0 or not np.any(expected != BG):
        return None

    min_canvas_h = max(7, max(base_h, plug_h) + 2)
    min_canvas_w = max(6, max(base_w, plug_w) + 2)
    if min_canvas_h > 11 or min_canvas_w > 13:
        return None
    canvas_h = _rng.randint(min_canvas_h, 11)
    canvas_w = _rng.randint(min_canvas_w, 13)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    try:
        pr, pc = random_free_location(
            canvas,
            plug_sprite,
            transparent=BG,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=BG)

    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")

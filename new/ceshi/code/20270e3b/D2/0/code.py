# Variant of 20270e3b
# Dimension: D2 — 连接器方向 (match_cond)
# Change: 连接器为水平条，对接方向为垂直（plug在base上方对接）... → 垂直连接器水平对接

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the D2 variant
==================================

a) 具体化与行为追踪
   - 保留颜色与总体叙事不变：`bg=1, obj=4, ctrl=7`，仍然只有两个 8 连通精灵。
   - 变化点只放在 connector 的朝向与对接方向：
     * 原题：每个 connector 是 1 行横条，plug 的 connector 行放到 base connector 行正上方 1 格。
     * 变体：每个 connector 是 1 列竖条，plug 的 connector 列放到 base connector 列左侧或右侧 1 格。
   - 具体像素关系（以 “plug 在 base 左侧对接” 为例）：
     * base connector 是竖条，connector 左边紧贴 base body，右边为空。
     * plug connector 也是同长度竖条，connector 右边紧贴 plug body，左边为空。
     * 平移后：plug connector 列 = base connector 列 - 1，同一批行对齐。
       于是发生两列宽的对接带：
       - 左列：plug 的 ctrl 覆盖 base 的 obj
       - 右列：plug 的 obj 覆盖 base 的 ctrl
     * 最后所有 ctrl 统一改回 obj，再做紧裁剪。
   - 小例子追踪：
     * 原逻辑平移公式：`tr=(dr0-1)-pr0, tc=dc0-pc0`，因为 connector 要上下错开 1 行。
     * 新逻辑如果 plug 在左：`tr=dr0-pr0, tc=(dc0-1)-pc0`。
     * 新逻辑如果 plug 在右：`tr=dr0-pr0, tc=(dc0+1)-pc0`。
     * 差异的本质是：旧题在“行”方向推理，新题在“列”方向推理，而且需要判断左右分支。

b) 复杂度校准
   - 若 connector 长度允许大量为 1，则“横条/竖条”肉眼不可区分，D2 会退化。
   - 因此 generator 只生成长度 2 或 3 的竖直 connector，让方向变化稳定可见。
   - 左接 / 右接两个分支都保留，且 connector 长度在同一任务族内重复出现，解题者能从少量样例看到“同长度竖条水平拼接”的共性。

c) 隐式假设审查（针对原 `main()` 的方向相关代码）
   - `norm(conn_pts)` 本身与方向无关，仍可复用。
   - 原 `for r,c in connD: assert (r+1,c) not in base_body` 假设“free side 在下方”，变体中不成立，需改为水平方向检查。
   - 原 `tr=(dr0-1)-pr0, tc=dc0-pc0` 写死了“plug connector 在 base 上方一行”，变体必须改为同排、左右偏 1 列。
   - 原 `assert rr0 == br0 and cc0 == bc0` 依赖 plug 永不伸到 base 左边；左接分支会打破这个前提，所以必须删除。
   - 原 overlap 断言 `cur==ctrl or val==ctrl` 仍然成立，因为合法重叠仍只发生在 ctrl 参与的对接带。

d) 放置合法性
   - 生成器先构造 docked 状态，再把两个 sprite 裁开分别随机放回输入。
   - base 固定贴在输入左上角 `(0,0)`，保持角色判定方式不变。
   - plug 在画布其他空位随机放置，输入中两精灵不重叠。
   - 在 assembled 检查中，plug 与 base 的重叠只允许出现在 “plug ctrl vs base obj” 或 “plug obj vs base ctrl” 的对接带。

e) 视觉可观察性
   - connector 竖条长度固定为 2-3，避免 1 像素 connector 的旋转歧义。
   - 左右两个分支都出现，但都遵守相同的“竖条水平对接”规则，不会引入难以归纳的多套机制。

f) 可视化整洁度
   - 保持原题的单一 body 色和单一 ctrl 色，不额外引入标记。
   - 对接带仍只有两列宽，不会出现大面积叠压。

g) 颜色变化可推断性
   - 颜色规则未变：输入中只有 `bg / obj / ctrl`，输出把所有 ctrl 还原成 obj。
   - 颜色变化与 connector 的位置绑定，结构线索充足。

h) 新 solver 假设与 generator 保证
   - solver 假设每个 sprite 恰有一个竖直 ctrl 条，两个 ctrl 条等长。
   - solver 假设 base connector 的 body 在其一侧、plug connector 的 body 在另一侧，且两者方向相反。
   - generator 通过候选筛选和组装验证保证这些条件成立。

i) 额外保留的非核心性质
   - 保留两精灵、8 连通、thin-walled 生长、tight crop、base 贴输入原点这些原题骨架。
   - 只把对接几何从“上下”换成“左右”。
"""


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assert_vertical_bar(shape: set[tuple[int, int]]) -> None:
    rows = sorted(int(r) for r, c in shape if int(c) == 0)
    assert len(rows) == len(shape) >= 1
    assert rows == list(range(rows[0], rows[-1] + 1))


def _connector_body_side(
    sprite: set[tuple[int, int]],
    conn_pts: set[tuple[int, int]],
) -> int:
    body = sprite - conn_pts
    valid = []
    for side in (-1, 1):
        if all((int(r), int(c + side)) in body for r, c in conn_pts) and all(
            (int(r), int(c - side)) not in body for r, c in conn_pts
        ):
            valid.append(int(side))
    assert len(valid) == 1
    return int(valid[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching vertical connectors, then tight-crop.

    Variant rule:
    - The connector on each sprite is a vertical ctrl-colored bar.
    - The base is the sprite whose bbox top-left is closest to `(0,0)` and anchored there.
    - Infer whether docking is from the left or from the right from which horizontal side
      of each connector is occupied by body pixels.
    - Translate the plug so its connector sits exactly one column to the left or right of
      the base connector, sharing the same rows.
    - Stamp, recolor ctrl -> obj, and tight-crop.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        comps.append({(int(r), int(c)) for r, c in np.argwhere(lab == int(k))})
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB

    shpA, (ar0, ac0) = _norm_conn(connA)
    shpB, (br0, bc0) = _norm_conn(connB)
    _assert_vertical_bar(shpA)
    _assert_vertical_bar(shpB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_side = _connector_body_side(base, connD)
    plug_side = _connector_body_side(plug, connP)
    assert plug_side == -base_side

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)

    tr = int(dr0 - pr0)
    tc = int((dc0 + base_side) - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (output-first: build assembled state, then separate)
# ---------------------------------------------------------------------------
import random as _rng


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


def _crop_nonbg(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = 1, 4, 7

    # Step 1: build base sprite body.
    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    # Step 2: choose a vertical connector on the base.
    candidates: list[tuple[int, int, int, int]] = []
    for side in (-1, 1):
        for conn_len in range(2, min(3, base_h) + 1):
            for conn_col in range(base_w):
                body_col = int(conn_col + side)
                free_col = int(conn_col - side)
                if not (0 <= body_col < base_w):
                    continue
                for conn_r0 in range(base_h - conn_len + 1):
                    rows = list(range(conn_r0, conn_r0 + conn_len))
                    if not all(base[r, conn_col] for r in rows):
                        continue
                    if not all(base[r, body_col] for r in rows):
                        continue
                    if any(0 <= free_col < base_w and base[r, free_col] for r in rows):
                        continue
                    candidates.append((side, conn_len, conn_r0, conn_col))
    if not candidates:
        return None
    side, conn_len, conn_r0, conn_col = _rng.choice(candidates)
    conn_rows = list(range(conn_r0, conn_r0 + conn_len))

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for r in conn_rows:
        base_grid[r, conn_col] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None
    for r in conn_rows:
        if base_grid[r, conn_col + side] != obj:
            return None
        free_col = int(conn_col - side)
        if 0 <= free_col < base_w and base_grid[r, free_col] != bg:
            return None

    # Step 3: grow the plug so that it docks from the left or right.
    plug_h = _rng.randint(conn_len, 7)
    plug_w = _rng.randint(2, 10)
    plug_body_side = int(-side)
    if plug_body_side == 1:
        plug_conn_col = _rng.randint(0, plug_w - 2)
    else:
        plug_conn_col = _rng.randint(1, plug_w - 1)

    max_conn_r0 = min(plug_h - conn_len, conn_r0)
    if max_conn_r0 < 0:
        return None
    plug_conn_r0 = _rng.randint(0, max_conn_r0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    connector_pts = set()
    keep_plug = set()
    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        connector_pts.add((int(r), int(plug_conn_col)))
        keep_plug.add((int(r), int(plug_conn_col)))
        keep_plug.add((int(r), int(plug_conn_col + plug_body_side)))
        plug[r, plug_conn_col] = True
        plug[r, plug_conn_col + plug_body_side] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(max(5, int(plug.sum())), plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    _thin(plug, keep=keep_plug)
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    for r in range(plug_conn_r0, plug_conn_r0 + conn_len):
        if not plug[r, plug_conn_col + plug_body_side]:
            return None
        free_col = int(plug_conn_col - plug_body_side)
        if 0 <= free_col < plug_w and plug[r, free_col]:
            return None

    dock_r = int(conn_r0 - plug_conn_r0)
    dock_c = int(conn_col + side - plug_conn_col)

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(dock_r + r)
            dc = int(dock_c + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == obj:
                if (int(r), int(c)) in connector_pts:
                    continue
                return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for r, c in connector_pts:
        plug_grid[r, c] = ctrl
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_nonbg(base_grid, bg)
    plug_sprite = _crop_nonbg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    # Step 4: place on canvas.
    min_canvas_h = max(7, max(bh, ph) + 2)
    min_canvas_w = max(5, max(bw, pw) + 2)
    if min_canvas_h > 11 or min_canvas_w > 13:
        return None
    canvas_h = _rng.randint(min_canvas_h, 11)
    canvas_w = _rng.randint(min_canvas_w, 13)

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
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")

# Combo variant of 20270e3b (index 6)
# combo_spec: D4/2+D7/0
# Name: 重力下落固定画布

from __future__ import annotations

import random as _rng
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

"""
Design analysis for combo D4/2 + D7/0
=====================================

a) 具体化与行为追踪
------------------
- 保留原题的对象语义：bg=1, obj=4, ctrl=7；恰好两个 8-connected sprite；每个 sprite
  恰好一条水平 ctrl 连接器，且两条连接器长度相同。
- D4/2 改动：plug 的水平位置仍由连接器对齐决定，但垂直位置不再由
  `plug_conn_row = base_conn_row - 1` 决定，而是像俄罗斯方块那样从上方直落，停在
  “再往下走一格就会与 base 重叠”的位置。
- D7/0 在单变体里是“把紧裁剪结果贴到输出左上角”；但本融合规格明确写的是
  “固定画布，合并后精灵在原位渲染”。因此这里采用更强的组合语义：
  输出尺寸等于输入尺寸，base 保持原绝对坐标，gravity landing 后的 plug 也按绝对坐标
  渲染，其余位置填 bg。
- 小例子：
  * 原题：base connector 在全局 row 4, col 2..3；plug connector 在自身局部 row 0。
    则直接令 plug 落到 `tr = (4 - 1) - 0 = 3`，连接器 seam 立刻相邻。
  * 融合版：列仍对齐到 col 2..3，但若 base 在这些列更高处已有“屋顶”像素，plug 会更早
    停住，例如 `tr = 1`。此时 plug connector 可能离 base connector 还有 2 行空气。
  * 输出不再把这个合并体紧裁剪成小图，而是放在与输入同样大的画布中，右侧或下侧保留 bg。

b) 复杂度校准
--------------
- connector 长度仍限制在 1..3，角色识别不会变得含糊。
- 变化只引入两个离散信号：
  * 垂直落点由 gravity 决定，而不是由 connector row 直接决定。
  * 输出尺寸固定，且必须保留可见的右侧或下侧背景余量。
- generator 强制 `drop_delta = direct_tr - gravity_tr >= 1`，否则重力版会退化成原题。
- generator 还强制 `canvas_h > merged_h or canvas_w > merged_w`，否则固定画布不可见。

c) 对原 main() 隐式假设的逐项审查
----------------------------------
- 原 `tr = (dr0 - 1) - pr0` 失效：connector 只负责水平对齐，不能再直接给出最终垂直落点。
- 原 overlap 逻辑允许 ctrl-vs-obj 在 seam 处重叠；gravity 版最终位置必须完全不重叠，
  否则说明 plug 穿进了 base。
- 原 solver 先做最小 bbox 渲染，再把 tight crop 当最终答案；本融合版只能把 tight union
  当作中间量，最终答案必须是原始输入尺寸。
- 原 D7 单变体把 merged sprite 贴到输出左上角；融合规格要求“原位渲染”，所以绝对坐标
  不能丢失。这里保留 base 的输入坐标并直接在固定画布上渲染。
- 新 solver 假设：
  * 对齐 connector 列后，一定存在有限的 gravity 落点；
  * 最终 gravity 位置不越出画布；
  * 最终 gravity 位置与 base 不重叠，且至少存在一处“plug 下方一格是 base”的支撑。
  generator 会显式构造并验证这些条件。

d) 放置合法性
--------------
- 输入仍把 base 放在 (0,0)，plug 用 `random_free_location(..., padding=1)` 放到别处，
  保证输入里始终是两个独立 8-connected 组件。
- assembled 态先在 base 局部坐标中计算 gravity 落点；只有当 `tr >= 0` 且 `tc >= 0`
  时才接受，这样 merged 结果不会伸到 base 上方或左侧，固定画布可直接原位承载。
- 选 canvas 尺寸时同时满足：
  * base / plug / merged 都能放下；
  * 至少一个维度比 merged 更大，使固定画布规则肉眼可见。

e) 视觉可观察性
----------------
- 观察者会看到：输入中 plug 散落在别处；输出中它不是按 connector row 直接贴到 seam，
  而是停在 base 的“顶面”。
- 同时输出保留与输入同样大的蓝色背景区，而不是变成紧裁剪小块。
- 这两个信号在 2-4 组样例中都稳定出现，且只依赖可见的几何与背景，不依赖颜色编号大小。

f) 可视化整洁度
----------------
- 仍只有 3 种颜色，ctrl 在输出统一重涂为 obj。
- 最终合并态禁止任何 obj-obj 或 ctrl-obj 重叠，只有边接触，不会产生难读的覆盖层。
- 画布余量只留在右侧或下侧，没有额外标记或杂讯。

g) 变化可推断性
----------------
- 颜色规则没有扩张，仍然只是“所有 connector 失去接口语义后变成 body 色 4”。
- 真正需要推断的是两个结构规则：gravity landing 和 fixed-size output。
- 这两条都通过 generator 的硬约束反复出现，不会被样例偶然性掩盖。

h) 新 solver 假设与 generator 保证
-----------------------------------
- solver 假设输入里恰有两个前景组件，且两者 connector 归一化形状相同。generator 显式构造。
- solver 假设 base connector 自由侧朝下，plug connector 自由侧朝上。generator 显式构造。
- solver 假设 gravity landing 后 plug 完全在画布内且不与 base 重叠。generator 先算落点再筛样本。
- solver 假设固定画布输出与 tight-crop 输出不同。generator 用 `drop_delta` 和 `canvas_slack`
  两个指标联合过滤。

i) 融合行为量化指标
--------------------
- `drop_delta = direct_tr - gravity_tr`：要求 >= 1，证明垂直位置确实不是原题那种直接 seam。
- `canvas_slack = (H - merged_h, W - merged_w)`：要求至少一个分量 > 0，证明输出确实保留了
  固定画布的背景区域。
- 后续测试脚本会统计这两个指标的分布，防止生成器退化回“原题落点”或“看起来像 tight crop”。
"""


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_pts(grid: np.ndarray, sprite: set[tuple[int, int]], ctrl: int = CTRL) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for (r, c) in sprite if int(grid[int(r), int(c)]) == int(ctrl)}


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _extract_roles(
    grid: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    lab, n = label_mask(g != int(BG), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    a, b = comps

    conn_a = _conn_pts(g, a, CTRL)
    conn_b = _conn_pts(g, b, CTRL)
    assert conn_a and conn_b

    shp_a, _ = _norm_conn(conn_a)
    shp_b, _ = _norm_conn(conn_b)
    assert shp_a == shp_b

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(a)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(b)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, conn_base = a, conn_a
        plug, conn_plug = b, conn_b
    else:
        base, conn_base = b, conn_b
        plug, conn_plug = a, conn_a

    base_body = base - conn_base
    plug_body = plug - conn_plug
    for r, c in conn_base:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in conn_plug:
        assert (int(r - 1), int(c)) not in plug_body
    return base, conn_base, plug, conn_plug


def _gravity_shift(
    base: set[tuple[int, int]],
    conn_base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    conn_plug: set[tuple[int, int]],
) -> tuple[int, int, set[tuple[int, int]]]:
    """
    Align connector columns, then drop the plug until the next downward step would overlap.
    """
    _sb, (_dr0, dc0) = _norm_conn(conn_base)
    _sp, (_pr0, pc0) = _norm_conn(conn_plug)
    tc = int(dc0 - pc0)

    base_rows_by_col: dict[int, list[int]] = {}
    for r, c in base:
        base_rows_by_col.setdefault(int(c), []).append(int(r))

    tr_candidates: list[int] = []
    for rp, cp in plug:
        target_col = int(cp + tc)
        for rb in base_rows_by_col.get(target_col, []):
            tr_candidates.append(int(rb - rp - 1))
    assert tr_candidates, "aligned connector columns must yield a finite landing"

    tr = int(min(tr_candidates))
    shifted = {(int(r + tr), int(c + tc)) for r, c in plug}
    assert base.isdisjoint(shifted), "gravity landing must stop before overlap"
    assert any((int(r + 1), int(c)) in base for r, c in shifted), "gravity landing needs support"
    return int(tr), int(tc), shifted


def _render_fixed_canvas(
    grid: np.ndarray,
    base: set[tuple[int, int]],
    conn_base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    conn_plug: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    out = np.full((h, w), int(BG), dtype=int)

    tr, tc, plug_shift = _gravity_shift(base, conn_base, plug, conn_plug)
    _sb, (dr0, _dc0) = _norm_conn(conn_base)
    _sp, (pr0, _pc0) = _norm_conn(conn_plug)
    direct_tr = int((dr0 - 1) - pr0)

    all_pts = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(all_pts)
    assert 0 <= rr0 and 0 <= cc0 and rr1 < h and cc1 < w

    for r, c in base:
        out[int(r), int(c)] = int(g[int(r), int(c)])

    for r, c in plug:
        rr = int(r + tr)
        cc = int(c + tc)
        assert 0 <= rr < h and 0 <= cc < w
        assert out[int(rr), int(cc)] == int(BG)
        out[int(rr), int(cc)] = int(g[int(r), int(c)])

    out[out == int(CTRL)] = int(OBJ)
    return out, {
        "gravity_tr": int(tr),
        "gravity_tc": int(tc),
        "direct_tr": int(direct_tr),
        "drop_delta": int(direct_tr - tr),
        "merged_h": int(rr1 - rr0 + 1),
        "merged_w": int(cc1 - cc0 + 1),
        "merged_r0": int(rr0),
        "merged_c0": int(cc0),
    }


def _render_gravity_from_cropped_sprites(
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Render the assembled gravity state on a fixed canvas, assuming the base sprite sits at (0,0).
    """
    canvas_h, canvas_w = map(int, canvas_shape)
    base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(BG))}
    plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(BG))}
    conn_base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite == int(CTRL))}
    conn_plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite == int(CTRL))}
    assert conn_base and conn_plug

    tr, tc, plug_shift = _gravity_shift(base, conn_base, plug, conn_plug)
    _sb, (dr0, _dc0) = _norm_conn(conn_base)
    _sp, (pr0, _pc0) = _norm_conn(conn_plug)
    direct_tr = int((dr0 - 1) - pr0)

    all_pts = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(all_pts)
    assert 0 <= rr0 and 0 <= cc0
    assert rr1 < canvas_h and cc1 < canvas_w

    out = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(out, base_sprite, 0, 0, transparent=int(BG))
    for r, c in plug:
        rr = int(r + tr)
        cc = int(c + tc)
        assert out[int(rr), int(cc)] == int(BG)
        out[int(rr), int(cc)] = int(plug_sprite[int(r), int(c)])
    out[out == int(CTRL)] = int(OBJ)

    merged_mask = out != int(BG)
    assert is_contiguous(merged_mask, background=0, connectivity=8)
    return out, {
        "gravity_tr": int(tr),
        "gravity_tc": int(tc),
        "direct_tr": int(direct_tr),
        "drop_delta": int(direct_tr - tr),
        "merged_h": int(rr1 - rr0 + 1),
        "merged_w": int(cc1 - cc0 + 1),
        "merged_r0": int(rr0),
        "merged_c0": int(cc0),
    }


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a sample actually exercises the combo:
    - drop_delta > 0: gravity matters
    - canvas slack > 0 on at least one axis: fixed-canvas output matters
    """
    g = np.array(grid, dtype=int)
    h, w = map(int, g.shape)
    base, conn_base, plug, conn_plug = _extract_roles(g)
    _out, meta = _render_fixed_canvas(g, base, conn_base, plug, conn_plug)
    meta["canvas_h"] = int(h)
    meta["canvas_w"] = int(w)
    meta["slack_h"] = int(h - meta["merged_h"])
    meta["slack_w"] = int(w - meta["merged_w"])
    return meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the connector columns, drop the plug onto the base by gravity, recolor connectors
    to the body color, and keep the full input canvas size with the merged result rendered
    in place.
    """
    g = np.array(grid, dtype=int)
    base, conn_base, plug, conn_plug = _extract_roles(g)
    out, _meta = _render_fixed_canvas(g, base, conn_base, plug, conn_plug)
    return out


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[int(r + dr), int(c + dc)]
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
    h, w = mask.shape
    keep = keep or set()
    pixels = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _make_base_sprite() -> np.ndarray | None:
    conn_len = _rng.randint(1, 3)
    conn_row = _rng.randint(4, 6)
    roof_row = 2
    c0 = _rng.randint(1, 3)
    extra_right = _rng.randint(0, 2)
    end_col = int(c0 + conn_len - 1 + extra_right)
    base_h = int(conn_row + 1)
    base_w = int(end_col + 1)
    if base_w > 8:
        return None

    path_col = int(c0 - 1)
    base_grid = np.full((base_h, base_w), int(BG), dtype=int)

    base_grid[0, 0 : path_col + 1] = int(OBJ)
    base_grid[0 : roof_row + 1, path_col] = int(OBJ)
    base_grid[roof_row, path_col : end_col + 1] = int(OBJ)
    base_grid[roof_row : conn_row, end_col] = int(OBJ)
    base_grid[conn_row - 1, c0 : end_col + 1] = int(OBJ)
    base_grid[conn_row, c0 : c0 + conn_len] = int(CTRL)

    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(base_grid != int(BG))) < 5 or int(np.sum(base_grid != int(BG))) > 29:
        return None
    return base_grid


def _make_plug_sprite(conn_len: int) -> np.ndarray | None:
    plug_w = _rng.randint(max(4, conn_len), 7)
    if plug_w > 7:
        return None

    plug_grid = np.full((2, plug_w), int(BG), dtype=int)
    plug_grid[0, 0:conn_len] = int(CTRL)
    plug_grid[1, :] = int(OBJ)

    if not is_contiguous(plug_grid != int(BG), background=0, connectivity=8):
        return None
    if int(np.sum(plug_grid != int(BG))) < 5 or int(np.sum(plug_grid != int(BG))) > 29:
        return None
    return plug_grid


def _gen_attempt() -> np.ndarray | None:
    base_sprite = _make_base_sprite()
    if base_sprite is None:
        return None
    conn_base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite == int(CTRL))}
    conn_len = len(conn_base)

    plug_sprite = _make_plug_sprite(conn_len)
    if plug_sprite is None:
        return None

    base = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(BG))}
    plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(BG))}
    conn_plug = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite == int(CTRL))}
    tr, tc, _shifted = _gravity_shift(base, conn_base, plug, conn_plug)
    _sb, (dr0, _dc0) = _norm_conn(conn_base)
    _sp, (pr0, _pc0) = _norm_conn(conn_plug)
    direct_tr = int((dr0 - 1) - pr0)
    drop_delta = int(direct_tr - tr)

    if int(tr) < 0 or int(tc) < 0:
        return None
    if int(drop_delta) < 1:
        return None

    base_h, base_w = map(int, base_sprite.shape)
    plug_h, plug_w = map(int, plug_sprite.shape)

    merged_h = max(int(max(r for r, _c in base)), int(max(r for r, _c in _shifted))) + 1
    merged_w = max(int(max(c for _r, c in base)), int(max(c for _r, c in _shifted))) + 1

    h_min = max(7, base_h, plug_h, merged_h)
    w_min = max(3, base_w, plug_w, merged_w)
    size_choices = [
        (int(h), int(w))
        for h in range(h_min, 12)
        for w in range(w_min, 14)
        if h > merged_h or w > merged_w
    ]
    if not size_choices:
        return None
    canvas_h, canvas_w = _rng.choice(size_choices)

    try:
        expected, meta = _render_gravity_from_cropped_sprites(
            base_sprite, plug_sprite, (canvas_h, canvas_w)
        )
    except AssertionError:
        return None

    if int(meta["drop_delta"]) < 1:
        return None
    if int(meta["merged_r0"]) != 0 or int(meta["merged_c0"]) != 0:
        return None
    if not (int(canvas_h) > int(meta["merged_h"]) or int(canvas_w) > int(meta["merged_w"])):
        return None

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=int(BG), padding=1)
    except ValueError:
        return None
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=int(BG))

    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
        metrics = _fusion_metrics(canvas)
    except Exception:
        return None

    if not np.array_equal(result, expected):
        return None
    if int(metrics["drop_delta"]) < 1:
        return None
    if not (int(metrics["slack_h"]) > 0 or int(metrics["slack_w"]) > 0):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")

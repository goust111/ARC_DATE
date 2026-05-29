# Combo variant of 20270e3b (index 28)
# combo_spec: D1/3+D3/2+D6/0
# Name: 边缘匹配干扰过滤面积判定

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# 设计分析（先分析，再编码）
# ---------------------------------------------------------------------------
# a) 具体化与行为追踪
# - 原题的关键证据是每个 sprite 自带一条 ctrl 水平 connector；solver 先找两条
#   connector，再按 connector 的归一化形状匹配，并把 plug 平移到 base connector
#   的正上方一行。base 还由“谁贴着输入原点”决定。
# - 本融合版同时改三个环节：
#   1) D1/3：完全取消 connector 推理。两个主精灵只保留 obj=4，匹配依据改成
#      “base 的 bbox 最底行前景列集合 == plug 的 bbox 最顶行前景列集合”。
#      也就是看边缘轮廓，而不是看特殊颜色条。
#   2) D3/2：输入里额外加入 1-2 个小型干扰块，颜色来自非主色，面积 1-3。
#      solver 必须先过滤这些小块，只保留两个主精灵。
#   3) D6/0：base 不再由位置决定，而是由面积严格更大的主精灵决定。
# - 一个具体小例子：
#   原题：A 在原点，B 在右下；A/B 都有长 2 的 ctrl 条，于是“看颜色条 + 原点判 base”，
#   再把 B 放到 A 上方。
#   融合后：输入中有四个组件，两个大黄块 + 两个小彩色噪声。较大的黄块底边列模式是
#   {0,2,3}，较小黄块顶边列模式也是 {0,2,3}，于是先忽略两个小噪声，再数像素认定较大
#   黄块是 base，最后把较小黄块平移到 base 下方一行做瓷砖式拼接。
#
# b) 复杂度校准
# - 如果 seam pattern 总是连续实心横条，就会视觉上退化成“另一个 connector”。
#   因此 generator 强制边缘列模式至少有一个 gap，例如 {0,2} 或 {0,2,3}。
# - seam 列数只允许 2-4，sprite 宽度只允许 4-9。这样既能稳定肉眼观察到“边缘模式
#   匹配”，又不会自由度过大导致样例里每次都像完全不同的形状。
# - 面积判 base 需要可见但不含糊，generator 强制两个主精灵面积差至少为 3。
# - 干扰块固定为 1-2 个、每个 1-3 像素，确保“需要过滤”是稳定存在的，但又不会和主精灵
#   混成同一视觉层次。
#
# c) 隐式假设审查（逐段对应原 main 的关键前提）
# - 原 `conn(sprite)` / `norm(conn_pts)` / `assert shpA == shpB)` 依赖“存在 ctrl 条且它
#   就是匹配依据”。融合后这个前提彻底失效，必须删除。
# - 原 `assert len(best_comps) == 2` 依赖“没有额外前景组件”。融合后输入里必有 1-2 个
#   干扰块，所以 solver 必须先按面积筛出两个主组件。
# - 原 `r0+c0==0` 判 base 的逻辑失效，因为 base 改为“面积较大者”。如果继续用原点，
#   很多样例会得到相反角色。
# - 原 connector 上下自由侧检查失效，因为新规则没有 connector，也不存在共享一行重叠区。
# - 原 `tr = (dr0 - 1) - pr0` 是“把 plug 放到 base 上方并按 connector 行对齐”。
#   融合后应改成 `tr = base_bbox_bottom + 1 - plug_bbox_top`，即把 plug 放到 base 下方。
# - 原 overlap 断言允许 ctrl/obj 在接缝处重叠；融合后两个精灵只做边对边相邻，不允许重叠。
# - 多项改动叠加后的新假设是：只有两个主组件面积 >=5，且它们的 edge signature 唯一匹配。
#   generator 会显式保证这些条件成立。
#
# d) 放置合法性
# - 生成器采用 output-first：先构造“已拼好”的 base/plug 轮廓，再拆成两个 sprite，
#   最后散放到输入画布。
# - 两个主精灵用 `padding=1` 散放，干扰块也用 `padding=1` 散放，确保所有组件在 8 连通下
#   都彼此分离，不会重叠或对角粘连。
# - 生成器让主精灵尺寸和画布尺寸始终满足 `7..11 x 3..13` 的原题物理尺度，不越界。
#
# e) 视觉可观察性
# - 人类在 2-4 个样例里能稳定看到三件事：
#   1) 输入里有小彩色噪声，但输出里噪声消失；
#   2) 输入中没有可用的特殊 connector 色条，接缝来自上下边缘列模式的对应；
#   3) 贴着原点的块并不总是 base，较大的块才保持不动。
# - 为了让 D6 真的可观察，generator 大多数样例会把较小的 plug 放在原点，而较大的 base
#   放在别处；这样“看原点”会明显犯错。
#
# f) 可视化整洁度
# - 主规则只使用 `bg=1` 和 `obj=4`，干扰块每块单色且很小，避免多层叠加。
# - 输出中两个主精灵只在相邻两行形成 seam，不共享像素，轮廓关系清楚。
#
# g) 变化的可推断性
# - 没有新的颜色映射任务。颜色只承担“主精灵 vs 小噪声”的分层作用，而不是编码隐藏语义。
# - 主精灵始终同色，变化都来自面积、边缘列模式、以及小干扰块是否应被忽略，均可由结构
#   直接观察推断。
#
# 新 solver 假设与 generator 保证的一一对应
# - solver 假设：恰有两个面积 >=5 的前景组件。generator 保证：干扰块面积 <=3。
# - solver 假设：两个主组件面积严格不同。generator 保证：面积差 >=3。
# - solver 假设：较大组件的底边列模式与较小组件的顶边列模式相同且带 gap。
#   generator 先构造该 seam，再做 oracle 校验。
# - solver 假设：拼接后 plug 不会伸到 base 上方或左方。generator 通过统一的 bbox 左对齐
#   seam 参考系来保证这一点。


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    lab, n = label_mask(mask, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    return comps


def _edge_cols(pts: set[tuple[int, int]], which: str) -> tuple[int, ...]:
    r0, c0, r1, _c1 = _bbox_of_pts(pts)
    row = int(r0 if which == "top" else r1)
    cols = sorted(int(c - c0) for r, c in pts if int(r) == row)
    assert cols
    return tuple(cols)


def _has_gap(cols: tuple[int, ...]) -> bool:
    return any(int(b - a) > 1 for a, b in zip(cols, cols[1:]))


def _crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _render_docked(base: set[tuple[int, int]], plug_shift: set[tuple[int, int]], bg: int, obj: int) -> np.ndarray:
    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in pts_all:
        out[int(r - rr0), int(c - cc0)] = int(obj)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Ignore 1-2 tiny distractor blobs.
    2. Among the two main sprites, the larger-area one is base.
    3. Match base bottom-edge columns to plug top-edge columns.
    4. Move plug to sit exactly one row below base, left-bbox aligned by that pattern.
    5. Tight-crop the merged monochrome result.
    """
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    comps = _component_sets(g != int(bg))
    assert len(comps) >= 3

    mains = [pts for pts in comps if len(pts) >= 5]
    noise = [pts for pts in comps if len(pts) < 5]
    assert len(mains) == 2
    assert 1 <= len(noise) <= 2
    assert all(len(pts) <= 3 for pts in noise)

    A, B = mains
    assert len(A) != len(B)
    if len(A) > len(B):
        base, plug = A, B
    else:
        base, plug = B, A

    base_sig = _edge_cols(base, "bottom")
    plug_sig = _edge_cols(plug, "top")
    assert base_sig == plug_sig
    assert _has_gap(base_sig)

    br0, bc0, br1, _bc1 = _bbox_of_pts(base)
    pr0, pc0, _pr1, _pc1 = _bbox_of_pts(plug)
    tr = int(br1 + 1 - pr0)
    tc = int(bc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    assert base.isdisjoint(plug_shift)
    rr0, cc0, _rr1, _cc1 = _bbox_of_pts(base | plug_shift)
    assert rr0 == br0 and cc0 == bc0

    out = _render_docked(base, plug_shift, bg, obj)
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


# ---------------------------------------------------------------------------
# Generator (output-first: build a matched edge seam, then scatter + distract)
# ---------------------------------------------------------------------------


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
        nbrs: list[tuple[int, int]] = []
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
        rr, cc = int(r), int(c)
        if (rr, cc) in keep:
            continue
        adj_bg = any(
            not (0 <= rr + dr < h and 0 <= cc + dc < w) or not mask[rr + dr, cc + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[rr, cc] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[rr, cc] = True


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
        moves: list[tuple[int, int]] = []
        if r != gr:
            moves.append((int(r + (1 if gr > r else -1)), c))
        if c != gc:
            moves.append((r, int(c + (1 if gc > c else -1))))
        if not moves:
            break
        nr, nc = moves[0] if len(moves) == 1 or _rng.random() < 0.6 else moves[1]
        r, c = int(nr), int(nc)
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
    lower = max(6, int(mask.sum()))
    upper = min(26, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    forbidden = {(int(height) - 1, c) for c in range(int(width)) if c not in seam_cols}
    target = _rng.randint(lower, upper)
    _grow_shape(mask, target, list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    bottom_cols = tuple(sorted(int(c) for c in np.where(mask[-1])[0]))
    if bottom_cols != seam_cols:
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
    upper = min(22, int(height * width) - max(0, int(width) - len(seam_cols)))
    if upper < lower:
        return None
    forbidden = {(0, c) for c in range(int(width)) if c not in seam_cols}
    target = _rng.randint(lower, upper)
    _grow_shape(mask, target, list(zip(*np.where(mask))), forbidden=forbidden)
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        return None
    top_cols = tuple(sorted(int(c) for c in np.where(mask[0])[0]))
    if top_cols != seam_cols:
        return None
    return mask


def _make_noise_patch(color: int, bg: int) -> np.ndarray:
    options = [
        np.array([[color]], dtype=int),
        np.array([[color, color]], dtype=int),
        np.array([[color], [color]], dtype=int),
        np.array([[color, color], [color, bg]], dtype=int),
        np.array([[color, bg], [color, color]], dtype=int),
    ]
    return np.array(_rng.choice(options), dtype=int)


def _build_expected_output(base_sprite: np.ndarray, plug_sprite: np.ndarray, bg: int, obj: int) -> np.ndarray:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    out = np.full((int(bh + ph), int(max(bw, pw))), int(bg), dtype=int)
    paste(out, base_sprite, 0, 0, transparent=bg)
    paste(out, plug_sprite, bh, 0, transparent=bg)
    out[out != int(bg)] = int(obj)
    cropped = _crop_non_bg(out, bg)
    assert cropped is not None
    return cropped


def _gen_attempt(debug: bool = False) -> np.ndarray | tuple[None, str] | None:
    bg, obj = 1, 4
    noise_palette = [2, 3, 5, 6, 8, 9]

    def _fail(reason: str) -> np.ndarray | tuple[None, str] | None:
        if debug:
            return None, reason
        return None

    width = _rng.randint(4, 9)
    seam_cols = _choose_seam_cols(width)
    if seam_cols is None:
        return _fail("seam_cols")

    base_h = _rng.randint(3, 5)
    plug_h_max = min(5, 8 - int(base_h))
    if plug_h_max < 3:
        return _fail("plug_h_range")
    plug_h = _rng.randint(3, plug_h_max)

    base_mask = _make_base_mask(base_h, width, seam_cols)
    plug_mask = _make_plug_mask(plug_h, width, seam_cols)
    if base_mask is None or plug_mask is None:
        return _fail("mask_build")

    base_grid = np.full(base_mask.shape, int(bg), dtype=int)
    plug_grid = np.full(plug_mask.shape, int(bg), dtype=int)
    base_grid[base_mask] = int(obj)
    plug_grid[plug_mask] = int(obj)

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return _fail("crop")

    base_count = int(np.sum(base_sprite != int(bg)))
    plug_count = int(np.sum(plug_sprite != int(bg)))
    if base_count < 6 or plug_count < 5:
        return _fail("small_main")
    if base_count - plug_count < 3:
        return _fail("area_gap")

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))}
    base_sig = _edge_cols(base_pts, "bottom")
    plug_sig = _edge_cols(plug_pts, "top")
    if base_sig != plug_sig or not _has_gap(base_sig):
        return _fail("bad_signature")

    expected = _build_expected_output(base_sprite, plug_sprite, bg, obj)
    eh, ew = map(int, expected.shape)
    if not (5 <= eh <= 8 and 3 <= ew <= 12):
        return _fail("expected_shape")

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 3), 11)
    canvas_w = _rng.randint(max(6, max(bw, pw) + 2), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return _fail("canvas_too_small")

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    origin_holds_base = _rng.random() < 0.25
    if origin_holds_base:
        paste(canvas, base_sprite, 0, 0, transparent=bg)
        try:
            pr, pc = random_free_location(
                canvas, plug_sprite, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            return _fail("plug_place")
        paste(canvas, plug_sprite, pr, pc, transparent=bg)
    else:
        paste(canvas, plug_sprite, 0, 0, transparent=bg)
        try:
            br, bc = random_free_location(
                canvas, base_sprite, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            return _fail("base_place")
        paste(canvas, base_sprite, br, bc, transparent=bg)

    n_noise = _rng.randint(1, 2)
    for color in _rng.sample(noise_palette, n_noise):
        patch = _make_noise_patch(int(color), bg)
        try:
            nr, nc = random_free_location(
                canvas, patch, transparent=bg, padding=1, padding_connectivity=8
            )
        except ValueError:
            return _fail("noise_place")
        paste(canvas, patch, nr, nc, transparent=bg)

    comps = _component_sets(canvas != int(bg))
    if len(comps) != 2 + n_noise:
        return _fail("component_count")
    if sum(1 for pts in comps if len(pts) >= 5) != 2:
        return _fail("main_filter")
    if sum(1 for pts in comps if len(pts) <= 3) != n_noise:
        return _fail("noise_filter")

    if np.sum(canvas == int(bg)) <= canvas.size // 3:
        return _fail("too_dense")

    try:
        result = main(canvas)
    except Exception:
        return _fail("oracle_exception")
    if not np.array_equal(result, expected):
        return _fail("oracle_mismatch")
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 600 attempts")

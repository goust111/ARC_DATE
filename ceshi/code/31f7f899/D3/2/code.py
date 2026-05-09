# Variant of 31f7f899
# Dimension: D3 — 源对象：条形结构 (src)
# Change: 3-5个矩形单色垂直条，宽度1或3（奇数），高度1-11（奇数），关于基线上下对称... → T形条（带水平臂）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste


# Design analysis for the D3 variant
# a) 具体化与行为追踪
#    现在每个对象都不是纯矩形竖条，而是单色 T 形：
#    - 一根宽 1 的竖干，仍然穿过中间的基线行，竖干高度为奇数。
#    - 竖干最顶端再长出一条奇数宽的水平臂，水平臂以竖干为中心。
#    - 因此对象的 bbox 高度仍等于竖干高度，但 bbox 宽度由横臂决定。
#    输入/输出仍保留原任务的全宽基线与“槽位颜色固定”的结构；变化只在于被搬运的对象形状和排序键。
#    具体例子：
#    - 左槽红色对象: (arm=3, stem=9)
#    - 中槽绿色对象: (arm=5, stem=3)
#    - 右槽蓝色对象: (arm=3, stem=5)
#    原任务会按 stem 高度排序 => (5,3,9) 对应绿色、蓝色、红色。
#    本变体按横臂宽度优先，再按竖干高度次级排序 => (3,5),(3,9),(5,3)，
#    所以蓝色 T 先去最左槽，红色 T 去中槽，绿色宽臂 T 去最右槽。
#    差异点正好发生在 “shapes 的排序” 这个环节：旧代码用 bbox 高度排，新代码必须读出 T 顶部横臂宽度。
#
# b) 复杂度校准
#    如果所有 T 的横臂都同宽，那么变体会退化成“还是看高度”，视觉上不明显。
#    因此生成器强制：
#    - 每个样例至少出现两种横臂宽度。
#    - 对象的 (arm_width, stem_height) 二元组互不相同。
#    - 宽度优先排序与高度优先排序不能给出同一顺序。
#    这样既能让横臂宽度成为可见控制变量，又不至于引入太多自由度：横臂宽度只取 3/5/7 中的可行值，竖干高度仍是少量奇数档位。
#
# c) 隐式假设审查
#    原 main() 里与变体相关的旧前提如下：
#    - `height = bbox_h(b)`：旧任务中这就是唯一尺寸指标；现在它只能表示竖干高度，不能再当主排序键。
#    - `anchor` 取 base_r 上对象像素的中位列：这仍然成立，因为 T 的竖干仍穿过基线，基线处恰有中心列像素。
#    - `rel` 以 (base_r, anchor) 为参考进行重放：这也仍然成立，因为所有对象都以竖干中心列作为稳定锚点。
#    - `find_components(... monochromatic=True)`：仍然成立，每个 T 仍是单色连通块。
#    新 solver 因此改为额外读取：
#    - `top_r = min(r)`，`arm_width = 顶行像素数`
#    - `stem_height = bbox_h(b)`
#    并按 `(arm_width, stem_height, ...)` 排序。
#
# d) 放置合法性
#    生成器不再按“竖条宽度”给槽位留空间，而是按该样例允许的最大横臂宽度预留整段窗口。
#    每个槽位窗口宽度 = `slot_w`，对象锚点固定在窗口中心，所以任意 T 被搬到任意槽位时：
#    - 横臂都不会越界；
#    - 相邻 T 至少隔 1 列，不会重叠；
#    - 基线仍完整可见，只有竖干穿过基线的那个像素会覆写成对象色，这与原任务一致。
#
# e) 视觉可观察性
#    观察 2-4 组样例时，人能看到：
#    - 所有对象都像“棒棒糖反过来”的 T，而不是纯矩形条。
#    - 宽横臂的对象总是更靠右，窄横臂的对象更靠左；同宽时更矮的竖干更靠左。
#    - 颜色并不跟着形状走，而是槽位颜色保留不变。
#    由于生成器明确排除了“宽度排序与高度排序一致”的样例，解题者不会被误导回原题。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort T-shaped bars by arm width, breaking ties by stem height, then
    redistribute the silhouettes across fixed colored slots.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, base_col}:
            continue

        pts = [(int(r), int(c)) for r, c in comp["coords"]]
        base_cols = sorted(c for r, c in pts if r == base_r)
        assert base_cols, "each T stem must cross the base row"
        anchor = int(base_cols[len(base_cols) // 2])

        top_r = min(r for r, _c in pts)
        arm_width = sum(1 for r, _c in pts if r == top_r)
        stem_height = int(bbox_h(comp["bbox"]))
        rel = [(r - int(base_r), c - int(anchor)) for r, c in pts]

        objects.append(
            {
                "color": col,
                "anchor": anchor,
                "arm_width": int(arm_width),
                "stem_height": stem_height,
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o["arm_width"]),
            int(o["stem_height"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, slot_w: int) -> list[int]:
    used = int(k * slot_w + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(slot_w)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _make_t_patch(stem_height: int, arm_width: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((int(stem_height), int(arm_width)), int(bg), dtype=int)
    mid_c = int(arm_width) // 2
    patch[0, :] = int(color)
    patch[:, mid_c] = int(color)
    return patch


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    specs: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, (arm_width, stem_height) in zip(anchors, colors, specs):
        patch = _make_t_patch(stem_height, arm_width, color, bg)
        paste(
            grid,
            patch,
            int(base_r - stem_height // 2),
            int(anchor - arm_width // 2),
            transparent=int(bg),
        )
    return grid


def _sample_specs(k: int, slot_w: int, n: int) -> list[tuple[int, int]] | None:
    arm_choices = list(range(3, int(slot_w) + 1, 2))
    height_choices = list(range(3, min(int(n), 11) + 1, 2))
    all_pairs = [(int(arm_w), int(stem_h)) for arm_w in arm_choices for stem_h in height_choices]
    if len(all_pairs) < k:
        return None

    for _ in range(200):
        pick = np.random.choice(len(all_pairs), size=k, replace=False)
        specs = [all_pairs[int(i)] for i in pick]
        if len({arm_w for arm_w, _stem_h in specs}) < 2:
            continue
        new_order = sorted(specs, key=lambda x: (int(x[0]), int(x[1])))
        old_order = sorted(specs, key=lambda x: (int(x[1]), int(x[0])))
        if new_order == old_order:
            continue
        return specs
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(23, 30, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    configs = [
        (int(k), int(slot_w))
        for k in range(3, 6)
        for slot_w in (5, 7)
        if int(k * slot_w + (k - 1)) <= n
    ]
    if not configs:
        return None
    k, slot_w = configs[int(np.random.randint(len(configs)))]

    lefts = _slot_lefts(n, k, slot_w)
    anchors = [int(left + slot_w // 2) for left in lefts]
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    specs = _sample_specs(k, slot_w, n)
    if specs is None:
        return None
    sorted_specs = sorted(specs, key=lambda x: (int(x[0]), int(x[1])))

    out = _draw_scene(n, base_r, base_color, bg, anchors, colors, sorted_specs)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_specs = [sorted_specs[int(i)] for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, in_specs)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")

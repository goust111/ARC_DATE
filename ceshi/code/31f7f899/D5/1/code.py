from __future__ import annotations

# Variant of 31f7f899
# Dimension: D5 — 变换：重新着色 (transform)
# Change: recolor — 条形形状放入新槽位后，颜色变为该槽位的原始颜色... → 颜色交换（双向）
#
# Design analysis
# a) 具体化与行为追踪
#    - 原任务输入：一条横向基线横贯整行；若干竖条以奇数高度穿过基线，因此每个槽位在基线上
#      都有一段该条形自己的颜色。
#    - 原任务输出：按高度升序把条形形状放入从左到右的槽位，整根条形都改成槽位原色。
#    - 本变体输出：几何重排不变，但颜色不再是“单向改色”。
#      条形主体（相对基线 dr != 0 的像素）获得目标槽位色；
#      条形在基线上的交叉段（dr == 0 的像素）改成该条形原始色。
#      这样每次放置都显式地发生一次“双向交换”：槽位色进入条形主体，条形原色回写到槽位基线。
#    - 小例子：若左槽是红色短条，右槽是蓝色高条，则原规则会把短条放到左槽、全红；高条放到右槽、全蓝。
#      变体下，放到左槽的短条主体仍为左槽色红，但其基线交叉像素写成“该短条来自哪里”的原色；
#      放到右槽的高条主体为蓝，而右槽基线交叉像素写成该高条的原色。差异集中出现在“基线槽位颜色”。
# b) 复杂度校准
#    - 若允许高度为 1，则条形没有“主体/基线”分离，颜色交换几乎不可见，因此生成器只采样 >=3 的奇数高度。
#    - 若允许条形回到原槽位，则该槽位看起来与原任务一致，因此生成器使用 derangement，确保每个槽位都发生交换。
#    - 颜色仍保持每条唯一，避免基线回写后出现颜色身份冲突。
# c) 隐式假设审查
#    - base row 检测依赖“存在一行拥有最多的非背景色像素”，原任务如此，变体未改输入，因此仍成立。
#    - 组件提取依赖“输入条形是单色组件且不与其他条重叠”，变体只改输出着色，因此仍成立；生成器继续保证这一点。
#    - anchor 依赖“每个条形都穿过 base row”，否则 dr==0 像素不存在，交换无法发生。生成器用奇数高度并以 base row 为中心放置来保证。
#    - 原 solver 在输出阶段把整根条形涂成 slot color；这在原任务中无害，但在变体中会丢失‘回写条形原色到基线’的信息，
#      所以必须把 dr==0 与其他像素分开处理。
# d) 放置合法性
#    - 每个条形都是 width 为 1 或 3 的竖直矩形，所有槽位共享同一宽度。
#    - `_slot_lefts` 维持矩形槽位之间至少 1 列间距，因此输入和输出都不会重叠。
#    - 高度不超过网格边界，且始终围绕 base row 居中；因此 solver 提取的相对坐标可原样重放。
# e) 视觉可观察性
#    - 观察者在 2-4 个样例中会看到：条形高度排序规则没变，但输出中每个槽位的基线颜色不再等于该槽位的条形主体颜色，
#      而是等于“搬进来的那根条形”的原色。由于每个槽位都强制发生交换且每根条都有非基线主体，这一差异稳定可见。
#    - 参考了 `/home/lixinyu/snap/new_code/solvers/arc2/a25697e4.py` 中“变换会同时改变移动对象与承载位置的颜色身份”的机制；
#      这里不引入旋转/翻转，只把这种双向颜色交换思想投射到基线槽位上。

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and redistribute them across fixed colored slots.

    Variant rule:
    - Rank bar shapes by height (shortest to tallest).
    - Rank original x-positions (slots) left to right.
    - Place the i-th shortest shape into the i-th leftmost slot.
    - Dual color swap:
      * non-baseline pixels of the placed shape take the slot's original color;
      * baseline pixels of the placed shape take the moved shape's original color.

    Key insight: geometry still follows "shape by height -> slot by position", but color identity
    is split across two loci so that the slot and the incoming bar exchange colors.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # Base row/color = (row,color) with maximal count on that row (excluding bg).
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
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        b = comp["bbox"]
        height = int(bbox_h(b))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        shape_col = int(shape["color"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(shape_col) if int(dr) == 0 else int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _random_derangement(k: int) -> np.ndarray:
    while True:
        perm = np.random.permutation(k)
        if np.all(perm != np.arange(k)):
            return perm


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    body_colors: list[int],
    base_colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, body_color, base_seg_color, height in zip(lefts, body_colors, base_colors, heights):
        patch = np.full((int(height), int(width)), int(body_color), dtype=int)
        patch[int(height) // 2, :] = int(base_seg_color)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _expected_output(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    sorted_heights: list[int],
    perm: np.ndarray,
) -> np.ndarray:
    inv_perm = np.empty(len(perm), dtype=int)
    for src_slot, rank in enumerate(perm):
        inv_perm[int(rank)] = int(src_slot)
    source_colors = [int(colors[int(inv_perm[i])]) for i in range(len(colors))]
    return _draw_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        body_colors=[int(c) for c in colors],
        base_colors=source_colors,
        heights=sorted_heights,
    )


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    # Heights start at 3 so every bar has a visible body separate from its baseline crossing.
    height_choices = list(range(3, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    perm = _random_derangement(k)
    in_heights = [int(sorted_heights[int(i)]) for i in perm]

    inp = _draw_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        body_colors=colors,
        base_colors=colors,
        heights=in_heights,
    )
    out = _expected_output(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        sorted_heights=sorted_heights,
        perm=perm,
    )

    # Derangement makes every slot participate in the swap, so the variant cannot collapse to recolor.
    if any(int(colors[i]) == int(out[base_r, lefts[i]]) for i in range(k)):
        return None
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

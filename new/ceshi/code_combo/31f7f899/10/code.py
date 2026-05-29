# Combo variant of 31f7f899 (index 10)
# combo_spec: D1/0+D5/1
# Name: 颜色匹配+双向颜色交换

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是“整行基线 + 若干穿过基线的单色竖条”，但每个槽位顶部额外有一个 1 像素 marker。
#      marker 的颜色只负责匹配：颜色为 c 的条形必须移动到颜色为 c 的 marker 所在槽位。
#    - 融合点在于：目标槽位还保留“槽位原色”，这个颜色来自输入时该槽位原本那根条的颜色，
#      它与 marker 颜色是两套信息。输出时，搬入条形的非基线部分获得槽位原色，而该槽位基线交叉段
#      回写成来条的原色，于是发生双向颜色交换。
#    - 小例子：三个槽位从左到右的输入条色是 [红, 绿, 蓝]，顶部 marker 是 [蓝, 红, 绿]。
#      原任务会按高度排序决定去向；这里只看 marker：蓝条去左槽、红条去中槽、绿条去右槽。
#      其中蓝条到左槽后，条形主体变成左槽原色红，但左槽基线段改成蓝。可见“匹配依据”和“着色依据”
#      被拆开了，解题者必须同时追踪颜色配对和颜色交换。
# b) 复杂度校准
#    - marker 仅占 1 像素，竖条面积至少为 3，组件分离稳定。
#    - 每个样例只使用 3-5 个互异颜色，marker 颜色与条色一一对应，避免匹配歧义。
#    - 生成器强制 marker 顺序相对槽位原色是 derangement，使每个槽位都发生真实移动和可见交换。
#    - 条高使用互异奇数且与颜色随机绑定，避免颜色匹配与原始“按高度排序”偶然重合。
# c) 隐式假设审查
#    - 原 solver 假设“所有非基线组件都是条形”，融合后不成立；需要先把 area==1 且不在基线行的组件识别为 marker。
#    - 原 solver 用 `slots = sorted(objects, key=anchor)` 表示槽位身份；融合后槽位身份改由 marker 颜色决定，
#      但“槽位原色”仍来自该 anchor 上输入条形的颜色，因此还要保留 anchor -> 原色 映射。
#    - 原 solver 输出时把整根条形统一涂成 slot color；融合后必须把 `dr == 0` 与 `dr != 0` 分开写色，
#      否则会丢失双向交换中的“基线回写”。
#    - 新 solver 额外假设：每种条色都有且只有一个同色 marker；每个 marker 的列正对某个槽位 anchor。
#      生成器用同一颜色集构造 marker 和条形，并把 marker 放在每个槽位中心列来保证这一点。
# d) 放置合法性
#    - 所有条形共享宽度 1 或 3，由 `_slot_lefts` 保证槽位间至少隔 1 列，因此输入/输出均不重叠。
#    - 高度只取 >=3 的奇数，并限制最大高度，让最高条顶端落在第 2 行及以下；第 0 行 marker 与条形之间
#      始终留出第 1 行空白，在 4 连通下不会粘连。
#    - 输出复用输入条形的相对坐标，只改变目标 anchor 和颜色，不会越界。
# e) 视觉可观察性
#    - 观察 2-4 个样例时，能稳定看到：marker 决定“哪根条去哪里”；输出中每个槽位的条形主体颜色
#      等于该槽输入时原本的颜色，而基线交叉段颜色等于搬进来的那根条的原色。主体色与基线色分离，
#      且 marker 顺序不等于高度顺序，因此不会退化成原任务或任一单变体。


def _find_base(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(int(g.shape[0])):
        vals, counts = np.unique(g[int(r), :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_scene(
    g: np.ndarray, bg: int, base_r: int, base_col: int
) -> tuple[dict[int, tuple[int, int]], list[dict[str, object]]]:
    markers: dict[int, tuple[int, int]] = {}
    objects: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        area = int(comp["area"])
        if area == 1:
            r, c = (int(x) for x in pts[0])
            if int(r) != int(base_r):
                markers[int(col)] = (int(r), int(c))
                continue

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

    return markers, objects


def _slot_map(objects: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    slot_by_anchor: dict[int, dict[str, object]] = {}
    for obj in objects:
        anchor = int(obj["anchor"])
        slot_by_anchor[anchor] = obj
    return slot_by_anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each bar to the slot whose marker has the same color, then swap colors bidirectionally.

    Output rule:
    - geometry: a bar keeps its shape and moves to the anchor of the marker with the same color;
    - color swap:
      * non-baseline pixels take the target slot's original input color;
      * baseline pixels at that slot take the incoming bar's original color.
    - markers and the background/base line elsewhere are preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)

    assert len(markers) == len(objects) and len(objects) > 0

    slot_by_anchor = _slot_map(objects)
    marker_anchor_by_color = {int(color): int(pos[1]) for color, pos in markers.items()}
    assert set(marker_anchor_by_color) == {int(obj["color"]) for obj in objects}
    assert set(marker_anchor_by_color.values()) == set(slot_by_anchor)

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)
    for color, (mr, mc) in markers.items():
        out[int(mr), int(mc)] = int(color)

    for obj in objects:
        source_color = int(obj["color"])
        target_anchor = int(marker_anchor_by_color[source_color])
        target_slot_color = int(slot_by_anchor[target_anchor]["color"])
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(target_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(source_color) if int(dr) == 0 else int(target_slot_color)

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
    for i in range(int(k)):
        lefts.append(int(c))
        c += int(width)
        if i < int(k) - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _random_derangement(items: list[int]) -> list[int]:
    while True:
        perm = [int(items[i]) for i in np.random.permutation(len(items))]
        if all(int(a) != int(b) for a, b in zip(perm, items)):
            return perm


def _draw_input_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    marker_colors: list[int],
    slot_colors: list[int],
    heights_by_color: dict[int, int],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)
    for left, marker_color in zip(lefts, marker_colors):
        anchor = int(left) + int(width // 2)
        grid[0, int(anchor)] = int(marker_color)
    for left, slot_color in zip(lefts, slot_colors):
        height = int(heights_by_color[int(slot_color)])
        patch = np.full((int(height), int(width)), int(slot_color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _expected_output_from_layout(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    marker_colors: list[int],
    slot_colors: list[int],
    heights_by_color: dict[int, int],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)

    for left, marker_color in zip(lefts, marker_colors):
        anchor = int(left) + int(width // 2)
        grid[0, int(anchor)] = int(marker_color)

    for left, marker_color, slot_color in zip(lefts, marker_colors, slot_colors):
        height = int(heights_by_color[int(marker_color)])
        top = int(base_r - height // 2)
        mid = int(height // 2)
        patch = np.full((int(height), int(width)), int(slot_color), dtype=int)
        patch[int(mid), :] = int(marker_color)
        paste(grid, patch, top, int(left))

    return grid


def _color_match_keep_color_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)
    marker_anchor_by_color = {int(color): int(pos[1]) for color, pos in markers.items()}

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)
    for color, (mr, mc) in markers.items():
        out[int(mr), int(mc)] = int(color)

    for obj in objects:
        source_color = int(obj["color"])
        target_anchor = int(marker_anchor_by_color[source_color])
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(target_anchor) + int(dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[int(rr), int(cc)] = int(source_color)
    return out


def _height_swap_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)
    for color, (mr, mc) in markers.items():
        out[int(mr), int(mc)] = int(color)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    for slot, shape in zip(slots, shapes):
        slot_color = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        shape_color = int(shape["color"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[int(rr), int(cc)] = int(shape_color) if int(dr) == 0 else int(slot_color)
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)
    slot_by_anchor = _slot_map(objects)
    marker_anchor_by_color = {int(color): int(pos[1]) for color, pos in markers.items()}
    out = main(g)

    moved_slots = 0
    swapped_slots = 0
    for color, target_anchor in marker_anchor_by_color.items():
        source_anchor = int(next(int(obj["anchor"]) for obj in objects if int(obj["color"]) == int(color)))
        slot_color = int(slot_by_anchor[target_anchor]["color"])
        if int(source_anchor) != int(target_anchor):
            moved_slots += 1
        if int(slot_color) != int(color):
            swapped_slots += 1

    diff_vs_color_match = int(np.count_nonzero(out != _color_match_keep_color_output(g)))
    diff_vs_height_swap = int(np.count_nonzero(out != _height_swap_output(g)))
    return {
        "slot_count": int(len(objects)),
        "moved_slots": int(moved_slots),
        "swapped_slots": int(swapped_slots),
        "diff_vs_color_match": int(diff_vs_color_match),
        "diff_vs_height_swap": int(diff_vs_height_swap),
    }


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    base_r = int(n // 2)
    bg = 8
    base_color = 6

    max_height = min(11, int(2 * (base_r - 2) + 1))
    height_choices = list(range(3, int(max_height) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, int(max_count) + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    palette = [1, 2, 3, 4, 5, 7, 9]
    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]

    slot_colors = [int(colors[i]) for i in np.random.permutation(k)]
    marker_colors = _random_derangement(slot_colors)

    chosen_heights = [int(h) for h in np.random.choice(height_choices, size=k, replace=False)]
    height_colors = [int(colors[i]) for i in np.random.permutation(k)]
    heights_by_color = {int(color): int(height) for color, height in zip(height_colors, chosen_heights)}

    inp = _draw_input_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        marker_colors=marker_colors,
        slot_colors=slot_colors,
        heights_by_color=heights_by_color,
    )
    out = _expected_output_from_layout(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        marker_colors=marker_colors,
        slot_colors=slot_colors,
        heights_by_color=heights_by_color,
    )

    metrics = _behavior_metrics(inp)
    if metrics["moved_slots"] != k or metrics["swapped_slots"] != k:
        return None
    if metrics["diff_vs_color_match"] <= 0 or metrics["diff_vs_height_swap"] <= 0:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")

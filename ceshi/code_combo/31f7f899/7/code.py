# Combo variant of 31f7f899 (index 7)
# combo_spec: D1/5+D6/0
# Name: 垂直基线+行对齐匹配

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste

# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 原任务有一条横向基线，若干竖条站在基线上，求解时按高度排序后重新分配到从左到右的颜色槽位。
#    - 本融合把整个空间旋转 90 度：基线改成纵向，条形改成水平条，槽位改成从上到下的一组“行槽位”。
#    - 同时把“全局排序”改成“局部对齐匹配”：
#      1. 每个水平条上方有一个可移动单像素标记，它跟着条形一起上下移动。
#      2. 每个槽位下方有一个固定单像素标记，它留在该槽位，不跟条形走。
#      3. 某条形上方标记的绝对列 c，要去匹配那个下方固定标记也位于列 c 的槽位。
#    - 小例子：假设三个槽位从上到下颜色是红/绿/黄，三个固定下标记列分别是 2/8/5。
#      输入中三条水平条当前的上标记列分别是 8/5/2（被错放到别的行槽位）。
#      旧逻辑会看宽度排序或保留原行序；融合后逻辑则按列对齐：
#      列 8 的条去绿色槽位，列 5 的条去黄色槽位，列 2 的条去红色槽位。
#    - 差异点不只是“旋转”，而是旋转后连匹配线索也一起改轴：
#      原任务靠尺寸序；融合版靠“同列的单像素线索”做一一配对。
#
# b) 复杂度校准
#    - 若固定标记列有重复，则会出现多条形抢同一槽位的歧义，因此生成器强制所有标记列互不相同。
#    - 若让宽度继续承担主要信息，视觉上容易又被看成“按宽度排序”；因此本融合故意让同一实例里的条宽一致，
#      把控制变量收敛到“标记列”，从而突出 D1 的空间匹配，同时保留 D6 的几何旋转。
#    - 条宽只从较大的奇数里选（>=7），这样同一条上可以放多个候选标记列且避开中轴列；条厚只取 1 或 3，
#      让 2-4 个例子里都能稳定看出“水平条 + 上下标记”的结构。
#    - 输入使用错排（derangement）而不是简单交换，保证每个槽位都是错位的，避免样例中出现“偶然已经对齐”的弱信号。
#
# c) 原 main() 的旧前提审查
#    - 原代码逐行找基线；融合后这会直接找错，必须改成逐列找整列主色。
#    - 原代码假设对象是不含额外提示点的竖条；现在如果直接遍历所有非背景非基线组件，会把标记像素也当成对象，
#      因此必须先识别“全是面积 1 组件”的 marker color，再把它排除。
#    - 原代码把 anchor 设在基线行上的列位置，并按 x 排槽位；融合后条形沿水平方向展开，槽位应改成按 y 的行槽位解析。
#    - 原代码核心是 `shapes = sorted(...height...)`；融合后这个前提失效，真正的匹配键变成 `bar_col -> slot_col`。
#    - 原代码输出只重画基线和条形；融合后若不重画上下两类标记，关键线索会丢失，输出不再可解释。
#
# d) 放置合法性
#    - 生成器使用 output-first。先构造“每条形的上标记列 = 该槽位下标记列”的正确输出，再用错排生成输入。
#    - 每个槽位的纵向 footprint 为 `1(上标记) + thickness(条体) + 1(下标记)`，槽位之间再留至少 1 行空白，
#      因而条体、移动标记、固定标记彼此都不重叠。
#    - 标记列一定落在公共条宽覆盖范围内，但不落在纵向基线列上；这样既不会越界，也不会破坏基线检测。
#    - 颜色槽位由行位置决定，条形被移动后会重着色为目标槽位颜色；输入输出都保留固定下标记。
#
# e) 视觉可观察性
#    - 人类看到几组样例时，会先注意到整列竖直基线和若干水平条，能明显区分出它已不是原始横基线任务。
#    - 再看输入输出，可见颜色槽位固定在各行，但上方单点标记跟着条移动，下方单点标记留在原槽位；
#      输出时每个槽位上下两点落在同一列，形成很直观的“列对齐”证据。
#    - 由于同一实例里条宽相同，任何“按宽度排序”的解释都会失效，解题者只能依赖空间对齐线索。
#
# 新 solver 的隐式假设审查（并由 generator 保证）
#    - 存在且仅存在一种 marker color，它的所有组件面积都为 1，且数量为偶数（每条形一个上标记、每槽位一个下标记）。
#    - 每个条形组件都是单色水平矩形，并且正上方恰有一个移动标记、正下方恰有一个固定标记。
#    - 所有条形共享同一纵向基线列；标记列都不等于该基线列。
#    - 固定下标记列两两不同，因此 `bar_col -> slot_col` 是双射。
#    - 条形之间竖向间隔足够大，使得上标记、条体、下标记都不会与别的槽位粘连成一个组件。


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, col, color)
    _h, w = grid.shape
    for c in range(w):
        vals, counts = np.unique(grid[:, c], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(c), int(v))
    assert best is not None
    _cnt, base_c, base_col = best
    return int(base_c), int(base_col)


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_c, base_col = _find_base(g, bg)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in comps:
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue
        by_color.setdefault(int(col), []).append(comp)

    marker_candidates = [
        (len(parts), int(col))
        for col, parts in by_color.items()
        if len(parts) >= 6 and len(parts) % 2 == 0 and all(int(part["area"]) == 1 for part in parts)
    ]
    assert marker_candidates
    marker_color = max(marker_candidates, key=lambda x: (int(x[0]), -int(x[1])))[1]

    marker_coords = {
        (int(comp["coords"][0][0]), int(comp["coords"][0][1]))
        for comp in by_color[int(marker_color)]
    }

    objects: list[dict[str, object]] = []
    for comp in comps:
        col = int(comp["color"])
        if col in {int(bg), int(base_col), int(marker_color)}:
            continue

        pts = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        body_cols = {int(c) for _r, c in pts}
        top = int(r0)
        thickness = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        bar_cols = [int(c) for (r, c) in marker_coords if int(r) == int(top - 1) and int(c) in body_cols]
        slot_cols = [int(c) for (r, c) in marker_coords if int(r) == int(r1 + 1) and int(c) in body_cols]

        assert len(bar_cols) == 1
        assert len(slot_cols) == 1

        rel = [(int(r) - int(top), int(c) - int(base_c)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "top": int(top),
                "thickness": int(thickness),
                "width": int(width),
                "size": int(len(rel)),
                "bar_col": int(bar_cols[0]),
                "slot_col": int(slot_cols[0]),
                "rel": rel,
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "base_c": int(base_c),
        "base_col": int(base_col),
        "marker_color": int(marker_color),
        "objects": objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each horizontal bar to the row-slot whose fixed marker shares the same column.

    Input structure:
    - Background plus a full-height vertical base line.
    - Several colored horizontal bars centered on that base line.
    - Each bar has one movable marker pixel above it.
    - Each row-slot has one fixed marker pixel below it.

    Rule:
    - Keep row-slot positions, colors, and fixed lower markers unchanged.
    - For each bar, read the column of its movable upper marker.
    - Move that bar vertically to the unique slot whose fixed lower marker is on the same column.
    - Recolor the moved bar to the destination slot color; preserve both kinds of markers.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    base_c = int(scene["base_c"])
    base_col = int(scene["base_col"])
    marker_color = int(scene["marker_color"])
    objects = list(scene["objects"])
    h, w = g.shape

    slot_by_col = {int(obj["slot_col"]): obj for obj in objects}
    assert len(slot_by_col) == len(objects)

    out = np.full_like(g, int(bg))
    out[:, int(base_c)] = int(base_col)

    for slot in sorted(objects, key=lambda o: int(o["top"])):
        rr = int(slot["top"]) + int(slot["thickness"])
        cc = int(slot["slot_col"])
        if 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = int(marker_color)

    for shape in objects:
        slot = slot_by_col[int(shape["bar_col"])]
        slot_top = int(slot["top"])
        slot_color = int(slot["color"])

        marker_r = int(slot_top) - 1
        marker_c = int(shape["bar_col"])
        if 0 <= marker_r < h and 0 <= marker_c < w:
            out[marker_r, marker_c] = int(marker_color)

        for dr, dc in shape["rel"]:
            rr = int(slot_top) + int(dr)
            cc = int(base_c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(slot_color)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_tops(n: int, k: int, thickness: int) -> list[int]:
    footprint = int(thickness + 2)  # upper marker + body + lower marker
    used = int(k * footprint + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars plus markers do not fit")
    gaps = _rand_composition(extra, k + 1)
    tops: list[int] = []
    start = int(gaps[0])
    for i in range(k):
        tops.append(int(start + 1))
        start += int(footprint)
        if i < k - 1:
            start += 1 + int(gaps[i + 1])
    return tops


def _allowed_marker_cols(base_c: int, width: int) -> list[int]:
    half = int(width // 2)
    cols = [int(c) for c in range(int(base_c - half), int(base_c + half + 1)) if int(c) != int(base_c)]
    return cols


def _is_monotonic(xs: list[int]) -> bool:
    return xs == sorted(xs) or xs == sorted(xs, reverse=True)


def _random_derangement(k: int) -> list[int]:
    while True:
        perm = [int(x) for x in np.random.permutation(k)]
        if all(int(perm[i]) != i for i in range(k)):
            return perm


def _draw_scene(
    n: int,
    base_c: int,
    base_color: int,
    bg: int,
    marker_color: int,
    tops: list[int],
    thickness: int,
    colors: list[int],
    width: int,
    cols: list[int],
    shape_idx_at_slot: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[:, int(base_c)] = int(base_color)

    left = int(base_c - width // 2)
    for slot_idx, top in enumerate(tops):
        shape_idx = int(shape_idx_at_slot[slot_idx])
        color = int(colors[slot_idx])
        bar_col = int(cols[shape_idx])
        slot_col = int(cols[slot_idx])

        patch = np.full((int(thickness), int(width)), int(color), dtype=int)
        paste(grid, patch, int(top), int(left))
        grid[int(top) - 1, int(bar_col)] = int(marker_color)
        grid[int(top) + int(thickness), int(slot_col)] = int(marker_color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    base_c = n // 2
    bg = 8
    base_color = 6
    marker_color = 0

    configs: list[tuple[int, int]] = []
    for k in range(3, 6):
        for thickness in (1, 3):
            used = int(k * (thickness + 2) + (k - 1))
            if used <= n:
                configs.append((int(k), int(thickness)))
    if not configs:
        return None

    k, thickness = configs[int(np.random.randint(len(configs)))]

    width_choices = [
        int(w)
        for w in range(7, min(n, 11) + 1, 2)
        if len(_allowed_marker_cols(base_c, int(w))) >= k
    ]
    if not width_choices:
        return None
    width = int(np.random.choice(width_choices))

    tops = _slot_tops(n, k, thickness)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    cols = [int(c) for c in np.random.choice(_allowed_marker_cols(base_c, width), size=k, replace=False)]
    if _is_monotonic(cols):
        return None

    out = _draw_scene(
        n=n,
        base_c=base_c,
        base_color=base_color,
        bg=bg,
        marker_color=marker_color,
        tops=tops,
        thickness=thickness,
        colors=colors,
        width=width,
        cols=cols,
        shape_idx_at_slot=list(range(k)),
    )

    perm = _random_derangement(k)
    inp = _draw_scene(
        n=n,
        base_c=base_c,
        base_color=base_color,
        bg=bg,
        marker_color=marker_color,
        tops=tops,
        thickness=thickness,
        colors=colors,
        width=width,
        cols=cols,
        shape_idx_at_slot=perm,
    )

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

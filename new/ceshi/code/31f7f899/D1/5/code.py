# Variant of 31f7f899
# Dimension: D1 — 匹配条件：按高度排序映射槽位 (match_cond)
# Change: size_ordering — 将条形按高度升序排序，槽位按列位置从左到右排序，第i短的条形放入第i个槽位... → 行对齐匹配

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis for the D1 variant
# a) 具体化与行为追踪
#    - 仍保留原任务的整体骨架：背景色、贯穿整行的基线、若干彩色竖条，且槽位颜色仍由原列位置决定。
#    - 变体新增一种“标记色”单像素：
#      1. 每个可移动条形左侧有一个单像素标记，它跟着条形一起横向移动。
#      2. 每个槽位右侧也有一个单像素标记，它固定在该槽位旁，不随条形变化。
#    - 输出规则不再看高度全局排序，而是看“行对齐”：
#      某条形左侧标记所在的绝对行 r，要被放到右侧固定标记也位于同一行 r 的槽位。
#    - 小例子：假设三个槽位从左到右颜色是蓝/红/黄，固定右标记行分别是 2/6/4。
#      输入里三个条形左标记行分别是 6/4/2（错位摆放）。
#      原逻辑会按高度排序后放回蓝/红/黄槽位；新逻辑则按标记行匹配：
#      行6的条形去红槽位，行4的条形去黄槽位，行2的条形去蓝槽位。
#    - 与原任务的核心差异体现在“匹配阶段”：
#      原任务：shape -> slot 的映射由全局高度序决定。
#      新任务：shape -> slot 的映射由局部空间线索（同一绝对行的标记）决定。
#
# b) 复杂度校准
#    - 若只加一个颜色提示但不保证唯一性，会出现多条形可匹配同一槽位的歧义，因此生成器强制所有标记行互不相同。
#    - 若允许高度为1，则条形本身也可能退化成单像素，和标记像素混淆；因此高度只取 >=3 的奇数，保证“条形组件面积>1”。
#    - 若槽位右标记和相邻条形左标记距离过近，单像素提示会挤在一起难以解析；因此横向布局保留“左标记 + 条形 + 右标记 + 至少1列空隙”的 footprint。
#    - 视觉上，输入中固定右标记和可移动左标记会明显错位；输出中二者在同一槽位上实现同一行对齐，区别于原任务的高度排序。
#
# c) 原 main() 的旧前提审查
#    - `objects` 以前默认只含真正条形组件；变体里若仍直接枚举所有非背景非基线组件，会把单像素标记也当作条形，错误。
#      现在必须先识别出“全为面积1的那种颜色”作为 marker color，再把它从条形组件里排除。
#    - `slots = sorted(objects, key=anchor)` 以前默认槽位定义只依赖 x 位置。
#      变体中槽位仍由 x 位置和颜色定义，但真正的匹配键改成 `slot_row`，所以不能再用 height-order zip。
#    - `shapes = sorted(objects, key=height)` 是原任务核心；在变体中该排序前提已失效，必须改成按 `bar_row -> slot_row` 做一一匹配。
#    - 以前输出时只重画基线和条形；现在还必须重画固定右标记以及随条形移动的左标记，否则输入中的新线索会在输出中消失。
#
# d) 放置合法性
#    - 每个槽位 footprint 宽度为 `1(左标记) + width(条形) + 1(右标记)`，槽位之间至少再留1列空白。
#    - 标记行必须位于该条形自身高度范围内，且不等于基线行，避免被基线覆盖。
#    - 槽位右标记固定在 `left + width`；条形左标记固定在 `left - 1`，生成器保证都在网格内。
#    - 输出优先构造“正确对齐态”，输入则通过置换条形形状/左标记对来制造错位，并用 `main(inp) == out` 验证。
#
# e) 视觉可观察性
#    - 观察者能在 2-4 个样例里看到：颜色槽位保持不变，右侧提示点也固定，但左侧提示点跟随不同高度的条形在各槽位间换位。
#    - 输出中每个槽位两侧的提示点处在同一行，形成清晰的“行对齐”证据；这比仅靠高度变化更直接。
#    - 生成器额外让输出中的高度顺序通常不是从左到右升序，从而避免样本视觉上退化成原始 size_ordering 任务。
#
# 参考任务 38007db0 的启发
#    - 该参考任务不是全局排序，而是“每个局部单元依赖一个唯一线索被挑出”。这里沿用同样的复杂度标尺：
#      用每个条形自己的唯一行标记去找唯一槽位，而不是做更复杂的多步推理或引入额外对象种类。
#
# 新 solver 的隐式假设审查（并由 generator 保证）
#    - 所有可移动条形都是单色、4连通、面积>1 的竖向矩形。
#    - 存在且仅存在一种“全部组件面积都为1”的颜色作为 marker color。
#    - 每个条形左侧恰有一个移动标记；每个槽位右侧恰有一个固定标记。
#    - 所有固定标记行互不相同，因此 `bar_row -> slot_row` 是双射。
#    - 槽位间距足够大，不会让一个槽位的右标记落到另一个条形的左标记列上。


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H, _W = grid.shape
    for r in range(H):
        vals, counts = np.unique(grid[r, :], return_counts=True)
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


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

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
        body_rows = {int(r) for r, _c in pts}
        left = int(c0)
        width = int(c1 - c0 + 1)

        bar_rows = [int(r) for (r, c) in marker_coords if int(c) == int(left - 1) and int(r) in body_rows]
        slot_rows = [int(r) for (r, c) in marker_coords if int(c) == int(c1 + 1)]

        assert len(bar_rows) == 1
        assert len(slot_rows) == 1

        rel = [(int(r) - int(base_r), int(c) - int(left)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "left": int(left),
                "width": int(width),
                "height": int(bbox_h(comp["bbox"])),
                "size": int(len(rel)),
                "bar_row": int(bar_rows[0]),
                "slot_row": int(slot_rows[0]),
                "rel": rel,
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "base_r": int(base_r),
        "base_col": int(base_col),
        "marker_color": int(marker_color),
        "objects": objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each bar to the slot whose fixed side-marker shares the same row.

    Input structure:
    - Background plus a full-width horizontal base line.
    - Several colored vertical bars anchored on that base line.
    - Each bar has one movable marker pixel on its left side.
    - Each slot position has one fixed marker pixel on its right side.

    Rule:
    - Keep slot x-positions, colors, and fixed right-side markers unchanged.
    - For each bar, read the row of its movable left-side marker.
    - Move that bar horizontally to the unique slot whose fixed right-side marker is on the same row.
    - Recolor the moved bar to the destination slot color; preserve both kinds of marker pixels.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    base_r = int(scene["base_r"])
    base_col = int(scene["base_col"])
    marker_color = int(scene["marker_color"])
    objects = list(scene["objects"])
    H, W = g.shape

    slot_by_row = {int(obj["slot_row"]): obj for obj in objects}
    assert len(slot_by_row) == len(objects)

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)

    for slot in sorted(objects, key=lambda o: int(o["left"])):
        rr = int(slot["slot_row"])
        cc = int(slot["left"]) + int(slot["width"])
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = int(marker_color)

    for shape in objects:
        slot = slot_by_row[int(shape["bar_row"])]
        slot_left = int(slot["left"])
        slot_color = int(slot["color"])

        marker_r = int(shape["bar_row"])
        marker_c = int(slot_left) - 1
        if 0 <= marker_r < H and 0 <= marker_c < W:
            out[marker_r, marker_c] = int(marker_color)

        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_left) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_color)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    footprint = int(width + 2)  # left marker + body + right marker
    used = int(k * footprint + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars plus markers do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    start = int(gaps[0])
    for i in range(k):
        lefts.append(int(start + 1))
        start += int(footprint)
        if i < k - 1:
            start += 1 + int(gaps[i + 1])
    return lefts


def _allowed_rows(base_r: int, height: int) -> list[int]:
    rad = int(height // 2)
    return [int(base_r + off) for off in range(-rad, rad + 1) if int(off) != 0]


def _choose_unique_rows(base_r: int, heights: list[int]) -> list[int] | None:
    allowed = [list(dict.fromkeys(_allowed_rows(base_r, int(h)))) for h in heights]
    order = sorted(range(len(heights)), key=lambda i: (len(allowed[i]), int(heights[i])))
    chosen: list[int | None] = [None] * len(heights)
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = int(order[pos])
        opts = allowed[idx][:]
        np.random.shuffle(opts)
        opts.sort(key=lambda r: (abs(int(r) - int(base_r)), int(r)), reverse=True)
        for row in opts:
            row = int(row)
            if row in used:
                continue
            used.add(row)
            chosen[idx] = int(row)
            if backtrack(pos + 1):
                return True
            used.remove(row)
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(x) for x in chosen if x is not None]


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    marker_color: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
    rows: list[int],
    shape_idx_at_slot: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)

    for slot_idx, left in enumerate(lefts):
        shape_idx = int(shape_idx_at_slot[slot_idx])
        height = int(heights[shape_idx])
        color = int(colors[slot_idx])
        row = int(rows[shape_idx])
        slot_row = int(rows[slot_idx])

        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
        grid[row, int(left) - 1] = int(marker_color)
        grid[slot_row, int(left) + int(width)] = int(marker_color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6
    marker_color = 0

    height_choices = list(range(3, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices))

    configs: list[tuple[int, int]] = []
    for k in range(3, max_count + 1):
        for width in (1, 3):
            used = int(k * (width + 2) + (k - 1))
            if used <= n:
                configs.append((int(k), int(width)))
    if not configs:
        return None

    k, width = configs[int(np.random.randint(len(configs)))]
    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    picked_heights = [int(h) for h in np.random.choice(height_choices, size=k, replace=False)]
    out_heights = [int(picked_heights[i]) for i in np.random.permutation(k)]
    if out_heights == sorted(out_heights):
        a, b = np.random.choice(k, size=2, replace=False)
        out_heights[a], out_heights[b] = out_heights[b], out_heights[a]

    rows = _choose_unique_rows(base_r, out_heights)
    if rows is None or len(set(rows)) != k:
        return None

    out = _draw_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        marker_color=marker_color,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=out_heights,
        rows=rows,
        shape_idx_at_slot=list(range(k)),
    )

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    inp = _draw_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        marker_color=marker_color,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=out_heights,
        rows=rows,
        shape_idx_at_slot=[int(x) for x in perm],
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

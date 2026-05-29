# Combo variant of 2ba387bc (index 9)
# combo_spec: D2/2+D13/0
# Name: 距离排序+对角线分布

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the combo variant

a) 具体化与行为追踪
   - 原题输入：背景上散落若干 4x4 方块，两类对象分别是
     1. frame：1 像素单色边框 + 2x2 背景内孔；
     2. solid：4x4 实心单色块。
   - 原题输出：仍把 frame 堆到左列、solid 堆到右列，组内按输入中的 (row, col) 排序。
   - 融合后输入：对象不再“任意散点”，而是整体沿主对角线附近排布，行列坐标正相关，视觉上像一串沿左上到右下的
     4x4 方块。frame 与 solid 交错穿插在这条对角线带上。
   - 融合后输出：分组规则不变，但组内顺序不看 row，而看“对象中心到整张网格中心的欧氏距离”，近者先堆到上面。
   - 具体例子：设 6 个对象沿对角线依次位于
     (2,3), (5,8), (10,11), (13,16), (18,19), (21,24)。
     原题会在每组内部按这些 row 从小到大堆叠；融合后则会优先拿靠近网格中心的第 3/4 个对象，
     再拿次近者，最后才是两端。对角线分布会诱导解题者误以为“仍然只是从上到下”，但真正起作用的是距离排序。

b) 复杂度校准
   - 只改两个维度：排序键改为“距中心”，空间分布改为“主对角线带”。对象形状、颜色语义、输出版式都保持原题，
     这样变化清晰但不混维。
   - 生成器固定让两组对象都跨越网格中心，并且交错分布。这样“按 row 排”和“按距离排”会稳定地产生不同结果，
     不是偶尔才显现的弱变化。
   - 对角线不是一条完全刚性的直线，而是一个窄带；允许小幅抖动，避免样例机械重复，同时仍然肉眼可见。

c) 隐式假设审查
   - `find_components(... monochromatic=False)` 假设对象之间不接触，否则不同颜色方块会并成一个组件。
     融合后仍需成立，所以生成器保证任意相邻对象至少在行或列方向留出 1 格空隙。
   - `bbox_h == bbox_w == 4` 假设对象仍是 4x4。两个变体都不改对象大小，这个前提继续成立。
   - frame 判定依赖“边框单色且 2x2 内部全是背景”。融合后对象内部结构没变，这个前提继续成立。
   - 原代码的 `sort(key=(row, col))` 假设题目规则就是位置顺序；D2/2 明确破坏了这一点，必须改成按距中心排序。
   - 新排序规则隐含一个新假设：同组对象最好不要恰好等距，否则主规则不明显。solver 用 `(dist2, row, col)` 做稳定
     tie-break，generator 则主动筛除同组等距样本。
   - D13/0 还引入一个额外可见性前提：对象应明显呈主对角线趋势，而不是普通散点。generator 用窄带约束与单调
     `(row, col)` 序列保证这一点。

d) 放置合法性
   - 生成器先构造一个严格递增的 `(row, col)` 序列；相邻步长只取 `(5,3)` 或 `(3,5)`。因为每对相邻对象至少有一个维度
     相差 5，所以 4x4 bbox 之间必然不会重叠，也不会边接触。
   - 起点会根据总步长回填到网格中部附近，并检查边界，确保所有对象都完整落在网格内。
   - 实际 paste 前后都会再次走 `_variant_visible()` 和 `main()` 校验，避免构造出 solver 不支持的输入。

e) 视觉可观察性
   - 距离排序依赖的是“离整张图中心近或远”，这是 ARC 中人类可见的空间关系，不依赖颜色编号、生成顺序等不可见信息。
   - 对角线分布会让列序和行序几乎同步，从视觉上制造一种“也许是沿对角线从左上到右下堆叠”的假象；但样例会显示
     输出优先取中部对象，因此真正规则仍可被归纳为“按距中心排序”。
   - 生成器要求两组都发生重排，避免只在一组上出现信号过弱的情况。

f) 可视化整洁度
   - 输入里只有背景和孤立的 4x4 方块，没有额外标记或噪声；对角线带宽有限，整体结构清楚。
   - frame 与 solid 仍由形状直接区分，且颜色互不重复，观察者可以轻松追踪每个对象在输出中的去向。
   - 输出继续沿用原题的两列堆叠，因此视觉负担集中在“顺序变化”，而不是版式变化。

g) 变化的可推断性
   - 本融合不引入颜色映射。颜色只承担“区分对象身份”的作用，不参与排序或分类判定。
   - 由于每个对象颜色唯一，解题者可以直接比对输入位置与输出顺序，从少量样例中归纳出“中间近、两端远”的排序。

h) 量化退化审查
   - 若样本退化成原题，则同一输入上“按 row 排”和“按距离排”会得到完全相同的输出。生成器用两个量化指标筛查：
     1. `reordered_groups`：有多少个组的距离排序签名不同于 row 排序签名；
     2. `diag_band`：所有对象中心的 `(col - row)` 离散范围，越小表示越贴近主对角线窄带。
   - 只有当 `reordered_groups == 2` 且 `diag_band` 足够小的样本才会被接受。

i) `main()` 的新隐式假设与 generator 保证
   - `main()` 新增的核心假设只有两个：
     1. 输入对象仍可被稳定分离成若干独立 4x4 组件；
     2. 同组对象的距中心排序最好没有肉眼不可分辨的完全平局。
   - `generate_input()` 通过单调对角线步长和唯一距离筛选保证这两点，因此 solver 的新增前提都被构造性满足。
"""


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.flatten()))
    return len(border_cols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _distance2_to_center(r0: int, c0: int, H: int, W: int) -> int:
    """
    Squared Euclidean distance in doubled coordinates.

    Object center: (r0 + 1.5, c0 + 1.5)
    Grid center:   ((H - 1) / 2, (W - 1) / 2)
    """
    dr2 = int(2 * int(r0) + 3 - (int(H) - 1))
    dc2 = int(2 * int(c0) + 3 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _collect_objects(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[dict] = []
    solids: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        obj = {
            "r0": int(r0),
            "c0": int(c0),
            "sub": sub.copy(),
            "dist2": _distance2_to_center(int(r0), int(c0), int(H), int(W)),
        }
        if _is_frame(sub, bg):
            frames.append(obj)
        else:
            solids.append(obj)
    return int(bg), frames, solids


def _distance_sorted(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"])))


def _position_order_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(group, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _distance_order_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    ordered = _distance_sorted(group)
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in ordered)


def _diagonal_band(objects: list[dict]) -> int:
    if not objects:
        return 0
    deltas = [int(obj["c0"]) - int(obj["r0"]) for obj in objects]
    return int(max(deltas) - min(deltas))


def _variant_visible(g: np.ndarray) -> bool:
    _bg, frames, solids = _collect_objects(np.array(g, dtype=int))
    groups = [frames, solids]
    if any(len(group) != 3 for group in groups):
        return False
    if any(len({int(obj["dist2"]) for obj in group}) != len(group) for group in groups):
        return False

    reordered_groups = 0
    for group in groups:
        if _position_order_signature(group) != _distance_order_signature(group):
            reordered_groups += 1
    if reordered_groups != 2:
        return False

    all_objects = sorted(frames + solids, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    cols = [int(obj["c0"]) for obj in all_objects]
    if not all(cols[i] < cols[i + 1] for i in range(len(cols) - 1)):
        return False

    if _diagonal_band(all_objects) > 8:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Classify each isolated 4x4 object as frame or solid exactly as in the original task.
    - Stack frames into the left output column and solids into the right output column.
    - Inside each group, order objects by distance from the object's center to the grid center.
    - Use (row, col) only as a deterministic tie-break, not as the semantic rule.
    """
    g = np.array(grid, dtype=int)
    bg, frames, solids = _collect_objects(g)
    assert frames or solids

    frames = _distance_sorted(frames)
    solids = _distance_sorted(solids)

    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, obj in enumerate(frames):
        paste(out, obj["sub"], int(i * 4), 0)
    for i, obj in enumerate(solids):
        paste(out, obj["sub"], int(i * 4), 4)

    return out


def _sample_unique_color(used_cols: set[int]) -> int:
    col = int(np.random.randint(1, 10))
    while col in used_cols:
        col = int(np.random.randint(1, 10))
    used_cols.add(col)
    return int(col)


def _sample_step_pattern(n_steps: int) -> list[tuple[int, int]]:
    heavy_rows = int(np.random.choice([2, 3]))
    steps = [(5, 3)] * heavy_rows + [(3, 5)] * (int(n_steps) - heavy_rows)
    order = np.random.permutation(len(steps))
    return [steps[int(i)] for i in order]


def _centered_start(limit: int, total_step: int) -> int:
    low = 1
    high = int(limit) - 5 - int(total_step)
    if high < low:
        raise ValueError("sequence cannot fit inside grid")
    base = int(low + (high - low) // 2)
    jitter = int(np.random.randint(-1, 2))
    return int(min(high, max(low, base + jitter)))


def _build_diagonal_positions(H: int, W: int, count: int) -> list[tuple[int, int]]:
    if count < 2:
        raise ValueError("count must be >= 2")

    for _ in range(80):
        steps = _sample_step_pattern(int(count) - 1)
        row_span = int(sum(dr for dr, _dc in steps))
        col_span = int(sum(dc for _dr, dc in steps))
        if H - 5 - row_span < 1 or W - 5 - col_span < 1:
            continue

        r = _centered_start(int(H), row_span)
        c = _centered_start(int(W), col_span)
        offset = int(np.random.randint(-2, 3))
        c = int(min(W - 5 - col_span, max(1, c + offset)))

        pos = [(int(r), int(c))]
        rr = int(r)
        cc = int(c)
        for dr, dc in steps:
            rr += int(dr)
            cc += int(dc)
            pos.append((int(rr), int(cc)))

        if max((c0 - r0) for r0, c0 in pos) - min((c0 - r0) for r0, c0 in pos) <= 8:
            return pos

    raise ValueError("failed to build diagonal positions")


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(27, 31))
    W = int(np.random.randint(27, 31))
    g = np.full((H, W), int(bg), dtype=int)

    n_frames = 3
    n_solids = 3
    total = n_frames + n_solids

    try:
        positions = _build_diagonal_positions(H, W, total)
        used_cols: set[int] = set()
        frame_cols = [_sample_unique_color(used_cols) for _ in range(n_frames)]
        solid_cols = [_sample_unique_color(used_cols) for _ in range(n_solids)]
        start_with_frame = bool(np.random.randint(0, 2))

        frame_idx = 0
        solid_idx = 0
        for idx, (r0, c0) in enumerate(positions):
            is_frame = (idx % 2 == 0) if start_with_frame else (idx % 2 == 1)
            if is_frame:
                patch = _make_frame(frame_cols[frame_idx], bg)
                frame_idx += 1
            else:
                patch = _make_solid(solid_cols[solid_idx])
                solid_idx += 1
            paste(g, patch, int(r0), int(c0), transparent=bg)

        if not _variant_visible(g):
            return None

        _ = main(g)
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")

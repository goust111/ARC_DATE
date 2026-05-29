# Variant of 2ba387bc
# Dimension: D2 — 排序规则——组内排列顺序 (match_cond)
# Change: 每组（框架组和实心组）内部按输入中的行位置(row)从上到下排序，行相同时按列位置(col)排序... → 按到网格中心的距离排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for the D2 variant

a) 具体化与行为追踪
   - 输入仍然只包含散落在背景上的 4x4 方块，分为两类：
     1. frame：单色边框 + 2x2 背景内孔。
     2. solid：4x4 全填充。
   - 输出仍然是 8 列：左列堆叠所有 frame，右列堆叠所有 solid。
   - 唯一核心变化是“每组内部的堆叠顺序”。原题按 (r0, c0) 从上到下、从左到右；本变体改为按
     “方块中心到整张网格中心的欧氏距离”从近到远。
   - 具体追踪例子：若三个 frame 的左上角分别为 (1,1)、(10,8)、(16,2)，在 24x24 网格中，
     原题顺序是 (1,1) -> (10,8) -> (16,2)；本题先比较三个 4x4 方块中心
     (2.5,2.5)、(11.5,9.5)、(17.5,3.5) 到网格中心 (11.5,11.5) 的距离，新的堆叠顺序会变为
     最近者先放。差异发生在“排序键”这一环，其余检测、分类、输出拼接都保持原样。

b) 复杂度校准
   - 不引入新的对象种类、标记或颜色语义，仍然只有 frame / solid 两类，避免把 D2 变化混入别的维度。
   - 距离排序比简单行序更复杂，但仍是 ARC 可见推理：观察者可从“靠近中心的先出现在输出上方”归纳规则。
   - 生成器要求每组距离唯一，并要求至少一组的距离顺序与旧行列顺序不同，避免退化成原题。

c) 隐式假设审查（针对 main 中与 D2 相关的部分）
   - `find_components(... monochromatic=False)` 依赖对象彼此不接触，否则多个方块会并成一个组件。
     该前提在变体里仍然成立，生成器继续使用 `padding=1` 保证分离。
   - `bbox_h == bbox_w == 4` 依赖输入对象仍是 4x4；D2 不改变形状，这个前提继续成立。
   - frame 判定依赖 “边框单色且 2x2 内部全是背景”。D2 不改变对象内部结构，这个前提继续成立。
   - 原来的 `sort(key=(row,col))` 依赖“题目要求按位置排序”；该前提不再成立，必须替换为距离键。
   - 新的距离排序额外引入一个温和假设：如果同组对象距离完全相等，就需要次级键保证确定性。
     solver 用 `(dist2, row, col)`；generator 则主动避免同组距离相等，以保持规则肉眼可见。

d) 放置合法性
   - 生成器仍通过 `random_free_location(... border=1, padding=1, padding_connectivity=4)` 放置 4x4 物体，
     保证不重叠、不越界、不共享边，从而不会被组件提取错误合并。
   - 颜色仍然在所有对象之间唯一，避免 frame/solid 之外出现额外歧义。
   - 生成后会再次用 solver 同步验证，确保输入可被当前 `main()` 正常处理。

e) 视觉可观察性
   - 排序依据是“离中心近还是远”，这是可见空间关系，不依赖颜色编号等不可见属性。
   - 为了让 2-3 个样例就能看出变化，生成器要求至少一组对象的“距离排序”与“行列排序”不同；
     否则虽然 solver 正确，但样例会看起来像原题，属于退化。

f) 可视化整洁度
   - 仍使用孤立的 4x4 方块，背景干净，无额外装饰。对象之间至少留 1 格空隙，输入不会粘连成团。
   - 输出结构完全沿用原题的两列堆叠，因此视觉负担只来自“顺序变化”，不会引入新的杂乱因素。

g) 颜色变化可推断性
   - 本变体不改变颜色映射；颜色只用于区分不同对象。颜色种类数量仍受限于最多 8 个对象，足够清晰。

h) 新增假设与 generator 保证
   - `main()` 新增的真正假设只有：“同组对象最好不要恰好等距，否则主规则不明显。”
   - `generate_input()` 用 `_variant_visible()` 保证：
     1. frame 组和 solid 组都至少有两个对象；
     2. 每组内部距离平方互不相同；
     3. 至少一组的距离排序与旧的 `(row,col)` 排序不同。
"""


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_cols = set(map(int, border.flatten()))
    return len(border_cols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _distance2_to_center(r0: int, c0: int, H: int, W: int) -> int:
    """
    Squared Euclidean distance on a doubled coordinate system.

    4x4 object's center is at (r0 + 1.5, c0 + 1.5).
    Grid center is at ((H - 1) / 2, (W - 1) / 2).
    Doubling keeps everything integral:
      obj center -> (2*r0 + 3, 2*c0 + 3)
      grid center -> (H - 1, W - 1)
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


def _variant_visible(g: np.ndarray) -> bool:
    _bg, frames, solids = _collect_objects(np.array(g, dtype=int))
    groups = [frames, solids]

    if any(len(group) < 2 for group in groups):
        return False
    if any(len({int(obj["dist2"]) for obj in group}) != len(group) for group in groups):
        return False

    reordered_groups = 0
    for group in groups:
        if _position_order_signature(group) != _distance_order_signature(group):
            reordered_groups += 1
    return reordered_groups >= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: frames left, solids right.

    Variant rule:
    - Classify 4x4 objects as frame or solid exactly as in the original task.
    - Inside each group, sort by distance from the object's center to the grid center.
    - Ties are broken by (row, col) only to keep the solver deterministic.
    """
    g = np.array(grid, dtype=int)
    bg, frames, solids = _collect_objects(g)
    assert frames or solids

    frames = _distance_sorted(frames)
    solids = _distance_sorted(solids)

    outH = 4 * max(int(len(frames)), int(len(solids)))
    outW = 8
    out = np.full((outH, outW), int(bg), dtype=int)

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


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(22, 31))
    W = int(np.random.randint(22, 31))
    g = np.full((H, W), int(bg), dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(2, 5))

    try:
        used_cols: set[int] = set()
        objects: list[np.ndarray] = []

        for _ in range(n_frames):
            objects.append(_make_frame(_sample_unique_color(used_cols), bg))
        for _ in range(n_solids):
            objects.append(_make_solid(_sample_unique_color(used_cols)))

        for idx in np.random.permutation(len(objects)):
            patch = objects[int(idx)]
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
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

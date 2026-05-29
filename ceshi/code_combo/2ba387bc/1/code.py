# Combo variant of 2ba387bc (index 1)
# combo_spec: D1/2+D2/2
# Name: 面积分类+距离排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for the combo variant

a) 具体化与行为追踪
   - 输入仍然是若干彼此分离的 4x4 物体，背景为整图最常见颜色。
   - 物体只分两档面积：
     1. 面积 12：4x4 中恰有 12 个非背景像素，视觉上通常是空心框。
     2. 面积 16：4x4 全填满，视觉上是实心块。
   - 输出仍为 8 列，两列各宽 4：
     左列堆叠所有面积 12 的物体，右列堆叠所有面积 16 的物体。
   - 融合后的关键变化有两处，而且必须一起生效：
     1. 分组不再看“2x2 中心是否全背景”，而是直接数非背景像素面积是 12 还是 16。
     2. 每组内部不再按输入中的行序，而是按“物体中心到整张网格中心的欧氏距离”从近到远。
   - 例子：在 24x24 网格里，三个面积 12 物体左上角分别是 (1,1)、(9,10)、(17,2)。
     它们中心分别是 (2.5,2.5)、(10.5,11.5)、(18.5,3.5)，到网格中心 (11.5,11.5)
     的距离顺序是 (9,10) 最近，其次 (17,2)，最后 (1,1)。原题会按 row 排成
     (1,1)->(9,10)->(17,2)，本变体会改成 (9,10)->(17,2)->(1,1)。

b) 复杂度校准
   - 只保留原题的 4x4 离散对象和两列输出，不引入额外标记，避免维度污染。
   - D1 使用“面积 12 vs 16”而不是读内部 2x2，D2 使用“中心距离”而不是行列顺序。
   - generator 强制两组都至少有 2 个对象，并且至少一组真的发生“距离序 != 行序”，
     否则样例会退化成原题或只体现单一维度。

c) 隐式假设审查
   - `find_components(... monochromatic=False)` 假设对象彼此不接触；generator 用 `padding=1`
     保证这一点。
   - `bbox_h == bbox_w == 4` 假设所有目标对象都还是 4x4；融合后仍成立。
   - 原题用“中心 2x2 是否全背景”做分类，这个前提在融合后被替换为“面积必须是 12 或 16”。
     因此 solver 新增的输入假设是：所有有效对象都属于这两档面积。
   - 距离排序新增一个确定性问题：若同组等距，会导致主规则不够清晰。solver 用
     `(dist2, row, col)` 做稳定 tie-break，而 generator 主动避免同组等距。
   - 输出高度 `4 * max(len(left_group), len(right_group))` 仍然依赖对象高度恒为 4；
     这个前提没有变化。

d) 放置合法性
   - 每个物体都通过 `random_free_location(... border=1, padding=1, padding_connectivity=4)`
     放置，因此不会重叠、越界、贴边粘连。
   - 每个对象颜色唯一，避免视觉上把两个相同颜色的物体误看成同类证据。

e) 视觉可观察性
   - 面积分组是可见的，因为一类占 12 格，一类占 16 格。
   - 排序依据也是可见的，因为“更靠近整图中心的物体在输出中更靠上”。
   - 为了让 2-4 个样例足以归纳，generator 禁止两组都恰好保持原来的行序。

f) 可视化整洁度
   - 输入仍是干净背景上的孤立 4x4 物体，没有叠加标记。
   - 输出结构沿用原题两列堆叠，视觉负担只来自“数面积”和“比中心距离”。

g) 变化的可推断性
   - 不引入颜色映射规则，颜色只用来区分对象身份。
   - 真正的控制变量只有两个：面积档位与中心距离。每个样例中两档面积都会重复出现，
     且至少一个组内顺序会和原题不同。

h) 量化退化检查
   - 使用两个指标：
     1. `reordered_groups`：两组中有多少组的“距离序”不同于“行序”。
     2. `same_as_row_baseline`：本变体输出是否与“按行序堆叠”的基线完全相同。
   - generator 要求 `reordered_groups >= 1`，因此 `same_as_row_baseline` 必须为 False。

i) main() 的新假设与 generator 保证
   - main() 新增假设：
     1. 目标组件是分离的 4x4 物体。
     2. 有效物体面积只会是 12 或 16。
     3. 若希望规则直观，同组最好不要等距。
   - generate_input() 分别通过组件间距、固定形状库、唯一距离约束来保证这些假设。
"""


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    arr = np.full((4, 4), int(col), dtype=int)
    arr[1:3, 1:3] = int(bg)
    return arr


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _distance2_to_center(r0: int, c0: int, H: int, W: int) -> int:
    """
    Squared Euclidean distance using doubled coordinates.

    Object center:
      (r0 + 1.5, c0 + 1.5) -> (2*r0 + 3, 2*c0 + 3)
    Grid center:
      ((H - 1)/2, (W - 1)/2) -> (H - 1, W - 1)
    """
    dr2 = int(2 * int(r0) + 3 - (int(H) - 1))
    dc2 = int(2 * int(c0) + 3 - (int(W) - 1))
    return int(dr2 * dr2 + dc2 * dc2)


def _collect_objects(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = int(get_bg(g))
    H, W = map(int, g.shape)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    area12: list[dict] = []
    area16: list[dict] = []
    for comp in comps:
        box = comp["bbox"]
        if bbox_h(box) != 4 or bbox_w(box) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, box)
        sub = crop_bbox(g, box)
        area = int(np.count_nonzero(sub != int(bg)))
        if area not in (12, 16):
            continue

        obj = {
            "r0": int(r0),
            "c0": int(c0),
            "sub": sub.copy(),
            "area": int(area),
            "dist2": _distance2_to_center(int(r0), int(c0), int(H), int(W)),
        }
        if area == 12:
            area12.append(obj)
        else:
            area16.append(obj)

    return int(bg), area12, area16


def _sort_by_distance(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["dist2"]), int(obj["r0"]), int(obj["c0"])))


def _sort_by_position(group: list[dict]) -> list[dict]:
    return sorted(group, key=lambda obj: (int(obj["r0"]), int(obj["c0"])))


def _render_output(bg: int, left_group: list[dict], right_group: list[dict]) -> np.ndarray:
    out_h = 4 * max(int(len(left_group)), int(len(right_group)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, obj in enumerate(left_group):
        paste(out, obj["sub"], int(i * 4), 0)
    for i, obj in enumerate(right_group):
        paste(out, obj["sub"], int(i * 4), 4)

    return out


def _position_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in _sort_by_position(group))


def _distance_signature(group: list[dict]) -> tuple[tuple[int, int], ...]:
    return tuple((int(obj["r0"]), int(obj["c0"])) for obj in _sort_by_distance(group))


def _variant_metrics(g: np.ndarray) -> dict[str, int | bool]:
    bg, area12, area16 = _collect_objects(np.array(g, dtype=int))
    left_row = _sort_by_position(area12)
    right_row = _sort_by_position(area16)
    left_dist = _sort_by_distance(area12)
    right_dist = _sort_by_distance(area16)

    reordered_groups = int(_position_signature(area12) != _distance_signature(area12)) + int(
        _position_signature(area16) != _distance_signature(area16)
    )
    same_as_row_baseline = bool(
        np.array_equal(
            _render_output(int(bg), left_dist, right_dist),
            _render_output(int(bg), left_row, right_row),
        )
    )

    return {
        "n_area12": int(len(area12)),
        "n_area16": int(len(area16)),
        "reordered_groups": int(reordered_groups),
        "same_as_row_baseline": bool(same_as_row_baseline),
    }


def _variant_visible(g: np.ndarray) -> bool:
    _bg, area12, area16 = _collect_objects(np.array(g, dtype=int))
    groups = [area12, area16]

    if any(len(group) < 2 for group in groups):
        return False
    if any(len({int(obj["dist2"]) for obj in group}) != len(group) for group in groups):
        return False

    metrics = _variant_metrics(g)
    return int(metrics["reordered_groups"]) >= 1 and not bool(metrics["same_as_row_baseline"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect all 4x4 objects.
    - Group them by non-background area: area 12 on the left, area 16 on the right.
    - Inside each area group, sort by distance from object center to grid center.
    - Ties use (row, col) only for deterministic output.
    """
    g = np.array(grid, dtype=int)
    bg, area12, area16 = _collect_objects(g)
    assert area12 or area16

    left_group = _sort_by_distance(area12)
    right_group = _sort_by_distance(area16)
    return _render_output(int(bg), left_group, right_group)


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

    n_area12 = int(np.random.randint(2, 5))
    n_area16 = int(np.random.randint(2, 5))

    try:
        used_cols: set[int] = set()
        objects: list[np.ndarray] = []

        for _ in range(n_area12):
            objects.append(_make_frame(_sample_unique_color(used_cols), bg))
        for _ in range(n_area16):
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
